"""Deterministic quality evaluation for review aspect analysis.

The evaluation pipeline is isolated from production data: its default inputs and
outputs live under ``evaluation/`` and importing this module never calls a model.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aspect_taxonomy import STANDARD_ASPECTS
from config import (
    LLM_PROVIDER,
    OLLAMA_BENCHMARK_MODELS,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    PROJECT_ROOT,
)
from llm_client import (
    LLMClientError,
    LLMResponseError,
    OllamaReviewClient,
    OpenAIReviewClient,
    ReviewAnalysisClient,
    create_review_client,
)
from review_analyzer import ReviewAnalysisError, analyze_review


EVALUATION_DIR = PROJECT_ROOT / "evaluation"
GOLD_CASES_PATH = EVALUATION_DIR / "gold_cases.json"
RESULTS_PATH = EVALUATION_DIR / "evaluation_results.json"
ERRORS_PATH = EVALUATION_DIR / "evaluation_errors.csv"
BASELINE_QWEN3_8B_PATH = EVALUATION_DIR / "baseline_qwen3_8b.json"

SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed")
EVIDENCE_MATCH_THRESHOLD = 0.60
WRONG_ASPECT_EVIDENCE_THRESHOLD = 0.80
REGRESSION_DROP_THRESHOLD = 0.05
MIN_EVIDENCE_VALIDITY = 0.95
BASELINE_METRICS = (
    "aspect_f1",
    "aspect_only_f1",
    "sentiment_accuracy",
    "evidence_validity_rate",
    "evidence_match_rate",
    "exact_case_accuracy",
)


@dataclass(frozen=True)
class _Mention:
    aspect: str
    sentiment: str
    evidences: tuple[str, ...]
    source_indices: tuple[int, ...]
    raw_aspect: str = ""
    derived_mixed: bool = False


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Return zero-safe precision, recall, and F1."""

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {
        actual: {predicted: 0 for predicted in SENTIMENT_LABELS}
        for actual in SENTIMENT_LABELS
    }


def normalize_evidence(value: str) -> str:
    """Case-fold, remove punctuation, and normalize whitespace."""

    if not isinstance(value, str):
        return ""
    normalized = re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE)
    return " ".join(normalized.split())


def evidence_similarity(expected: str, actual: str) -> float:
    """Score evidence using phrase containment and multiset token overlap.

    One-word containment is deliberately not treated as a perfect match: a quote
    such as ``battery`` is too weak to stand in for a full supporting statement.
    """

    normalized_expected = normalize_evidence(expected)
    normalized_actual = normalize_evidence(actual)
    if not normalized_expected or not normalized_actual:
        return 0.0
    if normalized_expected == normalized_actual:
        return 1.0
    expected_tokens = normalized_expected.split()
    actual_tokens = normalized_actual.split()
    if min(len(expected_tokens), len(actual_tokens)) >= 2 and (
        normalized_expected in normalized_actual
        or normalized_actual in normalized_expected
    ):
        return 1.0
    overlap = sum(
        (Counter(expected_tokens) & Counter(actual_tokens)).values()
    )
    return _safe_divide(overlap, max(len(expected_tokens), len(actual_tokens)))


def evidence_matches(
    expected: str,
    actual: str,
    *,
    threshold: float = EVIDENCE_MATCH_THRESHOLD,
) -> bool:
    return evidence_similarity(expected, actual) >= threshold


def evidence_is_valid(review: str, evidence: str) -> bool:
    """Enforce contiguous, verbatim source evidence (after edge trimming)."""

    if not isinstance(review, str) or not isinstance(evidence, str):
        return False
    candidate = evidence.strip()
    return bool(candidate) and candidate in review


def _evidence_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = value.strip()
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )
    return ()


def _display_evidence(values: tuple[str, ...]) -> str | list[str]:
    return values[0] if len(values) == 1 else list(values)


def _mention_dict(mention: _Mention) -> dict[str, Any]:
    result: dict[str, Any] = {
        "aspect": mention.aspect,
        "sentiment": mention.sentiment,
        "evidence": _display_evidence(mention.evidences),
    }
    if mention.raw_aspect:
        result["raw_aspect"] = mention.raw_aspect
    if mention.derived_mixed:
        result["derived_mixed"] = True
        result["source_indices"] = list(mention.source_indices)
    return result


def _best_evidence_similarity(expected: _Mention, actual: _Mention) -> float:
    if not expected.evidences or not actual.evidences:
        return 0.0
    return max(
        evidence_similarity(gold, predicted)
        for gold in expected.evidences
        for predicted in actual.evidences
    )


def _validate_gold_case(case: Mapping[str, Any], position: int) -> dict[str, Any]:
    if not isinstance(case, Mapping):
        raise ValueError(f"Gold case at index {position} must be an object.")
    case_id = case.get("case_id")
    review = case.get("review", case.get("review_text"))
    raw_expected = case.get("expected_aspects")
    context = f"Gold case at index {position}"
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{context} must have a non-empty case_id.")
    if not isinstance(review, str) or not review.strip():
        raise ValueError(f"{context} must have a non-empty review.")
    if not isinstance(raw_expected, list):
        raise ValueError(f"{context}.expected_aspects must be a list.")

    expected: list[dict[str, Any]] = []
    for index, item in enumerate(raw_expected):
        item_context = f"{context}.expected_aspects[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{item_context} must be an object.")
        aspect = item.get("aspect")
        sentiment = item.get("sentiment")
        evidences = _evidence_values(item.get("evidence"))
        if not isinstance(aspect, str) or aspect not in STANDARD_ASPECTS:
            raise ValueError(f"{item_context}.aspect must use the fixed taxonomy.")
        if sentiment not in SENTIMENT_LABELS:
            raise ValueError(
                f"{item_context}.sentiment must be one of: "
                f"{', '.join(SENTIMENT_LABELS)}."
            )
        if not evidences:
            raise ValueError(f"{item_context}.evidence must be non-empty.")
        expected.append(
            {
                "aspect": aspect,
                "sentiment": sentiment,
                "evidence": _display_evidence(evidences),
            }
        )
    return {
        "case_id": case_id.strip(),
        "review": review,
        "expected_aspects": expected,
    }


def load_gold_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the fixed human-labelled dataset; never invoke a model."""

    resolved = Path(path) if path is not None else GOLD_CASES_PATH
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Gold cases must be a non-empty JSON list.")
    cases = [_validate_gold_case(case, index) for index, case in enumerate(payload)]
    duplicate_ids = sorted(
        case_id
        for case_id, count in Counter(case["case_id"] for case in cases).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            f"Gold case_id values must be unique: {', '.join(duplicate_ids)}"
        )
    return cases


def _expected_mentions(case: Mapping[str, Any]) -> list[_Mention]:
    return [
        _Mention(
            aspect=str(item["aspect"]),
            sentiment=str(item["sentiment"]),
            evidences=_evidence_values(item.get("evidence")),
            source_indices=(index,),
        )
        for index, item in enumerate(case["expected_aspects"])
    ]


def _actual_mentions(
    actual_analysis: Mapping[str, Any] | None,
) -> tuple[list[_Mention], list[dict[str, Any]], bool]:
    errors: list[dict[str, Any]] = []
    if not isinstance(actual_analysis, Mapping):
        return [], [{
            "error_type": "SCHEMA_ERROR",
            "message": "Analysis result must be an object.",
        }], True
    raw_aspects = actual_analysis.get("aspects")
    if not isinstance(raw_aspects, list):
        return [], [{
            "error_type": "SCHEMA_ERROR",
            "message": "Analysis result aspects must be a list.",
        }], True

    mentions: list[_Mention] = []
    for index, item in enumerate(raw_aspects):
        if not isinstance(item, Mapping):
            errors.append({
                "error_type": "SCHEMA_ERROR",
                "actual_index": index,
                "message": f"aspects[{index}] must be an object.",
            })
            continue
        aspect = item.get("aspect")
        sentiment = item.get("sentiment")
        evidences = _evidence_values(item.get("evidence"))
        if not isinstance(aspect, str) or not aspect.strip():
            errors.append({
                "error_type": "SCHEMA_ERROR",
                "actual_index": index,
                "message": f"aspects[{index}].aspect must be non-empty text.",
            })
            continue
        if sentiment not in SENTIMENT_LABELS:
            errors.append({
                "error_type": "SCHEMA_ERROR",
                "actual_index": index,
                "message": f"aspects[{index}].sentiment is invalid.",
            })
            continue
        if not evidences:
            errors.append({
                "error_type": "SCHEMA_ERROR",
                "actual_index": index,
                "message": f"aspects[{index}].evidence must be non-empty text.",
            })
        raw_aspect = item.get("raw_aspect")
        mentions.append(
            _Mention(
                aspect=aspect.strip(),
                sentiment=str(sentiment),
                evidences=evidences,
                source_indices=(index,),
                raw_aspect=(
                    raw_aspect.strip()
                    if isinstance(raw_aspect, str) and raw_aspect.strip()
                    else ""
                ),
            )
        )
    return mentions, errors, False


def _project_actual_mixed(
    expected: Sequence[_Mention],
    actual: Sequence[_Mention],
) -> list[_Mention]:
    """Derive mixed only for an aspect explicitly marked mixed in gold."""

    expected_mixed = Counter(
        item.aspect for item in expected if item.sentiment == "mixed"
    )
    existing_mixed = Counter(
        item.aspect for item in actual if item.sentiment == "mixed"
    )
    replacements: dict[int, _Mention] = {}
    consumed: set[int] = set()
    for aspect in sorted(expected_mixed):
        needed = max(0, expected_mixed[aspect] - existing_mixed[aspect])
        positive = [
            index for index, item in enumerate(actual)
            if item.aspect == aspect and item.sentiment == "positive"
        ]
        negative = [
            index for index, item in enumerate(actual)
            if item.aspect == aspect and item.sentiment == "negative"
        ]
        for pos_index, neg_index in zip(positive[:needed], negative[:needed]):
            first = min(pos_index, neg_index)
            items = sorted(
                (actual[pos_index], actual[neg_index]),
                key=lambda item: item.source_indices,
            )
            replacements[first] = _Mention(
                aspect=aspect,
                sentiment="mixed",
                evidences=tuple(
                    evidence for item in items for evidence in item.evidences
                ),
                source_indices=tuple(
                    source for item in items for source in item.source_indices
                ),
                raw_aspect=" / ".join(
                    item.raw_aspect for item in items if item.raw_aspect
                ),
                derived_mixed=True,
            )
            consumed.update((pos_index, neg_index))
    return [
        replacements[index] if index in replacements else mention
        for index, mention in enumerate(actual)
        if index in replacements or index not in consumed
    ]


def _pair_same_aspect(
    expected: Sequence[_Mention],
    actual: Sequence[_Mention],
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    remaining_expected = set(range(len(expected)))
    remaining_actual = set(range(len(actual)))
    pairs: list[tuple[int, int, float]] = []
    # Exact sentiment first makes the pairing equivalent to Counter intersection;
    # evidence similarity then resolves duplicates deterministically.
    for require_equal_sentiment in (True, False):
        candidates: list[tuple[float, int, int]] = []
        for expected_index in sorted(remaining_expected):
            for actual_index in sorted(remaining_actual):
                gold = expected[expected_index]
                predicted = actual[actual_index]
                if gold.aspect != predicted.aspect:
                    continue
                if require_equal_sentiment != (
                    gold.sentiment == predicted.sentiment
                ):
                    continue
                candidates.append((
                    _best_evidence_similarity(gold, predicted),
                    expected_index,
                    actual_index,
                ))
        for similarity, expected_index, actual_index in sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2])
        ):
            if (
                expected_index not in remaining_expected
                or actual_index not in remaining_actual
            ):
                continue
            pairs.append((expected_index, actual_index, similarity))
            remaining_expected.remove(expected_index)
            remaining_actual.remove(actual_index)
    return pairs, remaining_expected, remaining_actual


def _pair_wrong_aspects(
    expected: Sequence[_Mention],
    actual: Sequence[_Mention],
    expected_indexes: set[int],
    actual_indexes: set[int],
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Consume strong-evidence taxonomy errors instead of MISS plus EXTRA."""

    remaining_expected = set(expected_indexes)
    remaining_actual = set(actual_indexes)
    candidates: list[tuple[float, int, int]] = []
    for expected_index in sorted(remaining_expected):
        for actual_index in sorted(remaining_actual):
            gold = expected[expected_index]
            predicted = actual[actual_index]
            if gold.aspect == predicted.aspect:
                continue
            similarity = _best_evidence_similarity(gold, predicted)
            if similarity >= WRONG_ASPECT_EVIDENCE_THRESHOLD:
                candidates.append((similarity, expected_index, actual_index))
    pairs: list[tuple[int, int, float]] = []
    for similarity, expected_index, actual_index in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if (
            expected_index not in remaining_expected
            or actual_index not in remaining_actual
        ):
            continue
        pairs.append((expected_index, actual_index, similarity))
        remaining_expected.remove(expected_index)
        remaining_actual.remove(actual_index)
    return pairs, remaining_expected, remaining_actual


def _finite_nonnegative(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) and numeric >= 0.0 else 0.0


def evaluate_case(
    case: Mapping[str, Any],
    actual_analysis: Mapping[str, Any] | None,
    *,
    analysis_time_seconds: float = 0.0,
    model_error: str | None = None,
    schema_error: str | None = None,
) -> dict[str, Any]:
    """Evaluate one case without I/O or model calls."""

    validated = _validate_gold_case(case, 0)
    expected = _expected_mentions(validated)
    errors: list[dict[str, Any]] = []
    if model_error:
        actual: list[_Mention] = []
        errors.append({"error_type": "MODEL_ERROR", "message": str(model_error)})
        fatal_error = True
    elif schema_error:
        actual = []
        errors.append({"error_type": "SCHEMA_ERROR", "message": str(schema_error)})
        fatal_error = True
    else:
        actual, schema_errors, fatal_error = _actual_mentions(actual_analysis)
        errors.extend(schema_errors)

    projected = _project_actual_mixed(expected, actual)
    review = validated["review"]
    evidence_evaluations: list[dict[str, Any]] = []
    valid_evidence = 0
    for mention in actual:
        evidence = mention.evidences[0] if mention.evidences else ""
        valid = evidence_is_valid(review, evidence)
        valid_evidence += int(valid)
        evidence_evaluations.append({
            "actual_index": mention.source_indices[0],
            "aspect": mention.aspect,
            "evidence": evidence,
            "valid": valid,
        })
        if not valid:
            errors.append({
                "error_type": "INVALID_EVIDENCE",
                "actual_index": mention.source_indices[0],
                "actual_aspect": mention.aspect,
                "actual_evidence": evidence,
                "message": "Predicted evidence is not a contiguous review substring.",
            })

    same_pairs, expected_left, actual_left = _pair_same_aspect(expected, projected)
    confusion = empty_confusion_matrix()
    wrong_sentiments: list[dict[str, Any]] = []
    evidence_match_rows: list[dict[str, Any]] = []
    sentiment_correct = 0
    matched_evidence = 0
    for expected_index, actual_index, similarity in same_pairs:
        gold = expected[expected_index]
        predicted = projected[actual_index]
        confusion[gold.sentiment][predicted.sentiment] += 1
        if gold.sentiment == predicted.sentiment:
            sentiment_correct += 1
        else:
            wrong_sentiments.append({
                "aspect": gold.aspect,
                "expected": gold.sentiment,
                "actual": predicted.sentiment,
                "expected_evidence": _display_evidence(gold.evidences),
                "actual_evidence": _display_evidence(predicted.evidences),
            })
            errors.append({
                "error_type": "WRONG_SENTIMENT",
                "expected_aspect": gold.aspect,
                "actual_aspect": predicted.aspect,
                "expected_sentiment": gold.sentiment,
                "actual_sentiment": predicted.sentiment,
                "expected_evidence": _display_evidence(gold.evidences),
                "actual_evidence": _display_evidence(predicted.evidences),
                "message": "Aspect matched but sentiment did not.",
            })
        matched = similarity >= EVIDENCE_MATCH_THRESHOLD
        matched_evidence += int(matched)
        evidence_match_rows.append({
            "aspect": gold.aspect,
            "expected_evidence": _display_evidence(gold.evidences),
            "actual_evidence": _display_evidence(predicted.evidences),
            "similarity": similarity,
            "matched": matched,
        })
        if not matched:
            errors.append({
                "error_type": "WEAK_EVIDENCE_MATCH",
                "expected_aspect": gold.aspect,
                "actual_aspect": predicted.aspect,
                "expected_evidence": _display_evidence(gold.evidences),
                "actual_evidence": _display_evidence(predicted.evidences),
                "similarity": similarity,
                "message": f"Evidence similarity is below {EVIDENCE_MATCH_THRESHOLD:.0%}.",
            })

    wrong_pairs: list[tuple[int, int, float]] = []
    if not fatal_error:
        wrong_pairs, expected_left, actual_left = _pair_wrong_aspects(
            expected, projected, expected_left, actual_left
        )
    wrong_aspects: list[dict[str, Any]] = []
    for expected_index, actual_index, similarity in wrong_pairs:
        gold = expected[expected_index]
        predicted = projected[actual_index]
        wrong_aspects.append({
            "expected": gold.aspect,
            "actual": predicted.aspect,
            "expected_sentiment": gold.sentiment,
            "actual_sentiment": predicted.sentiment,
            "expected_evidence": _display_evidence(gold.evidences),
            "actual_evidence": _display_evidence(predicted.evidences),
            "similarity": similarity,
        })
        errors.append({
            "error_type": "WRONG_ASPECT",
            "expected_aspect": gold.aspect,
            "actual_aspect": predicted.aspect,
            "expected_sentiment": gold.sentiment,
            "actual_sentiment": predicted.sentiment,
            "expected_evidence": _display_evidence(gold.evidences),
            "actual_evidence": _display_evidence(predicted.evidences),
            "similarity": similarity,
            "message": "Strong evidence match indicates a taxonomy classification error.",
        })

    missing_aspects = [expected[index].aspect for index in sorted(expected_left)]
    extra_aspects = [projected[index].aspect for index in sorted(actual_left)]
    if not fatal_error:
        for index in sorted(expected_left):
            gold = expected[index]
            errors.append({
                "error_type": "MISS_ASPECT",
                "expected_aspect": gold.aspect,
                "expected_sentiment": gold.sentiment,
                "expected_evidence": _display_evidence(gold.evidences),
                "message": "Expected aspect mention was not predicted.",
            })
        for index in sorted(actual_left):
            predicted = projected[index]
            errors.append({
                "error_type": "EXTRA_ASPECT",
                "actual_aspect": predicted.aspect,
                "actual_sentiment": predicted.sentiment,
                "actual_evidence": _display_evidence(predicted.evidences),
                "message": "Predicted aspect mention has no gold match.",
            })

    expected_pairs = Counter((item.aspect, item.sentiment) for item in expected)
    actual_pairs = Counter((item.aspect, item.sentiment) for item in projected)
    exact_tp = sum((expected_pairs & actual_pairs).values())
    expected_aspects = Counter(item.aspect for item in expected)
    actual_aspects = Counter(item.aspect for item in projected)
    aspect_only_tp = sum((expected_aspects & actual_aspects).values())
    evidence_total = len(actual)
    evidence_compared = len(same_pairs)
    exact_match = (
        not fatal_error
        and not any(error["error_type"] == "SCHEMA_ERROR" for error in errors)
        and expected_pairs == actual_pairs
    )
    counts = {
        "exact_tp": exact_tp,
        "exact_fp": sum(actual_pairs.values()) - exact_tp,
        "exact_fn": sum(expected_pairs.values()) - exact_tp,
        "aspect_only_tp": aspect_only_tp,
        "aspect_only_fp": sum(actual_aspects.values()) - aspect_only_tp,
        "aspect_only_fn": sum(expected_aspects.values()) - aspect_only_tp,
        "sentiment_correct": sentiment_correct,
        "sentiment_total": len(same_pairs),
        "evidence_valid": valid_evidence,
        "evidence_total": evidence_total,
        "evidence_matched": matched_evidence,
        "evidence_compared": evidence_compared,
    }
    return {
        "case_id": validated["case_id"],
        "review": review,
        "expected_aspects": [_mention_dict(item) for item in expected],
        "actual_aspects": [_mention_dict(item) for item in actual],
        "evaluation_aspects": [_mention_dict(item) for item in projected],
        "analysis_time_seconds": _finite_nonnegative(analysis_time_seconds),
        "exact_match": exact_match,
        "missing_aspects": missing_aspects,
        "extra_aspects": extra_aspects,
        "wrong_sentiments": wrong_sentiments,
        "wrong_aspects": wrong_aspects,
        "evidence_evaluations": evidence_evaluations,
        "evidence_matches": evidence_match_rows,
        "evidence_validity_rate": _safe_divide(valid_evidence, evidence_total),
        "evidence_match_rate": _safe_divide(matched_evidence, evidence_compared),
        "confusion_matrix": confusion,
        "errors": errors,
        "counts": counts,
    }


def _merge_confusion_matrices(
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    combined = empty_confusion_matrix()
    for result in case_results:
        matrix = result.get("confusion_matrix", {})
        if not isinstance(matrix, Mapping):
            continue
        for actual in SENTIMENT_LABELS:
            row = matrix.get(actual, {})
            if not isinstance(row, Mapping):
                continue
            for predicted in SENTIMENT_LABELS:
                try:
                    combined[actual][predicted] += int(row.get(predicted, 0))
                except (TypeError, ValueError):
                    pass
    return combined


def calculate_metrics(
    case_results: Sequence[Mapping[str, Any]],
    *,
    model: str = "",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Aggregate multiset counts, quality rates, and serial runtimes."""

    totals: Counter[str] = Counter()
    runtimes: list[float] = []
    exact_matches = 0
    cases_with_errors = 0
    failed_cases = 0
    count_keys = (
        "exact_tp",
        "exact_fp",
        "exact_fn",
        "aspect_only_tp",
        "aspect_only_fp",
        "aspect_only_fn",
        "sentiment_correct",
        "sentiment_total",
        "evidence_valid",
        "evidence_total",
        "evidence_matched",
        "evidence_compared",
    )
    for result in case_results:
        raw_counts = result.get("counts", {})
        if isinstance(raw_counts, Mapping):
            for key in count_keys:
                try:
                    totals[key] += int(raw_counts.get(key, 0))
                except (TypeError, ValueError):
                    pass
        runtimes.append(
            _finite_nonnegative(result.get("analysis_time_seconds", 0.0))
        )
        exact_matches += int(bool(result.get("exact_match")))
        raw_errors = result.get("errors", [])
        errors = raw_errors if isinstance(raw_errors, list) else []
        cases_with_errors += int(bool(errors))
        failed_cases += int(any(
            isinstance(error, Mapping)
            and error.get("error_type") in {"MODEL_ERROR", "SCHEMA_ERROR"}
            for error in errors
        ))

    aspect_rates = precision_recall_f1(
        totals["exact_tp"], totals["exact_fp"], totals["exact_fn"]
    )
    aspect_only_rates = precision_recall_f1(
        totals["aspect_only_tp"],
        totals["aspect_only_fp"],
        totals["aspect_only_fn"],
    )
    total_cases = len(case_results)
    return {
        "model": model,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "total_cases": total_cases,
        "successful_cases": total_cases - failed_cases,
        "failed_cases": failed_cases,
        "exact_matches": exact_matches,
        "cases_with_errors": cases_with_errors,
        "aspect_true_positives": totals["exact_tp"],
        "aspect_false_positives": totals["exact_fp"],
        "aspect_false_negatives": totals["exact_fn"],
        "aspect_precision": aspect_rates["precision"],
        "aspect_recall": aspect_rates["recall"],
        "aspect_f1": aspect_rates["f1"],
        "aspect_only_true_positives": totals["aspect_only_tp"],
        "aspect_only_false_positives": totals["aspect_only_fp"],
        "aspect_only_false_negatives": totals["aspect_only_fn"],
        "aspect_only_precision": aspect_only_rates["precision"],
        "aspect_only_recall": aspect_only_rates["recall"],
        "aspect_only_f1": aspect_only_rates["f1"],
        "sentiment_correct": totals["sentiment_correct"],
        "sentiment_total": totals["sentiment_total"],
        "sentiment_accuracy": _safe_divide(
            totals["sentiment_correct"], totals["sentiment_total"]
        ),
        "evidence_valid": totals["evidence_valid"],
        "evidence_total": totals["evidence_total"],
        "evidence_validity_rate": _safe_divide(
            totals["evidence_valid"], totals["evidence_total"]
        ),
        "evidence_matched": totals["evidence_matched"],
        "evidence_compared": totals["evidence_compared"],
        "evidence_match_rate": _safe_divide(
            totals["evidence_matched"], totals["evidence_compared"]
        ),
        "exact_case_accuracy": _safe_divide(exact_matches, total_cases),
        "average_analysis_time": _safe_divide(sum(runtimes), total_cases),
        "median_analysis_time": statistics.median(runtimes) if runtimes else 0.0,
        "total_analysis_time": sum(runtimes),
        "confusion_matrix": _merge_confusion_matrices(case_results),
    }


def _configured_model_name(
    client: ReviewAnalysisClient | None,
    model: str | None,
) -> str:
    if isinstance(model, str) and model.strip():
        return model.strip()
    client_model = getattr(client, "model", None)
    if isinstance(client_model, str) and client_model.strip():
        return client_model.strip()
    return OPENAI_MODEL if LLM_PROVIDER == "openai" else OLLAMA_MODEL


def _create_client_for_model(model: str | None) -> ReviewAnalysisClient:
    if model is None:
        return create_review_client()
    if LLM_PROVIDER == "openai":
        return OpenAIReviewClient(model=model)
    return OllamaReviewClient(model=model)


def evaluate_model(
    cases: Sequence[Mapping[str, Any]] | None = None,
    *,
    model: str | None = None,
    client: ReviewAnalysisClient | None = None,
    client_factory: Callable[[str], ReviewAnalysisClient] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    gold_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one model over cases strictly serially, without writing files."""

    resolved_cases = list(cases) if cases is not None else load_gold_cases(gold_path)
    model_name = _configured_model_name(client, model)
    if client is not None and client_factory is not None:
        raise ValueError("Provide client or client_factory, not both.")
    resolved_client = client
    if resolved_client is None:
        resolved_client = (
            client_factory(model_name)
            if client_factory is not None
            else _create_client_for_model(model)
        )

    case_results: list[dict[str, Any]] = []
    for position, raw_case in enumerate(resolved_cases):
        case = _validate_gold_case(raw_case, position)
        started = clock()
        actual: Mapping[str, Any] | None = None
        model_error: str | None = None
        schema_error: str | None = None
        try:
            actual = analyze_review(case["review"], llm_client=resolved_client)
        except (LLMResponseError, ReviewAnalysisError) as exc:
            schema_error = str(exc)
        except LLMClientError as exc:
            model_error = str(exc)
        except Exception as exc:  # SDK adapters can expose provider-specific types.
            model_error = f"{type(exc).__name__}: {exc}"
        elapsed = _finite_nonnegative(clock() - started)
        case_results.append(evaluate_case(
            case,
            actual,
            analysis_time_seconds=elapsed,
            model_error=model_error,
            schema_error=schema_error,
        ))

    summary = calculate_metrics(case_results, model=model_name)
    error_counts = Counter(
        str(error.get("error_type"))
        for result in case_results
        for error in result["errors"]
        if isinstance(error, Mapping) and error.get("error_type")
    )
    return {
        "summary": summary,
        "confusion_matrix": summary["confusion_matrix"],
        "error_counts": dict(sorted(error_counts.items())),
        "cases": case_results,
    }


def evaluate_models(
    models: Sequence[str] = OLLAMA_BENCHMARK_MODELS,
    cases: Sequence[Mapping[str, Any]] | None = None,
    *,
    client_factory: Callable[[str], ReviewAnalysisClient] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    gold_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate models serially and preserve the caller's model order."""

    resolved_cases = list(cases) if cases is not None else load_gold_cases(gold_path)
    factory = client_factory or (lambda name: OllamaReviewClient(model=name))
    reports: list[dict[str, Any]] = []
    for model in models:
        reports.append(evaluate_model(
            resolved_cases,
            model=model,
            client_factory=factory,
            clock=clock,
        ))
    return reports


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _error_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = report.get("summary", {})
    model = summary.get("model", "") if isinstance(summary, Mapping) else ""
    timestamp = summary.get("timestamp", "") if isinstance(summary, Mapping) else ""
    rows: list[dict[str, Any]] = []
    raw_cases = report.get("cases", [])
    if not isinstance(raw_cases, Sequence):
        return rows
    for case in raw_cases:
        if not isinstance(case, Mapping):
            continue
        raw_errors = case.get("errors", [])
        if not isinstance(raw_errors, list):
            continue
        for error in raw_errors:
            if not isinstance(error, Mapping):
                continue
            rows.append({
                "timestamp": timestamp,
                "model": model,
                "case_id": case.get("case_id", ""),
                "error_type": error.get("error_type", ""),
                "review": case.get("review", ""),
                "expected_aspect": error.get("expected_aspect", ""),
                "actual_aspect": error.get("actual_aspect", ""),
                "expected_sentiment": error.get("expected_sentiment", ""),
                "actual_sentiment": error.get("actual_sentiment", ""),
                "expected_evidence": _csv_value(error.get("expected_evidence", "")),
                "actual_evidence": _csv_value(error.get("actual_evidence", "")),
                "similarity": error.get("similarity", ""),
                "message": error.get("message", ""),
            })
    return rows


def export_evaluation_results(
    report: Mapping[str, Any],
    path: str | Path | None = None,
) -> Path:
    resolved = Path(path) if path is not None else RESULTS_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


def export_evaluation_errors(
    report: Mapping[str, Any],
    path: str | Path | None = None,
) -> Path:
    resolved = Path(path) if path is not None else ERRORS_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "timestamp",
        "model",
        "case_id",
        "error_type",
        "review",
        "expected_aspect",
        "actual_aspect",
        "expected_sentiment",
        "actual_sentiment",
        "expected_evidence",
        "actual_evidence",
        "similarity",
        "message",
    )
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_error_rows(report))
    return resolved


def save_evaluation_report(
    report: Mapping[str, Any],
    *,
    results_path: str | Path | None = None,
    errors_path: str | Path | None = None,
) -> dict[str, str]:
    saved_results = export_evaluation_results(report, results_path)
    saved_errors = export_evaluation_errors(report, errors_path)
    # Short names are the public API; *_path aliases keep the report self-explanatory.
    return {
        "results": str(saved_results),
        "errors": str(saved_errors),
        "results_path": str(saved_results),
        "errors_path": str(saved_errors),
    }


def _summary_from_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = report.get("summary")
    return summary if isinstance(summary, Mapping) else report


def save_baseline(
    report: Mapping[str, Any],
    path: str | Path | None = None,
) -> Path:
    summary = _summary_from_report(report)
    baseline = {
        "model": summary.get("model", ""),
        "timestamp": summary.get("timestamp", ""),
        **{metric: float(summary.get(metric, 0.0)) for metric in BASELINE_METRICS},
    }
    resolved = Path(path) if path is not None else BASELINE_QWEN3_8B_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


def run_evaluation(
    cases: Sequence[Mapping[str, Any]] | None = None,
    *,
    model: str | None = None,
    client: ReviewAnalysisClient | None = None,
    client_factory: Callable[[str], ReviewAnalysisClient] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    gold_path: str | Path | None = None,
    results_path: str | Path | None = None,
    errors_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    create_baseline: bool = True,
) -> dict[str, Any]:
    """Evaluate, export, and create the qwen3:8b baseline only once."""

    report = evaluate_model(
        cases,
        model=model,
        client=client,
        client_factory=client_factory,
        clock=clock,
        gold_path=gold_path,
    )
    report["outputs"] = save_evaluation_report(
        report,
        results_path=results_path,
        errors_path=errors_path,
    )
    resolved_baseline = (
        Path(baseline_path) if baseline_path is not None else BASELINE_QWEN3_8B_PATH
    )
    if (
        create_baseline
        and report["summary"]["model"] == "qwen3:8b"
        and not resolved_baseline.exists()
    ):
        baseline = save_baseline(report, resolved_baseline)
        report["outputs"]["baseline"] = str(baseline)
        report["outputs"]["baseline_path"] = str(baseline)
    return report


def _load_json_mapping(path: Path, description: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must contain a JSON object.")
    return payload


def check_regression(
    current: Mapping[str, Any] | str | Path | None = None,
    *,
    baseline_path: str | Path | None = None,
    threshold: float = REGRESSION_DROP_THRESHOLD,
    minimum_evidence_validity: float = MIN_EVIDENCE_VALIDITY,
) -> dict[str, Any]:
    """Return non-blocking warnings against the saved baseline."""

    resolved_baseline = (
        Path(baseline_path) if baseline_path is not None else BASELINE_QWEN3_8B_PATH
    )
    baseline = _load_json_mapping(resolved_baseline, "Regression baseline")
    if current is None:
        current_report: Mapping[str, Any] = _load_json_mapping(
            RESULTS_PATH, "Current evaluation result"
        )
    elif isinstance(current, (str, Path)):
        current_report = _load_json_mapping(
            Path(current), "Current evaluation result"
        )
    elif isinstance(current, Mapping):
        current_report = current
    else:
        raise ValueError("Current evaluation must be a report, summary, or JSON path.")

    summary = _summary_from_report(current_report)
    deltas: dict[str, float] = {}
    for metric in BASELINE_METRICS:
        if metric in baseline and metric in summary:
            deltas[metric] = float(summary[metric]) - float(baseline[metric])

    warnings: list[str] = []
    for metric, label in (
        ("aspect_f1", "Aspect F1"),
        ("sentiment_accuracy", "Sentiment Accuracy"),
    ):
        drop = -deltas.get(metric, 0.0)
        # A tiny tolerance makes "exactly five points" reliably non-warning.
        if drop > threshold + 1e-12:
            warnings.append(
                f"{label} decreased by {drop * 100:.1f} percentage points "
                f"(threshold: more than {threshold * 100:.1f})."
            )

    validity = float(summary.get("evidence_validity_rate", 0.0))
    if validity < minimum_evidence_validity:
        warnings.append(
            f"Evidence Validity is {validity:.1%}, "
            f"below {minimum_evidence_validity:.1%}."
        )
    return {
        "has_regression": bool(warnings),
        "warnings": warnings,
        "deltas": deltas,
        "baseline_path": str(resolved_baseline),
        "threshold": threshold,
        "minimum_evidence_validity": minimum_evidence_validity,
    }


def format_evaluation_report(report: Mapping[str, Any]) -> str:
    """Render the compact CLI quality report without printing it."""

    summary = _summary_from_report(report)
    raw_error_counts = report.get("error_counts", {})
    error_counts = raw_error_counts if isinstance(raw_error_counts, Mapping) else {}
    raw_outputs = report.get("outputs", {})
    outputs = raw_outputs if isinstance(raw_outputs, Mapping) else {}
    lines = [
        f"Model: {summary.get('model', '')}",
        f"Gold cases: {summary.get('total_cases', 0)}",
        "",
        "Evaluation Results",
        "--------------------------------",
        f"Aspect Precision:       {float(summary.get('aspect_precision', 0.0)):.1%}",
        f"Aspect Recall:          {float(summary.get('aspect_recall', 0.0)):.1%}",
        f"Aspect F1:              {float(summary.get('aspect_f1', 0.0)):.1%}",
        "",
        f"Aspect-only F1:         {float(summary.get('aspect_only_f1', 0.0)):.1%}",
        f"Sentiment Accuracy:     {float(summary.get('sentiment_accuracy', 0.0)):.1%}",
        "",
        f"Evidence Validity:      {float(summary.get('evidence_validity_rate', 0.0)):.1%}",
        f"Evidence Match:         {float(summary.get('evidence_match_rate', 0.0)):.1%}",
        "",
        f"Exact Case Accuracy:    {float(summary.get('exact_case_accuracy', 0.0)):.1%}",
        "",
        "Cases:",
        f"{summary.get('total_cases', 0)} total",
        f"{summary.get('exact_matches', 0)} exact",
        f"{summary.get('cases_with_errors', 0)} with errors",
        "",
        "Performance:",
        f"Total:   {float(summary.get('total_analysis_time', 0.0)):.2f}s",
        f"Average: {float(summary.get('average_analysis_time', 0.0)):.2f}s/review",
        f"Median:  {float(summary.get('median_analysis_time', 0.0)):.2f}s/review",
        "",
        "Top Error Types:",
    ]
    ranked = sorted(
        ((str(name), int(count)) for name, count in error_counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    lines.extend(
        (f"{name:<24}{count}" for name, count in ranked[:5])
        if ranked
        else ("None",)
    )
    output_paths: list[str] = []
    for key in ("results", "errors", "baseline"):
        if outputs.get(key):
            output_paths.append(str(outputs[key]))
    if output_paths:
        lines.extend(["", "Output:", *output_paths])
    raw_cases = report.get("cases", [])
    cases = raw_cases if isinstance(raw_cases, Sequence) else []
    error_cases = [
        case
        for case in cases
        if isinstance(case, Mapping) and case.get("errors")
    ][:5]
    if error_cases:
        lines.extend(["", "Error Examples:"])
        for case in error_cases:
            expected = "; ".join(
                f"{item.get('aspect', '')} → {item.get('sentiment', '')}"
                for item in case.get("expected_aspects", [])
                if isinstance(item, Mapping)
            ) or "(no specific aspect)"
            actual = "; ".join(
                f"{item.get('aspect', '')} → {item.get('sentiment', '')}"
                for item in case.get("evaluation_aspects", [])
                if isinstance(item, Mapping)
            ) or "(no specific aspect)"
            error_types = ", ".join(dict.fromkeys(
                str(error.get("error_type", ""))
                for error in case.get("errors", [])
                if isinstance(error, Mapping) and error.get("error_type")
            ))
            lines.extend([
                f"Case {case.get('case_id', '')}",
                f"Review: {case.get('review', '')}",
                f"Expected: {expected}",
                f"Actual: {actual}",
                f"Error: {error_types}",
            ])
    return "\n".join(lines)


def format_model_comparison(reports: Sequence[Mapping[str, Any]]) -> str:
    """Render objective model rows without choosing a winner."""

    lines = [
        "Model Comparison",
        "",
        f"{'Model':<24}{'Aspect F1':>12}{'Sentiment':>13}"
        f"{'Exact Case':>14}{'Avg Time':>12}",
    ]
    for report in reports:
        summary = _summary_from_report(report)
        lines.append(
            f"{str(summary.get('model', '')):<24}"
            f"{float(summary.get('aspect_f1', 0.0)):>11.1%}"
            f"{float(summary.get('sentiment_accuracy', 0.0)):>13.1%}"
            f"{float(summary.get('exact_case_accuracy', 0.0)):>14.1%}"
            f"{float(summary.get('average_analysis_time', 0.0)):>10.2f}s"
        )
    return "\n".join(lines)


def format_regression_report(result: Mapping[str, Any]) -> str:
    if result.get("has_regression"):
        warnings = result.get("warnings", [])
        return "\n".join(["REGRESSION WARNING", *map(str, warnings)])
    return "No regression warning."
