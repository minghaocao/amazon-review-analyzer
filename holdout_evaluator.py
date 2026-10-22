"""Blind holdout evaluation for the frozen Step 8.6 analysis baseline.

This module is intentionally measurement-only.  It reuses the production
``analyze_review`` entry point and the established evaluator metrics without
changing prompts, taxonomy, normalization, or post-processing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import LLM_PROVIDER, OLLAMA_MODEL, PROJECT_ROOT
from evaluator import calculate_metrics, evaluate_case, load_gold_cases
from llm_client import (
    LLMClientError,
    LLMConnectionError,
    LLMResponseError,
    ReviewAnalysisClient,
    create_review_client,
)
from review_analyzer import ReviewAnalysisError, analyze_review


HOLDOUT_CASES_PATH = PROJECT_ROOT / "evaluation" / "holdout_cases.json"
HOLDOUT_MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "holdout_manifest.json"
DEVELOPMENT_REPORT_PATH = (
    PROJECT_ROOT / "evaluation" / "evaluation_step8_6_qwen3_8b.json"
)
HOLDOUT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "holdout"
HOLDOUT_REPORT_PATH = HOLDOUT_OUTPUT_DIR / "holdout_evaluation_qwen3_8b.json"
HOLDOUT_ERRORS_PATH = HOLDOUT_OUTPUT_DIR / "holdout_errors.csv"
HOLDOUT_PREDICTIONS_PATH = HOLDOUT_OUTPUT_DIR / "holdout_predictions.json"

EXPECTED_HOLDOUT_CASE_COUNT = 50
HOLDOUT_ID_PREFIX = "H"
TRANSIENT_RETRY_DELAY_SECONDS = 1.0

GAP_METRICS = (
    "aspect_precision",
    "aspect_recall",
    "aspect_f1",
    "sentiment_accuracy",
    "exact_case_accuracy",
)

ERROR_PRIORITY = {
    "WRONG_ASPECT": 0,
    "MISS_ASPECT": 1,
    "EXTRA_ASPECT": 2,
    "WRONG_SENTIMENT": 3,
    "WEAK_EVIDENCE_MATCH": 4,
    "INVALID_EVIDENCE": 5,
    "MODEL_ERROR": 6,
    "SCHEMA_ERROR": 7,
    "POSSIBLE_GOLD_ISSUE": 8,
}

ERROR_CSV_FIELDS = (
    "case_id",
    "review",
    "error_type",
    "expected_aspects",
    "actual_aspects",
    "expected_sentiments",
    "actual_sentiments",
    "expected_evidence",
    "actual_evidence",
    "notes",
    "message",
)


class HoldoutIntegrityError(ValueError):
    """The immutable holdout dataset or its manifest failed validation."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_case_ids(count: int) -> list[str]:
    return [f"{HOLDOUT_ID_PREFIX}{index:03d}" for index in range(1, count + 1)]


def _raw_holdout_payload(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HoldoutIntegrityError(f"Holdout dataset was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HoldoutIntegrityError(
            f"Holdout dataset is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise HoldoutIntegrityError("Holdout dataset must be a JSON list.")
    return payload


def _validate_holdout_dataset(
    dataset_path: Path,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    raw_cases = _raw_holdout_payload(dataset_path)
    if len(raw_cases) != expected_count:
        raise HoldoutIntegrityError(
            f"Holdout dataset must contain exactly {expected_count} cases; "
            f"found {len(raw_cases)}."
        )

    try:
        validated_cases = load_gold_cases(dataset_path)
    except (OSError, ValueError) as exc:
        raise HoldoutIntegrityError(str(exc)) from exc

    actual_ids = [case["case_id"] for case in validated_cases]
    expected_ids = _expected_case_ids(expected_count)
    if actual_ids != expected_ids:
        raise HoldoutIntegrityError(
            "Holdout case IDs must be ordered consecutively from "
            f"{expected_ids[0]} through {expected_ids[-1]}."
        )
    normalized_reviews = [
        " ".join(case["review"].casefold().split()) for case in validated_cases
    ]
    if len(normalized_reviews) != len(set(normalized_reviews)):
        raise HoldoutIntegrityError("Holdout review texts must be unique.")

    enriched: list[dict[str, Any]] = []
    for position, (raw_case, validated) in enumerate(
        zip(raw_cases, validated_cases, strict=True)
    ):
        if not isinstance(raw_case, Mapping):
            raise HoldoutIntegrityError(
                f"Holdout case at index {position} must be an object."
            )
        notes = raw_case.get("notes", "")
        if not isinstance(notes, str) or not notes.strip():
            raise HoldoutIntegrityError(
                f"{validated['case_id']} must include non-empty annotation notes."
            )
        tags = raw_case.get("tags", [])
        if not isinstance(tags, list) or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            raise HoldoutIntegrityError(
                f"{validated['case_id']}.tags must be a list of non-empty strings."
            )
        review = validated["review"]
        for mention in validated["expected_aspects"]:
            if mention["sentiment"] not in {"positive", "negative", "neutral"}:
                raise HoldoutIntegrityError(
                    f"{validated['case_id']} uses an unsupported holdout sentiment: "
                    f"{mention['sentiment']!r}"
                )
            evidence = mention["evidence"]
            evidence_values = evidence if isinstance(evidence, list) else [evidence]
            for value in evidence_values:
                if value not in review:
                    raise HoldoutIntegrityError(
                        f"{validated['case_id']} evidence is not a contiguous review "
                        f"substring: {value!r}"
                    )
        enriched.append(
            {
                **validated,
                "notes": notes.strip(),
                "tags": [tag.strip() for tag in tags],
            }
        )
    return enriched


def create_holdout_manifest(
    dataset_path: str | Path = HOLDOUT_CASES_PATH,
    manifest_path: str | Path = HOLDOUT_MANIFEST_PATH,
    *,
    expected_count: int = EXPECTED_HOLDOUT_CASE_COUNT,
) -> dict[str, Any]:
    """Create the one-time pre-evaluation integrity manifest.

    This helper validates the complete dataset before recording its exact-byte
    hash.  The normal evaluation path never rewrites the manifest.
    """

    dataset = Path(dataset_path)
    _validate_holdout_dataset(dataset, expected_count=expected_count)
    manifest = {
        "dataset": dataset.name,
        "case_count": expected_count,
        "sha256": file_sha256(dataset),
        "created_before_evaluation": True,
    }
    destination = Path(manifest_path)
    if destination.exists():
        raise HoldoutIntegrityError(
            "Holdout manifest already exists and will not be overwritten."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_holdout_manifest(
    dataset_path: str | Path = HOLDOUT_CASES_PATH,
    manifest_path: str | Path = HOLDOUT_MANIFEST_PATH,
    *,
    expected_count: int = EXPECTED_HOLDOUT_CASE_COUNT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the frozen hash and all Gold invariants before any model call."""

    dataset = Path(dataset_path)
    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HoldoutIntegrityError(
            f"Holdout manifest was not found: {manifest_file}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise HoldoutIntegrityError(
            f"Holdout manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise HoldoutIntegrityError("Holdout manifest must be a JSON object.")
    if manifest.get("dataset") != dataset.name:
        raise HoldoutIntegrityError("Holdout manifest dataset name does not match.")
    if manifest.get("case_count") != expected_count:
        raise HoldoutIntegrityError("Holdout manifest case count does not match.")
    if manifest.get("created_before_evaluation") is not True:
        raise HoldoutIntegrityError(
            "Holdout manifest must confirm creation before evaluation."
        )
    recorded_hash = manifest.get("sha256")
    actual_hash = file_sha256(dataset)
    if not isinstance(recorded_hash, str) or recorded_hash != actual_hash:
        raise HoldoutIntegrityError(
            "Holdout dataset SHA-256 does not match its frozen manifest."
        )
    cases = _validate_holdout_dataset(dataset, expected_count=expected_count)
    return dict(manifest), cases


def _strict_exact_match(case_result: Mapping[str, Any]) -> bool:
    """Require aspect, sentiment, mention count, and Evidence for exactness."""

    if not bool(case_result.get("exact_match")):
        return False
    expected = case_result.get("expected_aspects", [])
    actual = case_result.get("actual_aspects", [])
    if not isinstance(expected, list) or not isinstance(actual, list):
        return False
    if not expected and not actual:
        return True
    counts = case_result.get("counts", {})
    if not isinstance(counts, Mapping):
        return False
    expected_count = len(expected)
    return (
        int(counts.get("evidence_total", 0)) == expected_count
        and int(counts.get("evidence_valid", 0)) == expected_count
        and int(counts.get("evidence_compared", 0)) == expected_count
        and int(counts.get("evidence_matched", 0)) == expected_count
    )


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _load_development_summary(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HoldoutIntegrityError(
            f"Step 8.6 development report was not found: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise HoldoutIntegrityError(
            f"Step 8.6 development report is not valid JSON: {exc}"
        ) from exc
    summary = payload.get("summary") if isinstance(payload, Mapping) else None
    if not isinstance(summary, Mapping):
        raise HoldoutIntegrityError(
            "Step 8.6 development report has no valid summary."
        )
    return dict(summary)


def _generalization_gap(
    development: Mapping[str, Any], holdout: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for metric in GAP_METRICS:
        development_value = float(development.get(metric, 0.0))
        holdout_value = float(holdout.get(metric, 0.0))
        comparison[metric] = {
            "development": development_value,
            "holdout": holdout_value,
            "difference": holdout_value - development_value,
            "difference_percentage_points": (
                holdout_value - development_value
            ) * 100.0,
        }
    return comparison


def dataset_distribution(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe the frozen Gold labels without inspecting model predictions."""

    aspect_counts: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()
    hard_case_counts: Counter[str] = Counter()
    aspect_count_per_case: Counter[str] = Counter()
    mixed_sentiment_cases = 0
    repeated_aspect_cases = 0
    for case in cases:
        mentions = case.get("expected_aspects", [])
        if not isinstance(mentions, list):
            mentions = []
        count = len(mentions)
        bucket = "0" if count == 0 else "1" if count == 1 else "2" if count == 2 else "3-5" if count <= 5 else "6+"
        aspect_count_per_case[bucket] += 1
        aspects_for_case: list[str] = []
        sentiments_for_case: set[str] = set()
        for mention in mentions:
            if not isinstance(mention, Mapping):
                continue
            aspect = str(mention.get("aspect", ""))
            sentiment = str(mention.get("sentiment", ""))
            aspect_counts[aspect] += 1
            sentiment_counts[sentiment] += 1
            aspects_for_case.append(aspect)
            sentiments_for_case.add(sentiment)
        mixed_sentiment_cases += int(
            "positive" in sentiments_for_case and "negative" in sentiments_for_case
        )
        repeated_aspect_cases += int(
            len(aspects_for_case) != len(set(aspects_for_case))
        )
        tags = case.get("tags", [])
        if isinstance(tags, list):
            hard_case_counts.update(str(tag) for tag in tags)
    return {
        "case_count": len(cases),
        "aspect_mentions": dict(sorted(aspect_counts.items())),
        "sentiment_mentions": dict(sorted(sentiment_counts.items())),
        "cases_by_expected_mention_count": dict(sorted(aspect_count_per_case.items())),
        "mixed_positive_negative_cases": mixed_sentiment_cases,
        "repeated_aspect_cases": repeated_aspect_cases,
        "hard_case_tags": dict(sorted(hard_case_counts.items())),
    }


def _error_explanation(
    error: Mapping[str, Any],
) -> str:
    error_type = str(error.get("error_type", ""))
    expected_aspect = str(error.get("expected_aspect", ""))
    actual_aspect = str(error.get("actual_aspect", ""))
    if error_type == "WRONG_ASPECT":
        return (
            f"Gold labels the supported opinion as {expected_aspect}, while the "
            f"prediction labels it as {actual_aspect}."
        )
    if error_type == "MISS_ASPECT":
        return f"The expected {expected_aspect} mention is absent from the prediction."
    if error_type == "EXTRA_ASPECT":
        return f"The prediction adds an unmatched {actual_aspect} mention."
    if error_type == "WRONG_SENTIMENT":
        return (
            f"The Aspect matches, but Gold is {error.get('expected_sentiment')} "
            f"and the prediction is {error.get('actual_sentiment')}."
        )
    if error_type == "WEAK_EVIDENCE_MATCH":
        return "The predicted quote does not match the Gold supporting span closely enough."
    if error_type == "INVALID_EVIDENCE":
        return "The predicted Evidence is not a contiguous substring of the review."
    if error_type == "MODEL_ERROR":
        return "The provider did not return an analyzable response for this case."
    if error_type == "SCHEMA_ERROR":
        return "The model response failed the frozen structured-output contract."
    return str(error.get("message", "Prediction differs from the frozen Gold label."))


def select_top_errors(
    case_results: Sequence[Mapping[str, Any]], limit: int = 10
) -> list[dict[str, Any]]:
    """Select at most one highest-priority representative error per case."""

    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    for case_position, case in enumerate(case_results):
        raw_errors = case.get("errors", [])
        if not isinstance(raw_errors, list):
            continue
        case_candidates: list[tuple[int, int, int, dict[str, Any]]] = []
        for error_position, error in enumerate(raw_errors):
            if not isinstance(error, Mapping):
                continue
            error_type = str(error.get("error_type", ""))
            entry = {
                "case_id": case.get("case_id", ""),
                "review": case.get("review", ""),
                "expected": case.get("expected_aspects", []),
                "actual": case.get("actual_aspects", []),
                "error_type": error_type,
                "error_details": dict(error),
                "explanation": _error_explanation(error),
            }
            case_candidates.append(
                (
                    ERROR_PRIORITY.get(error_type, 99),
                    case_position,
                    error_position,
                    entry,
                )
            )
        if case_candidates:
            candidates.append(min(case_candidates, key=lambda item: item[:3]))
    candidates.sort(key=lambda item: item[:3])
    return [item[3] for item in candidates[: max(0, limit)]]


def _prediction_error(
    *, model_error: str | None, schema_error: str | None
) -> tuple[str, str]:
    if schema_error:
        return "SCHEMA_ERROR", schema_error
    if model_error:
        return "MODEL_ERROR", model_error
    return "", ""


def run_holdout_evaluation(
    *,
    dataset_path: str | Path = HOLDOUT_CASES_PATH,
    manifest_path: str | Path = HOLDOUT_MANIFEST_PATH,
    development_report_path: str | Path = DEVELOPMENT_REPORT_PATH,
    expected_count: int = EXPECTED_HOLDOUT_CASE_COUNT,
    provider: str = LLM_PROVIDER,
    model: str = OLLAMA_MODEL,
    client: ReviewAnalysisClient | None = None,
    client_factory: Callable[[], ReviewAnalysisClient] | None = None,
    analyzer: Callable[..., Mapping[str, Any]] = analyze_review,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    retry_delay_seconds: float = TRANSIENT_RETRY_DELAY_SECONDS,
    progress: Callable[[int, int, str, str, float], None] | None = None,
) -> dict[str, Any]:
    """Run the frozen analyzer serially after verifying the holdout manifest."""

    # Integrity and the development comparison are resolved before client
    # creation.  A broken freeze therefore cannot trigger even one model call.
    manifest, cases = verify_holdout_manifest(
        dataset_path,
        manifest_path,
        expected_count=expected_count,
    )
    development_summary = _load_development_summary(development_report_path)
    if client is not None and client_factory is not None:
        raise ValueError("Provide client or client_factory, not both.")
    resolved_client = client or (
        client_factory() if client_factory is not None else create_review_client()
    )

    case_results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    retries = 0
    for position, case in enumerate(cases, start=1):
        started = clock()
        actual: Mapping[str, Any] | None = None
        model_error: str | None = None
        schema_error: str | None = None
        attempts = 0
        while attempts < 2:
            attempts += 1
            try:
                actual = analyzer(case["review"], llm_client=resolved_client)
                break
            except LLMConnectionError as exc:
                if attempts == 1:
                    retries += 1
                    sleeper(max(0.0, retry_delay_seconds))
                    continue
                model_error = str(exc)
                break
            except (LLMResponseError, ReviewAnalysisError) as exc:
                schema_error = str(exc)
                break
            except LLMClientError as exc:
                model_error = str(exc)
                break
            except Exception as exc:  # Preserve one failed case without stopping 50.
                model_error = f"{type(exc).__name__}: {exc}"
                break

        elapsed = max(0.0, float(clock() - started))
        result = evaluate_case(
            case,
            actual,
            analysis_time_seconds=elapsed,
            model_error=model_error,
            schema_error=schema_error,
        )
        result["base_exact_match"] = bool(result["exact_match"])
        result["exact_match"] = _strict_exact_match(result)
        result["notes"] = case["notes"]
        result["tags"] = case["tags"]
        result["attempts"] = attempts
        result["retried"] = attempts > 1
        error_type, error_message = _prediction_error(
            model_error=model_error,
            schema_error=schema_error,
        )
        status = "failed" if error_type else "success"
        prediction = {
            "case_id": case["case_id"],
            "review": case["review"],
            "status": status,
            "attempts": attempts,
            "retried": attempts > 1,
            "analysis_time_seconds": elapsed,
            "analysis": dict(actual) if isinstance(actual, Mapping) else None,
            "error_type": error_type,
            "error_message": error_message,
        }
        predictions.append(prediction)
        case_results.append(result)
        if progress is None:
            print(
                f"[{position}/{len(cases)}] {case['case_id']} {status} | "
                f"{elapsed:.2f}s"
            )
        else:
            progress(position, len(cases), case["case_id"], status, elapsed)

    summary = calculate_metrics(case_results, model=model)
    runtimes = [float(item["analysis_time_seconds"]) for item in predictions]
    slowest = sorted(
        (
            {
                "case_id": item["case_id"],
                "analysis_time_seconds": item["analysis_time_seconds"],
                "status": item["status"],
            }
            for item in predictions
        ),
        key=lambda item: (-float(item["analysis_time_seconds"]), item["case_id"]),
    )[:5]
    error_counts = Counter(
        str(error.get("error_type"))
        for result in case_results
        for error in result.get("errors", [])
        if isinstance(error, Mapping) and error.get("error_type")
    )
    runtime_statistics = {
        "total_seconds": sum(runtimes),
        "average_seconds_per_review": (
            sum(runtimes) / len(runtimes) if runtimes else 0.0
        ),
        "median_seconds_per_review": statistics.median(runtimes) if runtimes else 0.0,
        "p90_seconds_per_review": _percentile_nearest_rank(runtimes, 0.90),
        "p90_method": "nearest_rank",
        "slowest_five": slowest,
        "retry_count": retries,
        "model_failure_count": error_counts.get("MODEL_ERROR", 0),
        "schema_failure_count": error_counts.get("SCHEMA_ERROR", 0),
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    summary["timestamp"] = timestamp
    report = {
        "evaluation_type": "blind_holdout",
        "baseline_version": "step8.6-v1",
        "provider": provider,
        "model": model,
        "dataset": Path(dataset_path).name,
        "dataset_sha256": manifest["sha256"],
        "timestamp": timestamp,
        "case_count": len(cases),
        "dataset_distribution": dataset_distribution(cases),
        "summary": summary,
        "runtime_statistics": runtime_statistics,
        "confusion_matrix": summary["confusion_matrix"],
        "error_counts": dict(sorted(error_counts.items())),
        "development_set": {
            "label": "Step 8.6 development evaluation set",
            "case_count": int(development_summary.get("total_cases", 0)),
            "summary": development_summary,
        },
        "holdout_set": {
            "label": "Step 8.7 blind holdout set",
            "case_count": len(cases),
        },
        "generalization_gap": _generalization_gap(
            development_summary, summary
        ),
        "top_errors": select_top_errors(case_results),
        "possible_gold_issues": [],
        "cases": case_results,
        "predictions": predictions,
    }
    return report


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _error_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = report.get("cases", [])
    if not isinstance(cases, list):
        return rows
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        expected = case.get("expected_aspects", [])
        actual = case.get("actual_aspects", [])
        expected_list = expected if isinstance(expected, list) else []
        actual_list = actual if isinstance(actual, list) else []
        errors = case.get("errors", [])
        if not isinstance(errors, list):
            continue
        for error in errors:
            if not isinstance(error, Mapping):
                continue
            rows.append(
                {
                    "case_id": case.get("case_id", ""),
                    "review": case.get("review", ""),
                    "error_type": error.get("error_type", ""),
                    "expected_aspects": _json_cell(
                        [item.get("aspect", "") for item in expected_list]
                    ),
                    "actual_aspects": _json_cell(
                        [item.get("aspect", "") for item in actual_list]
                    ),
                    "expected_sentiments": _json_cell(
                        [item.get("sentiment", "") for item in expected_list]
                    ),
                    "actual_sentiments": _json_cell(
                        [item.get("sentiment", "") for item in actual_list]
                    ),
                    "expected_evidence": _json_cell(
                        [item.get("evidence", "") for item in expected_list]
                    ),
                    "actual_evidence": _json_cell(
                        [item.get("evidence", "") for item in actual_list]
                    ),
                    "notes": case.get("notes", ""),
                    "message": error.get("message", ""),
                }
            )
    return rows


def save_holdout_outputs(
    report: Mapping[str, Any],
    *,
    report_path: str | Path = HOLDOUT_REPORT_PATH,
    errors_path: str | Path = HOLDOUT_ERRORS_PATH,
    predictions_path: str | Path = HOLDOUT_PREDICTIONS_PATH,
) -> dict[str, str]:
    """Save all holdout artifacts without touching legacy evaluation files."""

    report_destination = Path(report_path)
    errors_destination = Path(errors_path)
    predictions_destination = Path(predictions_path)
    for destination in (
        report_destination,
        errors_destination,
        predictions_destination,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "report": str(report_destination),
        "errors": str(errors_destination),
        "predictions": str(predictions_destination),
    }
    report_payload = dict(report)
    report_payload["outputs"] = output_paths
    report_destination.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with errors_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_error_rows(report))

    predictions_payload = {
        "provider": report_payload.get("provider", ""),
        "model": report_payload.get("model", ""),
        "dataset_sha256": report_payload.get("dataset_sha256", ""),
        "timestamp": report_payload.get("timestamp", ""),
        "case_count": report_payload.get("case_count", 0),
        "predictions": report_payload.get("predictions", []),
    }
    predictions_destination.write_text(
        json.dumps(
            predictions_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_paths


def _format_metric(label: str, value: Any) -> str:
    return f"{label}: {float(value):.1%}"


def format_holdout_report(report: Mapping[str, Any]) -> str:
    """Return the concise terminal report, including development comparison."""

    summary = report.get("summary", {})
    development = report.get("development_set", {})
    development_summary = (
        development.get("summary", {}) if isinstance(development, Mapping) else {}
    )
    runtime = report.get("runtime_statistics", {})
    development_count = int(development_summary.get("total_cases", 0))
    holdout_count = int(summary.get("total_cases", 0))
    lines = [
        f"\nDevelopment Set ({development_count} cases)",
        "Step 8.6 development evaluation set",
        _format_metric("Aspect F1", development_summary.get("aspect_f1", 0.0)),
        (
            "Exact Case: "
            f"{int(development_summary.get('exact_matches', 0))}/"
            f"{int(development_summary.get('total_cases', 0))} "
            f"({float(development_summary.get('exact_case_accuracy', 0.0)):.1%})"
        ),
        f"\nBlind Holdout Set ({holdout_count} cases)",
        "Step 8.7 blind holdout set",
        _format_metric("Aspect Precision", summary.get("aspect_precision", 0.0)),
        _format_metric("Aspect Recall", summary.get("aspect_recall", 0.0)),
        _format_metric("Aspect F1", summary.get("aspect_f1", 0.0)),
        _format_metric(
            "Aspect-only Precision", summary.get("aspect_only_precision", 0.0)
        ),
        _format_metric("Aspect-only Recall", summary.get("aspect_only_recall", 0.0)),
        _format_metric("Aspect-only F1", summary.get("aspect_only_f1", 0.0)),
        _format_metric(
            "Sentiment Accuracy", summary.get("sentiment_accuracy", 0.0)
        ),
        _format_metric(
            "Evidence Validity", summary.get("evidence_validity_rate", 0.0)
        ),
        _format_metric("Evidence Match", summary.get("evidence_match_rate", 0.0)),
        (
            "Exact Case: "
            f"{int(summary.get('exact_matches', 0))}/"
            f"{int(summary.get('total_cases', 0))} "
            f"({float(summary.get('exact_case_accuracy', 0.0)):.1%})"
        ),
        (
            "TP / FP / FN: "
            f"{int(summary.get('aspect_true_positives', 0))} / "
            f"{int(summary.get('aspect_false_positives', 0))} / "
            f"{int(summary.get('aspect_false_negatives', 0))}"
        ),
        "\nGeneralization Gap (holdout - development)",
    ]
    gaps = report.get("generalization_gap", {})
    if isinstance(gaps, Mapping):
        for metric in GAP_METRICS:
            values = gaps.get(metric, {})
            if not isinstance(values, Mapping):
                continue
            lines.append(
                f"{metric}: {float(values.get('development', 0.0)):.1%} -> "
                f"{float(values.get('holdout', 0.0)):.1%} "
                f"({float(values.get('difference_percentage_points', 0.0)):+.1f} pp)"
            )
    lines.extend(
        [
            "\nPerformance",
            f"Total: {float(runtime.get('total_seconds', 0.0)):.2f}s",
            (
                "Average: "
                f"{float(runtime.get('average_seconds_per_review', 0.0)):.2f}s/review"
            ),
            f"Median: {float(runtime.get('median_seconds_per_review', 0.0)):.2f}s",
            f"P90: {float(runtime.get('p90_seconds_per_review', 0.0)):.2f}s",
            f"Retries: {int(runtime.get('retry_count', 0))}",
            (
                "Model/schema failures: "
                f"{int(runtime.get('model_failure_count', 0))}/"
                f"{int(runtime.get('schema_failure_count', 0))}"
            ),
            "Slowest five:",
            *[
                f"- {item.get('case_id', '')}: "
                f"{float(item.get('analysis_time_seconds', 0.0)):.2f}s"
                for item in runtime.get("slowest_five", [])
                if isinstance(item, Mapping)
            ],
            "\nTop 10 errors",
        ]
    )
    top_errors = report.get("top_errors", [])
    if not top_errors:
        lines.append("(none)")
    elif isinstance(top_errors, list):
        for item in top_errors:
            if not isinstance(item, Mapping):
                continue
            lines.extend(
                [
                    f"\n{item.get('case_id', '')} | {item.get('error_type', '')}",
                    f"Review: {item.get('review', '')}",
                    f"Expected: {_json_cell(item.get('expected', []))}",
                    f"Actual: {_json_cell(item.get('actual', []))}",
                    f"Explanation: {item.get('explanation', '')}",
                ]
            )
    return "\n".join(lines)
