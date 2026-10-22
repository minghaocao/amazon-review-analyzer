"""Offline-only Step 8.10 generalization error diagnostics.

This module treats every Step 8.9 artifact as immutable input.  It never imports
or creates an LLM client, opens a browser, or performs network I/O.  The study
explains existing Final Blind results; it does not change model behaviour or
evaluation metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aspect_taxonomy import STANDARD_ASPECTS
from config import PROJECT_ROOT


FINAL_DATASET_PATH = PROJECT_ROOT / "evaluation" / "final_blind_cases.json"
FINAL_MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "final_blind_manifest.json"
FINAL_FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT / "evaluation" / "final_blind_freeze_manifest.json"
)
FINAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "final_blind"
FINAL_PREDICTIONS_PATH = FINAL_OUTPUT_DIR / "final_blind_predictions.json"
FINAL_REPORT_PATH = FINAL_OUTPUT_DIR / "final_blind_evaluation.json"
FINAL_ERRORS_PATH = FINAL_OUTPUT_DIR / "final_blind_errors.csv"

DEVELOPMENT_DATASET_PATH = (
    PROJECT_ROOT / "evaluation" / "generalization_dev_cases.json"
)
DEVELOPMENT_MANIFEST_PATH = (
    PROJECT_ROOT / "evaluation" / "generalization_dev_manifest.json"
)
DEVELOPMENT_REPORT_PATH = (
    PROJECT_ROOT / "evaluation" / "evaluation_generalization_dev_qwen3_8b.json"
)

STUDY_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "evaluation" / "generalization_error_study"
)
GENERALIZATION_REPORT_PATH = STUDY_OUTPUT_DIR / "generalization_error_report.json"
CASE_ANALYSIS_PATH = STUDY_OUTPUT_DIR / "generalization_case_analysis.csv"
ERROR_SUMMARY_PATH = STUDY_OUTPUT_DIR / "generalization_error_summary.csv"
ASPECT_CONFUSION_PATH = STUDY_OUTPUT_DIR / "aspect_confusion_matrix.csv"
ASPECT_METRICS_PATH = STUDY_OUTPUT_DIR / "aspect_metrics.csv"
MULTI_MENTION_PATH = STUDY_OUTPUT_DIR / "same_aspect_multi_mention_analysis.csv"
SENTIMENT_ERROR_PATH = STUDY_OUTPUT_DIR / "sentiment_error_analysis.csv"
DIFFICULTY_PATH = STUDY_OUTPUT_DIR / "dataset_difficulty_comparison.json"
DEV_V2_SPEC_PATH = STUDY_OUTPUT_DIR / "generalization_dev_v2_spec.json"

EVALUATION_ONLY_NOTICE = (
    "Step 8.10 is an offline diagnostic study. No model optimization was "
    "performed and no Final Blind score was changed."
)

FINAL_METRIC_KEYS = (
    "aspect_precision",
    "aspect_recall",
    "aspect_f1",
    "aspect_only_precision",
    "aspect_only_recall",
    "aspect_only_f1",
    "sentiment_accuracy",
    "evidence_validity_rate",
    "evidence_match_rate",
    "exact_case_accuracy",
    "exact_matches",
    "aspect_true_positives",
    "aspect_false_positives",
    "aspect_false_negatives",
)

DEMO_FROZEN_PATHS = (
    PROJECT_ROOT / "data" / "raw_reviews.csv",
    PROJECT_ROOT / "outputs" / "analyzed_reviews.csv",
    PROJECT_ROOT / "outputs" / "aspect_summary.csv",
    PROJECT_ROOT / "outputs" / "business_summary.json",
    PROJECT_ROOT / "outputs" / "aspect_evidence.json",
    PROJECT_ROOT / "outputs" / "keyword_summary.csv",
    PROJECT_ROOT / "outputs" / "sentiment_keywords.json",
    PROJECT_ROOT / "outputs" / "aspect_keywords.json",
)

SECOND_LEVEL_CATEGORIES = (
    "ASPECT_BOUNDARY_CONFUSION",
    "OVER_EXTRACTION",
    "UNDER_EXTRACTION",
    "MENTION_OVER_SPLITTING",
    "MENTION_OVER_MERGING",
    "MULTI_ASPECT_SENTENCE_FAILURE",
    "NEGATION_SCOPE_ERROR",
    "CONTRAST_ERROR",
    "COMPARATIVE_ERROR",
    "TEMPORAL_CHANGE_ERROR",
    "WEAK_SENTIMENT_ERROR",
    "ATTRIBUTE_VS_EVALUATION_ERROR",
    "PRODUCT_VS_EXPERIENCE_CONFUSION",
    "OTHER_OVERUSE",
    "OTHER_UNDERUSE",
    "GOLD_AMBIGUITY",
    "UNRESOLVED",
)

PRODUCT_ASPECTS = frozenset(
    {
        "Picture Quality",
        "Sound Quality",
        "Price / Value",
        "Remote Control",
        "System Performance",
        "Ease of Use",
        "Design",
        "Build Quality",
        "Reliability",
        "Connectivity",
        "Setup / Installation",
    }
)
EXPERIENCE_ASPECTS = frozenset({"Delivery", "Packaging", "Customer Service"})

NEGATION_RE = re.compile(
    r"\b(?:not|no|never|didn['’]t|doesn['’]t|without|hardly|barely|cannot|can['’]t|isn['’]t|hasn['’]t)\b",
    re.IGNORECASE,
)
CONTRAST_RE = re.compile(
    r"\b(?:but|however|although|though|while|yet|even though|on the other hand)\b",
    re.IGNORECASE,
)
COMPARISON_RE = re.compile(
    r"\b(?:better than|worse than|cheaper than|more responsive than|compared (?:with|to)|unlike|less than|more than|twice as|than my|than the)\b",
    re.IGNORECASE,
)
TEMPORAL_RE = re.compile(
    r"\b(?:initially|at first|after (?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)|later|now|used to|since then|still|eventually|over time|every (?:day|night|week|month)|this week|overnight|day before)\b",
    re.IGNORECASE,
)
WEAK_SENTIMENT_RE = re.compile(
    r"\b(?:fine|okay|ok|decent|acceptable|adequate|ordinary|average|nothing special|not bad|could be better|entirely ordinary|in line with)\b",
    re.IGNORECASE,
)
class GeneralizationStudyError(ValueError):
    """The frozen inputs or offline diagnostic output are invalid."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_label(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_json(path: str | Path, *, expected: type | None = None) -> Any:
    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeneralizationStudyError(f"Required offline input was not found: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise GeneralizationStudyError(f"Invalid JSON input {resolved}: {exc}") from exc
    if expected is not None and not isinstance(payload, expected):
        raise GeneralizationStudyError(
            f"Unexpected JSON shape in {resolved}; expected {expected.__name__}."
        )
    return payload


def create_final_blind_freeze_manifest(
    destination: str | Path,
    *,
    dataset_path: str | Path,
    manifest_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
    errors_path: str | Path,
    project_root: str | Path | None = None,
    created_at: str | None = None,
    demo_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Create a portable hash manifest for already-existing Step 8.9 files."""

    destination_path = Path(destination)
    if destination_path.exists():
        raise GeneralizationStudyError(
            "Final Blind freeze manifest already exists and will not be overwritten."
        )
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    paths = {
        "dataset": Path(dataset_path),
        "manifest": Path(manifest_path),
        "predictions": Path(predictions_path),
        "evaluation_report": Path(report_path),
        "errors_csv": Path(errors_path),
    }
    for path in paths.values():
        if not path.is_file():
            raise GeneralizationStudyError(f"Cannot freeze missing Step 8.9 file: {path}")

    report = _read_json(paths["evaluation_report"], expected=dict)
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise GeneralizationStudyError("Final Blind report has no summary object.")
    timestamp = report.get("timestamp", summary.get("timestamp"))
    case_count = report.get("case_count", summary.get("total_cases"))
    if report.get("provider") != "ollama" or report.get("model") != "qwen3:8b":
        raise GeneralizationStudyError(
            "Final Blind provider/model must be ollama / qwen3:8b."
        )
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise GeneralizationStudyError("Final Blind report has no timestamp.")
    if not isinstance(case_count, int) or case_count <= 0:
        raise GeneralizationStudyError("Final Blind report has no valid case count.")
    missing_metrics = [
        key
        for key in FINAL_METRIC_KEYS
        if not isinstance(summary.get(key), (int, float))
    ]
    if missing_metrics:
        raise GeneralizationStudyError(
            "Final Blind report is missing required metrics: "
            + ", ".join(missing_metrics)
        )
    payload: dict[str, Any] = {
        "freeze_version": "step8.10",
        "purpose": (
            "Freeze Step 8.9 artifacts before the offline Step 8.10 diagnostic study."
        ),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "provider": report.get("provider", ""),
        "model": report.get("model", ""),
        "step8_9_timestamp": timestamp,
        "case_count": case_count,
        "metrics": {key: summary[key] for key in FINAL_METRIC_KEYS},
    }
    for name, path in paths.items():
        payload[f"{name}_path"] = _relative_label(path, root)
        payload[f"{name}_sha256"] = file_sha256(path)

    protected_demo_files: dict[str, str] = {}
    for raw_path in demo_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise GeneralizationStudyError(f"Cannot freeze missing Demo file: {path}")
        protected_demo_files[_relative_label(path, root)] = file_sha256(path)
    if protected_demo_files:
        payload["protected_demo_files"] = dict(sorted(protected_demo_files.items()))

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_final_blind_freeze(
    manifest_path: str | Path = FINAL_FREEZE_MANIFEST_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed if any frozen Step 8.9 byte has changed."""

    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    payload = _read_json(manifest_path, expected=dict)
    required_names = (
        "dataset",
        "manifest",
        "predictions",
        "evaluation_report",
        "errors_csv",
    )
    for name in required_names:
        path_value = payload.get(f"{name}_path")
        expected_hash = payload.get(f"{name}_sha256")
        if not isinstance(path_value, str) or not path_value:
            raise GeneralizationStudyError(f"Freeze manifest is missing {name}_path.")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise GeneralizationStudyError(f"Freeze manifest is missing {name}_sha256.")
        path = Path(path_value)
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            raise GeneralizationStudyError(f"Frozen Step 8.9 file is missing: {resolved}")
        actual_hash = file_sha256(resolved)
        if actual_hash != expected_hash:
            raise GeneralizationStudyError(
                f"Frozen Step 8.9 file changed: {path_value}."
            )
    if payload.get("provider") != "ollama" or payload.get("model") != "qwen3:8b":
        raise GeneralizationStudyError(
            "Frozen Step 8.9 provider/model must be ollama / qwen3:8b."
        )
    if not isinstance(payload.get("step8_9_timestamp"), str) or not str(
        payload.get("step8_9_timestamp")
    ).strip():
        raise GeneralizationStudyError("Freeze manifest has no Step 8.9 timestamp.")
    if not isinstance(payload.get("case_count"), int) or int(payload["case_count"]) <= 0:
        raise GeneralizationStudyError("Freeze manifest has no valid case count.")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise GeneralizationStudyError("Freeze manifest has no metrics object.")
    invalid_metrics = [
        key for key in FINAL_METRIC_KEYS if not isinstance(metrics.get(key), (int, float))
    ]
    if invalid_metrics:
        raise GeneralizationStudyError(
            "Freeze manifest is missing required metrics: "
            + ", ".join(invalid_metrics)
        )
    return dict(payload)


def verify_demo_freeze(
    freeze_manifest: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
    required: bool = True,
) -> dict[str, str]:
    """Verify the Step 10B.7 Demo files without reading or writing their contents."""

    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    recorded = freeze_manifest.get("protected_demo_files")
    if not isinstance(recorded, Mapping) or not recorded:
        if required:
            raise GeneralizationStudyError(
                "Freeze manifest has no protected Demo file hashes."
            )
        return {}
    verified: dict[str, str] = {}
    for label, expected in recorded.items():
        if not isinstance(label, str) or not isinstance(expected, str) or len(expected) != 64:
            raise GeneralizationStudyError("Freeze manifest has invalid Demo hash metadata.")
        candidate = Path(label)
        path = candidate if candidate.is_absolute() else root / candidate
        if not path.is_file():
            raise GeneralizationStudyError(f"Protected Demo file is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise GeneralizationStudyError(f"Protected Demo file changed: {label}.")
        verified[label] = actual
    return dict(sorted(verified.items()))


def _assert_frozen_input_paths(
    freeze_manifest: Mapping[str, Any],
    *,
    project_root: Path,
    dataset_path: Path,
    pre_evaluation_manifest_path: Path,
    predictions_path: Path,
    final_report_path: Path,
    errors_path: Path,
) -> None:
    """Prevent callers from verifying one frozen set and analyzing another."""

    supplied = {
        "dataset": dataset_path,
        "manifest": pre_evaluation_manifest_path,
        "predictions": predictions_path,
        "evaluation_report": final_report_path,
        "errors_csv": errors_path,
    }
    for name, path in supplied.items():
        recorded = freeze_manifest.get(f"{name}_path")
        if not isinstance(recorded, str) or not recorded:
            raise GeneralizationStudyError(f"Freeze manifest is missing {name}_path.")
        candidate = Path(recorded)
        expected = candidate if candidate.is_absolute() else project_root / candidate
        if path.resolve() != expected.resolve():
            raise GeneralizationStudyError(
                f"Supplied {name} path is not the frozen Step 8.9 file."
            )


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _rates(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _evidence_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(str(item) for item in value if isinstance(item, str))
    return ""


def _normalized_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def _aspect_counter(mentions: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(
        str(item.get("aspect", ""))
        for item in mentions
        if str(item.get("aspect", "")) in STANDARD_ASPECTS
    )


def _pair_counter(mentions: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter(
        (str(item.get("aspect", "")), str(item.get("sentiment", "")))
        for item in mentions
        if str(item.get("aspect", "")) in STANDARD_ASPECTS
    )


def _case_error_types(case: Mapping[str, Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for error in _list_of_mappings(case.get("errors")):
        value = str(error.get("error_type", ""))
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _count_mismatches(case: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    gold_mentions = _list_of_mappings(case.get("expected_aspects"))
    predicted_mentions = _list_of_mappings(case.get("actual_aspects"))
    gold = _aspect_counter(gold_mentions)
    predicted = _aspect_counter(predicted_mentions)
    over_split: set[str] = set()
    over_merged: set[str] = set()

    # Count differences caused by a taxonomy substitution are not mention
    # granularity errors.  Require an evaluator EXTRA/MISS record plus evidence
    # containment within a same-aspect Gold/predicted opinion.
    for error in _list_of_mappings(case.get("errors")):
        error_type = str(error.get("error_type", ""))
        if error_type == "EXTRA_ASPECT":
            aspect = str(error.get("actual_aspect", ""))
            extra_evidence = _normalized_text(error.get("actual_evidence", ""))
            same_gold = [
                _normalized_text(_evidence_text(item.get("evidence")))
                for item in gold_mentions
                if str(item.get("aspect", "")) == aspect
            ]
            if (
                aspect
                and gold[aspect] > 0
                and predicted[aspect] > gold[aspect]
                and extra_evidence
                and any(
                    extra_evidence in evidence or evidence in extra_evidence
                    for evidence in same_gold
                    if evidence
                )
            ):
                over_split.add(aspect)
        elif error_type == "MISS_ASPECT":
            aspect = str(error.get("expected_aspect", ""))
            missed_evidence = _normalized_text(error.get("expected_evidence", ""))
            same_predictions = [
                _normalized_text(_evidence_text(item.get("evidence")))
                for item in predicted_mentions
                if str(item.get("aspect", "")) == aspect
            ]
            if (
                aspect
                and gold[aspect] > predicted[aspect] > 0
                and missed_evidence
                and any(
                    missed_evidence in evidence or evidence in missed_evidence
                    for evidence in same_predictions
                    if evidence
                )
            ):
                over_merged.add(aspect)
    return sorted(over_split), sorted(over_merged)


def _has_duplicate_predictions(case: Mapping[str, Any]) -> bool:
    keys = [
        (
            str(item.get("aspect", "")),
            str(item.get("sentiment", "")),
            _normalized_text(_evidence_text(item.get("evidence"))),
        )
        for item in _list_of_mappings(case.get("actual_aspects"))
    ]
    return any(count > 1 for count in Counter(keys).values())


def _sentences(review: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+", review) if part.strip()]


def _has_multi_aspect_sentence(case: Mapping[str, Any]) -> bool:
    review = str(case.get("review", ""))
    gold = _list_of_mappings(case.get("expected_aspects"))
    for sentence in _sentences(review):
        sentence_aspects = {
            str(item.get("aspect", ""))
            for item in gold
            if _evidence_text(item.get("evidence"))
            and _evidence_text(item.get("evidence")) in sentence
        }
        if len(sentence_aspects) >= 2:
            return True
    return False


def _crosses_product_experience(gold: str, predicted: str) -> bool:
    return (gold in PRODUCT_ASPECTS and predicted in EXPERIENCE_ASPECTS) or (
        gold in EXPERIENCE_ASPECTS and predicted in PRODUCT_ASPECTS
    )


def _possible_gold_ambiguity(case: Mapping[str, Any]) -> bool:
    tags = " ".join(str(tag) for tag in case.get("tags", [])).casefold()
    notes = str(case.get("notes", "")).casefold()
    return any(marker in f"{tags} {notes}" for marker in ("ambiguous", "uncertain"))


def _affected_aspects(
    case: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    gold: set[str] = set()
    predicted: set[str] = set()
    for error in _list_of_mappings(case.get("errors")):
        expected = str(error.get("expected_aspect", ""))
        actual = str(error.get("actual_aspect", ""))
        if expected:
            gold.add(expected)
        if actual:
            predicted.add(actual)
    return sorted(gold), sorted(predicted)


def _affected_text(case: Mapping[str, Any]) -> str:
    review = str(case.get("review", ""))
    snippets: list[str] = []
    for error in _list_of_mappings(case.get("errors")):
        for field in ("expected_evidence", "actual_evidence"):
            value = _evidence_text(error.get(field))
            if value:
                snippets.append(value)
    affected_sentences = [
        sentence
        for sentence in _sentences(review)
        if any(snippet in sentence for snippet in snippets)
    ]
    return " ".join(affected_sentences or snippets or [review])


def _observed_linguistic_cues(text: str) -> list[str]:
    """Describe nearby syntax without claiming that it caused an error."""

    patterns = (
        ("negation", NEGATION_RE),
        ("contrast", CONTRAST_RE),
        ("comparison", COMPARISON_RE),
        ("temporal_change", TEMPORAL_RE),
        ("weak_sentiment", WEAK_SENTIMENT_RE),
    )
    return [name for name, pattern in patterns if pattern.search(text)]


def _sentiment_error_diagnostic(
    case: Mapping[str, Any], error: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify one sentiment mismatch from its aligned evidence only.

    Review-wide cues are intentionally excluded: their presence does not prove
    that they caused this particular mention error.
    """

    expected = str(error.get("expected_sentiment", ""))
    actual = str(error.get("actual_sentiment", ""))
    expected_evidence = _evidence_text(error.get("expected_evidence"))
    actual_evidence = _evidence_text(error.get("actual_evidence"))
    local_text = " ".join(
        part for part in (expected_evidence, actual_evidence) if part
    )
    aspect = str(error.get("expected_aspect", ""))
    gold_mentions = _list_of_mappings(case.get("expected_aspects"))
    same_aspect_polarities = {
        str(item.get("sentiment", ""))
        for item in gold_mentions
        if str(item.get("aspect", "")) == aspect
    }
    normalized_actual = _normalized_text(actual_evidence)
    alignment_disagreement = bool(normalized_actual) and any(
        _normalized_text(_evidence_text(item.get("evidence"))) == normalized_actual
        and (
            str(item.get("aspect", "")) != aspect
            or str(item.get("sentiment", "")) != expected
        )
        for item in gold_mentions
    )

    flags: list[str] = []
    if NEGATION_RE.search(local_text):
        flags.append("negation")
    if WEAK_SENTIMENT_RE.search(local_text):
        flags.append("weak_sentiment")
    if CONTRAST_RE.search(local_text):
        flags.append("contrast")
    if TEMPORAL_RE.search(local_text):
        flags.append("temporal_change")
    if COMPARISON_RE.search(local_text):
        flags.append("comparison")
    if {"positive", "negative"}.issubset(same_aspect_polarities):
        flags.append("mixed_polarity")
    if "neutral" in {expected, actual}:
        flags.append("neutral_boundary")

    if alignment_disagreement:
        primary = "other"
        confidence = "low"
        note = "Evidence aligns with a different Gold opinion; causal sentiment attribution is unsafe."
    elif "neutral_boundary" in flags:
        primary = "neutral_boundary"
        confidence = "high"
        note = "The mismatch crosses the explicit neutral-label boundary."
    else:
        primary_order = (
            "negation",
            "temporal_change",
            "comparison",
            "weak_sentiment",
            "mixed_polarity",
            "contrast",
        )
        primary = next((name for name in primary_order if name in flags), "other")
        confidence = "medium" if primary != "other" else "low"
        note = (
            "Primary category is supported by the aligned evidence span."
            if primary != "other"
            else "No evidence-local construction reliably explains the sentiment mismatch."
        )
    return {
        "primary_category": primary,
        "secondary_categories": [name for name in flags if name != primary],
        "classification_confidence": confidence,
        "needs_manual_review": confidence == "low",
        "alignment_disagreement": alignment_disagreement,
        "diagnostic_note": note,
    }


def classify_non_exact_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Assign transparent, multi-label root-cause categories to one case."""

    if bool(case.get("exact_match")):
        raise GeneralizationStudyError("Only non-exact cases can be classified.")
    original_error_types = _case_error_types(case)
    error_set = set(original_error_types)
    categories: set[str] = set()
    over_split, over_merged = _count_mismatches(case)
    wrong_aspects = _list_of_mappings(case.get("wrong_aspects"))
    affected_text = _affected_text(case)

    if wrong_aspects:
        categories.add("ASPECT_BOUNDARY_CONFUSION")
    if "EXTRA_ASPECT" in error_set:
        categories.add("OVER_EXTRACTION")
    if "MISS_ASPECT" in error_set:
        categories.add("UNDER_EXTRACTION")
    if over_split:
        categories.add("MENTION_OVER_SPLITTING")
    if over_merged:
        categories.add("MENTION_OVER_MERGING")
    # Linguistic categories are causal labels, not mere cue-presence labels.
    # Only evidence-local WRONG_SENTIMENT diagnostics may add them here.
    sentiment_category_map = {
        "negation": "NEGATION_SCOPE_ERROR",
        "contrast": "CONTRAST_ERROR",
        "comparison": "COMPARATIVE_ERROR",
        "temporal_change": "TEMPORAL_CHANGE_ERROR",
        "weak_sentiment": "WEAK_SENTIMENT_ERROR",
    }
    sentiment_diagnostics = [
        _sentiment_error_diagnostic(case, error)
        for error in _list_of_mappings(case.get("errors"))
        if error.get("error_type") == "WRONG_SENTIMENT"
    ]
    for diagnostic in sentiment_diagnostics:
        mapped = sentiment_category_map.get(str(diagnostic["primary_category"]))
        if mapped:
            categories.add(mapped)
    if _has_multi_aspect_sentence(case) and (
        error_set.intersection({"WRONG_ASPECT", "EXTRA_ASPECT", "MISS_ASPECT"})
        or any(
            bool(diagnostic.get("alignment_disagreement"))
            for diagnostic in sentiment_diagnostics
        )
    ):
        categories.add("MULTI_ASPECT_SENTENCE_FAILURE")

    gold_mentions = _list_of_mappings(case.get("expected_aspects"))
    actual_mentions = _list_of_mappings(case.get("actual_aspects"))
    if "EXTRA_ASPECT" in error_set and not gold_mentions and actual_mentions:
        # A Gold-empty review with predictions is the only high-confidence
        # attribute-vs-evaluation signal available without human adjudication.
        categories.add("ATTRIBUTE_VS_EVALUATION_ERROR")

    for wrong in wrong_aspects:
        gold_aspect = str(wrong.get("expected", ""))
        predicted_aspect = str(wrong.get("actual", ""))
        if _crosses_product_experience(gold_aspect, predicted_aspect):
            categories.add("PRODUCT_VS_EXPERIENCE_CONFUSION")
        if predicted_aspect == "Other" and gold_aspect != "Other":
            categories.add("OTHER_OVERUSE")
        if gold_aspect == "Other" and predicted_aspect != "Other":
            categories.add("OTHER_UNDERUSE")
    ambiguity = _possible_gold_ambiguity(case)
    if ambiguity:
        categories.add("GOLD_AMBIGUITY")

    primary_order = (
        "ASPECT_BOUNDARY_CONFUSION",
        "MENTION_OVER_SPLITTING",
        "MENTION_OVER_MERGING",
        "OVER_EXTRACTION",
        "UNDER_EXTRACTION",
        "NEGATION_SCOPE_ERROR",
        "TEMPORAL_CHANGE_ERROR",
        "COMPARATIVE_ERROR",
        "WEAK_SENTIMENT_ERROR",
        "CONTRAST_ERROR",
        "MULTI_ASPECT_SENTENCE_FAILURE",
        "ATTRIBUTE_VS_EVALUATION_ERROR",
        "GOLD_AMBIGUITY",
    )
    primary = next((item for item in primary_order if item in categories), "UNRESOLVED")
    ordered_categories = [item for item in SECOND_LEVEL_CATEGORIES if item in categories]
    if not ordered_categories:
        ordered_categories = ["UNRESOLVED"]
    secondary = [item for item in ordered_categories if item != primary]

    if ambiguity or primary == "UNRESOLVED":
        confidence = "low"
    elif wrong_aspects or over_split or over_merged:
        confidence = "high"
    else:
        confidence = "medium"
    needs_manual_review = confidence == "low" or any(
        bool(item.get("needs_manual_review")) for item in sentiment_diagnostics
    ) or (
        "WEAK_EVIDENCE_MATCH" in error_set and not wrong_aspects
    )
    affected_gold, affected_predicted = _affected_aspects(case)
    notes: list[str] = []
    if wrong_aspects:
        notes.append(
            "Directional taxonomy substitutions: "
            + ", ".join(
                f"{item.get('expected')} -> {item.get('actual')}"
                for item in wrong_aspects
            )
            + "."
        )
    if over_split:
        notes.append("Gold opinion span was split for: " + ", ".join(over_split) + ".")
    if over_merged:
        notes.append("Independent Gold spans were merged for: " + ", ".join(over_merged) + ".")
    if "EXTRA_ASPECT" in error_set and not over_split:
        notes.append("One or more predicted mentions lacked an independent Gold opinion.")
    if "MISS_ASPECT" in error_set and not over_merged:
        notes.append("One or more independently annotated Gold opinions were omitted.")
    if "WRONG_SENTIMENT" in error_set:
        notes.append("At least one same-Aspect mention received the wrong sentiment label.")
    if "WEAK_EVIDENCE_MATCH" in error_set:
        notes.append("Gold and predicted evidence spans did not align under the frozen metric.")
    if ambiguity:
        notes.append("Gold wording explicitly indicates possible annotation ambiguity.")
    observed_cues = _observed_linguistic_cues(affected_text)
    if observed_cues:
        notes.append(
            "Observed nearby linguistic cues (context only, not asserted causes): "
            + ", ".join(observed_cues)
            + "."
        )

    return {
        "case_id": str(case.get("case_id", "")),
        "review_text": str(case.get("review", "")),
        "gold_mentions": [dict(item) for item in gold_mentions],
        "predicted_mentions": [dict(item) for item in actual_mentions],
        "original_error_types": original_error_types,
        "primary_error_category": primary,
        "secondary_error_categories": secondary,
        "all_error_categories": ordered_categories,
        "affected_gold_aspects": affected_gold,
        "affected_predicted_aspects": affected_predicted,
        "possible_gold_ambiguity": ambiguity,
        "classification_confidence": confidence,
        "needs_manual_review": needs_manual_review,
        "observed_linguistic_cues": observed_cues,
        "over_split_aspects": over_split,
        "over_merged_aspects": over_merged,
        "notes": " ".join(notes) or "No reliable second-level cause was inferred.",
    }


def build_directional_confusions(
    case_results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use evaluator-paired WRONG_ASPECT records only; never infer pairings."""

    directional: dict[tuple[str, str], dict[str, Any]] = {}
    pair_level: dict[tuple[str, str], dict[str, Any]] = {}
    total = sum(
        len(_list_of_mappings(case.get("wrong_aspects"))) for case in case_results
    )
    for case in case_results:
        case_id = str(case.get("case_id", ""))
        for wrong in _list_of_mappings(case.get("wrong_aspects")):
            gold = str(wrong.get("expected", ""))
            predicted = str(wrong.get("actual", ""))
            if not gold or not predicted or gold == predicted:
                continue
            key = (gold, predicted)
            record = directional.setdefault(
                key,
                {
                    "gold_aspect": gold,
                    "predicted_aspect": predicted,
                    "count": 0,
                    "case_ids": [],
                },
            )
            record["count"] += 1
            if case_id not in record["case_ids"]:
                record["case_ids"].append(case_id)

            pair_key = tuple(sorted((gold, predicted)))
            pair = pair_level.setdefault(
                pair_key,
                {
                    "pair": " <-> ".join(pair_key),
                    "aspects": list(pair_key),
                    "count": 0,
                    "directions": Counter(),
                    "case_ids": [],
                },
            )
            pair["count"] += 1
            pair["directions"][f"{gold} -> {predicted}"] += 1
            if case_id not in pair["case_ids"]:
                pair["case_ids"].append(case_id)

    directional_rows = sorted(
        (
            {
                **record,
                "case_ids": sorted(record["case_ids"]),
                "percentage_of_wrong_aspect": _safe_divide(record["count"], total),
            }
            for record in directional.values()
        ),
        key=lambda item: (
            -int(item["count"]),
            str(item["gold_aspect"]),
            str(item["predicted_aspect"]),
        ),
    )
    pair_rows = sorted(
        (
            {
                **record,
                "directions": dict(sorted(record["directions"].items())),
                "case_ids": sorted(record["case_ids"]),
                "percentage_of_wrong_aspect": _safe_divide(record["count"], total),
            }
            for record in pair_level.values()
        ),
        key=lambda item: (-int(item["count"]), str(item["pair"])),
    )
    return directional_rows, pair_rows


def calculate_aspect_metrics(
    case_results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Calculate aspect-only and Aspect+Sentiment multiset metrics by label."""

    totals = {
        aspect: {
            "support": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "aspect_sentiment_tp": 0,
            "aspect_sentiment_fp": 0,
            "aspect_sentiment_fn": 0,
        }
        for aspect in STANDARD_ASPECTS
    }
    unknown: set[str] = set()
    for case in case_results:
        gold_mentions = _list_of_mappings(case.get("expected_aspects"))
        predicted_mentions = _list_of_mappings(
            case.get("evaluation_aspects", case.get("actual_aspects"))
        )
        for item in predicted_mentions:
            aspect = str(item.get("aspect", ""))
            if aspect and aspect not in STANDARD_ASPECTS:
                unknown.add(aspect)
        gold_aspects = _aspect_counter(gold_mentions)
        predicted_aspects = _aspect_counter(predicted_mentions)
        gold_pairs = _pair_counter(gold_mentions)
        predicted_pairs = _pair_counter(predicted_mentions)
        for aspect in STANDARD_ASPECTS:
            tp = min(gold_aspects[aspect], predicted_aspects[aspect])
            totals[aspect]["support"] += gold_aspects[aspect]
            totals[aspect]["tp"] += tp
            totals[aspect]["fp"] += predicted_aspects[aspect] - tp
            totals[aspect]["fn"] += gold_aspects[aspect] - tp
            gold_pair_count = sum(
                count for (label, _), count in gold_pairs.items() if label == aspect
            )
            predicted_pair_count = sum(
                count for (label, _), count in predicted_pairs.items() if label == aspect
            )
            pair_tp = sum(
                min(count, predicted_pairs[(label, sentiment)])
                for (label, sentiment), count in gold_pairs.items()
                if label == aspect
            )
            totals[aspect]["aspect_sentiment_tp"] += pair_tp
            totals[aspect]["aspect_sentiment_fp"] += predicted_pair_count - pair_tp
            totals[aspect]["aspect_sentiment_fn"] += gold_pair_count - pair_tp

    rows: list[dict[str, Any]] = []
    for index, aspect in enumerate(STANDARD_ASPECTS):
        values = totals[aspect]
        aspect_rates = _rates(values["tp"], values["fp"], values["fn"])
        pair_rates = _rates(
            values["aspect_sentiment_tp"],
            values["aspect_sentiment_fp"],
            values["aspect_sentiment_fn"],
        )
        rows.append(
            {
                "aspect": aspect,
                "taxonomy_index": index,
                **values,
                "precision": aspect_rates["precision"],
                "recall": aspect_rates["recall"],
                "f1": aspect_rates["f1"],
                "aspect_sentiment_precision": pair_rates["precision"],
                "aspect_sentiment_recall": pair_rates["recall"],
                "aspect_sentiment_f1": pair_rates["f1"],
                "metric_scope": "aspect_only_multiset",
            }
        )
    return rows, sorted(unknown)


def rank_aspect_stability(
    aspect_rows: Sequence[Mapping[str, Any]],
    *,
    count: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [dict(row) for row in aspect_rows if int(row.get("support", 0)) > 0]
    stable = sorted(
        eligible,
        key=lambda row: (
            -float(row.get("f1", 0.0)),
            -int(row.get("support", 0)),
            int(row.get("taxonomy_index", 0)),
        ),
    )[:count]
    difficult = sorted(
        eligible,
        key=lambda row: (
            float(row.get("f1", 0.0)),
            -int(row.get("support", 0)),
            int(row.get("taxonomy_index", 0)),
        ),
    )[:count]
    fields = ("aspect", "support", "tp", "fp", "fn", "precision", "recall", "f1")
    return (
        [{field: row[field] for field in fields} for row in stable],
        [{field: row[field] for field in fields} for row in difficult],
    )


def strict_exact_failure_breakdown(
    case_results: Sequence[Mapping[str, Any]],
    classifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    non_exact = [case for case in case_results if not bool(case.get("exact_match"))]
    presence = Counter()
    for case in non_exact:
        presence.update(set(_case_error_types(case)))
    primary = Counter(
        str(item.get("primary_error_category", "UNRESOLVED"))
        for item in classifications
    )
    count_mismatch = sum(
        len(_list_of_mappings(case.get("expected_aspects")))
        != len(_list_of_mappings(case.get("actual_aspects")))
        for case in non_exact
    )
    aspect_vector_mismatch = sum(
        _aspect_counter(_list_of_mappings(case.get("expected_aspects")))
        != _aspect_counter(_list_of_mappings(case.get("actual_aspects")))
        for case in non_exact
    )
    over_split = sum(bool(item.get("over_split_aspects")) for item in classifications)
    over_merge = sum(bool(item.get("over_merged_aspects")) for item in classifications)
    classification_primary = {
        "ASPECT_BOUNDARY_CONFUSION",
        "OVER_EXTRACTION",
        "UNDER_EXTRACTION",
        "PRODUCT_VS_EXPERIENCE_CONFUSION",
        "OTHER_OVERUSE",
        "OTHER_UNDERUSE",
        "ATTRIBUTE_VS_EVALUATION_ERROR",
    }
    granularity_primary = {"MENTION_OVER_SPLITTING", "MENTION_OVER_MERGING"}
    # These are an exclusive attribution of the primary cause. EXTRA/MISS
    # records caused by a detected split/merge are not counted again as a
    # classification failure.
    aspect_classification_cases = sum(
        str(item.get("primary_error_category", "")) in classification_primary
        for item in classifications
    )
    granularity_cases = sum(
        str(item.get("primary_error_category", "")) in granularity_primary
        for item in classifications
    )
    evidence_only = sum(
        bool(case.get("base_exact_match")) and not bool(case.get("exact_match"))
        for case in non_exact
    )
    multiple_types = sum(len(_case_error_types(case)) > 1 for case in non_exact)
    multiple_records = sum(
        len(_list_of_mappings(case.get("errors"))) > 1 for case in non_exact
    )
    ambiguity = sum(bool(item.get("possible_gold_ambiguity")) for item in classifications)
    conclusion = (
        "Aspect classification is the dominant source of strict-exact failure; "
        "high-confidence mention granularity failures form a smaller, separately "
        "attributed primary-cause group."
        if aspect_classification_cases > granularity_cases
        else "Mention granularity is at least as common as Aspect classification failure."
    )
    return {
        "non_exact_cases": len(non_exact),
        "case_presence_by_original_error": dict(sorted(presence.items())),
        "mutually_exclusive_primary_category": dict(sorted(primary.items())),
        "mention_count_mismatch_cases": count_mismatch,
        "aspect_count_vector_mismatch_cases": aspect_vector_mismatch,
        "high_confidence_over_split_cases": over_split,
        "high_confidence_over_merge_cases": over_merge,
        "aspect_classification_failure_cases": aspect_classification_cases,
        "mention_granularity_failure_cases": granularity_cases,
        "exclusive_primary_attribution": True,
        "evidence_only_strict_failures": evidence_only,
        "possible_gold_ambiguity_cases": ambiguity,
        "multiple_simultaneous_error_type_cases": multiple_types,
        "multiple_error_record_cases": multiple_records,
        "counts_are_nonexclusive_except_primary_category": True,
        "conclusion": conclusion,
    }


def analyze_same_aspect_multi_mentions(
    case_results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in case_results:
        gold_mentions = _list_of_mappings(case.get("expected_aspects"))
        actual_mentions = _list_of_mappings(case.get("actual_aspects"))
        gold_counts = _aspect_counter(gold_mentions)
        repeated = sorted(aspect for aspect, count in gold_counts.items() if count > 1)
        if not repeated:
            continue
        over_split, over_merge = _count_mismatches(case)
        errors = _list_of_mappings(case.get("errors"))
        repeated_set = set(repeated)
        relevant_errors = [
            error
            for error in errors
            if (
                str(error.get("expected_aspect", "")) in repeated_set
                or str(error.get("actual_aspect", "")) in repeated_set
            )
        ]
        wrong_sentiment = any(
            error.get("error_type") == "WRONG_SENTIMENT"
            and str(error.get("expected_aspect", "")) in repeated_set
            for error in errors
        )
        wrong_aspect = any(
            error.get("error_type") == "WRONG_ASPECT"
            and (
                str(error.get("expected_aspect", "")) in repeated_set
                or str(error.get("actual_aspect", "")) in repeated_set
            )
            for error in errors
        )
        explicit_missing = [
            error
            for error in errors
            if error.get("error_type") == "MISS_ASPECT"
            and str(error.get("expected_aspect", "")) in repeated_set
        ]
        total_repeated_gold = sum(gold_counts[aspect] for aspect in repeated)
        actual_counts = _aspect_counter(actual_mentions)
        repeated_predicted = sum(actual_counts[aspect] for aspect in repeated)
        one_missed = len(explicit_missing) == 1 and not wrong_aspect
        all_missed = total_repeated_gold > 0 and repeated_predicted == 0
        row = {
            "case_id": str(case.get("case_id", "")),
            "repeated_aspects": repeated,
            "gold_repeated_mention_count": total_repeated_gold,
            "predicted_repeated_mention_count": repeated_predicted,
            "whole_case_strict_exact": bool(case.get("exact_match")),
            # Reuse the frozen evaluator alignment/evidence decisions.  Exact
            # string equality would wrongly reject valid contained spans.
            "repeated_component_fully_correct": not relevant_errors,
            "one_mention_missed": one_missed,
            "all_repeated_mentions_missed": all_missed,
            "over_merged": bool(set(over_merge).intersection(repeated_set)),
            "over_split": bool(set(over_split).intersection(repeated_set)),
            "wrong_sentiment": wrong_sentiment,
            "wrong_aspect": wrong_aspect,
            "duplicate_generated": _has_duplicate_predictions(case),
        }
        rows.append(row)

    summary_fields = (
        "whole_case_strict_exact",
        "repeated_component_fully_correct",
        "one_mention_missed",
        "all_repeated_mentions_missed",
        "over_merged",
        "over_split",
        "wrong_sentiment",
        "wrong_aspect",
        "duplicate_generated",
    )
    summary = {
        "total_cases": len(rows),
        **{field: sum(bool(row[field]) for row in rows) for field in summary_fields},
    }
    summary["cases_with_repeated_component_failure"] = sum(
        not bool(row["repeated_component_fully_correct"]) for row in rows
    )
    return rows, summary


def classify_sentiment_errors(
    case_results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in case_results:
        for error in _list_of_mappings(case.get("errors")):
            if error.get("error_type") != "WRONG_SENTIMENT":
                continue
            expected = str(error.get("expected_sentiment", ""))
            actual = str(error.get("actual_sentiment", ""))
            expected_evidence = _evidence_text(error.get("expected_evidence"))
            actual_evidence = _evidence_text(error.get("actual_evidence"))
            aspect = str(error.get("expected_aspect", ""))
            diagnostic = _sentiment_error_diagnostic(case, error)
            rows.append(
                {
                    "case_id": str(case.get("case_id", "")),
                    "aspect": aspect,
                    "expected_sentiment": expected,
                    "predicted_sentiment": actual,
                    "expected_evidence": expected_evidence,
                    "predicted_evidence": actual_evidence,
                    **diagnostic,
                }
            )
    counts = Counter(str(row["primary_category"]) for row in rows)
    refs: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in refs[str(row["primary_category"])]:
            refs[str(row["primary_category"])].append(case_id)
    categories = (
        "negation",
        "weak_sentiment",
        "contrast",
        "temporal_change",
        "comparison",
        "mixed_polarity",
        "neutral_boundary",
        "other",
    )
    summary = {
        "total_wrong_sentiment_mentions": len(rows),
        "primary_category_counts": {name: counts[name] for name in categories},
        "case_refs": {name: sorted(refs[name]) for name in categories},
        "classification_policy": "Each wrong-sentiment mention has one primary category; secondary cues may overlap.",
    }
    return rows, summary


def analyze_weak_evidence(
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in case_results:
        gold_mentions = _list_of_mappings(case.get("expected_aspects"))
        for error in _list_of_mappings(case.get("errors")):
            if error.get("error_type") != "WEAK_EVIDENCE_MATCH":
                continue
            expected = _evidence_text(error.get("expected_evidence"))
            actual = _evidence_text(error.get("actual_evidence"))
            normalized_expected = _normalized_text(expected)
            normalized_actual = _normalized_text(actual)
            if normalized_expected and normalized_expected in normalized_actual:
                category = "span_too_broad"
            elif normalized_actual and normalized_actual in normalized_expected:
                category = "span_too_narrow"
            elif any(
                _normalized_text(_evidence_text(item.get("evidence")))
                == normalized_actual
                and str(item.get("aspect", ""))
                != str(error.get("expected_aspect", ""))
                for item in gold_mentions
            ):
                category = "none_of_requested_categories"
                alignment_note = "Predicted span belongs to a different Gold opinion."
                alignment_subtype = "cross_opinion_alignment"
            elif _count_mismatches(case)[0] or _count_mismatches(case)[1]:
                category = "mention_granularity_disagreement"
                alignment_note = "Evidence mismatch overlaps a mention-count disagreement."
                alignment_subtype = ""
            else:
                category = "correct_meaning_different_gold_span"
                alignment_note = "Meaning is related, but the frozen Gold span differs."
                alignment_subtype = ""
            if category in {"span_too_broad", "span_too_narrow"}:
                alignment_note = "Predicted and Gold spans have a direct containment relation."
                alignment_subtype = ""
            records.append(
                {
                    "case_id": str(case.get("case_id", "")),
                    "aspect": str(error.get("expected_aspect", "")),
                    "expected_evidence": expected,
                    "predicted_evidence": actual,
                    "similarity": float(error.get("similarity", 0.0) or 0.0),
                    "classification": category,
                    "alignment_subtype": alignment_subtype,
                    "note": alignment_note,
                }
            )
    counts = Counter(str(record["classification"]) for record in records)
    return {
        "weak_evidence_count": len(records),
        "classification_counts": dict(sorted(counts.items())),
        "records": records,
        "requested_categories": [
            "span_too_broad",
            "span_too_narrow",
            "correct_meaning_different_gold_span",
            "mention_granularity_disagreement",
        ],
        "fallback_policy": (
            "Use none_of_requested_categories when the frozen alignment is a "
            "cross-opinion swap rather than a span-width or granularity issue."
        ),
        "optimization_performed": False,
    }


def dataset_difficulty_statistics(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe Gold dataset structure with identical rules for both sets."""

    review_count = len(cases)
    word_counts: list[int] = []
    mention_counts: list[int] = []
    repeated_cases = 0
    mixed_cases = 0
    same_aspect_mixed_cases = 0
    neutral_mentions = 0
    total_mentions = 0
    structures = Counter()
    cardinality = Counter()
    long_reviews = 0
    for case in cases:
        review = str(case.get("review", case.get("review_text", "")))
        mentions = _list_of_mappings(case.get("expected_aspects"))
        words = re.findall(r"\b[\w’'-]+\b", review, flags=re.UNICODE)
        word_counts.append(len(words))
        mention_counts.append(len(mentions))
        total_mentions += len(mentions)
        neutral_mentions += sum(
            str(item.get("sentiment", "")) == "neutral" for item in mentions
        )
        count = len(mentions)
        cardinality[
            "0" if count == 0 else "1" if count == 1 else "2" if count == 2 else "3-5"
        ] += 1
        aspect_counts = _aspect_counter(mentions)
        repeated_cases += int(any(value > 1 for value in aspect_counts.values()))
        sentiments = {str(item.get("sentiment", "")) for item in mentions}
        mixed_cases += int({"positive", "negative"}.issubset(sentiments))
        same_aspect_mixed_cases += int(
            any(
                {"positive", "negative"}.issubset(
                    {
                        str(item.get("sentiment", ""))
                        for item in mentions
                        if str(item.get("aspect", "")) == aspect
                    }
                )
                for aspect in aspect_counts
            )
        )
        structures["negation"] += int(bool(NEGATION_RE.search(review)))
        structures["contrast"] += int(bool(CONTRAST_RE.search(review)))
        structures["comparison"] += int(bool(COMPARISON_RE.search(review)))
        structures["temporal_language"] += int(bool(TEMPORAL_RE.search(review)))
        structures["multi_aspect_sentence"] += int(_has_multi_aspect_sentence(case))
        long_reviews += int(len(words) >= 35)

    def count_rate(count: int) -> dict[str, Any]:
        return {"count": count, "ratio": _safe_divide(count, review_count)}

    return {
        "review_count": review_count,
        "average_words_per_review": (
            statistics.mean(word_counts) if word_counts else 0.0
        ),
        "median_words_per_review": (
            statistics.median(word_counts) if word_counts else 0.0
        ),
        "gold_mentions": total_mentions,
        "mentions_per_review": _safe_divide(total_mentions, review_count),
        "mention_cardinality": {
            key: count_rate(cardinality[key]) for key in ("0", "1", "2", "3-5")
        },
        "same_aspect_repeated_mentions": count_rate(repeated_cases),
        "mixed_positive_negative_cases": count_rate(mixed_cases),
        "same_aspect_mixed_polarity_cases": count_rate(same_aspect_mixed_cases),
        "neutral_mentions": neutral_mentions,
        "neutral_mention_ratio": _safe_divide(neutral_mentions, total_mentions),
        "language_structures": {
            key: count_rate(structures[key])
            for key in (
                "negation",
                "contrast",
                "comparison",
                "temporal_language",
                "multi_aspect_sentence",
            )
        },
        "long_reviews_35_plus_words": count_rate(long_reviews),
        "calculation_method": {
            "words": "Unicode word-token regex: \\b[\\w’'-]+\\b",
            "repeated_mentions": "A case where a Gold Aspect label occurs more than once.",
            "mixed_positive_negative": "Any case containing both positive and negative Gold mentions.",
            "multi_aspect_sentence": "A sentence containing exact Gold evidence for at least two distinct Aspects.",
            "language_frequency": "Case-level regex presence; each case counts at most once per structure.",
            "long_review_threshold_words": 35,
        },
    }


def compare_dataset_difficulty(
    development_cases: Sequence[Mapping[str, Any]],
    final_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    development = dataset_difficulty_statistics(development_cases)
    final = dataset_difficulty_statistics(final_cases)
    scalar_fields = (
        "average_words_per_review",
        "median_words_per_review",
        "mentions_per_review",
        "neutral_mention_ratio",
    )
    differences: dict[str, Any] = {
        field: float(final[field]) - float(development[field]) for field in scalar_fields
    }
    for key in ("0", "1", "2", "3-5"):
        differences[f"mention_cardinality_{key}_ratio"] = (
            float(final["mention_cardinality"][key]["ratio"])
            - float(development["mention_cardinality"][key]["ratio"])
        )
    for key in (
        "same_aspect_repeated_mentions",
        "mixed_positive_negative_cases",
        "same_aspect_mixed_polarity_cases",
        "long_reviews_35_plus_words",
    ):
        differences[f"{key}_ratio"] = (
            float(final[key]["ratio"]) - float(development[key]["ratio"])
        )
    for key in development["language_structures"]:
        differences[f"{key}_ratio"] = (
            float(final["language_structures"][key]["ratio"])
            - float(development["language_structures"][key]["ratio"])
        )
    materially_harder = [
        name
        for name, value in differences.items()
        if value > (2.0 if "words" in name else 0.05)
    ]
    materially_lower = [
        name
        for name, value in differences.items()
        if value < (-2.0 if "words" in name else -0.05)
    ]
    return {
        "development": development,
        "final_blind": final,
        "final_minus_development": differences,
        "features_materially_higher_in_final": materially_harder,
        "features_materially_lower_in_final": materially_lower,
        "interpretation": (
            "Final Blind is longer and has more high-cardinality and repeated-Aspect "
            "cases, but not every measured language structure is more frequent."
        ),
    }


def build_development_final_gap_diagnosis(
    development_report: Mapping[str, Any],
    final_report: Mapping[str, Any],
    development_aspects: Sequence[Mapping[str, Any]],
    final_aspects: Sequence[Mapping[str, Any]],
    classifications: Sequence[Mapping[str, Any]],
    difficulty: Mapping[str, Any],
) -> dict[str, Any]:
    development_summary = development_report.get("summary", {})
    final_summary = final_report.get("summary", {})
    if not isinstance(development_summary, Mapping) or not isinstance(final_summary, Mapping):
        raise GeneralizationStudyError("Development or Final report has no summary.")
    development_by_aspect = {
        str(row.get("aspect", "")): row for row in development_aspects
    }
    final_by_aspect = {str(row.get("aspect", "")): row for row in final_aspects}
    aspect_drops = []
    for aspect in STANDARD_ASPECTS:
        dev = development_by_aspect.get(aspect, {})
        final = final_by_aspect.get(aspect, {})
        aspect_drops.append(
            {
                "aspect": aspect,
                "development_f1": float(dev.get("f1", 0.0)),
                "final_blind_f1": float(final.get("f1", 0.0)),
                "f1_change": float(final.get("f1", 0.0)) - float(dev.get("f1", 0.0)),
                "development_support": int(dev.get("support", 0)),
                "final_blind_support": int(final.get("support", 0)),
            }
        )
    aspect_drops.sort(key=lambda row: (float(row["f1_change"]), str(row["aspect"])))
    category_counts = Counter(
        category
        for row in classifications
        for category in row.get("all_error_categories", [])
    )
    final_cases = _list_of_mappings(final_report.get("cases"))
    gold_mentions = sum(
        len(_list_of_mappings(case.get("expected_aspects"))) for case in final_cases
    )
    predicted_mentions = sum(
        len(_list_of_mappings(case.get("actual_aspects"))) for case in final_cases
    )
    gaps = {
        metric: float(final_summary.get(metric, 0.0))
        - float(development_summary.get(metric, 0.0))
        for metric in (
            "aspect_precision",
            "aspect_recall",
            "aspect_f1",
            "aspect_only_f1",
            "sentiment_accuracy",
            "exact_case_accuracy",
        )
    }
    findings = [
        {
            "finding": "taxonomy_boundary_generalization",
            "evidence": {
                "wrong_aspect_case_count": sum(
                    "WRONG_ASPECT" in row.get("original_error_types", [])
                    for row in classifications
                ),
                "wrong_aspect_record_count": int(
                    final_report.get("error_counts", {}).get("WRONG_ASPECT", 0)
                ),
                "largest_aspect_f1_drops": aspect_drops[:6],
            },
            "interpretation": (
                "Several taxonomy boundaries that were stable in development did not "
                "transfer to new phrasing and product contexts."
            ),
        },
        {
            "finding": "mild_over_extraction_and_splitting_tendency",
            "evidence": {
                "gold_mentions": gold_mentions,
                "predicted_mentions": predicted_mentions,
                "predicted_minus_gold": predicted_mentions - gold_mentions,
                "extra_aspect_records": int(
                    final_report.get("error_counts", {}).get("EXTRA_ASPECT", 0)
                ),
                "miss_aspect_records": int(
                    final_report.get("error_counts", {}).get("MISS_ASPECT", 0)
                ),
                "over_split_cases": category_counts["MENTION_OVER_SPLITTING"],
                "over_merge_cases": category_counts["MENTION_OVER_MERGING"],
            },
            "interpretation": (
                "The model predicted more mentions than Gold and more often split or "
                "added opinions than it omitted them."
            ),
        },
        {
            "finding": "dataset_structure_shift",
            "evidence": difficulty.get("final_minus_development", {}),
            "interpretation": (
                "Final Blind contains longer reviews and more repeated/high-cardinality "
                "Gold structures, while some other measured structures are not higher."
            ),
        },
        {
            "finding": "other_and_implicit_boundary_instability",
            "evidence": {
                "other_overuse_cases": category_counts["OTHER_OVERUSE"],
                "other_underuse_cases": category_counts["OTHER_UNDERUSE"],
                "other_final_metrics": final_by_aspect.get("Other", {}),
                "price_value_final_metrics": final_by_aspect.get("Price / Value", {}),
            },
            "interpretation": (
                "The open-ended Other boundary and implicit value language generalize "
                "poorly beyond development examples."
            ),
        },
    ]
    return {
        "development_label": "Step 8.8 development",
        "final_label": "Step 8.9 final blind",
        "metric_gaps_final_minus_development": gaps,
        "largest_aspect_only_f1_drops": aspect_drops[:10],
        "findings": findings,
        "conclusion": (
            "The 94.70% to 73.42% Aspect+Sentiment F1 gap is best explained by "
            "taxonomy-boundary transfer, an over-extraction/splitting tendency, and "
            "a harder mix of longer and repeated-mention reviews. This is a diagnostic "
            "finding, not a post-Final model adjustment."
        ),
        "overfitting_claim": (
            "The evidence supports development-set coverage and lexical-boundary "
            "dependence; it does not by itself prove generic statistical overfitting."
        ),
    }


def build_generalization_dev_v2_spec(
    pair_confusions: Sequence[Mapping[str, Any]],
    aspect_metrics: Sequence[Mapping[str, Any]],
    *,
    classifications: Sequence[Mapping[str, Any]] = (),
    difficulty_comparison: Mapping[str, Any] | None = None,
    repeated_summary: Mapping[str, Any] | None = None,
    sentiment_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe a future dataset only; never generate cases or copy Final text."""

    target_cases = 110
    target_mentions = 210
    metrics_by_aspect = {
        str(row.get("aspect", "")): row for row in aspect_metrics
    }
    confusion_pressure: Counter[str] = Counter()
    for row in pair_confusions:
        count = int(row.get("count", 0))
        for aspect in row.get("aspects", []):
            if str(aspect) in STANDARD_ASPECTS:
                confusion_pressure[str(aspect)] += count
    weights: dict[str, float] = {}
    for aspect in STANDARD_ASPECTS:
        row = metrics_by_aspect.get(aspect, {})
        support = int(row.get("support", 0))
        errors = int(row.get("fp", 0)) + int(row.get("fn", 0))
        f1 = float(row.get("f1", 0.0))
        weights[aspect] = max(
            1.0,
            support + (2 * errors) + (2 * confusion_pressure[aspect]) + (4 * (1 - f1)),
        )
    minimum_per_aspect = 8
    remainder = target_mentions - (minimum_per_aspect * len(STANDARD_ASPECTS))
    weight_total = sum(weights.values())
    raw_allocations = {
        aspect: remainder * weights[aspect] / weight_total for aspect in STANDARD_ASPECTS
    }
    aspect_targets = {
        aspect: minimum_per_aspect + int(math.floor(raw_allocations[aspect]))
        for aspect in STANDARD_ASPECTS
    }
    unallocated = target_mentions - sum(aspect_targets.values())
    allocation_order = sorted(
        STANDARD_ASPECTS,
        key=lambda aspect: (
            -(raw_allocations[aspect] - math.floor(raw_allocations[aspect])),
            list(STANDARD_ASPECTS).index(aspect),
        ),
    )
    for aspect in allocation_order[:unallocated]:
        aspect_targets[aspect] += 1
    if sum(aspect_targets.values()) != 210:
        raise GeneralizationStudyError("Dev V2 aspect targets must sum to 210.")
    confusion_targets = [
        {
            "pair": str(row.get("pair", "")),
            "observed_final_count": int(row.get("count", 0)),
            "target_cases": 4 if int(row.get("count", 0)) >= 2 else 3,
        }
        for row in pair_confusions[:10]
    ]
    difficult = sorted(
        (
            row for row in aspect_metrics if int(row.get("support", 0)) > 0
        ),
        key=lambda row: (float(row.get("f1", 0.0)), -int(row.get("support", 0))),
    )[:6]
    category_counts = Counter(
        category
        for row in classifications
        for category in row.get("all_error_categories", [])
    )
    difficulty = (difficulty_comparison or {}).get("final_blind", {})
    repeated = repeated_summary or {}
    sentiment = sentiment_summary or {}
    final_review_count = max(int(difficulty.get("review_count", 0)), 1)

    def scaled_structure(name: str, floor: int, extra: int = 0) -> int:
        structure = difficulty.get("language_structures", {}).get(name, {})
        observed = int(structure.get("count", 0))
        scaled = int(round(observed * target_cases / final_review_count)) + extra
        return min(target_cases, max(floor, scaled))

    cardinality = difficulty.get("mention_cardinality", {})
    no_aspect = min(20, max(
        10,
        int(round(int(cardinality.get("0", {}).get("count", 0)) * target_cases / final_review_count)),
    ))
    one_aspect = min(40, max(
        25,
        int(round(int(cardinality.get("1", {}).get("count", 0)) * target_cases / final_review_count)),
    ))
    three_to_five = min(30, max(
        20,
        int(round(int(cardinality.get("3-5", {}).get("count", 0)) * target_cases / final_review_count)),
    ))
    two_aspect = target_cases - no_aspect - one_aspect - three_to_five
    if two_aspect < 0:
        raise GeneralizationStudyError("Derived Dev V2 case cardinality is invalid.")

    repeated_target = min(
        target_cases,
        max(
            24,
            int(round(int(repeated.get("total_cases", 0)) * target_cases / final_review_count))
            + int(repeated.get("cases_with_repeated_component_failure", 0)),
        ),
    )
    wrong_sentiment_counts = sentiment.get("primary_category_counts", {})
    weak_error_count = int(wrong_sentiment_counts.get("weak_sentiment", 0))
    product_experience_target = max(
        15, 4 * category_counts["PRODUCT_VS_EXPERIENCE_CONFUSION"]
    )
    product_experience_target = min(target_cases, product_experience_target)
    taxonomy_boundary_target = min(
        target_cases,
        max(30, category_counts["ASPECT_BOUNDARY_CONFUSION"] + 10),
    )

    return {
        "specification_only": True,
        "generation_status": "No cases were generated in Step 8.10.",
        "target_cases": target_cases,
        "target_mentions": target_mentions,
        "target_aspect_distribution": aspect_targets,
        "aspect_allocation_method": (
            "Minimum coverage plus observed support, FP/FN, taxonomy confusion, "
            "and low-F1 pressure from the frozen Final Blind diagnostics."
        ),
        "case_cardinality_distribution": {
            "no_aspect": no_aspect,
            "one_aspect_mention": one_aspect,
            "two_aspect_mentions": two_aspect,
            "three_to_five_aspect_mentions": three_to_five,
        },
        "boundary_case_distribution": {
            "taxonomy_boundary_cases": taxonomy_boundary_target,
            "other_boundary_cases": max(10, aspect_targets.get("Other", 0)),
            "price_value_implicit_or_comparative_cases": max(
                12, aspect_targets.get("Price / Value", 0)
            ),
            "product_vs_experience_cases": product_experience_target,
            "top_observed_pair_targets": confusion_targets,
        },
        "overlapping_language_and_structure_targets": {
            "same_aspect_multi_mention": repeated_target,
            "multi_aspect_sentences": scaled_structure("multi_aspect_sentence", 30),
            "mixed_positive_negative": min(
                target_cases,
                max(
                    35,
                    int(
                        round(
                            int(difficulty.get("mixed_positive_negative_cases", {}).get("count", 0))
                            * target_cases
                            / final_review_count
                        )
                    ),
                ),
            ),
            "same_aspect_mixed_polarity": min(
                target_cases,
                max(
                    16,
                    int(
                        round(
                            int(difficulty.get("same_aspect_mixed_polarity_cases", {}).get("count", 0))
                            * target_cases
                            / final_review_count
                        )
                    ),
                ),
            ),
            "negation": scaled_structure("negation", 22),
            "contrast": scaled_structure("contrast", 28),
            "comparison": scaled_structure("comparison", 15),
            "weak_sentiment": max(16, weak_error_count * 5),
            "temporal_change": scaled_structure("temporal_language", 18),
            "no_aspect": no_aspect,
            "other_cases": max(10, aspect_targets.get("Other", 0)),
            "experience_vs_product": product_experience_target,
            "long_reviews": max(
                20,
                int(
                    round(
                        int(difficulty.get("long_reviews_35_plus_words", {}).get("count", 0))
                        * target_cases
                        / final_review_count
                    )
                ),
            ),
            "ambiguous_cases_for_dual_human_adjudication": max(
                8, category_counts["GOLD_AMBIGUITY"]
            ),
        },
        "observed_final_error_basis": {
            "second_level_case_counts": dict(sorted(category_counts.items())),
            "same_aspect_repeated_case_count": int(repeated.get("total_cases", 0)),
            "same_aspect_repeated_failure_count": int(
                repeated.get("cases_with_repeated_component_failure", 0)
            ),
            "sentiment_primary_category_counts": dict(wrong_sentiment_counts),
        },
        "priority_aspects_from_final_metrics": [
            {
                "aspect": row.get("aspect"),
                "final_support": row.get("support"),
                "final_f1": row.get("f1"),
            }
            for row in difficult
        ],
        "design_principles": [
            "Learn error types and taxonomy boundaries, never Final Blind answers.",
            "Do not copy any Step 8.9 review sentence, Gold Evidence span, or case ID.",
            "Do not create a superficial paraphrase by replacing isolated adjectives.",
            "Use new products, discourse structures, and vocabulary for every case.",
            "Separate atomic opinion cardinality from sentence count.",
            "Include independent human annotation and adjudication before evaluation.",
            "Keep a future holdout disjoint from Dev V2 and freeze it before inference.",
        ],
        "overlap_note": (
            "Language and boundary targets intentionally overlap; their counts do not "
            "sum to target_cases."
        ),
        "forbidden_sources": [
            "Step 8.9 exact review phrasing",
            "Step 8.9 Gold Evidence",
            "Step 8.9 case IDs",
            "near-paraphrases of Step 8.9 cases",
        ],
    }


def _load_offline_inputs(
    *,
    dataset_path: Path,
    pre_evaluation_manifest_path: Path,
    predictions_path: Path,
    final_report_path: Path,
    errors_path: Path,
    development_dataset_path: Path,
    development_manifest_path: Path,
    development_report_path: Path,
    freeze_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    final_cases = _read_json(dataset_path, expected=list)
    pre_manifest = _read_json(pre_evaluation_manifest_path, expected=dict)
    predictions = _read_json(predictions_path, expected=dict)
    final_report = _read_json(final_report_path, expected=dict)
    development_cases = _read_json(development_dataset_path, expected=list)
    development_manifest = _read_json(development_manifest_path, expected=dict)
    development_report = _read_json(development_report_path, expected=dict)
    try:
        with errors_path.open(encoding="utf-8", newline="") as handle:
            error_rows = list(csv.DictReader(handle))
    except FileNotFoundError as exc:
        raise GeneralizationStudyError(
            f"Required Final Blind errors CSV was not found: {errors_path}"
        ) from exc

    dataset_hash = file_sha256(dataset_path)
    if pre_manifest.get("sha256") != dataset_hash:
        raise GeneralizationStudyError(
            "Final Blind pre-evaluation manifest does not match the frozen dataset."
        )
    if final_report.get("dataset_sha256") != dataset_hash:
        raise GeneralizationStudyError("Final Blind report dataset hash is inconsistent.")
    if predictions.get("dataset_sha256") != dataset_hash:
        raise GeneralizationStudyError(
            "Final Blind predictions dataset hash is inconsistent."
        )
    provider_model = {
        (str(final_report.get("provider", "")), str(final_report.get("model", ""))),
        (str(predictions.get("provider", "")), str(predictions.get("model", ""))),
        (str(freeze_manifest.get("provider", "")), str(freeze_manifest.get("model", ""))),
    }
    if provider_model != {("ollama", "qwen3:8b")}:
        raise GeneralizationStudyError("Final Blind provider/model metadata is inconsistent.")

    report_cases = _list_of_mappings(final_report.get("cases"))
    prediction_rows = _list_of_mappings(predictions.get("predictions"))
    dataset_ids = [str(case.get("case_id", "")) for case in final_cases]
    report_ids = [str(case.get("case_id", "")) for case in report_cases]
    prediction_ids = [str(case.get("case_id", "")) for case in prediction_rows]
    expected_count = int(freeze_manifest.get("case_count", 0))
    if not expected_count or not (
        len(final_cases)
        == len(report_cases)
        == len(prediction_rows)
        == expected_count
    ):
        raise GeneralizationStudyError("Frozen Final Blind case counts are inconsistent.")
    if dataset_ids != report_ids or dataset_ids != prediction_ids:
        raise GeneralizationStudyError("Frozen Final Blind case IDs are not aligned.")
    if len(error_rows) != sum(
        len(_list_of_mappings(case.get("errors"))) for case in report_cases
    ):
        raise GeneralizationStudyError(
            "Final Blind error CSV does not match structured report errors."
        )
    if any(row.get("case_id") not in set(dataset_ids) for row in error_rows):
        raise GeneralizationStudyError("Final Blind error CSV contains an unknown case ID.")

    summary = final_report.get("summary", {})
    frozen_metrics = freeze_manifest.get("metrics", {})
    if not isinstance(summary, Mapping) or not isinstance(frozen_metrics, Mapping):
        raise GeneralizationStudyError("Final Blind summary or frozen metrics are missing.")
    for name in FINAL_METRIC_KEYS:
        expected = frozen_metrics.get(name)
        actual = summary.get(name)
        if not isinstance(expected, (int, float)) or not isinstance(actual, (int, float)):
            raise GeneralizationStudyError(
                f"Frozen Final Blind metric is missing or non-numeric: {name}."
            )
        if not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-12):
            raise GeneralizationStudyError(
                f"Frozen Final Blind metric changed: {name}."
            )

    development_hash = file_sha256(development_dataset_path)
    if development_manifest.get("sha256") != development_hash:
        raise GeneralizationStudyError(
            "Step 8.8 manifest does not match the Development dataset."
        )
    if development_report.get("dataset_sha256") != development_hash:
        raise GeneralizationStudyError(
            "Step 8.8 report does not match the Development dataset."
        )
    if (
        development_report.get("provider") != "ollama"
        or development_report.get("model") != "qwen3:8b"
        or development_report.get("stage") != "optimized"
    ):
        raise GeneralizationStudyError(
            "Step 8.8 report must be the optimized ollama / qwen3:8b run."
        )
    development_results = _list_of_mappings(development_report.get("cases"))
    development_ids = [str(case.get("case_id", "")) for case in development_cases]
    development_report_ids = [
        str(case.get("case_id", "")) for case in development_results
    ]
    manifest_count = development_manifest.get("case_count")
    report_count = development_report.get("case_count")
    if not isinstance(manifest_count, int) or not isinstance(report_count, int) or not (
        len(development_cases)
        == len(development_results)
        == manifest_count
        == report_count
    ):
        raise GeneralizationStudyError("Step 8.8 case counts are inconsistent.")
    if development_ids != development_report_ids:
        raise GeneralizationStudyError("Step 8.8 case IDs are not aligned.")

    # Join annotation metadata from the authoritative combined Gold dataset.
    gold_by_id = {str(case.get("case_id", "")): case for case in final_cases}
    enriched_report_cases: list[dict[str, Any]] = []
    for case in report_cases:
        case_id = str(case.get("case_id", ""))
        gold = gold_by_id[case_id]
        enriched_report_cases.append(
            {
                **dict(case),
                "tags": list(gold.get("tags", [])),
                "notes": str(gold.get("notes", case.get("notes", ""))),
            }
        )
    final_report = {**final_report, "cases": enriched_report_cases}

    return {
        "final_cases": final_cases,
        "pre_evaluation_manifest": pre_manifest,
        "predictions": predictions,
        "final_report": final_report,
        "error_rows": error_rows,
        "development_cases": development_cases,
        "development_manifest": development_manifest,
        "development_report": development_report,
    }


def _second_level_summary_rows(
    classifications: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    total = len(classifications)
    case_refs: dict[str, list[str]] = defaultdict(list)
    for row in classifications:
        for category in row.get("all_error_categories", []):
            case_id = str(row.get("case_id", ""))
            if case_id not in case_refs[str(category)]:
                case_refs[str(category)].append(case_id)
    return [
        {
            "error_category": category,
            "case_count": len(case_refs[category]),
            "percentage_of_non_exact": _safe_divide(len(case_refs[category]), total),
            "case_ids": sorted(case_refs[category]),
        }
        for category in SECOND_LEVEL_CATEGORIES
    ]


def _representative_errors(
    classifications: Sequence[Mapping[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    ordered = sorted(
        classifications,
        key=lambda row: (
            -len(row.get("original_error_types", [])),
            -len(row.get("all_error_categories", [])),
            str(row.get("case_id", "")),
        ),
    )
    fields = (
        "case_id",
        "review_text",
        "original_error_types",
        "primary_error_category",
        "secondary_error_categories",
        "classification_confidence",
        "notes",
    )
    return [{field: row.get(field) for field in fields} for row in ordered[:limit]]


def build_generalization_error_report(inputs: Mapping[str, Any]) -> dict[str, Any]:
    final_report = inputs["final_report"]
    development_report = inputs["development_report"]
    case_results = _list_of_mappings(final_report.get("cases"))
    development_results = _list_of_mappings(development_report.get("cases"))
    non_exact = [case for case in case_results if not bool(case.get("exact_match"))]
    classifications = [classify_non_exact_case(case) for case in non_exact]
    directional, pairs = build_directional_confusions(case_results)
    final_aspect_metrics, final_unknown = calculate_aspect_metrics(case_results)
    development_aspect_metrics, development_unknown = calculate_aspect_metrics(
        development_results
    )
    if final_unknown or development_unknown:
        raise GeneralizationStudyError(
            "Evaluation report contains out-of-taxonomy predicted labels: "
            + ", ".join(sorted(set(final_unknown + development_unknown)))
        )
    stable, difficult = rank_aspect_stability(final_aspect_metrics)
    repeated_rows, repeated_summary = analyze_same_aspect_multi_mentions(case_results)
    sentiment_rows, sentiment_summary = classify_sentiment_errors(case_results)
    evidence_analysis = analyze_weak_evidence(case_results)
    difficulty_comparison = compare_dataset_difficulty(
        inputs["development_cases"], inputs["final_cases"]
    )
    strict_breakdown = strict_exact_failure_breakdown(
        case_results, classifications
    )
    diagnosis = build_development_final_gap_diagnosis(
        development_report,
        final_report,
        development_aspect_metrics,
        final_aspect_metrics,
        classifications,
        difficulty_comparison,
    )
    dev_v2_spec = build_generalization_dev_v2_spec(
        pairs,
        final_aspect_metrics,
        classifications=classifications,
        difficulty_comparison=difficulty_comparison,
        repeated_summary=repeated_summary,
        sentiment_summary=sentiment_summary,
    )
    second_level_summary = _second_level_summary_rows(classifications)
    final_summary = final_report.get("summary", {})
    return {
        "study": "Step 8.10 Generalization Error Study",
        "study_policy": EVALUATION_ONLY_NOTICE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": final_report.get("provider", ""),
        "model": final_report.get("model", ""),
        "final_blind_timestamp": final_report.get("timestamp", ""),
        "development_metrics": dict(development_report.get("summary", {})),
        "final_blind_metrics": dict(final_summary),
        "final_blind_error_counts": dict(final_report.get("error_counts", {})),
        "non_exact_case_count": len(non_exact),
        "unresolved_case_count": sum(
            row.get("primary_error_category") == "UNRESOLVED"
            for row in classifications
        ),
        "needs_manual_review_count": sum(
            bool(row.get("needs_manual_review")) for row in classifications
        ),
        "needs_manual_review_case_ids": [
            str(row.get("case_id", ""))
            for row in classifications
            if bool(row.get("needs_manual_review"))
        ],
        "second_level_error_distribution": second_level_summary,
        "case_analysis": classifications,
        "top_10_representative_errors": _representative_errors(classifications),
        "aspect_confusion_directions": directional,
        "taxonomy_confusion_pairs": pairs,
        "aspect_metrics": final_aspect_metrics,
        "development_aspect_metrics": development_aspect_metrics,
        "most_stable_5_aspects": stable,
        "most_difficult_5_aspects": difficult,
        "strict_exact_failure_analysis": strict_breakdown,
        "same_aspect_multi_mention_analysis": repeated_summary,
        "same_aspect_multi_mention_rows": repeated_rows,
        "sentiment_error_study": sentiment_summary,
        "sentiment_error_rows": sentiment_rows,
        "evidence_analysis": evidence_analysis,
        "dataset_difficulty_comparison": difficulty_comparison,
        "development_final_gap_diagnosis": diagnosis,
        "generalization_dev_v2_spec": dev_v2_spec,
        "input_cross_checks": {
            "final_gold_case_count": len(inputs["final_cases"]),
            "standalone_prediction_count": len(
                _list_of_mappings(inputs["predictions"].get("predictions"))
            ),
            "errors_csv_row_count": len(inputs["error_rows"]),
            "development_case_count": len(inputs["development_cases"]),
            "development_manifest_verified": True,
            "final_frozen_paths_bound_to_manifest": True,
            "model_inference_performed": False,
            "network_access_performed": False,
        },
        "no_model_optimization_performed": True,
    }


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    *,
    json_fields: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for raw in rows:
            row = dict(raw)
            for field in json_fields:
                row[field] = _json_cell(row.get(field, []))
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    temporary.replace(path)


def save_generalization_error_outputs(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path = STUDY_OUTPUT_DIR,
) -> dict[str, str]:
    root = Path(output_dir)
    paths = {
        "generalization_error_report": root / "generalization_error_report.json",
        "generalization_case_analysis": root / "generalization_case_analysis.csv",
        "generalization_error_summary": root / "generalization_error_summary.csv",
        "aspect_confusion_matrix": root / "aspect_confusion_matrix.csv",
        "aspect_metrics": root / "aspect_metrics.csv",
        "same_aspect_multi_mention_analysis": root
        / "same_aspect_multi_mention_analysis.csv",
        "sentiment_error_analysis": root / "sentiment_error_analysis.csv",
        "dataset_difficulty_comparison": root / "dataset_difficulty_comparison.json",
        "generalization_dev_v2_spec": root / "generalization_dev_v2_spec.json",
    }
    report_payload = dict(report)
    report_payload["outputs"] = {key: str(path) for key, path in paths.items()}
    _atomic_write_json(paths["generalization_error_report"], report_payload)

    _atomic_write_csv(
        paths["generalization_case_analysis"],
        report.get("case_analysis", []),
        (
            "case_id",
            "review_text",
            "gold_mentions",
            "predicted_mentions",
            "original_error_types",
            "primary_error_category",
            "secondary_error_categories",
            "affected_gold_aspects",
            "affected_predicted_aspects",
            "possible_gold_ambiguity",
            "classification_confidence",
            "needs_manual_review",
            "observed_linguistic_cues",
            "over_split_aspects",
            "over_merged_aspects",
            "notes",
        ),
        json_fields=(
            "gold_mentions",
            "predicted_mentions",
            "original_error_types",
            "secondary_error_categories",
            "affected_gold_aspects",
            "affected_predicted_aspects",
            "observed_linguistic_cues",
            "over_split_aspects",
            "over_merged_aspects",
        ),
    )
    _atomic_write_csv(
        paths["generalization_error_summary"],
        report.get("second_level_error_distribution", []),
        ("error_category", "case_count", "percentage_of_non_exact", "case_ids"),
        json_fields=("case_ids",),
    )
    _atomic_write_csv(
        paths["aspect_confusion_matrix"],
        report.get("aspect_confusion_directions", []),
        (
            "gold_aspect",
            "predicted_aspect",
            "count",
            "case_ids",
            "percentage_of_wrong_aspect",
        ),
        json_fields=("case_ids",),
    )
    _atomic_write_csv(
        paths["aspect_metrics"],
        report.get("aspect_metrics", []),
        (
            "aspect",
            "support",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "aspect_sentiment_tp",
            "aspect_sentiment_fp",
            "aspect_sentiment_fn",
            "aspect_sentiment_precision",
            "aspect_sentiment_recall",
            "aspect_sentiment_f1",
            "metric_scope",
        ),
    )
    _atomic_write_csv(
        paths["same_aspect_multi_mention_analysis"],
        report.get("same_aspect_multi_mention_rows", []),
        (
            "case_id",
            "repeated_aspects",
            "gold_repeated_mention_count",
            "predicted_repeated_mention_count",
            "whole_case_strict_exact",
            "repeated_component_fully_correct",
            "one_mention_missed",
            "all_repeated_mentions_missed",
            "over_merged",
            "over_split",
            "wrong_sentiment",
            "wrong_aspect",
            "duplicate_generated",
        ),
        json_fields=("repeated_aspects",),
    )
    _atomic_write_csv(
        paths["sentiment_error_analysis"],
        report.get("sentiment_error_rows", []),
        (
            "case_id",
            "aspect",
            "expected_sentiment",
            "predicted_sentiment",
            "expected_evidence",
            "predicted_evidence",
            "primary_category",
            "secondary_categories",
            "classification_confidence",
            "needs_manual_review",
            "alignment_disagreement",
            "diagnostic_note",
        ),
        json_fields=("secondary_categories",),
    )
    _atomic_write_json(
        paths["dataset_difficulty_comparison"],
        report.get("dataset_difficulty_comparison", {}),
    )
    _atomic_write_json(
        paths["generalization_dev_v2_spec"],
        report.get("generalization_dev_v2_spec", {}),
    )
    return {key: str(path) for key, path in paths.items()}


def run_generalization_error_study(
    *,
    project_root: str | Path = PROJECT_ROOT,
    freeze_manifest_path: str | Path = FINAL_FREEZE_MANIFEST_PATH,
    dataset_path: str | Path = FINAL_DATASET_PATH,
    pre_evaluation_manifest_path: str | Path = FINAL_MANIFEST_PATH,
    predictions_path: str | Path = FINAL_PREDICTIONS_PATH,
    final_report_path: str | Path = FINAL_REPORT_PATH,
    errors_path: str | Path = FINAL_ERRORS_PATH,
    development_dataset_path: str | Path = DEVELOPMENT_DATASET_PATH,
    development_manifest_path: str | Path = DEVELOPMENT_MANIFEST_PATH,
    development_report_path: str | Path = DEVELOPMENT_REPORT_PATH,
    output_dir: str | Path = STUDY_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the entire diagnostic using disk artifacts only."""

    root = Path(project_root)
    freeze_start = verify_final_blind_freeze(
        freeze_manifest_path, project_root=root
    )
    _assert_frozen_input_paths(
        freeze_start,
        project_root=root,
        dataset_path=Path(dataset_path),
        pre_evaluation_manifest_path=Path(pre_evaluation_manifest_path),
        predictions_path=Path(predictions_path),
        final_report_path=Path(final_report_path),
        errors_path=Path(errors_path),
    )
    demo_start = verify_demo_freeze(freeze_start, project_root=root)
    protected_development_paths = (
        Path(development_dataset_path),
        Path(development_manifest_path),
        Path(development_report_path),
    )
    development_hashes_start = {
        str(path.resolve()): file_sha256(path) for path in protected_development_paths
    }
    inputs = _load_offline_inputs(
        dataset_path=Path(dataset_path),
        pre_evaluation_manifest_path=Path(pre_evaluation_manifest_path),
        predictions_path=Path(predictions_path),
        final_report_path=Path(final_report_path),
        errors_path=Path(errors_path),
        development_dataset_path=Path(development_dataset_path),
        development_manifest_path=Path(development_manifest_path),
        development_report_path=Path(development_report_path),
        freeze_manifest=freeze_start,
    )
    report = build_generalization_error_report(inputs)
    paths = save_generalization_error_outputs(report, output_dir=output_dir)
    freeze_end = verify_final_blind_freeze(
        freeze_manifest_path, project_root=root
    )
    demo_end = verify_demo_freeze(freeze_end, project_root=root)
    development_hashes_end = {
        str(path.resolve()): file_sha256(path) for path in protected_development_paths
    }
    if development_hashes_start != development_hashes_end:
        raise GeneralizationStudyError("A protected Step 8.8 input changed during analysis.")
    report = {
        **report,
        "outputs": paths,
        "freeze_verification": {
            "verified_before_analysis": True,
            "verified_after_outputs": True,
            "hashes_unchanged": freeze_start == freeze_end,
            "demo_hashes_unchanged": demo_start == demo_end,
            "protected_demo_file_count": len(demo_end),
            "development_hashes_unchanged": True,
        },
    }
    # Refresh the main report with final verification and output paths.
    _atomic_write_json(Path(paths["generalization_error_report"]), report)
    verify_final_blind_freeze(freeze_manifest_path, project_root=root)
    return report


def format_generalization_error_study(report: Mapping[str, Any]) -> str:
    metrics = report.get("final_blind_metrics", {})
    strict = report.get("strict_exact_failure_analysis", {})
    categories = sorted(
        report.get("second_level_error_distribution", []),
        key=lambda row: (-int(row.get("case_count", 0)), str(row.get("error_category", ""))),
    )
    confusions = report.get("aspect_confusion_directions", [])
    difficult = report.get("most_difficult_5_aspects", [])
    repeated = report.get("same_aspect_multi_mention_analysis", {})
    ambiguity = int(strict.get("possible_gold_ambiguity_cases", 0))
    lines = [
        "Step 8.10 Generalization Error Study",
        "",
        f"Final Blind F1: {float(metrics.get('aspect_f1', 0.0)):.2%}",
        f"Aspect-only F1: {float(metrics.get('aspect_only_f1', 0.0)):.2%}",
        f"Strict Exact: {float(metrics.get('exact_case_accuracy', 0.0)):.2%}",
        "",
        f"Non-exact cases: {int(report.get('non_exact_case_count', 0))}",
        "",
        "Top Error Categories:",
    ]
    for row in categories[:8]:
        lines.append(f"  {row.get('error_category')}: {int(row.get('case_count', 0))}")
    lines.extend(["", "Top Aspect Confusions:"])
    for row in confusions[:5]:
        lines.append(
            f"  {row.get('gold_aspect')} -> {row.get('predicted_aspect')}: "
            f"{int(row.get('count', 0))}"
        )
    lines.extend(["", "Most Difficult Aspects:"])
    for row in difficult:
        lines.append(
            f"  {row.get('aspect')}: F1 {float(row.get('f1', 0.0)):.2%} "
            f"(support {int(row.get('support', 0))})"
        )
    lines.extend(
        [
            "",
            "Same-Aspect Multi-Mention Failures: "
            f"{int(repeated.get('cases_with_repeated_component_failure', 0))}/"
            f"{int(repeated.get('total_cases', 0))}",
            f"Possible Gold Ambiguities: {ambiguity}",
            "",
            "No model inference was performed.",
            "No network access was performed.",
            "No model optimization was performed in Step 8.10.",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "GeneralizationStudyError",
    "SECOND_LEVEL_CATEGORIES",
    "FINAL_METRIC_KEYS",
    "create_final_blind_freeze_manifest",
    "verify_final_blind_freeze",
    "verify_demo_freeze",
    "classify_non_exact_case",
    "build_directional_confusions",
    "calculate_aspect_metrics",
    "rank_aspect_stability",
    "strict_exact_failure_breakdown",
    "analyze_same_aspect_multi_mentions",
    "classify_sentiment_errors",
    "analyze_weak_evidence",
    "dataset_difficulty_statistics",
    "compare_dataset_difficulty",
    "build_generalization_dev_v2_spec",
    "build_generalization_error_report",
    "save_generalization_error_outputs",
    "run_generalization_error_study",
    "format_generalization_error_study",
]
