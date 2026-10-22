"""Evaluation runner for the Step 8.8 development expansion set.

The dataset is development data, not a final test set.  Metrics reuse the
existing evaluator implementation while this module adds dataset integrity,
bounded provider retry, stage-specific persistence, and baseline comparison.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import LLM_PROVIDER, OLLAMA_MODEL, PROJECT_ROOT
from evaluator import calculate_metrics, evaluate_case, load_gold_cases
from holdout_evaluator import (
    ERROR_CSV_FIELDS,
    dataset_distribution,
    file_sha256,
    select_top_errors,
)
from llm_client import (
    LLMClientError,
    LLMConnectionError,
    LLMResponseError,
    ReviewAnalysisClient,
    create_review_client,
)
from review_analyzer import ReviewAnalysisError, analyze_review


GENERALIZATION_CASES_PATH = (
    PROJECT_ROOT / "evaluation" / "generalization_dev_cases.json"
)
GENERALIZATION_MANIFEST_PATH = (
    PROJECT_ROOT / "evaluation" / "generalization_dev_manifest.json"
)
GENERALIZATION_BASELINE_PATH = (
    PROJECT_ROOT / "evaluation" / "baseline_generalization_dev_qwen3_8b.json"
)
GENERALIZATION_BASELINE_ERRORS_PATH = (
    PROJECT_ROOT / "evaluation" / "baseline_generalization_dev_errors.csv"
)
GENERALIZATION_RESULTS_PATH = (
    PROJECT_ROOT / "evaluation" / "evaluation_generalization_dev_qwen3_8b.json"
)
GENERALIZATION_ERRORS_PATH = (
    PROJECT_ROOT / "evaluation" / "evaluation_generalization_dev_errors.csv"
)

EXPECTED_GENERALIZATION_CASE_COUNT = 80
GENERALIZATION_ID_PREFIX = "G"
RETRY_DELAY_SECONDS = 1.0

COMPARISON_METRICS = (
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
)


class GeneralizationIntegrityError(ValueError):
    """The development expansion set or its freeze manifest is invalid."""


def _expected_ids(count: int) -> list[str]:
    return [f"{GENERALIZATION_ID_PREFIX}{index:03d}" for index in range(1, count + 1)]


def _load_raw_cases(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeneralizationIntegrityError(
            f"Generalization development dataset was not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GeneralizationIntegrityError(
            f"Generalization development dataset is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise GeneralizationIntegrityError(
            "Generalization development dataset must be a JSON list."
        )
    return payload


def validate_generalization_cases(
    path: str | Path = GENERALIZATION_CASES_PATH,
    *,
    expected_count: int = EXPECTED_GENERALIZATION_CASE_COUNT,
) -> list[dict[str, Any]]:
    """Validate structure, IDs, sentiments, metadata, and verbatim Evidence."""

    resolved = Path(path)
    raw_cases = _load_raw_cases(resolved)
    if len(raw_cases) != expected_count:
        raise GeneralizationIntegrityError(
            f"Generalization development dataset must contain exactly "
            f"{expected_count} cases; found {len(raw_cases)}."
        )
    try:
        validated = load_gold_cases(resolved)
    except (OSError, ValueError) as exc:
        raise GeneralizationIntegrityError(str(exc)) from exc
    if [case["case_id"] for case in validated] != _expected_ids(expected_count):
        raise GeneralizationIntegrityError(
            "Generalization case IDs must be consecutive and ordered from "
            f"G001 through G{expected_count:03d}."
        )
    normalized_reviews = [
        " ".join(case["review"].casefold().split()) for case in validated
    ]
    if len(normalized_reviews) != len(set(normalized_reviews)):
        raise GeneralizationIntegrityError(
            "Generalization development review texts must be unique."
        )

    enriched: list[dict[str, Any]] = []
    for raw, case in zip(raw_cases, validated, strict=True):
        if not isinstance(raw, Mapping):
            raise GeneralizationIntegrityError(
                f"{case['case_id']} must be a JSON object."
            )
        notes = raw.get("notes", "")
        tags = raw.get("tags", [])
        if not isinstance(notes, str) or not notes.strip():
            raise GeneralizationIntegrityError(
                f"{case['case_id']} must include annotation notes."
            )
        if not isinstance(tags, list) or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            raise GeneralizationIntegrityError(
                f"{case['case_id']}.tags must contain non-empty strings."
            )
        for mention in case["expected_aspects"]:
            if mention["sentiment"] not in {"positive", "negative", "neutral"}:
                raise GeneralizationIntegrityError(
                    f"{case['case_id']} uses an unsupported sentiment."
                )
            evidences = (
                mention["evidence"]
                if isinstance(mention["evidence"], list)
                else [mention["evidence"]]
            )
            for evidence in evidences:
                if evidence not in case["review"]:
                    raise GeneralizationIntegrityError(
                        f"{case['case_id']} Evidence is not a contiguous review "
                        f"substring: {evidence!r}"
                    )
        enriched.append(
            {
                **case,
                "notes": notes.strip(),
                "tags": [tag.strip() for tag in tags],
            }
        )
    return enriched


def create_generalization_manifest(
    dataset_path: str | Path = GENERALIZATION_CASES_PATH,
    manifest_path: str | Path = GENERALIZATION_MANIFEST_PATH,
    *,
    expected_count: int = EXPECTED_GENERALIZATION_CASE_COUNT,
) -> dict[str, Any]:
    """Create the one-time dataset manifest; existing manifests are immutable."""

    dataset = Path(dataset_path)
    validate_generalization_cases(dataset, expected_count=expected_count)
    destination = Path(manifest_path)
    if destination.exists():
        raise GeneralizationIntegrityError(
            "Generalization manifest already exists and will not be overwritten."
        )
    manifest = {
        "dataset": dataset.name,
        "case_count": expected_count,
        "sha256": file_sha256(dataset),
        "created_before_baseline": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_generalization_manifest(
    dataset_path: str | Path = GENERALIZATION_CASES_PATH,
    manifest_path: str | Path = GENERALIZATION_MANIFEST_PATH,
    *,
    expected_count: int = EXPECTED_GENERALIZATION_CASE_COUNT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify exact bytes before creating an LLM client or calling a model."""

    dataset = Path(dataset_path)
    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeneralizationIntegrityError(
            f"Generalization manifest was not found: {manifest_file}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GeneralizationIntegrityError(
            f"Generalization manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise GeneralizationIntegrityError(
            "Generalization manifest must be a JSON object."
        )
    if manifest.get("dataset") != dataset.name:
        raise GeneralizationIntegrityError("Manifest dataset name does not match.")
    if manifest.get("case_count") != expected_count:
        raise GeneralizationIntegrityError("Manifest case count does not match.")
    if manifest.get("created_before_baseline") is not True:
        raise GeneralizationIntegrityError(
            "Manifest must confirm creation before the baseline evaluation."
        )
    actual_hash = file_sha256(dataset)
    if manifest.get("sha256") != actual_hash:
        raise GeneralizationIntegrityError(
            "Generalization dataset SHA-256 does not match its frozen manifest."
        )
    cases = validate_generalization_cases(dataset, expected_count=expected_count)
    return dict(manifest), cases


def _strict_exact(case_result: Mapping[str, Any]) -> bool:
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
    count = len(expected)
    return all(
        int(counts.get(key, 0)) == count
        for key in (
            "evidence_total",
            "evidence_valid",
            "evidence_compared",
            "evidence_matched",
        )
    )


def run_generalization_evaluation(
    *,
    dataset_path: str | Path = GENERALIZATION_CASES_PATH,
    manifest_path: str | Path = GENERALIZATION_MANIFEST_PATH,
    expected_count: int = EXPECTED_GENERALIZATION_CASE_COUNT,
    provider: str = LLM_PROVIDER,
    model: str = OLLAMA_MODEL,
    client: ReviewAnalysisClient | None = None,
    client_factory: Callable[[], ReviewAnalysisClient] | None = None,
    analyzer: Callable[..., Mapping[str, Any]] = analyze_review,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    retry_delay_seconds: float = RETRY_DELAY_SECONDS,
    progress: Callable[[int, int, str, str, float], None] | None = None,
) -> dict[str, Any]:
    """Run all cases serially with one retry only for connection failures."""

    manifest, cases = verify_generalization_manifest(
        dataset_path,
        manifest_path,
        expected_count=expected_count,
    )
    if client is not None and client_factory is not None:
        raise ValueError("Provide client or client_factory, not both.")
    resolved_client = client or (
        client_factory() if client_factory is not None else create_review_client()
    )

    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    retry_count = 0
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
                    retry_count += 1
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
            except Exception as exc:
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
        result["exact_match"] = _strict_exact(result)
        result["notes"] = case["notes"]
        result["tags"] = case["tags"]
        result["attempts"] = attempts
        result["retried"] = attempts > 1
        status = "failed" if model_error or schema_error else "success"
        predictions.append(
            {
                "case_id": case["case_id"],
                "review": case["review"],
                "status": status,
                "attempts": attempts,
                "retried": attempts > 1,
                "analysis_time_seconds": elapsed,
                "analysis": dict(actual) if isinstance(actual, Mapping) else None,
                "error_type": (
                    "SCHEMA_ERROR" if schema_error else "MODEL_ERROR" if model_error else ""
                ),
                "error_message": schema_error or model_error or "",
            }
        )
        results.append(result)
        if progress is None:
            print(
                f"[{position}/{len(cases)}] {case['case_id']} {status} | "
                f"{elapsed:.2f}s"
            )
        else:
            progress(position, len(cases), case["case_id"], status, elapsed)

    summary = calculate_metrics(results, model=model)
    error_counts = Counter(
        str(error.get("error_type"))
        for result in results
        for error in result.get("errors", [])
        if isinstance(error, Mapping) and error.get("error_type")
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    summary["timestamp"] = timestamp
    return {
        "evaluation_type": "generalization_development",
        "provider": provider,
        "model": model,
        "dataset": Path(dataset_path).name,
        "dataset_sha256": manifest["sha256"],
        "timestamp": timestamp,
        "case_count": len(cases),
        "dataset_distribution": dataset_distribution(cases),
        "summary": summary,
        "runtime_statistics": {
            "total_seconds": summary["total_analysis_time"],
            "average_seconds_per_review": summary["average_analysis_time"],
            "median_seconds_per_review": summary["median_analysis_time"],
            "retry_count": retry_count,
            "model_failure_count": error_counts.get("MODEL_ERROR", 0),
            "schema_failure_count": error_counts.get("SCHEMA_ERROR", 0),
        },
        "confusion_matrix": summary["confusion_matrix"],
        "error_counts": dict(sorted(error_counts.items())),
        "top_errors": select_top_errors(results),
        "cases": results,
        "predictions": predictions,
    }


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


def _comparison(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_summary = baseline.get("summary", {})
    current_summary = current.get("summary", {})
    baseline_errors = baseline.get("error_counts", {})
    current_errors = current.get("error_counts", {})
    return {
        "metrics": {
            metric: {
                "baseline": float(baseline_summary.get(metric, 0.0)),
                "optimized": float(current_summary.get(metric, 0.0)),
                "difference": float(current_summary.get(metric, 0.0))
                - float(baseline_summary.get(metric, 0.0)),
            }
            for metric in COMPARISON_METRICS
        },
        "errors": {
            error_type: {
                "baseline": int(baseline_errors.get(error_type, 0)),
                "optimized": int(current_errors.get(error_type, 0)),
                "difference": int(current_errors.get(error_type, 0))
                - int(baseline_errors.get(error_type, 0)),
            }
            for error_type in (
                "EXTRA_ASPECT",
                "MISS_ASPECT",
                "WRONG_ASPECT",
                "WRONG_SENTIMENT",
                "WEAK_EVIDENCE_MATCH",
                "INVALID_EVIDENCE",
                "MODEL_ERROR",
                "SCHEMA_ERROR",
            )
        },
    }


def save_generalization_evaluation(
    report: Mapping[str, Any],
    *,
    stage: str,
    baseline_path: str | Path = GENERALIZATION_BASELINE_PATH,
    baseline_errors_path: str | Path = GENERALIZATION_BASELINE_ERRORS_PATH,
    results_path: str | Path = GENERALIZATION_RESULTS_PATH,
    errors_path: str | Path = GENERALIZATION_ERRORS_PATH,
) -> dict[str, str]:
    """Persist an immutable baseline or the post-optimization evaluation."""

    if stage not in {"baseline", "optimized"}:
        raise ValueError("stage must be 'baseline' or 'optimized'.")
    baseline_destination = Path(baseline_path)
    if stage == "baseline":
        report_destination = baseline_destination
        errors_destination = Path(baseline_errors_path)
        if report_destination.exists() or errors_destination.exists():
            raise GeneralizationIntegrityError(
                "Generalization baseline already exists and will not be overwritten."
            )
        payload = dict(report)
        payload["stage"] = "baseline"
    else:
        if not baseline_destination.is_file():
            raise GeneralizationIntegrityError(
                "Run and save the generalization baseline before optimization."
            )
        try:
            baseline = json.loads(baseline_destination.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GeneralizationIntegrityError(
                f"Generalization baseline is invalid JSON: {exc}"
            ) from exc
        if baseline.get("dataset_sha256") != report.get("dataset_sha256"):
            raise GeneralizationIntegrityError(
                "Optimized evaluation dataset does not match the baseline dataset."
            )
        report_destination = Path(results_path)
        errors_destination = Path(errors_path)
        payload = dict(report)
        payload["stage"] = "optimized"
        payload["baseline_comparison"] = _comparison(baseline, report)

    report_destination.parent.mkdir(parents=True, exist_ok=True)
    errors_destination.parent.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "results": str(report_destination),
        "errors": str(errors_destination),
    }
    payload["outputs"] = output_paths
    report_destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with errors_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_error_rows(payload))
    return output_paths


def format_generalization_report(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    runtime = report.get("runtime_statistics", {})
    error_counts = report.get("error_counts", {})
    return "\n".join(
        [
            "Generalization Development Evaluation",
            f"Cases: {int(summary.get('total_cases', 0))}",
            f"Successful: {int(summary.get('successful_cases', 0))}",
            f"Failed: {int(summary.get('failed_cases', 0))}",
            f"Aspect Precision: {float(summary.get('aspect_precision', 0.0)):.2%}",
            f"Aspect Recall: {float(summary.get('aspect_recall', 0.0)):.2%}",
            f"Aspect F1: {float(summary.get('aspect_f1', 0.0)):.2%}",
            f"Aspect-only Precision: {float(summary.get('aspect_only_precision', 0.0)):.2%}",
            f"Aspect-only Recall: {float(summary.get('aspect_only_recall', 0.0)):.2%}",
            f"Aspect-only F1: {float(summary.get('aspect_only_f1', 0.0)):.2%}",
            f"Sentiment Accuracy: {float(summary.get('sentiment_accuracy', 0.0)):.2%}",
            f"Evidence Validity: {float(summary.get('evidence_validity_rate', 0.0)):.2%}",
            f"Evidence Match: {float(summary.get('evidence_match_rate', 0.0)):.2%}",
            (
                f"Strict Exact: {int(summary.get('exact_matches', 0))}/"
                f"{int(summary.get('total_cases', 0))} "
                f"({float(summary.get('exact_case_accuracy', 0.0)):.2%})"
            ),
            (
                "TP / FP / FN: "
                f"{int(summary.get('aspect_true_positives', 0))} / "
                f"{int(summary.get('aspect_false_positives', 0))} / "
                f"{int(summary.get('aspect_false_negatives', 0))}"
            ),
            f"Error counts: {json.dumps(error_counts, sort_keys=True)}",
            f"Total runtime: {float(runtime.get('total_seconds', 0.0)):.2f}s",
            (
                "Average runtime: "
                f"{float(runtime.get('average_seconds_per_review', 0.0)):.2f}s/review"
            ),
            f"Median runtime: {float(runtime.get('median_seconds_per_review', 0.0)):.2f}s",
            f"Retries: {int(runtime.get('retry_count', 0))}",
        ]
    )
