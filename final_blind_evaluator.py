"""One-time, evaluation-only runner for the Step 8.9 Final Blind Test.

The Final dataset and the production analysis fingerprint are SHA-locked before
an LLM client can be created.  This module reuses the established evaluator
metrics and never changes prompts, taxonomy, normalization, or post-processing.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    LLM_PROVIDER,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    PROJECT_ROOT,
)
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


FINAL_CASES_PATH = PROJECT_ROOT / "evaluation" / "final_blind_cases.json"
FINAL_MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "final_blind_manifest.json"
FINAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "final_blind"
FINAL_REPORT_PATH = FINAL_OUTPUT_DIR / "final_blind_evaluation.json"
FINAL_ERRORS_PATH = FINAL_OUTPUT_DIR / "final_blind_errors.csv"
FINAL_PREDICTIONS_PATH = FINAL_OUTPUT_DIR / "final_blind_predictions.json"

EXPECTED_FINAL_CASE_COUNT = 100
FINAL_ID_PREFIX = "F"
SIMILARITY_THRESHOLD = 0.75
TRANSIENT_RETRY_DELAY_SECONDS = 1.0
TAXONOMY_VERSION = "step8.8-frozen"
EVALUATION_ONLY_NOTICE = (
    "Final Blind Test is evaluation-only. No model changes were made based "
    "on these results."
)

FROZEN_ANALYSIS_FILES = (
    PROJECT_ROOT / "llm_client.py",
    PROJECT_ROOT / "aspect_taxonomy.py",
    PROJECT_ROOT / "review_analyzer.py",
    PROJECT_ROOT / "evaluator.py",
    PROJECT_ROOT / "config.py",
)
COMPARISON_DATASETS = (
    PROJECT_ROOT / "evaluation" / "gold_cases.json",
    PROJECT_ROOT / "evaluation" / "holdout_cases.json",
    PROJECT_ROOT / "evaluation" / "generalization_dev_cases.json",
)
DEMO_REVIEW_FILES = (
    PROJECT_ROOT / "data" / "raw_reviews.csv",
)
DEVELOPMENT_REPORTS = {
    "step8.6": PROJECT_ROOT / "evaluation" / "evaluation_step8_6_qwen3_8b.json",
    "step8.8": (
        PROJECT_ROOT / "evaluation" / "evaluation_generalization_dev_qwen3_8b.json"
    ),
}
GAP_METRICS = (
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
FINAL_ERROR_TYPES = (
    "EXTRA_ASPECT",
    "MISS_ASPECT",
    "WRONG_ASPECT",
    "WRONG_SENTIMENT",
    "WEAK_EVIDENCE_MATCH",
    "INVALID_EVIDENCE",
    "MODEL_ERROR",
    "SCHEMA_ERROR",
)


class FinalBlindIntegrityError(ValueError):
    """The Final dataset, manifest, fingerprint, or output freeze is invalid."""


def _expected_ids(count: int) -> list[str]:
    return [f"{FINAL_ID_PREFIX}{index:03d}" for index in range(1, count + 1)]


def _normalized_review(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _stable_file_label(path: Path) -> str:
    """Use a project-relative label when possible so manifests are portable."""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_raw_cases(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FinalBlindIntegrityError(
            f"Final Blind dataset was not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FinalBlindIntegrityError(
            f"Final Blind dataset is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise FinalBlindIntegrityError("Final Blind dataset must be a JSON list.")
    return payload


def validate_final_blind_cases(
    path: str | Path = FINAL_CASES_PATH,
    *,
    expected_count: int = EXPECTED_FINAL_CASE_COUNT,
) -> list[dict[str, Any]]:
    """Validate IDs, schema, sentiments, metadata, and verbatim Gold Evidence."""

    resolved = Path(path)
    raw_cases = _load_raw_cases(resolved)
    if len(raw_cases) != expected_count:
        raise FinalBlindIntegrityError(
            f"Final Blind dataset must contain exactly {expected_count} cases; "
            f"found {len(raw_cases)}."
        )
    try:
        validated = load_gold_cases(resolved)
    except (OSError, ValueError) as exc:
        raise FinalBlindIntegrityError(str(exc)) from exc
    if [case["case_id"] for case in validated] != _expected_ids(expected_count):
        raise FinalBlindIntegrityError(
            "Final Blind case IDs must be consecutive and ordered from "
            f"F001 through F{expected_count:03d}."
        )

    normalized = [_normalized_review(case["review"]) for case in validated]
    if len(normalized) != len(set(normalized)):
        raise FinalBlindIntegrityError(
            "Final Blind review texts must be unique after normalization."
        )

    enriched: list[dict[str, Any]] = []
    for raw, case in zip(raw_cases, validated, strict=True):
        if not isinstance(raw, Mapping):
            raise FinalBlindIntegrityError(
                f"{case['case_id']} must be a JSON object."
            )
        notes = raw.get("notes", "")
        tags = raw.get("tags", raw.get("coverage_tags", []))
        if not isinstance(notes, str) or not notes.strip():
            raise FinalBlindIntegrityError(
                f"{case['case_id']} must include annotation notes."
            )
        if not isinstance(tags, list) or not tags or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            raise FinalBlindIntegrityError(
                f"{case['case_id']}.tags must contain non-empty strings."
            )
        for mention in case["expected_aspects"]:
            if mention["sentiment"] not in {"positive", "negative", "neutral"}:
                raise FinalBlindIntegrityError(
                    f"{case['case_id']} uses an unsupported sentiment."
                )
            evidence = mention["evidence"]
            values = evidence if isinstance(evidence, list) else [evidence]
            for value in values:
                if value not in case["review"]:
                    raise FinalBlindIntegrityError(
                        f"{case['case_id']} Evidence is not a contiguous review "
                        f"substring: {value!r}"
                    )
        enriched.append(
            {
                **case,
                "notes": notes.strip(),
                "tags": [tag.strip() for tag in tags],
            }
        )
    return enriched


def _source_records(
    dataset_paths: Sequence[str | Path],
    demo_paths: Sequence[str | Path],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    records: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    for raw_path in dataset_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FinalBlindIntegrityError(
                f"Leakage comparison dataset was not found: {path}"
            )
        hashes[_stable_file_label(path)] = file_sha256(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FinalBlindIntegrityError(
                f"Leakage comparison dataset is invalid JSON: {path}: {exc}"
            ) from exc
        if not isinstance(payload, list):
            raise FinalBlindIntegrityError(
                f"Leakage comparison dataset must be a list: {path}"
            )
        for position, item in enumerate(payload):
            if not isinstance(item, Mapping):
                continue
            review = item.get("review", item.get("review_text", ""))
            if isinstance(review, str) and review.strip():
                identifier = item.get("case_id", position)
                records.append(
                    {
                        "source": path.name,
                        "source_id": str(identifier),
                        "review": review.strip(),
                    }
                )

    for raw_path in demo_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FinalBlindIntegrityError(
                f"Demo leakage comparison file was not found: {path}"
            )
        hashes[_stable_file_label(path)] = file_sha256(path)
        with path.open(encoding="utf-8", newline="") as handle:
            for position, row in enumerate(csv.DictReader(handle), start=1):
                review = row.get("content") or row.get("original_content") or ""
                if review.strip():
                    records.append(
                        {
                            "source": path.name,
                            "source_id": str(row.get("review_id") or position),
                            "review": review.strip(),
                        }
                    )
    return records, hashes


def _fingerprint_mapping(values: Mapping[str, Any]) -> str:
    serialized = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def check_final_blind_leakage(
    dataset_path: str | Path = FINAL_CASES_PATH,
    *,
    expected_count: int = EXPECTED_FINAL_CASE_COUNT,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
) -> dict[str, Any]:
    """Compare Final reviews with every previous evaluation and Demo review."""

    cases = validate_final_blind_cases(
        dataset_path,
        expected_count=expected_count,
    )
    sources, source_hashes = _source_records(
        comparison_dataset_paths,
        demo_paths,
    )
    exact_duplicates: list[dict[str, str]] = []
    normalized_duplicates: list[dict[str, str]] = []
    maximum_similarity = 0.0
    closest_source: dict[str, Any] | None = None

    for case in cases:
        final_review = case["review"].strip()
        final_normalized = _normalized_review(final_review)
        for source in sources:
            old_review = source["review"].strip()
            old_normalized = _normalized_review(old_review)
            identity = {
                "final_case_id": case["case_id"],
                "source": source["source"],
                "source_id": source["source_id"],
            }
            if final_review == old_review:
                exact_duplicates.append(identity)
            if final_normalized == old_normalized:
                normalized_duplicates.append(identity)
            similarity = difflib.SequenceMatcher(
                None,
                final_normalized,
                old_normalized,
            ).ratio()
            if similarity > maximum_similarity:
                maximum_similarity = similarity
                closest_source = {**identity, "similarity": similarity}

    source_fingerprint = _fingerprint_mapping(source_hashes)
    return {
        "exact_duplicate_count": len(exact_duplicates),
        "normalized_duplicate_count": len(normalized_duplicates),
        "exact_duplicates": exact_duplicates,
        "normalized_duplicates": normalized_duplicates,
        "maximum_similarity": maximum_similarity,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "closest_source": closest_source,
        "source_review_count": len(sources),
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "source_corpus_fingerprint": source_fingerprint,
        "passed": (
            not exact_duplicates
            and not normalized_duplicates
            and maximum_similarity < SIMILARITY_THRESHOLD
        ),
    }


def current_analysis_fingerprint(
    paths: Sequence[str | Path] = FROZEN_ANALYSIS_FILES,
) -> dict[str, Any]:
    """Fingerprint the exact analysis implementation and relevant runtime config."""

    file_hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FinalBlindIntegrityError(
                f"Frozen analysis file was not found: {path}"
            )
        file_hashes[_stable_file_label(path)] = file_sha256(path)
    runtime = {
        "provider": LLM_PROVIDER,
        "model": OLLAMA_MODEL,
        "ollama_num_ctx": OLLAMA_NUM_CTX,
        "ollama_keep_alive": OLLAMA_KEEP_ALIVE,
        "taxonomy_version": TAXONOMY_VERSION,
    }
    fingerprint = _fingerprint_mapping(
        {"files": dict(sorted(file_hashes.items())), "runtime": runtime}
    )
    return {
        "fingerprint": fingerprint,
        "files": dict(sorted(file_hashes.items())),
        "runtime": runtime,
    }


def create_final_blind_manifest(
    dataset_path: str | Path = FINAL_CASES_PATH,
    manifest_path: str | Path = FINAL_MANIFEST_PATH,
    *,
    expected_count: int = EXPECTED_FINAL_CASE_COUNT,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
    analysis_paths: Sequence[str | Path] = FROZEN_ANALYSIS_FILES,
) -> dict[str, Any]:
    """Create the one-time manifest before any real model invocation."""

    dataset = Path(dataset_path)
    validate_final_blind_cases(dataset, expected_count=expected_count)
    leakage = check_final_blind_leakage(
        dataset,
        expected_count=expected_count,
        comparison_dataset_paths=comparison_dataset_paths,
        demo_paths=demo_paths,
    )
    if not leakage["passed"]:
        raise FinalBlindIntegrityError(
            "Final Blind leakage check failed; regenerate overlapping cases before "
            "creating the manifest."
        )
    destination = Path(manifest_path)
    if destination.exists():
        raise FinalBlindIntegrityError(
            "Final Blind manifest already exists and will not be overwritten."
        )
    analysis = current_analysis_fingerprint(analysis_paths)
    manifest = {
        "dataset": dataset.name,
        "case_count": expected_count,
        "sha256": file_sha256(dataset),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_before_evaluation": True,
        "taxonomy_version": TAXONOMY_VERSION,
        "analysis_fingerprint": analysis["fingerprint"],
        "analysis_file_sha256": analysis["files"],
        "runtime_config": analysis["runtime"],
        "leakage_check": leakage,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_final_blind_manifest(
    dataset_path: str | Path = FINAL_CASES_PATH,
    manifest_path: str | Path = FINAL_MANIFEST_PATH,
    *,
    expected_count: int = EXPECTED_FINAL_CASE_COUNT,
    analysis_paths: Sequence[str | Path] = FROZEN_ANALYSIS_FILES,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify dataset and analysis hashes before an LLM client can be created."""

    dataset = Path(dataset_path)
    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FinalBlindIntegrityError(
            f"Final Blind manifest was not found: {manifest_file}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FinalBlindIntegrityError(
            f"Final Blind manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise FinalBlindIntegrityError("Final Blind manifest must be a JSON object.")
    if manifest.get("dataset") != dataset.name:
        raise FinalBlindIntegrityError("Final Blind manifest dataset name does not match.")
    if manifest.get("case_count") != expected_count:
        raise FinalBlindIntegrityError("Final Blind manifest case count does not match.")
    if manifest.get("created_before_evaluation") is not True:
        raise FinalBlindIntegrityError(
            "Final Blind manifest must confirm creation before evaluation."
        )
    if manifest.get("sha256") != file_sha256(dataset):
        raise FinalBlindIntegrityError(
            "Final Blind dataset SHA-256 does not match its frozen manifest."
        )
    if not isinstance(manifest.get("created_at"), str):
        raise FinalBlindIntegrityError("Final Blind manifest has no creation timestamp.")
    if manifest.get("taxonomy_version") != TAXONOMY_VERSION:
        raise FinalBlindIntegrityError("Final Blind taxonomy version does not match.")
    analysis = current_analysis_fingerprint(analysis_paths)
    if manifest.get("analysis_fingerprint") != analysis["fingerprint"]:
        raise FinalBlindIntegrityError(
            "Frozen analysis fingerprint changed after Final manifest creation."
        )
    leakage = manifest.get("leakage_check")
    if not isinstance(leakage, Mapping) or leakage.get("passed") is not True:
        raise FinalBlindIntegrityError(
            "Final Blind manifest does not contain a passing leakage check."
        )
    _, current_source_hashes = _source_records(
        comparison_dataset_paths,
        demo_paths,
    )
    current_source_fingerprint = _fingerprint_mapping(current_source_hashes)
    if leakage.get("source_corpus_fingerprint") != current_source_fingerprint:
        raise FinalBlindIntegrityError(
            "Leakage comparison corpus changed after Final manifest creation."
        )
    cases = validate_final_blind_cases(dataset, expected_count=expected_count)
    return dict(manifest), cases


def strict_exact_case(case_result: Mapping[str, Any]) -> bool:
    """Require order-independent Aspect, sentiment, count, and Evidence matches."""

    if not bool(case_result.get("exact_match")):
        return False
    expected = case_result.get("expected_aspects", [])
    actual = case_result.get("actual_aspects", [])
    counts = case_result.get("counts", {})
    if not isinstance(expected, list) or not isinstance(actual, list):
        return False
    if not expected and not actual:
        return True
    if not isinstance(counts, Mapping) or len(expected) != len(actual):
        return False
    expected_count = len(expected)
    return all(
        int(counts.get(key, 0)) == expected_count
        for key in (
            "evidence_total",
            "evidence_valid",
            "evidence_compared",
            "evidence_matched",
        )
    )


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _load_development_summaries(
    paths: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for stage, raw_path in paths.items():
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise FinalBlindIntegrityError(
                f"Development comparison report is unavailable: {path}: {exc}"
            ) from exc
        summary = payload.get("summary") if isinstance(payload, Mapping) else None
        if not isinstance(summary, Mapping):
            raise FinalBlindIntegrityError(
                f"Development comparison report has no summary: {path}"
            )
        summaries[stage] = dict(summary)
    return summaries


def _generalization_gaps(
    developments: Mapping[str, Mapping[str, Any]],
    final_summary: Mapping[str, Any],
) -> dict[str, Any]:
    gaps: dict[str, Any] = {}
    for stage, summary in developments.items():
        gaps[stage] = {
            metric: {
                "development": float(summary.get(metric, 0.0)),
                "final_blind": float(final_summary.get(metric, 0.0)),
                "difference": (
                    float(final_summary.get(metric, 0.0))
                    - float(summary.get(metric, 0.0))
                ),
            }
            for metric in GAP_METRICS
        }
    return gaps


def run_final_blind_evaluation(
    *,
    dataset_path: str | Path = FINAL_CASES_PATH,
    manifest_path: str | Path = FINAL_MANIFEST_PATH,
    expected_count: int = EXPECTED_FINAL_CASE_COUNT,
    provider: str = LLM_PROVIDER,
    model: str = OLLAMA_MODEL,
    analysis_paths: Sequence[str | Path] = FROZEN_ANALYSIS_FILES,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
    development_report_paths: Mapping[str, str | Path] = DEVELOPMENT_REPORTS,
    client: ReviewAnalysisClient | None = None,
    client_factory: Callable[[], ReviewAnalysisClient] | None = None,
    analyzer: Callable[..., Mapping[str, Any]] = analyze_review,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    retry_delay_seconds: float = TRANSIENT_RETRY_DELAY_SECONDS,
    progress: Callable[[int, int, str, str, float], None] | None = None,
) -> dict[str, Any]:
    """Run the frozen analyzer serially, with one connection retry at most."""

    manifest, cases = verify_final_blind_manifest(
        dataset_path,
        manifest_path,
        expected_count=expected_count,
        analysis_paths=analysis_paths,
        comparison_dataset_paths=comparison_dataset_paths,
        demo_paths=demo_paths,
    )
    if provider != "ollama" or model != "qwen3:8b":
        raise FinalBlindIntegrityError(
            "Final Blind evaluation is locked to local Ollama qwen3:8b."
        )
    developments = _load_development_summaries(development_report_paths)
    if client is not None and client_factory is not None:
        raise ValueError("Provide client or client_factory, not both.")
    resolved_client = client or (
        client_factory() if client_factory is not None else create_review_client()
    )

    case_results: list[dict[str, Any]] = []
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
        result["exact_match"] = strict_exact_case(result)
        result["notes"] = case["notes"]
        result["tags"] = case["tags"]
        result["attempts"] = attempts
        result["retried"] = attempts > 1
        error_type = "SCHEMA_ERROR" if schema_error else "MODEL_ERROR" if model_error else ""
        status = "failed" if error_type else "success"
        predictions.append(
            {
                "case_id": case["case_id"],
                "review": case["review"],
                "status": status,
                "attempts": attempts,
                "retried": attempts > 1,
                "analysis_time_seconds": elapsed,
                "analysis": dict(actual) if isinstance(actual, Mapping) else None,
                "error_type": error_type,
                "error_message": schema_error or model_error or "",
            }
        )
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
    observed_errors = Counter(
        str(error.get("error_type"))
        for result in case_results
        for error in result.get("errors", [])
        if isinstance(error, Mapping) and error.get("error_type")
    )
    error_counts = {
        error_type: int(observed_errors.get(error_type, 0))
        for error_type in FINAL_ERROR_TYPES
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    summary["timestamp"] = timestamp
    runtime_statistics = {
        "total_seconds": sum(runtimes),
        "mean_seconds_per_review": (
            sum(runtimes) / len(runtimes) if runtimes else 0.0
        ),
        "median_seconds_per_review": statistics.median(runtimes) if runtimes else 0.0,
        "p90_seconds_per_review": _percentile_nearest_rank(runtimes, 0.90),
        "p90_method": "nearest_rank",
        "retry_count": retry_count,
        "cold_start_first_case_seconds": runtimes[0] if runtimes else 0.0,
        "model_failure_count": error_counts["MODEL_ERROR"],
        "schema_failure_count": error_counts["SCHEMA_ERROR"],
    }
    return {
        "evaluation_type": "final_blind",
        "evaluation_policy": EVALUATION_ONLY_NOTICE,
        "provider": provider,
        "model": model,
        "dataset": Path(dataset_path).name,
        "dataset_sha256": manifest["sha256"],
        "manifest_created_at": manifest["created_at"],
        "analysis_fingerprint": manifest["analysis_fingerprint"],
        "taxonomy_version": manifest["taxonomy_version"],
        "timestamp": timestamp,
        "case_count": len(cases),
        "dataset_distribution": dataset_distribution(cases),
        "leakage_check": manifest["leakage_check"],
        "summary": summary,
        "runtime_statistics": runtime_statistics,
        "confusion_matrix": summary["confusion_matrix"],
        "error_counts": error_counts,
        "top_errors": select_top_errors(case_results, limit=10),
        "development_comparison": {
            "step8.6": {
                "label": "Step 8.6 development set",
                "summary": developments.get("step8.6", {}),
            },
            "step8.8": {
                "label": "Step 8.8 development expansion set",
                "summary": developments.get("step8.8", {}),
            },
            "policy": "Development and Final Blind metrics are not pooled.",
        },
        "final_generalization_gap": _generalization_gaps(developments, summary),
        "cases": case_results,
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
        if not isinstance(case, Mapping) or bool(case.get("exact_match")):
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


def final_outputs_exist(
    paths: Sequence[str | Path] = (
        FINAL_REPORT_PATH,
        FINAL_ERRORS_PATH,
        FINAL_PREDICTIONS_PATH,
    ),
) -> bool:
    return any(Path(path).exists() for path in paths)


def save_final_blind_outputs(
    report: Mapping[str, Any],
    *,
    report_path: str | Path = FINAL_REPORT_PATH,
    errors_path: str | Path = FINAL_ERRORS_PATH,
    predictions_path: str | Path = FINAL_PREDICTIONS_PATH,
) -> dict[str, str]:
    """Persist the one-time Final artifacts without overwriting any result."""

    destinations = (Path(report_path), Path(errors_path), Path(predictions_path))
    if any(path.exists() for path in destinations):
        raise FinalBlindIntegrityError(
            "Final Blind outputs already exist and will not be overwritten."
        )
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "report": str(destinations[0]),
        "errors": str(destinations[1]),
        "predictions": str(destinations[2]),
    }
    report_payload = dict(report)
    report_payload["outputs"] = output_paths
    destinations[0].write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with destinations[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_error_rows(report_payload))
    predictions_payload = {
        "evaluation_policy": EVALUATION_ONLY_NOTICE,
        "provider": report_payload.get("provider", ""),
        "model": report_payload.get("model", ""),
        "dataset_sha256": report_payload.get("dataset_sha256", ""),
        "analysis_fingerprint": report_payload.get("analysis_fingerprint", ""),
        "timestamp": report_payload.get("timestamp", ""),
        "case_count": report_payload.get("case_count", 0),
        "predictions": report_payload.get("predictions", []),
    }
    destinations[2].write_text(
        json.dumps(predictions_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output_paths


def format_final_blind_report(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    runtime = report.get("runtime_statistics", {})
    return "\n".join(
        [
            "Final Blind Test — evaluation only",
            EVALUATION_ONLY_NOTICE,
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
            f"Error counts: {json.dumps(report.get('error_counts', {}), sort_keys=True)}",
            f"Total runtime: {float(runtime.get('total_seconds', 0.0)):.2f}s",
            f"Mean runtime: {float(runtime.get('mean_seconds_per_review', 0.0)):.2f}s/review",
            f"Median runtime: {float(runtime.get('median_seconds_per_review', 0.0)):.2f}s",
            f"P90 runtime: {float(runtime.get('p90_seconds_per_review', 0.0)):.2f}s",
            f"Cold-start first case: {float(runtime.get('cold_start_first_case_seconds', 0.0)):.2f}s",
            f"Retries: {int(runtime.get('retry_count', 0))}",
        ]
    )
