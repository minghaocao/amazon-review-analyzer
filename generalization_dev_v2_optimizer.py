"""Step 8.12 optimization-round evaluator for Generalization Dev V2.

The Step 8.11 dataset, manifest, and untouched baseline are immutable inputs.
Production analysis files may change *between* optimization rounds, but are
snapshotted at the start of a round and must remain unchanged while that round
is running.  Importing this module performs no model or network work.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import LLM_PROVIDER, OLLAMA_MODEL, PROJECT_ROOT
from evaluator import calculate_metrics, evaluate_case
from final_blind_evaluator import strict_exact_case
from generalization_dev_v2_evaluator import (
    CORE_MODEL_FILES,
    ERROR_TYPES,
    EXPECTED_V2_CASE_COUNT,
    PRIORITY_ASPECTS,
    TRANSIENT_RETRY_DELAY_SECONDS,
    V2_BASELINE_PATH,
    V2_CASES_PATH,
    V2_ERRORS_PATH,
    V2_MANIFEST_PATH,
    V2_OUTPUT_DIR,
    V2_PREDICTIONS_PATH,
    GeneralizationDevV2IntegrityError,
    _error_rows,
    _percentile_nearest_rank,
    dataset_distribution,
    snapshot_file_hashes,
    validate_generalization_dev_v2_cases,
    verify_file_hashes,
)
from holdout_evaluator import ERROR_CSV_FIELDS, file_sha256, select_top_errors
from llm_client import (
    LLMClientError,
    LLMConnectionError,
    LLMResponseError,
    ReviewAnalysisClient,
    create_review_client,
)
from review_analyzer import ReviewAnalysisError, analyze_review


EXPECTED_V2_DATASET_SHA256 = (
    "ea6d264db2dea92fa6e0e2e469e2de572cac56654e916fcc3502bfaf23f650f2"
)
FROZEN_STEP_811_SHA256: Mapping[str, str] = {
    "dataset": EXPECTED_V2_DATASET_SHA256,
    "manifest": "33287c4dfe2b74ea36bfc2d874a1306a481dac874bac77dc5961f21f7edad724",
    "baseline": "80c0f2bce9f160fe5c4667856eb13db8bde6edcba5010fca3971a179e94609ba",
    "baseline_errors": "571a374973e3f9b1a2cc0736f3b10760bda2c3d5a70f7c310c83f3438b615820",
    "baseline_predictions": "d455abf9f67ab58e4f77c534bc755aacaff594f7cbf90e45f33a89df0da3d5b8",
}
FROZEN_STEP_811_PATHS: Mapping[str, Path] = {
    "dataset": V2_CASES_PATH,
    "manifest": V2_MANIFEST_PATH,
    "baseline": V2_BASELINE_PATH,
    "baseline_errors": V2_ERRORS_PATH,
    "baseline_predictions": V2_PREDICTIONS_PATH,
}
STABLE_ASPECTS = (
    "Sound Quality",
    "Delivery",
    "Connectivity",
    "System Performance",
    "Remote Control",
    "Setup / Installation",
)
MAX_OPTIMIZATION_ROUNDS = 3
REGRESSION_WARNING_THRESHOLD = 0.05
MIN_MEANINGFUL_F1_GAIN = 0.01
TARGET_ASPECT_SENTIMENT_F1 = 0.85
BEST_REPORT_PATH = V2_OUTPUT_DIR / "best_generalization_dev_v2_qwen3_8b.json"
BASELINE_DIAGNOSTIC_PATH = V2_OUTPUT_DIR / "optimization_baseline_diagnostic.json"


def optimization_round_paths(
    round_number: int,
    *,
    output_dir: str | Path = V2_OUTPUT_DIR,
) -> dict[str, Path]:
    """Return non-overlapping destinations for one bounded optimization round."""

    if not isinstance(round_number, int) or not 1 <= round_number <= MAX_OPTIMIZATION_ROUNDS:
        raise ValueError("Optimization round must be an integer from 1 through 3.")
    root = Path(output_dir)
    stem = f"optimization_round{round_number}"
    return {
        "report": root / f"{stem}_generalization_dev_v2.json",
        "errors": root / f"{stem}_errors.csv",
        "predictions": root / f"{stem}_predictions.json",
    }


def optimization_round_outputs_exist(
    round_number: int,
    *,
    output_dir: str | Path = V2_OUTPUT_DIR,
) -> bool:
    """Return true when any output for a round already exists."""

    return any(
        path.exists()
        for path in optimization_round_paths(
            round_number, output_dir=output_dir
        ).values()
    )


def load_optimization_round_reports(
    completed_rounds: int,
    *,
    output_dir: str | Path = V2_OUTPUT_DIR,
) -> list[dict[str, Any]]:
    """Load consecutive saved reports for a later round or final selection."""

    if not isinstance(completed_rounds, int) or not 0 <= completed_rounds <= 3:
        raise ValueError("completed_rounds must be an integer from 0 through 3.")
    reports: list[dict[str, Any]] = []
    for round_number in range(1, completed_rounds + 1):
        path = optimization_round_paths(
            round_number, output_dir=output_dir
        )["report"]
        report = _load_json_object(path, label=f"optimization round {round_number}")
        if report.get("round_number") != round_number:
            raise GeneralizationDevV2IntegrityError(
                f"Optimization round {round_number} report identity changed."
            )
        if report.get("provider") != "ollama" or report.get("model") != "qwen3:8b":
            raise GeneralizationDevV2IntegrityError(
                f"Optimization round {round_number} is not Ollama qwen3:8b."
            )
        if report.get("dataset_sha256") != EXPECTED_V2_DATASET_SHA256:
            raise GeneralizationDevV2IntegrityError(
                f"Optimization round {round_number} used a different dataset."
            )
        reports.append(report)
    _validate_round_sequence(reports)
    return reports


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeneralizationDevV2IntegrityError(
            f"Frozen {label} was not found: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GeneralizationDevV2IntegrityError(
            f"Frozen {label} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise GeneralizationDevV2IntegrityError(
            f"Frozen {label} must be a JSON object."
        )
    return dict(payload)


def verify_frozen_step_811_inputs(
    *,
    paths: Mapping[str, str | Path] = FROZEN_STEP_811_PATHS,
    expected_sha256: Mapping[str, str] = FROZEN_STEP_811_SHA256,
    expected_count: int = EXPECTED_V2_CASE_COUNT,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Fail closed if any Step 8.11 input changed.

    Deliberately does not compare current production-file hashes with the
    hashes stored in the Step 8.11 manifest: changing those production files
    is the purpose of Step 8.12.  It instead freezes the manifest and baseline
    files themselves byte-for-byte.
    """

    required = set(FROZEN_STEP_811_PATHS)
    if set(paths) != required or set(expected_sha256) != required:
        raise GeneralizationDevV2IntegrityError(
            "The Step 8.11 frozen-file set is incomplete or unexpected."
        )
    observed: dict[str, str] = {}
    for label in sorted(required):
        path = Path(paths[label])
        if not path.is_file():
            raise GeneralizationDevV2IntegrityError(
                f"Frozen Step 8.11 {label} was not found: {path}"
            )
        observed[label] = file_sha256(path)
        if observed[label] != expected_sha256[label]:
            raise GeneralizationDevV2IntegrityError(
                f"Frozen Step 8.11 {label} SHA-256 changed."
            )

    dataset_path = Path(paths["dataset"])
    cases = validate_generalization_dev_v2_cases(
        dataset_path, expected_count=expected_count
    )
    manifest = _load_json_object(paths["manifest"], label="V2 manifest")
    baseline = _load_json_object(paths["baseline"], label="V2 baseline")
    if manifest.get("sha256") != observed["dataset"]:
        raise GeneralizationDevV2IntegrityError(
            "Frozen V2 manifest no longer identifies the frozen dataset."
        )
    if manifest.get("case_count") != expected_count:
        raise GeneralizationDevV2IntegrityError(
            "Frozen V2 manifest case count changed."
        )
    if baseline.get("stage") != "baseline":
        raise GeneralizationDevV2IntegrityError(
            "Frozen V2 baseline has an invalid stage."
        )
    if baseline.get("provider") != "ollama" or baseline.get("model") != "qwen3:8b":
        raise GeneralizationDevV2IntegrityError(
            "Frozen V2 baseline is not the Ollama qwen3:8b baseline."
        )
    if baseline.get("dataset_sha256") != observed["dataset"]:
        raise GeneralizationDevV2IntegrityError(
            "Frozen V2 baseline no longer identifies the frozen dataset."
        )
    if baseline.get("case_count") != expected_count:
        raise GeneralizationDevV2IntegrityError(
            "Frozen V2 baseline case count changed."
        )
    baseline_cases = baseline.get("cases")
    if not isinstance(baseline_cases, list) or len(baseline_cases) != expected_count:
        raise GeneralizationDevV2IntegrityError(
            "Frozen V2 baseline case results are missing or incomplete."
        )
    return baseline, cases, observed


def build_baseline_optimization_diagnostic(
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize frozen baseline errors before any optimization inference."""

    directions: Counter[str] = Counter()
    extra_aspects: Counter[str] = Counter()
    missed_aspects: Counter[str] = Counter()
    wrong_sentiment_aspects: Counter[str] = Counter()
    no_aspect_extra_mentions = 0
    no_aspect_cases_with_extras = 0
    exact_duplicate_mentions = 0
    over_predicted_cases = 0
    under_predicted_cases = 0

    cases = baseline.get("cases", [])
    for case in cases if isinstance(cases, list) else []:
        if not isinstance(case, Mapping):
            continue
        expected = case.get("expected_aspects", [])
        actual = case.get("evaluation_aspects", case.get("actual_aspects", []))
        expected_items = expected if isinstance(expected, list) else []
        actual_items = actual if isinstance(actual, list) else []
        if len(actual_items) > len(expected_items):
            over_predicted_cases += 1
        elif len(actual_items) < len(expected_items):
            under_predicted_cases += 1
        if not expected_items and actual_items:
            no_aspect_cases_with_extras += 1
            no_aspect_extra_mentions += len(actual_items)

        seen: Counter[tuple[str, str, str]] = Counter()
        for item in actual_items:
            if isinstance(item, Mapping):
                key = (
                    str(item.get("aspect", "")),
                    str(item.get("sentiment", "")),
                    str(item.get("evidence", "")),
                )
                seen[key] += 1
        exact_duplicate_mentions += sum(max(0, count - 1) for count in seen.values())

        errors = case.get("errors", [])
        for error in errors if isinstance(errors, list) else []:
            if not isinstance(error, Mapping):
                continue
            error_type = str(error.get("error_type", ""))
            expected_aspect = str(error.get("expected_aspect", ""))
            actual_aspect = str(error.get("actual_aspect", ""))
            if error_type == "WRONG_ASPECT":
                directions[f"{expected_aspect} -> {actual_aspect}"] += 1
            elif error_type == "EXTRA_ASPECT":
                extra_aspects[actual_aspect] += 1
            elif error_type == "MISS_ASPECT":
                missed_aspects[expected_aspect] += 1
            elif error_type == "WRONG_SENTIMENT":
                wrong_sentiment_aspects[expected_aspect] += 1

    return {
        "source": "frozen Step 8.11 Generalization Dev V2 baseline",
        "case_count": int(baseline.get("case_count", 0)),
        "dataset_sha256": baseline.get("dataset_sha256", ""),
        "baseline_error_counts": baseline.get("error_counts", {}),
        "wrong_aspect_directions": dict(directions.most_common()),
        "extra_aspects_by_label": dict(extra_aspects.most_common()),
        "missed_aspects_by_label": dict(missed_aspects.most_common()),
        "wrong_sentiment_by_aspect": dict(wrong_sentiment_aspects.most_common()),
        "mention_granularity": {
            "cases_with_more_predictions_than_gold": over_predicted_cases,
            "cases_with_fewer_predictions_than_gold": under_predicted_cases,
            "exact_duplicate_prediction_mentions": exact_duplicate_mentions,
            "no_aspect_cases_with_extra_mentions": no_aspect_cases_with_extras,
            "extra_mentions_in_no_aspect_cases": no_aspect_extra_mentions,
        },
        "policy": (
            "Diagnostic only. It was derived from the frozen Step 8.11 baseline "
            "before Step 8.12 model evaluation."
        ),
    }


def save_baseline_optimization_diagnostic(
    diagnostic: Mapping[str, Any],
    *,
    destination: str | Path = BASELINE_DIAGNOSTIC_PATH,
) -> str:
    """Persist the pre-optimization diagnostic without overwriting any file."""

    path = Path(destination)
    if path.exists():
        raise GeneralizationDevV2IntegrityError(
            "Optimization baseline diagnostic already exists and will not be overwritten."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(diagnostic), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def calculate_aspect_metrics(
    case_results: Sequence[Mapping[str, Any]],
    aspects: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Calculate Aspect+sentiment multiset metrics for named taxonomy labels."""

    ordered_aspects = tuple(dict.fromkeys(str(aspect) for aspect in aspects))
    totals = {
        aspect: {"support": 0, "tp": 0, "fp": 0, "fn": 0}
        for aspect in ordered_aspects
    }
    for case in case_results:
        expected_raw = case.get("expected_aspects", [])
        actual_raw = case.get("evaluation_aspects", case.get("actual_aspects", []))
        expected = expected_raw if isinstance(expected_raw, list) else []
        actual = actual_raw if isinstance(actual_raw, list) else []
        expected_pairs = Counter(
            (str(item.get("aspect", "")), str(item.get("sentiment", "")))
            for item in expected
            if isinstance(item, Mapping)
        )
        actual_pairs = Counter(
            (str(item.get("aspect", "")), str(item.get("sentiment", "")))
            for item in actual
            if isinstance(item, Mapping)
        )
        for aspect in ordered_aspects:
            gold = sum(
                count for (label, _), count in expected_pairs.items() if label == aspect
            )
            predicted = sum(
                count for (label, _), count in actual_pairs.items() if label == aspect
            )
            tp = sum(
                min(count, actual_pairs[(label, sentiment)])
                for (label, sentiment), count in expected_pairs.items()
                if label == aspect
            )
            totals[aspect]["support"] += gold
            totals[aspect]["tp"] += tp
            totals[aspect]["fp"] += predicted - tp
            totals[aspect]["fn"] += gold - tp

    metrics: dict[str, dict[str, Any]] = {}
    for aspect, values in totals.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[aspect] = {
            **values,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def calculate_report_deltas(
    report: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe round changes relative to the untouched Step 8.11 baseline."""

    summary = report.get("summary", {})
    baseline_summary = baseline.get("summary", {})
    current_errors = report.get("error_counts", {})
    baseline_errors = baseline.get("error_counts", {})
    current_priority = report.get("priority_aspect_metrics", {})
    baseline_priority = baseline.get("priority_aspect_metrics", {})
    metric_names = (
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
    return {
        "summary": {
            name: float(summary.get(name, 0.0))
            - float(baseline_summary.get(name, 0.0))
            for name in metric_names
        },
        "counts": {
            name: int(summary.get(name, 0)) - int(baseline_summary.get(name, 0))
            for name in (
                "aspect_true_positives",
                "aspect_false_positives",
                "aspect_false_negatives",
                "exact_matches",
            )
        },
        "error_counts": {
            error_type: int(current_errors.get(error_type, 0))
            - int(baseline_errors.get(error_type, 0))
            for error_type in ERROR_TYPES
        },
        "priority_aspects": {
            aspect: {
                key: float(current_priority.get(aspect, {}).get(key, 0.0))
                - float(baseline_priority.get(aspect, {}).get(key, 0.0))
                for key in ("precision", "recall", "f1")
            }
            for aspect in PRIORITY_ASPECTS
        },
    }


def stable_aspect_regressions(
    current_metrics: Mapping[str, Mapping[str, Any]],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float = REGRESSION_WARNING_THRESHOLD,
) -> list[dict[str, Any]]:
    """Flag stable-Aspect F1 drops strictly greater than five points."""

    warnings: list[dict[str, Any]] = []
    for aspect in STABLE_ASPECTS:
        before = float(baseline_metrics.get(aspect, {}).get("f1", 0.0))
        after = float(current_metrics.get(aspect, {}).get("f1", 0.0))
        drop = before - after
        # Avoid turning an exact five-point drop into a warning solely because
        # of binary floating-point representation (for example 0.9 - 0.85).
        if drop > threshold + 1e-12:
            warnings.append(
                {
                    "warning": "REGRESSION_WARNING",
                    "aspect": aspect,
                    "baseline_f1": before,
                    "round_f1": after,
                    "drop": drop,
                    "threshold": threshold,
                }
            )
    return warnings


def _error_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    observed = Counter(
        str(error.get("error_type"))
        for result in results
        for error in result.get("errors", [])
        if isinstance(error, Mapping) and error.get("error_type")
    )
    return {error_type: int(observed.get(error_type, 0)) for error_type in ERROR_TYPES}


def run_generalization_dev_v2_optimization_round(
    round_number: int,
    *,
    frozen_paths: Mapping[str, str | Path] = FROZEN_STEP_811_PATHS,
    frozen_sha256: Mapping[str, str] = FROZEN_STEP_811_SHA256,
    expected_count: int = EXPECTED_V2_CASE_COUNT,
    provider: str = LLM_PROVIDER,
    model: str = OLLAMA_MODEL,
    core_model_paths: Sequence[str | Path] = CORE_MODEL_FILES,
    client: ReviewAnalysisClient | None = None,
    client_factory: Callable[[], ReviewAnalysisClient] | None = None,
    analyzer: Callable[..., Mapping[str, Any]] = analyze_review,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    retry_delay_seconds: float = TRANSIENT_RETRY_DELAY_SECONDS,
    progress: Callable[[int, int, str, str, float], None] | None = None,
    prior_round_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run one serial qwen3:8b round with only one transient retry per case."""

    if not 1 <= round_number <= MAX_OPTIMIZATION_ROUNDS:
        raise ValueError("Optimization round must be from 1 through 3.")
    if provider != "ollama" or model != "qwen3:8b":
        raise GeneralizationDevV2IntegrityError(
            "Step 8.12 optimization is locked to Ollama qwen3:8b."
        )
    if client is not None and client_factory is not None:
        raise ValueError("Provide client or client_factory, not both.")
    _validate_round_sequence(prior_round_reports)
    if len(prior_round_reports) != round_number - 1:
        raise ValueError(
            f"Round {round_number} requires exactly {round_number - 1} prior "
            "optimization report(s)."
        )

    baseline, cases, frozen_before = verify_frozen_step_811_inputs(
        paths=frozen_paths,
        expected_sha256=frozen_sha256,
        expected_count=expected_count,
    )
    # Important: construct the real provider only after immutable inputs pass.
    resolved_client = client or (
        client_factory() if client_factory is not None else create_review_client()
    )
    core_before = snapshot_file_hashes(core_model_paths)
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
            except Exception as exc:  # one failed case must not abort the round
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
        result["notes"] = case.get("notes", "")
        result["tags"] = case.get("tags", [])
        result["attempts"] = attempts
        result["retried"] = attempts > 1
        fatal_types = {
            str(error.get("error_type"))
            for error in result.get("errors", [])
            if isinstance(error, Mapping)
        }
        fatal_type = (
            "SCHEMA_ERROR"
            if "SCHEMA_ERROR" in fatal_types
            else "MODEL_ERROR"
            if "MODEL_ERROR" in fatal_types
            else ""
        )
        status = "failed" if fatal_type else "success"
        predictions.append(
            {
                "case_id": case["case_id"],
                "review": case["review"],
                "status": status,
                "attempts": attempts,
                "retried": attempts > 1,
                "analysis_time_seconds": elapsed,
                "analysis": dict(actual) if isinstance(actual, Mapping) else None,
                "error_type": fatal_type,
                "error_message": schema_error or model_error or "",
            }
        )
        results.append(result)
        if progress is None:
            print(
                f"[{position}/{len(cases)}] V2 round {round_number} "
                f"{case['case_id']} {status} | {elapsed:.2f}s"
            )
        else:
            progress(position, len(cases), case["case_id"], status, elapsed)

    core_after = verify_file_hashes(core_before, core_model_paths)
    _, _, frozen_after = verify_frozen_step_811_inputs(
        paths=frozen_paths,
        expected_sha256=frozen_sha256,
        expected_count=expected_count,
    )
    if frozen_before != frozen_after:
        raise GeneralizationDevV2IntegrityError(
            "A frozen Step 8.11 input changed during the optimization round."
        )

    summary = calculate_metrics(results, model=model)
    runtimes = [float(item["analysis_time_seconds"]) for item in predictions]
    priority_metrics = calculate_aspect_metrics(results, PRIORITY_ASPECTS)
    stable_metrics = calculate_aspect_metrics(results, STABLE_ASPECTS)
    baseline_stable = calculate_aspect_metrics(
        baseline.get("cases", []), STABLE_ASPECTS
    )
    warnings = stable_aspect_regressions(stable_metrics, baseline_stable)
    timestamp = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "evaluation_type": "generalization_dev_v2_optimization",
        "evaluation_policy": (
            "Step 8.12 optimized only against Generalization Dev V2. "
            "Final Blind was not rerun."
        ),
        "stage": f"optimization_round{round_number}",
        "round_number": round_number,
        "provider": provider,
        "model": model,
        "dataset": Path(frozen_paths["dataset"]).name,
        "dataset_sha256": frozen_before["dataset"],
        "timestamp": timestamp,
        "case_count": len(cases),
        "dataset_distribution": dataset_distribution(cases),
        "frozen_step_811_sha256": frozen_before,
        "core_model_file_sha256_before": core_before,
        "core_model_file_sha256_after": core_after,
        "core_model_hashes_unchanged_during_round": core_before == core_after,
        "summary": summary,
        "runtime_statistics": {
            "total_seconds": sum(runtimes),
            "average_seconds_per_review": sum(runtimes) / len(runtimes)
            if runtimes
            else 0.0,
            "median_seconds_per_review": statistics.median(runtimes)
            if runtimes
            else 0.0,
            "p90_seconds_per_review": _percentile_nearest_rank(runtimes, 0.90),
            "p90_method": "nearest_rank",
            "cold_start_first_case_seconds": runtimes[0] if runtimes else 0.0,
            "retry_count": retry_count,
            "successful_cases": int(summary.get("successful_cases", 0)),
            "failed_cases": int(summary.get("failed_cases", 0)),
        },
        "confusion_matrix": summary["confusion_matrix"],
        "error_counts": _error_counts(results),
        "priority_aspect_metrics": priority_metrics,
        "stable_aspect_metrics": stable_metrics,
        "baseline_stable_aspect_metrics": baseline_stable,
        "regression_warnings": warnings,
        "has_regression_warning": bool(warnings),
        "top_errors": select_top_errors(results, limit=10),
        "cases": results,
        "predictions": predictions,
    }
    report["baseline_deltas"] = calculate_report_deltas(report, baseline)
    report["stop_evaluation"] = evaluate_stop_conditions(
        baseline, [*prior_round_reports, report]
    )
    return report


def _validate_round_sequence(round_reports: Sequence[Mapping[str, Any]]) -> None:
    observed = [report.get("round_number") for report in round_reports]
    expected = list(range(1, len(round_reports) + 1))
    if observed != expected:
        raise ValueError(
            "Optimization reports must be ordered, consecutive rounds starting at 1."
        )


def evaluate_stop_conditions(
    baseline: Mapping[str, Any],
    round_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the four explicit Step 8.12 stop conditions."""

    _validate_round_sequence(round_reports)
    if not round_reports:
        return {"should_stop": False, "conditions": [], "latest_round": 0}
    latest = round_reports[-1]
    summary = latest.get("summary", {})
    reasons: list[dict[str, Any]] = []
    if (
        float(summary.get("aspect_f1", 0.0)) >= TARGET_ASPECT_SENTIMENT_F1
        and float(summary.get("evidence_validity_rate", 0.0)) == 1.0
        and not latest.get("regression_warnings")
    ):
        reasons.append(
            {
                "condition": "A",
                "message": "F1 target met with valid Evidence and no stable regression.",
            }
        )

    if len(round_reports) >= 2:
        prior_reports = [baseline, *round_reports[:-1]]
        last_two_gains = []
        for offset in (-2, -1):
            current = round_reports[offset]
            previous = prior_reports[offset]
            gain = float(current.get("summary", {}).get("aspect_f1", 0.0)) - float(
                previous.get("summary", {}).get("aspect_f1", 0.0)
            )
            last_two_gains.append(gain)
        if all(gain < MIN_MEANINGFUL_F1_GAIN for gain in last_two_gains):
            reasons.append(
                {
                    "condition": "B",
                    "message": "Two consecutive rounds improved F1 by less than one point.",
                    "last_two_f1_gains": last_two_gains,
                }
            )

    previous = baseline if len(round_reports) == 1 else round_reports[-2]
    overall_gain = float(summary.get("aspect_f1", 0.0)) - float(
        previous.get("summary", {}).get("aspect_f1", 0.0)
    )
    if overall_gain > 0.0 and latest.get("regression_warnings"):
        reasons.append(
            {
                "condition": "C",
                "message": "Overall F1 increased while a stable Aspect regressed.",
                "round_over_round_f1_gain": overall_gain,
            }
        )
    if len(round_reports) >= MAX_OPTIMIZATION_ROUNDS:
        reasons.append(
            {"condition": "D", "message": "Maximum of three rounds reached."}
        )
    return {
        "should_stop": bool(reasons),
        "conditions": reasons,
        "latest_round": int(latest.get("round_number", len(round_reports))),
    }


def _best_round_key(report: Mapping[str, Any]) -> tuple[float, ...]:
    summary = report.get("summary", {})
    fp = int(summary.get("aspect_false_positives", 0))
    fn = int(summary.get("aspect_false_negatives", 0))
    warning_count = len(report.get("regression_warnings", []))
    return (
        float(summary.get("aspect_f1", 0.0)),
        float(summary.get("aspect_only_f1", 0.0)),
        -float(abs(fp - fn)),
        -float(fp + fn),
        -float(warning_count),
        float(summary.get("exact_case_accuracy", 0.0)),
        -float(report.get("round_number", 0)),
    )


def select_best_round(
    round_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the best optimization round using the required ordering."""

    if not round_reports:
        raise ValueError("At least one optimization round is required.")
    _validate_round_sequence(round_reports)
    if len(round_reports) > MAX_OPTIMIZATION_ROUNDS:
        raise ValueError("No more than three optimization rounds are allowed.")
    return dict(max(round_reports, key=_best_round_key))


def save_optimization_round_outputs(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path = V2_OUTPUT_DIR,
) -> dict[str, str]:
    """Persist a round atomically by policy: existing outputs are never replaced."""

    round_number = report.get("round_number")
    if not isinstance(round_number, int):
        raise ValueError("Optimization report has no integer round_number.")
    destinations = optimization_round_paths(round_number, output_dir=output_dir)
    if any(path.exists() for path in destinations.values()):
        raise GeneralizationDevV2IntegrityError(
            f"Optimization round {round_number} output already exists and will not "
            "be overwritten."
        )
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    output_strings = {key: str(path) for key, path in destinations.items()}
    payload = dict(report)
    payload["outputs"] = output_strings
    destinations["report"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with destinations["errors"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_error_rows(payload))
    predictions_payload = {
        "evaluation_policy": payload.get("evaluation_policy", ""),
        "stage": payload.get("stage", ""),
        "round_number": round_number,
        "provider": payload.get("provider", ""),
        "model": payload.get("model", ""),
        "dataset_sha256": payload.get("dataset_sha256", ""),
        "timestamp": payload.get("timestamp", ""),
        "case_count": payload.get("case_count", 0),
        "predictions": payload.get("predictions", []),
    }
    destinations["predictions"].write_text(
        json.dumps(
            predictions_payload, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return output_strings


def save_best_round(
    round_reports: Sequence[Mapping[str, Any]],
    *,
    baseline: Mapping[str, Any],
    destination: str | Path = BEST_REPORT_PATH,
) -> dict[str, Any]:
    """Save one final best-round artifact without altering any round report."""

    path = Path(destination)
    if path.exists():
        raise GeneralizationDevV2IntegrityError(
            "Best Generalization Dev V2 output already exists and will not be overwritten."
        )
    best = select_best_round(round_reports)
    stop = evaluate_stop_conditions(baseline, round_reports)
    payload = {
        "evaluation_type": "generalization_dev_v2_best_optimization_round",
        "evaluation_policy": (
            "Step 8.12 optimized only against Generalization Dev V2. "
            "Final Blind was not rerun."
        ),
        "selected_round": best["round_number"],
        "selection_order": [
            "aspect_sentiment_f1",
            "aspect_only_f1",
            "fp_fn_balance",
            "stable_aspect_regression",
            "strict_exact_accuracy",
        ],
        "rounds_considered": [
            int(report["round_number"]) for report in round_reports
        ],
        "stop_evaluation": stop,
        "baseline_summary": baseline.get("summary", {}),
        "best_round": best,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def format_optimization_round_report(report: Mapping[str, Any]) -> str:
    """Produce a compact CLI summary without changing evaluation data."""

    summary = report.get("summary", {})
    runtime = report.get("runtime_statistics", {})
    warnings = report.get("regression_warnings", [])
    return "\n".join(
        [
            f"Generalization Dev V2 — optimization round {report.get('round_number', '?')}",
            f"Aspect P/R/F1: {float(summary.get('aspect_precision', 0.0)):.2%} / "
            f"{float(summary.get('aspect_recall', 0.0)):.2%} / "
            f"{float(summary.get('aspect_f1', 0.0)):.2%}",
            f"Aspect-only F1: {float(summary.get('aspect_only_f1', 0.0)):.2%}",
            f"Sentiment Accuracy: {float(summary.get('sentiment_accuracy', 0.0)):.2%}",
            f"Evidence Validity: {float(summary.get('evidence_validity_rate', 0.0)):.2%}",
            f"Strict Exact: {int(summary.get('exact_matches', 0))}/"
            f"{int(summary.get('total_cases', 0))} "
            f"({float(summary.get('exact_case_accuracy', 0.0)):.2%})",
            "TP / FP / FN: "
            f"{int(summary.get('aspect_true_positives', 0))} / "
            f"{int(summary.get('aspect_false_positives', 0))} / "
            f"{int(summary.get('aspect_false_negatives', 0))}",
            f"REGRESSION_WARNING count: {len(warnings)}",
            f"Runtime: {float(runtime.get('total_seconds', 0.0)):.2f}s",
        ]
    )


__all__ = [
    "BASELINE_DIAGNOSTIC_PATH",
    "BEST_REPORT_PATH",
    "EXPECTED_V2_DATASET_SHA256",
    "FROZEN_STEP_811_PATHS",
    "FROZEN_STEP_811_SHA256",
    "MAX_OPTIMIZATION_ROUNDS",
    "STABLE_ASPECTS",
    "build_baseline_optimization_diagnostic",
    "calculate_aspect_metrics",
    "calculate_report_deltas",
    "evaluate_stop_conditions",
    "format_optimization_round_report",
    "load_optimization_round_reports",
    "optimization_round_outputs_exist",
    "optimization_round_paths",
    "run_generalization_dev_v2_optimization_round",
    "save_baseline_optimization_diagnostic",
    "save_best_round",
    "save_optimization_round_outputs",
    "select_best_round",
    "stable_aspect_regressions",
    "verify_frozen_step_811_inputs",
]
