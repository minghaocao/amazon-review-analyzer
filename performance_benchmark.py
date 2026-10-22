"""Isolated throughput/stability benchmark for the frozen production analyzer."""

from __future__ import annotations

import json
import os
import resource
import statistics
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from analysis_runtime import TimedReviewClient
from aggregator import aggregate_reviews, save_aggregation_outputs
from batch_analyzer import build_unique_aspects
from cleaner import clean_reviews
from config import ANALYSIS_CANDIDATE_VERSION, OLLAMA_MODEL, OUTPUTS_DIR, PROJECT_ROOT
from input_loader import STANDARD_COLUMNS, validate_reviews
from keyword_analyzer import analyze_keywords, save_keyword_outputs
from llm_client import LLMConnectionError, ReviewAnalysisClient, create_review_client
from review_analyzer import analyze_review


PERFORMANCE_DATASET_PATH = PROJECT_ROOT / "performance_benchmark_reviews.json"
PERFORMANCE_OUTPUT_DIR = OUTPUTS_DIR / "performance"
PERFORMANCE_BASELINE_PATH = PERFORMANCE_OUTPUT_DIR / "performance_baseline.json"
PERFORMANCE_TIMINGS_PATH = PERFORMANCE_OUTPUT_DIR / "performance_request_timings.csv"
SUPPORTED_REVIEW_COUNTS = frozenset({10, 20})
SUPPORTED_WORKERS = (1, 2)
TRANSIENT_RETRY_DELAY_SECONDS = 1.0


class PerformanceBenchmarkError(RuntimeError):
    """The isolated benchmark cannot be completed safely."""


@dataclass(frozen=True)
class _RequestResult:
    source_row_id: int
    record: dict[str, Any]
    timing: dict[str, Any]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_performance_reviews(
    path: str | Path = PERFORMANCE_DATASET_PATH,
    *,
    review_count: int = 20,
) -> pd.DataFrame:
    if review_count not in SUPPORTED_REVIEW_COUNTS:
        raise ValueError("review_count must be 10 or 20")
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PerformanceBenchmarkError(
            f"Performance dataset was not found: {source}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PerformanceBenchmarkError(
            "Performance dataset could not be read safely."
        ) from exc
    if not isinstance(payload, list) or len(payload) != 20:
        raise PerformanceBenchmarkError(
            "Performance dataset must contain exactly 20 review objects."
        )
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise PerformanceBenchmarkError(
                f"Performance case {index} must be a JSON object."
            )
        missing = [column for column in STANDARD_COLUMNS if column not in item]
        if missing:
            raise PerformanceBenchmarkError(
                f"Performance case {index} is missing: {', '.join(missing)}"
            )
        rows.append({column: item.get(column) for column in STANDARD_COLUMNS})
    frame = pd.DataFrame(rows[:review_count], columns=list(STANDARD_COLUMNS))
    validation = validate_reviews(frame)
    if not validation["is_valid"]:
        raise PerformanceBenchmarkError(" ".join(validation["errors"]))
    return frame


def _evidence_is_valid(review_text: str, aspects: object) -> bool:
    if not isinstance(aspects, list):
        return False
    for aspect in aspects:
        if not isinstance(aspect, dict):
            return False
        evidence = aspect.get("evidence")
        if not isinstance(evidence, str) or evidence not in review_text:
            return False
    return True


def _base_record(row: pd.Series) -> dict[str, Any]:
    record = row.to_dict()
    record.setdefault("original_content", record.get("content", pd.NA))
    record.setdefault("cleaned_content", "")
    record.setdefault("skip_reason", "")
    return record


def _skipped_result(row: pd.Series) -> _RequestResult:
    record = _base_record(row)
    record.update(
        {
            "analysis_status": "skipped",
            "summary": "",
            "aspects": "[]",
            "unique_aspects": "[]",
            "positive_points": "[]",
            "negative_points": "[]",
            "error_message": "",
            "analysis_time_seconds": 0.0,
        }
    )
    source_row_id = int(record["source_row_id"])
    return _RequestResult(
        source_row_id=source_row_id,
        record=record,
        timing={
            "source_row_id": source_row_id,
            "review_id": str(record.get("review_id", "")),
            "status": "skipped",
            "latency_seconds": 0.0,
            "retry_count": 0,
            "error_type": "",
            "schema_valid": True,
            "evidence_valid": True,
        },
    )


def _analyze_one(
    row: pd.Series,
    *,
    client: ReviewAnalysisClient,
    sleeper: Callable[[float], None],
    retry_delay_seconds: float,
    clock: Callable[[], float],
) -> _RequestResult:
    record = _base_record(row)
    source_row_id = int(record["source_row_id"])
    review_id = str(record.get("review_id", ""))
    review_text = str(record.get("cleaned_content", ""))
    started = clock()
    provider_before = float(getattr(client, "total_provider_seconds", 0.0))
    retry_count = 0
    try:
        try:
            analysis = analyze_review(review_text, llm_client=client)
        except LLMConnectionError:
            retry_count = 1
            sleeper(retry_delay_seconds)
            analysis = analyze_review(review_text, llm_client=client)
        latency = max(0.0, clock() - started)
        aspects = analysis.get("aspects", [])
        schema_valid = (
            isinstance(analysis.get("summary"), str)
            and isinstance(aspects, list)
            and isinstance(analysis.get("positive_points"), list)
            and isinstance(analysis.get("negative_points"), list)
        )
        evidence_valid = _evidence_is_valid(review_text, aspects)
        record.update(
            {
                "analysis_status": "success",
                "summary": str(analysis.get("summary", "")),
                "aspects": json.dumps(aspects, ensure_ascii=False),
                "unique_aspects": json.dumps(
                    build_unique_aspects(aspects), ensure_ascii=False
                ),
                "positive_points": json.dumps(
                    analysis.get("positive_points", []), ensure_ascii=False
                ),
                "negative_points": json.dumps(
                    analysis.get("negative_points", []), ensure_ascii=False
                ),
                "error_message": "",
                "analysis_time_seconds": latency,
            }
        )
        status = "success"
        error_type = ""
        oom_error = False
    except Exception as exc:
        latency = max(0.0, clock() - started)
        schema_valid = False
        evidence_valid = False
        status = "failed"
        error_type = type(exc).__name__
        oom_error = isinstance(exc, MemoryError) or "out of memory" in str(exc).casefold()
        record.update(
            {
                "analysis_status": status,
                "summary": "",
                "aspects": "[]",
                "unique_aspects": "[]",
                "positive_points": "[]",
                "negative_points": "[]",
                "error_message": str(exc) or error_type,
                "analysis_time_seconds": latency,
            }
        )
    provider_after = float(getattr(client, "total_provider_seconds", 0.0))
    provider_seconds = max(0.0, provider_after - provider_before)
    return _RequestResult(
        source_row_id=source_row_id,
        record=record,
        timing={
            "source_row_id": source_row_id,
            "review_id": review_id,
            "status": status,
            "latency_seconds": latency,
            "llm_time_seconds": provider_seconds,
            "postprocessing_time_seconds": max(0.0, latency - provider_seconds),
            "retry_count": retry_count,
            "error_type": error_type,
            "oom_error": oom_error,
            "schema_valid": schema_valid,
            "evidence_valid": evidence_valid,
        },
    )


def _run_requests(
    cleaned: pd.DataFrame,
    *,
    workers: int,
    client_factory: Callable[[], ReviewAnalysisClient],
    checkpoint_path: Path,
    sleeper: Callable[[float], None],
    clock: Callable[[], float],
) -> tuple[pd.DataFrame, list[dict[str, Any]], float, float, float]:
    prepared = cleaned.copy(deep=True).reset_index(drop=True)
    prepared.insert(0, "source_row_id", range(len(prepared)))
    rows = [row for _, row in prepared.iterrows()]
    records: dict[int, dict[str, Any]] = {}
    timings: list[dict[str, Any]] = []
    checkpoint_time = 0.0
    postprocessing_time = 0.0
    thread_local = threading.local()

    def client_for_thread() -> ReviewAnalysisClient:
        if not hasattr(thread_local, "client"):
            thread_local.client = TimedReviewClient(
                client_factory(),
                clock=clock,
            )
        return thread_local.client

    def run_row(row: pd.Series) -> _RequestResult:
        try:
            analyzable = bool(row.get("is_analyzable"))
        except (TypeError, ValueError):
            analyzable = False
        if not analyzable:
            return _skipped_result(row)
        return _analyze_one(
            row,
            client=client_for_thread(),
            sleeper=sleeper,
            retry_delay_seconds=TRANSIENT_RETRY_DELAY_SECONDS,
            clock=clock,
        )

    def persist_checkpoint() -> None:
        nonlocal checkpoint_time
        checkpoint_started = clock()
        ordered = [records[position] for position in sorted(records)]
        _atomic_csv(pd.DataFrame.from_records(ordered), checkpoint_path)
        checkpoint_time += max(0.0, clock() - checkpoint_started)

    analysis_started = clock()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = iter(rows)
        pending: dict[Future[_RequestResult], int] = {}
        for _ in range(min(workers, len(rows))):
            try:
                row = next(iterator)
            except StopIteration:
                break
            future = executor.submit(run_row, row)
            pending[future] = int(row["source_row_id"])

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                process_started = clock()
                result = future.result()
                records[result.source_row_id] = result.record
                timings.append(result.timing)
                postprocessing_time += max(0.0, clock() - process_started)
                persist_checkpoint()
                try:
                    row = next(iterator)
                except StopIteration:
                    continue
                next_future = executor.submit(run_row, row)
                pending[next_future] = int(row["source_row_id"])
    analysis_wall_time = max(0.0, clock() - analysis_started)
    ordered_records = [records[position] for position in sorted(records)]
    frame = pd.DataFrame.from_records(ordered_records)
    timings.sort(key=lambda item: int(item["source_row_id"]))
    return frame, timings, analysis_wall_time, checkpoint_time, postprocessing_time


def _warm_up(
    client: ReviewAnalysisClient,
    *,
    clock: Callable[[], float],
) -> tuple[float | None, str | None]:
    """Load the model without producing a benchmark review or checkpoint row."""

    started = clock()
    try:
        analyze_review("The product works well.", llm_client=client)
    except Exception as exc:
        return max(0.0, clock() - started), type(exc).__name__
    return max(0.0, clock() - started), None


def run_performance_benchmark(
    *,
    review_count: int,
    workers: int,
    output_dir: str | Path,
    dataset_path: str | Path = PERFORMANCE_DATASET_PATH,
    client_factory: Callable[[], ReviewAnalysisClient] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if workers not in SUPPORTED_WORKERS:
        raise ValueError("workers must be 1 or 2")
    factory = client_factory or (lambda: create_review_client(provider="ollama"))
    run_directory = Path(output_dir) / f"workers_{workers}"
    run_directory.mkdir(parents=True, exist_ok=True)

    # Prepare the exact clients used by the worker threads before timing. This
    # keeps per-client health checks out of the measured request path and makes
    # the 1-worker and 2-worker comparison fair.
    prepared_clients: deque[ReviewAnalysisClient] = deque()
    for _ in range(workers):
        checked_client = factory()
        checker = getattr(checked_client, "check_connection", None)
        if not callable(checker):
            raise PerformanceBenchmarkError("Ollama health check is unavailable.")
        health = dict(checker())
        if str(health.get("model", OLLAMA_MODEL)) != OLLAMA_MODEL:
            raise PerformanceBenchmarkError(
                f"Performance benchmark requires {OLLAMA_MODEL}."
            )
        prepared_clients.append(checked_client)

    # One non-persisted request loads the shared Ollama model. A warm-up error
    # is recorded but does not make the batch unavailable, as required by the
    # production UX contract.
    warmup_latency, warmup_error = _warm_up(prepared_clients[0], clock=clock)
    client_lock = threading.Lock()

    def prepared_client_factory() -> ReviewAnalysisClient:
        with client_lock:
            if not prepared_clients:
                raise PerformanceBenchmarkError(
                    "No prepared benchmark client is available."
                )
            return prepared_clients.popleft()

    memory_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    total_started = clock()
    phase_started = clock()
    reviews = load_performance_reviews(dataset_path, review_count=review_count)
    validation_time = max(0.0, clock() - phase_started)

    phase_started = clock()
    cleaned = clean_reviews(reviews)
    cleaning_time = max(0.0, clock() - phase_started)
    (
        analyzed,
        timings,
        analysis_wall_time,
        checkpoint_time,
        row_postprocessing_time,
    ) = _run_requests(
        cleaned,
        workers=workers,
        client_factory=prepared_client_factory,
        checkpoint_path=run_directory / "analyzed_reviews_checkpoint.csv",
        sleeper=sleeper,
        clock=clock,
    )
    _atomic_csv(analyzed, run_directory / "analyzed_reviews.csv")

    phase_started = clock()
    aggregation = aggregate_reviews(analyzed)
    save_aggregation_outputs(
        aggregation,
        aspect_summary_path=run_directory / "aspect_summary.csv",
        evidence_path=run_directory / "aspect_evidence.json",
        business_summary_path=run_directory / "business_summary.json",
    )
    aggregation_time = max(0.0, clock() - phase_started)

    phase_started = clock()
    keywords = analyze_keywords(analyzed, min_review_count=1)
    save_keyword_outputs(
        keywords,
        keyword_summary_path=run_directory / "keyword_summary.csv",
        sentiment_keywords_path=run_directory / "sentiment_keywords.json",
        aspect_keywords_path=run_directory / "aspect_keywords.json",
    )
    keyword_time = max(0.0, clock() - phase_started)
    total_runtime = max(0.0, clock() - total_started)
    memory_after = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    statuses = analyzed["analysis_status"].astype(str)
    successful = int(statuses.eq("success").sum())
    failed = int(statuses.eq("failed").sum())
    skipped = int(statuses.eq("skipped").sum())
    latencies = [
        float(item["latency_seconds"])
        for item in timings
        if item["status"] in {"success", "failed"}
    ]
    retries = sum(int(item["retry_count"]) for item in timings)
    attempted = successful + failed
    schema_valid = successful > 0 and all(
        bool(item["schema_valid"])
        for item in timings
        if item["status"] == "success"
    )
    evidence_valid = successful > 0 and all(
        bool(item["evidence_valid"])
        for item in timings
        if item["status"] == "success"
    )
    first_case = next(
        (item for item in timings if int(item["source_row_id"]) == 0), None
    )
    report: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "candidate": ANALYSIS_CANDIDATE_VERSION,
        "workers": workers,
        "total_reviews": len(analyzed),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "total_runtime": total_runtime,
        "analysis_wall_time": analysis_wall_time,
        "mean_latency": statistics.fmean(latencies) if latencies else 0.0,
        "median_latency": statistics.median(latencies) if latencies else 0.0,
        "P90_latency": _percentile(latencies, 0.90),
        "min_latency": min(latencies, default=0.0),
        "max_latency": max(latencies, default=0.0),
        # Only the first (serial) run can represent the service cold start;
        # workers=2 follows it against the already-loaded model.
        "cold_start_latency": warmup_latency if workers == 1 else None,
        "first_case_latency": (
            float(first_case["latency_seconds"]) if first_case else 0.0
        ),
        "warmup_latency": warmup_latency,
        "warmup_error": warmup_error,
        "retries": retries,
        "model_failures": failed,
        "throughput_reviews_per_minute": (
            successful / analysis_wall_time * 60 if analysis_wall_time else 0.0
        ),
        "llm_time": sum(
            float(item.get("llm_time_seconds", 0.0)) for item in timings
        ),
        "validation_time": validation_time,
        "postprocessing_time": (
            cleaning_time
            + row_postprocessing_time
            + sum(
                float(item.get("postprocessing_time_seconds", 0.0))
                for item in timings
            )
        ),
        "cleaning_time": cleaning_time,
        "checkpoint_time": checkpoint_time,
        "aggregation_time": aggregation_time,
        "keyword_time": keyword_time,
        "schema_valid": schema_valid,
        "evidence_valid": evidence_valid,
        "memory_max_rss_before": memory_before,
        "memory_max_rss_after": memory_after,
        "oom_or_crash": any(bool(item.get("oom_error")) for item in timings),
        "performance_regression": workers == 1
        and statistics.median(latencies or [0.0]) > 10.0,
        "output_order_preserved": analyzed["source_row_id"].astype(int).tolist()
        == list(range(len(analyzed))),
    }
    for timing in timings:
        timing["workers"] = workers
        timing["model"] = OLLAMA_MODEL
    _atomic_json(report, run_directory / "performance_run.json")
    return report, timings


def select_runtime_strategy(
    serial: dict[str, Any],
    parallel: dict[str, Any],
) -> dict[str, Any]:
    serial_throughput = float(serial.get("throughput_reviews_per_minute", 0.0))
    parallel_throughput = float(parallel.get("throughput_reviews_per_minute", 0.0))
    improvement = (
        (parallel_throughput / serial_throughput) - 1.0
        if serial_throughput > 0
        else 0.0
    )
    serial_p90 = float(serial.get("P90_latency", 0.0))
    parallel_p90 = float(parallel.get("P90_latency", 0.0))
    p90_limit = max(15.0, serial_p90 * 1.75)
    serial_total = int(serial.get("total_reviews", 0))
    parallel_total = int(parallel.get("total_reviews", 0))
    serial_successful = int(serial.get("successful", serial_total))
    parallel_successful = int(parallel.get("successful", parallel_total))
    serial_skipped = int(serial.get("skipped", 0))
    parallel_skipped = int(parallel.get("skipped", 0))
    gates = {
        "throughput_improvement_at_least_20_percent": improvement >= 0.20,
        "failures_not_increased": int(parallel.get("failed", 0))
        <= int(serial.get("failed", 0)),
        "retries_not_materially_increased": int(parallel.get("retries", 0))
        <= int(serial.get("retries", 0)) + 1,
        "all_attempted_reviews_succeeded": (
            int(serial.get("failed", 0)) == 0
            and int(parallel.get("failed", 0)) == 0
            and serial_successful + serial_skipped == serial_total
            and parallel_successful + parallel_skipped == parallel_total
        ),
        "no_oom_or_crash": not bool(parallel.get("oom_or_crash", False)),
        "schema_valid": bool(parallel.get("schema_valid", False)),
        "evidence_valid": bool(parallel.get("evidence_valid", False)),
        "P90_controlled": parallel_p90 <= p90_limit,
        "output_order_preserved": bool(parallel.get("output_order_preserved", False)),
    }
    optional_faster_mode_allowed = all(gates.values())
    return {
        "throughput_improvement_ratio": improvement,
        "gates": gates,
        "optional_faster_mode_allowed": optional_faster_mode_allowed,
        # Product default remains conservative even when the optional gate passes.
        "production_default_workers": 1,
        "recommended_strategy": (
            "Stable mode (1 worker) with optional validated Faster mode"
            if optional_faster_mode_allowed
            else "Stable mode (1 worker)"
        ),
    }


def run_performance_comparison(
    *,
    review_count: int = 20,
    output_dir: str | Path = PERFORMANCE_OUTPUT_DIR,
    dataset_path: str | Path = PERFORMANCE_DATASET_PATH,
    client_factory: Callable[[], ReviewAnalysisClient] | None = None,
    allow_overwrite: bool = False,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    destination = Path(output_dir)
    is_default_baseline = review_count == 20
    baseline_path = destination / (
        "performance_baseline.json"
        if is_default_baseline
        else f"performance_baseline_{review_count}.json"
    )
    timings_path = destination / (
        "performance_request_timings.csv"
        if is_default_baseline
        else f"performance_request_timings_{review_count}.csv"
    )
    run_root = destination if is_default_baseline else destination / f"reviews_{review_count}"
    lock_path = destination / f".performance_benchmark_{review_count}.lock"
    protected_targets = (
        baseline_path,
        timings_path,
        run_root / "workers_1",
        run_root / "workers_2",
        lock_path,
    )
    if not allow_overwrite and any(path.exists() for path in protected_targets):
        raise PerformanceBenchmarkError(
            "Performance benchmark output already exists or an earlier run was "
            "interrupted; no result was overwritten or rerun."
        )
    destination.mkdir(parents=True, exist_ok=True)
    lock_descriptor: int | None = None
    if not allow_overwrite:
        try:
            lock_descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.write(lock_descriptor, b"benchmark in progress\n")
        except FileExistsError as exc:
            raise PerformanceBenchmarkError(
                "A performance benchmark is already running."
            ) from exc

    try:
        reports: list[dict[str, Any]] = []
        all_timings: list[dict[str, Any]] = []
        for workers in SUPPORTED_WORKERS:
            report, timings = run_performance_benchmark(
                review_count=review_count,
                workers=workers,
                output_dir=run_root,
                dataset_path=dataset_path,
                client_factory=client_factory,
                clock=clock,
                sleeper=sleeper,
            )
            reports.append(report)
            all_timings.extend(timings)
        selection = select_runtime_strategy(reports[0], reports[1])
        throughput = float(
            reports[selection["production_default_workers"] - 1].get(
                "throughput_reviews_per_minute", 0.0
            )
        )
        comparison = {
            "benchmark_purpose": "throughput_and_stability_only",
            "accuracy_evaluated": False,
            "dataset": Path(dataset_path).name,
            "review_count": review_count,
            "model": OLLAMA_MODEL,
            "candidate": ANALYSIS_CANDIDATE_VERSION,
            "runs": reports,
            "selection": selection,
            "estimated_100_review_runtime_minutes": (
                100.0 / throughput if throughput > 0 else None
            ),
            "benchmark_order_note": (
                "Both modes received a non-persisted warm-up. Workers=1 ran first, "
                "followed by workers=2 on the same reviews."
            ),
        }
        _atomic_json(comparison, baseline_path)
        _atomic_csv(pd.DataFrame(all_timings), timings_path)
        return comparison
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
            # A retained lock is intentional evidence of an interrupted run;
            # remove it only after both final comparison files exist.
            if baseline_path.is_file() and timings_path.is_file():
                lock_path.unlink(missing_ok=True)


def format_performance_report(report: dict[str, Any]) -> str:
    lines = [
        "Performance Benchmark",
        f"Model: {report.get('model', OLLAMA_MODEL)}",
        f"Reviews: {report.get('review_count', 0)}",
    ]
    for run in report.get("runs", []):
        lines.extend(
            [
                "",
                f"Workers: {run['workers']}",
                f"Successful / failed / skipped: {run['successful']} / {run['failed']} / {run['skipped']}",
                f"Total runtime: {run['total_runtime']:.2f}s",
                f"Mean / median / P90: {run['mean_latency']:.2f}s / {run['median_latency']:.2f}s / {run['P90_latency']:.2f}s",
                f"Throughput: {run['throughput_reviews_per_minute']:.2f} reviews/min",
                f"Retries: {run['retries']}",
                f"Schema valid: {run['schema_valid']}",
                f"Evidence valid: {run['evidence_valid']}",
            ]
        )
    lines.extend(
        [
            "",
            f"Runtime decision: {report['selection']['recommended_strategy']}",
            "Production default: 1 worker",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "PERFORMANCE_BASELINE_PATH",
    "PERFORMANCE_DATASET_PATH",
    "PERFORMANCE_OUTPUT_DIR",
    "PERFORMANCE_TIMINGS_PATH",
    "PerformanceBenchmarkError",
    "format_performance_report",
    "load_performance_reviews",
    "run_performance_benchmark",
    "run_performance_comparison",
    "select_runtime_strategy",
]
