"""Shared production runtime for long, local review-analysis jobs.

This module owns operational concerns only: progress, ETA, checkpoints, run
metadata, non-sensitive logs, and downstream publication.  It deliberately
does not contain prompts, taxonomy rules, or review-classification logic.
"""

from __future__ import annotations

import json
import os
import secrets
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from aggregator import aggregate_reviews, save_aggregation_outputs
from batch_analyzer import (
    BatchRunResult,
    ProgressCallback,
    RuntimeEventCallback,
    run_checkpointed_batch,
)
from config import (
    ANALYSIS_CANDIDATE_VERSION,
    KEYWORD_MIN_REVIEW_COUNT,
    MIN_ASPECT_REVIEWS_FOR_RANKING,
    OLLAMA_MODEL,
    PRODUCTION_CHECKPOINT_INTERVAL,
)
from keyword_analyzer import analyze_keywords, save_keyword_outputs
from llm_client import ReviewAnalysisClient


class AnalysisState(StrEnum):
    IDLE = "idle"
    VALIDATING = "validating"
    PREPARING = "preparing"
    ANALYZING = "analyzing"
    PAUSED = "paused"
    RETRYING = "retrying"
    AGGREGATING = "aggregating"
    GENERATING_KEYWORDS = "generating_keywords"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


TERMINAL_REVIEW_STATUSES = frozenset({"success", "failed", "skipped"})


class AnalysisRunBusyError(RuntimeError):
    """Another process or browser session owns this dataset workspace."""


def _process_is_alive(pid: object) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return True
    if value <= 0:
        return True
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


@contextmanager
def analysis_run_lock(
    root: str | Path,
    *,
    operation: str = "analysis",
) -> Iterator[Path]:
    """Own one Current workspace without exposing review content.

    The lock lives beside ``current`` so a fresh-session reset cannot remove an
    active owner's lock. A lock whose recorded process no longer exists is
    recovered once; an unreadable lock fails closed.
    """

    workspace = Path(root).resolve()
    workspace.parent.mkdir(parents=True, exist_ok=True)
    lock_path = workspace.parent / f".{workspace.name}.analysis.lock"
    token = secrets.token_hex(8)
    payload = {
        "pid": os.getpid(),
        "token": token,
        "operation": str(operation),
        "started_at": _utc_now(),
    }

    descriptor: int | None = None
    for attempt in range(2):
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.write(
                descriptor,
                (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"),
            )
            os.fsync(descriptor)
            break
        except FileExistsError as exc:
            if attempt:
                raise AnalysisRunBusyError(
                    "Another analysis is already using this dataset workspace."
                ) from exc
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise AnalysisRunBusyError(
                    "Another analysis may already be using this dataset workspace."
                ) from exc
            if not isinstance(existing, dict) or _process_is_alive(existing.get("pid")):
                raise AnalysisRunBusyError(
                    "Another analysis is already using this dataset workspace."
                ) from exc
            try:
                lock_path.unlink()
            except OSError as unlink_error:
                raise AnalysisRunBusyError(
                    "A stale analysis lock could not be recovered safely."
                ) from unlink_error

    if descriptor is None:
        raise AnalysisRunBusyError(
            "Another analysis is already using this dataset workspace."
        )
    try:
        yield lock_path
    finally:
        os.close(descriptor)
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("token") == token:
            lock_path.unlink(missing_ok=True)


def _with_workspace_lock(function: Callable[..., RuntimeExecutionResult]) -> Callable[..., RuntimeExecutionResult]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> RuntimeExecutionResult:
        lock_already_held = bool(kwargs.pop("lock_already_held", False))
        paths = kwargs.get("paths")
        if not isinstance(paths, RuntimePaths):
            raise TypeError("paths must be provided as RuntimePaths")
        if lock_already_held:
            return function(*args, **kwargs)
        with analysis_run_lock(paths.root):
            return function(*args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    checkpoint: Path
    checkpoint_meta: Path
    analyzed_reviews: Path
    aspect_summary: Path
    aspect_evidence: Path
    business_summary: Path
    keyword_summary: Path
    sentiment_keywords: Path
    aspect_keywords: Path
    run_metadata: Path
    event_log: Path

    @classmethod
    def from_session(cls, session: object) -> "RuntimePaths":
        root = Path(getattr(session, "root"))
        return cls(
            root=root,
            checkpoint=Path(getattr(session, "checkpoint")),
            checkpoint_meta=Path(getattr(session, "checkpoint_meta")),
            analyzed_reviews=Path(getattr(session, "analyzed_reviews")),
            aspect_summary=Path(getattr(session, "aspect_summary")),
            aspect_evidence=Path(getattr(session, "aspect_evidence")),
            business_summary=Path(getattr(session, "business_summary")),
            keyword_summary=Path(getattr(session, "keyword_summary")),
            sentiment_keywords=Path(getattr(session, "sentiment_keywords")),
            aspect_keywords=Path(getattr(session, "aspect_keywords")),
            run_metadata=Path(
                getattr(session, "run_metadata", root / "analysis_run_metadata.json")
            ),
            event_log=Path(
                getattr(session, "event_log", root / "analysis_run_events.jsonl")
            ),
        )


@dataclass(frozen=True)
class ProgressSnapshot:
    state: AnalysisState
    total_reviews: int
    completed_reviews: int
    successful: int
    failed: int
    skipped: int
    current_review_id: str | None
    elapsed_seconds: float
    eta_seconds: float | None
    retry_count: int
    operation_completed: int
    operation_total: int

    @property
    def percent_complete(self) -> float:
        denominator = self.operation_total
        if denominator <= 0:
            return 1.0 if self.completed_reviews >= self.total_reviews else 0.0
        return min(1.0, max(0.0, self.operation_completed / denominator))


ProgressSnapshotCallback = Callable[[ProgressSnapshot], None]


@dataclass(frozen=True)
class CheckpointSummary:
    total_reviews: int
    already_completed: int
    remaining: int
    successful: int
    failed: int
    skipped: int


@dataclass(frozen=True)
class RuntimeExecutionResult:
    batch: BatchRunResult
    successful_reviews: int
    failed_reviews: int
    skipped_reviews: int
    coverage: float
    elapsed_seconds: float
    state: AnalysisState
    run_metadata: dict[str, Any]
    downstream_completed: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def format_duration(seconds: float | None) -> str:
    """Return compact, stable UI text for an elapsed or ETA duration."""

    if seconds is None:
        return "Estimating remaining time..."
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _safe_callback(
    callback: ProgressSnapshotCallback | None,
    snapshot: ProgressSnapshot,
) -> None:
    if callback is None:
        return
    try:
        callback(snapshot)
    except Exception:
        return


class ProgressTracker:
    """Translate row callbacks into stable counts and rolling-median ETA."""

    def __init__(
        self,
        total_reviews: int,
        *,
        initial_statuses: Mapping[int, str] | None = None,
        operation_positions: set[int] | None = None,
        state: AnalysisState = AnalysisState.IDLE,
        concurrency: int = 1,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if total_reviews < 0:
            raise ValueError("total_reviews cannot be negative")
        if concurrency not in {1, 2}:
            raise ValueError("concurrency must be 1 or 2")
        self.total_reviews = total_reviews
        self._statuses = {
            int(position): str(status).casefold()
            for position, status in (initial_statuses or {}).items()
            if str(status).casefold() in TERMINAL_REVIEW_STATUSES
        }
        self._operation_positions = (
            set(range(total_reviews))
            if operation_positions is None
            else {int(position) for position in operation_positions}
        )
        self._operation_completed: set[int] = set()
        self._state = state
        self._clock = clock
        self._started_at = clock()
        self._request_started: dict[int, float] = {}
        self._recent_success_latencies: deque[float] = deque(maxlen=10)
        self._current_review_id: str | None = None
        self._retry_count = 0
        self._concurrency = concurrency

    def set_state(self, state: AnalysisState) -> ProgressSnapshot:
        self._state = AnalysisState(state)
        if state not in {AnalysisState.ANALYZING, AnalysisState.RETRYING}:
            self._current_review_id = None
        return self.snapshot()

    def add_retry(self, count: int = 1) -> ProgressSnapshot:
        self._retry_count += max(0, int(count))
        return self.snapshot()

    def handle(
        self,
        current: int,
        total: int,
        review_id: str,
        status: str,
    ) -> ProgressSnapshot:
        if total != self.total_reviews:
            self.total_reviews = max(0, int(total))
        position = max(0, int(current) - 1)
        normalized = str(status).casefold()
        now = self._clock()
        self._current_review_id = str(review_id)
        if normalized == "analyzing":
            self._request_started[position] = now
            if self._state != AnalysisState.RETRYING:
                self._state = AnalysisState.ANALYZING
        elif normalized in TERMINAL_REVIEW_STATUSES:
            self._statuses[position] = normalized
            if position in self._operation_positions:
                self._operation_completed.add(position)
            started = self._request_started.pop(position, None)
            if normalized == "success" and started is not None:
                self._recent_success_latencies.append(max(0.0, now - started))
        return self.snapshot(now=now)

    def snapshot(self, *, now: float | None = None) -> ProgressSnapshot:
        current_time = self._clock() if now is None else now
        counts = {
            status: sum(value == status for value in self._statuses.values())
            for status in TERMINAL_REVIEW_STATUSES
        }
        remaining = max(0, len(self._operation_positions) - len(self._operation_completed))
        eta: float | None = None
        if len(self._recent_success_latencies) >= 3 and remaining:
            eta = (
                statistics.median(self._recent_success_latencies)
                * remaining
                / self._concurrency
            )
        elif remaining == 0:
            eta = 0.0
        return ProgressSnapshot(
            state=self._state,
            total_reviews=self.total_reviews,
            completed_reviews=len(self._statuses),
            successful=counts["success"],
            failed=counts["failed"],
            skipped=counts["skipped"],
            current_review_id=self._current_review_id,
            elapsed_seconds=max(0.0, current_time - self._started_at),
            eta_seconds=eta,
            retry_count=self._retry_count,
            operation_completed=len(self._operation_completed),
            operation_total=len(self._operation_positions),
        )


def load_checkpoint_statuses(path: str | Path) -> dict[int, str]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        return {}
    try:
        frame = pd.read_csv(
            checkpoint,
            usecols=["source_row_id", "analysis_status"],
            keep_default_na=False,
        )
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return {}
    statuses: dict[int, str] = {}
    for row in frame.itertuples(index=False):
        try:
            position = int(row.source_row_id)
        except (TypeError, ValueError):
            continue
        status = str(row.analysis_status).casefold()
        if status in TERMINAL_REVIEW_STATUSES:
            statuses[position] = status
    return statuses


def inspect_checkpoint(
    checkpoint_path: str | Path,
    *,
    total_reviews: int | None = None,
) -> CheckpointSummary:
    statuses = load_checkpoint_statuses(checkpoint_path)
    if total_reviews is None:
        total_reviews = max(statuses, default=-1) + 1
    total = max(0, int(total_reviews))
    successful = sum(status == "success" for status in statuses.values())
    failed = sum(status == "failed" for status in statuses.values())
    skipped = sum(status == "skipped" for status in statuses.values())
    completed = successful + skipped
    # A normal resume leaves existing failures unchanged; only rows absent from
    # the checkpoint are pending. Failed-only retry is an explicit operation.
    remaining = max(0, total - len(statuses))
    return CheckpointSummary(
        total_reviews=total,
        already_completed=completed,
        remaining=remaining,
        successful=successful,
        failed=failed,
        skipped=skipped,
    )


def make_run_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(3)}"


def _atomic_write_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_run_metadata(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, dict) else None


class StructuredEventLog:
    """Append JSON events containing IDs and timing, never review text."""

    def __init__(self, path: Path, run_id: str, *, append: bool) -> None:
        self.path = path
        self.run_id = run_id
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            path.write_text("", encoding="utf-8")

    def __call__(self, event: dict[str, Any]) -> None:
        safe = {
            "timestamp": _utc_now(),
            "run_id": self.run_id,
            "source_row_id": event.get("source_row_id"),
            "review_id": event.get("review_id"),
            "status": event.get("status"),
            "latency_seconds": event.get("latency_seconds"),
            "retry_count": event.get("retry_count", 0),
            "error_type": event.get("error_type"),
        }
        line = json.dumps(safe, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)


class TimedReviewClient:
    """Measure provider calls without changing their inputs or outputs."""

    def __init__(
        self,
        client: ReviewAnalysisClient,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._client = client
        self._clock = clock
        self._lock = threading.Lock()
        self.main_request_seconds = 0.0
        self.evidence_repair_seconds = 0.0
        self.main_request_count = 0
        self.evidence_repair_count = 0

    @property
    def total_provider_seconds(self) -> float:
        return self.main_request_seconds + self.evidence_repair_seconds

    def analyze_review(
        self,
        review_text: str,
        allowed_aspects: Sequence[str],
    ) -> Any:
        started = self._clock()
        try:
            return self._client.analyze_review(review_text, allowed_aspects)
        finally:
            elapsed = max(0.0, self._clock() - started)
            with self._lock:
                self.main_request_seconds += elapsed
                self.main_request_count += 1

    def repair_evidence(
        self,
        review_text: str,
        invalid_aspects: Sequence[Mapping[str, Any]],
    ) -> Any:
        started = self._clock()
        try:
            return self._client.repair_evidence(review_text, invalid_aspects)
        finally:
            elapsed = max(0.0, self._clock() - started)
            with self._lock:
                self.evidence_repair_seconds += elapsed
                self.evidence_repair_count += 1


def _initial_operation_positions(
    total: int,
    statuses: Mapping[int, str],
    *,
    resume: bool,
    retry_failed: bool,
) -> set[int]:
    all_positions = set(range(total))
    if retry_failed and resume:
        return {
            position
            for position in all_positions
            if position not in statuses or statuses.get(position) == "failed"
        }
    if retry_failed:
        return {
            position for position, status in statuses.items() if status == "failed"
        }
    if resume:
        return all_positions.difference(statuses)
    return all_positions


def _status_counts(frame: pd.DataFrame) -> tuple[int, int, int]:
    statuses = frame.get("analysis_status", pd.Series(dtype="string")).astype(str)
    return (
        int(statuses.eq("success").sum()),
        int(statuses.eq("failed").sum()),
        int(statuses.eq("skipped").sum()),
    )


def _latency_statistics(frame: pd.DataFrame) -> dict[str, float]:
    if "analysis_status" not in frame or "analysis_time_seconds" not in frame:
        values: list[float] = []
    else:
        attempted = frame["analysis_status"].astype(str).isin({"success", "failed"})
        series = pd.to_numeric(
            frame.loc[attempted, "analysis_time_seconds"], errors="coerce"
        ).dropna()
        values = [max(0.0, float(value)) for value in series.tolist()]
    return {
        "mean_latency": statistics.fmean(values) if values else 0.0,
        "median_latency": statistics.median(values) if values else 0.0,
        "P90_latency": _percentile(values, 0.90),
    }


_TIMING_KEYS = (
    "validation_time",
    "preparing_time",
    "llm_time",
    "postprocessing_time",
    "checkpoint_time",
    "aggregation_time",
    "keyword_time",
)


def _normalized_timings(value: object) -> dict[str, float]:
    source = value if isinstance(value, Mapping) else {}
    normalized: dict[str, float] = {}
    for key in _TIMING_KEYS:
        try:
            normalized[key] = max(0.0, float(source.get(key, 0.0)))
        except (TypeError, ValueError):
            normalized[key] = 0.0
    return normalized


def _merge_timings(
    previous: Mapping[str, float],
    current: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: max(0.0, float(previous.get(key, 0.0)))
        + max(0.0, float(current.get(key, 0.0)))
        for key in _TIMING_KEYS
    }


def _emit_state(
    tracker: ProgressTracker,
    state: AnalysisState,
    callback: ProgressSnapshotCallback | None,
) -> ProgressSnapshot:
    snapshot = tracker.set_state(state)
    _safe_callback(callback, snapshot)
    return snapshot


@_with_workspace_lock
def run_analysis_runtime(
    cleaned_reviews: pd.DataFrame,
    *,
    source_type: str,
    llm_client: ReviewAnalysisClient,
    paths: RuntimePaths,
    resume: bool = False,
    retry_failed: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_snapshot_callback: ProgressSnapshotCallback | None = None,
    checkpoint_interval: int = PRODUCTION_CHECKPOINT_INTERVAL,
    min_aspect_reviews: int = MIN_ASPECT_REVIEWS_FOR_RANKING,
    min_keyword_reviews: int = KEYWORD_MIN_REVIEW_COUNT,
    concurrency: int = 1,
    validation_time_seconds: float = 0.0,
    preparing_time_seconds: float = 0.0,
    pre_runtime_elapsed_seconds: float = 0.0,
    clock: Callable[[], float] = time.perf_counter,
) -> RuntimeExecutionResult:
    """Run the common batch -> aggregation -> keywords production pipeline."""

    if concurrency != 1:
        raise ValueError(
            "Production runtime remains in Stable mode with concurrency=1."
        )
    if resume and retry_failed:
        raise ValueError("resume and retry_failed cannot both be true")
    paths.root.mkdir(parents=True, exist_ok=True)
    runtime_started = clock()
    previous = load_run_metadata(paths.run_metadata) if (resume or retry_failed) else None
    previous_duration = float(previous.get("duration_seconds", 0.0)) if previous else 0.0
    previous_timings = _normalized_timings(previous.get("timings") if previous else None)
    run_id = str(previous.get("run_id")) if previous and previous.get("run_id") else make_run_id()
    started_at = (
        str(previous.get("started_at"))
        if previous and previous.get("started_at")
        else _utc_now()
    )
    source = (
        str(previous.get("source_type"))
        if previous and previous.get("source_type")
        else str(source_type)
    )
    statuses = load_checkpoint_statuses(paths.checkpoint) if (resume or retry_failed) else {}
    operation_positions = _initial_operation_positions(
        len(cleaned_reviews), statuses, resume=resume, retry_failed=retry_failed
    )
    tracker = ProgressTracker(
        len(cleaned_reviews),
        initial_statuses=statuses,
        operation_positions=operation_positions,
        state=AnalysisState.RETRYING if retry_failed else AnalysisState.PREPARING,
        concurrency=concurrency,
        clock=clock,
    )
    _safe_callback(progress_snapshot_callback, tracker.snapshot())
    event_log = StructuredEventLog(
        paths.event_log,
        run_id,
        append=bool(previous),
    )
    total_retry_count = int(previous.get("retry_count", 0)) if previous else 0
    invocation_retry_count = 0

    def runtime_event(event: dict[str, Any]) -> None:
        nonlocal invocation_retry_count
        retries = int(event.get("retry_count", 0) or 0)
        invocation_retry_count += retries
        if retries:
            tracker.add_retry(retries)
        event_log(event)

    def row_progress(current: int, total: int, review_id: str, status: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(current, total, review_id, status)
            except Exception:
                pass
        snapshot = tracker.handle(current, total, review_id, status)
        _safe_callback(progress_snapshot_callback, snapshot)

    initial_metadata: dict[str, Any] = {
        "run_id": run_id,
        "source_type": source,
        "model": OLLAMA_MODEL,
        "candidate": ANALYSIS_CANDIDATE_VERSION,
        "total_reviews": len(cleaned_reviews),
        "successful": sum(value == "success" for value in statuses.values()),
        "failed": sum(value == "failed" for value in statuses.values()),
        "skipped": sum(value == "skipped" for value in statuses.values()),
        "started_at": started_at,
        "completed_at": None,
        "duration_seconds": previous_duration,
        "mean_latency": float(previous.get("mean_latency", 0.0)) if previous else 0.0,
        "median_latency": float(previous.get("median_latency", 0.0)) if previous else 0.0,
        "P90_latency": float(previous.get("P90_latency", 0.0)) if previous else 0.0,
        "retry_count": total_retry_count,
        "concurrency": concurrency,
        "resumed": bool(resume),
        "retry_failed": bool(retry_failed),
        "status": AnalysisState.RETRYING if retry_failed else AnalysisState.ANALYZING,
        "warmup_latency": None,
        "timings": _merge_timings(previous_timings, {
            "validation_time": max(0.0, validation_time_seconds),
            "preparing_time": max(0.0, preparing_time_seconds),
            "llm_time": 0.0,
            "postprocessing_time": 0.0,
            "checkpoint_time": 0.0,
            "aggregation_time": 0.0,
            "keyword_time": 0.0,
        }),
        "last_invocation_timings": {
            "validation_time": max(0.0, validation_time_seconds),
            "preparing_time": max(0.0, preparing_time_seconds),
            "llm_time": 0.0,
            "postprocessing_time": 0.0,
            "checkpoint_time": 0.0,
            "aggregation_time": 0.0,
            "keyword_time": 0.0,
        },
    }
    _atomic_write_json(initial_metadata, paths.run_metadata)
    _emit_state(
        tracker,
        AnalysisState.RETRYING if retry_failed else AnalysisState.ANALYZING,
        progress_snapshot_callback,
    )

    timed_client = TimedReviewClient(llm_client, clock=clock)

    def invocation_timings(
        batch_result: BatchRunResult | None = None,
        *,
        aggregation_seconds: float = 0.0,
        keyword_seconds: float = 0.0,
    ) -> dict[str, float]:
        measured_provider_time = timed_client.total_provider_seconds
        analyzer_postprocessing_time = 0.0
        checkpoint_seconds = 0.0
        if batch_result is not None:
            if measured_provider_time <= 0.0 and batch_result.llm_time_seconds > 0.0:
                # Batch-runner mocks and legacy adapters may provide aggregate
                # timing without actually invoking the wrapper.
                measured_provider_time = batch_result.llm_time_seconds
                analyzer_postprocessing_time = batch_result.postprocessing_time_seconds
            else:
                analyzer_postprocessing_time = (
                    max(0.0, batch_result.llm_time_seconds - measured_provider_time)
                    + batch_result.postprocessing_time_seconds
                )
            checkpoint_seconds = batch_result.checkpoint_time_seconds
        return {
            "validation_time": max(0.0, validation_time_seconds),
            "preparing_time": max(0.0, preparing_time_seconds),
            "llm_time": max(0.0, measured_provider_time),
            "postprocessing_time": max(0.0, analyzer_postprocessing_time),
            "checkpoint_time": max(0.0, checkpoint_seconds),
            "aggregation_time": max(0.0, aggregation_seconds),
            "keyword_time": max(0.0, keyword_seconds),
        }

    def persist_failure(
        exc: BaseException,
        state: AnalysisState,
        current_timings: Mapping[str, float],
        *,
        counts: tuple[int, int, int] | None = None,
    ) -> None:
        invocation_duration = max(0.0, clock() - runtime_started) + max(
            0.0, pre_runtime_elapsed_seconds
        )
        failure_metadata = {
            **initial_metadata,
            "completed_at": _utc_now(),
            "duration_seconds": previous_duration + invocation_duration,
            "last_invocation_duration_seconds": invocation_duration,
            "retry_count": total_retry_count + invocation_retry_count,
            "status": state,
            "error_type": type(exc).__name__,
            "timings": _merge_timings(previous_timings, current_timings),
            "last_invocation_timings": dict(current_timings),
        }
        if counts is not None:
            failure_metadata.update(
                {
                    "successful": counts[0],
                    "failed": counts[1],
                    "skipped": counts[2],
                }
            )
        _atomic_write_json(failure_metadata, paths.run_metadata)
        _emit_state(tracker, state, progress_snapshot_callback)

    try:
        batch = run_checkpointed_batch(
            cleaned_reviews,
            resume=resume,
            retry_failed=retry_failed,
            llm_client=timed_client,
            checkpoint_path=paths.checkpoint,
            meta_path=paths.checkpoint_meta,
            final_output_path=paths.analyzed_reviews,
            checkpoint_interval=checkpoint_interval,
            progress_callback=row_progress,
            runtime_event_callback=runtime_event,
            show_progress=False,
            clock=clock,
            checkpoint_context={
                "model": OLLAMA_MODEL,
                "candidate": ANALYSIS_CANDIDATE_VERSION,
            },
        )
    except BaseException as exc:
        failure_state = (
            AnalysisState.INTERRUPTED
            if isinstance(exc, KeyboardInterrupt)
            else AnalysisState.FAILED
        )
        persist_failure(exc, failure_state, invocation_timings())
        raise
    successful, failed, skipped = _status_counts(batch.reviews)
    aggregate_time = 0.0
    keyword_time = 0.0
    downstream_completed = False

    if batch.interrupted:
        final_state = AnalysisState.INTERRUPTED
    elif batch.stopped_for_failures or not batch.is_complete or successful <= 0:
        final_state = AnalysisState.FAILED
    else:
        _emit_state(tracker, AnalysisState.AGGREGATING, progress_snapshot_callback)
        phase_started = clock()
        try:
            aggregation = aggregate_reviews(
                batch.reviews,
                min_aspect_reviews=min_aspect_reviews,
            )
            save_aggregation_outputs(
                aggregation,
                aspect_summary_path=paths.aspect_summary,
                evidence_path=paths.aspect_evidence,
                business_summary_path=paths.business_summary,
            )
        except BaseException as exc:
            aggregate_time = max(0.0, clock() - phase_started)
            persist_failure(
                exc,
                AnalysisState.FAILED,
                invocation_timings(
                    batch,
                    aggregation_seconds=aggregate_time,
                ),
                counts=(successful, failed, skipped),
            )
            raise
        aggregate_time = max(0.0, clock() - phase_started)

        _emit_state(
            tracker,
            AnalysisState.GENERATING_KEYWORDS,
            progress_snapshot_callback,
        )
        phase_started = clock()
        try:
            keywords = analyze_keywords(
                batch.reviews,
                min_review_count=min_keyword_reviews,
            )
            save_keyword_outputs(
                keywords,
                keyword_summary_path=paths.keyword_summary,
                sentiment_keywords_path=paths.sentiment_keywords,
                aspect_keywords_path=paths.aspect_keywords,
            )
        except BaseException as exc:
            keyword_time = max(0.0, clock() - phase_started)
            persist_failure(
                exc,
                AnalysisState.FAILED,
                invocation_timings(
                    batch,
                    aggregation_seconds=aggregate_time,
                    keyword_seconds=keyword_time,
                ),
                counts=(successful, failed, skipped),
            )
            raise
        keyword_time = max(0.0, clock() - phase_started)
        downstream_completed = True
        final_state = (
            AnalysisState.COMPLETED_WITH_FAILURES
            if failed
            else AnalysisState.COMPLETED
        )

    invocation_duration = max(0.0, clock() - runtime_started) + max(
        0.0, pre_runtime_elapsed_seconds
    )
    latencies = _latency_statistics(batch.reviews)
    last_invocation_timings = invocation_timings(
        batch,
        aggregation_seconds=aggregate_time,
        keyword_seconds=keyword_time,
    )
    timings = _merge_timings(previous_timings, last_invocation_timings)
    metadata = {
        **initial_metadata,
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "completed_at": _utc_now(),
        "duration_seconds": previous_duration + invocation_duration,
        "last_invocation_duration_seconds": invocation_duration,
        **latencies,
        "retry_count": total_retry_count + invocation_retry_count,
        "model_failures": failed,
        "status": final_state,
        "timings": timings,
        "last_invocation_timings": last_invocation_timings,
    }
    _atomic_write_json(metadata, paths.run_metadata)
    _emit_state(tracker, final_state, progress_snapshot_callback)
    total = len(batch.reviews)
    return RuntimeExecutionResult(
        batch=batch,
        successful_reviews=successful,
        failed_reviews=failed,
        skipped_reviews=skipped,
        coverage=successful / total if total else 0.0,
        elapsed_seconds=invocation_duration,
        state=final_state,
        run_metadata=metadata,
        downstream_completed=downstream_completed,
    )


__all__ = [
    "AnalysisRunBusyError",
    "AnalysisState",
    "CheckpointSummary",
    "ProgressSnapshot",
    "ProgressSnapshotCallback",
    "ProgressTracker",
    "RuntimeExecutionResult",
    "RuntimePaths",
    "StructuredEventLog",
    "TimedReviewClient",
    "analysis_run_lock",
    "format_duration",
    "inspect_checkpoint",
    "load_checkpoint_statuses",
    "load_run_metadata",
    "make_run_id",
    "run_analysis_runtime",
]
