"""Isolated persistence and analysis orchestration for Amazon imports.

Amazon scraping is deliberately outside this module.  A successful scraper
result is saved here, then the existing cleaner, checkpointed batch analyzer,
aggregator, keyword analyzer, and dashboard loader are reused unchanged.

Every mutating entry point validates that its destination is a narrowly scoped
``amazon/current`` directory.  In particular, the frozen Demo Dataset and the
Upload/Paste ``session/current`` directory can never be accepted as Amazon
write targets.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, fields
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

import analysis_runtime as runtime
from batch_analyzer import BatchRunResult, ProgressCallback
from cleaner import clean_reviews
from config import (
    AMAZON_CURRENT_DIR,
    ANALYZED_REVIEWS_PATH,
    DEMO_OUTPUT_PATHS,
    KEYWORD_MIN_REVIEW_COUNT,
    MIN_ASPECT_REVIEWS_FOR_RANKING,
    OUTPUTS_DIR,
    PRODUCTION_CHECKPOINT_INTERVAL,
    RAW_REVIEWS_PATH,
)
from input_loader import load_reviews, validate_reviews
from interactive_analysis import CURRENT_SESSION_DIR, DASHBOARD_OUTPUT_NAMES
from llm_client import ReviewAnalysisClient
from visualization import DashboardData, load_dashboard_data


# This is both executable documentation and a convenient test surface.  These
# paths are never passed to a writer in this module.
DEMO_PROTECTED_PATHS = DEMO_OUTPUT_PATHS

_SAFE_METADATA_FIELDS = (
    "asin",
    "marketplace",
    "source_url",
    "requested_limit",
    "reviews_collected",
    "scrape_status",
    "reason",
    "pages_fetched",
    "retrieved_review_count",
    "unique_review_count",
    "target_review_count",
    "availability_status",
    "availability_reason",
    "stop_reason",
)

_SAFE_BROWSER_METADATA_FIELDS = (
    "marketplace",
    "asin",
    "product_url",
    "reviews_url",
    "review_count",
    "retrieved_review_count",
    "unique_review_count",
    "target_review_count",
    "availability_status",
    "availability_reason",
    "stop_reason",
    "pages_visited",
    "challenge_encountered",
    "login_required",
    "status",
    "started_at",
    "completed_at",
)


class AmazonAnalysisError(RuntimeError):
    """Base error for Amazon import persistence and pipeline orchestration."""


class AmazonPathSafetyError(AmazonAnalysisError):
    """An Amazon operation was pointed at a protected or overly broad path."""


class AmazonImportError(AmazonAnalysisError):
    """Saved Amazon raw reviews or scrape metadata are missing or invalid."""


class AmazonPipelineError(AmazonAnalysisError):
    """The Amazon review analysis could not publish a complete dashboard."""


def _validated_amazon_root(root: str | Path) -> Path:
    """Return a canonical, narrowly scoped ``amazon/current`` directory."""

    resolved = Path(root).expanduser().resolve()
    if resolved.name != "current" or resolved.parent.name != "amazon":
        raise AmazonPathSafetyError(
            "Amazon data may only be written to an amazon/current directory."
        )

    demo_root = OUTPUTS_DIR.resolve()
    session_root = CURRENT_SESSION_DIR.resolve()
    if resolved == session_root or session_root in resolved.parents:
        raise AmazonPathSafetyError(
            "The Upload/Paste session directory is protected from Amazon writes."
        )

    # Within the real project outputs tree, only the official Amazon current
    # directory is accepted.  Temporary amazon/current roots remain available
    # for deterministic offline tests.
    if demo_root in resolved.parents and resolved != AMAZON_CURRENT_DIR.resolve():
        raise AmazonPathSafetyError(
            "The frozen Demo outputs are protected from Amazon writes."
        )
    return resolved


@dataclass(frozen=True)
class AmazonSessionPaths:
    """All raw, checkpoint, analysis, and metadata files for one Amazon import."""

    root: Path
    raw_reviews: Path
    scrape_metadata: Path
    browser_import_metadata: Path
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
    def from_root(cls, root: str | Path) -> "AmazonSessionPaths":
        directory = _validated_amazon_root(root)
        return cls(
            root=directory,
            raw_reviews=directory / "raw_reviews.csv",
            scrape_metadata=directory / "scrape_metadata.json",
            browser_import_metadata=directory / "browser_import_metadata.json",
            checkpoint=directory / "analyzed_reviews_checkpoint.csv",
            checkpoint_meta=directory / "analyzed_reviews_checkpoint.meta.json",
            analyzed_reviews=directory / "analyzed_reviews.csv",
            aspect_summary=directory / "aspect_summary.csv",
            aspect_evidence=directory / "aspect_evidence.json",
            business_summary=directory / "business_summary.json",
            keyword_summary=directory / "keyword_summary.csv",
            sentiment_keywords=directory / "sentiment_keywords.json",
            aspect_keywords=directory / "aspect_keywords.json",
            run_metadata=directory / "analysis_run_metadata.json",
            event_log=directory / "analysis_run_events.jsonl",
        )

    @classmethod
    def current(cls) -> "AmazonSessionPaths":
        return cls.from_root(AMAZON_CURRENT_DIR)


@dataclass(frozen=True)
class AmazonImport:
    """A saved Amazon import ready for preview or analysis."""

    reviews: pd.DataFrame
    metadata: dict[str, Any]
    paths: AmazonSessionPaths


@dataclass(frozen=True)
class AmazonPipelineResult:
    """Completion summary for the Amazon analysis UI."""

    paths: AmazonSessionPaths
    batch: BatchRunResult
    submitted_reviews: int
    successful_reviews: int
    failed_reviews: int
    skipped_reviews: int
    coverage: float
    elapsed_seconds: float
    run_metadata: dict[str, Any]
    state: runtime.AnalysisState


def _canonical_paths(paths: AmazonSessionPaths | None) -> AmazonSessionPaths:
    candidate = paths or AmazonSessionPaths.current()
    if not isinstance(candidate, AmazonSessionPaths):
        raise TypeError("paths must be an AmazonSessionPaths instance")
    canonical = AmazonSessionPaths.from_root(candidate.root)
    for field in fields(AmazonSessionPaths):
        value = getattr(candidate, field.name)
        expected = getattr(canonical, field.name)
        if Path(value).resolve() != Path(expected).resolve():
            raise AmazonPathSafetyError(
                "Amazon session paths must use the canonical filenames."
            )
    return canonical


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(value: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _json_scalar(value: Any, field_name: str) -> str | int | float | bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    raise AmazonImportError(f"Scrape metadata field {field_name} must be a scalar.")


def _safe_metadata(metadata: Mapping[str, Any], review_count: int) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    safe = {
        key: _json_scalar(metadata.get(key), key)
        for key in _SAFE_METADATA_FIELDS
        if key in metadata
    }
    safe["reviews_collected"] = review_count
    safe.setdefault("scrape_status", "success")
    return safe


def _safe_browser_metadata(
    metadata: Mapping[str, Any],
    review_count: int,
) -> dict[str, Any]:
    """Return only the non-sensitive browser import audit fields."""

    if not isinstance(metadata, Mapping):
        raise TypeError("browser_metadata must be a mapping")
    safe = {
        key: _json_scalar(metadata.get(key), key)
        for key in _SAFE_BROWSER_METADATA_FIELDS
        if key in metadata
    }
    raw_count = metadata.get("retrieved_review_count", review_count)
    try:
        retrieved_review_count = int(raw_count)
    except (TypeError, ValueError, OverflowError):
        retrieved_review_count = review_count
    safe["review_count"] = review_count
    safe["retrieved_review_count"] = max(review_count, retrieved_review_count)
    safe["unique_review_count"] = review_count
    # Quick Scan has no user-requested target.  Keep the compatibility field
    # explicit so older metadata readers do not infer one from the row count.
    safe["target_review_count"] = None
    safe.setdefault("availability_status", "unknown")
    safe.setdefault("status", "success")
    for required in ("marketplace", "asin", "product_url", "reviews_url"):
        if not str(safe.get(required) or "").strip():
            raise AmazonImportError(
                f"Browser import metadata is missing {required}."
            )
    return safe


def _reset_amazon_analysis_unlocked(root: str | Path) -> None:
    target = _validated_amazon_root(root)
    if target.exists():
        if not target.is_dir():
            raise AmazonPathSafetyError("The Amazon current target is not a directory.")
        shutil.rmtree(target)


def reset_amazon_analysis(root: str | Path = AMAZON_CURRENT_DIR) -> None:
    """Delete only one idle, validated ``amazon/current`` tree."""

    target = _validated_amazon_root(root)
    with runtime.analysis_run_lock(target, operation="reset"):
        _reset_amazon_analysis_unlocked(target)


def _with_amazon_lock(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        session = _canonical_paths(kwargs.get("paths"))
        with runtime.analysis_run_lock(session.root, operation=function.__name__):
            return function(*args, **kwargs)

    return wrapped


@_with_amazon_lock
def save_amazon_import(
    reviews: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    paths: AmazonSessionPaths | None = None,
    browser_metadata: Mapping[str, Any] | None = None,
) -> AmazonImport:
    """Replace only Amazon Current with one validated raw import and metadata."""

    session = _canonical_paths(paths)
    if not isinstance(reviews, pd.DataFrame):
        raise TypeError("reviews must be a pandas DataFrame")
    validation = validate_reviews(reviews)
    if not validation["is_valid"]:
        raise AmazonImportError(" ".join(validation["errors"]))
    if not len(reviews):
        raise AmazonImportError("No Amazon reviews were collected.")

    snapshot = reviews.copy(deep=True)
    safe_metadata = _safe_metadata(metadata, len(snapshot))
    safe_browser_metadata = (
        _safe_browser_metadata(browser_metadata, len(snapshot))
        if browser_metadata is not None
        else None
    )

    # A new import invalidates only the previous Amazon analysis.  Validation
    # occurs first, so bad scraper data cannot erase the last usable import.
    _reset_amazon_analysis_unlocked(session.root)
    session.root.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(snapshot, session.raw_reviews)
    _atomic_write_json(safe_metadata, session.scrape_metadata)
    if safe_browser_metadata is not None:
        _atomic_write_json(
            safe_browser_metadata,
            session.browser_import_metadata,
        )
    return AmazonImport(snapshot, safe_metadata, session)


def save_browser_amazon_import(
    reviews: pd.DataFrame,
    browser_metadata: Mapping[str, Any],
    *,
    paths: AmazonSessionPaths | None = None,
) -> AmazonImport:
    """Persist a user-confirmed browser preview in the Amazon-only workspace.

    ``scrape_metadata.json`` remains as a compatibility bridge for the existing
    preview and analysis pipeline.  The richer browser audit record is written
    separately and contains no browser storage, cookies, passwords, or tokens.
    """

    if not isinstance(browser_metadata, Mapping):
        raise TypeError("browser_metadata must be a mapping")
    review_count = len(reviews) if isinstance(reviews, pd.DataFrame) else 0
    safe_browser = _safe_browser_metadata(browser_metadata, review_count)
    legacy_metadata = {
        "asin": safe_browser.get("asin"),
        "marketplace": safe_browser.get("marketplace"),
        "source_url": safe_browser.get("product_url"),
        "requested_limit": None,
        "reviews_collected": review_count,
        "scrape_status": safe_browser.get("status", "success"),
        "reason": safe_browser.get("availability_reason"),
        "pages_fetched": safe_browser.get("pages_visited", 0),
        "retrieved_review_count": safe_browser.get(
            "retrieved_review_count", review_count
        ),
        "unique_review_count": review_count,
        "target_review_count": None,
        "availability_status": safe_browser.get(
            "availability_status", "unknown"
        ),
        "availability_reason": safe_browser.get("availability_reason"),
        "stop_reason": safe_browser.get("stop_reason"),
    }
    return save_amazon_import(
        reviews,
        legacy_metadata,
        paths=paths,
        browser_metadata=safe_browser,
    )


def amazon_import_available(
    root: str | Path = AMAZON_CURRENT_DIR,
) -> bool:
    session = AmazonSessionPaths.from_root(root)
    return session.raw_reviews.is_file() and session.scrape_metadata.is_file()


def load_amazon_import(
    paths: AmazonSessionPaths | None = None,
) -> AmazonImport:
    """Load an isolated raw Amazon import without touching other datasets."""

    session = _canonical_paths(paths)
    if not amazon_import_available(session.root):
        raise AmazonImportError("No saved Amazon review import is available.")
    reviews = load_reviews(session.raw_reviews)
    validation = validate_reviews(reviews)
    if not validation["is_valid"]:
        raise AmazonImportError(" ".join(validation["errors"]))
    try:
        metadata = json.loads(session.scrape_metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AmazonImportError("Amazon scrape metadata could not be read.") from exc
    if not isinstance(metadata, dict):
        raise AmazonImportError("Amazon scrape metadata must be a JSON object.")
    return AmazonImport(reviews, metadata, session)


def load_browser_import_metadata(
    paths: AmazonSessionPaths | None = None,
) -> dict[str, Any]:
    """Load the non-sensitive Quick Scan audit record when it exists."""

    session = _canonical_paths(paths)
    if not session.browser_import_metadata.is_file():
        return {}
    try:
        metadata = json.loads(
            session.browser_import_metadata.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AmazonImportError(
            "Amazon browser import metadata could not be read."
        ) from exc
    if not isinstance(metadata, dict):
        raise AmazonImportError(
            "Amazon browser import metadata must be a JSON object."
        )
    return metadata


def amazon_analysis_available(
    root: str | Path = AMAZON_CURRENT_DIR,
) -> bool:
    session = AmazonSessionPaths.from_root(root)
    return all((session.root / name).is_file() for name in DASHBOARD_OUTPUT_NAMES)


def failed_amazon_review_count(
    root: str | Path = AMAZON_CURRENT_DIR,
) -> int:
    session = AmazonSessionPaths.from_root(root)
    if not session.analyzed_reviews.is_file():
        return 0
    try:
        statuses = pd.read_csv(
            session.analyzed_reviews,
            usecols=["analysis_status"],
            dtype="string",
            keep_default_na=False,
        )["analysis_status"]
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return 0
    return int(statuses.str.casefold().eq("failed").sum())


def interrupted_amazon_analysis_available(
    root: str | Path = AMAZON_CURRENT_DIR,
) -> bool:
    session = AmazonSessionPaths.from_root(root)
    if amazon_analysis_available(session.root):
        return False
    return all(
        path.is_file()
        for path in (
            session.raw_reviews,
            session.scrape_metadata,
            session.checkpoint,
            session.checkpoint_meta,
        )
    )


def checkpoint_summary(
    paths: AmazonSessionPaths | None = None,
) -> runtime.CheckpointSummary | None:
    """Return safe resume counts for Amazon Current without modifying it."""

    session = _canonical_paths(paths)
    if not session.raw_reviews.is_file() or not session.checkpoint.is_file():
        return None
    try:
        total = len(load_reviews(session.raw_reviews))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return runtime.inspect_checkpoint(session.checkpoint, total_reviews=total)


def _cleaned_import(session: AmazonSessionPaths) -> pd.DataFrame:
    imported = load_amazon_import(session)
    cleaned = clean_reviews(imported.reviews)
    if not cleaned["is_analyzable"].eq(True).any():
        raise AmazonImportError("No Amazon reviews contain analyzable text.")
    return cleaned


def _clear_analysis_artifacts(session: AmazonSessionPaths) -> None:
    """Clear prior analysis while preserving raw reviews and scrape metadata."""

    for path in (
        session.checkpoint,
        session.checkpoint_meta,
        session.analyzed_reviews,
        session.aspect_summary,
        session.aspect_evidence,
        session.business_summary,
        session.keyword_summary,
        session.sentiment_keywords,
        session.aspect_keywords,
        session.run_metadata,
        session.event_log,
    ):
        path.unlink(missing_ok=True)


@_with_amazon_lock
def run_amazon_pipeline(
    *,
    llm_client: ReviewAnalysisClient,
    paths: AmazonSessionPaths | None = None,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_snapshot_callback: runtime.ProgressSnapshotCallback | None = None,
    checkpoint_interval: int = PRODUCTION_CHECKPOINT_INTERVAL,
    min_aspect_reviews: int = MIN_ASPECT_REVIEWS_FOR_RANKING,
    min_keyword_reviews: int = KEYWORD_MIN_REVIEW_COUNT,
    clock: Callable[[], float] = time.perf_counter,
) -> AmazonPipelineResult:
    """Run or resume the existing analysis pipeline for Amazon Current."""

    session = _canonical_paths(paths)
    if resume:
        if not interrupted_amazon_analysis_available(session.root):
            raise AmazonPipelineError("No interrupted Amazon analysis is available.")
    elif not amazon_import_available(session.root):
        raise AmazonImportError("Import Amazon reviews before starting analysis.")

    started = clock()
    phase_started = clock()
    cleaned = _cleaned_import(session)
    validation_time = max(0.0, clock() - phase_started)
    preparing_time = 0.0
    if not resume:
        phase_started = clock()
        _clear_analysis_artifacts(session)
        preparing_time = max(0.0, clock() - phase_started)
    execution = runtime.run_analysis_runtime(
        cleaned,
        source_type="amazon",
        llm_client=llm_client,
        paths=runtime.RuntimePaths.from_session(session),
        resume=resume,
        retry_failed=False,
        checkpoint_interval=checkpoint_interval,
        progress_callback=progress_callback,
        progress_snapshot_callback=progress_snapshot_callback,
        min_aspect_reviews=min_aspect_reviews,
        min_keyword_reviews=min_keyword_reviews,
        validation_time_seconds=validation_time,
        preparing_time_seconds=preparing_time,
        pre_runtime_elapsed_seconds=max(0.0, clock() - started),
        clock=clock,
        lock_already_held=True,
    )
    if not execution.batch.is_complete or execution.batch.final_output_path is None:
        raise AmazonPipelineError(
            "Amazon analysis stopped before completion. Its checkpoint was preserved."
        )
    if execution.successful_reviews <= 0:
        raise AmazonPipelineError(
            "No Amazon reviews were analyzed successfully. The checkpoint was preserved."
        )
    if not execution.downstream_completed or not amazon_analysis_available(session.root):
        raise AmazonPipelineError(
            "Amazon analysis finished but dashboard outputs are incomplete."
        )
    return AmazonPipelineResult(
        paths=session,
        batch=execution.batch,
        submitted_reviews=len(execution.batch.reviews),
        successful_reviews=execution.successful_reviews,
        failed_reviews=execution.failed_reviews,
        skipped_reviews=execution.skipped_reviews,
        coverage=execution.coverage,
        elapsed_seconds=execution.elapsed_seconds,
        run_metadata=execution.run_metadata,
        state=execution.state,
    )


@_with_amazon_lock
def retry_failed_amazon_reviews(
    *,
    llm_client: ReviewAnalysisClient,
    paths: AmazonSessionPaths | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_snapshot_callback: runtime.ProgressSnapshotCallback | None = None,
    checkpoint_interval: int = PRODUCTION_CHECKPOINT_INTERVAL,
    min_aspect_reviews: int = MIN_ASPECT_REVIEWS_FOR_RANKING,
    min_keyword_reviews: int = KEYWORD_MIN_REVIEW_COUNT,
    clock: Callable[[], float] = time.perf_counter,
) -> AmazonPipelineResult:
    """Retry only failed Amazon rows, then rebuild all dashboard artifacts."""

    session = _canonical_paths(paths)
    if not amazon_analysis_available(session.root):
        raise AmazonPipelineError("No complete Amazon analysis is available.")
    if failed_amazon_review_count(session.root) <= 0:
        raise AmazonPipelineError("Amazon analysis has no failed reviews to retry.")

    started = clock()
    cleaned = _cleaned_import(session)
    execution = runtime.run_analysis_runtime(
        cleaned,
        source_type="amazon",
        llm_client=llm_client,
        paths=runtime.RuntimePaths.from_session(session),
        resume=False,
        retry_failed=True,
        checkpoint_interval=checkpoint_interval,
        progress_callback=progress_callback,
        progress_snapshot_callback=progress_snapshot_callback,
        min_aspect_reviews=min_aspect_reviews,
        min_keyword_reviews=min_keyword_reviews,
        pre_runtime_elapsed_seconds=max(0.0, clock() - started),
        clock=clock,
        lock_already_held=True,
    )
    if not execution.downstream_completed:
        raise AmazonPipelineError(
            "Failed Amazon reviews could not be republished. The checkpoint was preserved."
        )
    return AmazonPipelineResult(
        paths=session,
        batch=execution.batch,
        submitted_reviews=len(execution.batch.reviews),
        successful_reviews=execution.successful_reviews,
        failed_reviews=execution.failed_reviews,
        skipped_reviews=execution.skipped_reviews,
        coverage=execution.coverage,
        elapsed_seconds=execution.elapsed_seconds,
        run_metadata=execution.run_metadata,
        state=execution.state,
    )


def load_amazon_dashboard(
    root: str | Path = AMAZON_CURRENT_DIR,
) -> DashboardData:
    """Reuse the Step 9 read-only dashboard loader for Amazon Current."""

    session = AmazonSessionPaths.from_root(root)
    return load_dashboard_data(session.root)


__all__ = [
    "AMAZON_CURRENT_DIR",
    "DEMO_PROTECTED_PATHS",
    "AmazonAnalysisError",
    "AmazonImport",
    "AmazonImportError",
    "AmazonPathSafetyError",
    "AmazonPipelineError",
    "AmazonPipelineResult",
    "AmazonSessionPaths",
    "amazon_analysis_available",
    "checkpoint_summary",
    "amazon_import_available",
    "failed_amazon_review_count",
    "interrupted_amazon_analysis_available",
    "load_amazon_dashboard",
    "load_amazon_import",
    "load_browser_import_metadata",
    "reset_amazon_analysis",
    "retry_failed_amazon_reviews",
    "run_amazon_pipeline",
    "save_amazon_import",
    "save_browser_amazon_import",
]
