"""Pure helpers for Step 10A interactive input and session analysis.

This module deliberately contains no Streamlit code.  The UI supplies input,
an already checked local Ollama client, and an optional progress callback.  All
generated artifacts are written below one explicit session directory so the
saved Demo Dataset is never used as a write target.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, fields
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

import analysis_runtime as runtime
from batch_analyzer import BatchRunResult, ProgressCallback
from cleaner import clean_reviews
from config import (
    DEMO_OUTPUT_PATHS,
    KEYWORD_MIN_REVIEW_COUNT,
    MIN_ASPECT_REVIEWS_FOR_RANKING,
    OLLAMA_MODEL,
    OUTPUTS_DIR,
    PRODUCTION_CHECKPOINT_INTERVAL,
    SESSION_CURRENT_DIR,
)
from input_loader import STANDARD_COLUMNS, load_reviews, validate_reviews
from llm_client import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    ReviewAnalysisClient,
    create_review_client,
)


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_INTERACTIVE_REVIEWS = 50
LONG_ANALYSIS_REVIEW_COUNT = 20
CURRENT_SESSION_DIR = SESSION_CURRENT_DIR

DASHBOARD_OUTPUT_NAMES = (
    "analyzed_reviews.csv",
    "aspect_summary.csv",
    "business_summary.json",
    "aspect_evidence.json",
    "keyword_summary.csv",
    "sentiment_keywords.json",
    "aspect_keywords.json",
)

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "review_id": ("review_id", "reviewid", "id"),
    "rating": ("rating", "stars", "star_rating", "score"),
    "title": ("title", "review_title", "headline", "subject"),
    "content": ("content", "review", "review_text", "text", "comment", "body"),
    "date": ("date", "review_date", "reviewed_at", "created_at"),
    "verified_purchase": (
        "verified_purchase",
        "verified",
        "verified purchase",
        "is_verified",
    ),
}


class InteractiveInputError(ValueError):
    """Interactive data cannot safely enter the analysis pipeline."""


class InteractiveLimitConfirmationRequired(InteractiveInputError):
    """The UI must obtain explicit consent before truncating the input."""


class SessionSafetyError(ValueError):
    """A reset target is not a narrowly scoped ``session/current`` folder."""


class InteractivePipelineError(RuntimeError):
    """The interactive pipeline could not publish a complete result."""


@dataclass(frozen=True)
class SessionPaths:
    """Every input, checkpoint, and dashboard artifact for one current run."""

    root: Path
    raw_reviews: Path
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
    def from_root(cls, root: str | Path) -> "SessionPaths":
        directory = Path(root)
        return cls(
            root=directory,
            raw_reviews=directory / "raw_reviews.csv",
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
    def current(cls) -> "SessionPaths":
        return cls.from_root(CURRENT_SESSION_DIR)


@dataclass(frozen=True)
class InteractivePipelineResult:
    """Concise completion summary consumed by the Streamlit page."""

    paths: SessionPaths
    batch: BatchRunResult
    submitted_reviews: int
    successful_reviews: int
    failed_reviews: int
    skipped_reviews: int
    coverage: float
    elapsed_seconds: float
    run_metadata: dict[str, Any]
    state: runtime.AnalysisState


def read_uploaded_csv(
    payload: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> pd.DataFrame:
    """Parse a small uploaded CSV with predictable, friendly failures."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if not payload:
        raise InteractiveInputError("Unable to read this CSV file. The file is empty.")
    if len(payload) > max_bytes:
        raise InteractiveInputError(
            "This CSV file is larger than the 10 MB upload limit."
        )

    last_error: Exception | None = None
    # UTF-8-SIG also accepts ordinary UTF-8.  CP1252 is one bounded fallback
    # for common exported review files; this is intentionally not an encoding
    # guessing system.
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            frame = pd.read_csv(
                BytesIO(payload),
                dtype="string",
                encoding=encoding,
                keep_default_na=False,
            )
            break
        except UnicodeDecodeError as exc:
            last_error = exc
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise InteractiveInputError("Unable to read this CSV file.") from exc
    else:
        raise InteractiveInputError("Unable to read this CSV file.") from last_error

    if frame.empty:
        raise InteractiveInputError("The CSV must contain at least one review row.")
    return frame


def suggest_column_mapping(columns: Sequence[object]) -> dict[str, str | None]:
    """Return conservative exact-name suggestions that the user may override."""

    available: dict[str, str] = {}
    for column in columns:
        name = str(column)
        available.setdefault(name.strip().casefold(), name)

    suggestions: dict[str, str | None] = {}
    for field in STANDARD_COLUMNS:
        suggestions[field] = next(
            (
                available[alias.casefold()]
                for alias in _COLUMN_ALIASES[field]
                if alias.casefold() in available
            ),
            None,
        )
    return suggestions


def _blank_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return series.isna() | text.str.strip().eq("").fillna(True)


def _generated_ids(
    existing: pd.Series,
    *,
    prefix: str,
) -> pd.Series:
    values = existing.astype("string").copy(deep=True)
    blank = _blank_mask(values)
    used = {
        str(value).strip()
        for value in values.loc[~blank].tolist()
        if str(value).strip()
    }
    next_number = 1
    for index in values.index[blank]:
        while True:
            candidate = f"{prefix}_{next_number:04d}"
            next_number += 1
            if candidate not in used:
                break
        values.loc[index] = candidate
        used.add(candidate)
    return values


def normalize_uploaded_reviews(
    frame: pd.DataFrame,
    mapping: Mapping[str, str | None],
    *,
    id_prefix: str = "UPLOAD",
) -> pd.DataFrame:
    """Map an uploaded table into only the six stable review input columns."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    content_column = mapping.get("content")
    if not isinstance(content_column, str) or not content_column.strip():
        raise InteractiveInputError("Please select a Review Content column.")
    if content_column not in frame.columns:
        raise InteractiveInputError("The selected Review Content column was not found.")

    normalized = pd.DataFrame(index=frame.index)
    for field in STANDARD_COLUMNS:
        source = mapping.get(field)
        if isinstance(source, str) and source:
            if source not in frame.columns:
                raise InteractiveInputError(
                    f"The selected column for {field} was not found."
                )
            normalized[field] = frame[source].copy(deep=True)
        else:
            normalized[field] = pd.Series(pd.NA, index=frame.index, dtype="object")

    normalized["review_id"] = _generated_ids(
        normalized["review_id"],
        prefix=id_prefix,
    )
    return normalized.loc[:, list(STANDARD_COLUMNS)].reset_index(drop=True)


def parse_pasted_reviews(
    text: object,
    *,
    id_prefix: str = "PASTE",
) -> pd.DataFrame:
    """Treat each non-empty line as one review and generate stable IDs."""

    if text is None:
        lines: list[str] = []
    else:
        # Whitespace decides whether a line is empty, but the stored content is
        # the user's original line.  Cleaner owns normalization and preserves
        # this value later as original_content.
        lines = [line for line in str(text).splitlines() if line.strip()]
    if not lines:
        raise InteractiveInputError("Please paste at least one review.")

    count = len(lines)
    return pd.DataFrame(
        {
            "review_id": [f"{id_prefix}_{index:04d}" for index in range(1, count + 1)],
            "rating": [pd.NA] * count,
            "title": [pd.NA] * count,
            "content": lines,
            "date": [pd.NA] * count,
            "verified_purchase": [pd.NA] * count,
        },
        columns=list(STANDARD_COLUMNS),
    )


def validate_interactive_reviews(frame: pd.DataFrame) -> dict[str, Any]:
    """Reuse Step 2 validation and add the UI's all-empty-content guard."""

    result = dict(validate_reviews(frame))
    errors = list(result.get("errors", []))
    if frame.empty and "CSV contains no review rows." not in errors:
        errors.append("The input must contain at least one review.")
    if "content" in frame.columns and len(frame) and _blank_mask(frame["content"]).all():
        errors.append("All review content is empty.")
    result["errors"] = errors
    result["is_valid"] = not errors
    return result


def apply_interactive_limit(
    frame: pd.DataFrame,
    *,
    allow_truncate: bool = False,
    max_reviews: int = MAX_INTERACTIVE_REVIEWS,
) -> tuple[pd.DataFrame, bool]:
    """Return a complete, bounded dataset after explicit truncation consent."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if isinstance(max_reviews, bool) or not isinstance(max_reviews, int) or max_reviews <= 0:
        raise ValueError("max_reviews must be a positive integer")
    if len(frame) <= max_reviews:
        return frame.copy(deep=True), False
    if not allow_truncate:
        raise InteractiveLimitConfirmationRequired(
            f"This interactive demo is limited to the first {max_reviews} reviews."
        )
    return frame.iloc[:max_reviews].copy(deep=True).reset_index(drop=True), True


def create_checked_ollama_client() -> tuple[ReviewAnalysisClient, dict[str, str]]:
    """Create only the local provider and verify it before any session write."""

    client = create_review_client(provider="ollama")
    checker = getattr(client, "check_connection", None)
    if not callable(checker):
        raise LLMConfigurationError("The local Ollama client cannot check its connection.")
    status = checker()
    return client, dict(status)


def friendly_ollama_error(error: BaseException) -> str:
    """Translate expected local-model failures into user-facing UI messages."""

    if isinstance(error, LLMModelNotFoundError):
        return f"Model {OLLAMA_MODEL} is not installed."
    if isinstance(error, LLMConnectionError):
        return (
            "Local AI is currently unavailable.\n\n"
            "Please start Ollama and try again."
        )
    if isinstance(error, LLMConfigurationError):
        return "Local AI model configuration is unavailable."
    return "The local AI analysis could not be started. Please try again."


def _validated_reset_target(root: str | Path) -> Path:
    resolved = Path(root).resolve()
    if resolved.name != "current" or resolved.parent.name != "session":
        raise SessionSafetyError(
            "Only a narrowly scoped session/current directory can be reset."
        )
    outputs_root = OUTPUTS_DIR.resolve()
    if outputs_root in resolved.parents and resolved != CURRENT_SESSION_DIR.resolve():
        raise SessionSafetyError(
            "Only the official Upload/Paste session directory may be modified."
        )
    return resolved


def _canonical_session_paths(paths: SessionPaths | None) -> SessionPaths:
    """Reject filename overrides that could target frozen Demo artifacts."""

    candidate = paths or SessionPaths.current()
    if not isinstance(candidate, SessionPaths):
        raise TypeError("paths must be a SessionPaths instance")
    canonical = SessionPaths.from_root(_validated_reset_target(candidate.root))
    for field in fields(SessionPaths):
        actual = Path(getattr(candidate, field.name)).resolve()
        expected = Path(getattr(canonical, field.name)).resolve()
        if actual != expected:
            raise SessionSafetyError(
                "Current Analysis paths must use the canonical filenames."
            )
        if field.name != "root" and actual in DEMO_OUTPUT_PATHS:
            raise SessionSafetyError(
                "The frozen Demo Dataset is protected from session writes."
            )
    return canonical


def _reset_current_session_unlocked(root: str | Path) -> None:
    target = _validated_reset_target(root)
    if target.exists():
        if not target.is_dir():
            raise SessionSafetyError("The current session target is not a directory.")
        shutil.rmtree(target)


def reset_current_session(root: str | Path = CURRENT_SESSION_DIR) -> None:
    """Delete only an idle, explicitly scoped ``session/current`` directory."""

    target = _validated_reset_target(root)
    with runtime.analysis_run_lock(target, operation="reset"):
        _reset_current_session_unlocked(target)


def _with_current_session_lock(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        session = _canonical_session_paths(kwargs.get("paths"))
        with runtime.analysis_run_lock(session.root, operation=function.__name__):
            return function(*args, **kwargs)

    return wrapped


def current_analysis_available(root: str | Path = CURRENT_SESSION_DIR) -> bool:
    """A Current Analysis is available only when all seven dashboard files exist."""

    directory = Path(root)
    return all((directory / name).is_file() for name in DASHBOARD_OUTPUT_NAMES)


def failed_review_count(root: str | Path = CURRENT_SESSION_DIR) -> int:
    """Return the published Current Analysis failed-row count for UI gating."""

    analyzed_path = Path(root) / "analyzed_reviews.csv"
    if not analyzed_path.is_file():
        return 0
    try:
        statuses = pd.read_csv(
            analyzed_path,
            usecols=["analysis_status"],
            dtype="string",
            keep_default_na=False,
        )["analysis_status"]
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return 0
    return int(statuses.str.casefold().eq("failed").sum())


def interrupted_analysis_available(root: str | Path = CURRENT_SESSION_DIR) -> bool:
    """Return true when a resumable session checkpoint exists but is incomplete."""

    directory = Path(root)
    if current_analysis_available(directory):
        return False
    required = (
        "raw_reviews.csv",
        "analyzed_reviews_checkpoint.csv",
        "analyzed_reviews_checkpoint.meta.json",
    )
    return all((directory / name).is_file() for name in required)


def resolve_dashboard_directory(
    dataset: str,
    *,
    demo_directory: str | Path = OUTPUTS_DIR,
    current_directory: str | Path = CURRENT_SESSION_DIR,
) -> Path:
    """Resolve a user-facing saved dataset name without mutating either source."""

    normalized = dataset.strip().casefold()
    if normalized in {"demo", "demo dataset"}:
        return Path(demo_directory)
    if normalized in {"current", "current analysis"}:
        return Path(current_directory)
    raise ValueError(f"Unsupported dashboard dataset: {dataset}")


def small_sample_message(successful_reviews: int) -> str | None:
    """Return the shared directional-warning copy for a small analysis."""

    if successful_reviews < 30:
        return (
            "Small sample size. Insights should be treated as directional "
            "rather than representative."
        )
    return None


def large_dataset_message(review_count: int) -> str | None:
    """Return transparent long-run guidance without truncating user data."""

    if review_count >= 500:
        return (
            "Large local analysis detected. Progress is checkpointed and can be "
            "resumed if the app or computer is interrupted."
        )
    if review_count > 100:
        return (
            "Large dataset detected. This analysis may take some time. Progress "
            "is checkpointed and can be resumed."
        )
    if review_count == 100:
        return (
            "100 reviews will be analyzed locally using qwen3:8b. You can leave "
            "this page open while analysis runs."
        )
    if review_count > LONG_ANALYSIS_REVIEW_COUNT:
        return (
            f"{review_count} reviews will be analyzed locally. Keep this page "
            "open; a live estimate will appear after the first few reviews."
        )
    return None


def checkpoint_summary(
    paths: SessionPaths | None = None,
) -> runtime.CheckpointSummary | None:
    """Return safe resume counts for the current Upload/Paste session."""

    session = _canonical_session_paths(paths)
    if not session.raw_reviews.is_file() or not session.checkpoint.is_file():
        return None
    try:
        total = len(load_reviews(session.raw_reviews))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return runtime.inspect_checkpoint(session.checkpoint, total_reviews=total)


def _atomic_write_raw_reviews(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_cleaned_session_reviews(session: SessionPaths) -> pd.DataFrame:
    loaded = load_reviews(session.raw_reviews)
    validation = validate_interactive_reviews(loaded)
    if not validation["is_valid"]:
        raise InteractiveInputError(" ".join(validation["errors"]))
    cleaned = clean_reviews(loaded)
    if not cleaned["is_analyzable"].eq(True).any():
        raise InteractiveInputError("No reviews contain analyzable text after cleaning.")
    return cleaned


@_with_current_session_lock
def run_interactive_pipeline(
    reviews: pd.DataFrame | None,
    *,
    llm_client: ReviewAnalysisClient,
    paths: SessionPaths | None = None,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_snapshot_callback: runtime.ProgressSnapshotCallback | None = None,
    source_type: str = "interactive",
    checkpoint_interval: int = PRODUCTION_CHECKPOINT_INTERVAL,
    min_aspect_reviews: int = MIN_ASPECT_REVIEWS_FOR_RANKING,
    min_keyword_reviews: int = KEYWORD_MIN_REVIEW_COUNT,
    clock: Callable[[], float] = time.perf_counter,
) -> InteractivePipelineResult:
    """Run or resume validation through keyword outputs in one isolated session."""

    session = _canonical_session_paths(paths)
    started = clock()
    validation_time = 0.0
    preparing_time = 0.0

    if resume:
        if reviews is not None:
            raise ValueError("reviews must be None when resume is true")
        if not interrupted_analysis_available(session.root):
            raise InteractivePipelineError(
                "No interrupted Current Analysis checkpoint is available."
            )
        phase_started = clock()
        cleaned = _load_cleaned_session_reviews(session)
        validation_time = max(0.0, clock() - phase_started)
    else:
        if not isinstance(reviews, pd.DataFrame):
            raise TypeError("reviews must be a pandas DataFrame")
        phase_started = clock()
        validation = validate_interactive_reviews(reviews)
        if not validation["is_valid"]:
            raise InteractiveInputError(" ".join(validation["errors"]))
        precleaned = clean_reviews(reviews)
        if not precleaned["is_analyzable"].eq(True).any():
            raise InteractiveInputError(
                "No reviews contain analyzable text after cleaning."
            )
        validation_time = max(0.0, clock() - phase_started)

        # The UI checks Ollama before calling this function, so an unavailable
        # model cannot erase a previous Current Analysis or its checkpoint.
        phase_started = clock()
        _reset_current_session_unlocked(session.root)
        session.root.mkdir(parents=True, exist_ok=True)
        _atomic_write_raw_reviews(reviews.copy(deep=True), session.raw_reviews)
        preparing_time = max(0.0, clock() - phase_started)
        cleaned = precleaned

    execution = runtime.run_analysis_runtime(
        cleaned,
        source_type=source_type,
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
        raise InteractivePipelineError(
            "The review analysis stopped before the session was complete. "
            "Its checkpoint was preserved."
        )
    if execution.successful_reviews <= 0:
        raise InteractivePipelineError(
            "No reviews were analyzed successfully. The checkpoint was preserved."
        )
    if not execution.downstream_completed or not current_analysis_available(session.root):
        raise InteractivePipelineError(
            "The session finished but one or more dashboard outputs are missing."
        )
    return InteractivePipelineResult(
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


@_with_current_session_lock
def retry_failed_reviews(
    *,
    llm_client: ReviewAnalysisClient,
    paths: SessionPaths | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_snapshot_callback: runtime.ProgressSnapshotCallback | None = None,
    checkpoint_interval: int = PRODUCTION_CHECKPOINT_INTERVAL,
    min_aspect_reviews: int = MIN_ASPECT_REVIEWS_FOR_RANKING,
    min_keyword_reviews: int = KEYWORD_MIN_REVIEW_COUNT,
    clock: Callable[[], float] = time.perf_counter,
) -> InteractivePipelineResult:
    """Retry only failed Current Analysis rows and republish dashboard files."""

    session = _canonical_session_paths(paths)
    if not current_analysis_available(session.root):
        raise InteractivePipelineError(
            "No complete Current Analysis is available for failed-review retry."
        )
    if failed_review_count(session.root) <= 0:
        raise InteractivePipelineError(
            "Current Analysis has no failed reviews to retry."
        )

    started = clock()
    cleaned = _load_cleaned_session_reviews(session)
    execution = runtime.run_analysis_runtime(
        cleaned,
        source_type="interactive",
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
        raise InteractivePipelineError(
            "Failed reviews could not be republished. The checkpoint was preserved."
        )
    return InteractivePipelineResult(
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


__all__ = [
    "CURRENT_SESSION_DIR",
    "DASHBOARD_OUTPUT_NAMES",
    "InteractiveInputError",
    "InteractiveLimitConfirmationRequired",
    "InteractivePipelineError",
    "InteractivePipelineResult",
    "LONG_ANALYSIS_REVIEW_COUNT",
    "MAX_INTERACTIVE_REVIEWS",
    "MAX_UPLOAD_BYTES",
    "SessionPaths",
    "SessionSafetyError",
    "apply_interactive_limit",
    "create_checked_ollama_client",
    "current_analysis_available",
    "checkpoint_summary",
    "failed_review_count",
    "friendly_ollama_error",
    "interrupted_analysis_available",
    "large_dataset_message",
    "normalize_uploaded_reviews",
    "parse_pasted_reviews",
    "read_uploaded_csv",
    "reset_current_session",
    "resolve_dashboard_directory",
    "retry_failed_reviews",
    "run_interactive_pipeline",
    "small_sample_message",
    "suggest_column_mapping",
    "validate_interactive_reviews",
]
