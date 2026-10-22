"""串行批量分析、checkpoint、恢复与失败重试。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    ANALYZED_REVIEWS_PATH,
    ANALYSIS_CHECKPOINT_META_PATH,
    ANALYSIS_CHECKPOINT_PATH,
    BATCH_CHECKPOINT_INTERVAL,
    MAX_CONSECUTIVE_LLM_FAILURES,
)
from llm_client import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    ReviewAnalysisClient,
    create_review_client,
)
from review_analyzer import analyze_review


ProgressCallback = Callable[[int, int, str, str], None]
RuntimeEventCallback = Callable[[dict[str, Any]], None]

ANALYSIS_COLUMNS = (
    "analysis_status",
    "summary",
    "aspects",
    "unique_aspects",
    "positive_points",
    "negative_points",
    "error_message",
    "analysis_time_seconds",
)
_REQUIRED_CLEANING_COLUMNS = ("is_analyzable", "cleaned_content")
CHECKPOINT_JSON_COLUMNS = (
    "aspects",
    "unique_aspects",
    "positive_points",
    "negative_points",
)
_ALLOWED_STATUSES = frozenset({"success", "failed", "skipped"})
_PROVIDER_AVAILABILITY_ERRORS = (
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotFoundError,
)
TRANSIENT_PROVIDER_RETRY_DELAY_SECONDS = 1.0

# Future aggregators should exclude these categories from default concern,
# selling-point, and pain-point rankings unless the user opts in.
DEFAULT_EXCLUDED_BUSINESS_ASPECTS = frozenset({"Other"})


def _notify_progress(
    callback: ProgressCallback | None,
    current: int,
    total: int,
    review_id: str,
    status: str,
) -> None:
    """Treat UI progress reporting as best-effort, never as analysis logic."""

    if callback is None:
        return
    try:
        callback(current, total, review_id, status)
    except Exception:
        # A disconnected or rerunning UI must not turn a completed LLM result
        # into a failed review or prevent the next checkpoint from being saved.
        return


class CheckpointError(RuntimeError):
    """Base class for checkpoint loading and compatibility failures."""


class CheckpointInvalidError(CheckpointError):
    """The checkpoint cannot be trusted or parsed safely."""


class CheckpointMismatchError(CheckpointError):
    """The checkpoint belongs to a different input dataset."""


@dataclass(frozen=True)
class BatchRunResult:
    """Outcome of one fresh, resume, or retry-failed invocation."""

    reviews: pd.DataFrame
    total_input_rows: int
    previously_completed: int
    newly_processed: int
    attempted_reviews: int
    is_complete: bool
    interrupted: bool
    stopped_for_failures: bool
    checkpoint_path: Path
    final_output_path: Path | None
    retry_count: int = 0
    model_failures: int = 0
    llm_time_seconds: float = 0.0
    postprocessing_time_seconds: float = 0.0
    checkpoint_time_seconds: float = 0.0


def _is_true(value: object) -> bool:
    """Only a real, non-missing truthy value is analyzable."""

    try:
        return False if pd.isna(value) else bool(value)
    except (TypeError, ValueError):
        return False


def _validate_limit(limit: int | None) -> None:
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")


def _validate_provider_retry_delay(delay_seconds: float) -> None:
    if (
        isinstance(delay_seconds, bool)
        or not isinstance(delay_seconds, (int, float))
        or delay_seconds < 0
    ):
        raise ValueError("provider_retry_delay_seconds must be a non-negative number")


def _analyze_with_transient_provider_retry(
    review_text: str,
    *,
    client: ReviewAnalysisClient,
    sleeper: Callable[[float], None],
    retry_delay_seconds: float,
    retry_observer: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Retry one explicit provider connection failure exactly once.

    Provider adapters classify connection resets, timeouts, and temporary
    service availability failures as ``LLMConnectionError``. Validation,
    schema, evidence, configuration, model, and ordinary analysis failures do
    not enter this bounded retry path.
    """

    try:
        return analyze_review(review_text, llm_client=client)
    except LLMConnectionError:
        if retry_observer is not None:
            retry_observer()
        sleeper(retry_delay_seconds)
        return analyze_review(review_text, llm_client=client)


def _notify_runtime_event(
    callback: RuntimeEventCallback | None,
    event: dict[str, Any],
) -> None:
    """Publish non-sensitive runtime telemetry without affecting analysis."""

    if callback is None:
        return
    try:
        callback(dict(event))
    except Exception:
        return


def add_source_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stable IDs derived from the current source row order."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if "source_row_id" in df.columns:
        raise ValueError(
            "Input already contains the reserved source_row_id column."
        )

    result = df.copy(deep=True)
    result.insert(
        0,
        "source_row_id",
        pd.Series(range(len(result)), index=result.index, dtype="int64"),
    )
    return result


def _signature_value(value: object) -> str | None:
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def calculate_input_signature(df: pd.DataFrame) -> str:
    """Hash stable row identity fields to bind a checkpoint to one dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    prepared = df if "source_row_id" in df.columns else add_source_row_ids(df)
    required = ("source_row_id", "review_id", "cleaned_content")
    missing = [column for column in required if column not in prepared.columns]
    if missing:
        raise ValueError(
            f"Cannot calculate input signature without: {', '.join(missing)}"
        )

    rows = [
        {
            "source_row_id": int(row.source_row_id),
            "review_id": _signature_value(row.review_id),
            "cleaned_content": _signature_value(row.cleaned_content),
        }
        for row in prepared.loc[:, list(required)].itertuples(index=False)
    ]
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_batch_rows(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Return the input prefix ending at the Nth analyzable review.

    Skipped rows encountered before that point remain in the result and do not
    consume the limit. Rows beyond the limit are outside this run and are not
    incorrectly labelled as skipped.
    """

    if limit is None:
        return df

    positions: list[int] = []
    analyzable_count = 0
    for position, value in enumerate(df["is_analyzable"].tolist()):
        positions.append(position)
        if _is_true(value):
            analyzable_count += 1
            if analyzable_count == limit:
                break
    return df.iloc[positions]


def _json_list(value: object) -> str:
    """Serialize list-shaped analysis fields as stable JSON for CSV output."""

    return json.dumps(value if isinstance(value, list) else [], ensure_ascii=False)


def build_unique_aspects(aspects: object) -> list[dict[str, Any]]:
    """Build review-level aspect counts without changing mention-level data.

    Mentions are grouped by their already-normalized ``aspect`` and
    ``sentiment`` values. First-seen order is preserved, while different
    sentiments for the same aspect remain separate review-level records.
    """

    if not isinstance(aspects, list):
        return []

    unique: list[dict[str, Any]] = []
    positions: dict[tuple[str, str], int] = {}
    for mention in aspects:
        if not isinstance(mention, dict):
            continue
        aspect = mention.get("aspect")
        sentiment = mention.get("sentiment")
        if not isinstance(aspect, str) or not isinstance(sentiment, str):
            continue
        key = (aspect, sentiment)
        if key in positions:
            unique[positions[key]]["mention_count"] += 1
            continue

        positions[key] = len(unique)
        unique.append(
            {
                "aspect": aspect,
                "sentiment": sentiment,
                "mention_count": 1,
                "include_in_business_metrics": (
                    aspect not in DEFAULT_EXCLUDED_BUSINESS_ASPECTS
                ),
            }
        )

    return unique


def _review_label(value: object) -> str:
    try:
        if bool(pd.isna(value)):
            return "<missing_review_id>"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or "<missing_review_id>"


def _skip_reason(value: object) -> str:
    try:
        if bool(pd.isna(value)):
            return "not_analyzable"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or "not_analyzable"


def _output_columns(df: pd.DataFrame) -> list[str]:
    """Preserve every input column and append the stable analysis contract."""

    columns = list(df.columns)
    fallback_columns = (
        "review_id",
        "rating",
        "title",
        "original_content",
        "cleaned_content",
        "skip_reason",
    )
    for column in fallback_columns + ANALYSIS_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def analyze_reviews(
    df: pd.DataFrame,
    limit: int | None = None,
    *,
    llm_client: ReviewAnalysisClient | None = None,
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    provider_retry_delay_seconds: float = TRANSIENT_PROVIDER_RETRY_DELAY_SECONDS,
) -> pd.DataFrame:
    """Analyze cleaned reviews serially while isolating failures per row.

    Only rows whose ``is_analyzable`` value is true call the existing
    :func:`review_analyzer.analyze_review`. The input DataFrame is never
    modified.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = [
        column
        for column in _REQUIRED_CLEANING_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing cleaned review columns: {', '.join(missing)}")
    _validate_limit(limit)
    _validate_provider_retry_delay(provider_retry_delay_seconds)

    prepared = add_source_row_ids(df)
    selected = _select_batch_rows(prepared, limit)
    output_columns = _output_columns(prepared)
    records: list[dict[str, Any]] = []
    active_client = llm_client
    total_rows = len(selected)

    for progress_index, (_, row) in enumerate(selected.iterrows(), start=1):
        record = row.to_dict()
        record.setdefault("review_id", pd.NA)
        record.setdefault("rating", pd.NA)
        record.setdefault("title", pd.NA)
        record.setdefault("original_content", record.get("content", pd.NA))
        record.setdefault("cleaned_content", "")
        record.setdefault("skip_reason", "")
        review_id = _review_label(record["review_id"])

        if not _is_true(record.get("is_analyzable")):
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
            if show_progress:
                reason = _skip_reason(record.get("skip_reason"))
                print(f"[{progress_index}/{total_rows}] {review_id} skipped | {reason}")
            records.append(record)
            _notify_progress(
                progress_callback,
                progress_index,
                total_rows,
                review_id,
                "skipped",
            )
            continue

        _notify_progress(
            progress_callback,
            progress_index,
            total_rows,
            review_id,
            "analyzing",
        )
        started = clock()
        try:
            if active_client is None:
                active_client = create_review_client()
            analysis = _analyze_with_transient_provider_retry(
                str(record["cleaned_content"]),
                client=active_client,
                sleeper=sleeper,
                retry_delay_seconds=provider_retry_delay_seconds,
            )
            elapsed = max(0.0, clock() - started)
            aspects = analysis["aspects"]
            record.update(
                {
                    "analysis_status": "success",
                    "summary": str(analysis["summary"]),
                    "aspects": _json_list(aspects),
                    "unique_aspects": _json_list(build_unique_aspects(aspects)),
                    "positive_points": _json_list(analysis["positive_points"]),
                    "negative_points": _json_list(analysis["negative_points"]),
                    "error_message": "",
                    "analysis_time_seconds": elapsed,
                }
            )
            if show_progress:
                print(
                    f"[{progress_index}/{total_rows}] {review_id} success | "
                    f"{elapsed:.2f}s"
                )
        except Exception as exc:
            elapsed = max(0.0, clock() - started)
            error_message = str(exc) or type(exc).__name__
            record.update(
                {
                    "analysis_status": "failed",
                    "summary": "",
                    "aspects": "[]",
                    "unique_aspects": "[]",
                    "positive_points": "[]",
                    "negative_points": "[]",
                    "error_message": error_message,
                    "analysis_time_seconds": elapsed,
                }
            )
            if show_progress:
                print(
                    f"[{progress_index}/{total_rows}] {review_id} failed | "
                    f"{elapsed:.2f}s"
                )

        _notify_progress(
            progress_callback,
            progress_index,
            total_rows,
            review_id,
            str(record["analysis_status"]),
        )

        records.append(record)

    return pd.DataFrame.from_records(records, columns=output_columns)


def _temporary_path(path: Path) -> Path:
    return path.with_suffix(".tmp")


def _write_csv_temp(df: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_text_temp(text: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _atomic_write_csv(df: pd.DataFrame, destination: Path) -> Path:
    temporary = _write_csv_temp(df, destination)
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def save_analyzed_reviews(
    df: pd.DataFrame,
    file_path: str | Path = ANALYZED_REVIEWS_PATH,
) -> Path:
    """Atomically save analysis results without touching the raw input CSV."""

    return _atomic_write_csv(df, Path(file_path))


def save_analysis_checkpoint(
    df: pd.DataFrame,
    input_signature: str,
    total_rows: int,
    checkpoint_path: str | Path = ANALYSIS_CHECKPOINT_PATH,
    meta_path: str | Path = ANALYSIS_CHECKPOINT_META_PATH,
    checkpoint_context: Mapping[str, str] | None = None,
) -> Path:
    """Atomically replace checkpoint data and its dataset metadata."""

    checkpoint = Path(checkpoint_path)
    metadata = Path(meta_path)
    ordered = (
        df.sort_values("source_row_id", kind="stable").reset_index(drop=True)
        if "source_row_id" in df.columns
        else df.copy(deep=True)
    )
    meta_payload = json.dumps(
        {
            "schema_version": 2,
            "input_signature": input_signature,
            "total_rows": total_rows,
            "analysis_context": dict(checkpoint_context or {}),
            "signature_fields": [
                "source_row_id",
                "review_id",
                "cleaned_content",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )

    csv_temp = _write_csv_temp(ordered, checkpoint)
    try:
        meta_temp = _write_text_temp(meta_payload + "\n", metadata)
    except BaseException:
        csv_temp.unlink(missing_ok=True)
        raise

    try:
        os.replace(csv_temp, checkpoint)
        os.replace(meta_temp, metadata)
    except BaseException:
        csv_temp.unlink(missing_ok=True)
        meta_temp.unlink(missing_ok=True)
        raise
    return checkpoint


def _invalid_checkpoint(detail: str) -> CheckpointInvalidError:
    return CheckpointInvalidError(
        f"Checkpoint is invalid or corrupted. {detail}"
    )


def load_analysis_checkpoint(
    expected_signature: str,
    expected_total_rows: int,
    *,
    checkpoint_path: str | Path = ANALYSIS_CHECKPOINT_PATH,
    meta_path: str | Path = ANALYSIS_CHECKPOINT_META_PATH,
    expected_input_columns: list[str] | tuple[str, ...] = (),
    expected_source_rows: pd.DataFrame | None = None,
    expected_checkpoint_context: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load and strictly validate a checkpoint before any LLM call."""

    checkpoint = Path(checkpoint_path)
    metadata = Path(meta_path)
    if not checkpoint.is_file() or not metadata.is_file():
        raise _invalid_checkpoint("Checkpoint CSV or metadata file is missing.")

    try:
        meta_payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _invalid_checkpoint("Metadata could not be read.") from exc
    if not isinstance(meta_payload, dict):
        raise _invalid_checkpoint("Metadata must be a JSON object.")

    signature = meta_payload.get("input_signature")
    total_rows = meta_payload.get("total_rows")
    if not isinstance(signature, str) or isinstance(total_rows, bool):
        raise _invalid_checkpoint("Metadata fields are missing or invalid.")
    try:
        stored_total_rows = int(total_rows)
    except (TypeError, ValueError) as exc:
        raise _invalid_checkpoint("Metadata total_rows is invalid.") from exc
    if signature != expected_signature or stored_total_rows != expected_total_rows:
        raise CheckpointMismatchError(
            "Checkpoint does not match the current input dataset."
        )
    if expected_checkpoint_context is not None:
        stored_context = meta_payload.get("analysis_context")
        if not isinstance(stored_context, dict) or stored_context != dict(
            expected_checkpoint_context
        ):
            raise CheckpointMismatchError(
                "Checkpoint does not match the current model and analysis version."
            )

    try:
        # Read every persisted value as text so an untouched checkpoint row is
        # written back byte-for-byte equivalently.  Without an explicit dtype,
        # pandas infers values such as ``true`` as the Python boolean ``True``;
        # a later resume save would then silently rewrite that source field.
        frame = pd.read_csv(
            checkpoint,
            dtype="string",
            keep_default_na=False,
            encoding="utf-8",
        )
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise _invalid_checkpoint("Checkpoint CSV could not be read.") from exc

    required_columns = {
        "source_row_id",
        "review_id",
        "analysis_status",
        "summary",
        "error_message",
        "analysis_time_seconds",
        *CHECKPOINT_JSON_COLUMNS,
        *expected_input_columns,
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise _invalid_checkpoint(
            f"Missing required columns: {', '.join(missing)}."
        )

    source_ids = pd.to_numeric(frame["source_row_id"], errors="coerce")
    if (
        source_ids.isna().any()
        or source_ids.lt(0).any()
        or source_ids.ge(expected_total_rows).any()
        or source_ids.mod(1).ne(0).any()
    ):
        raise _invalid_checkpoint("source_row_id contains invalid values.")
    frame["source_row_id"] = source_ids.astype("int64")
    if frame["source_row_id"].duplicated().any():
        raise _invalid_checkpoint("source_row_id contains duplicates.")

    invalid_statuses = set(frame["analysis_status"]).difference(_ALLOWED_STATUSES)
    if invalid_statuses:
        raise _invalid_checkpoint("analysis_status contains unsupported values.")

    for column in CHECKPOINT_JSON_COLUMNS:
        for row_number, value in enumerate(frame[column], start=2):
            try:
                decoded = json.loads(value)
            except (TypeError, json.JSONDecodeError) as exc:
                raise _invalid_checkpoint(
                    f"{column} contains invalid JSON at CSV row {row_number}."
                ) from exc
            if not isinstance(decoded, list):
                raise _invalid_checkpoint(
                    f"{column} must contain JSON arrays."
                )

    analysis_times = pd.to_numeric(
        frame["analysis_time_seconds"],
        errors="coerce",
    )
    if analysis_times.isna().any() or analysis_times.lt(0).any():
        raise _invalid_checkpoint("analysis_time_seconds contains invalid values.")

    if expected_source_rows is not None:
        expected_by_id = expected_source_rows.set_index("source_row_id")
        for _, checkpoint_row in frame.iterrows():
            source_id = int(checkpoint_row["source_row_id"])
            expected_row = expected_by_id.loc[source_id]
            for column in ("review_id", "cleaned_content"):
                if _signature_value(checkpoint_row[column]) != _signature_value(
                    expected_row[column]
                ):
                    raise _invalid_checkpoint(
                        f"Row identity differs for source_row_id {source_id}."
                    )
    return frame.sort_values("source_row_id", kind="stable").reset_index(drop=True)


def _base_record(row: pd.Series) -> dict[str, Any]:
    record = row.to_dict()
    record.setdefault("review_id", pd.NA)
    record.setdefault("rating", pd.NA)
    record.setdefault("title", pd.NA)
    record.setdefault("original_content", record.get("content", pd.NA))
    record.setdefault("cleaned_content", "")
    record.setdefault("skip_reason", "")
    return record


def _skipped_record(row: pd.Series) -> dict[str, Any]:
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
    return record


def _success_record(
    row: pd.Series,
    analysis: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    record = _base_record(row)
    aspects = analysis["aspects"]
    record.update(
        {
            "analysis_status": "success",
            "summary": str(analysis["summary"]),
            "aspects": _json_list(aspects),
            "unique_aspects": _json_list(build_unique_aspects(aspects)),
            "positive_points": _json_list(analysis["positive_points"]),
            "negative_points": _json_list(analysis["negative_points"]),
            "error_message": "",
            "analysis_time_seconds": elapsed,
        }
    )
    return record


def _failed_record(
    row: pd.Series,
    error: BaseException,
    elapsed: float,
) -> dict[str, Any]:
    record = _base_record(row)
    record.update(
        {
            "analysis_status": "failed",
            "summary": "",
            "aspects": "[]",
            "unique_aspects": "[]",
            "positive_points": "[]",
            "negative_points": "[]",
            "error_message": str(error) or type(error).__name__,
            "analysis_time_seconds": elapsed,
        }
    )
    return record


def _checkpoint_frame(
    records: dict[int, dict[str, Any]],
    columns: list[str],
) -> pd.DataFrame:
    ordered_records = [records[source_id] for source_id in sorted(records)]
    return pd.DataFrame.from_records(ordered_records, columns=columns)


def run_checkpointed_batch(
    df: pd.DataFrame,
    limit: int | None = None,
    *,
    resume: bool = False,
    retry_failed: bool = False,
    llm_client: ReviewAnalysisClient | None = None,
    checkpoint_path: str | Path = ANALYSIS_CHECKPOINT_PATH,
    meta_path: str | Path = ANALYSIS_CHECKPOINT_META_PATH,
    final_output_path: str | Path = ANALYZED_REVIEWS_PATH,
    checkpoint_interval: int = BATCH_CHECKPOINT_INTERVAL,
    max_consecutive_failures: int = MAX_CONSECUTIVE_LLM_FAILURES,
    progress_callback: ProgressCallback | None = None,
    runtime_event_callback: RuntimeEventCallback | None = None,
    show_progress: bool = True,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    provider_retry_delay_seconds: float = TRANSIENT_PROVIDER_RETRY_DELAY_SECONDS,
    checkpoint_context: Mapping[str, str] | None = None,
) -> BatchRunResult:
    """Run a recoverable serial batch without reprocessing completed rows."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = [
        column
        for column in _REQUIRED_CLEANING_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing cleaned review columns: {', '.join(missing)}")
    _validate_limit(limit)
    _validate_provider_retry_delay(provider_retry_delay_seconds)
    for value, name in (
        (checkpoint_interval, "checkpoint_interval"),
        (max_consecutive_failures, "max_consecutive_failures"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    prepared = add_source_row_ids(df)
    output_columns = _output_columns(prepared)
    input_signature = calculate_input_signature(prepared)
    total_rows = len(prepared)
    checkpoint = Path(checkpoint_path)
    metadata = Path(meta_path)
    final_output = Path(final_output_path)

    if resume or retry_failed:
        loaded = load_analysis_checkpoint(
            input_signature,
            total_rows,
            checkpoint_path=checkpoint,
            meta_path=metadata,
            expected_input_columns=list(prepared.columns),
            expected_source_rows=prepared,
            expected_checkpoint_context=checkpoint_context,
        )
    else:
        loaded = pd.DataFrame(columns=output_columns)

    records: dict[int, dict[str, Any]] = {
        int(row["source_row_id"]): row.to_dict()
        for _, row in loaded.iterrows()
    }
    existing_statuses = {
        source_id: str(record["analysis_status"])
        for source_id, record in records.items()
    }
    previously_completed = sum(
        status in {"success", "skipped"}
        for status in existing_statuses.values()
    )
    all_source_ids = prepared["source_row_id"].astype(int).tolist()
    unprocessed_ids = [
        source_id for source_id in all_source_ids if source_id not in records
    ]
    failed_ids = [
        source_id
        for source_id in sorted(records)
        if existing_statuses[source_id] == "failed"
    ]

    if retry_failed and resume:
        target_ids = failed_ids + unprocessed_ids
    elif retry_failed:
        target_ids = failed_ids
    elif resume:
        target_ids = unprocessed_ids
    else:
        target_ids = all_source_ids

    if show_progress and resume:
        print("Resume detected.")
        print(f"{previously_completed} rows already completed.")
        print(f"{len(unprocessed_ids)} rows remaining.")
    if show_progress and retry_failed:
        print("Retry failed detected.")
        print(f"{len(failed_ids)} failed rows queued for retry.")

    rows_by_id = {
        int(row["source_row_id"]): row
        for _, row in prepared.iterrows()
    }
    newly_processed_ids: set[int] = set()
    attempted_reviews = 0
    consecutive_failures = 0
    active_client = llm_client
    interrupted = False
    stopped_for_failures = False
    retry_count = 0
    model_failures = 0
    llm_time_seconds = 0.0
    postprocessing_time_seconds = 0.0
    checkpoint_time_seconds = 0.0

    def save_current_checkpoint() -> pd.DataFrame:
        nonlocal checkpoint_time_seconds
        checkpoint_started = clock()
        current = _checkpoint_frame(records, output_columns)
        save_analysis_checkpoint(
            current,
            input_signature,
            total_rows,
            checkpoint_path=checkpoint,
            meta_path=metadata,
            checkpoint_context=checkpoint_context,
        )
        checkpoint_time_seconds += max(0.0, clock() - checkpoint_started)
        return current

    current_checkpoint = _checkpoint_frame(records, output_columns)
    for source_id in target_ids:
        row = rows_by_id[source_id]
        analyzable = _is_true(row.get("is_analyzable"))
        if analyzable and limit is not None and attempted_reviews >= limit:
            break

        review_id = _review_label(row.get("review_id"))
        progress = f"[{source_id + 1}/{total_rows}] {review_id}"
        if not analyzable:
            postprocess_started = clock()
            records[source_id] = _skipped_record(row)
            postprocessing_time_seconds += max(0.0, clock() - postprocess_started)
            newly_processed_ids.add(source_id)
            if show_progress:
                print(f"{progress} skipped | {_skip_reason(row.get('skip_reason'))}")
            _notify_progress(
                progress_callback,
                source_id + 1,
                total_rows,
                review_id,
                "skipped",
            )
            _notify_runtime_event(
                runtime_event_callback,
                {
                    "source_row_id": source_id,
                    "review_id": review_id,
                    "status": "skipped",
                    "latency_seconds": 0.0,
                    "retry_count": 0,
                    "error_type": None,
                },
            )
            continue

        _notify_progress(
            progress_callback,
            source_id + 1,
            total_rows,
            review_id,
            "analyzing",
        )
        started = clock()
        row_retry_count = 0

        def observe_retry() -> None:
            nonlocal row_retry_count, retry_count
            row_retry_count += 1
            retry_count += 1

        try:
            if active_client is None:
                active_client = create_review_client()
            analysis = _analyze_with_transient_provider_retry(
                str(row["cleaned_content"]),
                client=active_client,
                sleeper=sleeper,
                retry_delay_seconds=provider_retry_delay_seconds,
                retry_observer=observe_retry,
            )
        except KeyboardInterrupt:
            interrupted = True
            current_checkpoint = save_current_checkpoint()
            if show_progress:
                print("Batch interrupted by user.")
                print("Checkpoint saved.")
            break
        except Exception as exc:
            elapsed = max(0.0, clock() - started)
            llm_time_seconds += elapsed
            postprocess_started = clock()
            records[source_id] = _failed_record(row, exc, elapsed)
            postprocessing_time_seconds += max(0.0, clock() - postprocess_started)
            newly_processed_ids.add(source_id)
            attempted_reviews += 1
            model_failures += 1
            if isinstance(exc, _PROVIDER_AVAILABILITY_ERRORS):
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if show_progress:
                print(f"{progress} failed | {elapsed:.2f}s")
            _notify_progress(
                progress_callback,
                source_id + 1,
                total_rows,
                review_id,
                "failed",
            )
            _notify_runtime_event(
                runtime_event_callback,
                {
                    "source_row_id": source_id,
                    "review_id": review_id,
                    "status": "failed",
                    "latency_seconds": elapsed,
                    "retry_count": row_retry_count,
                    "error_type": type(exc).__name__,
                },
            )
        else:
            elapsed = max(0.0, clock() - started)
            llm_time_seconds += elapsed
            postprocess_started = clock()
            records[source_id] = _success_record(row, analysis, elapsed)
            postprocessing_time_seconds += max(0.0, clock() - postprocess_started)
            newly_processed_ids.add(source_id)
            attempted_reviews += 1
            consecutive_failures = 0
            if show_progress:
                print(f"{progress} success | {elapsed:.2f}s")
            _notify_progress(
                progress_callback,
                source_id + 1,
                total_rows,
                review_id,
                "success",
            )
            _notify_runtime_event(
                runtime_event_callback,
                {
                    "source_row_id": source_id,
                    "review_id": review_id,
                    "status": "success",
                    "latency_seconds": elapsed,
                    "retry_count": row_retry_count,
                    "error_type": None,
                },
            )

        if attempted_reviews % checkpoint_interval == 0:
            current_checkpoint = save_current_checkpoint()

        if consecutive_failures >= max_consecutive_failures:
            stopped_for_failures = True
            current_checkpoint = save_current_checkpoint()
            if show_progress:
                print(
                    "Batch stopped after "
                    f"{max_consecutive_failures} consecutive LLM failures."
                )
                print("Checkpoint saved.")
                print("Run with --resume after Ollama is available again.")
            break

    if not interrupted and not stopped_for_failures:
        current_checkpoint = save_current_checkpoint()

    checkpoint_ids = set(current_checkpoint["source_row_id"].astype(int))
    is_complete = (
        not interrupted
        and not stopped_for_failures
        and checkpoint_ids == set(all_source_ids)
    )
    written_final: Path | None = None
    if is_complete:
        current_checkpoint = current_checkpoint.sort_values(
            "source_row_id",
            kind="stable",
        ).reset_index(drop=True)
        written_final = save_analyzed_reviews(current_checkpoint, final_output)

    return BatchRunResult(
        reviews=current_checkpoint,
        total_input_rows=total_rows,
        previously_completed=previously_completed,
        newly_processed=len(newly_processed_ids),
        attempted_reviews=attempted_reviews,
        is_complete=is_complete,
        interrupted=interrupted,
        stopped_for_failures=stopped_for_failures,
        checkpoint_path=checkpoint,
        final_output_path=written_final,
        retry_count=retry_count,
        model_failures=model_failures,
        llm_time_seconds=llm_time_seconds,
        postprocessing_time_seconds=postprocessing_time_seconds,
        checkpoint_time_seconds=checkpoint_time_seconds,
    )
