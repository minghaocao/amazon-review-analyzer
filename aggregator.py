"""Step 6 商品级 Aspect、情绪和 Evidence 聚合。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

from aspect_taxonomy import business_category_for_aspect
from business_intelligence import build_business_intelligence, enrich_aspect_summary
from config import (
    ANALYZED_REVIEWS_PATH,
    ASPECT_EVIDENCE_PATH,
    ASPECT_SUMMARY_PATH,
    BUSINESS_SUMMARY_PATH,
    MIN_ASPECT_REVIEWS_FOR_RANKING,
)


BASE_SENTIMENTS = ("positive", "negative", "neutral")
AGGREGATED_SENTIMENTS = (*BASE_SENTIMENTS, "mixed")
REVIEW_ASPECT_COLUMNS = (
    "source_row_id",
    "review_id",
    "aspect",
    "business_category",
    "review_sentiment",
    "mention_count",
    "include_in_business_metrics",
)
ASPECT_SUMMARY_COLUMNS = (
    "aspect",
    "business_category",
    "aspect_review_count",
    "mention_count",
    "aspect_share_of_reviews",
    "positive_review_count",
    "negative_review_count",
    "neutral_review_count",
    "mixed_review_count",
    "positive_rate",
    "negative_rate",
    "neutral_rate",
    "mixed_rate",
    "include_in_business_metrics",
    "review_share",
    "avg_rating",
    "opportunity_score",
    "evidence_level",
)
REQUIRED_ANALYZED_COLUMNS = (
    "source_row_id",
    "review_id",
    "analysis_status",
    "aspects",
    "unique_aspects",
)
RANKING_RECORD_FIELDS = (
    "aspect",
    "business_category",
    "aspect_review_count",
    "mention_count",
    "positive_review_count",
    "negative_review_count",
    "neutral_review_count",
    "mixed_review_count",
    "positive_rate",
    "negative_rate",
    "neutral_rate",
    "mixed_rate",
    "aspect_share_of_reviews",
)


class AggregationError(ValueError):
    """The analyzed-review input cannot be aggregated safely."""


@dataclass(frozen=True)
class AggregationResult:
    """All derived Step 6 outputs before they are persisted."""

    review_aspect_table: pd.DataFrame
    aspect_summary: pd.DataFrame
    evidence: dict[str, dict[str, list[dict[str, Any]]]]
    business_summary: dict[str, Any]


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _python_scalar(value: object) -> object:
    """Convert pandas/numpy-style scalar values into JSON-safe Python values."""

    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    return value


def _source_row_id(value: object) -> int | str | None:
    value = _python_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return int(value)
    text = str(value).strip()
    try:
        numeric = int(text)
    except ValueError:
        return text
    return numeric if str(numeric) == text else text


def _review_id(value: object) -> str | None:
    value = _python_scalar(value)
    return None if value is None else str(value)


def _issue(
    row: pd.Series,
    column: str,
    error: str,
) -> dict[str, Any]:
    return {
        "source_row_id": _source_row_id(row.get("source_row_id")),
        "review_id": _review_id(row.get("review_id")),
        "column": column,
        "error": error,
    }


def _append_issue(
    issues: list[dict[str, Any]],
    row: pd.Series,
    column: str,
    error: str,
) -> None:
    candidate = _issue(row, column, error)
    if candidate not in issues:
        issues.append(candidate)


def _parse_json_list(
    value: object,
    *,
    row: pd.Series,
    column: str,
    issues: list[dict[str, Any]],
) -> list[Any] | None:
    """Decode one JSON-list field and record an unusable payload by row."""

    if isinstance(value, list):
        return value
    if _is_missing(value) or not isinstance(value, str) or not value.strip():
        _append_issue(issues, row, column, "missing JSON array")
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        _append_issue(
            issues,
            row,
            column,
            f"invalid JSON: {exc.msg}",
        )
        return None
    if not isinstance(decoded, list):
        _append_issue(issues, row, column, "JSON value must be an array")
        return None
    return decoded


def _positive_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        return None
    return number if number > 0 else None


def _business_flag(value: object, aspect: str) -> bool:
    """Read the persisted flag, while always excluding Other as a fail-safe."""

    if aspect.casefold() == "other":
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    # Backward compatibility for older analyzed files that predate this flag.
    return True


def _validate_analyzed_reviews(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = [column for column in REQUIRED_ANALYZED_COLUMNS if column not in df.columns]
    if missing:
        raise AggregationError(
            f"Missing analyzed review columns: {', '.join(missing)}"
        )
    source_ids = df["source_row_id"]
    if source_ids.map(_is_missing).any() or source_ids.astype(str).str.strip().eq("").any():
        raise AggregationError("source_row_id contains missing values")
    if source_ids.astype(str).duplicated().any():
        raise AggregationError("source_row_id must be unique")
    statuses = set(df["analysis_status"].astype(str))
    invalid_statuses = sorted(statuses.difference({"success", "failed", "skipped"}))
    if invalid_statuses:
        raise AggregationError(
            "analysis_status contains unsupported values: "
            + ", ".join(invalid_statuses)
        )


def load_analyzed_reviews(
    file_path: str | Path = ANALYZED_REVIEWS_PATH,
) -> pd.DataFrame:
    """Load the completed batch output without consulting a checkpoint."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Analyzed reviews file not found: {path}")
    try:
        frame = pd.read_csv(
            path,
            dtype="string",
            keep_default_na=False,
            encoding="utf-8",
        )
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise AggregationError(f"Analyzed reviews CSV could not be read: {exc}") from exc
    _validate_analyzed_reviews(frame)
    return frame


def build_review_aspect_table(
    df: pd.DataFrame,
    *,
    issues: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build one derived row per ``source_row_id + normalized aspect``.

    Customer counts come from ``unique_aspects``. Different base sentiments
    for the same review and aspect are represented as the derived ``mixed``
    state, without modifying the original analysis fields.
    """

    _validate_analyzed_reviews(df)
    issue_list = issues if issues is not None else []
    grouped: dict[tuple[int | str | None, str], dict[str, Any]] = {}
    successful = df.loc[df["analysis_status"].astype(str).eq("success")]

    for _, row in successful.iterrows():
        entries = _parse_json_list(
            row["unique_aspects"],
            row=row,
            column="unique_aspects",
            issues=issue_list,
        )
        if entries is None:
            continue

        for position, entry in enumerate(entries):
            if not isinstance(entry, dict):
                _append_issue(
                    issue_list,
                    row,
                    "unique_aspects",
                    f"item {position} must be an object",
                )
                continue
            aspect_value = entry.get("aspect")
            sentiment_value = entry.get("sentiment")
            aspect = aspect_value.strip() if isinstance(aspect_value, str) else ""
            sentiment = (
                sentiment_value.strip().casefold()
                if isinstance(sentiment_value, str)
                else ""
            )
            mention_count = _positive_integer(entry.get("mention_count"))
            if not aspect:
                _append_issue(
                    issue_list,
                    row,
                    "unique_aspects",
                    f"item {position} has no aspect",
                )
                continue
            if sentiment not in BASE_SENTIMENTS:
                _append_issue(
                    issue_list,
                    row,
                    "unique_aspects",
                    f"item {position} has unsupported sentiment",
                )
                continue
            if mention_count is None:
                _append_issue(
                    issue_list,
                    row,
                    "unique_aspects",
                    f"item {position} has invalid mention_count",
                )
                continue

            source_id = _source_row_id(row["source_row_id"])
            key = (source_id, aspect)
            include = _business_flag(
                entry.get("include_in_business_metrics"),
                aspect,
            )
            if key not in grouped:
                grouped[key] = {
                    "source_row_id": source_id,
                    "review_id": _review_id(row["review_id"]),
                    "aspect": aspect,
                    "business_category": business_category_for_aspect(aspect),
                    "sentiments": set(),
                    "mention_count": 0,
                    "include_flags": [],
                }
            grouped[key]["sentiments"].add(sentiment)
            grouped[key]["mention_count"] += mention_count
            grouped[key]["include_flags"].append(include)

    records: list[dict[str, Any]] = []
    for item in grouped.values():
        sentiments = item.pop("sentiments")
        include_flags = item.pop("include_flags")
        records.append(
            {
                **item,
                "review_sentiment": (
                    next(iter(sentiments)) if len(sentiments) == 1 else "mixed"
                ),
                "include_in_business_metrics": all(include_flags),
            }
        )

    table = pd.DataFrame.from_records(records, columns=REVIEW_ASPECT_COLUMNS)
    table.attrs["invalid_json_details"] = list(issue_list)
    return table


def build_aspect_summary(
    review_aspect_table: pd.DataFrame,
    successful_reviews: int,
) -> pd.DataFrame:
    """Aggregate review-level sentiments and mention-level counts by aspect."""

    if isinstance(successful_reviews, bool) or successful_reviews < 0:
        raise ValueError("successful_reviews must be a non-negative integer")
    missing = [
        column
        for column in REVIEW_ASPECT_COLUMNS
        if column not in review_aspect_table.columns
    ]
    if missing:
        raise AggregationError(
            f"Missing review-aspect columns: {', '.join(missing)}"
        )
    if review_aspect_table.empty:
        return pd.DataFrame(columns=ASPECT_SUMMARY_COLUMNS)

    records: list[dict[str, Any]] = []
    for aspect, group in review_aspect_table.groupby("aspect", sort=False):
        aspect_review_count = int(group["source_row_id"].nunique())
        mention_count = int(pd.to_numeric(group["mention_count"]).sum())
        sentiment_counts = {
            sentiment: int(group["review_sentiment"].eq(sentiment).sum())
            for sentiment in AGGREGATED_SENTIMENTS
        }
        if sum(sentiment_counts.values()) != aspect_review_count:
            raise AggregationError(
                f"Sentiment counts do not reconcile for aspect: {aspect}"
            )

        record: dict[str, Any] = {
            "aspect": str(aspect),
            "business_category": business_category_for_aspect(str(aspect)),
            "aspect_review_count": aspect_review_count,
            "mention_count": mention_count,
            "aspect_share_of_reviews": (
                aspect_review_count / successful_reviews * 100
                if successful_reviews
                else 0.0
            ),
        }
        for sentiment in AGGREGATED_SENTIMENTS:
            count = sentiment_counts[sentiment]
            record[f"{sentiment}_review_count"] = count
            record[f"{sentiment}_rate"] = (
                count / aspect_review_count * 100 if aspect_review_count else 0.0
            )
        record["include_in_business_metrics"] = bool(
            group["include_in_business_metrics"].map(bool).all()
        )
        records.append(record)

    summary = pd.DataFrame.from_records(records, columns=ASPECT_SUMMARY_COLUMNS)
    summary = enrich_aspect_summary(summary)
    return summary.sort_values(
        ["aspect_review_count", "mention_count", "aspect"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def collect_aspect_evidence(
    df: pd.DataFrame,
    *,
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Collect validated verbatim Evidence from original aspect mentions."""

    _validate_analyzed_reviews(df)
    issue_list = issues if issues is not None else []
    evidence: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen: set[tuple[int | str | None, str, str, str]] = set()
    successful = df.loc[df["analysis_status"].astype(str).eq("success")]

    for _, row in successful.iterrows():
        entries = _parse_json_list(
            row["aspects"],
            row=row,
            column="aspects",
            issues=issue_list,
        )
        if entries is None:
            continue
        for position, entry in enumerate(entries):
            if not isinstance(entry, dict):
                _append_issue(
                    issue_list,
                    row,
                    "aspects",
                    f"item {position} must be an object",
                )
                continue
            aspect_value = entry.get("aspect")
            sentiment_value = entry.get("sentiment")
            evidence_value = entry.get("evidence")
            aspect = aspect_value.strip() if isinstance(aspect_value, str) else ""
            sentiment = (
                sentiment_value.strip().casefold()
                if isinstance(sentiment_value, str)
                else ""
            )
            if not aspect or sentiment not in BASE_SENTIMENTS:
                _append_issue(
                    issue_list,
                    row,
                    "aspects",
                    f"item {position} has invalid aspect or sentiment",
                )
                continue
            if not isinstance(evidence_value, str) or not evidence_value:
                _append_issue(
                    issue_list,
                    row,
                    "aspects",
                    f"item {position} has no evidence",
                )
                continue

            source_id = _source_row_id(row["source_row_id"])
            deduplication_key = (source_id, aspect, sentiment, evidence_value)
            if deduplication_key in seen:
                continue
            seen.add(deduplication_key)
            if aspect not in evidence:
                evidence[aspect] = {
                    sentiment_name: [] for sentiment_name in BASE_SENTIMENTS
                }
            evidence[aspect][sentiment].append(
                {
                    "source_row_id": source_id,
                    "review_id": _review_id(row["review_id"]),
                    "evidence": evidence_value,
                }
            )
    return {aspect: evidence[aspect] for aspect in sorted(evidence)}


def _ranking_records(
    summary: pd.DataFrame,
    *,
    ranking: str,
    minimum_reviews: int,
    top_n: int,
) -> list[dict[str, Any]]:
    eligible = summary.loc[summary["include_in_business_metrics"].map(bool)].copy()
    if ranking == "concern":
        eligible = eligible.loc[
            eligible["business_category"].isin({"PRODUCT", "EXPERIENCE"})
        ].sort_values(
            ["aspect_review_count", "mention_count", "aspect"],
            ascending=[False, False, True],
            kind="stable",
        )
    elif ranking in {"selling", "product_selling"}:
        eligible = eligible.loc[
            eligible["business_category"].eq("PRODUCT")
            & eligible["aspect_review_count"].ge(minimum_reviews)
            & eligible["positive_review_count"].gt(
                eligible["negative_review_count"]
            )
        ].sort_values(
            ["positive_review_count", "positive_rate", "aspect_review_count", "aspect"],
            ascending=[False, False, False, True],
            kind="stable",
        )
    elif ranking in {"pain", "product_pain"}:
        eligible = eligible.loc[
            eligible["business_category"].eq("PRODUCT")
            & eligible["aspect_review_count"].ge(minimum_reviews)
            & eligible["negative_review_count"].gt(
                eligible["positive_review_count"]
            )
        ].sort_values(
            ["negative_review_count", "negative_rate", "aspect_review_count", "aspect"],
            ascending=[False, False, False, True],
            kind="stable",
        )
    elif ranking == "experience_positive":
        eligible = eligible.loc[
            eligible["business_category"].eq("EXPERIENCE")
            & eligible["aspect_review_count"].ge(minimum_reviews)
            & eligible["positive_review_count"].gt(
                eligible["negative_review_count"]
            )
        ].sort_values(
            ["positive_review_count", "positive_rate", "aspect_review_count", "aspect"],
            ascending=[False, False, False, True],
            kind="stable",
        )
    elif ranking == "experience_negative":
        eligible = eligible.loc[
            eligible["business_category"].eq("EXPERIENCE")
            & eligible["aspect_review_count"].ge(minimum_reviews)
            & eligible["negative_review_count"].gt(
                eligible["positive_review_count"]
            )
        ].sort_values(
            ["negative_review_count", "negative_rate", "aspect_review_count", "aspect"],
            ascending=[False, False, False, True],
            kind="stable",
        )
    else:
        raise ValueError(f"Unsupported ranking: {ranking}")

    records: list[dict[str, Any]] = []
    selected = eligible.head(top_n).loc[:, list(RANKING_RECORD_FIELDS)]
    for row in selected.to_dict("records"):
        records.append({key: _python_scalar(value) for key, value in row.items()})
    return records


def build_business_summary(
    df: pd.DataFrame,
    aspect_summary: pd.DataFrame,
    issues: list[dict[str, Any]],
    *,
    min_aspect_reviews: int,
    top_n: int,
) -> dict[str, Any]:
    """Build transparent commercial rankings and data-quality metadata."""

    statuses = df["analysis_status"].astype(str)
    total_rows = int(len(df))
    successful_reviews = int(statuses.eq("success").sum())
    failed_reviews = int(statuses.eq("failed").sum())
    skipped_reviews = int(statuses.eq("skipped").sum())
    unique_issue_rows = {
        str(issue.get("source_row_id")) for issue in issues
    }
    data_quality = {
        "total_input_rows": total_rows,
        "successful_reviews": successful_reviews,
        "failed_reviews": failed_reviews,
        "skipped_reviews": skipped_reviews,
        "analysis_coverage": (
            successful_reviews / total_rows if total_rows else 0.0
        ),
        "invalid_json_rows": len(unique_issue_rows),
        "invalid_json_details": issues,
    }
    customer_concerns = _ranking_records(
        aspect_summary,
        ranking="concern",
        minimum_reviews=min_aspect_reviews,
        top_n=top_n,
    )
    product_selling_points = _ranking_records(
        aspect_summary,
        ranking="product_selling",
        minimum_reviews=min_aspect_reviews,
        top_n=top_n,
    )
    product_pain_points = _ranking_records(
        aspect_summary,
        ranking="product_pain",
        minimum_reviews=min_aspect_reviews,
        top_n=top_n,
    )
    experience_positive = _ranking_records(
        aspect_summary,
        ranking="experience_positive",
        minimum_reviews=min_aspect_reviews,
        top_n=top_n,
    )
    experience_negative = _ranking_records(
        aspect_summary,
        ranking="experience_negative",
        minimum_reviews=min_aspect_reviews,
        top_n=top_n,
    )
    return {
        "data_quality": data_quality,
        "ranking_rules": {
            "minimum_aspect_reviews": min_aspect_reviews,
            "top_n": top_n,
            "customer_concerns_order": "aspect_review_count descending",
            "product_selling_eligibility": (
                "PRODUCT with positive_review_count greater than "
                "negative_review_count"
            ),
            "product_selling_order": "positive_review_count descending",
            "product_pain_eligibility": (
                "PRODUCT with negative_review_count greater than "
                "positive_review_count"
            ),
            "product_pain_order": "negative_review_count descending",
            "shopping_experience_eligibility": (
                "EXPERIENCE with the directional review count greater than "
                "the opposite review count"
            ),
            "minimum_sample_applies_to": (
                "product and shopping-experience directional rankings"
            ),
        },
        "top_customer_concerns": customer_concerns,
        "product_selling_points": product_selling_points,
        "product_pain_points": product_pain_points,
        "shopping_experience": {
            "positive": experience_positive,
            "negative": experience_negative,
        },
        # Temporary aliases for Step 6 consumers; both now mean product-only.
        "top_selling_points": product_selling_points,
        "top_pain_points": product_pain_points,
    }


def aggregate_reviews(
    df: pd.DataFrame,
    *,
    min_aspect_reviews: int = MIN_ASPECT_REVIEWS_FOR_RANKING,
    top_n: int = 5,
) -> AggregationResult:
    """Create every Step 6 result without mutating input or calling an LLM."""

    _validate_analyzed_reviews(df)
    for value, name in (
        (min_aspect_reviews, "min_aspect_reviews"),
        (top_n, "top_n"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    issues: list[dict[str, Any]] = []
    review_aspects = build_review_aspect_table(df, issues=issues)
    evidence = collect_aspect_evidence(df, issues=issues)
    successful_reviews = int(df["analysis_status"].astype(str).eq("success").sum())
    aspect_summary = build_aspect_summary(review_aspects, successful_reviews)
    intelligence = build_business_intelligence(
        df,
        aspect_summary,
        minimum_support=min_aspect_reviews,
        top_n=top_n,
    )
    aspect_summary = enrich_aspect_summary(
        aspect_summary,
        df,
        rating_insights=intelligence.get("rating_insights"),
    )
    business_summary = build_business_summary(
        df,
        aspect_summary,
        issues,
        min_aspect_reviews=min_aspect_reviews,
        top_n=top_n,
    )
    business_summary.update(intelligence)
    return AggregationResult(
        review_aspect_table=review_aspects,
        aspect_summary=aspect_summary,
        evidence=evidence,
        business_summary=business_summary,
    )


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.tmp")


def _write_text_temp(destination: Path, text: str) -> Path:
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


def _write_csv_temp(destination: Path, frame: pd.DataFrame) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def save_aggregation_outputs(
    result: AggregationResult,
    *,
    aspect_summary_path: str | Path = ASPECT_SUMMARY_PATH,
    evidence_path: str | Path = ASPECT_EVIDENCE_PATH,
    business_summary_path: str | Path = BUSINESS_SUMMARY_PATH,
) -> dict[str, Path]:
    """Atomically publish the three completed aggregation artifacts."""

    destinations = {
        "aspect_summary": Path(aspect_summary_path),
        "aspect_evidence": Path(evidence_path),
        "business_summary": Path(business_summary_path),
    }
    temporary_files: list[tuple[Path, Path]] = []
    try:
        temporary_files.append(
            (
                _write_csv_temp(
                    destinations["aspect_summary"],
                    result.aspect_summary,
                ),
                destinations["aspect_summary"],
            )
        )
        temporary_files.append(
            (
                _write_text_temp(
                    destinations["aspect_evidence"],
                    json.dumps(result.evidence, ensure_ascii=False, indent=2) + "\n",
                ),
                destinations["aspect_evidence"],
            )
        )
        temporary_files.append(
            (
                _write_text_temp(
                    destinations["business_summary"],
                    json.dumps(
                        result.business_summary,
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                ),
                destinations["business_summary"],
            )
        )
    except BaseException:
        for temporary, _ in temporary_files:
            temporary.unlink(missing_ok=True)
        raise

    try:
        for temporary, destination in temporary_files:
            os.replace(temporary, destination)
    except BaseException:
        for temporary, _ in temporary_files:
            temporary.unlink(missing_ok=True)
        raise
    return destinations
