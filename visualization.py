"""Read-only data helpers for the Step 9 Streamlit interview dashboard.

This module deliberately has no Streamlit, Ollama, or OpenAI dependency. It
loads previously generated outputs, safely parses row-level JSON, and exposes
small deterministic helpers that the presentation layer in ``app.py`` uses.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from business_intelligence import (
    DEFAULT_EVIDENCE_DISPLAY_LIMIT,
    EXPANDED_EVIDENCE_DISPLAY_LIMIT,
    build_business_intelligence,
    enrich_aspect_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUT_FILES = {
    "analyzed_reviews": "analyzed_reviews.csv",
    "aspect_summary": "aspect_summary.csv",
    "business_summary": "business_summary.json",
    "aspect_evidence": "aspect_evidence.json",
    "keyword_summary": "keyword_summary.csv",
    "sentiment_keywords": "sentiment_keywords.json",
    "aspect_keywords": "aspect_keywords.json",
}
OUTPUT_COMMANDS = {
    "analyzed_reviews": "python main.py --batch --resume",
    "aspect_summary": "python main.py --aggregate",
    "business_summary": "python main.py --aggregate",
    "aspect_evidence": "python main.py --aggregate",
    "keyword_summary": "python main.py --keywords",
    "sentiment_keywords": "python main.py --keywords",
    "aspect_keywords": "python main.py --keywords",
}
CSV_REQUIRED_COLUMNS = {
    "analyzed_reviews": {
        "source_row_id",
        "review_id",
        "analysis_status",
        "cleaned_content",
        "aspects",
    },
    "aspect_summary": {
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
    },
    "keyword_summary": {
        "term",
        "term_type",
        "count",
        "review_count",
        "review_share",
    },
}
REVIEW_JSON_COLUMNS = (
    "aspects",
    "unique_aspects",
    "positive_points",
    "negative_points",
)
SENTIMENTS = ("positive", "negative", "neutral", "mixed")


def _copy_nested_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy nested lists/dicts stored in object columns after a DataFrame copy."""

    result = df.copy(deep=True)
    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column].dtype):
            result[column] = result[column].map(copy.deepcopy)
    return result


@dataclass(frozen=True)
class OutputIssue:
    """One missing or unreadable dashboard source."""

    output_name: str
    path: Path
    message: str
    command: str


@dataclass(frozen=True)
class DashboardData:
    """All available read-only dashboard inputs."""

    analyzed_reviews: pd.DataFrame | None
    aspect_summary: pd.DataFrame | None
    business_summary: dict[str, Any] | None
    aspect_evidence: dict[str, Any] | None
    keyword_summary: pd.DataFrame | None
    sentiment_keywords: dict[str, Any] | None
    aspect_keywords: dict[str, Any] | None
    issues: tuple[OutputIssue, ...]
    row_json_warnings: tuple[dict[str, Any], ...]
    business_intelligence: dict[str, Any] | None = None
    enriched_aspect_summary: pd.DataFrame | None = None

    def issue_for(self, output_name: str) -> OutputIssue | None:
        return next(
            (issue for issue in self.issues if issue.output_name == output_name),
            None,
        )


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_text(value: object) -> str:
    if value is None or _is_missing(value):
        return ""
    return str(value)


def safe_parse_json_list(value: object) -> tuple[list[Any], str | None]:
    """Decode a JSON list without propagating row-level data errors."""

    if isinstance(value, list):
        return copy.deepcopy(value), None
    if value is None or _is_missing(value):
        return [], None
    if not isinstance(value, str):
        return [], "value is not a JSON array"
    if not value.strip():
        return [], None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        return [], f"invalid JSON: {exc.msg}"
    if not isinstance(decoded, list):
        return [], "JSON value must be an array"
    return copy.deepcopy(decoded), None


def prepare_review_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return a copy with safe parsed JSON columns and warning metadata."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    prepared = df.copy(deep=True)
    warnings: list[dict[str, Any]] = []

    for column in REVIEW_JSON_COLUMNS:
        parsed_values: list[list[Any]] = []
        raw_values: Iterable[object]
        if column in prepared.columns:
            raw_values = prepared[column]
        else:
            raw_values = [None] * len(prepared)
        for row_position, value in enumerate(raw_values):
            parsed, error = safe_parse_json_list(value)
            parsed_values.append(parsed)
            if error:
                row = prepared.iloc[row_position]
                warnings.append(
                    {
                        "source_row_id": row.get("source_row_id"),
                        "review_id": row.get("review_id"),
                        "column": column,
                        "error": error,
                    }
                )
        prepared[f"parsed_{column}"] = parsed_values

    return prepared, warnings


def _read_csv_output(
    output_name: str,
    path: Path,
) -> tuple[pd.DataFrame | None, OutputIssue | None]:
    if not path.is_file():
        return None, OutputIssue(
            output_name,
            path,
            "Analysis output not found.",
            OUTPUT_COMMANDS[output_name],
        )
    try:
        dtype = (
            {"source_row_id": "string", "review_id": "string"}
            if output_name == "analyzed_reviews"
            else None
        )
        frame = pd.read_csv(
            path,
            keep_default_na=False,
            encoding="utf-8",
            dtype=dtype,
        )
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        return None, OutputIssue(
            output_name,
            path,
            f"Analysis output could not be read: {exc}",
            OUTPUT_COMMANDS[output_name],
        )
    missing = sorted(CSV_REQUIRED_COLUMNS[output_name].difference(frame.columns))
    if missing:
        return None, OutputIssue(
            output_name,
            path,
            f"Analysis output is missing columns: {', '.join(missing)}",
            OUTPUT_COMMANDS[output_name],
        )
    return frame, None


def _read_json_output(
    output_name: str,
    path: Path,
) -> tuple[dict[str, Any] | None, OutputIssue | None]:
    if not path.is_file():
        return None, OutputIssue(
            output_name,
            path,
            "Analysis output not found.",
            OUTPUT_COMMANDS[output_name],
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, OutputIssue(
            output_name,
            path,
            f"Analysis output could not be read: {exc}",
            OUTPUT_COMMANDS[output_name],
        )
    if not isinstance(value, dict):
        return None, OutputIssue(
            output_name,
            path,
            "Analysis output must contain a JSON object.",
            OUTPUT_COMMANDS[output_name],
        )
    structure_error = _json_structure_error(output_name, value)
    if structure_error:
        return None, OutputIssue(
            output_name,
            path,
            f"Analysis output has an invalid structure: {structure_error}",
            OUTPUT_COMMANDS[output_name],
        )
    return value, None


def _is_record_list(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(record, Mapping) for record in value
    )


def _json_structure_error(
    output_name: str,
    value: Mapping[str, Any],
) -> str | None:
    """Return a concise schema error for a syntactically valid JSON output."""

    if output_name == "business_summary":
        for field in ("data_quality", "ranking_rules"):
            if not isinstance(value.get(field), Mapping):
                return f"{field} must be an object"
        for field in (
            "top_customer_concerns",
            "product_selling_points",
            "product_pain_points",
        ):
            if not _is_record_list(value.get(field)):
                return f"{field} must be an array of objects"
        shopping = value.get("shopping_experience")
        if not isinstance(shopping, Mapping):
            return "shopping_experience must be an object"
        for direction in ("positive", "negative"):
            if not _is_record_list(shopping.get(direction)):
                return f"shopping_experience.{direction} must be an array of objects"
    elif output_name == "aspect_evidence":
        for aspect, buckets in value.items():
            if not isinstance(buckets, Mapping):
                return f"{aspect} must contain sentiment arrays"
            for sentiment, records in buckets.items():
                if sentiment not in SENTIMENTS or not _is_record_list(records):
                    return f"{aspect}.{sentiment} must be an Evidence array"
    elif output_name == "sentiment_keywords":
        for sentiment in ("positive", "negative"):
            if not _is_record_list(value.get(sentiment)):
                return f"{sentiment} must be an array of keyword objects"
    elif output_name == "aspect_keywords":
        for aspect, records in value.items():
            if not _is_record_list(records):
                return f"{aspect} must be an array of keyword objects"
    return None


def load_dashboard_data(
    outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR,
) -> DashboardData:
    """Load every available output while converting failures into issues."""

    root = Path(outputs_dir)
    issues: list[OutputIssue] = []
    loaded: dict[str, Any] = {}
    for output_name, filename in OUTPUT_FILES.items():
        path = root / filename
        if path.suffix == ".csv":
            value, issue = _read_csv_output(output_name, path)
        else:
            value, issue = _read_json_output(output_name, path)
        loaded[output_name] = value
        if issue:
            issues.append(issue)

    row_warnings: list[dict[str, Any]] = []
    analyzed = loaded["analyzed_reviews"]
    if analyzed is not None:
        analyzed, row_warnings = prepare_review_data(analyzed)

    persisted_business = loaded["business_summary"]
    minimum_support: int | None = None
    generated_at: str | None = None
    if isinstance(persisted_business, Mapping):
        ranking_rules = persisted_business.get("ranking_rules")
        if isinstance(ranking_rules, Mapping):
            try:
                candidate = int(ranking_rules.get("minimum_aspect_reviews", 0))
            except (TypeError, ValueError):
                candidate = 0
            if candidate > 0:
                minimum_support = candidate
        persisted_generated = persisted_business.get("generated_at")
        if isinstance(persisted_generated, str) and persisted_generated:
            generated_at = persisted_generated
    intelligence = (
        build_business_intelligence(
            analyzed,
            loaded["aspect_summary"],
            **(
                {"minimum_support": minimum_support}
                if minimum_support is not None
                else {}
            ),
            generated_at=generated_at,
        )
        if analyzed is not None and loaded["aspect_summary"] is not None
        else None
    )
    enriched_summary = (
        enrich_aspect_summary(
            loaded["aspect_summary"],
            analyzed,
            rating_insights=(
                intelligence.get("rating_insights")
                if isinstance(intelligence, Mapping)
                else None
            ),
        )
        if loaded["aspect_summary"] is not None
        else None
    )

    return DashboardData(
        analyzed_reviews=analyzed,
        aspect_summary=loaded["aspect_summary"],
        business_summary=loaded["business_summary"],
        aspect_evidence=loaded["aspect_evidence"],
        keyword_summary=loaded["keyword_summary"],
        sentiment_keywords=loaded["sentiment_keywords"],
        aspect_keywords=loaded["aspect_keywords"],
        issues=tuple(issues),
        row_json_warnings=tuple(row_warnings),
        business_intelligence=intelligence,
        enriched_aspect_summary=enriched_summary,
    )


def output_fingerprint(
    outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR,
) -> tuple[tuple[str, int, int], ...]:
    """Return a cache key that changes when a dashboard source changes."""

    root = Path(outputs_dir)
    values: list[tuple[str, int, int]] = []
    for filename in OUTPUT_FILES.values():
        path = root / filename
        try:
            stat = path.stat()
            values.append((filename, stat.st_mtime_ns, stat.st_size))
        except OSError:
            values.append((filename, -1, -1))
    return tuple(values)


def count_validated_evidence(aspect_evidence: Mapping[str, Any] | None) -> int:
    """Count the stored verbatim Evidence records without reinterpretation."""

    if not isinstance(aspect_evidence, Mapping):
        return 0
    total = 0
    for sentiment_buckets in aspect_evidence.values():
        if not isinstance(sentiment_buckets, Mapping):
            continue
        for sentiment in SENTIMENTS:
            entries = sentiment_buckets.get(sentiment, [])
            if isinstance(entries, list):
                total += sum(isinstance(entry, dict) for entry in entries)
    return total


def calculate_kpis(data: DashboardData) -> dict[str, int | float]:
    """Derive dashboard KPIs from persisted output values only."""

    frame = data.analyzed_reviews
    quality_value = (
        data.business_summary.get("data_quality", {})
        if isinstance(data.business_summary, Mapping)
        else {}
    )
    quality = quality_value if isinstance(quality_value, Mapping) else {}
    total = int(quality.get("total_input_rows", len(frame) if frame is not None else 0))
    if frame is not None and "analysis_status" in frame.columns:
        statuses = frame["analysis_status"].astype(str).str.casefold()
        success_fallback = int(statuses.eq("success").sum())
        failed_fallback = int(statuses.eq("failed").sum())
        skipped_fallback = int(statuses.eq("skipped").sum())
    else:
        success_fallback = failed_fallback = skipped_fallback = 0
    successful = int(quality.get("successful_reviews", success_fallback))
    failed = int(quality.get("failed_reviews", failed_fallback))
    skipped = int(quality.get("skipped_reviews", skipped_fallback))
    coverage = quality.get(
        "analysis_coverage",
        successful / total if total else 0.0,
    )
    try:
        coverage_value = float(coverage)
    except (TypeError, ValueError):
        coverage_value = successful / total if total else 0.0

    product_aspects = 0
    if data.aspect_summary is not None:
        summary = data.aspect_summary
        categories = summary["business_category"].astype(str).str.upper()
        reviews = pd.to_numeric(summary["aspect_review_count"], errors="coerce").fillna(0)
        product_aspects = int((categories.eq("PRODUCT") & reviews.gt(0)).sum())

    return {
        "total_reviews": total,
        "successful_reviews": successful,
        "failed_reviews": failed,
        "skipped_reviews": skipped,
        "analysis_coverage": coverage_value,
        "product_aspects_identified": product_aspects,
        "validated_evidence_count": count_validated_evidence(data.aspect_evidence),
    }


def category_label(value: object) -> str:
    labels = {
        "PRODUCT": "Product Aspect",
        "EXPERIENCE": "Shopping Experience",
        "OTHER": "Other",
    }
    return labels.get(_as_text(value).upper(), "Other")


def filter_aspect_summary(
    df: pd.DataFrame,
    category: str = "All",
) -> pd.DataFrame:
    """Return a sorted concern view without recalculating stored metrics."""

    result = _copy_nested_object_columns(df)
    required = {
        "aspect",
        "business_category",
        "aspect_review_count",
        "mention_count",
        "aspect_share_of_reviews",
    }
    if result.empty or not required.issubset(result.columns):
        return pd.DataFrame(columns=[*sorted(required), "category_label"])
    categories = result["business_category"].astype(str).str.upper()
    if category == "Product":
        result = result.loc[categories.eq("PRODUCT")]
    elif category == "Shopping Experience":
        result = result.loc[categories.eq("EXPERIENCE")]
    else:
        # Formal customer-concern views contain product and shopping-experience
        # Aspects; the catch-all Other category remains available in raw data.
        result = result.loc[categories.isin({"PRODUCT", "EXPERIENCE"})]
    result["category_label"] = result["business_category"].map(category_label)
    result["aspect_share_of_reviews"] = pd.to_numeric(
        result["aspect_share_of_reviews"], errors="coerce"
    ).fillna(0.0)
    result["aspect_review_count"] = pd.to_numeric(
        result["aspect_review_count"], errors="coerce"
    ).fillna(0).astype(int)
    result["mention_count"] = pd.to_numeric(
        result["mention_count"], errors="coerce"
    ).fillna(0).astype(int)
    return result.sort_values(
        ["aspect_review_count", "mention_count", "aspect"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _normalized_bool(value: object) -> str:
    normalized = _as_text(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return "Verified"
    if normalized in {"false", "0", "no", "n"}:
        return "Not verified"
    return "Unknown"


def review_filter_options(df: pd.DataFrame) -> dict[str, list[Any]]:
    """Return stable options derived from safely parsed review rows."""

    ratings: list[int | float] = []
    if "rating" in df.columns:
        numeric = pd.to_numeric(df["rating"], errors="coerce").dropna().unique()
        ratings = sorted(
            int(value) if float(value).is_integer() else float(value)
            for value in numeric
        )
    statuses = sorted(df["analysis_status"].astype(str).dropna().unique().tolist())
    aspects: set[str] = set()
    sentiments: set[str] = set()
    for entries in df.get("parsed_aspects", pd.Series([[]] * len(df))):
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("aspect"), str):
                aspects.add(entry["aspect"])
            if isinstance(entry.get("sentiment"), str):
                sentiments.add(entry["sentiment"].casefold())
    return {
        "ratings": ratings,
        "statuses": statuses,
        "aspects": sorted(aspects),
        "sentiments": sorted(sentiments),
    }


def filter_reviews(
    df: pd.DataFrame,
    *,
    ratings: Sequence[int | float] | None = None,
    statuses: Sequence[str] | None = None,
    aspects: Sequence[str] | None = None,
    sentiments: Sequence[str] | None = None,
    verified_purchase: str | None = None,
    query: str = "",
) -> pd.DataFrame:
    """Apply review explorer filters without mutating the caller's DataFrame."""

    result = _copy_nested_object_columns(df)
    mask = pd.Series(True, index=result.index)

    if ratings:
        mask &= pd.to_numeric(result.get("rating"), errors="coerce").isin(ratings)
    if statuses:
        allowed = {str(value) for value in statuses}
        mask &= result["analysis_status"].astype(str).isin(allowed)
    if verified_purchase and verified_purchase != "All":
        verified = result.get(
            "verified_purchase",
            pd.Series([""] * len(result), index=result.index),
        ).map(_normalized_bool)
        mask &= verified.eq(verified_purchase)

    allowed_aspects = {str(value) for value in aspects or []}
    allowed_sentiments = {str(value).casefold() for value in sentiments or []}
    if allowed_aspects or allowed_sentiments:
        parsed_series = result.get(
            "parsed_aspects",
            pd.Series([[]] * len(result), index=result.index),
        )

        def row_matches(entries: object) -> bool:
            if not isinstance(entries, list):
                return False
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                aspect_match = (
                    not allowed_aspects or entry.get("aspect") in allowed_aspects
                )
                sentiment = _as_text(entry.get("sentiment")).casefold()
                sentiment_match = (
                    not allowed_sentiments or sentiment in allowed_sentiments
                )
                if aspect_match and sentiment_match:
                    return True
            return False

        mask &= parsed_series.map(row_matches)

    normalized_query = query.strip().casefold()
    if normalized_query:
        searchable_columns = [
            column
            for column in (
                "review_id",
                "title",
                "original_title",
                "content",
                "original_content",
                "cleaned_content",
                "summary",
            )
            if column in result.columns
        ]
        searchable = pd.Series("", index=result.index)
        for column in searchable_columns:
            searchable = searchable + " " + result[column].astype(str)
        mask &= searchable.str.casefold().str.contains(
            normalized_query,
            regex=False,
            na=False,
        )

    return result.loc[mask].copy(deep=True)


def evidence_for_aspect(
    aspect_evidence: Mapping[str, Any] | None,
    aspect: str,
) -> dict[str, list[dict[str, Any]]]:
    """Return only stored Evidence objects for one existing Aspect."""

    empty = {sentiment: [] for sentiment in SENTIMENTS}
    if not isinstance(aspect_evidence, Mapping):
        return empty
    value = aspect_evidence.get(aspect)
    if not isinstance(value, Mapping):
        return empty
    result: dict[str, list[dict[str, Any]]] = {}
    for sentiment in SENTIMENTS:
        entries = value.get(sentiment, [])
        result[sentiment] = [
            copy.deepcopy(dict(entry))
            for entry in entries
            if isinstance(entry, Mapping)
        ] if isinstance(entries, list) else []
    return result


def evidence_with_review_metadata(
    entries: Sequence[Mapping[str, Any]],
    analyzed_reviews: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Join rating and purchase metadata by source row without changing Evidence."""

    lookup: dict[str, dict[str, Any]] = {}
    if isinstance(analyzed_reviews, pd.DataFrame) and "source_row_id" in analyzed_reviews:
        for _, row in analyzed_reviews.iterrows():
            key = _as_text(row.get("source_row_id"))
            lookup[key] = {
                "rating": _as_text(row.get("rating")),
                "verified_purchase": _normalized_bool(row.get("verified_purchase")),
            }
    result: list[dict[str, Any]] = []
    for entry in entries:
        copied = copy.deepcopy(dict(entry))
        metadata = lookup.get(_as_text(copied.get("source_row_id")), {})
        copied["rating"] = metadata.get("rating", "")
        copied["verified_purchase"] = metadata.get("verified_purchase", "Unknown")
        result.append(copied)
    return result


def aspect_language_by_sentiment(
    aspect_keywords: Mapping[str, Any] | None,
    aspect_evidence: Mapping[str, Any] | None,
    aspect: str,
    *,
    limit: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Group existing Aspect phrases by the sentiment of their review refs."""

    result = {"positive": [], "negative": []}
    if limit <= 0 or not isinstance(aspect_keywords, Mapping):
        return result
    evidence = evidence_for_aspect(aspect_evidence, aspect)
    evidence_by_ref: dict[str, dict[str, str]] = {
        "positive": {},
        "negative": {},
    }
    for sentiment in evidence_by_ref:
        fragments: dict[str, list[str]] = {}
        for record in evidence.get(sentiment, []):
            if not isinstance(record, Mapping):
                continue
            source_id = _as_text(record.get("source_row_id"))
            text = _as_text(record.get("evidence")).casefold()
            fragments.setdefault(source_id, []).append(text)
        evidence_by_ref[sentiment] = {
            source_id: "\n".join(values)
            for source_id, values in fragments.items()
        }
    records = aspect_keywords.get(aspect, [])
    if not isinstance(records, list):
        return result
    for sentiment in ("positive", "negative"):
        selected: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            if record.get("term_type") not in {"bigram", "trigram"}:
                continue
            refs = record.get("review_refs", [])
            if not isinstance(refs, list):
                continue
            source_ids = {
                _as_text(ref.get("source_row_id"))
                for ref in refs
                if isinstance(ref, Mapping)
            }
            term = _as_text(record.get("term")).casefold().strip()
            overlap = {
                source_id
                for source_id in source_ids
                if term and term in evidence_by_ref[sentiment].get(source_id, "")
            }
            if not overlap:
                continue
            copied = copy.deepcopy(dict(record))
            copied["sentiment_review_count"] = len(overlap)
            selected.append(copied)
        selected.sort(
            key=lambda record: (
                -_safe_int(record.get("sentiment_review_count")),
                -_safe_int(record.get("count")),
                _as_text(record.get("term")).casefold(),
            )
        )
        result[sentiment] = selected[:limit]
    return result


def apply_review_quick_filter(
    df: pd.DataFrame,
    aspect_summary: pd.DataFrame | None,
    quick_filter: str,
) -> pd.DataFrame:
    """Apply one Review Explorer shortcut using the existing filter engine."""

    if quick_filter == "All":
        return _copy_nested_object_columns(df)
    if quick_filter == "Positive Evidence":
        return filter_reviews(df, sentiments=["positive"])
    if quick_filter == "Negative Evidence":
        return filter_reviews(df, sentiments=["negative"])
    if quick_filter == "1–2 Star":
        return filter_reviews(df, ratings=[1, 2])
    if quick_filter == "4–5 Star":
        return filter_reviews(df, ratings=[4, 5])

    category = {
        "Product Issues": "PRODUCT",
        "Shopping Experience": "EXPERIENCE",
    }.get(quick_filter)
    if category is None or not isinstance(aspect_summary, pd.DataFrame):
        return _copy_nested_object_columns(df)
    required = {"aspect", "business_category"}
    if not required.issubset(aspect_summary.columns):
        return _copy_nested_object_columns(df)
    categories = aspect_summary["business_category"].astype(str).str.upper()
    aspects = aspect_summary.loc[categories.eq(category), "aspect"].astype(str).tolist()
    if not aspects:
        return _copy_nested_object_columns(df.iloc[0:0])
    return filter_reviews(
        df,
        aspects=aspects,
        sentiments=["negative"] if quick_filter == "Product Issues" else None,
    )


def top_term_records(
    records: object,
    *,
    term_types: Sequence[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Safely select already ranked keyword records for display."""

    if not isinstance(records, list) or limit <= 0:
        return []
    allowed_types = set(term_types or [])
    selected = [
        copy.deepcopy(dict(record))
        for record in records
        if isinstance(record, Mapping)
        and (not allowed_types or record.get("term_type") in allowed_types)
    ]
    return selected[:limit]


def deterministic_business_implications(
    business_summary: Mapping[str, Any] | None,
) -> list[str]:
    """Create restrained implications from transparent persisted counts."""

    if not isinstance(business_summary, Mapping):
        return []
    messages: list[str] = []
    concerns = business_summary.get("top_customer_concerns", [])
    if isinstance(concerns, list) and concerns:
        first = concerns[0]
        if isinstance(first, Mapping):
            aspect = _as_text(first.get("aspect"))
            positive = _safe_int(first.get("positive_review_count"))
            negative = _safe_int(first.get("negative_review_count"))
            if aspect and positive > negative:
                messages.append(
                    f"{aspect} is both highly discussed and predominantly positive, "
                    "making it a potential listing strength."
                )
            elif aspect and negative > positive:
                messages.append(
                    f"{aspect} is highly discussed and has more negative than positive "
                    "feedback, so it may warrant product or listing investigation."
                )

    pain_points = business_summary.get("product_pain_points", [])
    if isinstance(pain_points, list):
        for record in pain_points[:2]:
            if not isinstance(record, Mapping):
                continue
            aspect = _as_text(record.get("aspect"))
            if aspect:
                message = (
                    f"{aspect} shows more negative than positive feedback and may "
                    "warrant product or listing investigation."
                )
                if message not in messages:
                    messages.append(message)
    return messages


def _safe_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 0
