"""Deterministic business-intelligence derivations for analyzed reviews.

The functions in this module never call an LLM and never reinterpret review
text.  They combine the persisted review-level analysis and Aspect summary
into transparent rankings, rating views, sample labels and restrained
recommendation templates used by both the aggregator and the dashboard.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import UTC, datetime
from numbers import Integral, Real
from typing import Any, Mapping

import pandas as pd

from aspect_taxonomy import business_category_for_aspect
from config import (
    BI_EVIDENCE_HIGH_MIN_REVIEWS,
    BI_EVIDENCE_MEDIUM_MIN_REVIEWS,
    BI_DEFAULT_EVIDENCE_LIMIT,
    BI_EXPANDED_EVIDENCE_LIMIT,
    BI_FULL_INSIGHT_MIN_REVIEWS,
    BI_HIGH_NEGATIVE_RATE_PERCENT,
    BI_HIGH_POSITIVE_RATE_PERCENT,
    BI_HIGH_REVIEW_SHARE_PERCENT,
    BI_OPPORTUNITY_HIGH_SCORE,
    BI_OPPORTUNITY_MEDIUM_SCORE,
    BI_QUICK_SCAN_MAX_REVIEWS,
    MIN_ASPECT_REVIEWS_FOR_RANKING,
)


BASE_SENTIMENTS = ("positive", "negative", "neutral")
FORMAL_CATEGORIES = {"PRODUCT", "EXPERIENCE"}
BI_ASPECT_COLUMNS = (
    "review_share",
    "avg_rating",
    "opportunity_score",
    "evidence_level",
)
DEFAULT_EVIDENCE_DISPLAY_LIMIT = BI_DEFAULT_EVIDENCE_LIMIT
EXPANDED_EVIDENCE_DISPLAY_LIMIT = BI_EXPANDED_EVIDENCE_LIMIT


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: object, default: int = 0) -> int:
    return int(_number(value, float(default)))


def _business_flag(value: object, aspect: str) -> bool:
    """Read a persisted eligibility flag without truthy-string surprises."""

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
    return True


def _harmonic_percent(first: object, second: object) -> float:
    """Balance prevalence and directional rate on a transparent 0–100 scale."""

    left = _number(first)
    right = _number(second)
    if left <= 0 or right <= 0:
        return 0.0
    return 2 * left * right / (left + right)


def _json_entries(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        decoded = value
    elif isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        return []
    if not isinstance(decoded, list):
        return []
    return [copy.deepcopy(item) for item in decoded if isinstance(item, dict)]


def _row_entries(row: pd.Series, column: str) -> list[dict[str, Any]]:
    parsed_column = f"parsed_{column}"
    if parsed_column in row.index:
        parsed = _json_entries(row.get(parsed_column))
        if parsed or isinstance(row.get(parsed_column), list):
            return parsed
    return _json_entries(row.get(column))


def _successful_reviews(df: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if "analysis_status" not in df.columns:
        return df.copy(deep=True)
    return df.loc[
        df["analysis_status"].astype(str).str.casefold().eq("success")
    ].copy(deep=True)


def _valid_rating(value: object) -> int | None:
    number = _number(value, -1.0)
    if not number.is_integer():
        return None
    rating = int(number)
    return rating if 1 <= rating <= 5 else None


def _normalized_verified(value: object) -> bool | None:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def evidence_level(review_count: object) -> str:
    """Return a plain-language sample-size label, not statistical confidence."""

    count = _integer(review_count)
    if count >= BI_EVIDENCE_HIGH_MIN_REVIEWS:
        return "High"
    if count >= BI_EVIDENCE_MEDIUM_MIN_REVIEWS:
        return "Medium"
    return "Low"


def sample_level(successful_reviews: object) -> dict[str, Any]:
    """Describe how cautiously the dashboard should word the current sample."""

    count = max(0, _integer(successful_reviews))
    if count <= BI_QUICK_SCAN_MAX_REVIEWS:
        return {
            "key": "early_signals",
            "label": "Quick Scan / Early Signals",
            "analyzed_reviews": count,
            "message": (
                "Small sample. These insights are directional and may not "
                "represent the full customer population."
            ),
        }
    if count < BI_FULL_INSIGHT_MIN_REVIEWS:
        return {
            "key": "limited_sample",
            "label": "Limited Sample",
            "analyzed_reviews": count,
            "message": (
                "Limited sample. Treat rankings as directional until more "
                "reviews are analyzed."
            ),
        }
    return {
        "key": "full_insight",
        "label": "Full Insight View",
        "analyzed_reviews": count,
        "message": "The dataset meets the configured full-insight sample threshold.",
    }


def opportunity_level(score: object) -> str:
    value = _number(score)
    if value >= BI_OPPORTUNITY_HIGH_SCORE:
        return "High"
    if value >= BI_OPPORTUNITY_MEDIUM_SCORE:
        return "Medium"
    return "Low"


def _review_aspect_states(
    analyzed_reviews: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Return one deterministic review-level state used for rating insights."""

    records: list[dict[str, Any]] = []
    for row_position, (_, row) in enumerate(
        _successful_reviews(analyzed_reviews).iterrows()
    ):
        entries = _row_entries(row, "unique_aspects")
        uses_unique_aspects = bool(entries)
        if not entries:
            entries = _row_entries(row, "aspects")
        aspects: dict[str, dict[str, int]] = {}
        for entry in entries:
            raw_aspect = entry.get("aspect")
            aspect = raw_aspect.strip() if isinstance(raw_aspect, str) else ""
            sentiment = str(entry.get("sentiment", "")).strip().casefold()
            if not aspect or sentiment not in BASE_SENTIMENTS:
                continue
            weight = (
                max(1, _integer(entry.get("mention_count"), 1))
                if uses_unique_aspects
                else 1
            )
            sentiment_counts = aspects.setdefault(aspect, {})
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + weight
        source_row_id = row.get("source_row_id")
        if _is_missing(source_row_id) or str(source_row_id).strip() == "":
            source_row_id = f"position:{row_position}"
        records.append(
            {
                "row_identity": row_position,
                "source_row_id": source_row_id,
                "review_id": row.get("review_id"),
                "rating": _valid_rating(row.get("rating")),
                "aspects": aspects,
            }
        )
    return records


def build_rating_insights(
    analyzed_reviews: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build Aspect x star-rating counts and cross-signal observations."""

    states = _review_aspect_states(analyzed_reviews)
    rated = [record for record in states if record["rating"] is not None]
    if not rated:
        return {
            "available": False,
            "rated_reviews": 0,
            "rating_matrix": [],
            "hidden_complaints": [],
            "strengths_in_negative_reviews": [],
            "average_rating_by_aspect": {},
        }

    total_high_rating_reviews = sum(record["rating"] >= 4 for record in rated)
    total_low_rating_reviews = sum(record["rating"] <= 2 for record in rated)
    aspect_ratings: dict[str, list[int]] = {}
    hidden: dict[str, dict[str, Any]] = {}
    strengths: dict[str, dict[str, Any]] = {}
    distribution: dict[str, dict[int, int]] = {}

    for record in rated:
        rating = int(record["rating"])
        identity = record["row_identity"]
        for aspect, sentiment_counts in record["aspects"].items():
            aspect_ratings.setdefault(aspect, []).append(rating)
            distribution.setdefault(aspect, {star: 0 for star in range(1, 6)})[
                rating
            ] += 1
            if rating >= 4 and sentiment_counts.get("negative", 0):
                state = hidden.setdefault(
                    aspect, {"review_ids": set(), "mention_count": 0}
                )
                state["review_ids"].add(identity)
                state["mention_count"] += sentiment_counts["negative"]
            if rating <= 2 and sentiment_counts.get("positive", 0):
                state = strengths.setdefault(
                    aspect, {"review_ids": set(), "mention_count": 0}
                )
                state["review_ids"].add(identity)
                state["mention_count"] += sentiment_counts["positive"]

    matrix: list[dict[str, Any]] = []
    for aspect, counts in distribution.items():
        total = sum(counts.values())
        matrix.append(
            {
                "aspect": aspect,
                "business_category": business_category_for_aspect(aspect),
                "review_count": total,
                "avg_rating": round(sum(aspect_ratings[aspect]) / total, 4),
                **{f"rating_{star}": counts[star] for star in range(1, 6)},
            }
        )
    matrix.sort(key=lambda item: (-item["review_count"], item["aspect"].casefold()))

    hidden_records = [
        {
            "aspect": aspect,
            "business_category": business_category_for_aspect(aspect),
            "review_count": len(state["review_ids"]),
            "mention_count": state["mention_count"],
            "share_of_high_rating_reviews": (
                len(state["review_ids"]) / total_high_rating_reviews * 100
                if total_high_rating_reviews
                else 0.0
            ),
        }
        for aspect, state in hidden.items()
    ]
    hidden_records.sort(
        key=lambda item: (
            -item["review_count"],
            -item["mention_count"],
            item["aspect"].casefold(),
        )
    )
    strength_records = [
        {
            "aspect": aspect,
            "business_category": business_category_for_aspect(aspect),
            "review_count": len(state["review_ids"]),
            "mention_count": state["mention_count"],
            "share_of_low_rating_reviews": (
                len(state["review_ids"]) / total_low_rating_reviews * 100
                if total_low_rating_reviews
                else 0.0
            ),
        }
        for aspect, state in strengths.items()
    ]
    strength_records.sort(
        key=lambda item: (
            -item["review_count"],
            -item["mention_count"],
            item["aspect"].casefold(),
        )
    )
    averages = {
        aspect: round(sum(values) / len(values), 4)
        for aspect, values in aspect_ratings.items()
    }
    return {
        "available": bool(matrix),
        "rated_reviews": len(rated),
        "rating_matrix": matrix,
        "hidden_complaints": hidden_records,
        "strengths_in_negative_reviews": strength_records,
        "average_rating_by_aspect": averages,
    }


def _mention_sentiment_summary(
    analyzed_reviews: pd.DataFrame | None,
    *,
    category: str | None = None,
) -> dict[str, Any]:
    counts = {sentiment: 0 for sentiment in BASE_SENTIMENTS}
    for _, row in _successful_reviews(analyzed_reviews).iterrows():
        entries = _row_entries(row, "aspects")
        weighted = False
        if not entries:
            entries = _row_entries(row, "unique_aspects")
            weighted = True
        for entry in entries:
            raw_aspect = entry.get("aspect")
            aspect = raw_aspect.strip() if isinstance(raw_aspect, str) else ""
            sentiment = str(entry.get("sentiment", "")).strip().casefold()
            if sentiment not in counts or not aspect:
                continue
            if category and business_category_for_aspect(aspect) != category:
                continue
            weight = max(1, _integer(entry.get("mention_count"), 1)) if weighted else 1
            counts[sentiment] += weight
    total = sum(counts.values())
    return {
        "basis": "aspect_mentions",
        "label": "Aspect Mention Sentiment",
        "scope": category or "ALL_ASPECTS_INCLUDING_OTHER",
        "total_mentions": total,
        **{f"{sentiment}_count": counts[sentiment] for sentiment in BASE_SENTIMENTS},
        **{
            f"{sentiment}_rate": counts[sentiment] / total * 100 if total else 0.0
            for sentiment in BASE_SENTIMENTS
        },
    }


def _verified_purchase_summary(
    analyzed_reviews: pd.DataFrame | None,
) -> dict[str, Any]:
    reviews = _successful_reviews(analyzed_reviews)
    if reviews.empty or "verified_purchase" not in reviews.columns:
        return {"available": False, "known_reviews": 0}
    values = [_normalized_verified(value) for value in reviews["verified_purchase"]]
    known = [value for value in values if value is not None]
    if not known:
        return {"available": False, "known_reviews": 0}
    verified = sum(value is True for value in known)
    return {
        "available": True,
        "known_reviews": len(known),
        "verified_reviews": verified,
        "unverified_reviews": len(known) - verified,
        "verified_rate": verified / len(known) * 100,
        "note": "Purchase verification is metadata, not a credibility score.",
    }


def enrich_aspect_summary(
    aspect_summary: pd.DataFrame | None,
    analyzed_reviews: pd.DataFrame | None = None,
    *,
    rating_insights: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Append BI columns while preserving every existing Aspect summary field."""

    if not isinstance(aspect_summary, pd.DataFrame):
        return pd.DataFrame()
    result = aspect_summary.copy(deep=True)
    if result.empty:
        for column in BI_ASPECT_COLUMNS:
            if column not in result.columns:
                result[column] = pd.Series(dtype="object")
        return result

    review_share_source = (
        result["aspect_share_of_reviews"]
        if "aspect_share_of_reviews" in result.columns
        else pd.Series(0.0, index=result.index)
    )
    result["review_share"] = pd.to_numeric(
        review_share_source, errors="coerce"
    ).fillna(0.0)
    negative_rate = pd.to_numeric(
        result.get("negative_rate", pd.Series(0.0, index=result.index)),
        errors="coerce",
    ).fillna(0.0)
    result["opportunity_score"] = result["review_share"] * negative_rate / 100
    counts = pd.to_numeric(
        result.get("aspect_review_count", pd.Series(0, index=result.index)),
        errors="coerce",
    ).fillna(0)
    result["evidence_level"] = counts.map(evidence_level)

    rating_payload = (
        rating_insights
        if isinstance(rating_insights, Mapping)
        else build_rating_insights(analyzed_reviews)
    )
    ratings = rating_payload.get("average_rating_by_aspect", {})
    if not isinstance(ratings, Mapping):
        ratings = {}
    if "aspect" in result.columns:
        result["avg_rating"] = result["aspect"].astype(str).map(ratings)
    else:
        result["avg_rating"] = None
    return result


def _metric_records(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in enriched.to_dict("records"):
        raw_aspect = row.get("aspect")
        aspect = raw_aspect.strip() if isinstance(raw_aspect, str) else ""
        if not aspect:
            continue
        review_count = _integer(row.get("aspect_review_count"))
        review_share = _number(
            row.get("review_share", row.get("aspect_share_of_reviews"))
        )
        positive_rate = _number(row.get("positive_rate"))
        negative_rate = _number(row.get("negative_rate"))
        avg_rating_value = row.get("avg_rating")
        avg_rating = None if _is_missing(avg_rating_value) else _number(avg_rating_value)
        opportunity = review_share * negative_rate / 100
        raw_category = row.get("business_category")
        category = (
            raw_category.strip().upper()
            if isinstance(raw_category, str) and raw_category.strip()
            else business_category_for_aspect(aspect)
        )
        selling_score = _harmonic_percent(review_share, positive_rate)
        pain_score = _harmonic_percent(review_share, negative_rate)
        record = {
            "aspect": aspect,
            "business_category": category,
            "aspect_review_count": review_count,
            "review_count": review_count,
            "mention_count": _integer(row.get("mention_count")),
            "review_share": review_share,
            "aspect_share_of_reviews": review_share,
            "positive_review_count": _integer(row.get("positive_review_count")),
            "negative_review_count": _integer(row.get("negative_review_count")),
            "neutral_review_count": _integer(row.get("neutral_review_count")),
            "mixed_review_count": _integer(row.get("mixed_review_count")),
            "positive_rate": positive_rate,
            "negative_rate": negative_rate,
            "neutral_rate": _number(row.get("neutral_rate")),
            "mixed_rate": _number(row.get("mixed_rate")),
            "avg_rating": avg_rating,
            "opportunity_score": opportunity,
            "opportunity_level": opportunity_level(opportunity),
            "selling_score": selling_score,
            "pain_score": pain_score,
            "evidence_level": evidence_level(review_count),
            "include_in_business_metrics": _business_flag(
                row.get("include_in_business_metrics"), aspect
            ),
        }
        record["net_sentiment"] = positive_rate - negative_rate
        records.append(record)
    return records


def _recommendation(record: Mapping[str, Any], minimum_support: int) -> dict[str, Any]:
    aspect = str(record.get("aspect", "Unknown"))
    category = str(record.get("business_category", "OTHER"))
    reviews = _integer(record.get("review_count"))
    share = _number(record.get("review_share"))
    positive = _number(record.get("positive_rate"))
    negative = _number(record.get("negative_rate"))
    if reviews < minimum_support or category == "OTHER":
        recommendation_type = "Insufficient Evidence"
        message = f"More reviews are needed before acting on {aspect}."
    elif (
        category == "PRODUCT"
        and share >= BI_HIGH_REVIEW_SHARE_PERCENT
        and positive >= BI_HIGH_POSITIVE_RATE_PERCENT
        and positive > negative
    ):
        recommendation_type = "Marketing Opportunity"
        message = f"Consider emphasizing {aspect} in listing copy and ad creatives."
    elif (
        category == "PRODUCT"
        and share >= BI_HIGH_REVIEW_SHARE_PERCENT
        and negative >= BI_HIGH_NEGATIVE_RATE_PERCENT
        and negative > positive
    ):
        recommendation_type = "Product Improvement Priority"
        message = (
            f"{aspect} is a frequent source of negative feedback and should be "
            "investigated."
        )
    else:
        recommendation_type = "Monitor"
        message = f"Continue monitoring {aspect} as more reviews are collected."
    return {
        "aspect": aspect,
        "business_category": category,
        "type": recommendation_type,
        "message": message,
        "review_count": reviews,
        "review_share": share,
        "evidence_level": evidence_level(reviews),
    }


def _executive_summary(
    mode: Mapping[str, Any],
    selling: list[dict[str, Any]],
    pain: list[dict[str, Any]],
) -> str:
    strengths = [record["aspect"] for record in selling[:2]]
    issues = [record["aspect"] for record in pain[:2]]
    if not strengths and not issues:
        return (
            "The current sample does not provide enough supported product signals "
            "for a formal selling-point or pain-point summary."
        )
    prefix = {
        "early_signals": "Early signals show that customers",
        "limited_sample": "Within this limited sample, customers",
        "full_insight": "Customers",
    }.get(str(mode.get("key")), "Customers")
    if strengths and issues:
        issue_names = " and ".join(issues)
        issue_verb = "generates" if len(issues) == 1 else "generate"
        return (
            f"{prefix} most frequently praise {' and '.join(strengths)}, while "
            f"{issue_names} {issue_verb} the strongest negative feedback."
        )
    if strengths:
        return f"{prefix} most frequently praise {' and '.join(strengths)}."
    return f"{prefix} most frequently criticize {' and '.join(issues)}."


def build_business_intelligence(
    analyzed_reviews: pd.DataFrame | None,
    aspect_summary: pd.DataFrame | None,
    *,
    minimum_support: int = MIN_ASPECT_REVIEWS_FOR_RANKING,
    top_n: int = 5,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create the complete Step 12 schema from existing deterministic outputs."""

    if (
        isinstance(minimum_support, bool)
        or not isinstance(minimum_support, Integral)
        or minimum_support <= 0
    ):
        raise ValueError("minimum_support must be a positive integer")
    if isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    successful = _successful_reviews(analyzed_reviews)
    total_rows = len(analyzed_reviews) if isinstance(analyzed_reviews, pd.DataFrame) else 0
    mode = sample_level(len(successful))
    rating = build_rating_insights(analyzed_reviews)
    enriched = enrich_aspect_summary(
        aspect_summary,
        analyzed_reviews,
        rating_insights=rating,
    )
    metrics = _metric_records(enriched)
    formal = [
        record
        for record in metrics
        if record["business_category"] in FORMAL_CATEGORIES
        and record["include_in_business_metrics"]
    ]

    selling = [
        copy.deepcopy(record)
        for record in formal
        if record["business_category"] == "PRODUCT"
        and record["review_count"] >= minimum_support
        and record["positive_review_count"] > record["negative_review_count"]
    ]
    selling.sort(
        key=lambda record: (
            -record["selling_score"],
            -record["review_count"],
            -record["positive_review_count"],
            -record["mention_count"],
            record["aspect"].casefold(),
        )
    )
    selling = selling[:top_n]

    pain = [
        copy.deepcopy(record)
        for record in formal
        if record["business_category"] == "PRODUCT"
        and record["review_count"] >= minimum_support
        and record["negative_review_count"] > record["positive_review_count"]
    ]
    pain.sort(
        key=lambda record: (
            -record["pain_score"],
            -record["review_count"],
            -record["negative_review_count"],
            -record["mention_count"],
            record["aspect"].casefold(),
        )
    )
    pain = pain[:top_n]

    opportunities = [
        copy.deepcopy(record)
        for record in formal
        if record["review_count"] >= minimum_support
        and record["negative_review_count"] > 0
    ]
    opportunities.sort(
        key=lambda record: (
            -record["opportunity_score"],
            -record["review_count"],
            -record["negative_review_count"],
            -record["mention_count"],
            record["aspect"].casefold(),
        )
    )

    matrix: list[dict[str, Any]] = []
    for record in formal:
        point = copy.deepcopy(record)
        high_share = record["review_share"] >= BI_HIGH_REVIEW_SHARE_PERCENT
        positive_direction = record["net_sentiment"] > 0
        negative_direction = record["net_sentiment"] < 0
        if math.isclose(record["net_sentiment"], 0.0, abs_tol=1e-9):
            quadrant = "Balanced / Monitor"
        elif high_share and positive_direction:
            quadrant = "Strength"
        elif high_share:
            quadrant = "Improvement Priority"
        elif positive_direction:
            quadrant = "Minor Positive"
        else:
            quadrant = "Emerging Concern"
        point["quadrant"] = quadrant
        matrix.append(point)
    matrix.sort(key=lambda record: (-record["review_count"], record["aspect"].casefold()))

    recommendations = [
        _recommendation(record, minimum_support)
        for record in sorted(
            metrics,
            key=lambda record: (-record["review_count"], record["aspect"].casefold()),
        )
    ]
    return {
        "schema_version": "step12.v1",
        "dataset_size": total_rows,
        "analyzed_review_count": len(successful),
        "sample_level": mode,
        "overall_sentiment": _mention_sentiment_summary(analyzed_reviews),
        "product_experience_summary": _mention_sentiment_summary(
            analyzed_reviews, category="PRODUCT"
        ),
        "shopping_experience_summary": _mention_sentiment_summary(
            analyzed_reviews, category="EXPERIENCE"
        ),
        "selling_points": selling,
        "pain_points": pain,
        "opportunity_ranking": opportunities,
        "aspect_business_metrics": metrics,
        "priority_matrix": matrix,
        "rating_insights": rating,
        "verified_purchase": _verified_purchase_summary(analyzed_reviews),
        "recommendations": recommendations,
        "executive_summary": _executive_summary(mode, selling, pain),
        "business_intelligence_rules": {
            "minimum_support": minimum_support,
            "selling_score": (
                "harmonic_mean(review_share_percent, positive_rate_percent)"
            ),
            "pain_score": (
                "harmonic_mean(review_share_percent, negative_rate_percent)"
            ),
            "opportunity_score": "review_share * negative_rate / 100",
            "priority_high_share_percent": BI_HIGH_REVIEW_SHARE_PERCENT,
            "evidence_level": {
                "low": f"review_count < {BI_EVIDENCE_MEDIUM_MIN_REVIEWS}",
                "medium": (
                    f"{BI_EVIDENCE_MEDIUM_MIN_REVIEWS} <= review_count < "
                    f"{BI_EVIDENCE_HIGH_MIN_REVIEWS}"
                ),
                "high": f"review_count >= {BI_EVIDENCE_HIGH_MIN_REVIEWS}",
            },
            "score_disclaimer": (
                "Opportunity Score is a deterministic prioritization aid, not "
                "statistical significance."
            ),
        },
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
    }
