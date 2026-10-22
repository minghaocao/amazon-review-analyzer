"""Deterministic two-product comparison built from existing analysis outputs.

Step 13 deliberately sits above the frozen analysis pipeline.  It reads the
seven dashboard artifacts for two products, aligns their canonical Aspects,
normalizes comparisons to review/rate percentages, and writes only compact
comparison artifacts.  It contains no prompt, model client, scraper, or review
analysis code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from aspect_taxonomy import (
    STANDARD_ASPECTS,
    business_category_for_aspect,
    is_standard_aspect,
)
from config import (
    ANALYSIS_CANDIDATE_VERSION,
    COMPARISON_CLEAR_GAP_PERCENT,
    COMPARISON_CURRENT_DIR,
    COMPARISON_DEFAULT_EVIDENCE_LIMIT,
    COMPARISON_EARLY_SAMPLE_MAX_REVIEWS,
    COMPARISON_EVIDENCE_HIGH_MIN_REVIEWS,
    COMPARISON_EVIDENCE_MEDIUM_MIN_REVIEWS,
    COMPARISON_LANGUAGE_DISTINCTIVE_SHARE_GAP,
    COMPARISON_LANGUAGE_MIN_REVIEW_COUNT,
    COMPARISON_MIN_SUPPORT,
    COMPARISON_SLIGHT_GAP_PERCENT,
    COMPARISON_UNEVEN_SAMPLE_RATIO,
    COMPARISON_WHITE_SPACE_NEGATIVE_RATE,
    OUTPUTS_DIR,
    PROJECT_ROOT,
)
COMPARISON_SCHEMA_VERSION = "step13.v1"
# Saved Analysis manifests are a source compatibility contract, distinct from
# the comparison-output schema above.  Keep the accepted set explicit so a
# future library format cannot silently change the meaning of an A/B result.
SUPPORTED_ANALYSIS_MANIFEST_SCHEMAS = frozenset({"step13.1.analysis.v1"})
FIXTURE_PATH = PROJECT_ROOT / "data" / "comparison_fixture.json"
SOURCE_OUTPUT_FILES = {
    "analyzed_reviews": "analyzed_reviews.csv",
    "aspect_summary": "aspect_summary.csv",
    "business_summary": "business_summary.json",
    "aspect_evidence": "aspect_evidence.json",
    "keyword_summary": "keyword_summary.csv",
    "sentiment_keywords": "sentiment_keywords.json",
    "aspect_keywords": "aspect_keywords.json",
}
SOURCE_REQUIRED_COLUMNS = {
    "analyzed_reviews": {"analysis_status"},
    "aspect_summary": {
        "aspect",
        "aspect_review_count",
        "aspect_share_of_reviews",
        "positive_rate",
        "negative_rate",
        "neutral_rate",
        "mixed_rate",
    },
    "keyword_summary": {"term", "review_count", "review_share"},
}
COMPARISON_OUTPUT_COLUMNS = (
    "aspect",
    "business_category",
    "a_review_count",
    "b_review_count",
    "a_review_share",
    "b_review_share",
    "review_share_gap",
    "a_positive_rate",
    "b_positive_rate",
    "a_negative_rate",
    "b_negative_rate",
    "positive_rate_gap",
    "negative_rate_gap",
    "a_net_sentiment",
    "b_net_sentiment",
    "net_sentiment_gap",
    "a_avg_rating",
    "b_avg_rating",
    "comparison_status",
    "comparison_evidence",
)


class ComparisonError(ValueError):
    """The selected saved analyses cannot produce a safe comparison."""


class AnalysisVersionMismatch(ComparisonError):
    """The two products were produced by incompatible analysis versions."""


class ComparisonOutputSafetyError(ComparisonError):
    """Comparison output was pointed outside a narrow comparison/current root."""


@dataclass(frozen=True)
class ProductAnalysis:
    """Compact, language-independent view of one completed analysis run."""

    product_id: str
    product_name: str
    source_type: str
    analysis_run_id: str
    source_reference: str
    analysis_candidate: str
    taxonomy_version: str
    total_reviews: int
    successful_reviews: int
    coverage: float
    sample_level: str
    aspect_metrics: tuple[dict[str, Any], ...]
    aspect_evidence: dict[str, Any]
    language_by_aspect: dict[str, Any]
    rating_summary: dict[str, Any]
    # Stable Saved Analysis identity.  It is optional for legacy current
    # workspaces and fixtures so existing direct constructors remain valid.
    analysis_id: str = ""
    analysis_schema_version: str = ""

    def public_metadata(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "source_type": self.source_type,
            "analysis_run_id": self.analysis_run_id,
            "analysis_id": self.analysis_id,
            "source_reference": self.source_reference,
            "analysis_candidate": self.analysis_candidate,
            "taxonomy_version": self.taxonomy_version,
            "analysis_schema_version": self.analysis_schema_version,
            "total_reviews": self.total_reviews,
            "successful_reviews": self.successful_reviews,
            "coverage": self.coverage,
            "sample_level": self.sample_level,
        }


@dataclass(frozen=True)
class ComparisonWorkspacePaths:
    root: Path
    summary: Path
    aspect_comparison: Path
    metadata: Path

    @classmethod
    def from_root(cls, root: str | Path = COMPARISON_CURRENT_DIR) -> "ComparisonWorkspacePaths":
        directory = Path(root)
        return cls(
            root=directory,
            summary=directory / "comparison_summary.json",
            aspect_comparison=directory / "aspect_comparison.csv",
            metadata=directory / "comparison_metadata.json",
        )


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(number) else number


def _integer(value: object, default: int = 0) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return default


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _optional_number(value: object) -> float | None:
    if value is None or _is_missing(value) or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "product"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Unable to read comparison source metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError("Comparison source metadata must contain a JSON object.")
    return value


def _read_required_json(path: Path, output_name: str) -> dict[str, Any]:
    """Read one persisted JSON artifact without rebuilding upstream results."""

    if not path.is_file():
        raise ComparisonError(
            f"Comparison source is missing required output: {output_name}"
        )
    return copy.deepcopy(_read_json_object(path))


def _read_required_csv(path: Path, output_name: str) -> pd.DataFrame:
    """Read one persisted CSV and validate only the comparison contract."""

    if not path.is_file():
        raise ComparisonError(
            f"Comparison source is missing required output: {output_name}"
        )
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ComparisonError(f"Unable to read {output_name}: {exc}") from exc
    missing = SOURCE_REQUIRED_COLUMNS.get(output_name, set()).difference(frame.columns)
    if missing:
        raise ComparisonError(
            f"Comparison source {output_name} is missing columns: "
            + ", ".join(sorted(missing))
        )
    return frame.copy(deep=True)


def _safe_source_reference(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _rating_summary_from_reviews(frame: pd.DataFrame | None) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "rating" not in frame:
        return {"available": False, "rated_reviews": 0, "distribution": []}
    source = frame
    if "analysis_status" in source:
        source = source.loc[source["analysis_status"].astype(str).str.casefold().eq("success")]
    ratings = pd.to_numeric(source["rating"], errors="coerce")
    ratings = ratings.loc[ratings.between(1, 5, inclusive="both")].dropna()
    if ratings.empty:
        return {"available": False, "rated_reviews": 0, "distribution": []}
    total = len(ratings)
    distribution = [
        {
            "rating": rating,
            "count": int((ratings.round().astype(int) == rating).sum()),
            "share": float((ratings.round().astype(int) == rating).sum() / total * 100),
        }
        for rating in range(1, 6)
    ]
    return {
        "available": True,
        "rated_reviews": total,
        "average_rating": float(ratings.mean()),
        "distribution": distribution,
    }


def _metric_records(aspect_summary: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    """Normalize persisted Aspect rows; never re-aggregate review results."""

    records = aspect_summary.to_dict("records")
    normalized: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        aspect = str(raw.get("aspect", "")).strip()
        if not aspect or not is_standard_aspect(aspect):
            continue
        review_count = _integer(raw.get("aspect_review_count", raw.get("review_count")))
        review_share = (
            _number(raw.get("review_share", raw.get("aspect_share_of_reviews")))
            if review_count
            else 0.0
        )
        # A persisted zero-mention row may contain numeric zero placeholders.
        # They are not satisfaction measurements.  Preserve absence as null so
        # consumers cannot render it as "0% positive" or infer a weakness.
        positive = _optional_number(raw.get("positive_rate")) if review_count else None
        negative = _optional_number(raw.get("negative_rate")) if review_count else None
        normalized.append(
            {
                "aspect": aspect,
                "business_category": business_category_for_aspect(aspect),
                "review_count": review_count,
                "review_share": review_share,
                "positive_rate": positive,
                "negative_rate": negative,
                "neutral_rate": (
                    _optional_number(raw.get("neutral_rate")) if review_count else None
                ),
                "mixed_rate": (
                    _optional_number(raw.get("mixed_rate")) if review_count else None
                ),
                "avg_rating": (
                    _optional_number(raw.get("avg_rating")) if review_count else None
                ),
                "evidence_level": (
                    str(raw.get("evidence_level") or evidence_level(review_count))
                    if review_count
                    else "no_evidence"
                ),
            }
        )
    order = {aspect: index for index, aspect in enumerate(STANDARD_ASPECTS)}
    normalized.sort(key=lambda record: order.get(record["aspect"], len(order)))
    return tuple(normalized)


def _as_text(value: object) -> str:
    if value is None or _is_missing(value):
        return ""
    return str(value)


def _aspect_language_by_sentiment(
    aspect_keywords: Mapping[str, Any],
    aspect_evidence: Mapping[str, Any],
    aspect: str,
    *,
    limit: int = 25,
) -> dict[str, list[dict[str, Any]]]:
    """Group persisted phrases by persisted Evidence refs, without BI rebuilds."""

    result: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": []}
    if limit <= 0:
        return result
    evidence_buckets = aspect_evidence.get(aspect, {})
    if not isinstance(evidence_buckets, Mapping):
        evidence_buckets = {}
    phrases = aspect_keywords.get(aspect, [])
    if not isinstance(phrases, list):
        return result
    for sentiment in ("positive", "negative"):
        fragments: dict[str, list[str]] = {}
        saved_evidence = evidence_buckets.get(sentiment, [])
        if isinstance(saved_evidence, list):
            for record in saved_evidence:
                if not isinstance(record, Mapping):
                    continue
                source_id = _as_text(record.get("source_row_id"))
                fragments.setdefault(source_id, []).append(
                    _as_text(record.get("evidence")).casefold()
                )
        evidence_by_ref = {
            source_id: "\n".join(values) for source_id, values in fragments.items()
        }
        selected: list[dict[str, Any]] = []
        for record in phrases:
            if not isinstance(record, Mapping) or record.get("term_type") not in {
                "bigram",
                "trigram",
            }:
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
                if term and term in evidence_by_ref.get(source_id, "")
            }
            if not overlap:
                continue
            copied = copy.deepcopy(dict(record))
            copied["sentiment_review_count"] = len(overlap)
            selected.append(copied)
        selected.sort(
            key=lambda record: (
                -_integer(record.get("sentiment_review_count")),
                -_integer(record.get("count")),
                _as_text(record.get("term")).casefold(),
            )
        )
        result[sentiment] = selected[:limit]
    return result


def load_product_analysis(
    outputs_dir: str | Path,
    *,
    product_name: str,
    product_id: str | None = None,
    source_type: str | None = None,
    analysis_run_id: str | None = None,
    analysis_candidate: str | None = None,
    taxonomy_version: str | None = None,
    analysis_id: str | None = None,
    assume_current_version: bool = False,
) -> ProductAnalysis:
    """Load one completed seven-output analysis without running any pipeline."""

    root = Path(outputs_dir)
    frame = _read_required_csv(
        root / SOURCE_OUTPUT_FILES["analyzed_reviews"], "analyzed_reviews"
    )
    aspect_summary = _read_required_csv(
        root / SOURCE_OUTPUT_FILES["aspect_summary"], "aspect_summary"
    )
    business_summary = _read_required_json(
        root / SOURCE_OUTPUT_FILES["business_summary"], "business_summary"
    )
    evidence = _read_required_json(
        root / SOURCE_OUTPUT_FILES["aspect_evidence"], "aspect_evidence"
    )
    # A comparison source is deliberately a complete saved seven-output run.
    # Reading these files validates that contract; it never regenerates them.
    _read_required_csv(
        root / SOURCE_OUTPUT_FILES["keyword_summary"], "keyword_summary"
    )
    _read_required_json(
        root / SOURCE_OUTPUT_FILES["sentiment_keywords"], "sentiment_keywords"
    )
    aspect_keywords = _read_required_json(
        root / SOURCE_OUTPUT_FILES["aspect_keywords"], "aspect_keywords"
    )
    metadata = _read_json_object(root / "analysis_run_metadata.json")
    manifest = _read_json_object(root / "analysis_manifest.json")
    manifest_schema = str(manifest.get("schema_version") or "").strip()
    if manifest and manifest_schema not in SUPPORTED_ANALYSIS_MANIFEST_SCHEMAS:
        raise AnalysisVersionMismatch(
            "Saved Analysis schema is incompatible with this comparison version."
        )
    quality = business_summary.get("data_quality", {})
    if not isinstance(quality, Mapping):
        quality = {}
    total = _integer(quality.get("total_input_rows"), len(frame))
    successful = _integer(
        quality.get("successful_reviews"),
        int(frame["analysis_status"].astype(str).str.casefold().eq("success").sum()),
    )
    total = max(total, successful)
    sample = business_summary.get("sample_level", {})
    sample_level = str(sample.get("key", "")) if isinstance(sample, Mapping) else ""
    candidate_value = (
        analysis_candidate
        or manifest.get("analysis_candidate")
        or metadata.get("candidate")
    )
    taxonomy_value = (
        taxonomy_version
        or manifest.get("taxonomy_version")
        or metadata.get("taxonomy_version")
    )
    if assume_current_version:
        candidate_value = candidate_value or ANALYSIS_CANDIDATE_VERSION
        taxonomy_value = taxonomy_value or ANALYSIS_CANDIDATE_VERSION
    if not candidate_value or not taxonomy_value:
        raise AnalysisVersionMismatch(
            "Analysis version metadata is unavailable. Supply both the analysis "
            "candidate and taxonomy version before comparing this saved result."
        )
    candidate = str(candidate_value)
    taxonomy = str(taxonomy_value)
    run_id = str(
        analysis_run_id
        or manifest.get("analysis_run_id")
        or metadata.get("run_id")
        or hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:12]
    )
    source = str(
        source_type
        or manifest.get("source_type")
        or metadata.get("source_type")
        or "saved_analysis"
    )
    language = {
        aspect: _aspect_language_by_sentiment(
            aspect_keywords,
            evidence,
            aspect,
            limit=25,
        )
        for aspect in STANDARD_ASPECTS
    }
    return ProductAnalysis(
        product_id=(
            product_id
            or str(manifest.get("product_id") or "").strip()
            or str(manifest.get("analysis_id") or "").strip()
            or _slug(product_name)
        ),
        product_name=product_name.strip() or "Product",
        source_type=source,
        analysis_run_id=run_id,
        source_reference=_safe_source_reference(root),
        analysis_candidate=candidate,
        taxonomy_version=taxonomy,
        total_reviews=total,
        successful_reviews=successful,
        coverage=(successful / total if total else 0.0),
        sample_level=sample_level or ("early_signals" if successful < 10 else "limited_sample"),
        aspect_metrics=_metric_records(aspect_summary),
        aspect_evidence=copy.deepcopy(evidence),
        language_by_aspect=language,
        rating_summary=_rating_summary_from_reviews(frame),
        analysis_id=str(
            analysis_id or manifest.get("analysis_id") or ""
        ).strip(),
        analysis_schema_version=manifest_schema,
    )


def load_fixture_product(
    fixture_key: str,
    *,
    fixture_path: str | Path = FIXTURE_PATH,
    product_name: str | None = None,
) -> ProductAnalysis:
    """Load one compact offline fixture used only for UI/tests."""

    payload = _read_json_object(Path(fixture_path))
    products = payload.get("products")
    if not isinstance(products, Mapping) or not isinstance(products.get(fixture_key), Mapping):
        raise ComparisonError(f"Unknown comparison fixture: {fixture_key}")
    raw = products[fixture_key]
    name = str(product_name or raw.get("product_name") or fixture_key)
    metrics = tuple(copy.deepcopy(raw.get("aspect_metrics", [])))
    candidate = raw.get("analysis_candidate")
    taxonomy = raw.get("taxonomy_version")
    if not candidate or not taxonomy:
        raise AnalysisVersionMismatch(
            "Comparison fixture must declare its analysis candidate and taxonomy version."
        )
    return ProductAnalysis(
        product_id=str(raw.get("product_id") or _slug(name)),
        product_name=name,
        source_type=str(raw.get("source_type") or "synthetic_fixture"),
        analysis_run_id=str(raw.get("analysis_run_id") or fixture_key),
        source_reference=f"fixture:{fixture_key}",
        analysis_candidate=str(candidate),
        taxonomy_version=str(taxonomy),
        total_reviews=_integer(raw.get("total_reviews")),
        successful_reviews=_integer(raw.get("successful_reviews")),
        coverage=_number(raw.get("coverage")),
        sample_level=str(raw.get("sample_level") or "full_insight"),
        aspect_metrics=metrics,
        aspect_evidence=copy.deepcopy(raw.get("aspect_evidence", {})),
        language_by_aspect=copy.deepcopy(raw.get("language_by_aspect", {})),
        rating_summary=copy.deepcopy(raw.get("rating_summary", {})),
        analysis_id=str(raw.get("analysis_id") or ""),
        analysis_schema_version="",
    )


def analysis_versions_compatible(
    product_a: ProductAnalysis,
    product_b: ProductAnalysis,
    *,
    required_candidate: str = ANALYSIS_CANDIDATE_VERSION,
) -> bool:
    a_taxonomy = product_a.taxonomy_version.strip()
    b_taxonomy = product_b.taxonomy_version.strip()
    schemas = tuple(
        schema
        for schema in (
            product_a.analysis_schema_version.strip(),
            product_b.analysis_schema_version.strip(),
        )
        if schema
    )
    schema_compatible = (
        all(schema in SUPPORTED_ANALYSIS_MANIFEST_SCHEMAS for schema in schemas)
        and (len(schemas) < 2 or schemas[0] == schemas[1])
    )
    return (
        product_a.analysis_candidate == required_candidate
        and product_b.analysis_candidate == required_candidate
        and product_a.analysis_candidate == product_b.analysis_candidate
        and bool(a_taxonomy)
        and bool(b_taxonomy)
        and a_taxonomy.casefold() != "unknown"
        and b_taxonomy.casefold() != "unknown"
        and a_taxonomy == b_taxonomy
        and schema_compatible
    )


def evidence_level(a_review_count: int, b_review_count: int | None = None) -> str:
    """Return a support label; with B supplied, both sides must meet a tier."""

    counts = [max(0, int(a_review_count))]
    if b_review_count is not None:
        counts.append(max(0, int(b_review_count)))
    minimum = min(counts)
    if minimum == 0:
        return "no_evidence"
    if minimum >= COMPARISON_EVIDENCE_HIGH_MIN_REVIEWS:
        return "High"
    if minimum >= COMPARISON_EVIDENCE_MEDIUM_MIN_REVIEWS:
        return "Medium"
    return "Low"


def classify_comparison(
    *,
    a_review_count: int,
    b_review_count: int,
    net_sentiment_gap: float | None,
    early_comparison: bool = False,
) -> str:
    """Classify A relative to B using rates and explicit support guards."""

    if (
        a_review_count < COMPARISON_MIN_SUPPORT
        or b_review_count < COMPARISON_MIN_SUPPORT
        or net_sentiment_gap is None
    ):
        return "insufficient_evidence"
    gap = float(net_sentiment_gap)
    if abs(gap) < COMPARISON_SLIGHT_GAP_PERCENT:
        return "similar"
    if early_comparison:
        return "directional_signal_a" if gap > 0 else "directional_signal_b"
    high_evidence = evidence_level(a_review_count, b_review_count) == "High"
    if gap >= COMPARISON_CLEAR_GAP_PERCENT and high_evidence:
        return "clear_advantage"
    if gap >= COMPARISON_SLIGHT_GAP_PERCENT:
        return "slight_advantage"
    if gap <= -COMPARISON_CLEAR_GAP_PERCENT and high_evidence:
        return "clear_weakness"
    return "slight_weakness"


def _metric_index(product: ProductAnalysis) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("aspect")): copy.deepcopy(record)
        for record in product.aspect_metrics
        if isinstance(record, Mapping) and record.get("aspect")
    }


def _aspect_comparison(
    aspect: str,
    a: Mapping[str, Any] | None,
    b: Mapping[str, Any] | None,
    *,
    early_comparison: bool,
) -> dict[str, Any]:
    a_count = _integer(a.get("review_count")) if a else 0
    b_count = _integer(b.get("review_count")) if b else 0
    a_positive = _optional_number(a.get("positive_rate")) if a and a_count else None
    b_positive = _optional_number(b.get("positive_rate")) if b and b_count else None
    a_negative = _optional_number(a.get("negative_rate")) if a and a_count else None
    b_negative = _optional_number(b.get("negative_rate")) if b and b_count else None
    a_share = _number(a.get("review_share")) if a and a_count else 0.0
    b_share = _number(b.get("review_share")) if b and b_count else 0.0
    a_net = (a_positive - a_negative) if a_positive is not None and a_negative is not None else None
    b_net = (b_positive - b_negative) if b_positive is not None and b_negative is not None else None
    net_gap = (a_net - b_net) if a_net is not None and b_net is not None else None
    status = classify_comparison(
        a_review_count=a_count,
        b_review_count=b_count,
        net_sentiment_gap=net_gap,
        early_comparison=early_comparison,
    )
    leader = (
        "product_a"
        if status in {"clear_advantage", "slight_advantage", "directional_signal_a"}
        else "product_b"
        if status in {"clear_weakness", "slight_weakness", "directional_signal_b"}
        else "none"
    )
    return {
        "aspect": aspect,
        "business_category": business_category_for_aspect(aspect),
        "a_review_count": a_count,
        "b_review_count": b_count,
        "a_review_share": a_share,
        "b_review_share": b_share,
        "review_share_gap": a_share - b_share,
        "a_positive_rate": a_positive,
        "b_positive_rate": b_positive,
        "a_negative_rate": a_negative,
        "b_negative_rate": b_negative,
        "a_neutral_rate": _optional_number(a.get("neutral_rate")) if a and a_count else None,
        "b_neutral_rate": _optional_number(b.get("neutral_rate")) if b and b_count else None,
        "positive_rate_gap": (
            a_positive - b_positive if a_positive is not None and b_positive is not None else None
        ),
        "negative_rate_gap": (
            a_negative - b_negative if a_negative is not None and b_negative is not None else None
        ),
        "a_net_sentiment": a_net,
        "b_net_sentiment": b_net,
        "net_sentiment_gap": net_gap,
        "a_avg_rating": _optional_number(a.get("avg_rating")) if a and a_count else None,
        "b_avg_rating": _optional_number(b.get("avg_rating")) if b and b_count else None,
        "a_evidence_level": (
            str(a.get("evidence_level") or evidence_level(a_count))
            if a and a_count
            else "no_evidence"
        ),
        "b_evidence_level": (
            str(b.get("evidence_level") or evidence_level(b_count))
            if b and b_count
            else "no_evidence"
        ),
        "comparison_status": status,
        "comparison_evidence": evidence_level(a_count, b_count),
        "leader": leader,
        "average_review_share": (a_share + b_share) / 2,
        "signal": (
            "unique_strength"
            if status == "clear_advantage"
            else "competitive_risk"
            if status == "clear_weakness"
            else "directional"
            if status.startswith("directional_signal")
            else "none"
        ),
    }


def comparison_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    gap = record.get("net_sentiment_gap")
    supported = record.get("comparison_status") != "insufficient_evidence"
    return (
        0 if supported else 1,
        -abs(_number(gap)) if gap is not None else 0.0,
        -_number(record.get("average_review_share")),
        str(record.get("aspect", "")).casefold(),
    )


def _phrase_index(records: object) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(records, list):
        return index
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        term = str(raw.get("term", "")).strip()
        normalized = re.sub(r"[^\w]+", " ", term.casefold()).strip()
        if not normalized:
            continue
        review_count = _integer(raw.get("sentiment_review_count", raw.get("review_count")))
        candidate = {
            "term": term,
            "normalized_term": normalized,
            "review_count": review_count,
            "review_share": _number(raw.get("review_share")),
            "term_type": str(raw.get("term_type", "")),
        }
        current = index.get(normalized)
        if current is None or (candidate["review_count"], candidate["term"]) > (
            current["review_count"], current["term"]
        ):
            index[normalized] = candidate
    return index


def compare_phrase_records(a_records: object, b_records: object) -> dict[str, Any]:
    """Compare existing phrase counts with normalized exact matching only."""

    a = _phrase_index(a_records)
    b = _phrase_index(b_records)
    shared: list[dict[str, Any]] = []
    a_distinctive: list[dict[str, Any]] = []
    b_distinctive: list[dict[str, Any]] = []
    for key in sorted(set(a).union(b)):
        left = a.get(key)
        right = b.get(key)
        if (
            left
            and right
            and left["review_count"] >= COMPARISON_LANGUAGE_MIN_REVIEW_COUNT
            and right["review_count"] >= COMPARISON_LANGUAGE_MIN_REVIEW_COUNT
        ):
            shared.append(
                {
                    "term": left["term"],
                    "normalized_term": key,
                    "a_review_count": left["review_count"],
                    "b_review_count": right["review_count"],
                    "a_review_share": left["review_share"],
                    "b_review_share": right["review_share"],
                }
            )
        if left and left["review_count"] >= COMPARISON_LANGUAGE_MIN_REVIEW_COUNT:
            other_share = right["review_share"] if right else 0.0
            if left["review_share"] - other_share >= COMPARISON_LANGUAGE_DISTINCTIVE_SHARE_GAP:
                a_distinctive.append(copy.deepcopy(left))
        if right and right["review_count"] >= COMPARISON_LANGUAGE_MIN_REVIEW_COUNT:
            other_share = left["review_share"] if left else 0.0
            if right["review_share"] - other_share >= COMPARISON_LANGUAGE_DISTINCTIVE_SHARE_GAP:
                b_distinctive.append(copy.deepcopy(right))
    shared.sort(
        key=lambda item: (
            -min(item["a_review_count"], item["b_review_count"]),
            item["normalized_term"],
        )
    )
    for records in (a_distinctive, b_distinctive):
        records.sort(key=lambda item: (-item["review_share"], -item["review_count"], item["normalized_term"]))
    return {
        "shared": shared,
        "product_a_distinctive": a_distinctive,
        "product_b_distinctive": b_distinctive,
    }


def compare_customer_language(
    product_a: ProductAnalysis,
    product_b: ProductAnalysis,
) -> dict[str, Any]:
    aspects: dict[str, Any] = {}
    for aspect in STANDARD_ASPECTS:
        a_groups = product_a.language_by_aspect.get(aspect, {})
        b_groups = product_b.language_by_aspect.get(aspect, {})
        if not isinstance(a_groups, Mapping):
            a_groups = {}
        if not isinstance(b_groups, Mapping):
            b_groups = {}
        comparison: dict[str, Any] = {}
        has_language = False
        for sentiment in ("positive", "negative"):
            left = copy.deepcopy(a_groups.get(sentiment, []))
            right = copy.deepcopy(b_groups.get(sentiment, []))
            if left or right:
                has_language = True
            comparison[sentiment] = {
                "product_a": left,
                "product_b": right,
                **compare_phrase_records(left, right),
            }
        if has_language:
            aspects[aspect] = comparison
    return {"matching_rule": "normalized_exact_phrase", "by_aspect": aspects}


def _rating_comparison(
    product_a: ProductAnalysis,
    product_b: ProductAnalysis,
    aspects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    a = product_a.rating_summary
    b = product_b.rating_summary
    if not a.get("available") or not b.get("available"):
        return {
            "available": False,
            "reason": "missing_rating",
            "product_a_available": bool(a.get("available")),
            "product_b_available": bool(b.get("available")),
        }
    by_aspect = [
        {
            "aspect": record["aspect"],
            "a_avg_rating": record["a_avg_rating"],
            "b_avg_rating": record["b_avg_rating"],
            "rating_gap": record["a_avg_rating"] - record["b_avg_rating"],
        }
        for record in aspects
        if record.get("a_avg_rating") is not None and record.get("b_avg_rating") is not None
    ]
    return {
        "available": True,
        "product_a": copy.deepcopy(a),
        "product_b": copy.deepcopy(b),
        "average_rating_gap": _number(a.get("average_rating")) - _number(b.get("average_rating")),
        "aspect_rating_comparison": by_aspect,
        "note": "Overall ratings are shown separately from Aspect sentiment.",
    }


def evidence_for_product(
    product: ProductAnalysis,
    aspect: str,
    sentiment: str,
    *,
    limit: int = COMPARISON_DEFAULT_EVIDENCE_LIMIT,
) -> list[dict[str, Any]]:
    buckets = product.aspect_evidence.get(aspect, {})
    records = buckets.get(sentiment, []) if isinstance(buckets, Mapping) else []
    if not isinstance(records, list) or limit <= 0:
        return []
    return copy.deepcopy([record for record in records if isinstance(record, Mapping)][:limit])


def _sample_warnings(product_a: ProductAnalysis, product_b: ProductAnalysis) -> list[dict[str, Any]]:
    a = product_a.successful_reviews
    b = product_b.successful_reviews
    warnings: list[dict[str, Any]] = []
    if min(a, b) <= COMPARISON_EARLY_SAMPLE_MAX_REVIEWS:
        warnings.append({"code": "early_comparison", "a_reviews": a, "b_reviews": b})
    smaller = min(a, b)
    larger = max(a, b)
    ratio = (larger / smaller) if smaller else (None if larger else 1.0)
    if ratio is None or ratio > COMPARISON_UNEVEN_SAMPLE_RATIO:
        warning: dict[str, Any] = {
            "code": "uneven_sample_sizes",
            "sample_size_ratio": ratio,
        }
        if ratio is None:
            warning["zero_sample_side"] = True
        warnings.append(warning)
    return warnings


def _recommendations(
    advantages: Sequence[Mapping[str, Any]],
    weaknesses: Sequence[Mapping[str, Any]],
    white_space: Sequence[Mapping[str, Any]],
    similar: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for record in advantages[:3]:
        recommendations.append(
            {
                "type": (
                    "monitor"
                    if str(record.get("comparison_status", "")).startswith("directional_signal")
                    else "protect_advantage"
                ),
                "aspect": record["aspect"],
            }
        )
    for record in weaknesses[:3]:
        recommendations.append(
            {
                "type": (
                    "monitor"
                    if str(record.get("comparison_status", "")).startswith("directional_signal")
                    else "improvement_priority"
                ),
                "aspect": record["aspect"],
            }
        )
    for record in white_space[:3]:
        recommendations.append({"type": "market_opportunity", "aspect": record["aspect"]})
    if not recommendations and similar:
        recommendations.append({"type": "monitor", "aspect": similar[0]["aspect"]})
    return recommendations


def _new_comparison_id(created_at: str) -> str:
    """Create a readable unique ID without deriving identity from product names."""

    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        timestamp = datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    timestamp = timestamp.astimezone(UTC)
    return f"CMP_{timestamp:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"


def build_comparison(
    product_a: ProductAnalysis,
    product_b: ProductAnalysis,
    *,
    generated_at: str | None = None,
    comparison_id: str | None = None,
) -> dict[str, Any]:
    """Build the complete Step 13 payload without mutating either product."""

    if (
        product_a.analysis_id
        and product_a.analysis_id == product_b.analysis_id
    ) or (
        product_a.product_id == product_b.product_id
        and product_a.analysis_run_id == product_b.analysis_run_id
    ):
        raise ComparisonError("Select two different product analyses.")
    if not analysis_versions_compatible(product_a, product_b):
        raise AnalysisVersionMismatch(
            "Analysis version mismatch. Both products must use "
            f"{ANALYSIS_CANDIDATE_VERSION} and the same taxonomy version."
        )
    early = min(product_a.successful_reviews, product_b.successful_reviews) <= COMPARISON_EARLY_SAMPLE_MAX_REVIEWS
    a_index = _metric_index(product_a)
    b_index = _metric_index(product_b)
    # This primary schema is intentionally fixed: every comparison has exactly
    # one row for every canonical Aspect, even when neither side has a mention.
    # Headline lists use a separately sorted view so the stable export order
    # does not alter Step 13 ranking semantics.
    records = [
        _aspect_comparison(
            aspect,
            a_index.get(aspect),
            b_index.get(aspect),
            early_comparison=early,
        )
        for aspect in STANDARD_ASPECTS
    ]
    ranked_records = sorted(records, key=comparison_sort_key)
    a_statuses = {"clear_advantage", "slight_advantage", "directional_signal_a"}
    b_statuses = {"clear_weakness", "slight_weakness", "directional_signal_b"}
    advantages = [
        copy.deepcopy(record)
        for record in ranked_records
        if record["business_category"] == "PRODUCT"
        and record["aspect"] != "Other"
        and record["comparison_status"] in a_statuses
    ]
    weaknesses = [
        copy.deepcopy(record)
        for record in ranked_records
        if record["business_category"] == "PRODUCT"
        and record["aspect"] != "Other"
        and record["comparison_status"] in b_statuses
    ]
    white_space = [
        copy.deepcopy(record)
        for record in records
        if record["business_category"] == "PRODUCT"
        and record["aspect"] != "Other"
        and record["a_review_count"] >= COMPARISON_MIN_SUPPORT
        and record["b_review_count"] >= COMPARISON_MIN_SUPPORT
        and record["a_negative_rate"] is not None
        and record["b_negative_rate"] is not None
        and record["a_negative_rate"] >= COMPARISON_WHITE_SPACE_NEGATIVE_RATE
        and record["b_negative_rate"] >= COMPARISON_WHITE_SPACE_NEGATIVE_RATE
    ]
    white_space.sort(
        key=lambda record: (
            -record["average_review_share"],
            -min(record["a_negative_rate"], record["b_negative_rate"]),
            record["aspect"].casefold(),
        )
    )
    shopping = [
        copy.deepcopy(record)
        for record in records
        if record["business_category"] == "EXPERIENCE"
    ]
    similar = [
        record
        for record in ranked_records
        if record["comparison_status"] == "similar"
    ]
    insufficient = [record for record in records if record["comparison_status"] == "insufficient_evidence"]
    warnings = _sample_warnings(product_a, product_b)
    summary_facts = {
        "product_a_advantages": [record["aspect"] for record in advantages[:3]],
        "product_b_advantages": [record["aspect"] for record in weaknesses[:3]],
        "shared_pain_points": [record["aspect"] for record in white_space[:3]],
    }
    created_at = generated_at or datetime.now(UTC).isoformat()
    result = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "comparison_id": comparison_id or _new_comparison_id(created_at),
        "product_a": product_a.public_metadata(),
        "product_b": product_b.public_metadata(),
        "analysis_compatibility": {
            "compatible": True,
            "required_candidate": ANALYSIS_CANDIDATE_VERSION,
            "taxonomy_version": product_a.taxonomy_version,
            "product_a_schema_version": product_a.analysis_schema_version or None,
            "product_b_schema_version": product_b.analysis_schema_version or None,
        },
        "sample_warnings": warnings,
        "comparison_mode": "early_comparison" if early else "standard_comparison",
        "head_to_head_summary": {
            "product_a_leads": len(advantages),
            "product_b_leads": len(weaknesses),
            "similar_aspects": len(similar),
            "insufficient_evidence_aspects": len(insufficient),
        },
        "executive_summary": summary_facts,
        "aspect_comparisons": records,
        "competitive_advantages": advantages,
        "competitive_weaknesses": weaknesses,
        "white_space_opportunities": white_space,
        "shopping_experience_comparison": shopping,
        "rating_comparison": _rating_comparison(product_a, product_b, records),
        "language_comparison": compare_customer_language(product_a, product_b),
        "recommendations": _recommendations(advantages, weaknesses, white_space, similar),
        "comparison_rules": {
            "basis": "review_share_and_sentiment_rates_not_raw_counts",
            "net_sentiment": "positive_rate - negative_rate",
            "minimum_support": COMPARISON_MIN_SUPPORT,
            "slight_gap_percent": COMPARISON_SLIGHT_GAP_PERCENT,
            "clear_gap_percent": COMPARISON_CLEAR_GAP_PERCENT,
            "uneven_sample_ratio": COMPARISON_UNEVEN_SAMPLE_RATIO,
            "white_space_negative_rate": COMPARISON_WHITE_SPACE_NEGATIVE_RATE,
            "significance_disclaimer": "Directional business rules, not statistical significance.",
        },
        "generated_at": created_at,
    }
    return result


def _validated_workspace_root(root: str | Path) -> Path:
    resolved = Path(root).expanduser().resolve()
    if resolved.name != "current" or resolved.parent.name != "comparison":
        raise ComparisonOutputSafetyError(
            "Comparison outputs may only be written to a comparison/current directory."
        )
    project_outputs = OUTPUTS_DIR.resolve()
    if project_outputs in resolved.parents and resolved != COMPARISON_CURRENT_DIR.resolve():
        raise ComparisonOutputSafetyError("The selected project output directory is protected.")
    return resolved


def _atomic_write_text(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_comparison_outputs(
    comparison: Mapping[str, Any],
    *,
    root: str | Path = COMPARISON_CURRENT_DIR,
) -> ComparisonWorkspacePaths:
    """Atomically publish only compact comparison artifacts."""

    directory = _validated_workspace_root(root)
    directory.mkdir(parents=True, exist_ok=True)
    paths = ComparisonWorkspacePaths.from_root(directory)
    payload = copy.deepcopy(dict(comparison))
    rows = payload.get("aspect_comparisons", [])
    if not isinstance(rows, list):
        raise ComparisonError("aspect_comparisons must be a list")
    frame = pd.DataFrame(rows)
    for column in COMPARISON_OUTPUT_COLUMNS:
        if column not in frame:
            frame[column] = None
    csv_text = frame.loc[:, COMPARISON_OUTPUT_COLUMNS].to_csv(index=False)
    metadata = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "comparison_id": payload.get("comparison_id"),
        "product_a_run_id": payload.get("product_a", {}).get("analysis_run_id"),
        "product_b_run_id": payload.get("product_b", {}).get("analysis_run_id"),
        "product_a_analysis_id": (
            payload.get("product_a", {}).get("analysis_id")
            or payload.get("product_a", {}).get("analysis_run_id")
        ),
        "product_b_analysis_id": (
            payload.get("product_b", {}).get("analysis_id")
            or payload.get("product_b", {}).get("analysis_run_id")
        ),
        # Stable aliases retained for consumers following the manifest naming
        # used in the Step 13.1 acceptance contract.
        "analysis_a_id": (
            payload.get("product_a", {}).get("analysis_id")
            or payload.get("product_a", {}).get("analysis_run_id")
        ),
        "analysis_b_id": (
            payload.get("product_b", {}).get("analysis_id")
            or payload.get("product_b", {}).get("analysis_run_id")
        ),
        "product_a": payload.get("product_a"),
        "product_b": payload.get("product_b"),
        "sample_sizes": {
            "product_a": payload.get("product_a", {}).get("successful_reviews"),
            "product_b": payload.get("product_b", {}).get("successful_reviews"),
        },
        "created_at": payload.get("generated_at"),
    }
    _atomic_write_text(
        paths.summary,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    _atomic_write_text(paths.aspect_comparison, csv_text)
    _atomic_write_text(
        paths.metadata,
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    return paths


def load_saved_comparison(
    root: str | Path = COMPARISON_CURRENT_DIR,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    paths = ComparisonWorkspacePaths.from_root(root)
    summary = _read_json_object(paths.summary)
    metadata = _read_json_object(paths.metadata)
    if not summary or not metadata or not paths.aspect_comparison.is_file():
        raise ComparisonError("No complete saved comparison is available.")
    try:
        frame = pd.read_csv(paths.aspect_comparison, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ComparisonError(f"Unable to read the saved Aspect comparison: {exc}") from exc
    return summary, frame, metadata


__all__ = [
    "COMPARISON_OUTPUT_COLUMNS",
    "COMPARISON_SCHEMA_VERSION",
    "SUPPORTED_ANALYSIS_MANIFEST_SCHEMAS",
    "FIXTURE_PATH",
    "AnalysisVersionMismatch",
    "ComparisonError",
    "ComparisonOutputSafetyError",
    "ComparisonWorkspacePaths",
    "ProductAnalysis",
    "analysis_versions_compatible",
    "build_comparison",
    "classify_comparison",
    "compare_customer_language",
    "compare_phrase_records",
    "comparison_sort_key",
    "evidence_for_product",
    "evidence_level",
    "load_fixture_product",
    "load_product_analysis",
    "load_saved_comparison",
    "save_comparison_outputs",
]
