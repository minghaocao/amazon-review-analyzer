"""Deterministic, offline HTML reports for completed review analyses.

Step 14 is a presentation layer over persisted Step 12 and Step 13 outputs.
This module never imports an LLM client, scraper, analyzer, aggregator, or
keyword analyzer.  It validates existing snapshots, reads their compact
artifacts, escapes all dynamic text, and emits a self-contained printable
HTML document plus a small metadata sidecar.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import html
import json
import math
import os
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from analysis_library import (
    REQUIRED_ANALYSIS_FILES,
    SavedAnalysis,
    SnapshotIntegrityError,
    validate_analysis,
)
from aspect_taxonomy import STANDARD_ASPECTS, business_category_for_aspect
from config import ANALYSIS_CANDIDATE_VERSION, REPORTS_DIR, REPORT_TEMPLATE_VERSION
from i18n import (
    display_analysis_source_type,
    display_aspect,
    display_business_category,
    display_comparison_status,
    display_evidence_level,
    display_quadrant,
    display_recommendation,
    display_sample_level,
    localized_comparison_summary,
    localized_executive_summary,
    localized_recommendation,
    normalize_language,
    t,
)
from visualization import load_dashboard_data


REPORT_METADATA_SCHEMA_VERSION = "step14.report-metadata.v1"
DEFAULT_REPORTS_DIR = REPORTS_DIR
MAX_REPORT_BYTES = 2_000_000
MAX_LANGUAGE_ITEMS = 10
MAX_EVIDENCE_PER_ASPECT = 3
MAX_HEADLINE_ITEMS = 5

_REPORT_ID_RE = re.compile(r"^(?:CMP_)?RPT_\d{8}_\d{6}_[a-f0-9]{4}$")
_LOCAL_PATH_RE = re.compile(
    r"(?:file://|/(?:Users|home|private|tmp|var)/|[A-Za-z]:\\\\Users\\\\)",
    re.IGNORECASE,
)


class ReportError(ValueError):
    """Base class for a safe, user-displayable report failure."""


class ReportSourceError(ReportError):
    """Required deterministic source artifacts are missing or invalid."""


class ReportIntegrityError(ReportError):
    """A source snapshot failed integrity validation."""


class ReportSizeError(ReportError):
    """The bounded report exceeded its configured V1 size guard."""


class ReportOutputError(ReportError):
    """A report could not be written safely."""


@dataclass(frozen=True)
class ReportDocument:
    """One in-memory report suitable for a Streamlit download button."""

    report_id: str
    report_type: str
    language: str
    html: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExportedReport:
    """The two files written for one immutable local report export."""

    root: Path
    html_path: Path
    metadata_path: Path
    document: ReportDocument


_REPORT_I18N_KEYS: dict[str, str] = {
    "single_title": "report.single.title",
    "comparison_title": "report.comparison.title",
    "source": "report.source",
    "reviews_analyzed": "report.reviews_analyzed",
    "coverage": "report.coverage",
    "sample_level": "report.sample_level",
    "analysis_date": "report.analysis_date",
    "report_language": "report.report_language",
    "english": "report.language.en",
    "chinese": "report.language.zh",
    "disclaimer": "report.disclaimer.single",
    "quick_scan": "report.disclaimer.quick_scan",
    "small_sample": "report.warning.small_sample",
    "early_comparison": "report.warning.early_comparison",
    "uneven_sample": "report.warning.uneven_samples",
    "executive_summary": "report.section.executive_summary",
    "dataset_quality": "report.section.dataset_quality",
    "reviews_imported": "report.reviews_imported",
    "successful": "report.successful",
    "failed": "report.failed",
    "skipped": "report.skipped",
    "overall_sentiment": "report.section.aspect_sentiment",
    "positive": "report.positive_rate",
    "negative": "report.negative_rate",
    "neutral": "report.neutral_rate",
    "mixed": "report.mixed_rate",
    "selling_points": "report.section.selling_points",
    "pain_points": "report.section.pain_points",
    "product_experience": "report.section.product_experience",
    "shopping_experience": "report.section.shopping_experience",
    "priority_matrix": "report.section.priority_matrix",
    "strengths": "report.priority.strengths",
    "strength": "report.priority.strengths",
    "improvement": "report.priority.improvement",
    "emerging": "report.priority.emerging",
    "monitor_minor": "report.priority.monitor",
    "aspect_performance": "report.section.aspect_performance",
    "aspect": "report.aspect",
    "review_share": "report.review_share",
    "positive_rate": "report.positive_rate",
    "negative_rate": "report.negative_rate",
    "review_count": "report.review_count",
    "opportunity_score": "report.opportunity_score",
    "average_rating": "report.average_rating",
    "evidence_level": "report.evidence_level",
    "no_evidence": "report.no_evidence",
    "not_available": "report.not_available",
    "mentioned_in": "report.mentioned_in",
    "rate_positive": "report.rate_positive",
    "rate_negative": "report.rate_negative",
    "rating_insights": "report.section.rating_insights",
    "rated_reviews": "report.rated_reviews",
    "hidden_complaints": "report.rating.hidden_complaints",
    "low_rating_strengths": "report.rating.positive_signals",
    "customer_language": "report.section.customer_language",
    "positive_language": "report.section.positive_language",
    "negative_language": "report.section.negative_language",
    "representative_evidence": "report.section.representative_evidence",
    "review_id": "report.review_id",
    "rating": "report.rating",
    "recommendations": "report.section.recommendations",
    "recommendation": "report.recommendation",
    "methodology": "report.section.methodology",
    "methodology_body": "report.methodology",
    "metadata": "report.section.analysis_metadata",
    "model": "report.model",
    "engine_version": "report.analysis_engine_version",
    "taxonomy_version": "report.taxonomy_version",
    "generated_at": "report.generated_at",
    "analysis_id": "report.analysis_id",
    "report_id": "report.report_id",
    "directional_note": "report.directional_note",
    "product_a": "report.product_a",
    "product_b": "report.product_b",
    "sample_size": "report.sample_size",
    "analysis_versions": "report.analysis_versions",
    "competitive_advantages": "report.section.competitive_advantages",
    "competitive_weaknesses": "report.section.competitive_weaknesses",
    "white_space": "report.section.white_space",
    "gap": "report.gap",
    "status": "report.status",
    "head_to_head": "report.section.head_to_head",
    "a_share": "report.a_share",
    "b_share": "report.b_share",
    "a_positive": "report.a_positive",
    "b_positive": "report.b_positive",
    "a_negative": "report.a_negative",
    "b_negative": "report.b_negative",
    "shared_phrases": "report.shared_phrases",
    "a_distinctive": "report.a_distinctive_phrases",
    "b_distinctive": "report.b_distinctive_phrases",
    "evidence_side_by_side": "report.section.evidence_side_by_side",
    "leads_summary": "report.leads_summary",
    "empty": "report.empty",
    "open_print": "report.pdf_instruction",
}


def report_t(key: str, language: object = "en", **values: object) -> str:
    """Resolve one report-only display phrase with safe English fallback."""

    lang = normalize_language(language)
    translation_key = _REPORT_I18N_KEYS.get(key, f"report.{key}")
    return t(translation_key, lang, **values)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _new_report_id(report_type: str, generated_at: datetime, token_factory: Callable[[], str] | None = None) -> str:
    prefix = "CMP_RPT" if report_type == "competitor_comparison" else "RPT"
    factory = token_factory or (lambda: secrets.token_hex(2))
    return f"{prefix}_{generated_at:%Y%m%d_%H%M%S}_{factory()}"


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _safe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: object) -> int:
    number = _safe_float(value)
    return max(0, int(number)) if number is not None else 0


def _percent(value: object, language: str, *, no_evidence: bool = False) -> str:
    if no_evidence:
        return report_t("no_evidence", language)
    number = _safe_float(value)
    if number is None:
        return report_t("not_available", language)
    return f"{number:.1f}%"


def _number(value: object, language: str, *, decimals: int = 1) -> str:
    number = _safe_float(value)
    if number is None:
        return report_t("not_available", language)
    return f"{number:.{decimals}f}"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportSourceError(f"Unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportSourceError(f"{label} must contain a JSON object.")
    return value


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(record) for record in csv.DictReader(handle)]
    except (FileNotFoundError, OSError, UnicodeError, csv.Error) as exc:
        raise ReportSourceError(f"Unable to read {label}: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReportSourceError(f"Unable to hash report source: {exc}") from exc
    return digest.hexdigest()


def _combined_hash(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(hashes[name]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _current_source_hash(root: Path) -> str:
    hashes: dict[str, str] = {}
    for filename in REQUIRED_ANALYSIS_FILES:
        path = root / filename
        if not path.is_file():
            raise ReportSourceError(f"Analysis source is missing required output: {filename}")
        hashes[filename] = _sha256(path)
    return _combined_hash(hashes)


def _validated_analysis_source(
    analysis_root: str | Path | SavedAnalysis,
    *,
    validate_integrity: bool,
) -> tuple[Path, dict[str, Any], str]:
    if isinstance(analysis_root, SavedAnalysis):
        root = analysis_root.root
    else:
        root = Path(analysis_root).expanduser().resolve(strict=False)
    manifest_path = root / "analysis_manifest.json"
    if validate_integrity:
        try:
            saved = validate_analysis(
                analysis_root,
                library_root=root.parent,
                verify_hashes=True,
            )
        except (SnapshotIntegrityError, ValueError) as exc:
            raise ReportIntegrityError("Saved analysis integrity check failed.") from exc
        return saved.root, copy.deepcopy(saved.manifest), str(saved.manifest["content_hash"])
    if not root.is_dir():
        raise ReportSourceError("Analysis source directory was not found.")
    manifest = _read_json(manifest_path, "analysis manifest") if manifest_path.is_file() else {}
    # Current Amazon workspaces predate the Analysis Library manifest.  Detect
    # their existing import metadata so the required Quick Scan limitation is
    # still disclosed without relying on a path name or rerunning a scraper.
    if not manifest and (
        (root / "scrape_metadata.json").is_file()
        or (root / "browser_import_metadata.json").is_file()
    ):
        manifest = {"source_type": "amazon"}
    return root, manifest, _current_source_hash(root)


def _validate_comparison_source(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    required = ("comparison_summary.json", "aspect_comparison.csv", "comparison_metadata.json")
    hashes: dict[str, str] = {}
    for filename in required:
        path = root / filename
        if not path.is_file():
            raise ReportSourceError(f"Comparison source is missing required output: {filename}")
        hashes[filename] = _sha256(path)
    return (
        _read_json(root / "comparison_summary.json", "comparison summary"),
        _read_json(root / "comparison_metadata.json", "comparison metadata"),
        _combined_hash(hashes),
    )


def _mapping_list(value: object, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    items = [copy.deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]
    return items if limit is None else items[: max(0, limit)]


def _section(title: str, body: str, *, section_id: str = "") -> str:
    identity = f' id="{_e(section_id)}"' if section_id else ""
    return f'<section class="report-section"{identity}><h2>{_e(title)}</h2>{body}</section>'


def _cards(items: Iterable[tuple[str, object]], *, compact: bool = False) -> str:
    card_class = "metric-card compact" if compact else "metric-card"
    return '<div class="metrics">' + "".join(
        f'<div class="{card_class}"><div class="metric-label">{_e(label)}</div>'
        f'<div class="metric-value">{_e(value)}</div></div>'
        for label, value in items
    ) + "</div>"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]], *, css_class: str = "") -> str:
    header = "".join(f"<th scope=\"col\">{_e(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_e(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-wrap {css_class}"><table><thead><tr>{header}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _empty(language: str) -> str:
    return f'<p class="muted">{_e(report_t("empty", language))}</p>'


_CSS = """
:root{color-scheme:light;--ink:#172033;--muted:#637083;--line:#dfe5ec;--soft:#f4f7fa;--blue:#1e5aa8;--green:#137a56;--red:#a33a3a;--amber:#9a6413}
*{box-sizing:border-box}html{background:#eef2f6}body{margin:0;color:var(--ink);background:white;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif;line-height:1.55}
.page{max-width:1080px;margin:0 auto;padding:54px 64px 80px}.cover{border-bottom:3px solid var(--blue);padding-bottom:28px;margin-bottom:34px}.eyebrow{color:var(--blue);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:12px}.cover h1{font-size:34px;line-height:1.15;margin:8px 0}.subtitle{font-size:20px;color:var(--muted);margin:0}.notice{border-left:4px solid var(--amber);background:#fff8e7;padding:12px 16px;margin:14px 0;border-radius:3px}.notice.info{border-color:var(--blue);background:#f2f7fd}.report-section{margin:38px 0;break-inside:avoid-page}.report-section h2{font-size:23px;margin:0 0 16px;padding-bottom:8px;border-bottom:1px solid var(--line)}h3{font-size:17px;margin:22px 0 10px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px}.metric-card,.signal-card,.evidence-card{border:1px solid var(--line);border-radius:8px;background:white;padding:16px;break-inside:avoid-page}.metric-card.compact{padding:12px}.metric-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.metric-value{font-size:22px;font-weight:700;margin-top:3px;overflow-wrap:anywhere;word-break:break-word}.signal-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}.signal-card h3{margin:0 0 8px}.signal-meta{font-size:13px;color:var(--muted)}.table-wrap{width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:7px}.table-wrap table{border-collapse:collapse;width:100%;font-size:13px}.table-wrap th,.table-wrap td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}.table-wrap th{background:var(--soft);font-weight:700}.table-wrap tr:last-child td{border-bottom:0}.evidence-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}.quote{font-size:14px;margin:5px 0 10px}.quote:before{content:"“";color:var(--blue);font-size:24px}.meta{font-size:12px;color:var(--muted)}.tag{display:inline-block;background:var(--soft);padding:2px 8px;margin:2px;border-radius:999px;font-size:12px}.muted{color:var(--muted)}.footer-note{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px;margin-top:36px}.print-help{font-size:12px;color:var(--muted);margin-top:20px}
@page{size:auto;margin:14mm}@media(max-width:720px){.page{padding:30px 22px}.cover h1{font-size:27px}.metrics,.signal-grid,.evidence-grid{grid-template-columns:1fr}}
@media print{html,body{background:white}.page{max-width:none;padding:0}.print-help,.no-print{display:none!important}.report-section,.metric-card,.signal-card,.evidence-card,.notice{break-inside:avoid-page;page-break-inside:avoid}.table-wrap{overflow:visible;border:0}table{break-inside:auto}thead{display:table-header-group}tr{break-inside:avoid;page-break-inside:avoid}a{color:inherit;text-decoration:none}}
"""


def _document_shell(title: str, product_line: str, body: str, language: str) -> str:
    lang_attr = "zh-CN" if language == "zh" else "en"
    return (
        "<!doctype html>\n"
        f'<html lang="{lang_attr}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)} — {_e(product_line)}</title><style>{_CSS}</style></head>"
        f'<body><main class="page">{body}<p class="print-help">{_e(report_t("open_print", language))}</p>'
        "</main></body></html>"
    )


def _assert_safe_html(rendered: str) -> None:
    encoded = rendered.encode("utf-8")
    if len(encoded) > MAX_REPORT_BYTES:
        raise ReportSizeError(
            f"Report exceeds the {MAX_REPORT_BYTES:,}-byte V1 size limit."
        )
    if _LOCAL_PATH_RE.search(rendered):
        raise ReportOutputError("Report contains a local file path.")
    if re.search(r"<(?:script|iframe|object|embed)\b", rendered, re.IGNORECASE):
        raise ReportOutputError("Report contains an unsafe executable element.")
    if re.search(r"<[^>]+\b(?:src|href)\s*=", rendered, re.IGNORECASE):
        raise ReportOutputError("Standalone reports cannot reference external resources.")


def _source_type_is_quick_scan(value: object) -> bool:
    normalized = str(value or "").strip().casefold().replace(" ", "_")
    return normalized in {"amazon", "amazon_quick_scan", "quick_scan"}


def _quality_from_source(
    manifest: Mapping[str, Any],
    business: Mapping[str, Any],
    analyzed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status_counts = {"success": 0, "failed": 0, "skipped": 0}
    for row in analyzed:
        status = str(row.get("analysis_status", "")).strip().casefold()
        if status in status_counts:
            status_counts[status] += 1
    data_quality = business.get("data_quality", {})
    if not isinstance(data_quality, Mapping):
        data_quality = {}
    imported = _safe_int(
        manifest.get("imported_reviews", data_quality.get("total_input_rows", len(analyzed)))
    )
    successful = _safe_int(
        manifest.get(
            "successful_reviews",
            data_quality.get("successful_reviews", business.get("analyzed_review_count", status_counts["success"])),
        )
    )
    failed = _safe_int(manifest.get("failed_reviews", data_quality.get("failed_reviews", status_counts["failed"])))
    skipped = _safe_int(manifest.get("skipped_reviews", data_quality.get("skipped_reviews", status_counts["skipped"])))
    coverage = _safe_float(manifest.get("coverage"))
    if coverage is None:
        coverage = successful / imported if imported else 0.0
    return {
        "imported": imported,
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "coverage": min(1.0, max(0.0, coverage)),
    }


def _sample_key(manifest: Mapping[str, Any], business: Mapping[str, Any], successful: int) -> str:
    raw = manifest.get("sample_level")
    if not raw:
        raw = business.get("sample_level")
    if isinstance(raw, Mapping):
        raw = raw.get("key")
    normalized = str(raw or "").strip().casefold().replace(" ", "_")
    if normalized in {"early_signals", "limited_sample", "full_insight"}:
        return normalized
    return "early_signals" if successful < 10 else ("limited_sample" if successful < 30 else "full_insight")


def _metric_records(business: Mapping[str, Any], aspect_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    source = business.get("aspect_business_metrics")
    records = _mapping_list(source)
    if records:
        return records
    result: list[dict[str, Any]] = []
    for row in aspect_rows:
        item = dict(row)
        aspect = str(item.get("aspect", "")).strip()
        if not aspect:
            continue
        item.setdefault("business_category", business_category_for_aspect(aspect))
        item.setdefault("review_count", item.get("aspect_review_count", 0))
        item.setdefault("review_share", item.get("aspect_share_of_reviews", 0.0))
        result.append(item)
    return result


def _single_signal_cards(records: Sequence[Mapping[str, Any]], language: str, kind: str) -> str:
    if not records:
        return _empty(language)
    cards: list[str] = []
    for record in records[:MAX_HEADLINE_ITEMS]:
        rate_key = "positive_rate" if kind == "positive" else "negative_rate"
        line_key = "rate_positive" if kind == "positive" else "rate_negative"
        details = [
            report_t("mentioned_in", language, share=_percent(record.get("review_share", record.get("aspect_share_of_reviews")), language)),
            report_t(line_key, language, rate=_percent(record.get(rate_key), language)),
            f'{report_t("review_count", language)}: {_safe_int(record.get("review_count", record.get("aspect_review_count")))}',
        ]
        if kind == "negative":
            details.append(f'{report_t("opportunity_score", language)}: {_number(record.get("opportunity_score"), language)}')
        details.append(
            f'{report_t("evidence_level", language)}: {display_evidence_level(record.get("evidence_level", "Low"), language)}'
        )
        cards.append(
            '<article class="signal-card"><h3>'
            + _e(display_aspect(record.get("aspect"), language))
            + '</h3><div class="signal-meta">'
            + "<br>".join(_e(item) for item in details)
            + "</div></article>"
        )
    return '<div class="signal-grid">' + "".join(cards) + "</div>"


def _sentiment_cards(summary: Mapping[str, Any], language: str) -> str:
    return _cards(
        [
            (report_t("positive", language), _percent(summary.get("positive_rate"), language)),
            (report_t("negative", language), _percent(summary.get("negative_rate"), language)),
            (report_t("neutral", language), _percent(summary.get("neutral_rate"), language)),
            (report_t("mixed", language), _percent(summary.get("mixed_rate"), language)),
        ],
        compact=True,
    )


def _experience_body(business: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]], language: str) -> str:
    product = business.get("product_experience_summary", {})
    shopping = business.get("shopping_experience_summary", {})
    if not isinstance(product, Mapping):
        product = {}
    if not isinstance(shopping, Mapping):
        shopping = {}
    experience_rows = [
        record for record in metrics if str(record.get("business_category", "")).upper() == "EXPERIENCE"
    ]
    shopping_table = _table(
        [report_t("aspect", language), report_t("review_share", language), report_t("positive_rate", language), report_t("negative_rate", language)],
        [
            [
                display_aspect(record.get("aspect"), language),
                _percent(record.get("review_share", record.get("aspect_share_of_reviews")), language),
                _percent(record.get("positive_rate"), language),
                _percent(record.get("negative_rate"), language),
            ]
            for record in experience_rows
        ],
    ) if experience_rows else _empty(language)
    return (
        f'<h3>{_e(report_t("product_experience", language))}</h3>{_sentiment_cards(product, language)}'
        f'<h3>{_e(report_t("shopping_experience", language))}</h3>{_sentiment_cards(shopping, language)}{shopping_table}'
    )


def _priority_body(records: Sequence[Mapping[str, Any]], language: str) -> str:
    groups: dict[str, list[str]] = {"strength": [], "improvement": [], "emerging": [], "monitor_minor": []}
    for record in records:
        quadrant = str(record.get("quadrant", "")).casefold()
        aspect = display_aspect(record.get("aspect"), language)
        if quadrant == "strength":
            groups["strength"].append(aspect)
        elif quadrant == "improvement priority":
            groups["improvement"].append(aspect)
        elif quadrant == "emerging concern":
            groups["emerging"].append(aspect)
        else:
            groups["monitor_minor"].append(aspect)
    return '<div class="signal-grid">' + "".join(
        '<article class="signal-card"><h3>' + _e(report_t(key, language)) + "</h3>"
        + ("".join(f'<span class="tag">{_e(value)}</span>' for value in values) if values else _empty(language))
        + "</article>"
        for key, values in groups.items()
    ) + "</div>"


def _aspect_table(metrics: Sequence[Mapping[str, Any]], language: str) -> str:
    index = {str(record.get("aspect", "")): record for record in metrics}
    rows: list[list[object]] = []
    for aspect in STANDARD_ASPECTS:
        if business_category_for_aspect(aspect) != "PRODUCT":
            continue
        record = index.get(aspect, {})
        count = _safe_int(record.get("review_count", record.get("aspect_review_count")))
        rows.append(
            [
                display_aspect(aspect, language),
                _percent(record.get("review_share", record.get("aspect_share_of_reviews")), language, no_evidence=count == 0),
                _percent(record.get("positive_rate"), language, no_evidence=count == 0),
                _percent(record.get("negative_rate"), language, no_evidence=count == 0),
                _number(record.get("avg_rating"), language),
                display_evidence_level(record.get("evidence_level", "no_evidence" if count == 0 else "Low"), language),
            ]
        )
    return _table(
        [report_t("aspect", language), report_t("review_share", language), report_t("positive_rate", language), report_t("negative_rate", language), report_t("average_rating", language), report_t("evidence_level", language)],
        rows,
        css_class="aspect-table",
    )


def _rating_body(rating: Mapping[str, Any], analyzed: Sequence[Mapping[str, Any]], language: str) -> str:
    ratings = [value for value in (_safe_float(row.get("rating")) for row in analyzed) if value is not None and 1 <= value <= 5]
    average = sum(ratings) / len(ratings) if ratings else None
    cards = _cards(
        [
            (report_t("average_rating", language), _number(average, language, decimals=2)),
            (report_t("rated_reviews", language), _safe_int(rating.get("rated_reviews", len(ratings)))),
        ]
    )
    hidden = _mapping_list(rating.get("hidden_complaints"), limit=MAX_HEADLINE_ITEMS)
    strengths = _mapping_list(rating.get("strengths_in_negative_reviews"), limit=MAX_HEADLINE_ITEMS)
    return cards + _table(
        [report_t("hidden_complaints", language), report_t("review_count", language)],
        [[display_aspect(item.get("aspect"), language), _safe_int(item.get("review_count"))] for item in hidden],
    ) + _table(
        [report_t("low_rating_strengths", language), report_t("review_count", language)],
        [[display_aspect(item.get("aspect"), language), _safe_int(item.get("review_count"))] for item in strengths],
    )


def _keyword_terms(sentiment_keywords: Mapping[str, Any], sentiment: str) -> list[str]:
    raw = sentiment_keywords.get(sentiment, [])
    if isinstance(raw, Mapping):
        for key in ("keywords", "phrases", "terms", "top_terms"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    result: list[str] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, Mapping):
                term = item.get("term", item.get("keyword", item.get("phrase")))
            else:
                term = item
            text = str(term or "").strip()
            if text and text not in result:
                result.append(text)
            if len(result) >= MAX_LANGUAGE_ITEMS:
                break
    return result


def _language_body(sentiment_keywords: Mapping[str, Any], language: str) -> str:
    blocks: list[str] = []
    for sentiment, label in (("positive", "positive_language"), ("negative", "negative_language")):
        terms = _keyword_terms(sentiment_keywords, sentiment)
        content = "".join(f'<span class="tag">{_e(term)}</span>' for term in terms) if terms else _empty(language)
        blocks.append(f'<article class="signal-card"><h3>{_e(report_t(label, language))}</h3>{content}</article>')
    return '<div class="signal-grid">' + "".join(blocks) + "</div>"


def _evidence_records(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    sentiment: str,
) -> list[tuple[str, dict[str, Any]]]:
    output: list[tuple[str, dict[str, Any]]] = []
    for record in records[:MAX_HEADLINE_ITEMS]:
        aspect = str(record.get("aspect", ""))
        buckets = evidence.get(aspect, {})
        values = buckets.get(sentiment, []) if isinstance(buckets, Mapping) else []
        for item in _mapping_list(values, limit=MAX_EVIDENCE_PER_ASPECT):
            if str(item.get("evidence", "")).strip():
                output.append((aspect, item))
    return output


def _evidence_body(evidence: Mapping[str, Any], selling: Sequence[Mapping[str, Any]], pain: Sequence[Mapping[str, Any]], language: str) -> str:
    records = _evidence_records(evidence, selling, "positive") + _evidence_records(evidence, pain, "negative")
    if not records:
        return _empty(language)
    cards: list[str] = []
    for aspect, record in records:
        meta = f'{report_t("review_id", language)}: {record.get("review_id") or report_t("not_available", language)}'
        rating = _safe_float(record.get("rating"))
        if rating is not None:
            meta += f' · {report_t("rating", language)}: {rating:g}'
        cards.append(
            '<article class="evidence-card"><strong>' + _e(display_aspect(aspect, language)) + '</strong>'
            + '<p class="quote">' + _e(record.get("evidence")) + '</p><div class="meta">' + _e(meta) + "</div></article>"
        )
    return '<div class="evidence-grid">' + "".join(cards) + "</div>"


def _recommendations_body(records: Sequence[Mapping[str, Any]], language: str) -> str:
    if not records:
        return _empty(language)
    rows: list[list[str]] = []
    for record in records[:10]:
        localized = localized_recommendation(record, language)
        rows.append(
            [
                display_aspect(record.get("aspect"), language),
                localized.get("type", display_recommendation(record.get("type"), language)),
                localized.get("message", ""),
                display_evidence_level(record.get("evidence_level", "Low"), language),
            ]
        )
    return _table(
        [report_t("aspect", language), report_t("recommendation", language), report_t("recommendation", language), report_t("evidence_level", language)],
        rows,
    )


def _single_metadata_body(manifest: Mapping[str, Any], report_id: str, generated_at: str, language: str) -> str:
    rows = [
        [report_t("model", language), manifest.get("model_name") or "qwen3:8b"],
        [
            report_t("engine_version", language),
            manifest.get("analysis_candidate") or ANALYSIS_CANDIDATE_VERSION,
        ],
        [
            report_t("taxonomy_version", language),
            manifest.get("taxonomy_version") or ANALYSIS_CANDIDATE_VERSION,
        ],
        [report_t("generated_at", language), generated_at],
        [report_t("analysis_id", language), manifest.get("analysis_id") or manifest.get("analysis_run_id") or "current"],
        [report_t("report_id", language), report_id],
    ]
    return _table([report_t("metadata", language), ""], rows)


def _make_document(
    *,
    report_id: str,
    report_type: str,
    language: str,
    html_text: str,
    generated_at: str,
    source_analysis_id: object,
    source_snapshot_hash: str,
    comparison_id: str | None = None,
) -> ReportDocument:
    if not _REPORT_ID_RE.fullmatch(report_id):
        raise ReportError("Invalid report ID.")
    _assert_safe_html(html_text)
    metadata: dict[str, Any] = {
        "report_id": report_id,
        "report_type": report_type,
        "schema_version": REPORT_METADATA_SCHEMA_VERSION,
        "template_version": REPORT_TEMPLATE_VERSION,
        "language": language,
        "source_analysis_id": copy.deepcopy(source_analysis_id),
        "comparison_id": comparison_id,
        "generated_at": generated_at,
        "source_snapshot_hash": source_snapshot_hash,
        "file_hash": _sha256_bytes(html_text.encode("utf-8")),
    }
    return ReportDocument(report_id, report_type, language, html_text, metadata)


def build_single_product_report(
    analysis_root: str | Path | SavedAnalysis,
    *,
    language: str = "en",
    product_name: str | None = None,
    generated_at: datetime | None = None,
    report_id: str | None = None,
    validate_integrity: bool = True,
) -> ReportDocument:
    """Build one bounded standalone report from persisted single-product outputs."""

    lang = normalize_language(language)
    root, manifest, source_hash = _validated_analysis_source(
        analysis_root, validate_integrity=validate_integrity
    )
    persisted_business = _read_json(root / "business_summary.json", "business summary")
    evidence = _read_json(root / "aspect_evidence.json", "Aspect Evidence")
    sentiment_keywords = _read_json(root / "sentiment_keywords.json", "sentiment keywords")
    analyzed = _read_csv(root / "analyzed_reviews.csv", "analyzed reviews")
    aspect_rows = _read_csv(root / "aspect_summary.csv", "Aspect summary")
    # Step 12 is intentionally calculated by the dashboard as a deterministic
    # view over the seven persisted outputs.  Reuse that same read-only view so
    # the exported report cannot drift from the on-screen Business Intelligence
    # while still avoiding Aggregator, Keyword Analyzer, model, and network work.
    dashboard = load_dashboard_data(root)
    business = dashboard.business_intelligence
    if not isinstance(business, Mapping):
        raise ReportSourceError(
            "Business Intelligence could not be derived from the saved outputs."
        )

    now = generated_at or _utc_now()
    generated = _iso_utc(now)
    identity = report_id or _new_report_id("single_product", now)
    name = str(product_name or manifest.get("product_name") or "Product").strip() or "Product"
    source_type = manifest.get("source_type") or "unknown"
    quality = _quality_from_source(manifest, persisted_business, analyzed)
    sample = _sample_key(manifest, business, quality["successful"])
    metrics = _metric_records(business, aspect_rows)
    selling = _mapping_list(business.get("selling_points"), limit=MAX_HEADLINE_ITEMS)
    pain = _mapping_list(business.get("pain_points"), limit=MAX_HEADLINE_ITEMS)
    matrix = _mapping_list(business.get("priority_matrix"))
    overall = business.get("overall_sentiment", {})
    overall = overall if isinstance(overall, Mapping) else {}
    rating = business.get("rating_insights", {})
    rating = rating if isinstance(rating, Mapping) else {}

    header_meta = _cards(
        [
            (report_t("source", lang), display_analysis_source_type(source_type, lang)),
            (report_t("reviews_analyzed", lang), quality["successful"]),
            (report_t("coverage", lang), _percent(quality["coverage"] * 100, lang)),
            (report_t("sample_level", lang), display_sample_level(sample, lang)),
            (report_t("analysis_date", lang), manifest.get("saved_at") or business.get("generated_at") or generated),
            (report_t("report_language", lang), report_t("chinese" if lang == "zh" else "english", lang)),
        ]
    )
    cover = (
        '<header class="cover"><div class="eyebrow">Amazon Review Analyzer</div>'
        f'<h1>{_e(report_t("single_title", lang))}</h1><p class="subtitle">{_e(name)}</p>{header_meta}</header>'
        f'<div class="notice info">{_e(report_t("disclaimer", lang))}</div>'
    )
    if _source_type_is_quick_scan(source_type):
        cover += f'<div class="notice">{_e(report_t("quick_scan", lang))}</div>'
    if sample == "early_signals":
        cover += f'<div class="notice">{_e(report_t("small_sample", lang))}</div>'

    body = cover
    body += _section(
        report_t("executive_summary", lang),
        f'<p>{_e(localized_executive_summary(business, lang))}</p>',
    )
    body += _section(
        report_t("dataset_quality", lang),
        _cards(
            [
                (report_t("reviews_imported", lang), quality["imported"]),
                (report_t("successful", lang), quality["successful"]),
                (report_t("failed", lang), quality["failed"]),
                (report_t("skipped", lang), quality["skipped"]),
                (report_t("coverage", lang), _percent(quality["coverage"] * 100, lang)),
            ]
        ),
    )
    body += _section(report_t("overall_sentiment", lang), _sentiment_cards(overall, lang))
    body += _section(report_t("selling_points", lang), _single_signal_cards(selling, lang, "positive"))
    body += _section(report_t("pain_points", lang), _single_signal_cards(pain, lang, "negative"))
    body += _section(
        f'{report_t("product_experience", lang)} / {report_t("shopping_experience", lang)}',
        _experience_body(business, metrics, lang),
    )
    body += _section(report_t("priority_matrix", lang), _priority_body(matrix, lang))
    body += _section(report_t("aspect_performance", lang), _aspect_table(metrics, lang))
    if bool(rating.get("available")):
        body += _section(report_t("rating_insights", lang), _rating_body(rating, analyzed, lang))
    body += _section(report_t("customer_language", lang), _language_body(sentiment_keywords, lang))
    body += _section(report_t("representative_evidence", lang), _evidence_body(evidence, selling, pain, lang))
    body += _section(report_t("recommendations", lang), _recommendations_body(_mapping_list(business.get("recommendations")), lang))
    body += f'<div class="notice info">{_e(report_t("directional_note", lang))}</div>'
    body += _section(report_t("methodology", lang), f'<p>{_e(report_t("methodology_body", lang))}</p>')
    body += _section(report_t("metadata", lang), _single_metadata_body(manifest, identity, generated, lang))

    rendered = _document_shell(report_t("single_title", lang), name, body, lang)
    return _make_document(
        report_id=identity,
        report_type="single_product",
        language=lang,
        html_text=rendered,
        generated_at=generated,
        source_analysis_id=manifest.get("analysis_id") or manifest.get("analysis_run_id") or "current",
        source_snapshot_hash=source_hash,
    )


def _comparison_listing(records: Sequence[Mapping[str, Any]], language: str, mode: str) -> str:
    if not records:
        return _empty(language)
    rows: list[list[object]] = []
    for record in records[:MAX_HEADLINE_ITEMS]:
        if mode == "white_space":
            rows.append(
                [
                    display_aspect(record.get("aspect"), language),
                    _percent(record.get("a_negative_rate"), language),
                    _percent(record.get("b_negative_rate"), language),
                    _percent(record.get("average_review_share"), language),
                    display_evidence_level(record.get("comparison_evidence"), language),
                ]
            )
        else:
            rows.append(
                [
                    display_aspect(record.get("aspect"), language),
                    _percent(record.get("a_positive_rate"), language),
                    _percent(record.get("b_positive_rate"), language),
                    _percent(record.get("positive_rate_gap"), language),
                    display_evidence_level(record.get("comparison_evidence"), language),
                ]
            )
    headers = (
        [report_t("aspect", language), report_t("a_negative", language), report_t("b_negative", language), report_t("review_share", language), report_t("evidence_level", language)]
        if mode == "white_space"
        else [report_t("aspect", language), report_t("a_positive", language), report_t("b_positive", language), report_t("gap", language), report_t("evidence_level", language)]
    )
    return _table(headers, rows)


def _comparison_head_to_head(records: Sequence[Mapping[str, Any]], language: str) -> str:
    index = {str(record.get("aspect", "")): record for record in records}
    rows: list[list[object]] = []
    for aspect in STANDARD_ASPECTS:
        record = index.get(aspect, {})
        a_missing = _safe_int(record.get("a_review_count")) == 0
        b_missing = _safe_int(record.get("b_review_count")) == 0
        rows.append(
            [
                display_aspect(aspect, language),
                _percent(record.get("a_review_share"), language, no_evidence=a_missing),
                _percent(record.get("b_review_share"), language, no_evidence=b_missing),
                _percent(record.get("a_positive_rate"), language, no_evidence=a_missing),
                _percent(record.get("b_positive_rate"), language, no_evidence=b_missing),
                display_comparison_status(record.get("comparison_status", "insufficient_evidence"), language),
            ]
        )
    return _table(
        [report_t("aspect", language), report_t("a_share", language), report_t("b_share", language), report_t("a_positive", language), report_t("b_positive", language), report_t("status", language)],
        rows,
        css_class="comparison-table",
    )


def _comparison_experience(records: Sequence[Mapping[str, Any]], language: str) -> str:
    grouped = {"PRODUCT": [], "EXPERIENCE": []}
    for record in records:
        category = str(record.get("business_category") or business_category_for_aspect(str(record.get("aspect", "")))).upper()
        if category in grouped:
            grouped[category].append(record)
    parts: list[str] = []
    for category, key in (("PRODUCT", "product_experience"), ("EXPERIENCE", "shopping_experience")):
        rows = [
            [
                display_aspect(item.get("aspect"), language),
                _percent(item.get("a_positive_rate"), language, no_evidence=_safe_int(item.get("a_review_count")) == 0),
                _percent(item.get("b_positive_rate"), language, no_evidence=_safe_int(item.get("b_review_count")) == 0),
                display_comparison_status(item.get("comparison_status"), language),
            ]
            for item in grouped[category]
        ]
        parts.append(f'<h3>{_e(report_t(key, language))}</h3>' + _table([report_t("aspect", language), report_t("a_positive", language), report_t("b_positive", language), report_t("status", language)], rows))
    return "".join(parts)


def _language_comparison_body(payload: Mapping[str, Any], language: str) -> str:
    shared: list[str] = []
    left: list[str] = []
    right: list[str] = []
    by_aspect = payload.get("by_aspect", {})
    if isinstance(by_aspect, Mapping):
        for values in by_aspect.values():
            if not isinstance(values, Mapping):
                continue
            for sentiment in ("positive", "negative"):
                group = values.get(sentiment, {})
                if not isinstance(group, Mapping):
                    continue
                for target, key in ((shared, "shared"), (left, "product_a_distinctive"), (right, "product_b_distinctive")):
                    for item in _mapping_list(group.get(key)):
                        term = str(item.get("term", item.get("normalized_term", ""))).strip()
                        if term and term not in target:
                            target.append(term)
    blocks = []
    for key, values in (("shared_phrases", shared), ("a_distinctive", left), ("b_distinctive", right)):
        content = "".join(f'<span class="tag">{_e(item)}</span>' for item in values[:MAX_LANGUAGE_ITEMS]) if values else _empty(language)
        blocks.append(f'<article class="signal-card"><h3>{_e(report_t(key, language))}</h3>{content}</article>')
    return '<div class="signal-grid">' + "".join(blocks) + "</div>"


def _load_evidence(root: Path) -> dict[str, Any]:
    return _read_json(root / "aspect_evidence.json", "Aspect Evidence")


def _comparison_evidence_body(
    summary: Mapping[str, Any],
    evidence_a: Mapping[str, Any],
    evidence_b: Mapping[str, Any],
    language: str,
) -> str:
    important = (
        _mapping_list(summary.get("competitive_advantages"), limit=2)
        + _mapping_list(summary.get("competitive_weaknesses"), limit=2)
        + _mapping_list(summary.get("white_space_opportunities"), limit=2)
    )
    seen: set[str] = set()
    cards: list[str] = []
    for record in important:
        aspect = str(record.get("aspect", ""))
        if not aspect or aspect in seen:
            continue
        seen.add(aspect)
        for side, source in ((report_t("product_a", language), evidence_a), (report_t("product_b", language), evidence_b)):
            buckets = source.get(aspect, {})
            if not isinstance(buckets, Mapping):
                continue
            candidates: list[dict[str, Any]] = []
            for sentiment in ("positive", "negative"):
                candidates.extend(_mapping_list(buckets.get(sentiment), limit=1))
            if candidates:
                item = candidates[0]
                cards.append(
                    '<article class="evidence-card"><strong>'
                    + _e(f"{side} · {display_aspect(aspect, language)}")
                    + '</strong><p class="quote">' + _e(item.get("evidence"))
                    + '</p><div class="meta">' + _e(f'{report_t("review_id", language)}: {item.get("review_id") or report_t("not_available", language)}')
                    + "</div></article>"
                )
    return '<div class="evidence-grid">' + "".join(cards) + "</div>" if cards else _empty(language)


def _comparison_metadata_body(summary: Mapping[str, Any], report_id: str, generated_at: str, language: str) -> str:
    compatibility = summary.get("analysis_compatibility", {})
    if not isinstance(compatibility, Mapping):
        compatibility = {}
    rows = [
        [report_t("engine_version", language), compatibility.get("required_candidate") or "unknown"],
        [report_t("taxonomy_version", language), compatibility.get("taxonomy_version") or "unknown"],
        [report_t("generated_at", language), generated_at],
        [t("report.comparison_id", language), summary.get("comparison_id") or "unknown"],
        [report_t("report_id", language), report_id],
    ]
    return _table([report_t("metadata", language), ""], rows)


def build_comparison_report(
    comparison_root: str | Path,
    *,
    product_a_root: str | Path | SavedAnalysis,
    product_b_root: str | Path | SavedAnalysis,
    language: str = "en",
    generated_at: datetime | None = None,
    report_id: str | None = None,
    validate_integrity: bool = True,
) -> ReportDocument:
    """Build a bounded comparison report without recalculating Step 13 logic."""

    lang = normalize_language(language)
    comparison_path = Path(comparison_root).expanduser().resolve(strict=False)
    summary, comparison_metadata, comparison_hash = _validate_comparison_source(comparison_path)
    root_a, manifest_a, hash_a = _validated_analysis_source(product_a_root, validate_integrity=validate_integrity)
    root_b, manifest_b, hash_b = _validated_analysis_source(product_b_root, validate_integrity=validate_integrity)

    expected_a = comparison_metadata.get("product_a_analysis_id") or comparison_metadata.get("analysis_a_id")
    expected_b = comparison_metadata.get("product_b_analysis_id") or comparison_metadata.get("analysis_b_id")
    actual_a = manifest_a.get("analysis_id") or manifest_a.get("analysis_run_id")
    actual_b = manifest_b.get("analysis_id") or manifest_b.get("analysis_run_id")
    if validate_integrity and ((expected_a and actual_a != expected_a) or (expected_b and actual_b != expected_b)):
        raise ReportIntegrityError("Comparison source does not match the selected saved analyses.")

    product_a = summary.get("product_a", {})
    product_b = summary.get("product_b", {})
    product_a = product_a if isinstance(product_a, Mapping) else {}
    product_b = product_b if isinstance(product_b, Mapping) else {}
    name_a = str(product_a.get("product_name") or manifest_a.get("product_name") or report_t("product_a", lang))
    name_b = str(product_b.get("product_name") or manifest_b.get("product_name") or report_t("product_b", lang))
    success_a = _safe_int(product_a.get("successful_reviews", manifest_a.get("successful_reviews")))
    success_b = _safe_int(product_b.get("successful_reviews", manifest_b.get("successful_reviews")))
    coverage_a = _safe_float(product_a.get("coverage", manifest_a.get("coverage")))
    coverage_b = _safe_float(product_b.get("coverage", manifest_b.get("coverage")))
    records = _mapping_list(summary.get("aspect_comparisons"))
    if len(records) != len(STANDARD_ASPECTS):
        raise ReportSourceError("Comparison report requires all 15 canonical Aspects.")

    now = generated_at or _utc_now()
    generated = _iso_utc(now)
    identity = report_id or _new_report_id("competitor_comparison", now)
    cover = (
        '<header class="cover"><div class="eyebrow">Amazon Review Analyzer</div>'
        f'<h1>{_e(report_t("comparison_title", lang))}</h1>'
        f'<p class="subtitle">{_e(name_a)} <span class="muted">vs</span> {_e(name_b)}</p>'
        + _cards(
            [
                (f'{report_t("product_a", lang)} · {report_t("sample_size", lang)}', success_a),
                (f'{report_t("product_b", lang)} · {report_t("sample_size", lang)}', success_b),
                (f'{report_t("product_a", lang)} · {report_t("source", lang)}', display_analysis_source_type(product_a.get("source_type") or manifest_a.get("source_type"), lang)),
                (f'{report_t("product_b", lang)} · {report_t("source", lang)}', display_analysis_source_type(product_b.get("source_type") or manifest_b.get("source_type"), lang)),
                (f'{report_t("product_a", lang)} · {report_t("coverage", lang)}', _percent((coverage_a or 0) * 100, lang)),
                (f'{report_t("product_b", lang)} · {report_t("coverage", lang)}', _percent((coverage_b or 0) * 100, lang)),
                (report_t("analysis_versions", lang), summary.get("analysis_compatibility", {}).get("taxonomy_version", "unknown")),
            ]
        )
        + "</header>"
    )
    warnings = _mapping_list(summary.get("sample_warnings"))
    warning_codes = {str(item.get("code", "")) for item in warnings}
    if "early_comparison" in warning_codes or min(success_a, success_b) < 10:
        cover += f'<div class="notice">{_e(report_t("early_comparison", lang))}</div>'
    ratio = max(success_a, success_b) / min(success_a, success_b) if min(success_a, success_b) else math.inf
    if "uneven_sample_sizes" in warning_codes or ratio > 3:
        cover += f'<div class="notice">{_e(report_t("uneven_sample", lang))}</div>'

    head = summary.get("head_to_head_summary", {})
    head = head if isinstance(head, Mapping) else {}
    summary_text = localized_comparison_summary(summary, lang)
    summary_text += " " + report_t(
        "leads_summary",
        lang,
        a=name_a,
        b=name_b,
        a_count=_safe_int(head.get("product_a_leads")),
        b_count=_safe_int(head.get("product_b_leads")),
    )
    body = cover
    body += _section(report_t("executive_summary", lang), f'<p>{_e(summary_text)}</p>')
    body += f'<div class="notice info">{_e(t("report.no_overall_winner", lang))}</div>'
    body += _section(report_t("competitive_advantages", lang), _comparison_listing(_mapping_list(summary.get("competitive_advantages")), lang, "advantage"))
    body += _section(report_t("competitive_weaknesses", lang), _comparison_listing(_mapping_list(summary.get("competitive_weaknesses")), lang, "weakness"))
    body += _section(report_t("white_space", lang), _comparison_listing(_mapping_list(summary.get("white_space_opportunities")), lang, "white_space"))
    body += _section(report_t("head_to_head", lang), _comparison_head_to_head(records, lang))
    body += _section(
        f'{report_t("product_experience", lang)} / {report_t("shopping_experience", lang)}',
        _comparison_experience(records, lang),
    )
    body += _section(report_t("customer_language", lang), _language_comparison_body(summary.get("language_comparison", {}) if isinstance(summary.get("language_comparison"), Mapping) else {}, lang))
    body += _section(report_t("evidence_side_by_side", lang), _comparison_evidence_body(summary, _load_evidence(root_a), _load_evidence(root_b), lang))
    body += f'<div class="notice info">{_e(report_t("directional_note", lang))}</div>'
    body += _section(report_t("methodology", lang), f'<p>{_e(report_t("methodology_body", lang))}</p>')
    body += _section(report_t("metadata", lang), _comparison_metadata_body(summary, identity, generated, lang))

    rendered = _document_shell(report_t("comparison_title", lang), f"{name_a} vs {name_b}", body, lang)
    source_hash = _combined_hash({"comparison": comparison_hash, "product_a": hash_a, "product_b": hash_b})
    return _make_document(
        report_id=identity,
        report_type="competitor_comparison",
        language=lang,
        html_text=rendered,
        generated_at=generated,
        source_analysis_id={"product_a": actual_a or expected_a or "current_a", "product_b": actual_b or expected_b or "current_b"},
        source_snapshot_hash=source_hash,
        comparison_id=str(summary.get("comparison_id") or comparison_metadata.get("comparison_id") or ""),
    )


def _safe_output_root(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve(strict=False)
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise ReportOutputError("Report output root must be a real directory.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        raise ReportOutputError(f"Unable to write report: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def export_report(
    document: ReportDocument,
    *,
    output_root: str | Path = DEFAULT_REPORTS_DIR,
) -> ExportedReport:
    """Atomically persist one already-built report without rerunning analysis."""

    if not isinstance(document, ReportDocument):
        raise TypeError("document must be a ReportDocument")
    _assert_safe_html(document.html)
    if document.metadata.get("file_hash") != _sha256_bytes(document.html.encode("utf-8")):
        raise ReportIntegrityError("In-memory report content does not match its metadata hash.")
    root = _safe_output_root(output_root)
    destination = root / document.report_id
    if destination.exists():
        raise ReportOutputError("A report with this ID already exists.")
    temporary = root / f".tmp-{document.report_id.casefold()}-{secrets.token_hex(4)}"
    if temporary.parent != root or destination.parent != root:
        raise ReportOutputError("Unsafe report destination.")
    try:
        temporary.mkdir(mode=0o700)
        _atomic_write(temporary / "report.html", document.html)
        metadata_text = json.dumps(document.metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        _atomic_write(temporary / "report_metadata.json", metadata_text)
        os.replace(temporary, destination)
    except Exception:
        if temporary.is_dir():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()
        raise
    return ExportedReport(
        root=destination,
        html_path=destination / "report.html",
        metadata_path=destination / "report_metadata.json",
        document=document,
    )


__all__ = [
    "REPORT_METADATA_SCHEMA_VERSION",
    "REPORT_TEMPLATE_VERSION",
    "DEFAULT_REPORTS_DIR",
    "MAX_REPORT_BYTES",
    "ReportError",
    "ReportSourceError",
    "ReportIntegrityError",
    "ReportSizeError",
    "ReportOutputError",
    "ReportDocument",
    "ExportedReport",
    "report_t",
    "build_single_product_report",
    "build_comparison_report",
    "export_report",
]
