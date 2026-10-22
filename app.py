"""Streamlit dashboard with isolated full-analysis and Amazon Quick Scan flows."""

from __future__ import annotations

import html
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis_library as library
import analysis_runtime as runtime
import amazon_analysis as amazon
import comparison_ui as comparison_dashboard
import interactive_analysis as interactive
from environment_health import run_environment_health, sanitize_diagnostics
from first_run import (
    DEFAULT_FIRST_RUN_STATE_PATH,
    is_first_run_completed,
    mark_first_run_completed,
)
from i18n import (
    DEFAULT_LANGUAGE,
    OLLAMA_MODEL,
    SUPPORTED_LANGUAGES,
    display_analysis_source_type,
    display_aspect,
    display_business_category,
    display_duration,
    display_evidence_level,
    display_quadrant,
    display_recommendation,
    display_sample_level,
    display_sentiment,
    display_status,
    localized_executive_summary,
    localized_error_message,
    localized_recommendation,
    t,
    t_count,
)
from amazon_url import AmazonURLValidationError, parse_amazon_url
from browser_amazon_import import (
    AmazonBrowserImportResult,
    AmazonBrowserSession,
    BrowserImportStatus,
    DEFAULT_NAVIGATION_TIMEOUT_MS,
    MAX_AMAZON_QUICK_SCAN_REVIEWS,
    MAX_BROWSER_REVIEW_PAGES,
    ReviewAvailabilityStatus,
    open_amazon_product,
)
from scraper import (
    TRUSTED_CA_BUNDLE_UNAVAILABLE,
    ScrapeStatus,
    scrape_amazon_reviews,
)
from visualization import (
    DEFAULT_OUTPUTS_DIR,
    DashboardData,
    DEFAULT_EVIDENCE_DISPLAY_LIMIT,
    EXPANDED_EVIDENCE_DISPLAY_LIMIT,
    apply_review_quick_filter,
    aspect_language_by_sentiment,
    calculate_kpis,
    deterministic_business_implications,
    evidence_for_aspect,
    evidence_with_review_metadata,
    filter_aspect_summary,
    filter_reviews,
    load_dashboard_data,
    output_fingerprint,
    review_filter_options,
    top_term_records,
)
from release_metadata import get_app_version


ACCENT = "#1F6F78"
NAVY = "#17324D"
POSITIVE = "#2E8B74"
NEGATIVE = "#D45D5D"
NEUTRAL = "#8A94A6"
MIXED = "#D5A63C"
CATEGORY_COLORS = {
    "Product Aspect": "#2D6CDF",
    "Shopping Experience": "#D98C32",
    "Other": "#8A94A6",
}
SENTIMENT_COLORS = {
    "Positive": POSITIVE,
    "Negative": NEGATIVE,
    "Neutral": NEUTRAL,
    "Mixed": MIXED,
}
UPLOAD_WORKSPACE = "Current Upload/Paste Analysis"
AMAZON_WORKSPACE = "Current Amazon Analysis"
AMAZON_QUICK_SCAN_SOURCE = "Amazon Quick Scan"
UPLOAD_DATASET_SOURCE = "Upload Dataset"
PASTE_REVIEWS_SOURCE = "Paste Reviews"
COMPARE_PRODUCTS_MODE = "Compare Products"
SAVED_ANALYSES_MODE = "Saved Analyses"
AMAZON_CHALLENGE_REASON = "amazon_challenge"
PRESERVED_AMAZON_URL_KEY = "preserved_amazon_product_url"
AMAZON_BROWSER_SESSION_KEY = "amazon_browser_session"
AMAZON_BROWSER_RESULT_KEY = "amazon_browser_result"
AMAZON_CHALLENGE_MESSAGE = (
    "Amazon currently prevents automated access to this product page.\n\n"
    "You can still analyze this product by pasting reviews or uploading a CSV."
)
AMAZON_QUICK_SCAN_DISCLAIMER = (
    "Quick Scan reflects reviews accessible to the current Amazon browser "
    "session and may not represent all reviews for this product."
)

LANGUAGE_STATE_KEY = "ui_language"
REPORT_LANGUAGES = ("en", "zh")
HEALTH_CACHE_TTL_SECONDS = 15
HEALTH_TEST_PROFILE_ENV = "ARA_UI_HEALTH_TEST_PROFILE"
FIRST_RUN_STATE_PATH_ENV = "ARA_FIRST_RUN_STATE_PATH"


def _language() -> str:
    """Return the display language without changing any analysis state."""

    value = st.session_state.get(LANGUAGE_STATE_KEY, DEFAULT_LANGUAGE)
    return value if value in SUPPORTED_LANGUAGES.values() else DEFAULT_LANGUAGE


def _label(key: str, fallback: str, **values: object) -> str:
    """Translate one UI label while retaining an explicit English fallback."""

    translated = t(key, _language(), **values)
    return fallback.format(**values) if translated == key else translated


def _offline_ui_health_profile(profile: str) -> dict[str, Any]:
    """Return a deterministic UI-only profile for offline AppTest runs.

    Production launchers never set this environment variable.  The profile
    prevents Streamlit's isolated AppTest runner from touching a real Ollama
    service; the environment health module itself is tested with injected
    probes instead.
    """

    limited = profile == "limited"
    ready_check = {
        "code": "ready",
        "status": "ready",
        "message": "Ready",
        "details": {},
    }
    checks = {
        name: dict(ready_check)
        for name in (
            "python",
            "venv",
            "dependencies",
            "ollama",
            "model",
            "port",
            "filesystem",
            "disk",
        )
    }
    checks["python"]["details"] = {"version": "3.13.14"}
    checks["ollama"]["details"] = {"version": "test metadata"}
    checks["model"]["details"] = {"model": OLLAMA_MODEL}
    if limited:
        checks["ollama"] = {
            "code": "ollama_unreachable",
            "status": "unavailable",
            "message": "Local AI unavailable",
            "details": {"installed": True},
        }
        checks["model"] = {
            "code": "model_unknown",
            "status": "unknown",
            "message": "Model availability unknown",
            "details": {"model": OLLAMA_MODEL},
        }
    return {
        "schema_version": "environment-health.v1",
        "app_version": get_app_version(),
        "overall_status": "limited" if limited else "ready",
        "checks": checks,
        "capabilities": {
            "dashboard": True,
            "demo": True,
            "saved_analyses": True,
            "reports": True,
            "comparison": True,
            "new_analysis": not limited,
            "amazon_import": True,
            "amazon_analysis": not limited,
        },
        "runtime": {"mode": "stable", "workers": 1},
    }


@st.cache_data(ttl=HEALTH_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_environment_health(test_profile: str = "") -> dict[str, Any]:
    """Run one short, non-inference health snapshot for the presentation layer."""

    profile = test_profile.strip().casefold()
    if profile in {"ready", "limited"}:
        return _offline_ui_health_profile(profile)
    return run_environment_health().to_dict()


def _health_report() -> dict[str, Any]:
    """Return a defensive health mapping without exposing technical errors."""

    try:
        profile = os.getenv(HEALTH_TEST_PROFILE_ENV, "").strip().casefold()
        report = _cached_environment_health(profile)
    except Exception:
        # The dashboard remains useful even if the diagnostic itself fails.
        report = _offline_ui_health_profile("limited")
        report["overall_status"] = "action_required"
        report["checks"]["filesystem"] = {
            "code": "health_check_unavailable",
            "status": "unknown",
            "message": "System status is temporarily unavailable",
            "details": {},
        }
    return report


def _new_analysis_available(report: Mapping[str, Any] | None = None) -> bool:
    """Return whether all prerequisites for new local inference are ready."""

    source = report if report is not None else _health_report()
    capabilities = source.get("capabilities", {})
    return bool(
        isinstance(capabilities, Mapping)
        and capabilities.get("new_analysis") is True
    )


def _first_run_state_path() -> Path:
    """Resolve the project-local first-run state, with a test-only override."""

    override = os.getenv(FIRST_RUN_STATE_PATH_ENV, "").strip()
    return Path(override) if override else DEFAULT_FIRST_RUN_STATE_PATH


def _first_run_is_complete() -> bool:
    if st.session_state.get("first_run_completed_session") is True:
        return True
    return is_first_run_completed(_first_run_state_path())


def _complete_first_run(destination: str, data_source: str | None = None) -> None:
    """Persist onboarding completion and select an existing application route."""

    try:
        mark_first_run_completed(get_app_version(), _first_run_state_path())
    except OSError:
        # A read-only state file must not trap the user on onboarding forever.
        pass
    st.session_state["first_run_completed_session"] = True
    st.session_state["dataset_mode"] = destination
    if destination == "Analyze New Dataset":
        st.session_state["analysis_workspace"] = "New Input"
        if data_source is not None:
            st.session_state["analysis_data_source"] = data_source


def _report_language_control(key_prefix: str) -> str:
    """Render a report-only language choice without changing dashboard state."""

    ui_language = _language()
    widget_key = f"{key_prefix}_report_language_{ui_language}"
    return st.selectbox(
        _label("report.language", "Report Language / 报告语言"),
        REPORT_LANGUAGES,
        index=REPORT_LANGUAGES.index(ui_language),
        format_func=lambda value: _label(
            f"report.language.{value}",
            "English" if value == "en" else "简体中文",
        ),
        key=widget_key,
    )


def _generate_single_product_report(
    analysis_root: Path,
    *,
    language: str,
    product_name: str,
    validate_integrity: bool,
):
    """Build and persist a report from completed outputs only.

    Importing lazily keeps report generation outside normal dashboard startup.
    The report module is deliberately given only an existing output directory;
    no analysis, aggregation, keyword, model, or network function is reachable
    through this adapter.
    """

    import report_generator

    document = report_generator.build_single_product_report(
        analysis_root,
        language=language,
        product_name=product_name,
        validate_integrity=validate_integrity,
    )
    exported = report_generator.export_report(document)
    return document, exported


def _report_error_message(exc: Exception) -> str:
    """Return a localized, non-technical message for report failures."""

    import report_generator

    if isinstance(exc, report_generator.ReportIntegrityError):
        return _label(
            "report.integrity_error",
            "Saved analysis integrity check failed.",
        )
    return _label(
        "report.generate_error",
        "The report could not be generated. Please try again.",
    )


def _render_single_report_controls(
    analysis_root: Path,
    *,
    product_name: str,
    key_prefix: str,
    validate_integrity: bool,
) -> None:
    """Offer deterministic HTML export for one completed analysis."""

    state_key = f"{key_prefix}_report_document"
    with st.expander(_label("report.export", "Export Report")):
        report_language = _report_language_control(key_prefix)
        st.caption(
            _label(
                "report.read_only_notice",
                "Reports use existing analysis outputs and do not rerun AI analysis.",
            )
        )
        if st.button(
            _label("report.generate", "Generate Report"),
            key=f"{key_prefix}_generate_report",
            type="primary",
        ):
            try:
                document, exported = _generate_single_product_report(
                    analysis_root,
                    language=report_language,
                    product_name=product_name,
                    validate_integrity=validate_integrity,
                )
            except Exception as exc:  # Report errors are contained at the UI boundary.
                st.error(_report_error_message(exc))
            else:
                st.session_state[state_key] = document
                st.session_state[f"{state_key}_export"] = exported
                st.success(
                    _label(
                        "report.generated",
                        "Report generated successfully.",
                    )
                )

        document = st.session_state.get(state_key)
        html_document = getattr(document, "html", None)
        report_id = str(getattr(document, "report_id", "report"))
        if isinstance(html_document, str) and html_document:
            st.download_button(
                _label("report.download_html", "Download HTML Report"),
                data=html_document,
                file_name=f"{report_id}.html",
                mime="text/html",
                key=f"{key_prefix}_download_report",
            )
            st.caption(
                _label(
                    "report.pdf_instruction",
                    "Open the report in your browser and choose Print → Save as PDF.",
                )
            )


def _localized_options(
    options: list[str],
    display,
) -> dict[str, str]:
    """Map canonical widget values to display-only labels."""

    return {option: str(display(option, _language())) for option in options}


def _display_aspect_option(value: object) -> str:
    return _label("common.all", "All") if str(value) == "All" else display_aspect(value, _language())


def _display_sentiment_option(value: object) -> str:
    return _label("common.all", "All") if str(value) == "All" else display_sentiment(value, _language())


def _display_dataset_label(value: str) -> str:
    return {
        "Demo Dataset": _label("dataset.demo", "Demo Dataset"),
        COMPARE_PRODUCTS_MODE: _label("dataset.compare_products", COMPARE_PRODUCTS_MODE),
        SAVED_ANALYSES_MODE: _label("dataset.saved_analyses", SAVED_ANALYSES_MODE),
        "Current Analysis": _label("dataset.current_analysis", "Current Analysis"),
        UPLOAD_WORKSPACE: _label("dataset.current_analysis", UPLOAD_WORKSPACE),
        AMAZON_WORKSPACE: _label("dataset.current_amazon", AMAZON_WORKSPACE),
        AMAZON_QUICK_SCAN_SOURCE: _label("dataset.amazon_quick_scan", AMAZON_QUICK_SCAN_SOURCE),
    }.get(value, value)


def _safe_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _display_verified_purchase(value: object) -> str:
    """Localize verified-purchase metadata without changing stored values."""

    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "verified", "verified purchase"}:
        return _label("explorer.verified", "Verified")
    if normalized in {"false", "0", "no", "not verified", "unverified"}:
        return _label("explorer.not_verified", "Not verified")
    return _label("common.unknown", "Unknown")


def _localized_review_preview(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy with localized headings and metadata."""

    display = frame.copy(deep=True)
    if _language() == "en":
        # Preserve the established English preview contract; Chinese uses
        # display-only aliases below.  The source DataFrame is never mutated.
        return display
    if "verified_purchase" in display.columns:
        display["verified_purchase"] = display["verified_purchase"].map(
            _display_verified_purchase
        )
    return display.rename(
        columns={
            "review_id": _label("common.review_id", "Review ID"),
            "rating": _label("rating.rating", "Rating"),
            "title": _label("common.title", "Title"),
            "content": _label("common.content", "Content"),
            "date": _label("common.date", "Date"),
            "verified_purchase": _label(
                "explorer.verified_purchase", "Verified Purchase"
            ),
        }
    )


def _friendly_local_ai_error(error: BaseException) -> str:
    """Localize known Ollama failures while preserving useful diagnostics."""

    friendly = interactive.friendly_ollama_error(error)
    if friendly.startswith("Model "):
        return _label(
            "runtime.model_missing",
            "Model {model} is not installed.",
            model=OLLAMA_MODEL,
        )
    if friendly.startswith("Local AI is currently unavailable"):
        return _label(
            "runtime.ollama_unavailable_detail",
            "Local AI is currently unavailable. Please start Ollama and try again.",
        )
    if friendly == "Local AI model configuration is unavailable.":
        return _label(
            "runtime.configuration_unavailable",
            "Local AI model configuration is unavailable.",
        )
    return _label(
        "runtime.analysis_start_error",
        "The local AI analysis could not be started. Please try again.",
    )


st.set_page_config(
    page_title=t(
        "app.title",
        st.session_state.get(LANGUAGE_STATE_KEY, DEFAULT_LANGUAGE),
    ),
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _load_cached_dashboard(
    outputs_dir: str,
    fingerprint: tuple[tuple[str, int, int], ...],
) -> DashboardData:
    """Cache file reads; fingerprint invalidates the cache after regeneration."""

    del fingerprint
    return load_dashboard_data(outputs_dir)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #F7F9FC; }
        .block-container { max-width: 1320px; padding-top: 1.7rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #17324D; letter-spacing: -0.02em; }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E4E9F0;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 5px 18px rgba(23, 50, 77, 0.05);
        }
        [data-testid="stMetricLabel"] { color: #58677A; }
        [data-testid="stMetricValue"] { color: #17324D; }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            color: white;
            background: linear-gradient(120deg, #17324D 0%, #1F6F78 100%);
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(23, 50, 77, 0.18);
        }
        .hero h1 { color: white; margin: 0; font-size: 2.25rem; }
        .hero p { color: #E6F4F5; margin: .35rem 0 0; font-size: 1.05rem; }
        .hero .source { color: #C7E2E5; margin-top: .8rem; font-size: .85rem; }
        .insight-card {
            background: #FFFFFF;
            border: 1px solid #E4E9F0;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            min-height: 132px;
            box-shadow: 0 5px 18px rgba(23, 50, 77, 0.05);
        }
        .insight-card h4 { color: #17324D; margin: 0 0 .4rem; }
        .insight-card .rate { color: #1F6F78; font-size: 1.35rem; font-weight: 700; }
        .insight-card .detail { color: #66758A; font-size: .88rem; }
        .phrase-positive, .phrase-negative {
            padding: .48rem .7rem;
            margin: .35rem 0;
            border-radius: 9px;
            font-size: .92rem;
        }
        .phrase-positive { background: #EAF6F1; color: #216550; }
        .phrase-negative { background: #FBEDED; color: #9B3F3F; }
        .footer { text-align: center; color: #738096; font-size: .82rem; padding-top: 2rem; }
        div[data-testid="stExpander"] { background: #FFFFFF; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_output_issue(data: DashboardData, output_name: str) -> None:
    issue = data.issue_for(output_name)
    if not issue:
        return
    st.warning(_label("errors.output_issue", "{message}\n\nPlease run:\n\n```bash\n{command}\n```", message=issue.message, command=issue.command))


def _render_kpis(kpis: Mapping[str, int | float]) -> None:
    columns = st.columns(5)
    values = (
        (_label("overview.total_reviews", "Total Reviews"), f"{int(kpis['total_reviews']):,}", None),
        (_label("overview.successfully_analyzed", "Successfully Analyzed"), f"{int(kpis['successful_reviews']):,}", None),
        (_label("overview.analysis_coverage", "Analysis Coverage"), f"{float(kpis['analysis_coverage']):.1%}", None),
        (_label("overview.product_aspects", "Product Aspects"), f"{int(kpis['product_aspects_identified']):,}", _label("common.identified", "identified")),
        (_label("overview.validated_evidence", "Validated Evidence"), f"{int(kpis['validated_evidence_count']):,}", _label("common.mentions", "mentions")),
    )
    for column, (label, value, help_text) in zip(columns, values):
        column.metric(label, value, help=help_text)


def _render_data_quality_notice(
    data: DashboardData,
    kpis: Mapping[str, int | float],
) -> None:
    coverage = float(kpis["analysis_coverage"])
    failed = int(kpis["failed_reviews"])
    skipped = int(kpis["skipped_reviews"])
    if coverage >= 1 and failed == 0 and skipped == 0:
        return

    parts = [_label("overview.analysis_coverage_notice", "Analysis coverage: {coverage:.1%}.", coverage=f"{coverage:.1%}")]
    if skipped:
        reason_text = ""
        frame = data.analyzed_reviews
        if frame is not None and "skip_reason" in frame.columns:
            skipped_rows = frame.loc[
                frame["analysis_status"].astype(str).str.casefold().eq("skipped")
            ]
            reasons = skipped_rows["skip_reason"].astype(str).str.strip()
            if len(skipped_rows) == skipped and reasons.isin(
                ["missing_content", "empty_after_cleaning"]
            ).all():
                reason_text = _label(
                    "runtime.no_analyzable_reason",
                    " because no analyzable content was available",
                )
        parts.append(_label("runtime.skipped_notice", "{count} review(s) were skipped{reason}.", count=skipped, reason=reason_text))
    if failed:
        parts.append(t_count("runtime.failed_count", "runtime.failed_count_plural", failed, _language()))
    st.info(" ".join(parts))


def _concern_chart(frame: pd.DataFrame) -> go.Figure:
    plot_data = frame.sort_values("aspect_review_count", ascending=True).copy()
    plot_data["aspect"] = plot_data["aspect"].map(lambda value: display_aspect(value, _language()))
    plot_data["category_label"] = plot_data["business_category"].map(
        lambda value: display_business_category(value, _language())
    )
    category_colors = {
        display_business_category("PRODUCT", _language()): "#2D6CDF",
        display_business_category("EXPERIENCE", _language()): "#D98C32",
        display_business_category("OTHER", _language()): "#8A94A6",
    }
    figure = px.bar(
        plot_data,
        x="aspect_share_of_reviews",
        y="aspect",
        color="category_label",
        orientation="h",
        color_discrete_map=category_colors,
        custom_data=[
            "aspect_review_count",
            "mention_count",
            "category_label",
        ],
    )
    figure.update_traces(
        hovertemplate=(
            f"<b>%{{y}}</b><br>{_label('chart.share_analyzed_reviews', 'Share of analyzed reviews')}: %{{x:.1f}}%"
            f"<br>{_label('chart.review_count', 'Review count')}: %{{customdata[0]}}"
            f"<br>{_label('chart.mention_count', 'Mention count')}: %{{customdata[1]}}"
            f"<br>{_label('common.category', 'Category')}: %{{customdata[2]}}<extra></extra>"
        ),
        marker_line_width=0,
    )
    figure.update_layout(
        height=max(360, 44 * len(plot_data)),
        margin=dict(l=8, r=20, t=15, b=10),
        xaxis_title=_label("chart.share_analyzed_reviews", "% of analyzed reviews"),
        yaxis_title=None,
        legend_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(range=[0, max(50, plot_data["aspect_share_of_reviews"].max() * 1.18)])
    return figure


def _sentiment_chart(frame: pd.DataFrame) -> go.Figure:
    plot_data = frame.sort_values("aspect_review_count", ascending=True).copy()
    plot_data["aspect"] = plot_data["aspect"].map(lambda value: display_aspect(value, _language()))
    figure = go.Figure()
    definitions = (
        ("positive", "positive_rate"),
        ("negative", "negative_rate"),
        ("neutral", "neutral_rate"),
        ("mixed", "mixed_rate"),
    )
    for label, column in definitions:
        values = pd.to_numeric(plot_data[column], errors="coerce").fillna(0.0)
        figure.add_bar(
            name=display_sentiment(label, _language()),
            y=plot_data["aspect"],
            x=values,
            orientation="h",
            marker_color=SENTIMENT_COLORS[label.title()],
            hovertemplate=f"<b>%{{y}}</b><br>{display_sentiment(label, _language())}: %{{x:.1f}}%<extra></extra>",
        )
    figure.update_layout(
        barmode="stack",
        height=max(410, 39 * len(plot_data)),
        margin=dict(l=8, r=20, t=20, b=10),
        xaxis_title=_label("chart.share_within_aspect", "Share within Aspect"),
        yaxis_title=None,
        legend_title=None,
        legend_orientation="h",
        legend_y=1.08,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(range=[0, 100], ticksuffix="%")
    return figure


def _render_directional_cards(
    title: str,
    records: object,
    *,
    direction: str,
    empty_message: str | None = None,
) -> None:
    st.subheader(title)
    if not isinstance(records, list) or not records:
        st.info(empty_message or _label(
            "bi.no_selling_point" if direction == "positive" else "bi.no_pain_point",
            "No product selling point meets the current minimum sample threshold."
            if direction == "positive" else
            "No product pain point meets the current minimum sample threshold.",
        ))
        return
    columns = st.columns(min(3, len(records)))
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        aspect = display_aspect(record.get("aspect", "Other"), _language())
        total = _safe_int(record.get("aspect_review_count"))
        directional_count = _safe_int(record.get(f"{direction}_review_count"))
        rate = _safe_float(record.get(f"{direction}_rate"))
        positive_rate = _safe_float(record.get("positive_rate"))
        negative_rate = _safe_float(record.get("negative_rate"))
        review_share = _safe_float(
            record.get("review_share", record.get("aspect_share_of_reviews"))
        )
        evidence_level = display_evidence_level(record.get("evidence_level", "Low"), _language())
        direction_label = display_sentiment(direction, _language())
        with columns[position % len(columns)]:
            st.markdown(
                f"""
                <div class="insight-card">
                  <h4>{html.escape(aspect)}</h4>
                  <div class="rate">{rate:.1f}% {html.escape(direction_label)}</div>
                  <div class="detail">{_label('cards.review_share', '{count} reviews · Mentioned in {share:.1f}% of analyzed reviews', count=total, share=review_share)}</div>
                  <div class="detail">{display_sentiment('positive', _language())} {positive_rate:.1f}% · {display_sentiment('negative', _language())} {negative_rate:.1f}%</div>
                  <div class="detail">{_label('cards.directional_evidence', '{count} {direction} reviews · Evidence: {level}', count=directional_count, direction=direction_label, level=html.escape(evidence_level))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_overall_sentiment(summary: Mapping[str, Any]) -> None:
    st.subheader(_label("overview.overall_sentiment", "Overall Sentiment"))
    st.caption(_label("overview.overall_sentiment_caption", "Aspect Mention Sentiment — each extracted Aspect mention is counted; this is not an overall product-satisfaction score."))
    columns = st.columns(3)
    for column, sentiment, label in zip(
        columns,
        ("positive", "negative", "neutral"),
        (display_sentiment("positive", _language()), display_sentiment("negative", _language()), display_sentiment("neutral", _language())),
    ):
        column.metric(
            label,
            f"{_safe_float(summary.get(f'{sentiment}_rate')):.1f}%",
            help=_label("overview.aspect_mentions", "{count} Aspect mentions", count=_safe_int(summary.get(f'{sentiment}_count'))),
        )


def _experience_comparison(data: Mapping[str, Any]) -> None:
    st.subheader(_label("overview.product_vs_shopping", "Product vs Shopping Experience"))
    st.caption(_label("overview.product_vs_shopping_caption", "Aspect mention sentiment. Delivery, Packaging and Customer Service are kept separate from product performance."))
    left, right = st.columns(2, gap="large")
    definitions = (
        (left, display_business_category("PRODUCT", _language()), data.get("product_experience_summary", {})),
        (
            right,
            display_business_category("EXPERIENCE", _language()),
            data.get("shopping_experience_summary", {}),
        ),
    )
    for column, title, raw_summary in definitions:
        summary = raw_summary if isinstance(raw_summary, Mapping) else {}
        column.markdown(f"**{title}**")
        metrics = column.columns(3)
        metrics[0].metric(display_sentiment("positive", _language()), f"{_safe_float(summary.get('positive_rate')):.1f}%")
        metrics[1].metric(display_sentiment("negative", _language()), f"{_safe_float(summary.get('negative_rate')):.1f}%")
        metrics[2].metric(_label("common.mentions", "Mentions"), f"{_safe_int(summary.get('total_mentions')):,}")


def _render_overview(
    data: DashboardData,
    compact_view: bool,
    key_prefix: str,
) -> None:
    intelligence = data.business_intelligence
    if not isinstance(intelligence, Mapping):
        _show_output_issue(data, "aspect_summary")
        _show_output_issue(data, "business_summary")
        if data.issue_for("aspect_summary") is None and data.issue_for(
            "business_summary"
        ) is None:
            st.warning(_label("overview.no_business_intelligence", "Business intelligence could not be derived from these outputs."))
        return

    st.header(_label("overview.executive_summary", "Executive Summary"))
    st.info(localized_executive_summary(intelligence, _language()), icon="💡")
    _render_overall_sentiment(
        intelligence.get("overall_sentiment", {})
        if isinstance(intelligence.get("overall_sentiment"), Mapping)
        else {}
    )

    mode = intelligence.get("sample_level", {})
    mode_key = str(mode.get("key", "")) if isinstance(mode, Mapping) else ""
    selling_title = (
        _label("overview.early_selling", "Early Selling Signals") if mode_key == "early_signals" else _label("overview.selling_points", "Top Selling Points")
    )
    pain_title = (
        _label("overview.early_pain", "Early Pain Signals") if mode_key == "early_signals" else _label("overview.pain_points", "Top Customer Pain Points")
    )
    left, right = st.columns(2, gap="large")
    with left:
        _render_directional_cards(
            selling_title,
            intelligence.get("selling_points", []),
            direction="positive",
        )
    with right:
        _render_directional_cards(
            pain_title,
            intelligence.get("pain_points", []),
            direction="negative",
        )

    _experience_comparison(intelligence)
    st.divider()
    st.subheader(_label("overview.customer_concerns", "Customer Concerns"))
    st.caption(_label("bi.opportunity_caption", "Ranked by Opportunity Score = review share × negative rate. This is a prioritization aid, not statistical significance."))
    category = st.radio(
        _label("overview.concern_category", "Concern category"),
        ["All", "Product", "Shopping Experience"],
        horizontal=True,
        label_visibility="collapsed" if compact_view else "visible",
        key=f"{key_prefix}_concern_category",
        format_func=lambda value: {
            "All": _label("common.all", "All"),
            "Product": display_business_category("PRODUCT", _language()),
            "Shopping Experience": display_business_category("EXPERIENCE", _language()),
        }[value],
    )
    concerns = pd.DataFrame(intelligence.get("opportunity_ranking", []))
    if not concerns.empty:
        if category == "Product":
            concerns = concerns.loc[concerns["business_category"].eq("PRODUCT")]
        elif category == "Shopping Experience":
            concerns = concerns.loc[concerns["business_category"].eq("EXPERIENCE")]
    if concerns.empty:
        st.caption(_label("overview.no_concern", "No concern meets the current minimum support threshold."))
    else:
        display = concerns[
            [
                "aspect",
                "review_count",
                "review_share",
                "negative_review_count",
                "negative_rate",
                "opportunity_score",
                "evidence_level",
            ]
        ].rename(
            columns={
                "aspect": "Aspect",
                "review_count": "Reviews",
                "review_share": "Review share (%)",
                "negative_review_count": "Negative reviews",
                "negative_rate": "Negative rate (%)",
                "opportunity_score": "Opportunity Score",
                "evidence_level": "Evidence Level",
            }
        )
        display["Aspect"] = display["Aspect"].map(lambda value: display_aspect(value, _language()))
        display["Evidence Level"] = display["Evidence Level"].map(lambda value: display_evidence_level(value, _language()))
        display = display.rename(columns={
            "Aspect": _label("common.aspect", "Aspect"),
            "Reviews": _label("common.reviews", "Reviews"),
            "Review share (%)": _label("common.review_share_percent", "Review share (%)"),
            "Negative reviews": _label("common.negative_reviews", "Negative reviews"),
            "Negative rate (%)": _label("common.negative_rate_percent", "Negative rate (%)"),
            "Opportunity Score": _label("common.opportunity_score", "Opportunity Score"),
            "Evidence Level": _label("common.evidence_level", "Evidence Level"),
        })
        st.dataframe(display, hide_index=True, width="stretch")


def _priority_matrix_figure(
    records: list[dict[str, Any]],
    high_share_threshold: float,
) -> go.Figure:
    figure = go.Figure()
    colors = {
        "Strength": POSITIVE,
        "Improvement Priority": NEGATIVE,
        "Emerging Concern": "#D98C32",
        "Minor Positive": ACCENT,
        "Balanced / Monitor": NEUTRAL,
    }
    max_reviews = max(
        1,
        max(
            (_safe_int(record.get("review_count")) for record in records),
            default=0,
        ),
    )
    for quadrant, color in colors.items():
        subset = [record for record in records if record.get("quadrant") == quadrant]
        if not subset:
            continue
        sizes = [max(12, 42 * _safe_int(record.get("review_count")) / max_reviews) for record in subset]
        figure.add_trace(
            go.Scatter(
                x=[_safe_float(record.get("net_sentiment")) for record in subset],
                y=[_safe_float(record.get("review_share")) for record in subset],
                text=[display_aspect(record.get("aspect", ""), _language()) for record in subset],
                customdata=[
                    [
                        _safe_int(record.get("review_count")),
                        _safe_float(record.get("positive_rate")),
                        _safe_float(record.get("negative_rate")),
                    ]
                    for record in subset
                ],
                mode="markers+text",
                textposition="top center",
                marker={"size": sizes, "color": color, "opacity": 0.82},
                name=display_quadrant(quadrant, _language()),
                hovertemplate=(
                    f"<b>%{{text}}</b><br>{_label('common.reviews', 'Reviews')}: %{{customdata[0]}}"
                    f"<br>{_label('bi.review_share', 'Review share')}: %{{y:.1f}}%"
                    f"<br>{display_sentiment('positive', _language())}: %{{customdata[1]:.1f}}%"
                    f"<br>{display_sentiment('negative', _language())}: %{{customdata[2]:.1f}}%<extra></extra>"
                ),
            )
        )
    figure.add_vline(x=0, line_dash="dash", line_color="#AAB3C2")
    figure.add_hline(
        y=high_share_threshold,
        line_dash="dash",
        line_color="#AAB3C2",
    )
    figure.update_layout(
        height=540,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title=_label("bi.sentiment_direction", "Sentiment balance (negative ← 0 → positive)"),
        yaxis_title=_label("chart.review_share", "Share of analyzed reviews"),
        legend_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(range=[-105, 105], ticksuffix="")
    figure.update_yaxes(rangemode="tozero", ticksuffix="%")
    return figure


def _rating_heatmap_figure(records: list[dict[str, Any]]) -> go.Figure:
    rows = sorted(records, key=lambda record: (-_safe_int(record.get("review_count")), str(record.get("aspect", ""))))
    figure = go.Figure(
        data=go.Heatmap(
            z=[
                [_safe_int(record.get(f"rating_{star}")) for star in range(1, 6)]
                for record in rows
            ],
            x=["1★", "2★", "3★", "4★", "5★"],
            y=[display_aspect(record.get("aspect", ""), _language()) for record in rows],
            texttemplate="%{z}",
            colorscale=[[0, "#F4F7FA"], [1, ACCENT]],
            colorbar={"title": _label("common.reviews", "Reviews")},
            hovertemplate=f"<b>%{{y}}</b><br>%{{x}}: %{{z}} {_label('common.reviews', 'reviews')}<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(380, 34 * len(rows)),
        margin=dict(l=20, r=20, t=15, b=20),
        xaxis_title=_label("rating.rating", "Original customer rating"),
        yaxis_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _render_insights(data: DashboardData) -> None:
    intelligence = data.business_intelligence
    if not isinstance(intelligence, Mapping):
        _show_output_issue(data, "aspect_summary")
        _show_output_issue(data, "business_summary")
        if data.issue_for("aspect_summary") is None and data.issue_for(
            "business_summary"
        ) is None:
            st.warning(_label("overview.no_business_intelligence", "Business intelligence could not be derived from these outputs."))
        return

    st.subheader(_label("bi.priority_matrix", "Customer Priority Matrix"))
    st.caption(_label("bi.priority_matrix_caption", "Bubble size represents unique review count. The dashed horizontal line marks the configured high-share threshold."))
    matrix = intelligence.get("priority_matrix", [])
    rules = intelligence.get("business_intelligence_rules", {})
    threshold = (
        _safe_float(rules.get("priority_high_share_percent"))
        if isinstance(rules, Mapping)
        else 20.0
    )
    if isinstance(matrix, list) and matrix:
        st.plotly_chart(
            _priority_matrix_figure(matrix, threshold),
            width="stretch",
            config={"displayModeBar": False},
        )
    else:
        st.info(_label("bi.no_priority_data", "No supported Aspect is available for the priority matrix."))

    st.subheader(_label("bi.opportunity_ranking", "Opportunity Ranking"))
    st.caption(_label("bi.opportunity_caption", "Opportunity Score = review share × negative rate. It is not a statistical significance measure."))
    opportunities = intelligence.get("opportunity_ranking", [])
    if isinstance(opportunities, list) and opportunities:
        opportunity_frame = pd.DataFrame(opportunities)
        display = opportunity_frame[
            [
                "aspect",
                "business_category",
                "review_count",
                "review_share",
                "negative_review_count",
                "negative_rate",
                "opportunity_score",
                "opportunity_level",
                "evidence_level",
            ]
        ].rename(
            columns={
                "aspect": "Aspect",
                "business_category": "Category",
                "review_count": "Reviews",
                "review_share": "Review share (%)",
                "negative_review_count": "Negative reviews",
                "negative_rate": "Negative rate (%)",
                "opportunity_score": "Opportunity Score",
                "opportunity_level": "Priority",
                "evidence_level": "Evidence Level",
            }
        )
        display["Aspect"] = display["Aspect"].map(lambda value: display_aspect(value, _language()))
        display["Category"] = display["Category"].map(lambda value: display_business_category(value, _language()))
        display["Priority"] = display["Priority"].map(lambda value: _label(f"priority.{str(value).casefold()}", str(value)))
        display["Evidence Level"] = display["Evidence Level"].map(lambda value: display_evidence_level(value, _language()))
        display = display.rename(
            columns={
                "Aspect": _label("common.aspect", "Aspect"),
                "Category": _label("common.category", "Category"),
                "Reviews": _label("common.reviews", "Reviews"),
                "Review share (%)": _label(
                    "common.review_share_percent", "Review share (%)"
                ),
                "Negative reviews": _label(
                    "common.negative_reviews", "Negative reviews"
                ),
                "Negative rate (%)": _label(
                    "common.negative_rate_percent", "Negative rate (%)"
                ),
                "Opportunity Score": _label(
                    "common.opportunity_score", "Opportunity Score"
                ),
                "Priority": _label("common.priority", "Priority"),
                "Evidence Level": _label(
                    "common.evidence_level", "Evidence Level"
                ),
            }
        )
        st.dataframe(display, hide_index=True, width="stretch")
    else:
        st.info(_label("overview.no_concern", "No issue meets the current minimum support threshold."))

    recommendations = intelligence.get("recommendations", [])
    actionable = [
        record
        for record in recommendations
        if isinstance(record, Mapping)
        and record.get("type") != "Insufficient Evidence"
    ] if isinstance(recommendations, list) else []
    st.subheader(_label("bi.business_recommendations", "Business Recommendations"))
    if actionable:
        localized = []
        for record in actionable[:8]:
            display_record = dict(record)
            translated = localized_recommendation(record, _language())
            display_record["type"] = translated.get("type", display_recommendation(record.get("type"), _language()))
            display_record["message"] = translated.get("message", str(record.get("message", "")))
            display_record["aspect"] = display_aspect(record.get("aspect"), _language())
            display_record["evidence_level"] = display_evidence_level(record.get("evidence_level"), _language())
            localized.append(display_record)
        recommendation_frame = pd.DataFrame(localized)
        st.dataframe(
            recommendation_frame[["type", "aspect", "message", "evidence_level"]].rename(
                columns={
                    "type": _label("recommendation.label", "Recommendation"),
                    "aspect": _label("common.aspect", "Aspect"),
                    "message": _label(
                        "recommendation.guidance", "Template guidance"
                    ),
                    "evidence_level": _label(
                        "common.evidence_level", "Evidence Level"
                    ),
                }
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption(_label("bi.no_recommendation", "No template recommendation has sufficient evidence yet."))

    rating = intelligence.get("rating_insights", {})
    if isinstance(rating, Mapping) and rating.get("available"):
        st.divider()
        st.subheader(_label("rating.aspect_by_star", "Aspect by Star Rating"))
        st.caption(_label("rating.separate_notice", "Original customer star ratings are shown separately from AI Aspect sentiment."))
        rating_matrix = rating.get("rating_matrix", [])
        if isinstance(rating_matrix, list) and rating_matrix:
            st.plotly_chart(
                _rating_heatmap_figure(rating_matrix),
                width="stretch",
                config={"displayModeBar": False},
            )
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(f"#### {_label('rating.hidden_complaints_heading', 'Hidden complaints in positive reviews')}")
            hidden = [
                record
                for record in rating.get("hidden_complaints", [])
                if isinstance(record, Mapping)
                and record.get("business_category") == "PRODUCT"
            ]
            if isinstance(hidden, list) and hidden:
                hidden_display = pd.DataFrame(hidden[:8]).rename(
                        columns={
                            "aspect": "Aspect",
                            "review_count": "Negative in 4–5★ reviews",
                            "mention_count": "Negative mentions",
                            "share_of_high_rating_reviews": "Share of 4–5★ reviews (%)",
                        }
                    )
                if "Aspect" in hidden_display:
                    hidden_display["Aspect"] = hidden_display["Aspect"].map(lambda value: display_aspect(value, _language()))
                hidden_display = hidden_display.rename(
                    columns={
                        "Aspect": _label("common.aspect", "Aspect"),
                        "Negative in 4–5★ reviews": _label(
                            "rating.negative_high_reviews",
                            "Negative in 4–5★ reviews",
                        ),
                        "Negative mentions": _label(
                            "rating.negative_mentions", "Negative mentions"
                        ),
                        "Share of 4–5★ reviews (%)": _label(
                            "rating.share_high_reviews",
                            "Share of 4–5★ reviews (%)",
                        ),
                    }
                )
                st.dataframe(
                    hidden_display,
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption(_label("rating.no_hidden_complaint", "No hidden complaint was detected in 4–5 star reviews."))
        with right:
            st.markdown(f"#### {_label('rating.strengths_negative', 'Strengths even in negative reviews')}")
            strengths = [
                record
                for record in rating.get("strengths_in_negative_reviews", [])
                if isinstance(record, Mapping)
                and record.get("business_category") == "PRODUCT"
            ]
            if isinstance(strengths, list) and strengths:
                strength_display = pd.DataFrame(strengths[:8]).rename(
                        columns={
                            "aspect": "Aspect",
                            "review_count": "Positive in 1–2★ reviews",
                            "mention_count": "Positive mentions",
                            "share_of_low_rating_reviews": "Share of 1–2★ reviews (%)",
                        }
                    )
                if "Aspect" in strength_display:
                    strength_display["Aspect"] = strength_display["Aspect"].map(lambda value: display_aspect(value, _language()))
                strength_display = strength_display.rename(
                    columns={
                        "Aspect": _label("common.aspect", "Aspect"),
                        "Positive in 1–2★ reviews": _label(
                            "rating.positive_low_reviews",
                            "Positive in 1–2★ reviews",
                        ),
                        "Positive mentions": _label(
                            "rating.positive_mentions", "Positive mentions"
                        ),
                        "Share of 1–2★ reviews (%)": _label(
                            "rating.share_low_reviews",
                            "Share of 1–2★ reviews (%)",
                        ),
                    }
                )
                st.dataframe(
                    strength_display,
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption(_label("rating.no_low_rating_strength", "No positive Aspect was detected in 1–2 star reviews."))

    st.divider()
    st.subheader(_label("bi.shopping_experience_issues", "Shopping Experience Issues"))
    shopping_issues = [
        record
        for record in opportunities
        if isinstance(record, Mapping) and record.get("business_category") == "EXPERIENCE"
    ] if isinstance(opportunities, list) else []
    if shopping_issues:
        shopping_display = pd.DataFrame(shopping_issues)[
                ["aspect", "review_count", "negative_review_count", "negative_rate", "evidence_level"]
            ].rename(
                columns={
                    "aspect": "Aspect",
                    "review_count": "Reviews",
                    "negative_review_count": "Negative",
                    "negative_rate": "Negative rate (%)",
                    "evidence_level": "Evidence Level",
                }
            )
        shopping_display["Aspect"] = shopping_display["Aspect"].map(lambda value: display_aspect(value, _language()))
        shopping_display["Evidence Level"] = shopping_display["Evidence Level"].map(lambda value: display_evidence_level(value, _language()))
        shopping_display = shopping_display.rename(
            columns={
                "Aspect": _label("common.aspect", "Aspect"),
                "Reviews": _label("common.reviews", "Reviews"),
                "Negative": _label("sentiment.negative", "Negative"),
                "Negative rate (%)": _label(
                    "common.negative_rate_percent", "Negative rate (%)"
                ),
                "Evidence Level": _label(
                    "common.evidence_level", "Evidence Level"
                ),
            }
        )
        st.dataframe(
            shopping_display,
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption(_label("bi.no_shopping_issue", "No shopping-experience issue meets the current support threshold."))

    st.divider()
    st.subheader(_label("overview.aspect_mention_sentiment", "Aspect Sentiment Breakdown"))
    summary_source = (
        data.enriched_aspect_summary
        if isinstance(data.enriched_aspect_summary, pd.DataFrame)
        else data.aspect_summary
    )
    if isinstance(summary_source, pd.DataFrame):
        summary = filter_aspect_summary(summary_source, "All")
        if not summary.empty:
            st.plotly_chart(
                _sentiment_chart(summary),
                width="stretch",
                config={"displayModeBar": False},
            )


def _phrase_html(record: Mapping[str, Any], css_class: str) -> str:
    term = html.escape(str(record.get("term", "")))
    reviews = int(
        record.get("sentiment_review_count", record.get("review_count", 0)) or 0
    )
    return (
        f'<div class="{css_class}"><b>{term}</b>'
        f'<span style="float:right">{_label("language.review_count", "{count} review(s)", count=reviews)}</span></div>'
    )


def _render_customer_language(data: DashboardData, key_prefix: str) -> None:
    st.header(_label("tab.customer_language", "Customer Language"))
    st.caption(_label("language.original_notice", "The words customers actually use — no additional model interpretation."))
    if data.keyword_summary is None:
        _show_output_issue(data, "keyword_summary")
    else:
        controls = st.columns([2, 3])
        selected_label = controls[0].radio(
            _label("language.type", "Term type"),
            ["Bigram", "Trigram", "Unigram"],
            horizontal=True,
            key=f"{key_prefix}_keyword_term_type",
            format_func=lambda value: _label(f"language.{value.casefold()}", value),
        )
        top_n = controls[1].slider(
            _label("language.top_n", "Top N"),
            5,
            25,
            10,
            key=f"{key_prefix}_keyword_top_n",
        )
        selected_type = selected_label.casefold()
        terms = data.keyword_summary.loc[
            data.keyword_summary["term_type"].astype(str).eq(selected_type)
        ].head(top_n)
        if terms.empty:
            st.info(
                _label(
                    "language.no_term_results",
                    "No {term_type} results are available.",
                    term_type=_label(
                        f"language.{selected_label.casefold()}", selected_label
                    ),
                )
            )
        else:
            display = terms[
                ["term", "review_count", "count", "review_share"]
            ].rename(
                columns={
                    "term": _label("language.term", "Term"),
                    "review_count": _label("common.reviews", "Reviews"),
                    "count": _label("language.occurrences", "Occurrences"),
                    "review_share": _label(
                        "common.review_share_percent", "Review share (%)"
                    ),
                }
            )
            share_column = _label(
                "common.review_share_percent", "Review share (%)"
            )
            st.dataframe(
                display,
                hide_index=True,
                width="stretch",
                column_config={
                    share_column: st.column_config.NumberColumn(format="%.1f%%")
                },
            )

    if data.sentiment_keywords is None:
        _show_output_issue(data, "sentiment_keywords")
        positive: list[dict[str, Any]] = []
        negative: list[dict[str, Any]] = []
    else:
        positive = top_term_records(
            data.sentiment_keywords.get("positive"),
            term_types=("bigram", "trigram"),
            limit=10,
        )
        negative = top_term_records(
            data.sentiment_keywords.get("negative"),
            term_types=("bigram", "trigram"),
            limit=10,
        )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(_label("language.positive_phrases", "Top Positive Phrases"))
        if positive:
            for record in positive:
                st.markdown(_phrase_html(record, "phrase-positive"), unsafe_allow_html=True)
        else:
            st.info(_label("language.no_positive_phrases", "No positive phrases are available."))
    with right:
        st.subheader(_label("language.negative_phrases", "Top Negative Phrases"))
        if negative:
            for record in negative:
                st.markdown(_phrase_html(record, "phrase-negative"), unsafe_allow_html=True)
        else:
            st.info(_label("language.no_negative_phrases", "No negative phrases are available."))

    st.divider()
    st.subheader(_label("language.by_aspect", "Language by Aspect"))
    st.caption(_label("language.by_aspect_caption", "Existing Aspect keywords are grouped using their stored review references and validated positive/negative Evidence."))
    if not isinstance(data.aspect_keywords, Mapping) or not data.aspect_keywords:
        _show_output_issue(data, "aspect_keywords")
        return
    aspect_options = sorted(str(aspect) for aspect in data.aspect_keywords)
    selected_aspect = st.selectbox(
        _label("language.aspect_language", "Aspect language"),
        aspect_options,
        key=f"{key_prefix}_language_aspect",
        format_func=lambda value: display_aspect(value, _language()),
    )
    grouped = aspect_language_by_sentiment(
        data.aspect_keywords,
        data.aspect_evidence,
        selected_aspect,
        limit=8,
    )
    positive_column, negative_column = st.columns(2, gap="large")
    with positive_column:
        st.markdown(f"#### {_label('language.positive_phrases', 'Common positive phrases')}")
        if grouped["positive"]:
            for record in grouped["positive"]:
                st.markdown(_phrase_html(record, "phrase-positive"), unsafe_allow_html=True)
        else:
            st.caption(_label("language.no_aspect_positive", "No positive phrase is linked to this Aspect yet."))
    with negative_column:
        st.markdown(f"#### {_label('language.negative_phrases', 'Common negative phrases')}")
        if grouped["negative"]:
            for record in grouped["negative"]:
                st.markdown(_phrase_html(record, "phrase-negative"), unsafe_allow_html=True)
        else:
            st.caption(_label("language.no_aspect_negative", "No negative phrase is linked to this Aspect yet."))


def _render_evidence_entries(entries: list[dict[str, Any]]) -> None:
    if not entries:
        st.caption(_label("deep_dive.no_evidence", "No validated Evidence for this sentiment."))
        return
    for entry in entries:
        evidence = str(entry.get("evidence", ""))
        review_id = str(entry.get("review_id", "Unknown review"))
        source_row_id = entry.get("source_row_id", "—")
        rating = str(entry.get("rating", "")).strip()
        verified = str(entry.get("verified_purchase", "Unknown"))
        metadata = [review_id, _label("evidence.source_row", "source row {row}", row=source_row_id)]
        if rating and rating.casefold() != "nan":
            metadata.append(_label("evidence.rating", "Rating {rating}", rating=rating))
        if verified.casefold() not in {"unknown", "nan", "", "none"}:
            metadata.append(_display_verified_purchase(verified))
        with st.container(border=True):
            st.write(f'“{evidence}”')
            st.caption(" · ".join(metadata))


def _render_aspect_deep_dive(data: DashboardData, key_prefix: str) -> None:
    st.header(_label("deep_dive.title", "Aspect Deep Dive"))
    summary = (
        data.enriched_aspect_summary
        if isinstance(data.enriched_aspect_summary, pd.DataFrame)
        else data.aspect_summary
    )
    if summary is None:
        _show_output_issue(data, "aspect_summary")
        return
    options = summary["aspect"].astype(str).tolist()
    if not options:
        st.info(_label("deep_dive.no_aspect", "No Aspect is available for deep dive."))
        return
    selected = st.selectbox(
        _label("deep_dive.select_aspect", "Select an Aspect"),
        options,
        key=f"{key_prefix}_selected_aspect",
        format_func=lambda value: display_aspect(value, _language()),
    )
    selected_rows = summary.loc[
        summary["aspect"].astype(str).eq(selected)
    ]
    if selected_rows.empty:
        return
    row = selected_rows.iloc[0]
    metrics = st.columns(4)
    metrics[0].metric(_label("deep_dive.reviews_mentioning", "Reviews Mentioning"), int(row["aspect_review_count"]))
    metrics[1].metric(_label("deep_dive.review_share", "Review Share"), f"{float(row['aspect_share_of_reviews']):.1f}%")
    metrics[2].metric(_label("deep_dive.mention_count", "Mention Count"), int(row["mention_count"]))
    metrics[3].metric(_label("deep_dive.evidence_level", "Evidence Level"), display_evidence_level(row.get("evidence_level", "Low"), _language()))
    sentiment_metrics = st.columns(4)
    sentiment_metrics[0].metric(_label("deep_dive.positive_rate", "Positive Rate"), f"{float(row['positive_rate']):.1f}%")
    sentiment_metrics[1].metric(_label("deep_dive.negative_rate", "Negative Rate"), f"{float(row['negative_rate']):.1f}%")
    sentiment_metrics[2].metric(_label("deep_dive.neutral_rate", "Neutral Rate"), f"{float(row['neutral_rate']):.1f}%")
    avg_rating = row.get("avg_rating")
    if avg_rating is not None and not pd.isna(avg_rating):
        sentiment_metrics[3].metric(_label("deep_dive.average_rating", "Average Rating"), f"{float(avg_rating):.2f} / 5")
    else:
        sentiment_metrics[3].metric(_label("common.category", "Category"), display_business_category(row["business_category"], _language()))

    sentiment_display = pd.DataFrame(
        {
            _label("explorer.sentiment", "Sentiment"): [display_sentiment(value, _language()) for value in ("positive", "negative", "neutral", "mixed")],
            _label("common.reviews", "Reviews"): [
                int(row["positive_review_count"]),
                int(row["negative_review_count"]),
                int(row["neutral_review_count"]),
                int(row["mixed_review_count"]),
            ],
            _label("common.rate_percent", "Rate (%)"): [
                float(row["positive_rate"]),
                float(row["negative_rate"]),
                float(row["neutral_rate"]),
                float(row["mixed_rate"]),
            ],
        }
    )
    st.dataframe(
        sentiment_display,
        hide_index=True,
        width="stretch",
        column_config={_label("common.rate_percent", "Rate (%)"): st.column_config.NumberColumn(format="%.1f%%")},
    )

    st.subheader(_label("deep_dive.keywords", "Top Keywords for this Aspect"))
    if data.aspect_keywords is None:
        _show_output_issue(data, "aspect_keywords")
    else:
        mode = st.radio(
            _label("language.keyword_view", "Keyword view"),
            ["Phrases", "Unigrams"],
            horizontal=True,
            key=f"{key_prefix}_aspect_keyword_mode",
            format_func=lambda value: _label(f"language.{value.casefold()}", value),
        )
        types = ("bigram", "trigram") if mode == "Phrases" else ("unigram",)
        records = top_term_records(
            data.aspect_keywords.get(selected, []),
            term_types=types,
            limit=10,
        )
        if records:
            keyword_display = pd.DataFrame(records)[
                ["term", "term_type", "review_count", "count"]
            ].copy()
            keyword_display["term_type"] = keyword_display["term_type"].map(
                lambda value: _label(f"language.{str(value).casefold()}", str(value))
            )
            keyword_display = keyword_display.rename(
                columns={
                    "term": _label("language.term", "Term"),
                    "term_type": _label("language.type", "Type"),
                    "review_count": _label("common.reviews", "Reviews"),
                    "count": _label("language.occurrences", "Occurrences"),
                }
            )
            st.dataframe(
                keyword_display,
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption(_label("language.no_keywords", "No keyword record is available for this view."))

    st.subheader(_label("deep_dive.customer_evidence", "Customer Evidence"))
    st.caption(_label("deep_dive.evidence_notice", "Every excerpt below is displayed verbatim from validated Step 4 Evidence."))
    if data.aspect_evidence is None:
        _show_output_issue(data, "aspect_evidence")
        return
    evidence = evidence_for_aspect(data.aspect_evidence, selected)
    show_more = st.toggle(
        _label("deep_dive.show_more_evidence", "Show more Evidence"),
        value=False,
        key=f"{key_prefix}_{selected}_show_more_evidence",
    )
    tabs = st.tabs(
        [
            _label("deep_dive.positive_evidence", "Positive Customer Evidence"),
            _label("deep_dive.negative_evidence", "Negative Customer Evidence"),
            _label("deep_dive.neutral_evidence", "Neutral Evidence"),
            _label("deep_dive.mixed_evidence", "Mixed Evidence"),
        ]
    )
    for tab, sentiment in zip(tabs, ("positive", "negative", "neutral", "mixed")):
        with tab:
            enriched_entries = evidence_with_review_metadata(
                evidence[sentiment], data.analyzed_reviews
            )
            display_limit = (
                EXPANDED_EVIDENCE_DISPLAY_LIMIT
                if show_more
                else DEFAULT_EVIDENCE_DISPLAY_LIMIT
            )
            visible = enriched_entries[:display_limit]
            _render_evidence_entries(visible)
            if len(enriched_entries) > len(visible):
                st.caption(
                    _label("deep_dive.showing_evidence", "Showing {shown} of {total} validated excerpts.", shown=len(visible), total=len(enriched_entries))
                )


def _first_text(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column not in row.index:
            continue
        text = str(row[column])
        if text and text.casefold() != "nan":
            return text
    return ""


def _render_review_explorer(
    data: DashboardData,
    compact_view: bool,
    key_prefix: str,
) -> None:
    st.header(_label("explorer.title", "Review Explorer"))
    frame = data.analyzed_reviews
    if frame is None:
        _show_output_issue(data, "analyzed_reviews")
        return
    options = review_filter_options(frame)
    search = st.text_input(
        _label("explorer.search_full", "Search review ID, title or content"),
        placeholder=_label("explorer.search_placeholder", "Try R003, remote, picture quality…"),
        key=f"{key_prefix}_review_search",
    )
    quick_filter = st.radio(
        _label("explorer.quick_filters", "Quick filters"),
        [
            "All",
            "Positive Evidence",
            "Negative Evidence",
            "1–2 Star",
            "4–5 Star",
            "Product Issues",
            "Shopping Experience",
        ],
        horizontal=True,
        key=f"{key_prefix}_review_quick_filter",
        format_func=lambda value: {
            "All": _label("common.all", "All"),
            "Positive Evidence": _label("explorer.positive_evidence", "Positive Evidence"),
            "Negative Evidence": _label("explorer.negative_evidence", "Negative Evidence"),
            "1–2 Star": _label("explorer.low_rating", "1–2 Star"),
            "4–5 Star": _label("explorer.high_rating", "4–5 Star"),
            "Product Issues": _label("explorer.product_issues", "Product Issues"),
            "Shopping Experience": _label("explorer.shopping_experience", "Shopping Experience"),
        }[value],
    )
    filter_columns = st.columns(5)
    ratings = filter_columns[0].multiselect(
        _label("explorer.rating", "Rating"),
        options["ratings"],
        default=options["ratings"],
        key=f"{key_prefix}_review_ratings",
    )
    statuses = filter_columns[1].multiselect(
        _label("explorer.analysis_status", "Analysis Status"),
        options["statuses"],
        default=options["statuses"],
        key=f"{key_prefix}_review_statuses",
        format_func=lambda value: display_status(value, _language()),
    )
    aspect = filter_columns[2].selectbox(
        _label("explorer.aspect", "Aspect"),
        ["All", *options["aspects"]],
        key=f"{key_prefix}_review_aspect",
        format_func=_display_aspect_option,
    )
    sentiment = filter_columns[3].selectbox(
        _label("explorer.sentiment", "Sentiment"),
        ["All", *options["sentiments"]],
        key=f"{key_prefix}_review_sentiment",
        format_func=_display_sentiment_option,
    )
    verified = filter_columns[4].selectbox(
        _label("explorer.verified_purchase", "Verified Purchase"),
        ["All", "Verified", "Not verified", "Unknown"],
        key=f"{key_prefix}_review_verified",
        format_func=lambda value: {
            "All": _label("common.all", "All"),
            "Verified": _label("explorer.verified", "Verified"),
            "Not verified": _label("explorer.not_verified", "Not verified"),
            "Unknown": _label("common.unknown", "Unknown"),
        }[value],
    )
    filtered = filter_reviews(
        frame,
        ratings=ratings,
        statuses=statuses,
        aspects=[] if aspect == "All" else [aspect],
        sentiments=[] if sentiment == "All" else [sentiment],
        verified_purchase=verified,
        query=search,
    )
    quick_filter_summary = (
        data.enriched_aspect_summary
        if isinstance(data.enriched_aspect_summary, pd.DataFrame)
        else data.aspect_summary
    )
    filtered = apply_review_quick_filter(
        filtered,
        quick_filter_summary,
        quick_filter,
    )
    st.caption(_label("explorer.showing_reviews", "Showing {shown} of {total} reviews", shown=len(filtered), total=len(frame)))
    if filtered.empty:
        st.info(_label("explorer.no_match", "No reviews match the current filters."))
    max_rows = 10 if compact_view else 25
    for _, row in filtered.head(max_rows).iterrows():
        review_id = _first_text(row, "review_id") or _label("common.unknown", "Unknown review")
        rating = _first_text(row, "rating") or "—"
        status = _first_text(row, "analysis_status") or "unknown"
        source_row = row.get("source_row_id", "—")
        label = _label("explorer.review_card_label", "{review_id} — Rating {rating} — {status} · row {row}", review_id=review_id, rating=rating, status=display_status(status, _language()), row=source_row)
        with st.expander(label):
            title = _first_text(row, "original_title", "title")
            if title:
                st.markdown(f"**{title}**")
            st.markdown(f"**{_label('explorer.original_review', 'Original Review')}**")
            original = _first_text(
                row,
                "original_content",
                "content",
                "cleaned_content",
            )
            st.write(original or _label("explorer.no_content", "No review content available."))
            summary = _first_text(row, "summary")
            if summary:
                st.markdown(f"**{_label('explorer.analysis_summary', 'Analysis Summary')}**")
                st.caption(
                    _label(
                        "explorer.summary_original_notice",
                        "AI summary remains in its original model-output language.",
                    )
                )
                st.write(summary)
            st.markdown(f"**{_label('explorer.detected_aspects', 'Detected Aspects')}**")
            aspects = row.get("parsed_aspects", [])
            if isinstance(aspects, list) and aspects:
                for entry in aspects:
                    if not isinstance(entry, Mapping):
                        continue
                    aspect_name = display_aspect(entry.get("aspect", "Other"), _language())
                    sentiment_name = display_sentiment(entry.get("sentiment", "unknown"), _language())
                    evidence = str(entry.get("evidence", ""))
                    st.markdown(f"- **{aspect_name} — {sentiment_name}**")
                    if evidence:
                        st.caption(f'{_label("explorer.evidence", "Evidence")}: “{evidence}”')
            else:
                st.caption(_label("explorer.no_aspect", "No specific Aspect was detected."))

    if len(filtered) > max_rows:
        st.caption(
            _label("explorer.first_results", "Showing the first {count} matching reviews. Refine filters to narrow the list.", count=max_rows)
        )

    st.divider()
    with st.expander(_label("explorer.data_quality", "System / Data Quality"), expanded=not compact_view):
        status_counts = (
            frame["analysis_status"].astype(str).value_counts().rename_axis("Status").reset_index(name="Reviews")
        )
        status_counts["Status"] = status_counts["Status"].map(lambda value: display_status(value, _language()))
        status_counts = status_counts.rename(columns={"Status": _label("explorer.status", "Status"), "Reviews": _label("common.reviews", "Reviews")})
        st.dataframe(status_counts, hide_index=True, width="stretch")
        if data.row_json_warnings:
            st.warning(_label("explorer.json_warning", "{count} row-level JSON field(s) could not be parsed. Affected fields were skipped without stopping the dashboard.", count=len(data.row_json_warnings)))
        else:
            st.success(_label("explorer.json_success", "All review-level JSON fields loaded successfully."))
        if data.issues:
            for issue in data.issues:
                st.caption(f"{issue.path.name}: {issue.message}")
        else:
            st.caption(_label("explorer.outputs_available", "All seven dashboard output files are available."))


def _render_dataset_summary_sidebar(
    data: DashboardData,
    kpis: Mapping[str, int | float],
    *,
    quick_scan: bool = False,
    outputs_dir: Path | None = None,
) -> None:
    with st.sidebar:
        st.divider()
        st.subheader(_label("dataset.summary", "Dataset Summary"))
        if quick_scan:
            st.write(f"**{_label('dataset.source_label', 'Source')}:** {_label('dataset.amazon_quick_scan', 'Amazon Quick Scan')}")
            st.write(f"**{_label('amazon.reviews_retrieved', 'Reviews Retrieved')}:** {int(kpis['total_reviews'])}")
            st.write(f"**{_label('dataset.analyzed', 'Reviews Analyzed')}:** {int(kpis['successful_reviews'])}")
            st.write(
                f"**{_label('overview.analysis_coverage', 'Analysis Coverage')}:** {float(kpis['analysis_coverage']):.1%}"
            )
        else:
            st.write(f"**{_label('dataset.reviews', 'Reviews')}:** {int(kpis['total_reviews'])}")
            st.write(f"**{_label('dataset.analyzed', 'Analyzed')}:** {int(kpis['successful_reviews'])}")
            st.write(f"**{_label('dataset.coverage', 'Coverage')}:** {float(kpis['analysis_coverage']):.1%}")
        minimum = "—"
        if data.business_summary:
            minimum = data.business_summary.get("ranking_rules", {}).get(
                "minimum_aspect_reviews", "—"
            )
        st.write(f"**{_label('dataset.ranking_minimum', 'Ranking minimum')}:** {minimum} {_label('common.reviews', 'reviews')}")
        st.subheader(_label("explorer.analysis_status", "Analysis Status"))
        st.write(f"**{_label('dataset.imported', 'Imported')}:** {int(kpis['total_reviews'])}")
        st.write(f"**{_label('runtime.successfully_analyzed', 'Successfully analyzed')}:** {int(kpis['successful_reviews'])}")
        st.write(f"**{_label('runtime.failed', 'Failed')}:** {int(kpis.get('failed_reviews', 0))}")
        st.write(f"**{_label('dataset.coverage', 'Coverage')}:** {float(kpis['analysis_coverage']):.1%}")
        if outputs_dir is not None:
            metadata = runtime.load_run_metadata(
                Path(outputs_dir) / "analysis_run_metadata.json"
            )
            if metadata is not None:
                duration = _safe_float(metadata.get("duration_seconds"))
                st.write(f"**{_label('runtime.analysis_time', 'Analysis time')}:** {display_duration(duration, _language())}")


def _render_quick_scan_disclaimer() -> None:
    """Explain the bounded Amazon source without presenting an error."""

    st.info(_label("amazon.quick_scan_disclaimer", AMAZON_QUICK_SCAN_DISCLAIMER))


def _health_status_label(status: object) -> str:
    canonical = str(status or "unknown").strip().casefold()
    return {
        "ready": _label("health.ready", "Ready"),
        "warning": _label("health.warning", "Warning"),
        "action_required": _label("health.action_required", "Action Required"),
        "unavailable": _label("health.unavailable", "Unavailable"),
        "limited": _label("health.limited", "Limited Mode"),
        "unknown": _label("health.unknown", "Unknown"),
    }.get(canonical, _label("health.unknown", "Unknown"))


def _safe_check(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    checks = report.get("checks", {})
    if not isinstance(checks, Mapping):
        return {}
    value = checks.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _safe_details(check: Mapping[str, Any]) -> Mapping[str, Any]:
    details = check.get("details", {})
    return details if isinstance(details, Mapping) else {}


def _sanitized_diagnostics(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build an allowlisted diagnostic payload with no user or review data."""

    safe_details = {
        "python": ("version", "required", "executable_available"),
        "venv": ("installed", "active"),
        "dependencies": (
            "required_count",
            "installed_count",
            "missing_required",
            "missing_optional",
            "missing_features",
        ),
        "ollama": ("installed", "running", "version", "host"),
        "model": ("model", "installed"),
        "port": ("port", "state"),
        "filesystem": ("directory_count", "writable"),
        "disk": ("available_gb",),
    }
    checks: dict[str, dict[str, Any]] = {}
    for name, allowed in safe_details.items():
        check = _safe_check(report, name)
        details = _safe_details(check)
        checks[name] = {
            "code": str(check.get("code") or "unknown"),
            "status": str(check.get("status") or "unknown"),
            "details": {
                key: details[key]
                for key in allowed
                if key in details
                and isinstance(details[key], (str, int, float, bool, list, tuple))
            },
        }
        if name == "dependencies":
            versions = details.get("versions", {})
            if isinstance(versions, Mapping):
                checks[name]["details"]["versions"] = {
                    str(package): str(version)
                    for package, version in versions.items()
                    if str(package) in {
                        "pandas",
                        "plotly",
                        "streamlit",
                        "playwright",
                        "ollama",
                        "pydantic",
                        "certifi",
                        "python-dotenv",
                    }
                }
    runtime = report.get("runtime", {})
    runtime = runtime if isinstance(runtime, Mapping) else {}
    return sanitize_diagnostics({
        "schema_version": str(report.get("schema_version") or "environment-health.v1"),
        "application_version": get_app_version(),
        "overall_status": str(report.get("overall_status") or "unknown"),
        "runtime": {
            "mode": str(runtime.get("mode") or "stable"),
            "workers": int(runtime.get("workers") or 1),
        },
        "checks": checks,
    })


def _diagnostics_text(report: Mapping[str, Any]) -> str:
    return json.dumps(
        _sanitized_diagnostics(report),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _render_system_health_sidebar(
    health_result: Mapping[str, Any] | None = None,
) -> None:
    """Render a compact accessible status and opt-in safe diagnostics."""

    report = health_result if health_result is not None else _health_report()
    overall = str(report.get("overall_status") or "unknown")
    overall_label = _health_status_label(overall)
    checks = {
        "python": _label("health.python", "Python"),
        "venv": _label("health.virtual_environment", "Virtual Environment"),
        "dependencies": _label("health.dependencies", "Dependencies"),
        "ollama": _label("health.local_ai_engine", "Local AI Engine"),
        "model": _label("health.ai_model", "AI Model"),
        "port": _label("health.port", "Dashboard Port"),
        "filesystem": _label("health.filesystem", "Local Storage"),
    }
    with st.sidebar:
        st.divider()
        st.subheader(_label("health.system", "System"))
        if overall == "ready":
            st.success(f"● {overall_label}")
        elif overall in {"limited", "warning"}:
            st.warning(f"● {overall_label}")
            if not _new_analysis_available(report):
                st.caption(
                    _label(
                        "health.limited_viewing",
                        "Local AI unavailable — viewing features remain available.",
                    )
                )
        else:
            st.error(f"● {overall_label}")

        with st.expander(_label("health.system_status", "System Status")):
            for name, display_name in checks.items():
                check = _safe_check(report, name)
                status = _health_status_label(check.get("status"))
                details = _safe_details(check)
                value = ""
                if name == "python":
                    value = str(details.get("version") or "")
                elif name == "ollama":
                    value = str(details.get("version") or "")
                elif name == "model":
                    value = str(details.get("model") or OLLAMA_MODEL)
                suffix = f" · {value}" if value else ""
                st.write(f"**{display_name}:** {status}{suffix}")
            st.write(
                f"**{_label('health.runtime_mode', 'Runtime Mode')}:** "
                f"{_label('health.runtime_value', 'Stable · 1 worker')}"
            )
            st.write(
                f"**{_label('health.application_version', 'Application Version')}:** "
                f"{get_app_version()}"
            )
            if st.button(
                _label("health.refresh", "Refresh System Status"),
                key="refresh_system_health",
                use_container_width=True,
            ):
                _cached_environment_health.clear()
                st.rerun()

        with st.expander(
            _label("health.advanced_diagnostics", "Advanced Diagnostics")
        ):
            diagnostic_text = _diagnostics_text(report)
            st.caption(
                _label(
                    "health.diagnostics_safe",
                    "This diagnostic summary excludes review text, credentials, cookies and personal file paths.",
                )
            )
            st.code(diagnostic_text, language="json")
            st.download_button(
                _label("health.download_diagnostics", "Download Diagnostics"),
                data=diagnostic_text,
                file_name=_label(
                    "diagnostics.filename",
                    "amazon_review_analyzer_diagnostics.txt",
                ),
                mime="text/plain",
                key="download_environment_diagnostics",
                use_container_width=True,
            )


def _render_model_information_sidebar(
    health_result: Mapping[str, Any] | None = None,
) -> None:
    """Compatibility wrapper retained for existing UI tests and callers."""

    _render_system_health_sidebar(health_result)


def _render_first_run_sidebar(report: Mapping[str, Any]) -> None:
    with st.sidebar:
        language_codes = list(SUPPORTED_LANGUAGES.values())
        if st.session_state.get(LANGUAGE_STATE_KEY) not in language_codes:
            st.session_state[LANGUAGE_STATE_KEY] = DEFAULT_LANGUAGE
        st.selectbox(
            t("language.selector", _language()),
            language_codes,
            format_func=lambda code: next(
                (
                    label
                    for label, value in SUPPORTED_LANGUAGES.items()
                    if value == code
                ),
                code,
            ),
            key=LANGUAGE_STATE_KEY,
        )
        st.caption(
            _label(
                "app.version",
                "Amazon Review Analyzer v{version}",
                version=get_app_version(),
            )
        )
    _render_system_health_sidebar(report)


def _render_first_run(report: Mapping[str, Any]) -> None:
    """Render one bilingual onboarding screen without running inference."""

    st.title(_label("first_run.title", "Welcome to Amazon Review Analyzer"))
    st.write(
        _label(
            "first_run.subtitle",
            "Local AI review intelligence and competitor analysis.",
        )
    )
    st.caption(
        _label(
            "first_run.local_only",
            "Analysis and saved results stay on this computer.",
        )
    )

    python_check = _safe_check(report, "python")
    ollama_check = _safe_check(report, "ollama")
    model_check = _safe_check(report, "model")
    cards = st.columns(3)
    for card, title, check in (
        (
            cards[0],
            _label("first_run.local_environment", "Local Environment"),
            python_check,
        ),
        (cards[1], _label("first_run.ai_engine", "AI Engine"), ollama_check),
        (cards[2], _label("first_run.ai_model", "AI Model"), model_check),
    ):
        card.metric(title, _health_status_label(check.get("status")))

    ready = _new_analysis_available(report)
    if ready:
        st.success(
            _label(
                "first_run.ready_message",
                "You're ready to analyze reviews.",
            )
        )
    else:
        st.warning(
            _label(
                "first_run.limited_message",
                "You can explore existing insights now and finish local AI setup later.",
            )
        )
        st.caption(
            _label(
                "health.limited_mode_detail",
                "Demo and saved reports are available, but new AI analysis is unavailable.",
            )
        )

    first_row = st.columns(2)
    first_row[0].button(
        _label("first_run.try_demo", "Try Demo Dataset"),
        type="primary",
        on_click=_complete_first_run,
        args=("Demo Dataset",),
        key="first_run_demo",
        use_container_width=True,
    )
    first_row[1].button(
        _label("first_run.saved_analyses", "Saved Analyses"),
        on_click=_complete_first_run,
        args=(SAVED_ANALYSES_MODE,),
        key="first_run_saved",
        use_container_width=True,
    )

    analysis_row = st.columns(3)
    analysis_row[0].button(
        _label("first_run.paste_reviews", "Paste Reviews"),
        on_click=_complete_first_run,
        args=("Analyze New Dataset", PASTE_REVIEWS_SOURCE),
        disabled=not ready,
        key="first_run_paste",
        use_container_width=True,
    )
    analysis_row[1].button(
        _label("first_run.upload_csv", "Upload CSV"),
        on_click=_complete_first_run,
        args=("Analyze New Dataset", UPLOAD_DATASET_SOURCE),
        disabled=not ready,
        key="first_run_upload",
        use_container_width=True,
    )
    analysis_row[2].button(
        _label("first_run.amazon_quick_scan", "Amazon Quick Scan"),
        on_click=_complete_first_run,
        args=("Analyze New Dataset", AMAZON_QUICK_SCAN_SOURCE),
        disabled=not ready,
        key="first_run_amazon",
        use_container_width=True,
    )

    guide_path = (
        Path(__file__).resolve().parent
        / "docs"
        / "Amazon_Review_Analyzer_v1.0_User_Guide_CN.md"
    )
    try:
        guide_data = guide_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        guide_data = f"Amazon Review Analyzer v{get_app_version()}"
    final_row = st.columns(2)
    final_row[0].download_button(
        _label("first_run.documentation", "Documentation"),
        data=guide_data,
        file_name="Amazon_Review_Analyzer_User_Guide.md",
        mime="text/markdown",
        key="first_run_documentation",
        use_container_width=True,
    )
    final_row[1].button(
        _label("first_run.continue_dashboard", "Continue to Dashboard"),
        on_click=_complete_first_run,
        args=("Demo Dataset",),
        key="first_run_continue",
        use_container_width=True,
    )


def _read_local_json(path: Path) -> dict[str, Any]:
    """Read optional local metadata without exposing a technical traceback."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _save_context(
    outputs_dir: Path,
    dataset_label: str,
) -> tuple[str, str, dict[str, Any]]:
    """Return display default, canonical source type, and safe manifest hints."""

    resolved = outputs_dir.resolve()
    metadata: dict[str, Any] = {}
    if resolved == Path(DEFAULT_OUTPUTS_DIR).resolve():
        return "Demo Product", "demo", metadata
    if resolved == Path(amazon.AmazonSessionPaths.current().root).resolve():
        imported = _read_local_json(outputs_dir / "browser_import_metadata.json")
        if not imported:
            imported = _read_local_json(outputs_dir / "scrape_metadata.json")
        asin = str(imported.get("asin") or "").strip()
        source_url = str(
            imported.get("product_url") or imported.get("source_url") or ""
        ).strip()
        if asin:
            metadata.update({"asin": asin, "product_id": asin})
        if source_url:
            metadata["source_url"] = source_url
        return (f"Amazon {asin}" if asin else "Amazon Product"), "amazon", metadata

    selected = str(st.session_state.get("analysis_data_source") or "")
    if selected == PASTE_REVIEWS_SOURCE:
        return "Pasted Reviews", "paste", metadata
    if selected == UPLOAD_DATASET_SOURCE:
        return "Uploaded Product", "upload", metadata
    # Old completed sessions did not persist the input control.  Keep the
    # source canonical and conservative instead of guessing from review text.
    return _label("library.untitled_product", "Untitled Product"), "upload", metadata


def _render_save_analysis_controls(
    outputs_dir: Path,
    *,
    dataset_label: str,
    key_prefix: str,
) -> None:
    """Snapshot one completed dashboard without invoking any pipeline."""

    complete = all(
        (outputs_dir / filename).is_file()
        for filename in library.REQUIRED_ANALYSIS_FILES
    )
    default_name, source_type, metadata = _save_context(outputs_dir, dataset_label)
    name_key = f"{key_prefix}_saved_analysis_product_name"
    if name_key not in st.session_state:
        st.session_state[name_key] = default_name
    with st.expander(_label("library.save_section", "Save This Analysis")):
        st.caption(
            _label(
                "library.local_notice",
                "Saved analyses are stored locally on this computer and are not uploaded to the cloud.",
            )
        )
        product_name = st.text_input(
            _label("library.product_name", "Product Name"),
            key=name_key,
        )
        if not complete:
            st.info(
                _label(
                    "library.incomplete",
                    "This analysis is not complete enough to save yet.",
                )
            )
        if st.button(
            _label("library.save", "Save Analysis"),
            key=f"{key_prefix}_save_analysis",
            type="primary",
            disabled=not complete,
        ):
            try:
                saved = library.save_analysis(
                    outputs_dir,
                    product_name,
                    source_type,
                    metadata,
                    trusted_current_outputs=True,
                )
            except library.DuplicateAnalysisError:
                st.info(
                    _label(
                        "library.already_saved",
                        "This analysis has already been saved.",
                    )
                )
            except library.AnalysisLibraryError:
                st.error(
                    _label(
                        "error.operation_failed",
                        "The analysis could not be saved. Please try again.",
                    )
                )
            else:
                st.success(
                    _label(
                        "library.saved_success",
                        "Analysis saved as {product_name}.",
                        product_name=saved.product_name,
                    )
                )


def _render_saved_analyses_page() -> None:
    """Show the lightweight local Analysis Library index."""

    st.title(_label("library.title", "Saved Analyses"))
    st.caption(
        _label(
            "library.subtitle",
            "Keep completed product analyses as stable local snapshots for future comparison.",
        )
    )
    st.info(
        _label(
            "library.local_notice",
            "Saved analyses are stored locally on this computer and are not uploaded to the cloud.",
        ),
        icon="🔒",
    )
    saved = library.list_saved_analyses()
    if not saved:
        st.info(
            _label(
                "library.empty",
                "No analyses have been saved yet. Open a completed dataset and choose Save Analysis.",
            )
        )
        return
    rows = []
    for item in saved:
        manifest = item.manifest
        rows.append(
            {
                _label("library.product", "Product Name"): item.product_name,
                _label("library.source", "Source"): display_analysis_source_type(
                    manifest.get("source_type"), _language()
                ),
                _label("library.reviews", "Reviews"): _safe_int(
                    manifest.get("imported_reviews")
                ),
                _label("library.coverage", "Coverage"): (
                    f"{_safe_float(manifest.get('coverage')):.1%}"
                ),
                _label("library.saved_at", "Saved"): str(
                    manifest.get("saved_at") or "—"
                ),
                _label("library.analysis_id", "Analysis ID"): item.analysis_id,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        _label(
            "library.count",
            "{count} saved analyses",
            count=len(saved),
        )
    )

    st.subheader(_label("report.saved_reports", "Generate a Saved Analysis Report"))
    st.caption(
        _label(
            "report.saved_reports_notice",
            "Generate a report directly from a verified snapshot without reopening or reanalyzing it.",
        )
    )
    report_language = _report_language_control("saved_library")
    for item in saved:
        manifest = item.manifest
        product_name = item.product_name
        analysis_id = item.analysis_id
        safe_key = "".join(
            character if character.isalnum() else "_"
            for character in analysis_id
        ).lower()
        state_key = f"saved_library_{safe_key}_report_document"
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"**{product_name}**")
            st.caption(
                f"{analysis_id} · "
                f"{display_analysis_source_type(manifest.get('source_type'), _language())}"
            )
        with right:
            generate = st.button(
                _label("report.generate", "Generate Report"),
                key=f"saved_library_{safe_key}_generate_report",
            )
        if generate:
            try:
                document, exported = _generate_single_product_report(
                    item.root,
                    language=report_language,
                    product_name=product_name,
                    validate_integrity=True,
                )
            except Exception as exc:  # Integrity/report failures remain local to this row.
                st.error(_report_error_message(exc))
            else:
                st.session_state[state_key] = document
                st.session_state[f"{state_key}_export"] = exported
                st.success(
                    _label(
                        "report.generated_for",
                        "Report generated for {product_name}.",
                        product_name=product_name,
                    )
                )
        document = st.session_state.get(state_key)
        html_document = getattr(document, "html", None)
        report_id = str(getattr(document, "report_id", "report"))
        if isinstance(html_document, str) and html_document:
            st.download_button(
                _label("report.download_html", "Download HTML Report"),
                data=html_document,
                file_name=f"{report_id}.html",
                mime="text/html",
                key=f"saved_library_{safe_key}_download_report",
            )


def _select_current_analysis() -> None:
    st.session_state["analysis_workspace"] = UPLOAD_WORKSPACE


def _select_amazon_analysis() -> None:
    st.session_state["analysis_workspace"] = AMAZON_WORKSPACE


def _select_analysis_data_source(source: str) -> None:
    """Route a fallback button to an existing Step 10A input workflow."""

    source_aliases = {
        "Paste Reviews": PASTE_REVIEWS_SOURCE,
        "Upload CSV": UPLOAD_DATASET_SOURCE,
        "Upload Dataset": UPLOAD_DATASET_SOURCE,
    }
    if source not in source_aliases:
        raise ValueError("Unsupported analysis data source")
    current_url = st.session_state.get("amazon_product_url", "")
    if isinstance(current_url, str):
        st.session_state[PRESERVED_AMAZON_URL_KEY] = current_url
    _close_amazon_browser_session(clear_result=True)
    st.session_state["dataset_mode"] = "Analyze New Dataset"
    st.session_state["analysis_workspace"] = "New Input"
    st.session_state["analysis_data_source"] = source_aliases[source]


def _reset_current_analysis() -> None:
    try:
        interactive.reset_current_session()
    except runtime.AnalysisRunBusyError:
        st.warning(_label("error.current_busy", "Current Analysis is still running. Stop or finish it before resetting."))
        return
    _load_cached_dashboard.clear()
    st.session_state["analysis_workspace"] = "New Input"
    st.session_state.pop("confirm_session_reset", None)


def _reset_amazon_analysis() -> None:
    _close_amazon_browser_session(clear_result=True)
    try:
        amazon.reset_amazon_analysis()
    except runtime.AnalysisRunBusyError:
        st.warning(_label("error.amazon_busy", "Amazon analysis is still running. Stop or finish it before resetting."))
        return
    _load_cached_dashboard.clear()
    st.session_state["analysis_workspace"] = "New Input"
    st.session_state.pop("confirm_amazon_reset", None)


def _close_amazon_browser_session(*, clear_result: bool) -> None:
    """Close one ephemeral browser without touching imported or Demo data."""

    session = st.session_state.pop(AMAZON_BROWSER_SESSION_KEY, None)
    if isinstance(session, AmazonBrowserSession):
        try:
            session.close()
        except Exception:
            # Cleanup is best-effort and must not expose a browser traceback.
            pass
    if clear_result:
        st.session_state.pop(AMAZON_BROWSER_RESULT_KEY, None)


def _render_sidebar_controls(
    current_available: bool,
    amazon_available: bool,
    amazon_present: bool,
    health_result: Mapping[str, Any] | None = None,
) -> tuple[str, str | None, str, bool]:
    with st.sidebar:
        language_codes = list(SUPPORTED_LANGUAGES.values())
        if st.session_state.get(LANGUAGE_STATE_KEY) not in language_codes:
            st.session_state[LANGUAGE_STATE_KEY] = DEFAULT_LANGUAGE
        st.selectbox(
            t("language.selector", _language()),
            language_codes,
            format_func=lambda code: next(
                (label for label, value in SUPPORTED_LANGUAGES.items() if value == code),
                code,
            ),
            key=LANGUAGE_STATE_KEY,
        )
        st.title(_label("app.analysis_settings", "Analysis Settings"))
        dataset_mode = st.radio(
            _label("dataset.mode", "Dataset Mode"),
            [
                "Demo Dataset",
                "Analyze New Dataset",
                SAVED_ANALYSES_MODE,
                COMPARE_PRODUCTS_MODE,
            ],
            index=0,
            key="dataset_mode",
            format_func=lambda value: {
                "Demo Dataset": _label("dataset.demo", "Demo Dataset"),
                "Analyze New Dataset": _label("dataset.analyze_new", "Analyze New Dataset"),
                SAVED_ANALYSES_MODE: _label("dataset.saved_analyses", SAVED_ANALYSES_MODE),
                COMPARE_PRODUCTS_MODE: _label("dataset.compare_products", COMPARE_PRODUCTS_MODE),
            }[value],
        )
        data_source: str | None = None
        workspace = "New Input"
        if dataset_mode == "Analyze New Dataset":
            workspace_options = ["New Input"]
            if current_available:
                workspace_options.append(UPLOAD_WORKSPACE)
            if amazon_available:
                workspace_options.append(AMAZON_WORKSPACE)
            if len(workspace_options) > 1:
                if st.session_state.get("analysis_workspace") not in workspace_options:
                    st.session_state["analysis_workspace"] = "New Input"
                workspace = st.radio(
                    _label("dataset.workspace", "Workspace"),
                    workspace_options,
                    key="analysis_workspace",
                    format_func=lambda value: {
                        "New Input": _label("dataset.new_input", "New Input"),
                        UPLOAD_WORKSPACE: _label("dataset.current_analysis", UPLOAD_WORKSPACE),
                        AMAZON_WORKSPACE: _label("dataset.current_amazon", AMAZON_WORKSPACE),
                    }[value],
                )
            else:
                st.caption(_label("dataset.none_yet", "No custom analysis has been run yet."))
                st.session_state["analysis_workspace"] = "New Input"

            if workspace == "New Input":
                source_options = [
                    AMAZON_QUICK_SCAN_SOURCE,
                    UPLOAD_DATASET_SOURCE,
                    PASTE_REVIEWS_SOURCE,
                ]
                legacy_source = st.session_state.get("analysis_data_source")
                if legacy_source == "Amazon URL":
                    st.session_state["analysis_data_source"] = (
                        AMAZON_QUICK_SCAN_SOURCE
                    )
                elif legacy_source == "Upload CSV":
                    st.session_state["analysis_data_source"] = (
                        UPLOAD_DATASET_SOURCE
                    )
                if st.session_state.get("analysis_data_source") not in source_options:
                    st.session_state["analysis_data_source"] = (
                        AMAZON_QUICK_SCAN_SOURCE
                    )
                data_source = st.radio(
                    _label("dataset.source", "Choose Analysis Source"),
                    source_options,
                    key="analysis_data_source",
                    format_func=lambda value: {
                        AMAZON_QUICK_SCAN_SOURCE: _label("dataset.amazon_quick_scan", AMAZON_QUICK_SCAN_SOURCE),
                        UPLOAD_DATASET_SOURCE: _label("dataset.upload", UPLOAD_DATASET_SOURCE),
                        PASTE_REVIEWS_SOURCE: _label("dataset.paste", PASTE_REVIEWS_SOURCE),
                    }[value],
                )

            if current_available:
                with st.expander(_label("dataset.reset_current", "Reset Current Analysis")):
                    confirmed = st.checkbox(
                        _label("dataset.reset_current_confirm", "I understand this removes only the current session."),
                        key="confirm_session_reset",
                    )
                    st.button(
                        _label("dataset.reset_current", "Reset Current Analysis"),
                        disabled=not confirmed,
                        on_click=_reset_current_analysis,
                        key="reset_current_analysis_button",
                    )

            if amazon_present:
                with st.expander(_label("dataset.reset_amazon", "Reset Amazon Analysis")):
                    confirmed = st.checkbox(
                        _label("dataset.reset_amazon_confirm", "I understand this removes only Amazon Current."),
                        key="confirm_amazon_reset",
                    )
                    st.button(
                        _label("dataset.reset_amazon", "Reset Amazon Analysis"),
                        disabled=not confirmed,
                        on_click=_reset_amazon_analysis,
                        key="reset_amazon_analysis_button",
                    )

        compact_view = st.toggle(
            _label("app.compact_view", "Compact interview view"),
            value=True,
            help=_label("app.compact_view_help", "Shows fewer review cards and keeps technical metadata compact."),
            key="compact_interview_view",
        )
        st.caption(
            _label(
                "app.version",
                "Amazon Review Analyzer v{version}",
                version=get_app_version(),
            )
        )
    _render_model_information_sidebar(health_result)
    return dataset_mode, data_source, workspace, compact_view


def _render_saved_dashboard(
    outputs_dir: Path,
    *,
    dataset_label: str,
    key_prefix: str,
    compact_view: bool,
    quick_scan: bool = False,
) -> None:
    data = _load_cached_dashboard(
        str(outputs_dir),
        output_fingerprint(outputs_dir),
    )
    kpis = calculate_kpis(data)
    _render_dataset_summary_sidebar(
        data,
        kpis,
        quick_scan=quick_scan,
        outputs_dir=outputs_dir,
    )

    safe_label = html.escape(_display_dataset_label(dataset_label))
    app_title = html.escape(_label("app.title", "Amazon Review Intelligence"))
    app_subtitle = html.escape(_label("app.subtitle", "Turn customer reviews into actionable product insights."))
    dataset_word = html.escape(_label("app.dataset", "Dataset"))
    st.markdown(
        f"""
        <div class="hero">
          <h1>{app_title}</h1>
          <p>{app_subtitle}</p>
          <div class="source">{dataset_word}: {safe_label} · analyzed_reviews.csv</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data.analyzed_reviews is None:
        _show_output_issue(data, "analyzed_reviews")
        return

    _render_save_analysis_controls(
        outputs_dir,
        dataset_label=dataset_label,
        key_prefix=key_prefix,
    )
    _render_single_report_controls(
        outputs_dir,
        product_name=str(
            st.session_state.get(f"{key_prefix}_saved_analysis_product_name")
            or _save_context(outputs_dir, dataset_label)[0]
        ),
        key_prefix=key_prefix,
        # Current dashboards are mutable completed-output workspaces rather
        # than frozen library snapshots.  The generator still hashes them,
        # but manifest integrity enforcement belongs to the Saved page.
        validate_integrity=False,
    )

    if quick_scan:
        _render_quick_scan_disclaimer()

    _render_kpis(kpis)
    sample = (
        data.business_intelligence.get("sample_level", {})
        if isinstance(data.business_intelligence, Mapping)
        else {}
    )
    if isinstance(sample, Mapping):
        sample_key = str(sample.get("key", ""))
        sample_label = display_sample_level(sample_key or sample.get("label", ""), _language())
        sample_message = _label({
            "early_signals": "sample.early_message",
            "limited_sample": "sample.limited_message",
            "full_insight": "sample.full_message",
        }.get(sample_key, "sample.full_message"), str(sample.get("message", "")))
        if sample.get("key") in {"early_signals", "limited_sample"}:
            st.warning(f"{sample_label} — {sample_message}", icon="⚠️")
        elif sample_message:
            st.caption(f"{sample_label} — {sample_message}")
    verified = (
        data.business_intelligence.get("verified_purchase", {})
        if isinstance(data.business_intelligence, Mapping)
        else {}
    )
    if isinstance(verified, Mapping) and verified.get("available"):
        st.caption(_label(
            "overview.verified_metadata",
            "Verified purchase metadata: {rate} of {count} reviews with known status. This is metadata, not a credibility score.",
            rate=f"{_safe_float(verified.get('verified_rate')):.1f}%",
            count=_safe_int(verified.get("known_reviews")),
        ))
    _render_data_quality_notice(data, kpis)

    overview, insights, language, deep_dive, reviews = st.tabs(
        [
            _label("tab.overview", "Overview"),
            _label("tab.product_insights", "Product Insights"),
            _label("tab.customer_language", "Customer Language"),
            _label("tab.aspect_deep_dive", "Aspect Deep Dive"),
            _label("tab.review_explorer", "Review Explorer"),
        ]
    )
    with overview:
        _render_overview(data, compact_view, key_prefix)
    with insights:
        _render_insights(data)
    with language:
        _render_customer_language(data, key_prefix)
    with deep_dive:
        _render_aspect_deep_dive(data, key_prefix)
    with reviews:
        _render_review_explorer(data, compact_view, key_prefix)

    st.markdown(
        f"""
        <div class="footer">
          {_label('app.footer_built', 'Built with Python, Streamlit and local Qwen3 via Ollama.')}<br>
          {_label('app.footer_local', 'Review analysis is performed locally.')}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _column_mapping_selectbox(
    label: str,
    columns: list[str],
    suggested: str | None,
    *,
    required: bool,
    key: str,
) -> str | None:
    placeholder = _label("input.select_column", "Select a column…") if required else _label("input.not_provided", "(Not provided)")
    options = [placeholder, *columns]
    index = options.index(suggested) if suggested in options else 0
    selected = st.selectbox(label, options, index=index, key=key)
    return None if selected == placeholder else selected


def _render_upload_input() -> tuple[pd.DataFrame, str] | None:
    st.caption(_label("input.upload_caption", "Upload a larger review dataset for deeper analysis without Amazon Quick Scan visibility limits."))
    uploaded = st.file_uploader(
        _label("input.upload_reviews_csv", "Upload Reviews CSV"),
        type=["csv"],
        help=_label("input.upload_help", "Maximum file size for this demo: 10 MB."),
        key="review_csv_upload",
    )
    if uploaded is None:
        st.info(_label("input.choose_csv", "Choose a CSV file to preview and map its columns."))
        return None

    payload = uploaded.getvalue()
    try:
        source = interactive.read_uploaded_csv(payload)
    except interactive.InteractiveInputError as exc:
        st.error(
            _label(
                "input.validation_error",
                "The reviews could not be validated: {error}",
                error=exc,
            )
        )
        return None

    columns = [str(column) for column in source.columns]
    token = hashlib.sha256(payload).hexdigest()[:12]
    suggestions = interactive.suggest_column_mapping(columns)
    first, second = st.columns(2)
    first.metric(_label("input.rows", "Rows"), f"{len(source):,}")
    second.metric(_label("input.columns", "Columns"), f"{len(columns):,}")
    st.caption(_label("input.reviews_uploaded", "{count} reviews uploaded", count=f"{len(source):,}"))
    st.markdown(f"**{_label('input.detected_columns', 'Detected columns')}**")
    st.write(", ".join(columns))
    st.dataframe(source.head(5), hide_index=True, width="stretch")

    st.subheader(_label("input.column_mapping", "Column Mapping"))
    st.caption(_label("input.mapping_caption", "Review Content is required. Every other field is optional."))
    mapping: dict[str, str | None] = {}
    mapping["content"] = _column_mapping_selectbox(
        _label("input.review_content", "Review Content"),
        columns,
        suggestions["content"],
        required=True,
        key=f"upload_{token}_content",
    )
    mapping_columns = st.columns(2)
    definitions = (
        ("review_id", _label("common.review_id", "Review ID")),
        ("rating", _label("rating.rating", "Rating")),
        ("title", _label("common.title", "Title")),
        ("date", _label("common.date", "Date")),
        ("verified_purchase", _label("explorer.verified_purchase", "Verified Purchase")),
    )
    for position, (field, label) in enumerate(definitions):
        with mapping_columns[position % 2]:
            mapping[field] = _column_mapping_selectbox(
                label,
                columns,
                suggestions[field],
                required=False,
                key=f"upload_{token}_{field}",
            )

    try:
        normalized = interactive.normalize_uploaded_reviews(source, mapping)
    except interactive.InteractiveInputError as exc:
        st.error(
            _label(
                "input.validation_error",
                "The reviews could not be validated: {error}",
                error=exc,
            )
        )
        return None
    mapping_token = hashlib.sha256(
        (token + repr(sorted(mapping.items()))).encode("utf-8")
    ).hexdigest()[:12]
    return normalized, f"upload_{mapping_token}"


def _render_paste_input() -> tuple[pd.DataFrame, str] | None:
    st.caption(_label("input.paste_caption", "Paste one or more reviews directly for analysis."))
    pasted = st.text_area(
        _label("input.paste_reviews", "Paste Reviews"),
        placeholder=_label(
            "input.paste_example",
            "The picture quality is excellent.\nThe remote feels cheap.\nSetup was easy and fast.",
        ),
        height=220,
        help=_label("input.paste_placeholder", "Enter one review per non-empty line."),
        key="pasted_reviews",
    )
    if not pasted.strip():
        st.info(_label("input.paste_preview", "Paste one review per line to see a preview."))
        return None
    try:
        normalized = interactive.parse_pasted_reviews(pasted)
    except interactive.InteractiveInputError as exc:
        st.error(
            _label(
                "input.validation_error",
                "The reviews could not be validated: {error}",
                error=exc,
            )
        )
        return None
    detected_key = "input.review_detected" if len(normalized) == 1 else "input.reviews_detected"
    detected_fallback = "{count} review detected" if len(normalized) == 1 else "{count} reviews detected"
    st.success(_label(detected_key, detected_fallback, count=f"{len(normalized):,}"))
    st.dataframe(
        _localized_review_preview(
            normalized.loc[:, ["review_id", "content"]].head(8)
        ),
        hide_index=True,
        width="stretch",
    )
    token = hashlib.sha256(pasted.encode("utf-8")).hexdigest()[:12]
    return normalized, f"paste_{token}"


def _runtime_progress_renderer() -> runtime.ProgressSnapshotCallback:
    """Create the shared long-task progress surface for every input source."""

    heading = st.empty()
    progress_bar = st.progress(0.0)
    count_slots = [column.empty() for column in st.columns(3)]
    current_slot = st.empty()
    timing_slot = st.empty()
    state_labels = {
        runtime.AnalysisState.VALIDATING: _label("runtime.validating", "Validating reviews"),
        runtime.AnalysisState.PREPARING: _label("runtime.preparing", "Preparing analysis"),
        runtime.AnalysisState.ANALYZING: _label("runtime.analyzing", "Analyzing reviews"),
        runtime.AnalysisState.RETRYING: _label("runtime.retrying", "Retrying failed reviews"),
        runtime.AnalysisState.AGGREGATING: _label("runtime.aggregating", "Aggregating product insights"),
        runtime.AnalysisState.GENERATING_KEYWORDS: _label("runtime.keywords", "Generating customer language"),
        runtime.AnalysisState.COMPLETED: _label("runtime.complete", "Analysis completed"),
        runtime.AnalysisState.COMPLETED_WITH_FAILURES: _label("runtime.completed_failures", "Analysis completed with some failures"),
        runtime.AnalysisState.INTERRUPTED: _label("runtime.interrupted_short", "Analysis interrupted"),
        runtime.AnalysisState.FAILED: _label("runtime.failed_analysis", "Analysis failed"),
    }

    def render(snapshot: runtime.ProgressSnapshot) -> None:
        label = state_labels.get(snapshot.state, snapshot.state.value.replace("_", " ").title())
        operation_total = snapshot.operation_total or snapshot.total_reviews
        operation_completed = (
            snapshot.operation_completed
            if snapshot.operation_total
            else snapshot.completed_reviews
        )
        heading.markdown(
            f"**{label}: {operation_completed:,} / {operation_total:,} "
            f"({snapshot.percent_complete:.0%})**"
        )
        progress_bar.progress(snapshot.percent_complete)
        count_slots[0].metric(_label("runtime.successfully_analyzed", "Successful"), snapshot.successful)
        count_slots[1].metric(_label("runtime.failed", "Failed"), snapshot.failed)
        count_slots[2].metric(_label("runtime.skipped", "Skipped"), snapshot.skipped)
        if snapshot.current_review_id and snapshot.state in {
            runtime.AnalysisState.ANALYZING,
            runtime.AnalysisState.RETRYING,
        }:
            current_slot.caption(
                _label("runtime.current_review", "Currently processing review ID: {review_id}", review_id=snapshot.current_review_id)
            )
        else:
            current_slot.empty()
        elapsed = display_duration(snapshot.elapsed_seconds, _language())
        if snapshot.eta_seconds is None and snapshot.operation_completed < 3:
            eta = _label("runtime.estimating", "Estimating remaining time...")
        else:
            eta = _label("runtime.estimated_remaining_value", "Estimated time remaining: ~{duration}", duration=display_duration(snapshot.eta_seconds, _language()))
        timing_slot.caption(_label("runtime.elapsed_value", "Elapsed: {elapsed} · {eta}", elapsed=elapsed, eta=eta))

    return render


def _execute_interactive_analysis(
    reviews: pd.DataFrame | None,
    key_token: str,
    *,
    resume: bool = False,
    retry_failed: bool = False,
    health_result: Mapping[str, Any] | None = None,
) -> interactive.InteractivePipelineResult | None:
    if resume and retry_failed:
        raise ValueError("resume and retry_failed cannot both be true")
    report = health_result if health_result is not None else _health_report()
    if not _new_analysis_available(report):
        _render_ai_unavailable_notice(report)
        return None
    try:
        with st.spinner(_label("runtime.checking_model", "Checking the local AI model…")):
            client, model_status = interactive.create_checked_ollama_client()
    except Exception as exc:
        st.error(_friendly_local_ai_error(exc))
        return None

    st.success(_label("runtime.model_ready", "Ollama is ready. Model found: {model}", model=model_status.get("model", "qwen3:8b")))
    update_runtime_progress = _runtime_progress_renderer()

    try:
        with st.spinner(_label("runtime.analyzing_model", "Analyzing reviews with local Qwen3 8B…")):
            if retry_failed:
                result = interactive.retry_failed_reviews(
                    llm_client=client,
                    progress_snapshot_callback=update_runtime_progress,
                    min_keyword_reviews=1,
                )
            else:
                result = interactive.run_interactive_pipeline(
                    reviews,
                    llm_client=client,
                    resume=resume,
                    progress_snapshot_callback=update_runtime_progress,
                    source_type=(
                        "upload" if key_token.startswith("upload_") else "paste"
                    ),
                    # Small interactive samples still need useful language output.
                    min_keyword_reviews=1,
                )
    except Exception as exc:
        friendly = _friendly_local_ai_error(exc)
        if friendly.startswith("The local AI analysis could not be started"):
            friendly = _label(
                "runtime.analysis_complete_error",
                "Analysis could not be completed. Please try again.",
            )
        st.error(friendly)
        return None

    st.success(_label("runtime.complete", "Analysis Complete"), icon="✅")
    summary_columns = st.columns(5)
    summary_columns[0].metric(_label("runtime.submitted", "Submitted"), result.submitted_reviews)
    summary_columns[1].metric(_label("runtime.successfully_analyzed", "Successful"), result.successful_reviews)
    summary_columns[2].metric(_label("runtime.failed", "Failed"), result.failed_reviews)
    summary_columns[3].metric(_label("runtime.skipped", "Skipped"), result.skipped_reviews)
    summary_columns[4].metric(_label("dataset.coverage", "Coverage"), f"{result.coverage:.1%}")
    st.caption(_label("runtime.completed_in", "Completed in {duration}", duration=display_duration(result.elapsed_seconds, _language())))
    _load_cached_dashboard.clear()
    if not retry_failed:
        st.button(
            _label("runtime.view_results", "View Results"),
            type="primary",
            on_click=_select_current_analysis,
            key=f"view_results_{key_token}",
        )
    return result


def _execute_amazon_analysis(
    *,
    resume: bool = False,
    retry_failed: bool = False,
    health_result: Mapping[str, Any] | None = None,
) -> amazon.AmazonPipelineResult | None:
    if resume and retry_failed:
        raise ValueError("resume and retry_failed cannot both be true")
    report = health_result if health_result is not None else _health_report()
    if not _new_analysis_available(report):
        _render_ai_unavailable_notice(report, imported=True)
        return None
    try:
        with st.spinner(_label("runtime.checking_model", "Checking the local AI model…")):
            client, model_status = interactive.create_checked_ollama_client()
    except Exception as exc:
        st.error(_friendly_local_ai_error(exc))
        return None

    st.success(_label("runtime.model_ready", "Ollama is ready. Model found: {model}", model=model_status.get("model", "qwen3:8b")))
    update_runtime_progress = _runtime_progress_renderer()

    try:
        with st.spinner(_label("runtime.analyzing_amazon", "Analyzing imported Amazon reviews with local Qwen3 8B…")):
            if retry_failed:
                result = amazon.retry_failed_amazon_reviews(
                    llm_client=client,
                    progress_snapshot_callback=update_runtime_progress,
                    min_keyword_reviews=1,
                )
            else:
                result = amazon.run_amazon_pipeline(
                    llm_client=client,
                    resume=resume,
                    progress_snapshot_callback=update_runtime_progress,
                    min_keyword_reviews=1,
                )
    except Exception as exc:
        friendly = _friendly_local_ai_error(exc)
        if friendly.startswith("The local AI analysis could not be started"):
            friendly = _label(
                "runtime.amazon_analysis_complete_error",
                "Amazon analysis could not be completed. Please try again.",
            )
        st.error(friendly)
        return None

    st.success(_label("runtime.amazon_complete", "Amazon Analysis Complete"), icon="✅")
    summary_columns = st.columns(5)
    summary_columns[0].metric(_label("runtime.submitted", "Submitted"), result.submitted_reviews)
    summary_columns[1].metric(_label("runtime.successfully_analyzed", "Successful"), result.successful_reviews)
    summary_columns[2].metric(_label("runtime.failed", "Failed"), result.failed_reviews)
    summary_columns[3].metric(_label("runtime.skipped", "Skipped"), result.skipped_reviews)
    summary_columns[4].metric(_label("dataset.coverage", "Coverage"), f"{result.coverage:.1%}")
    st.caption(_label("runtime.completed_in", "Completed in {duration}", duration=display_duration(result.elapsed_seconds, _language())))
    _load_cached_dashboard.clear()
    if not retry_failed:
        st.button(
            _label("runtime.view_amazon_results", "View Amazon Results"),
            type="primary",
            on_click=_select_amazon_analysis,
            key="view_amazon_results",
        )
    return result


def _render_failed_review_retry_panel(
    outputs_dir: Path,
    *,
    source: str = "upload",
    health_result: Mapping[str, Any] | None = None,
) -> None:
    if source == "amazon":
        failed = amazon.failed_amazon_review_count(outputs_dir)
    else:
        failed = interactive.failed_review_count(outputs_dir)
    if failed <= 0:
        return

    warning_slot = st.empty()
    button_slot = st.empty()
    warning_slot.warning(t_count("runtime.failed_count", "runtime.failed_count_plural", failed, _language()))
    ready = (
        True
        if health_result is None
        else _new_analysis_available(health_result)
    )
    if not ready:
        _render_ai_unavailable_notice(
            health_result,
            imported=source == "amazon",
        )
    button_options: dict[str, Any] = {
        "type": "primary",
        "key": f"retry_failed_reviews_button_{source}",
    }
    if health_result is not None:
        button_options["disabled"] = not ready
    if button_slot.button(
        _label("runtime.retry_failed", "Retry Failed Reviews"),
        **button_options,
    ):
        if source == "amazon":
            if health_result is None:
                result = _execute_amazon_analysis(retry_failed=True)
            else:
                result = _execute_amazon_analysis(
                    retry_failed=True,
                    health_result=health_result,
                )
        else:
            if health_result is None:
                result = _execute_interactive_analysis(
                    None,
                    "retry_failed",
                    retry_failed=True,
                )
            else:
                result = _execute_interactive_analysis(
                    None,
                    "retry_failed",
                    retry_failed=True,
                    health_result=health_result,
                )
        if result is None:
            return
        if result.failed_reviews:
            warning_slot.warning(t_count("runtime.failed_count", "runtime.failed_count_plural", result.failed_reviews, _language()))
        else:
            warning_slot.success(_label("runtime.retry_success", "All failed reviews were analyzed successfully."))
            button_slot.empty()


def _render_ai_unavailable_notice(
    report: Mapping[str, Any] | None,
    *,
    imported: bool = False,
) -> None:
    """Explain Limited Mode without exposing provider exceptions."""

    source = report if report is not None else _health_report()
    model = _safe_check(source, "model")
    ollama = _safe_check(source, "ollama")
    if str(model.get("code") or "") in {
        "model_missing",
        "configured_model_missing",
    }:
        st.warning(
            _label(
                "health.model_missing",
                "{model} is required for new review analysis.",
                model=OLLAMA_MODEL,
            )
        )
        st.code(f"ollama pull {OLLAMA_MODEL}", language="text")
    elif str(ollama.get("code") or "") == "ollama_not_installed":
        st.warning(
            _label(
                "health.ollama_not_installed",
                "Ollama is not installed.",
            )
        )
    else:
        st.warning(
            _label(
                "health.new_analysis_unavailable",
                "Local AI is currently unavailable. Your existing analyses are safe. Start Ollama and refresh System Status to try again.",
            )
        )
    st.caption(
        _label(
            (
                "health.import_saved_unavailable"
                if imported
                else "health.analysis_saved_unavailable"
            ),
            (
                "Reviews were imported successfully, but local AI analysis is currently unavailable. The imported reviews are preserved."
                if imported
                else "Your input is preserved. New AI analysis can continue after the local AI engine is ready."
            ),
        )
    )


def _render_analysis_action(
    reviews: pd.DataFrame,
    key_token: str,
    health_result: Mapping[str, Any] | None = None,
) -> None:
    validation = interactive.validate_interactive_reviews(reviews)
    if not validation["is_valid"]:
        for error in validation["errors"]:
            st.error(
                _label(
                    "input.validation_error",
                    "The reviews could not be validated: {error}",
                    error=error,
                )
            )
        return

    if message := interactive.large_dataset_message(len(reviews)):
        st.warning(
            _label(
                "runtime.large_dataset_warning",
                message,
                count=f"{len(reviews):,}",
            )
        )
        st.caption(_label("runtime.large_dataset_help", "Use Ctrl+C in the launcher Terminal to stop safely. The latest checkpoint can be resumed later."))

    # Upload and Paste are full-dataset workflows.  The Amazon Quick Scan
    # safety cap is deliberately not applied to user-provided datasets.
    bounded = reviews.copy(deep=True)
    ready = (
        True
        if health_result is None
        else _new_analysis_available(health_result)
    )
    if not ready:
        _render_ai_unavailable_notice(health_result)
    clicked = st.button(
        _label("runtime.start", "Analyze Reviews"),
        type="primary",
        disabled=bounded.empty or not ready,
        key=f"analyze_reviews_{key_token}",
    )
    if clicked:
        _execute_interactive_analysis(
            bounded,
            key_token,
            health_result=health_result,
        )


def _friendly_amazon_import_error(status: str, reason: str | None = None) -> str:
    """Translate scraper states into concise, actionable UI messages."""

    if status == ScrapeStatus.BLOCKED and reason == AMAZON_CHALLENGE_REASON:
        return _label("amazon.challenge", AMAZON_CHALLENGE_MESSAGE.splitlines()[0])
    if status == ScrapeStatus.BLOCKED:
        return _label("amazon.blocked", (
            "Amazon did not allow this review-page request. "
            "Please try again later or use another input method."
        ))
    if status == ScrapeStatus.DNS_ERROR:
        return _label("amazon.dns_error", "Unable to resolve Amazon hostname. Please check the network connection.")
    if status == ScrapeStatus.CONNECTION_TIMEOUT:
        return _label("amazon.timeout", "Connection to Amazon timed out. Please try again later.")
    if status == ScrapeStatus.CONNECTION_REFUSED:
        return _label("amazon.connection_refused", "Amazon refused the connection. Please try again later.")
    if status == ScrapeStatus.TLS_ERROR:
        if reason == TRUSTED_CA_BUNDLE_UNAVAILABLE:
            return _label("amazon.ca_unavailable", "Trusted CA certificate bundle is unavailable.")
        return _label("amazon.tls_detail", "TLS connection to Amazon failed. Check this computer's certificates.")
    if status == ScrapeStatus.PROXY_ERROR:
        return _label("amazon.proxy_error", "The configured proxy could not reach Amazon.")
    if status == ScrapeStatus.REDIRECT_ERROR:
        return _label("amazon.redirect_error", "Amazon's redirect could not be completed safely.")
    if status == ScrapeStatus.HTTP_ERROR:
        code = reason.removeprefix("http_") if reason else "unknown"
        return _label("amazon.http_error", "Amazon returned an HTTP error ({code}). Please try again later.", code=code)
    if status == ScrapeStatus.CONNECTION_ERROR:
        return _label("amazon.connection_error", (
            "Unable to reach Amazon right now. Please try again later, or "
            "upload a CSV or paste reviews instead."
        ))
    if status == ScrapeStatus.NO_REVIEWS:
        return _label("amazon.no_reviews", "No public reviews were found for this product.")
    if status == ScrapeStatus.STRUCTURE_NOT_RECOGNIZED:
        return _label("amazon.structure_detail", (
            "Amazon's review-page structure was not recognized. "
            "You can upload a CSV or paste reviews instead."
        ))
    return _label("amazon.generic_error", (
        "Amazon reviews could not be imported. Please try again later, or "
        "upload a CSV or paste reviews instead."
    ))


def _render_amazon_import_issue(status: str, reason: str | None = None) -> None:
    """Render one import outcome; only a true challenge gets fallback actions."""

    message = _friendly_amazon_import_error(status, reason)
    if status == ScrapeStatus.BLOCKED and reason == AMAZON_CHALLENGE_REASON:
        st.warning(f"{_label('amazon.challenge', message.splitlines()[0])}\n\n{_label('amazon.challenge_fallback', 'You can still analyze this product by pasting reviews or uploading a CSV.')}")
        paste, upload = st.columns(2)
        paste.button(
            _label("amazon.paste_instead", "Paste Reviews Instead"),
            on_click=_select_analysis_data_source,
            args=(PASTE_REVIEWS_SOURCE,),
            use_container_width=True,
            key="amazon_challenge_paste_fallback",
        )
        upload.button(
            _label("amazon.upload_instead", "Upload Dataset Instead"),
            on_click=_select_analysis_data_source,
            args=(UPLOAD_DATASET_SOURCE,),
            use_container_width=True,
            key="amazon_challenge_upload_fallback",
        )
        return
    if status == ScrapeStatus.NO_REVIEWS:
        st.info(message)
    elif status == ScrapeStatus.STRUCTURE_NOT_RECOGNIZED:
        st.warning(message)
    else:
        st.error(message)


def _render_amazon_import_preview(
    health_result: Mapping[str, Any] | None = None,
) -> bool:
    """Show the persisted Amazon import and return whether it is available."""

    if not amazon.amazon_import_available():
        return False
    try:
        imported = amazon.load_amazon_import()
    except amazon.AmazonAnalysisError:
        st.error(_label("error.operation_failed", "The saved Amazon import could not be loaded. Please try again."))
        return False

    metadata = imported.metadata
    st.subheader(_label("dataset.amazon_quick_scan", "Amazon Quick Scan"))
    summary = st.columns(3)
    summary[0].metric("ASIN", str(metadata.get("asin") or "—"))
    summary[1].metric(
        _label("amazon.marketplace", "Marketplace"),
        str(metadata.get("marketplace") or "—"),
    )
    summary[2].metric(_label("amazon.reviews_retrieved", "Reviews Retrieved"), f"{len(imported.reviews):,}")
    availability_status = str(metadata.get("availability_status") or "unknown")
    if availability_status == ReviewAvailabilityStatus.ACCESS_LIMITED:
        st.warning(_label("amazon.access_limited_detail", "Amazon currently exposes a limited set of reviews to this browser session. Quick Scan will analyze the reviews that are currently accessible."))
    st.caption(_label("amazon.preview_first", "Previewing the first 5 imported reviews."))
    preferred_columns = [
        "review_id",
        "rating",
        "title",
        "content",
        "date",
        "verified_purchase",
    ]
    preview_columns = [
        column for column in preferred_columns if column in imported.reviews.columns
    ]
    preview = _localized_review_preview(
        imported.reviews.loc[:, preview_columns].head(5)
    )
    st.dataframe(preview, hide_index=True, width="stretch")

    if amazon.interrupted_amazon_analysis_available():
        summary = amazon.checkpoint_summary()
        if summary is None:
            st.warning(
                _label("runtime.previous_session", "Previous analysis session found. You can continue without reprocessing completed reviews.")
            )
        else:
            st.warning(
                _label(
                    "runtime.previous_session_detail",
                    "Previous analysis session found. {completed} of {total} reviews were already completed; {remaining} remain and {failed} are currently failed.",
                    completed=summary.already_completed,
                    total=summary.total_reviews,
                    remaining=summary.remaining,
                    failed=summary.failed,
                )
            )
        ready = (
            True
            if health_result is None
            else _new_analysis_available(health_result)
        )
        if not ready:
            _render_ai_unavailable_notice(health_result, imported=True)
        resume_options: dict[str, Any] = {
            "type": "primary",
            "key": "resume_amazon_analysis",
        }
        if health_result is not None:
            resume_options["disabled"] = not ready
        if st.button(
            _label("runtime.resume", "Resume Amazon Analysis"),
            **resume_options,
        ):
            if health_result is None:
                _execute_amazon_analysis(resume=True)
            else:
                _execute_amazon_analysis(
                    resume=True,
                    health_result=health_result,
                )

    ready = (
        True
        if health_result is None
        else _new_analysis_available(health_result)
    )
    if not ready and not amazon.interrupted_amazon_analysis_available():
        _render_ai_unavailable_notice(health_result, imported=True)
    analyze_options: dict[str, Any] = {
        "type": "primary",
        "disabled": imported.reviews.empty or not ready,
        "key": "analyze_imported_amazon_reviews",
    }
    if st.button(
        _label("amazon.analyze_available", "Analyze Available Reviews"),
        **analyze_options,
    ):
        if health_result is None:
            _execute_amazon_analysis()
        else:
            _execute_amazon_analysis(health_result=health_result)
    st.caption(_label("amazon.import_notice", "Importing only fetches and previews public review data. The local AI model runs only after you click Analyze Available Reviews."))
    return True


def _friendly_browser_import_error(result: AmazonBrowserImportResult) -> str:
    """Return a concise browser-import message without raw browser output."""

    messages = {
        BrowserImportStatus.BROWSER_UNAVAILABLE: (
            _label(
                "amazon.browser_unavailable",
                "Amazon browser support is unavailable on this computer.",
            )
        ),
        BrowserImportStatus.NETWORK_ERROR: _label(
            "amazon.browser_network_error",
            "The browser could not reach Amazon. Check the network connection and try once more.",
        ),
        BrowserImportStatus.NAVIGATION_TIMEOUT: _label(
            "amazon.browser_timeout",
            "Amazon page navigation timed out. Please try again later.",
        ),
        BrowserImportStatus.REVIEWS_UNAVAILABLE: _label(
            "amazon.reviews_unavailable",
            "Amazon does not currently make reviews available for this product.",
        ),
        BrowserImportStatus.NO_REVIEWS_FOUND: _label(
            "amazon.no_reviews", "No public reviews were found for this product."
        ),
        BrowserImportStatus.PAGE_STRUCTURE_CHANGED: _label(
            "amazon.browser_structure_error",
            "Amazon's page structure was not recognized. The import was stopped without changing saved data.",
        ),
        BrowserImportStatus.PARSE_ERROR: _label(
            "amazon.parse_error",
            "The Amazon review page could not be read safely. The import was stopped.",
        ),
        BrowserImportStatus.CANCELLED: _label(
            "amazon.cancelled", "Amazon browser import was cancelled."
        ),
    }
    return messages.get(
        result.status,
        _label(
            "amazon.generic_error",
            result.error_message or "Amazon reviews could not be imported.",
        ),
    )


def _render_nonfatal_browser_availability(
    result: AmazonBrowserImportResult,
) -> bool:
    """Render a known access limitation as guidance, never a fatal error."""

    if (
        result.status == BrowserImportStatus.REVIEWS_UNAVAILABLE
        or result.availability_status == ReviewAvailabilityStatus.ACCESS_LIMITED
    ):
        st.warning(_label("amazon.access_limited_detail", "Amazon currently exposes a limited set of reviews to this browser session. Quick Scan can analyze any reviews that are currently accessible."))
        return True
    return False


def _render_browser_identity(result: AmazonBrowserImportResult) -> None:
    st.subheader(_label("amazon.browser_import", "Browser Import"))
    identity = st.columns(3)
    identity[0].metric(_label("amazon.marketplace", "Marketplace"), result.marketplace)
    identity[1].metric("ASIN", result.asin)
    status_label = _label(
        f"amazon.status.{result.status}",
        result.status.replace("_", " ").title(),
    )
    identity[2].metric(_label("amazon.browser_status", "Browser Status"), status_label)
    st.caption(_label("amazon.canonical_url", "Canonical product URL: {url}", url=result.product_url))


def _render_browser_verification(
    result: AmazonBrowserImportResult,
    session: AmazonBrowserSession,
) -> AmazonBrowserImportResult:
    """Render bounded manual verification controls for the existing browser."""

    if result.status == BrowserImportStatus.CHALLENGE_REQUIRED:
        st.warning(_label("amazon.manual_verification_detail", "Amazon requires manual verification in the open browser window. Complete the verification there, then continue here. The app will not solve or bypass the challenge."))
    else:
        st.warning(_label("amazon.login_required", "Amazon requires you to sign in in the open browser window. Sign in manually, then continue here. The app does not enter credentials."))

    first, second = st.columns(2)
    continued = first.button(
        _label("amazon.continue_verification", "Continue After Verification"),
        type="primary",
        use_container_width=True,
        key="continue_amazon_browser_verification",
    )
    cancelled = second.button(
        _label("amazon.cancel_import", "Cancel Amazon Import"),
        use_container_width=True,
        key="cancel_amazon_browser_verification",
    )
    if cancelled:
        _close_amazon_browser_session(clear_result=True)
        st.info(_label("amazon.cancelled_safe", "Amazon browser import was cancelled. Saved datasets were not changed."))
        return AmazonBrowserImportResult(
            status=BrowserImportStatus.CANCELLED,
            marketplace=result.marketplace,
            asin=result.asin,
            product_url=result.product_url,
            reviews_url=result.reviews_url,
        )
    if continued:
        with st.spinner(_label("amazon.checking_page", "Checking the existing Amazon browser page…")):
            checked = session.continue_after_verification()
        st.session_state[AMAZON_BROWSER_RESULT_KEY] = checked
        if checked.status in {
            BrowserImportStatus.CHALLENGE_REQUIRED,
            BrowserImportStatus.LOGIN_REQUIRED,
        }:
            st.info(_label("amazon.verification_pending", "Manual verification is still required in the open browser."))
        else:
            st.success(_label("amazon.verification_complete", "Manual verification is complete."))
        return checked
    return result


def _render_staged_browser_reviews(
    result: AmazonBrowserImportResult,
    health_result: Mapping[str, Any] | None = None,
) -> None:
    """Preview browser results; save and analyze only after user confirmation."""

    reviews = pd.DataFrame(result.reviews)
    st.success(_label("amazon.unique_count", "{count} unique reviews retrieved.", count=f"{result.review_count:,}"), icon="✅")
    if int(result.retrieved_review_count or 0) > result.review_count:
        st.caption(
            _label(
                "amazon.valid_containers",
                "{count} valid review containers were found before deduplication.",
                count=f"{int(result.retrieved_review_count or 0):,}",
            )
        )
    if result.availability_status == ReviewAvailabilityStatus.ACCESS_LIMITED:
        st.warning(_label("amazon.access_limited_detail", "Amazon currently exposes a limited set of reviews to this browser session. Quick Scan will analyze the reviews that are currently accessible."))
    elif result.availability_status == ReviewAvailabilityStatus.PARTIAL:
        st.info(
            _label(
                "amazon.partial_notice",
                "Quick Scan will analyze the unique reviews currently accessible to this browser session.",
            )
        )
    st.subheader(_label("amazon.preview", "Review Preview"))
    preferred_columns = [
        "review_id",
        "rating",
        "title",
        "content",
        "date",
        "verified_purchase",
    ]
    columns = [column for column in preferred_columns if column in reviews.columns]
    st.dataframe(
        _localized_review_preview(reviews.loc[:, columns].head(5)),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        _label(
            "amazon.no_ai_yet",
            "No local AI analysis has run yet. Review the imported fields before starting Qwen3 8B.",
        )
    )

    ready = (
        True
        if health_result is None
        else _new_analysis_available(health_result)
    )
    if not ready:
        try:
            amazon.save_browser_amazon_import(reviews, result.to_metadata())
        except (amazon.AmazonAnalysisError, runtime.AnalysisRunBusyError):
            st.error(
                _label(
                    "error.operation_failed",
                    "The imported reviews could not be saved. Please try again.",
                )
            )
            return
        _close_amazon_browser_session(clear_result=True)
        _render_ai_unavailable_notice(health_result, imported=True)
        return

    if st.button(
        _label("amazon.analyze_available", "Analyze Available Reviews"),
        type="primary",
        disabled=result.review_count <= 0,
        key="analyze_staged_amazon_browser_reviews",
    ):
        try:
            amazon.save_browser_amazon_import(reviews, result.to_metadata())
        except (amazon.AmazonAnalysisError, runtime.AnalysisRunBusyError):
            st.error(_label("error.operation_failed", "The imported reviews could not be saved. Please try again."))
            return
        _close_amazon_browser_session(clear_result=True)
        if health_result is None:
            _execute_amazon_analysis()
        else:
            _execute_amazon_analysis(health_result=health_result)


def _render_amazon_url_input(
    health_result: Mapping[str, Any] | None = None,
) -> None:
    st.caption(_label("amazon.url_caption", "Analyze reviews currently accessible from an Amazon product page. {notice}", notice=_label("amazon.live_accessibility", "Live Amazon import depends on current website accessibility.")))
    if "amazon_product_url" not in st.session_state:
        preserved_url = st.session_state.get(PRESERVED_AMAZON_URL_KEY, "")
        st.session_state["amazon_product_url"] = (
            preserved_url if isinstance(preserved_url, str) else ""
        )
    url = st.text_input(
        _label("amazon.product_url", "Amazon Product URL"),
        placeholder="https://www.amazon.com/dp/B0XXXXXXXX",
        key="amazon_product_url",
    )
    st.session_state[PRESERVED_AMAZON_URL_KEY] = url
    open_clicked = st.button(
        _label("amazon.open_product", "Open Amazon Product"),
        type="primary",
        disabled=not url.strip(),
        key="open_amazon_product",
    )
    if open_clicked:
        try:
            product = parse_amazon_url(url)
        except AmazonURLValidationError as exc:
            st.error(
                localized_error_message(exc, "amazon.invalid_url", _language())
            )
        else:
            _close_amazon_browser_session(clear_result=True)
            try:
                with st.spinner(_label("amazon.opening_product", "Opening product page in a visible browser…")):
                    session, result = open_amazon_product(
                        product,
                        navigation_timeout_ms=DEFAULT_NAVIGATION_TIMEOUT_MS,
                    )
            except Exception:
                st.error(
                    _label(
                        "amazon.browser_start_error",
                        "The Amazon browser could not be started. Saved datasets were not changed.",
                    )
                )
            else:
                st.session_state[AMAZON_BROWSER_SESSION_KEY] = session
                st.session_state[AMAZON_BROWSER_RESULT_KEY] = result
                if result.status in {
                    BrowserImportStatus.BROWSER_UNAVAILABLE,
                    BrowserImportStatus.NETWORK_ERROR,
                    BrowserImportStatus.NAVIGATION_TIMEOUT,
                    BrowserImportStatus.PAGE_STRUCTURE_CHANGED,
                }:
                    session.close()
                    st.session_state.pop(AMAZON_BROWSER_SESSION_KEY, None)

    session = st.session_state.get(AMAZON_BROWSER_SESSION_KEY)
    result = st.session_state.get(AMAZON_BROWSER_RESULT_KEY)
    if isinstance(result, AmazonBrowserImportResult):
        _render_browser_identity(result)
        verification_rendered = False
        if result.status in {
            BrowserImportStatus.CHALLENGE_REQUIRED,
            BrowserImportStatus.LOGIN_REQUIRED,
        } and isinstance(session, AmazonBrowserSession):
            result = _render_browser_verification(result, session)
            verification_rendered = True

        if result.status == BrowserImportStatus.PRODUCT_PAGE_LOADED and isinstance(
            session, AmazonBrowserSession
        ):
            st.success(_label("amazon.product_loaded", "Amazon product page loaded in the visible browser."))
            cancel_column, load_column = st.columns(2)
            if cancel_column.button(
                _label("amazon.cancel_import", "Cancel Amazon Import"),
                use_container_width=True,
                key="cancel_loaded_amazon_browser",
            ):
                _close_amazon_browser_session(clear_result=True)
                st.info(_label("amazon.cancelled", "Amazon browser import was cancelled."))
                return
            if load_column.button(
                _label("amazon.scan_reviews", "Scan Available Reviews"),
                type="primary",
                use_container_width=True,
                key="load_amazon_browser_reviews",
            ):
                progress_bar = st.progress(0.0)
                progress_text = st.empty()

                def update_progress(
                    page_number: int,
                    reviews_collected: int,
                    status: str,
                ) -> None:
                    progress_bar.progress(
                        min(0.95, page_number / MAX_BROWSER_REVIEW_PAGES)
                    )
                    if status == "opening_review_page":
                        progress_text.write(_label("amazon.reading_page", "Reading review page {page}…", page=page_number))
                    else:
                        progress_text.write(
                            _label("amazon.found_so_far", "Found {count} unique reviews so far.", count=reviews_collected)
                        )

                with st.spinner(_label("amazon.reading_reviews", "Reading public reviews in the visible browser…")):
                    result = session.load_reviews(
                        limit=MAX_AMAZON_QUICK_SCAN_REVIEWS,
                        max_pages=MAX_BROWSER_REVIEW_PAGES,
                        progress_callback=update_progress,
                    )
                st.session_state[AMAZON_BROWSER_RESULT_KEY] = result
                if result.status == BrowserImportStatus.SUCCESS:
                    progress_bar.progress(1.0)
                    progress_text.success(
                        _label("amazon.unique_count", "{count} unique reviews retrieved.", count=result.review_count)
                    )

        if result.status in {
            BrowserImportStatus.CHALLENGE_REQUIRED,
            BrowserImportStatus.LOGIN_REQUIRED,
        } and isinstance(session, AmazonBrowserSession) and not verification_rendered:
            # This branch also covers a challenge reached while opening the
            # review page after the normal product page had already loaded.
            result = _render_browser_verification(result, session)
            st.session_state[AMAZON_BROWSER_RESULT_KEY] = result

        if result.status == BrowserImportStatus.SUCCESS and result.reviews:
            _render_staged_browser_reviews(result, health_result)
            return
        if result.status in {
            BrowserImportStatus.CHALLENGE_REQUIRED,
            BrowserImportStatus.LOGIN_REQUIRED,
            BrowserImportStatus.PRODUCT_PAGE_LOADED,
        }:
            return
        if result.status != BrowserImportStatus.CANCELLED:
            if not _render_nonfatal_browser_availability(result):
                st.error(_friendly_browser_import_error(result))
            if isinstance(session, AmazonBrowserSession):
                session.close()
                st.session_state.pop(AMAZON_BROWSER_SESSION_KEY, None)

    # A prior, user-confirmed import remains available independently of any
    # current browser attempt.
    _render_amazon_import_preview(health_result)


def _render_new_analysis(
    data_source: str | None,
    health_result: Mapping[str, Any] | None = None,
) -> None:
    title = html.escape(_label("dataset.analyze_new", "Analyze New Dataset"))
    subtitle = html.escape(_label("input.new_analysis_subtitle", "Preview, validate and analyze reviews with local Qwen3 8B."))
    isolation = html.escape(_label("input.isolation_notice", "New results are isolated from the Demo Dataset."))
    st.markdown(
        f"""
        <div class="hero">
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <div class="source">{isolation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(_label("input.model_notice", "AI Model: Qwen3 8B · Provider: Local Ollama · Analysis starts only after you click Analyze Reviews."))
    source_descriptions = {
        AMAZON_QUICK_SCAN_SOURCE: (
            _label("input.amazon_description", "Analyze reviews currently accessible from an Amazon product page.")
        ),
        UPLOAD_DATASET_SOURCE: (
            _label("input.upload_description", "Upload a larger review dataset for full analysis.")
        ),
        PASTE_REVIEWS_SOURCE: _label("input.paste_description", "Paste reviews manually for custom analysis."),
    }
    if data_source in source_descriptions:
        st.caption(source_descriptions[data_source])

    ready = (
        True
        if health_result is None
        else _new_analysis_available(health_result)
    )

    if (
        data_source != AMAZON_QUICK_SCAN_SOURCE
        and interactive.interrupted_analysis_available()
    ):
        summary = interactive.checkpoint_summary()
        if summary is None:
            st.warning(
                _label("runtime.previous_session", "Previous analysis session found. You can continue without reprocessing completed reviews.")
            )
        else:
            st.warning(
                _label(
                    "runtime.previous_session_detail",
                    "Previous analysis session found. {completed} of {total} reviews were already completed; {remaining} remain and {failed} are currently failed.",
                    completed=summary.already_completed,
                    total=summary.total_reviews,
                    remaining=summary.remaining,
                    failed=summary.failed,
                )
            )
        if not ready:
            _render_ai_unavailable_notice(health_result)
        resume_options: dict[str, Any] = {
            "type": "primary",
            "key": "resume_interrupted_analysis",
        }
        if health_result is not None:
            resume_options["disabled"] = not ready
        if st.button(
            _label("runtime.resume", "Resume Interrupted Analysis"),
            **resume_options,
        ):
            if health_result is None:
                _execute_interactive_analysis(
                    None,
                    "resume_interrupted",
                    resume=True,
                )
            else:
                _execute_interactive_analysis(
                    None,
                    "resume_interrupted",
                    resume=True,
                    health_result=health_result,
                )
        st.divider()
    if data_source == UPLOAD_DATASET_SOURCE:
        prepared = _render_upload_input()
    elif data_source == PASTE_REVIEWS_SOURCE:
        prepared = _render_paste_input()
    elif data_source == AMAZON_QUICK_SCAN_SOURCE:
        _render_amazon_url_input(health_result)
        return
    else:
        st.info(_label("dataset.choose_source", "Choose a data source to continue."))
        return
    if prepared is not None:
        reviews, key_token = prepared
        _render_analysis_action(reviews, key_token, health_result)


def main() -> None:
    _inject_styles()
    health_result = _health_report()
    if not _first_run_is_complete():
        _render_first_run_sidebar(health_result)
        _render_first_run(health_result)
        return

    current_available = interactive.current_analysis_available()
    amazon_available = amazon.amazon_analysis_available()
    amazon_present = amazon.amazon_import_available()
    dataset_mode, data_source, workspace, compact_view = _render_sidebar_controls(
        current_available,
        amazon_available,
        amazon_present,
        health_result,
    )

    if dataset_mode == "Demo Dataset":
        _close_amazon_browser_session(clear_result=True)
        _render_saved_dashboard(
            Path(DEFAULT_OUTPUTS_DIR),
            dataset_label="Demo Dataset",
            key_prefix="demo",
            compact_view=compact_view,
        )
        return

    if dataset_mode == COMPARE_PRODUCTS_MODE:
        _close_amazon_browser_session(clear_result=True)
        comparison_dashboard.render_competitor_comparison(language=_language())
        return

    if dataset_mode == SAVED_ANALYSES_MODE:
        _close_amazon_browser_session(clear_result=True)
        _render_saved_analyses_page()
        return

    if workspace != "New Input" or data_source != AMAZON_QUICK_SCAN_SOURCE:
        _close_amazon_browser_session(clear_result=True)

    if workspace == UPLOAD_WORKSPACE:
        if not current_available:
            st.info(_label("dataset.none_yet", "No custom analysis has been run yet."))
            return
        current_directory = interactive.resolve_dashboard_directory(
            "Current Analysis"
        )
        _render_failed_review_retry_panel(
            current_directory,
            health_result=health_result,
        )
        _render_saved_dashboard(
            current_directory,
            dataset_label="Current Analysis",
            key_prefix="current",
            compact_view=compact_view,
        )
        return

    if workspace == AMAZON_WORKSPACE:
        if not amazon_available:
            st.info(_label("dataset.no_amazon_yet", "No Amazon analysis has been run yet."))
            return
        amazon_directory = amazon.AmazonSessionPaths.current().root
        _render_failed_review_retry_panel(
            amazon_directory,
            source="amazon",
            health_result=health_result,
        )
        _render_saved_dashboard(
            amazon_directory,
            dataset_label=AMAZON_QUICK_SCAN_SOURCE,
            key_prefix="amazon",
            compact_view=compact_view,
            quick_scan=True,
        )
        return

    _render_new_analysis(data_source, health_result)


if __name__ == "__main__":
    main()
