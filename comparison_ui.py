"""Read-only Streamlit UI for deterministic two-product comparison.

The module reads completed analysis artifacts (or the bundled offline fixture)
and delegates all calculations to :mod:`competitor_comparison`.  It never
imports or invokes model, review-analysis, aggregation, keyword, or scraping
pipelines.  Canonical values remain unchanged; localization is display-only,
and customer phrases/Evidence are always rendered verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis_library as library
import competitor_comparison as comparison
from config import (
    AMAZON_CURRENT_DIR,
    COMPARISON_CURRENT_DIR,
    OUTPUTS_DIR,
    SESSION_CURRENT_DIR,
)
from i18n import (
    display_analysis_source_type,
    display_aspect,
    display_business_category,
    display_comparison_status,
    display_evidence_level,
    display_sample_level,
    localized_comparison_recommendation,
    localized_comparison_summary,
    normalize_language,
    t,
)


COMPARISON_STATE_KEY = "competitor_comparison_payload"
PRODUCT_A_STATE_KEY = "competitor_comparison_product_a"
PRODUCT_B_STATE_KEY = "competitor_comparison_product_b"
SOURCE_A_STATE_KEY = "competitor_comparison_source_a"
SOURCE_B_STATE_KEY = "competitor_comparison_source_b"
PRODUCT_NAME_A_STATE_KEY = "competitor_comparison_name_a"
PRODUCT_NAME_B_STATE_KEY = "competitor_comparison_name_b"
SELECTION_STATE_KEY = "competitor_comparison_selection"
COMPARISON_REPORT_STATE_KEY = "competitor_comparison_report_document"
REPORT_LANGUAGES = ("en", "zh")

_REQUIRED_DASHBOARD_OUTPUTS = (
    "analyzed_reviews.csv",
    "aspect_summary.csv",
    "business_summary.json",
    "aspect_evidence.json",
    "keyword_summary.csv",
    "sentiment_keywords.json",
    "aspect_keywords.json",
)


@dataclass(frozen=True)
class ComparisonSource:
    """One trusted, completed analysis selectable by the comparison UI."""

    key: str
    label_key: str
    default_product_name: str
    source_type: str
    outputs_dir: Path | None = None
    fixture_key: str | None = None
    saved_analysis_id: str | None = None
    review_count: int = 0
    library_root: Path | None = None


def _is_complete_dashboard_source(root: Path) -> bool:
    return root.is_dir() and all(
        (root / filename).is_file() for filename in _REQUIRED_DASHBOARD_OUTPUTS
    )


def available_comparison_sources() -> tuple[ComparisonSource, ...]:
    """Return Saved Analyses first, then current workspaces and fixtures.

    The registry is read-only.  Loading this list never runs a pipeline or
    modifies a source snapshot.  Stable ``analysis_id`` values are used as
    widget keys so language and Dataset Mode changes preserve A/B selection.
    """

    sources: list[ComparisonSource] = []
    for saved in library.list_saved_analyses():
        manifest = saved.manifest
        sources.append(
            ComparisonSource(
                key=f"saved:{saved.analysis_id}",
                label_key="comparison.saved_source",
                default_product_name=saved.product_name,
                source_type=saved.source_type,
                outputs_dir=saved.root,
                saved_analysis_id=saved.analysis_id,
                review_count=int(manifest.get("imported_reviews", 0) or 0),
                library_root=saved.root.parent,
            )
        )

    fixture_sources = [
        ComparisonSource(
            "fixture_a",
            "comparison.source_fixture_a",
            "Aurora View TV",
            "synthetic_fixture",
            fixture_key="fixture_a",
        ),
        ComparisonSource(
            "fixture_b",
            "comparison.source_fixture_b",
            "Nimbus Smart TV",
            "synthetic_fixture",
            fixture_key="fixture_b",
        ),
    ]
    candidates = (
        ("demo", "comparison.source_demo", "Demo Product", "demo", OUTPUTS_DIR),
        (
            "current_upload",
            "comparison.source_session",
            "Current Upload/Paste Product",
            "upload_or_paste",
            SESSION_CURRENT_DIR,
        ),
        (
            "current_amazon",
            "comparison.source_amazon",
            "Current Amazon Product",
            "amazon",
            AMAZON_CURRENT_DIR,
        ),
    )
    for key, label_key, name, source_type, root in candidates:
        if _is_complete_dashboard_source(Path(root)):
            sources.append(
                ComparisonSource(
                    key,
                    label_key,
                    name,
                    source_type,
                    outputs_dir=Path(root),
                )
            )
    # Fixtures remain selectable for deterministic tests/development, but
    # never displace two available real Saved Analyses as production defaults.
    sources.extend(fixture_sources)
    return tuple(sources)


def load_comparison_source(
    source: ComparisonSource,
    *,
    product_name: str | None = None,
) -> comparison.ProductAnalysis:
    """Load one source without running inference or blessing unknown versions."""

    name = (product_name or source.default_product_name).strip()
    name = name or source.default_product_name
    if source.saved_analysis_id:
        saved = library.load_analysis(
            source.saved_analysis_id,
            library_root=source.library_root or library.DEFAULT_ANALYSIS_LIBRARY_DIR,
        )
        manifest = saved.manifest
        return comparison.load_product_analysis(
            saved.root,
            product_name=name,
            product_id=str(
                manifest.get("product_id")
                or manifest.get("asin")
                or saved.analysis_id
            ),
            source_type=saved.source_type,
            analysis_run_id=saved.analysis_run_id,
            analysis_candidate=str(manifest.get("analysis_candidate") or ""),
            taxonomy_version=str(manifest.get("taxonomy_version") or ""),
            analysis_id=saved.analysis_id,
            assume_current_version=False,
        )
    if source.fixture_key:
        return comparison.load_fixture_product(source.fixture_key, product_name=name)
    if source.outputs_dir is None:
        raise comparison.ComparisonError("Comparison source has no output directory.")
    return comparison.load_product_analysis(
        source.outputs_dir,
        product_name=name,
        source_type=source.source_type,
        assume_current_version=False,
    )


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _coverage(value: object) -> float:
    coverage = max(0.0, _as_float(value))
    return coverage * 100.0 if coverage <= 1.0 else coverage


def _percent(value: object) -> str:
    return "—" if value is None else f"{_as_float(value):.1f}%"


def _rate_or_no_evidence(value: object, language: object) -> str:
    return (
        t("comparison.no_evidence_value", language)
        if value is None
        else f"{_as_float(value):.1f}%"
    )


def _number_or_no_evidence(value: object, language: object, digits: int = 1) -> str:
    return (
        t("comparison.no_evidence_value", language)
        if value is None
        else f"{_as_float(value):.{digits}f}"
    )


def _number(value: object, digits: int = 1) -> str:
    return "—" if value is None else f"{_as_float(value):.{digits}f}"


def _generate_comparison_report(
    comparison_root: Path,
    *,
    product_a_root: Path,
    product_b_root: Path,
    language: str,
):
    """Build and persist a report from verified, completed snapshots only."""

    import report_generator

    document = report_generator.build_comparison_report(
        comparison_root,
        product_a_root=product_a_root,
        product_b_root=product_b_root,
        language=language,
        validate_integrity=True,
    )
    exported = report_generator.export_report(document)
    return document, exported


def _render_comparison_report_controls(
    *,
    output_root: Path,
    source_a: ComparisonSource,
    source_b: ComparisonSource,
    language: object,
) -> None:
    """Offer an HTML report for the active saved-snapshot comparison.

    V1 intentionally requires two Saved Analyses so both source manifests can
    be integrity-checked immediately before report generation.  The existing
    fixture/current-output comparison remains available for on-screen use.
    """

    st.subheader(t("report.comparison.export", language))
    st.caption(t("report.read_only_notice", language))
    ui_language = normalize_language(language)
    report_language = st.selectbox(
        t("report.language", language),
        REPORT_LANGUAGES,
        index=REPORT_LANGUAGES.index(ui_language),
        format_func=lambda value: t(f"report.language.{value}", language),
        key=f"competitor_comparison_report_language_{ui_language}",
    )
    saved_roots_available = bool(
        source_a.saved_analysis_id
        and source_b.saved_analysis_id
        and source_a.outputs_dir
        and source_b.outputs_dir
    )
    if not saved_roots_available:
        st.info(t("report.comparison.saved_required", language))
    if st.button(
        t("report.comparison.generate", language),
        key="competitor_comparison_generate_report",
        type="primary",
        disabled=not saved_roots_available,
    ):
        try:
            document, exported = _generate_comparison_report(
                Path(output_root),
                product_a_root=Path(source_a.outputs_dir),  # type: ignore[arg-type]
                product_b_root=Path(source_b.outputs_dir),  # type: ignore[arg-type]
                language=report_language,
            )
        except Exception as exc:  # Keep report/integrity failures inside the UI.
            import report_generator

            if isinstance(exc, report_generator.ReportIntegrityError):
                st.error(t("report.integrity_error", language))
            else:
                st.error(t("report.generate_error", language, error=str(exc)))
        else:
            st.session_state[COMPARISON_REPORT_STATE_KEY] = document
            st.session_state[f"{COMPARISON_REPORT_STATE_KEY}_export"] = exported
            st.success(t("report.comparison.generated", language))

    document = st.session_state.get(COMPARISON_REPORT_STATE_KEY)
    html_document = getattr(document, "html", None)
    report_id = str(getattr(document, "report_id", "comparison-report"))
    if isinstance(html_document, str) and html_document:
        st.download_button(
            t("report.comparison.download_html", language),
            data=html_document,
            file_name=f"{report_id}.html",
            mime="text/html",
            key="competitor_comparison_download_report",
        )
        st.caption(t("report.pdf_instruction", language))


def _source_label(source: ComparisonSource, language: object) -> str:
    if source.saved_analysis_id:
        return t(
            "comparison.saved_source",
            language,
            name=source.default_product_name,
            reviews=source.review_count,
            source=display_analysis_source_type(source.source_type, language),
        )
    return t(source.label_key, language)


def _render_product_summary(
    product: Mapping[str, Any],
    side_key: str,
    language: object,
) -> None:
    st.subheader(
        f"{t(side_key, language)} · {product.get('product_name', '')}"
    )
    first, second, third = st.columns(3)
    first.metric(t("comparison.reviews", language), int(product.get("total_reviews", 0)))
    second.metric(
        t("comparison.successfully_analyzed", language),
        int(product.get("successful_reviews", 0)),
    )
    third.metric(t("comparison.coverage", language), _percent(_coverage(product.get("coverage"))))
    st.caption(
        f"{t('comparison.sample_level', language)}: "
        f"{display_sample_level(product.get('sample_level'), language)} · "
        f"{t('comparison.analysis_version', language)}: "
        f"{product.get('analysis_candidate', '—')}"
    )


def _render_sample_warnings(payload: Mapping[str, Any], language: object) -> None:
    for warning in payload.get("sample_warnings", []):
        if not isinstance(warning, Mapping):
            continue
        if warning.get("code") == "early_comparison":
            st.warning(t("comparison.warning.early", language), icon="⚠️")
        elif warning.get("code") == "uneven_sample_sizes":
            st.warning(t("comparison.warning.uneven", language), icon="⚠️")
    st.caption(t("comparison.warning.directional", language))


def _support_caption(record: Mapping[str, Any], language: object) -> str:
    return t(
        "comparison.card.support",
        language,
        a_share=_as_float(record.get("a_review_share")),
        b_share=_as_float(record.get("b_review_share")),
        level=display_evidence_level(record.get("comparison_evidence"), language),
    )


def _render_advantage_cards(
    title: str,
    records: object,
    payload: Mapping[str, Any],
    language: object,
    *,
    weakness: bool,
) -> None:
    st.subheader(title)
    if not isinstance(records, list) or not records:
        key = "comparison.no_weaknesses" if weakness else "comparison.no_advantages"
        st.caption(t(key, language))
        return
    a_name = str(payload.get("product_a", {}).get("product_name", t("comparison.product_a", language)))
    b_name = str(payload.get("product_b", {}).get("product_name", t("comparison.product_b", language)))
    for record in records:
        if not isinstance(record, Mapping):
            continue
        st.markdown(f"**{display_aspect(record.get('aspect'), language)}**")
        if weakness:
            message = t(
                "comparison.card.weakness",
                language,
                a_name=a_name,
                b_name=b_name,
                a_rate=_as_float(record.get("a_negative_rate")),
                b_rate=_as_float(record.get("b_negative_rate")),
                gap=_as_float(record.get("negative_rate_gap")),
            )
        else:
            message = t(
                "comparison.card.advantage",
                language,
                a_name=a_name,
                b_name=b_name,
                a_rate=_as_float(record.get("a_positive_rate")),
                b_rate=_as_float(record.get("b_positive_rate")),
                gap=_as_float(record.get("positive_rate_gap")),
            )
        st.caption(message)
        st.caption(_support_caption(record, language))


def _render_white_space(payload: Mapping[str, Any], language: object) -> None:
    st.subheader(t("comparison.white_space", language))
    records = payload.get("white_space_opportunities", [])
    if not isinstance(records, list) or not records:
        st.caption(t("comparison.no_white_space", language))
        return
    for record in records:
        if isinstance(record, Mapping):
            aspect = display_aspect(record.get("aspect"), language)
            st.markdown(f"**{aspect}**")
            st.caption(t("comparison.white_space_message", language, aspect=aspect))
            st.caption(_support_caption(record, language))


def _render_shopping(payload: Mapping[str, Any], language: object) -> None:
    st.subheader(t("comparison.shopping_experience", language))
    records = payload.get("shopping_experience_comparison", [])
    if not isinstance(records, list) or not records:
        st.caption(t("comparison.no_shopping", language))
        return
    for record in records:
        if isinstance(record, Mapping):
            st.markdown(
                f"- **{display_aspect(record.get('aspect'), language)}** — "
                f"{display_comparison_status(record.get('comparison_status'), language)} · "
                f"{_support_caption(record, language)}"
            )


def _render_recommendations(payload: Mapping[str, Any], language: object) -> None:
    st.subheader(t("comparison.business_recommendations", language))
    records = payload.get("recommendations", [])
    if not isinstance(records, list) or not records:
        st.caption(t("comparison.no_recommendations", language))
        return
    a_name = payload.get("product_a", {}).get("product_name")
    b_name = payload.get("product_b", {}).get("product_name")
    for record in records:
        if not isinstance(record, Mapping):
            continue
        localized = localized_comparison_recommendation(
            record,
            language,
            product_a_name=a_name,
            product_b_name=b_name,
        )
        st.markdown(f"- **{localized['type']}** — {localized['message']}")


def _render_overview(payload: Mapping[str, Any], language: object) -> None:
    left, right = st.columns(2)
    with left:
        _render_product_summary(payload.get("product_a", {}), "comparison.product_a", language)
    with right:
        _render_product_summary(payload.get("product_b", {}), "comparison.product_b", language)
    _render_sample_warnings(payload, language)

    summary = payload.get("head_to_head_summary", {})
    st.subheader(t("comparison.head_to_head_summary", language))
    metrics = st.columns(4)
    metrics[0].metric(t("comparison.a_leads", language), int(summary.get("product_a_leads", 0)))
    metrics[1].metric(t("comparison.b_leads", language), int(summary.get("product_b_leads", 0)))
    metrics[2].metric(t("comparison.similar_aspects", language), int(summary.get("similar_aspects", 0)))
    metrics[3].metric(
        t("comparison.insufficient_aspects", language),
        int(summary.get("insufficient_evidence_aspects", 0)),
    )
    st.header(t("comparison.executive_summary", language))
    st.write(localized_comparison_summary(payload, language))

    a_name = str(payload.get("product_a", {}).get("product_name", t("comparison.product_a", language)))
    b_name = str(payload.get("product_b", {}).get("product_name", t("comparison.product_b", language)))
    a_column, b_column = st.columns(2)
    with a_column:
        _render_advantage_cards(
            f"{a_name} · {t('comparison.competitive_advantages', language)}",
            payload.get("competitive_advantages"),
            payload,
            language,
            weakness=False,
        )
    with b_column:
        _render_advantage_cards(
            f"{a_name} · {t('comparison.competitive_weaknesses', language)} ({b_name})",
            payload.get("competitive_weaknesses"),
            payload,
            language,
            weakness=True,
        )
    _render_white_space(payload, language)
    _render_shopping(payload, language)
    _render_recommendations(payload, language)
    st.info(t("comparison.no_overall_winner", language), icon="ℹ️")


def _comparison_frame(payload: Mapping[str, Any], language: object) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in payload.get("aspect_comparisons", []):
        if not isinstance(raw, Mapping):
            continue
        rows.append(
            {
                t("comparison.aspect", language): display_aspect(raw.get("aspect"), language),
                t("comparison.business_category", language): display_business_category(
                    raw.get("business_category"), language
                ),
                t("comparison.a_review_count", language): int(raw.get("a_review_count", 0)),
                t("comparison.b_review_count", language): int(raw.get("b_review_count", 0)),
                t("comparison.a_review_share", language): _percent(raw.get("a_review_share")),
                t("comparison.b_review_share", language): _percent(raw.get("b_review_share")),
                t("comparison.a_positive_rate", language): _rate_or_no_evidence(
                    raw.get("a_positive_rate"), language
                ),
                t("comparison.b_positive_rate", language): _rate_or_no_evidence(
                    raw.get("b_positive_rate"), language
                ),
                t("comparison.a_negative_rate", language): _rate_or_no_evidence(
                    raw.get("a_negative_rate"), language
                ),
                t("comparison.b_negative_rate", language): _rate_or_no_evidence(
                    raw.get("b_negative_rate"), language
                ),
                t("comparison.net_sentiment_gap", language): _number_or_no_evidence(
                    raw.get("net_sentiment_gap"), language
                ),
                t("comparison.status", language): display_comparison_status(
                    raw.get("comparison_status"), language
                ),
                t("comparison.evidence", language): display_evidence_level(
                    raw.get("comparison_evidence"), language
                ),
            }
        )
    return pd.DataFrame(rows)


def _chart_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        record
        for record in payload.get("aspect_comparisons", [])
        if isinstance(record, Mapping)
        and record.get("comparison_status") != "insufficient_evidence"
        and record.get("net_sentiment_gap") is not None
        and record.get("aspect") != "Other"
    ]


def _render_gap_chart(payload: Mapping[str, Any], language: object) -> None:
    records = _chart_records(payload)
    if not records:
        return
    gaps = [_as_float(record.get("net_sentiment_gap")) for record in records]
    figure = go.Figure(
        go.Bar(
            x=gaps,
            y=[display_aspect(record.get("aspect"), language) for record in records],
            orientation="h",
            marker_color=["#2E8B74" if gap >= 0 else "#D45D5D" for gap in gaps],
            text=[f"{gap:+.1f}" for gap in gaps],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.1f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color="#8A94A6", line_width=1)
    figure.update_layout(
        title=t("comparison.gap_chart", language),
        xaxis_title=t("comparison.gap_axis", language),
        yaxis_title="",
        height=max(360, 36 * len(records) + 110),
        margin=dict(l=15, r=45, t=55, b=35),
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(t("comparison.gap_positive", language))


def _render_sentiment_matrices(payload: Mapping[str, Any], language: object) -> None:
    """Show positive and negative rates side by side without filling missing values."""

    records = [
        record
        for record in payload.get("aspect_comparisons", [])
        if isinstance(record, Mapping) and record.get("aspect") != "Other"
    ]
    if not records:
        return
    a_name = str(
        payload.get("product_a", {}).get(
            "product_name", t("comparison.product_a", language)
        )
    )
    b_name = str(
        payload.get("product_b", {}).get(
            "product_name", t("comparison.product_b", language)
        )
    )
    aspects = [display_aspect(record.get("aspect"), language) for record in records]
    st.subheader(t("comparison.sentiment_matrices", language))
    columns = st.columns(2)
    for column, metric, title_key, colorscale in (
        (columns[0], "positive_rate", "comparison.positive_matrix", "Greens"),
        (columns[1], "negative_rate", "comparison.negative_matrix", "Reds"),
    ):
        values = [
            [record.get(f"a_{metric}"), record.get(f"b_{metric}")]
            for record in records
        ]
        labels = [
            [
                t("comparison.no_evidence_value", language)
                if value is None
                else f"{_as_float(value):.1f}%"
                for value in row
            ]
            for row in values
        ]
        figure = go.Figure(
            go.Heatmap(
                z=values,
                x=[a_name, b_name],
                y=aspects,
                text=labels,
                texttemplate="%{text}",
                zmin=0,
                zmax=100,
                colorscale=colorscale,
                colorbar=dict(title="%"),
                hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
            )
        )
        figure.update_layout(
            title=t(title_key, language),
            height=max(390, 32 * len(records) + 110),
            margin=dict(l=15, r=15, t=55, b=35),
        )
        with column:
            st.plotly_chart(figure, width="stretch")


def _render_opportunity_matrix(payload: Mapping[str, Any], language: object) -> None:
    records = _chart_records(payload)
    if not records:
        return
    figure = go.Figure()
    for record in records:
        gap = _as_float(record.get("net_sentiment_gap"))
        figure.add_trace(
            go.Scatter(
                x=[gap],
                y=[_as_float(record.get("average_review_share"))],
                mode="markers+text",
                text=[display_aspect(record.get("aspect"), language)],
                textposition="top center",
                marker=dict(size=13, color="#2E8B74" if gap >= 0 else "#D45D5D"),
                hovertemplate="%{text}<br>%{x:.1f}<br>%{y:.1f}%<extra></extra>",
                showlegend=False,
            )
        )
    figure.add_vline(x=0, line_color="#8A94A6", line_width=1)
    figure.update_layout(
        title=t("comparison.opportunity_matrix", language),
        xaxis_title=t("comparison.opportunity_x", language),
        yaxis_title=t("comparison.opportunity_y", language),
        height=470,
        margin=dict(l=15, r=15, t=55, b=35),
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(t("comparison.opportunity_caption", language))


def _render_head_to_head(payload: Mapping[str, Any], language: object) -> None:
    st.caption(t("comparison.raw_counts_context", language))
    st.caption(t("comparison.net_sentiment_notice", language))
    frame = _comparison_frame(payload, language)
    if frame.empty:
        st.info(t("comparison.summary.no_supported_signals", language))
        return
    st.dataframe(frame, width="stretch", hide_index=True)
    if any(
        isinstance(record, Mapping)
        and ((int(record.get("a_review_count", 0) or 0) == 0) !=
             (int(record.get("b_review_count", 0) or 0) == 0))
        for record in payload.get("aspect_comparisons", [])
    ):
        st.caption(t("comparison.one_sided_evidence", language))
    _render_sentiment_matrices(payload, language)
    _render_gap_chart(payload, language)
    _render_opportunity_matrix(payload, language)


def _render_rating_comparison(payload: Mapping[str, Any], language: object) -> None:
    rating = payload.get("rating_comparison", {})
    if not isinstance(rating, Mapping) or not rating.get("available"):
        st.info(t("comparison.rating_unavailable", language))
        return
    a_name = str(payload.get("product_a", {}).get("product_name", t("comparison.product_a", language)))
    b_name = str(payload.get("product_b", {}).get("product_name", t("comparison.product_b", language)))
    left, right = st.columns(2)
    left.metric(
        f"{a_name} · {t('rating.average', language)}",
        _number(rating.get("product_a", {}).get("average_rating"), 2),
    )
    right.metric(
        f"{b_name} · {t('rating.average', language)}",
        _number(rating.get("product_b", {}).get("average_rating"), 2),
    )
    a_distribution = {
        int(item.get("rating", 0)): _as_float(item.get("share"))
        for item in rating.get("product_a", {}).get("distribution", [])
        if isinstance(item, Mapping)
    }
    b_distribution = {
        int(item.get("rating", 0)): _as_float(item.get("share"))
        for item in rating.get("product_b", {}).get("distribution", [])
        if isinstance(item, Mapping)
    }
    stars = [1, 2, 3, 4, 5]
    figure = go.Figure(
        [
            go.Bar(name=a_name, x=stars, y=[a_distribution.get(star, 0.0) for star in stars]),
            go.Bar(name=b_name, x=stars, y=[b_distribution.get(star, 0.0) for star in stars]),
        ]
    )
    figure.update_layout(
        barmode="group",
        title=t("comparison.rating_distribution", language),
        xaxis_title=t("rating.rating", language),
        yaxis_title=t("comparison.review_share", language) + " (%)",
        height=420,
    )
    st.plotly_chart(figure, width="stretch")
    aspect_ratings = rating.get("aspect_rating_comparison", [])
    if isinstance(aspect_ratings, list) and aspect_ratings:
        rows = [
            {
                t("comparison.aspect", language): display_aspect(item.get("aspect"), language),
                f"A · {t('rating.average', language)}": _number(item.get("a_avg_rating"), 2),
                f"B · {t('rating.average', language)}": _number(item.get("b_avg_rating"), 2),
            }
            for item in aspect_ratings
            if isinstance(item, Mapping)
        ]
        st.subheader(t("comparison.aspect_rating", language))
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(t("comparison.rating_notice", language))


def _terms(records: object) -> str:
    if not isinstance(records, list):
        return ""
    return " · ".join(
        str(record.get("term", "")).strip()
        for record in records
        if isinstance(record, Mapping) and str(record.get("term", "")).strip()
    )


def _phrase_group(title: str, records: object, language: object, empty_key: str) -> None:
    st.markdown(f"**{title}**")
    st.caption(_terms(records) or t(empty_key, language))


def _render_customer_language(payload: Mapping[str, Any], language: object) -> None:
    language_payload = payload.get("language_comparison", {})
    by_aspect = language_payload.get("by_aspect", {}) if isinstance(language_payload, Mapping) else {}
    if not isinstance(by_aspect, Mapping) or not by_aspect:
        st.info(t("comparison.no_language", language))
        return
    st.caption(t("comparison.language_notice", language))
    aspect = st.selectbox(
        t("comparison.select_aspect", language),
        list(by_aspect),
        format_func=lambda value: display_aspect(value, language),
        key="competitor_comparison_language_aspect",
    )
    selected = by_aspect.get(aspect, {})
    a_name = str(payload.get("product_a", {}).get("product_name", t("comparison.product_a", language)))
    b_name = str(payload.get("product_b", {}).get("product_name", t("comparison.product_b", language)))
    for sentiment, title_key in (
        ("positive", "comparison.positive_phrases"),
        ("negative", "comparison.negative_phrases"),
    ):
        st.subheader(t(title_key, language))
        group = selected.get(sentiment, {}) if isinstance(selected, Mapping) else {}
        left, right = st.columns(2)
        with left:
            _phrase_group(a_name, group.get("product_a", []), language, "comparison.no_language")
            _phrase_group(
                t("comparison.a_distinctive", language),
                group.get("product_a_distinctive", []),
                language,
                "comparison.no_distinctive_language",
            )
        with right:
            _phrase_group(b_name, group.get("product_b", []), language, "comparison.no_language")
            _phrase_group(
                t("comparison.b_distinctive", language),
                group.get("product_b_distinctive", []),
                language,
                "comparison.no_distinctive_language",
            )
        _phrase_group(
            t("comparison.shared_language", language),
            group.get("shared", []),
            language,
            "comparison.no_shared_language",
        )


def _render_evidence_records(
    records: Sequence[Mapping[str, Any]],
    language: object,
) -> None:
    if not records:
        st.caption(t("comparison.no_evidence", language))
        return
    for record in records:
        evidence = str(record.get("evidence", "")).strip()
        if not evidence:
            continue
        st.markdown(f"> “{evidence}”")
        review_id = str(record.get("review_id", "")).strip()
        if review_id:
            st.caption(f"Review ID: {review_id}")


def _render_evidence_comparison(
    payload: Mapping[str, Any],
    product_a: comparison.ProductAnalysis | None,
    product_b: comparison.ProductAnalysis | None,
    language: object,
) -> None:
    if product_a is None or product_b is None:
        st.info(t("comparison.no_evidence", language))
        return
    aspects = sorted(
        set(product_a.aspect_evidence).union(product_b.aspect_evidence),
        key=str.casefold,
    )
    if not aspects:
        st.info(t("comparison.no_evidence", language))
        return
    st.caption(t("comparison.evidence_notice", language))
    aspect = st.selectbox(
        t("comparison.select_aspect", language),
        aspects,
        format_func=lambda value: display_aspect(value, language),
        key="competitor_comparison_evidence_aspect",
    )
    limit = 25 if st.toggle(
        t("comparison.show_more", language),
        key="competitor_comparison_show_more_evidence",
    ) else 5
    a_name = str(payload.get("product_a", {}).get("product_name", t("comparison.product_a", language)))
    b_name = str(payload.get("product_b", {}).get("product_name", t("comparison.product_b", language)))
    for sentiment, heading_key in (
        ("positive", "comparison.positive_evidence"),
        ("negative", "comparison.negative_evidence"),
    ):
        st.subheader(t(heading_key, language))
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{a_name}**")
            _render_evidence_records(
                comparison.evidence_for_product(product_a, aspect, sentiment, limit=limit),
                language,
            )
        with right:
            st.markdown(f"**{b_name}**")
            _render_evidence_records(
                comparison.evidence_for_product(product_b, aspect, sentiment, limit=limit),
                language,
            )


def _render_results(
    payload: Mapping[str, Any],
    product_a: comparison.ProductAnalysis | None,
    product_b: comparison.ProductAnalysis | None,
    language: object,
) -> None:
    tabs = st.tabs(
        [
            t("comparison.tab.overview", language),
            t("comparison.tab.head_to_head", language),
            t("comparison.tab.rating", language),
            t("comparison.tab.language", language),
            t("comparison.tab.evidence", language),
        ]
    )
    with tabs[0]:
        _render_overview(payload, language)
    with tabs[1]:
        _render_head_to_head(payload, language)
    with tabs[2]:
        _render_rating_comparison(payload, language)
    with tabs[3]:
        _render_customer_language(payload, language)
    with tabs[4]:
        _render_evidence_comparison(payload, product_a, product_b, language)


def _initialize_source_state(
    options: Mapping[str, ComparisonSource],
) -> None:
    keys = list(options)
    saved_keys = [
        key for key, source in options.items() if source.saved_analysis_id
    ]
    default_a = saved_keys[0] if len(saved_keys) >= 2 else (
        "fixture_a" if "fixture_a" in options else keys[0]
    )
    default_b = saved_keys[1] if len(saved_keys) >= 2 else (
        "fixture_b" if "fixture_b" in options else keys[1]
    )
    if st.session_state.get(SOURCE_A_STATE_KEY) not in options:
        st.session_state[SOURCE_A_STATE_KEY] = default_a
    if st.session_state.get(SOURCE_B_STATE_KEY) not in options:
        st.session_state[SOURCE_B_STATE_KEY] = default_b


def _source_and_name_controls(
    options: Mapping[str, ComparisonSource],
    *,
    side: str,
    language: object,
) -> tuple[str, ComparisonSource, str]:
    source_key_name = SOURCE_A_STATE_KEY if side == "a" else SOURCE_B_STATE_KEY
    name_key = PRODUCT_NAME_A_STATE_KEY if side == "a" else PRODUCT_NAME_B_STATE_KEY
    marker_key = f"{source_key_name}_previous"
    # Streamlit can retain the previously formatted label when the canonical
    # option values are unchanged across a language switch.  Use a
    # language-scoped presentation widget while keeping the canonical source
    # in ``source_key_name``.  This refreshes translated labels without losing
    # A/B selection when the user changes language or leaves this Dataset Mode.
    language_code = normalize_language(language)
    widget_key = f"{source_key_name}_widget_{language_code}"
    language_marker_key = f"{source_key_name}_widget_language"
    if (
        st.session_state.get(language_marker_key) != language_code
        or st.session_state.get(widget_key) not in options
    ):
        st.session_state[widget_key] = st.session_state[source_key_name]
        st.session_state[language_marker_key] = language_code
    source_key = st.selectbox(
        t("comparison.source", language),
        list(options),
        format_func=lambda key: _source_label(options[key], language),
        key=widget_key,
    )
    st.session_state[source_key_name] = source_key
    source = options[source_key]
    # Widget-bound state is removed by Streamlit while Compare Products is not
    # rendered.  Keep the durable display name in ``name_key`` and use a
    # separate transient widget key so Dataset Mode round-trips cannot blank
    # the selected products' names or invalidate the saved comparison state.
    name_widget_key = f"{name_key}_widget"
    source_changed = st.session_state.get(marker_key) != source_key
    if source_changed:
        st.session_state[name_key] = source.default_product_name
        st.session_state[name_widget_key] = source.default_product_name
        st.session_state[marker_key] = source_key
    elif name_widget_key not in st.session_state:
        st.session_state[name_widget_key] = (
            str(st.session_state.get(name_key) or "").strip()
            or source.default_product_name
        )
    name = st.text_input(
        t("comparison.product_name", language),
        key=name_widget_key,
    )
    st.session_state[name_key] = name
    return source_key, source, name


def render_competitor_comparison(
    *,
    language: object = "en",
    output_root: str | Path = COMPARISON_CURRENT_DIR,
    sources: Sequence[ComparisonSource] | None = None,
) -> Mapping[str, Any] | None:
    """Render Step 13; calculation occurs only on the explicit button click."""

    st.title(t("comparison.title", language))
    st.caption(t("comparison.subtitle", language))
    st.info(t("comparison.read_only_notice", language), icon="ℹ️")

    source_options = tuple(sources or available_comparison_sources())
    if len(source_options) < 2:
        st.error(t("comparison.source_unavailable", language))
        return None
    by_key = {source.key: source for source in source_options}
    if sum(bool(source.saved_analysis_id) for source in source_options) >= 2:
        st.caption(t("comparison.saved_priority", language))
    else:
        st.caption(t("comparison.fixture_group_notice", language))
    _initialize_source_state(by_key)
    left, right = st.columns(2)
    with left:
        st.markdown(f"### {t('comparison.product_a', language)}")
        source_a_key, source_a, name_a = _source_and_name_controls(
            by_key, side="a", language=language
        )
    with right:
        st.markdown(f"### {t('comparison.product_b', language)}")
        source_b_key, source_b, name_b = _source_and_name_controls(
            by_key, side="b", language=language
        )

    same_source = source_a_key == source_b_key
    if same_source:
        message_key = (
            "comparison.select_distinct_saved"
            if source_a.saved_analysis_id
            else "comparison.select_distinct"
        )
        st.warning(t(message_key, language))
    if st.button(
        t("comparison.compare", language),
        type="primary",
        disabled=same_source,
        key="competitor_comparison_run",
    ):
        try:
            product_a = load_comparison_source(source_a, product_name=name_a)
            product_b = load_comparison_source(source_b, product_name=name_b)
            payload = comparison.build_comparison(product_a, product_b)
            comparison.save_comparison_outputs(payload, root=output_root)
        except comparison.AnalysisVersionMismatch:
            st.error(t("comparison.version_mismatch", language))
        except (comparison.ComparisonError, OSError, ValueError) as exc:
            st.error(t("comparison.error", language, error=str(exc)))
        else:
            st.session_state[COMPARISON_STATE_KEY] = payload
            # A report belongs to one exact comparison payload.  Never offer
            # a previously generated download after A/B has been rebuilt.
            st.session_state.pop(COMPARISON_REPORT_STATE_KEY, None)
            st.session_state.pop(f"{COMPARISON_REPORT_STATE_KEY}_export", None)
            st.session_state[PRODUCT_A_STATE_KEY] = product_a
            st.session_state[PRODUCT_B_STATE_KEY] = product_b
            st.session_state[SELECTION_STATE_KEY] = (
                source_a_key,
                source_b_key,
                name_a.strip(),
                name_b.strip(),
            )
            st.success(t("comparison.saved", language))

    payload = st.session_state.get(COMPARISON_STATE_KEY)
    if not isinstance(payload, Mapping):
        st.caption(t("comparison.no_results", language))
        return None
    current_selection = (
        source_a_key,
        source_b_key,
        name_a.strip(),
        name_b.strip(),
    )
    if st.session_state.get(SELECTION_STATE_KEY) != current_selection:
        st.warning(t("comparison.source_changed", language))
        return payload
    product_a = st.session_state.get(PRODUCT_A_STATE_KEY)
    product_b = st.session_state.get(PRODUCT_B_STATE_KEY)
    _render_results(
        payload,
        product_a if isinstance(product_a, comparison.ProductAnalysis) else None,
        product_b if isinstance(product_b, comparison.ProductAnalysis) else None,
        language,
    )
    _render_comparison_report_controls(
        output_root=Path(output_root),
        source_a=source_a,
        source_b=source_b,
        language=language,
    )
    return payload


__all__ = [
    "COMPARISON_STATE_KEY",
    "COMPARISON_REPORT_STATE_KEY",
    "PRODUCT_NAME_A_STATE_KEY",
    "PRODUCT_NAME_B_STATE_KEY",
    "SELECTION_STATE_KEY",
    "SOURCE_A_STATE_KEY",
    "SOURCE_B_STATE_KEY",
    "ComparisonSource",
    "available_comparison_sources",
    "load_comparison_source",
    "render_competitor_comparison",
]
