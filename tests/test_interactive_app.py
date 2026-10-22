"""Small offline Streamlit smoke tests for Step 10A mode selection."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest


os.environ.setdefault("ARA_UI_HEALTH_TEST_PROFILE", "ready")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_UI_TEST_FIRST_RUN = (
    Path(tempfile.gettempdir()) / f"ara_ui_first_run_{os.getpid()}.json"
)
_UI_TEST_FIRST_RUN.write_text(
    '{"app_version":"1.0.0","first_run_completed":true}',
    encoding="utf-8",
)
os.environ["ARA_FIRST_RUN_STATE_PATH"] = str(_UI_TEST_FIRST_RUN)

import app as dashboard_app  # noqa: E402
from amazon_url import AmazonProductURL, AmazonURLValidationError  # noqa: E402
from config import DEMO_OUTPUT_PATHS  # noqa: E402
from scraper import (  # noqa: E402
    TRUSTED_CA_BUNDLE_UNAVAILABLE,
    ScrapeResult,
    ScrapeStatus,
)


class InteractiveAppSmokeTests(unittest.TestCase):
    def _app(self) -> AppTest:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=20)
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_default_mode_is_demo_dataset(self) -> None:
        app = self._app()
        dataset_mode = next(radio for radio in app.radio if radio.label == "Dataset Mode")
        self.assertEqual("Demo Dataset", dataset_mode.value)

    def test_analyze_mode_is_safe_with_or_without_a_current_session(self) -> None:
        app = self._app()
        next(radio for radio in app.radio if radio.label == "Dataset Mode").set_value(
            "Analyze New Dataset"
        ).run()
        self.assertFalse(app.exception)
        workspace = [radio for radio in app.radio if radio.label == "Workspace"]
        if workspace:
            workspace[0].set_value("New Input").run()
            self.assertFalse(app.exception)
        else:
            self.assertTrue(
                any("No custom analysis" in item.value for item in app.caption)
            )
        source = next(
            radio for radio in app.radio if radio.label == "Choose Analysis Source"
        )
        self.assertEqual("Amazon Quick Scan", source.value)
        self.assertEqual(
            ["Amazon Quick Scan", "Upload Dataset", "Paste Reviews"],
            source.options,
        )

    def test_paste_preview_does_not_start_analysis(self) -> None:
        app = self._app()
        next(radio for radio in app.radio if radio.label == "Dataset Mode").set_value(
            "Analyze New Dataset"
        ).run()
        next(
            radio for radio in app.radio if radio.label == "Choose Analysis Source"
        ).set_value("Paste Reviews").run()
        app.text_area(key="pasted_reviews").set_value("Great picture.\n\nCheap remote.").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("2 reviews detected" in item.value for item in app.success))
        analyze = next(button for button in app.button if button.label == "Analyze Reviews")
        self.assertFalse(analyze.disabled)

    def test_single_paste_review_is_accepted_and_analysis_is_enabled(self) -> None:
        app = self._app()
        next(radio for radio in app.radio if radio.label == "Dataset Mode").set_value(
            "Analyze New Dataset"
        ).run()
        next(
            radio for radio in app.radio if radio.label == "Choose Analysis Source"
        ).set_value("Paste Reviews").run()
        app.text_area(key="pasted_reviews").set_value(
            "The picture quality is excellent, but the remote feels cheap "
            "and the menu is slow."
        ).run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any(item.value == "1 review detected" for item in app.success)
        )
        analyze = next(button for button in app.button if button.label == "Analyze Reviews")
        self.assertFalse(analyze.disabled)

    def test_upload_csv_remains_available_after_leaving_demo(self) -> None:
        app = self._app()
        next(radio for radio in app.radio if radio.label == "Dataset Mode").set_value(
            "Analyze New Dataset"
        ).run()

        source = next(
            radio for radio in app.radio if radio.label == "Choose Analysis Source"
        )
        self.assertIn("Upload Dataset", source.options)
        source.set_value("Upload Dataset").run()
        uploader = next(
            item for item in app.file_uploader if item.label == "Upload Reviews CSV"
        )
        self.assertFalse(uploader.disabled)
        self.assertTrue(
            any("Choose a CSV file" in item.value for item in app.info)
        )

    def test_compact_demo_keeps_all_five_dashboards_populated(self) -> None:
        app = self._app()
        compact = next(
            toggle for toggle in app.toggle if toggle.label == "Compact interview view"
        )
        self.assertTrue(compact.value)

        top_level_labels = {
            "Overview",
            "Product Insights",
            "Customer Language",
            "Aspect Deep Dive",
            "Review Explorer",
        }
        rendered = {tab.label: tab for tab in app.tabs if tab.label in top_level_labels}
        self.assertEqual(top_level_labels, set(rendered))
        self.assertTrue(rendered["Overview"].info)
        self.assertTrue(rendered["Product Insights"].dataframe)
        self.assertTrue(rendered["Customer Language"].dataframe)
        self.assertTrue(rendered["Customer Language"].markdown)
        self.assertTrue(rendered["Aspect Deep Dive"].metric)
        self.assertTrue(rendered["Aspect Deep Dive"].dataframe)
        self.assertTrue(rendered["Aspect Deep Dive"].markdown)
        self.assertTrue(rendered["Review Explorer"].dataframe)
        self.assertTrue(rendered["Review Explorer"].expander)

    def test_demo_uses_limited_sample_wording_and_step12_sections(self) -> None:
        app = self._app()
        warnings = [item.value for item in app.warning]
        self.assertTrue(any("Limited Sample" in value for value in warnings))
        subheaders = {item.value for item in app.subheader}
        self.assertIn("Overall Sentiment", subheaders)
        self.assertIn("Product vs Shopping Experience", subheaders)
        self.assertIn("Customer Priority Matrix", subheaders)
        self.assertIn("Opportunity Ranking", subheaders)
        self.assertIn("Language by Aspect", subheaders)

    def test_review_explorer_exposes_all_six_step12_shortcuts(self) -> None:
        app = self._app()
        quick_filter = next(
            radio for radio in app.radio if radio.label == "Quick filters"
        )
        self.assertEqual(
            [
                "All",
                "Positive Evidence",
                "Negative Evidence",
                "1–2 Star",
                "4–5 Star",
                "Product Issues",
                "Shopping Experience",
            ],
            quick_filter.options,
        )

    def test_rendering_demo_does_not_modify_frozen_outputs(self) -> None:
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in DEMO_OUTPUT_PATHS
            if path.is_file()
        }
        self.assertTrue(before)

        self._app()

        after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in before
        }
        self.assertEqual(before, after)

    def test_demo_route_never_checks_ollama(self) -> None:
        with (
            patch.object(dashboard_app, "_inject_styles"),
            patch.object(
                dashboard_app.interactive,
                "current_analysis_available",
                return_value=False,
            ),
            patch.object(
                dashboard_app.amazon,
                "amazon_analysis_available",
                return_value=False,
            ),
            patch.object(
                dashboard_app.amazon,
                "amazon_import_available",
                return_value=False,
            ),
            patch.object(
                dashboard_app,
                "_render_sidebar_controls",
                return_value=("Demo Dataset", None, "New Input", True),
            ),
            patch.object(dashboard_app, "_render_saved_dashboard") as render,
            patch.object(
                dashboard_app.interactive,
                "create_checked_ollama_client",
            ) as check_ollama,
        ):
            dashboard_app.main()

        check_ollama.assert_not_called()
        render.assert_called_once_with(
            Path(dashboard_app.DEFAULT_OUTPUTS_DIR),
            dataset_label="Demo Dataset",
            key_prefix="demo",
            compact_view=True,
        )

    def test_ollama_is_only_checked_after_analyze_is_clicked(self) -> None:
        reviews = dashboard_app.interactive.parse_pasted_reviews("Great picture.")
        with (
            patch.object(dashboard_app.st, "button", return_value=False),
            patch.object(dashboard_app, "_execute_interactive_analysis") as execute,
        ):
            dashboard_app._render_analysis_action(reviews, "one_review")
        execute.assert_not_called()

        with (
            patch.object(dashboard_app.st, "button", return_value=True),
            patch.object(dashboard_app, "_execute_interactive_analysis") as execute,
        ):
            dashboard_app._render_analysis_action(reviews, "one_review")
        execute.assert_called_once()
        analyzed = execute.call_args.args[0]
        self.assertEqual(1, len(analyzed))

    def test_upload_or_paste_full_analysis_is_not_capped_at_one_hundred(self) -> None:
        reviews = dashboard_app.interactive.parse_pasted_reviews(
            "\n".join(f"Review number {index}." for index in range(1, 102))
        )
        fake_st = MagicMock()
        fake_st.button.return_value = False

        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(dashboard_app, "_execute_interactive_analysis") as execute,
        ):
            dashboard_app._render_analysis_action(reviews, "large_dataset")

        analyze_call = next(
            call
            for call in fake_st.button.call_args_list
            if call.args and call.args[0] == "Analyze Reviews"
        )
        self.assertFalse(analyze_call.kwargs["disabled"])
        fake_st.checkbox.assert_not_called()
        execute.assert_not_called()

    def test_amazon_url_mode_shows_import_controls_without_fetching(self) -> None:
        app = self._app()
        next(radio for radio in app.radio if radio.label == "Dataset Mode").set_value(
            "Analyze New Dataset"
        ).run()
        next(
            radio for radio in app.radio if radio.label == "Choose Analysis Source"
        ).set_value("Amazon Quick Scan").run()
        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Analyze reviews currently accessible" in item.value
                for item in app.caption
            )
        )
        self.assertFalse(
            any(
                selectbox.label in {"Maximum Reviews", "Target Reviews"}
                for selectbox in app.selectbox
            )
        )
        import_button = next(
            button for button in app.button if button.label == "Open Amazon Product"
        )
        self.assertTrue(import_button.disabled)
        app.text_input(key="amazon_product_url").set_value(
            "https://www.amazon.com/dp/B012345678"
        ).run()
        import_button = next(
            button for button in app.button if button.label == "Open Amazon Product"
        )
        self.assertFalse(import_button.disabled)

    def test_current_analysis_failed_notice_runs_the_failed_retry_action(self) -> None:
        warning_slot = Mock()
        button_slot = Mock()
        button_slot.button.return_value = True

        with (
            patch.object(
                dashboard_app.interactive,
                "failed_review_count",
                return_value=1,
            ),
            patch.object(
                dashboard_app.st,
                "empty",
                side_effect=[warning_slot, button_slot],
            ),
            patch.object(
                dashboard_app,
                "_execute_interactive_analysis",
                return_value=SimpleNamespace(failed_reviews=0),
            ) as execute,
        ):
            dashboard_app._render_failed_review_retry_panel(Path("/safe/current"))

        warning_slot.warning.assert_called_once_with(
            "1 review failed during analysis."
        )
        button_slot.button.assert_called_once_with(
            "Retry Failed Reviews",
            type="primary",
            key="retry_failed_reviews_button_upload",
        )
        execute.assert_called_once_with(
            None,
            "retry_failed",
            retry_failed=True,
        )
        warning_slot.success.assert_called_once_with(
            "All failed reviews were analyzed successfully."
        )
        button_slot.empty.assert_called_once_with()


class AmazonImportUITests(unittest.TestCase):
    def setUp(self) -> None:
        self.product = AmazonProductURL(
            marketplace="amazon.com",
            asin="B012345678",
            canonical_product_url="https://www.amazon.com/dp/B012345678",
        )
        self.reviews = [
            {
                "review_id": "R-AMZ-1",
                "rating": 5,
                "title": "Great",
                "content": "Great picture quality.",
                "date": "2026-09-01",
                "verified_purchase": True,
            }
        ]

    def _streamlit_for_import(self, url: str) -> MagicMock:
        def columns(count: int) -> tuple[MagicMock, ...]:
            rendered = tuple(MagicMock() for _ in range(count))
            for column in rendered:
                column.button.return_value = False
            return rendered

        fake_st = MagicMock()
        fake_st.session_state = {}
        fake_st.text_input.return_value = url
        fake_st.selectbox.return_value = 10
        fake_st.button.side_effect = (
            lambda *args, **kwargs: kwargs.get("key") == "open_amazon_product"
        )
        fake_st.columns.side_effect = columns
        return fake_st

    def test_scrape_states_have_friendly_non_traceback_messages(self) -> None:
        expected_cases = (
            (
                ScrapeStatus.BLOCKED,
                dashboard_app.AMAZON_CHALLENGE_REASON,
                "currently prevents automated access",
            ),
            (ScrapeStatus.DNS_ERROR, None, "Unable to resolve Amazon hostname"),
            (ScrapeStatus.TIMEOUT, None, "Connection to Amazon timed out"),
            (ScrapeStatus.CONNECTION_REFUSED, None, "refused the connection"),
            (ScrapeStatus.TLS_ERROR, None, "TLS connection to Amazon failed"),
            (ScrapeStatus.PROXY_ERROR, None, "configured proxy could not reach Amazon"),
            (ScrapeStatus.REDIRECT_ERROR, None, "redirect could not be completed"),
            (ScrapeStatus.HTTP_ERROR, None, "HTTP error"),
            (ScrapeStatus.CONNECTION_ERROR, None, "Unable to reach Amazon"),
            (ScrapeStatus.NO_REVIEWS, None, "No public reviews were found"),
            (
                ScrapeStatus.STRUCTURE_NOT_RECOGNIZED,
                None,
                "structure was not recognized",
            ),
        )
        for status, reason, phrase in expected_cases:
            with self.subTest(status=status, reason=reason):
                message = dashboard_app._friendly_amazon_import_error(status, reason)
                self.assertIn(phrase, message)
                self.assertNotIn("Traceback", message)

    def test_missing_ca_bundle_has_exact_safe_message(self) -> None:
        self.assertEqual(
            "Trusted CA certificate bundle is unavailable.",
            dashboard_app._friendly_amazon_import_error(
                ScrapeStatus.TLS_ERROR,
                TRUSTED_CA_BUNDLE_UNAVAILABLE,
            ),
        )

    def test_browser_access_limitation_is_warning_not_error(self) -> None:
        result = dashboard_app.AmazonBrowserImportResult(
            status=dashboard_app.BrowserImportStatus.REVIEWS_UNAVAILABLE,
            marketplace=self.product.marketplace,
            asin=self.product.asin,
            product_url=self.product.canonical_url,
            reviews_url="https://www.amazon.com/product-reviews/B012345678",
            availability_status=(
                dashboard_app.ReviewAvailabilityStatus.ACCESS_LIMITED
            ),
            stop_reason="amazon_access_limited",
        )
        fake_st = MagicMock()

        with patch.object(dashboard_app, "st", fake_st):
            handled = dashboard_app._render_nonfatal_browser_availability(result)

        self.assertTrue(handled)
        fake_st.warning.assert_called_once()
        fake_st.error.assert_not_called()

    def test_open_product_does_not_save_or_start_ollama(self) -> None:
        fake_st = self._streamlit_for_import(
            "https://www.amazon.com/dp/B012345678"
        )
        result = dashboard_app.AmazonBrowserImportResult(
            status=dashboard_app.BrowserImportStatus.PRODUCT_PAGE_LOADED,
            asin=self.product.asin,
            marketplace=self.product.marketplace,
            product_url=self.product.canonical_url,
            reviews_url="https://www.amazon.com/product-reviews/B012345678",
        )
        session = MagicMock(spec=dashboard_app.AmazonBrowserSession)
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(
                dashboard_app,
                "parse_amazon_url",
                return_value=self.product,
            ) as parse,
            patch.object(
                dashboard_app,
                "open_amazon_product",
                return_value=(session, result),
            ) as open_product,
            patch.object(
                dashboard_app.amazon,
                "save_browser_amazon_import",
            ) as save,
            patch.object(
                dashboard_app.amazon,
                "amazon_import_available",
                return_value=False,
            ),
            patch.object(dashboard_app, "_execute_amazon_analysis") as analyze,
        ):
            dashboard_app._render_amazon_url_input()

        parse.assert_called_once_with("https://www.amazon.com/dp/B012345678")
        open_product.assert_called_once_with(
            self.product,
            navigation_timeout_ms=dashboard_app.DEFAULT_NAVIGATION_TIMEOUT_MS,
        )
        save.assert_not_called()
        analyze.assert_not_called()

    def test_invalid_url_does_not_fetch_or_replace_saved_data(self) -> None:
        fake_st = self._streamlit_for_import("https://example.com/dp/B012345678")
        error = AmazonURLValidationError(
            "This does not appear to be a supported Amazon product URL.",
            code="unsupported_marketplace",
        )
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(
                dashboard_app,
                "parse_amazon_url",
                side_effect=error,
            ),
            patch.object(dashboard_app, "open_amazon_product") as open_product,
            patch.object(
                dashboard_app.amazon,
                "save_browser_amazon_import",
            ) as save,
            patch.object(
                dashboard_app.amazon,
                "amazon_import_available",
                return_value=False,
            ),
        ):
            dashboard_app._render_amazon_url_input()

        open_product.assert_not_called()
        save.assert_not_called()
        fake_st.error.assert_called_once_with(str(error))
        self.assertNotIn(
            dashboard_app.AMAZON_CHALLENGE_MESSAGE,
            fake_st.error.call_args.args[0],
        )
        rendered_button_labels = [
            call.args[0]
            for call in fake_st.button.call_args_list
            if call.args
        ]
        self.assertNotIn("Paste Reviews Instead", rendered_button_labels)
        self.assertNotIn("Upload Dataset Instead", rendered_button_labels)

    def test_challenge_waits_for_manual_continue_and_is_not_saved(self) -> None:
        fake_st = self._streamlit_for_import(
            "https://www.amazon.com/dp/B012345678"
        )
        result = dashboard_app.AmazonBrowserImportResult(
            status=dashboard_app.BrowserImportStatus.CHALLENGE_REQUIRED,
            asin=self.product.asin,
            marketplace=self.product.marketplace,
            product_url=self.product.canonical_url,
            reviews_url="https://www.amazon.com/product-reviews/B012345678",
            challenge_detected=True,
        )
        session = MagicMock(spec=dashboard_app.AmazonBrowserSession)
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(
                dashboard_app,
                "parse_amazon_url",
                return_value=self.product,
            ),
            patch.object(
                dashboard_app,
                "open_amazon_product",
                return_value=(session, result),
            ),
            patch.object(
                dashboard_app.amazon,
                "save_browser_amazon_import",
            ) as save,
            patch.object(
                dashboard_app.amazon,
                "amazon_import_available",
                return_value=False,
            ),
            patch.object(dashboard_app, "_execute_amazon_analysis") as analyze,
        ):
            dashboard_app._render_amazon_url_input()

        save.assert_not_called()
        analyze.assert_not_called()
        fake_st.error.assert_not_called()
        warning = fake_st.warning.call_args.args[0]
        self.assertIn("manual verification", warning)
        self.assertIn("will not solve or bypass", warning)
        self.assertNotIn("Traceback", warning)
        self.assertIs(
            session,
            fake_st.session_state[dashboard_app.AMAZON_BROWSER_SESSION_KEY],
        )

    def test_manual_continue_reuses_existing_browser_session(self) -> None:
        challenge = dashboard_app.AmazonBrowserImportResult(
            status=dashboard_app.BrowserImportStatus.CHALLENGE_REQUIRED,
            marketplace=self.product.marketplace,
            asin=self.product.asin,
            product_url=self.product.canonical_url,
            reviews_url="https://www.amazon.com/product-reviews/B012345678",
            challenge_detected=True,
        )
        loaded = dashboard_app.AmazonBrowserImportResult(
            status=dashboard_app.BrowserImportStatus.PRODUCT_PAGE_LOADED,
            marketplace=self.product.marketplace,
            asin=self.product.asin,
            product_url=self.product.canonical_url,
            reviews_url=challenge.reviews_url,
            challenge_detected=True,
        )
        session = MagicMock(spec=dashboard_app.AmazonBrowserSession)
        session.continue_after_verification.return_value = loaded
        continue_column = MagicMock()
        cancel_column = MagicMock()
        continue_column.button.return_value = True
        cancel_column.button.return_value = False
        fake_st = MagicMock()
        fake_st.session_state = {}
        fake_st.columns.return_value = (continue_column, cancel_column)

        with patch.object(dashboard_app, "st", fake_st):
            actual = dashboard_app._render_browser_verification(
                challenge,
                session,
            )

        self.assertIs(loaded, actual)
        session.continue_after_verification.assert_called_once_with()
        self.assertIs(
            loaded,
            fake_st.session_state[dashboard_app.AMAZON_BROWSER_RESULT_KEY],
        )

    def test_browser_preview_does_not_save_or_call_ollama(self) -> None:
        result = dashboard_app.AmazonBrowserImportResult(
            status=dashboard_app.BrowserImportStatus.SUCCESS,
            marketplace=self.product.marketplace,
            asin=self.product.asin,
            product_url=self.product.canonical_url,
            reviews_url="https://www.amazon.com/product-reviews/B012345678",
            reviews=self.reviews,
            pages_visited=1,
        )
        fake_st = MagicMock()
        fake_st.button.return_value = False
        fake_st.session_state = {}
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(
                dashboard_app.amazon,
                "save_browser_amazon_import",
            ) as save,
            patch.object(dashboard_app, "_execute_amazon_analysis") as analyze,
        ):
            dashboard_app._render_staged_browser_reviews(result)

        save.assert_not_called()
        analyze.assert_not_called()
        preview = fake_st.dataframe.call_args.args[0]
        self.assertEqual(["R-AMZ-1"], preview["review_id"].tolist())

    def test_access_limited_quick_scan_is_warning_and_remains_analyzable(self) -> None:
        reviews = [
            {**self.reviews[0], "review_id": f"R-AMZ-{index}"}
            for index in range(1, 9)
        ]
        result = dashboard_app.AmazonBrowserImportResult(
            status=dashboard_app.BrowserImportStatus.SUCCESS,
            marketplace=self.product.marketplace,
            asin=self.product.asin,
            product_url=self.product.canonical_url,
            reviews_url="https://www.amazon.com/product-reviews/B012345678",
            reviews=reviews,
            retrieved_review_count=10,
            unique_review_count=8,
            target_review_count=None,
            availability_status=(
                dashboard_app.ReviewAvailabilityStatus.ACCESS_LIMITED
            ),
            stop_reason="amazon_access_limited",
        )
        fake_st = MagicMock()
        fake_st.button.return_value = False
        fake_st.session_state = {}

        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(dashboard_app.amazon, "save_browser_amazon_import"),
            patch.object(dashboard_app, "_execute_amazon_analysis"),
        ):
            dashboard_app._render_staged_browser_reviews(result)

        fake_st.error.assert_not_called()
        self.assertIn(
            "limited set of reviews",
            fake_st.warning.call_args.args[0],
        )
        analyze_call = next(
            call
            for call in fake_st.button.call_args_list
            if call.args and call.args[0] == "Analyze Available Reviews"
        )
        self.assertFalse(analyze_call.kwargs["disabled"])

    def test_zero_unique_reviews_disables_available_review_analysis(self) -> None:
        result = dashboard_app.AmazonBrowserImportResult(
            status=dashboard_app.BrowserImportStatus.SUCCESS,
            marketplace=self.product.marketplace,
            asin=self.product.asin,
            product_url=self.product.canonical_url,
            reviews_url="https://www.amazon.com/product-reviews/B012345678",
            reviews=[],
            unique_review_count=0,
        )
        fake_st = MagicMock()
        fake_st.button.return_value = False
        fake_st.session_state = {}

        with patch.object(dashboard_app, "st", fake_st):
            dashboard_app._render_staged_browser_reviews(result)

        analyze_call = next(
            call
            for call in fake_st.button.call_args_list
            if call.args and call.args[0] == "Analyze Available Reviews"
        )
        self.assertTrue(analyze_call.kwargs["disabled"])

    def test_quick_scan_dataset_summary_uses_imported_review_coverage(self) -> None:
        fake_st = MagicMock()
        data = SimpleNamespace(business_summary={"ranking_rules": {}})
        kpis = {
            "total_reviews": 8,
            "successful_reviews": 8,
            "analysis_coverage": 1.0,
        }

        with patch.object(dashboard_app, "st", fake_st):
            dashboard_app._render_dataset_summary_sidebar(
                data,
                kpis,
                quick_scan=True,
            )

        rendered = [call.args[0] for call in fake_st.write.call_args_list]
        self.assertIn("**Source:** Amazon Quick Scan", rendered)
        self.assertIn("**Reviews Retrieved:** 8", rendered)
        self.assertIn("**Reviews Analyzed:** 8", rendered)
        self.assertIn("**Analysis Coverage:** 100.0%", rendered)
        self.assertFalse(any("Retrieval Completion" in value for value in rendered))

    def test_quick_scan_disclaimer_is_informational(self) -> None:
        fake_st = MagicMock()

        with patch.object(dashboard_app, "st", fake_st):
            dashboard_app._render_quick_scan_disclaimer()

        fake_st.info.assert_called_once_with(
            dashboard_app.AMAZON_QUICK_SCAN_DISCLAIMER
        )
        fake_st.warning.assert_not_called()
        fake_st.error.assert_not_called()

    def test_analyze_these_reviews_saves_then_reuses_existing_pipeline(self) -> None:
        result = dashboard_app.AmazonBrowserImportResult(
            status=dashboard_app.BrowserImportStatus.SUCCESS,
            marketplace=self.product.marketplace,
            asin=self.product.asin,
            product_url=self.product.canonical_url,
            reviews_url="https://www.amazon.com/product-reviews/B012345678",
            reviews=self.reviews,
            pages_visited=1,
        )
        fake_st = MagicMock()
        fake_st.button.return_value = True
        fake_st.session_state = {}
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(
                dashboard_app.amazon,
                "save_browser_amazon_import",
            ) as save,
            patch.object(
                dashboard_app,
                "_close_amazon_browser_session",
            ) as close,
            patch.object(dashboard_app, "_execute_amazon_analysis") as analyze,
        ):
            dashboard_app._render_staged_browser_reviews(result)

        saved_frame = save.call_args.args[0]
        self.assertEqual(["R-AMZ-1"], saved_frame["review_id"].tolist())
        self.assertEqual(result.to_metadata(), save.call_args.args[1])
        close.assert_called_once_with(clear_result=True)
        analyze.assert_called_once_with()

    def test_reset_amazon_closes_browser_before_resetting_files(self) -> None:
        fake_st = SimpleNamespace(
            session_state={"confirm_amazon_reset": True},
        )
        calls: list[str] = []
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(
                dashboard_app,
                "_close_amazon_browser_session",
                side_effect=lambda **kwargs: calls.append("close"),
            ),
            patch.object(
                dashboard_app.amazon,
                "reset_amazon_analysis",
                side_effect=lambda: calls.append("reset"),
            ),
            patch.object(dashboard_app._load_cached_dashboard, "clear"),
        ):
            dashboard_app._reset_amazon_analysis()

        self.assertEqual(["close", "reset"], calls)
        self.assertNotIn("confirm_amazon_reset", fake_st.session_state)

    def test_paste_fallback_reuses_existing_mode_and_preserves_url(self) -> None:
        original_url = "https://www.amazon.com/dp/B012345678"
        fake_st = SimpleNamespace(
            session_state={"amazon_product_url": original_url}
        )
        with patch.object(dashboard_app, "st", fake_st):
            dashboard_app._select_analysis_data_source("Paste Reviews")

        self.assertEqual("Analyze New Dataset", fake_st.session_state["dataset_mode"])
        self.assertEqual("New Input", fake_st.session_state["analysis_workspace"])
        self.assertEqual("Paste Reviews", fake_st.session_state["analysis_data_source"])
        self.assertEqual(
            original_url,
            fake_st.session_state[dashboard_app.PRESERVED_AMAZON_URL_KEY],
        )

    def test_upload_fallback_reuses_existing_mode_and_preserves_url(self) -> None:
        original_url = "https://www.amazon.com/dp/B012345678"
        fake_st = SimpleNamespace(
            session_state={"amazon_product_url": original_url}
        )
        with patch.object(dashboard_app, "st", fake_st):
            dashboard_app._select_analysis_data_source("Upload CSV")

        self.assertEqual("Analyze New Dataset", fake_st.session_state["dataset_mode"])
        self.assertEqual("New Input", fake_st.session_state["analysis_workspace"])
        self.assertEqual(
            "Upload Dataset",
            fake_st.session_state["analysis_data_source"],
        )
        self.assertEqual(
            original_url,
            fake_st.session_state[dashboard_app.PRESERVED_AMAZON_URL_KEY],
        )

    def test_amazon_url_is_restored_after_using_a_fallback(self) -> None:
        original_url = "https://www.amazon.com/dp/B012345678"
        fake_st = MagicMock()
        fake_st.session_state = {
            dashboard_app.PRESERVED_AMAZON_URL_KEY: original_url,
        }
        fake_st.text_input.side_effect = lambda *args, **kwargs: (
            fake_st.session_state[kwargs["key"]]
        )
        fake_st.selectbox.return_value = 50
        fake_st.button.return_value = False
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(
                dashboard_app.amazon,
                "amazon_import_available",
                return_value=False,
            ),
        ):
            dashboard_app._render_amazon_url_input()

        self.assertEqual(original_url, fake_st.session_state["amazon_product_url"])
        self.assertEqual(
            original_url,
            fake_st.session_state[dashboard_app.PRESERVED_AMAZON_URL_KEY],
        )

    def test_no_reviews_is_not_presented_as_an_amazon_challenge(self) -> None:
        fake_st = MagicMock()
        with patch.object(dashboard_app, "st", fake_st):
            dashboard_app._render_amazon_import_issue(
                ScrapeStatus.NO_REVIEWS,
                "no_public_reviews",
            )

        message = fake_st.info.call_args.args[0]
        self.assertEqual("No public reviews were found for this product.", message)
        self.assertNotIn("prevents automated access", message)
        fake_st.warning.assert_not_called()
        fake_st.error.assert_not_called()
        fake_st.columns.assert_not_called()

    def test_challenge_fallback_does_not_modify_demo_outputs(self) -> None:
        demo_directory = PROJECT_ROOT / dashboard_app.DEFAULT_OUTPUTS_DIR

        def output_hashes() -> dict[str, str]:
            return {
                str(path.relative_to(demo_directory)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(demo_directory.rglob("*"))
                if path.is_file()
            }

        before = output_hashes()
        fake_st = MagicMock()
        fake_st.columns.return_value = (MagicMock(), MagicMock())
        with patch.object(dashboard_app, "st", fake_st):
            dashboard_app._render_amazon_import_issue(
                ScrapeStatus.BLOCKED,
                dashboard_app.AMAZON_CHALLENGE_REASON,
            )

        self.assertTrue(before)
        self.assertEqual(before, output_hashes())

    def test_saved_preview_limits_rows_and_analysis_needs_its_own_click(self) -> None:
        fake_st = MagicMock()
        fake_st.button.return_value = True
        reviews = pd.DataFrame(
            [
                {
                    "review_id": f"R{index:03d}",
                    "rating": 5,
                    "title": f"Title {index}",
                    "content": f"Review {index}",
                    "date": "2026-09-01",
                    "verified_purchase": True,
                }
                for index in range(1, 8)
            ]
        )
        imported = SimpleNamespace(
            reviews=reviews,
            metadata={"asin": "B012345678", "marketplace": "amazon.com"},
        )
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(
                dashboard_app.amazon,
                "amazon_import_available",
                return_value=True,
            ),
            patch.object(
                dashboard_app.amazon,
                "load_amazon_import",
                return_value=imported,
            ),
            patch.object(
                dashboard_app.amazon,
                "interrupted_amazon_analysis_available",
                return_value=False,
            ),
            patch.object(dashboard_app, "_execute_amazon_analysis") as analyze,
        ):
            available = dashboard_app._render_amazon_import_preview()

        self.assertTrue(available)
        displayed_frame = fake_st.dataframe.call_args.args[0]
        self.assertEqual(5, len(displayed_frame))
        analyze.assert_called_once_with()

    def test_saved_checkpoint_can_resume_without_restarting(self) -> None:
        fake_st = MagicMock()
        fake_st.button.side_effect = [True, False]
        imported = SimpleNamespace(
            reviews=pd.DataFrame(self.reviews),
            metadata={"asin": "B012345678", "marketplace": "amazon.com"},
        )
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(
                dashboard_app.amazon,
                "amazon_import_available",
                return_value=True,
            ),
            patch.object(
                dashboard_app.amazon,
                "load_amazon_import",
                return_value=imported,
            ),
            patch.object(
                dashboard_app.amazon,
                "interrupted_amazon_analysis_available",
                return_value=True,
            ),
            patch.object(dashboard_app, "_execute_amazon_analysis") as analyze,
        ):
            dashboard_app._render_amazon_import_preview()

        analyze.assert_called_once_with(resume=True)

    def test_amazon_retry_panel_uses_amazon_retry_path(self) -> None:
        warning_slot = Mock()
        button_slot = Mock()
        button_slot.button.return_value = True
        with (
            patch.object(
                dashboard_app.amazon,
                "failed_amazon_review_count",
                return_value=2,
            ),
            patch.object(
                dashboard_app.st,
                "empty",
                side_effect=[warning_slot, button_slot],
            ),
            patch.object(
                dashboard_app,
                "_execute_amazon_analysis",
                return_value=SimpleNamespace(failed_reviews=0),
            ) as execute,
        ):
            dashboard_app._render_failed_review_retry_panel(
                Path("/safe/amazon/current"),
                source="amazon",
            )

        button_slot.button.assert_called_once_with(
            "Retry Failed Reviews",
            type="primary",
            key="retry_failed_reviews_button_amazon",
        )
        execute.assert_called_once_with(retry_failed=True)


if __name__ == "__main__":
    unittest.main()
