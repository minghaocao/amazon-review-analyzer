"""Offline acceptance tests for Step 16 setup, onboarding and Limited Mode."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import app as dashboard_app
from first_run import mark_first_run_completed
from i18n import TRANSLATIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


class FirstRunUITests(unittest.TestCase):
    def _first_run_app(self, profile: str = "ready") -> tuple[AppTest, tempfile.TemporaryDirectory]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_path = Path(temporary.name) / "first_run.json"
        environment = {
            "ARA_FIRST_RUN_STATE_PATH": str(state_path),
            "ARA_UI_HEALTH_TEST_PROFILE": profile,
        }
        patcher = patch.dict(os.environ, environment, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        app.run()
        self.assertFalse(app.exception)
        return app, temporary

    def test_ready_first_run_is_single_screen_with_all_entries(self) -> None:
        app, _ = self._first_run_app("ready")

        self.assertIn(
            "Welcome to Amazon Review Analyzer",
            {item.value for item in app.title},
        )
        for label in (
            "Try Demo Dataset",
            "Saved Analyses",
            "Paste Reviews",
            "Upload CSV",
            "Amazon Quick Scan",
            "Continue to Dashboard",
        ):
            self.assertFalse(_button(app, label).disabled, label)
        self.assertFalse(app.radio)

    def test_limited_first_run_enables_existing_content_only(self) -> None:
        app, _ = self._first_run_app("limited")

        self.assertFalse(_button(app, "Try Demo Dataset").disabled)
        self.assertFalse(_button(app, "Saved Analyses").disabled)
        self.assertFalse(_button(app, "Continue to Dashboard").disabled)
        self.assertTrue(_button(app, "Paste Reviews").disabled)
        self.assertTrue(_button(app, "Upload CSV").disabled)
        self.assertTrue(_button(app, "Amazon Quick Scan").disabled)
        self.assertTrue(
            any("finish local AI setup later" in item.value for item in app.warning)
        )

    def test_first_run_language_switch_is_live_and_stays_offline(self) -> None:
        app, _ = self._first_run_app("limited")
        app.selectbox(key="ui_language").set_value("zh").run()

        self.assertFalse(app.exception)
        self.assertIn("欢迎使用 Amazon Review Analyzer", {item.value for item in app.title})
        self.assertTrue(_button(app, "粘贴评论").disabled)
        self.assertIn("系统状态", {item.label for item in app.expander})

    def test_try_demo_completes_first_run_with_only_allowed_state(self) -> None:
        app, temporary = self._first_run_app("ready")
        _button(app, "Try Demo Dataset").click().run()

        self.assertFalse(app.exception)
        state_path = Path(temporary.name) / "first_run.json"
        self.assertEqual(
            {"app_version": "1.0.1", "first_run_completed": True},
            json.loads(state_path.read_text(encoding="utf-8")),
        )
        self.assertEqual("Demo Dataset", app.radio(key="dataset_mode").value)


class LimitedModeUITests(unittest.TestCase):
    def _completed_app(self, profile: str = "limited") -> AppTest:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_path = Path(temporary.name) / "first_run.json"
        mark_first_run_completed("1.0.0", state_path)
        patcher = patch.dict(
            os.environ,
            {
                "ARA_FIRST_RUN_STATE_PATH": str(state_path),
                "ARA_UI_HEALTH_TEST_PROFILE": profile,
            },
            clear=False,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_demo_all_five_tabs_work_without_ollama(self) -> None:
        app = self._completed_app()
        expected = {
            "Overview",
            "Product Insights",
            "Customer Language",
            "Aspect Deep Dive",
            "Review Explorer",
        }
        self.assertEqual(expected, {tab.label for tab in app.tabs if tab.label in expected})
        self.assertIn("Limited Mode", {item.value.replace("● ", "") for item in app.warning})

    def test_saved_analysis_and_comparison_routes_remain_available(self) -> None:
        app = self._completed_app()
        mode = app.radio(key="dataset_mode")
        mode.set_value("Saved Analyses").run()
        self.assertFalse(app.exception)
        self.assertIn("Saved Analyses", {item.value for item in app.title})

        app.radio(key="dataset_mode").set_value("Compare Products").run()
        self.assertFalse(app.exception)
        self.assertEqual("Compare Products", app.radio(key="dataset_mode").value)
        self.assertTrue(app.selectbox)

    def test_paste_validation_is_preserved_but_analysis_is_blocked(self) -> None:
        app = self._completed_app()
        app.radio(key="dataset_mode").set_value("Analyze New Dataset").run()
        app.radio(key="analysis_data_source").set_value("Paste Reviews").run()
        app.text_area(key="pasted_reviews").set_value("Great picture.").run()

        self.assertFalse(app.exception)
        self.assertTrue(any("1 review detected" in item.value for item in app.success))
        self.assertTrue(_button(app, "Analyze Reviews").disabled)
        self.assertEqual("Great picture.", app.text_area(key="pasted_reviews").value)

    def test_refresh_health_preserves_language_route_and_input(self) -> None:
        app = self._completed_app()
        app.radio(key="dataset_mode").set_value("Analyze New Dataset").run()
        app.radio(key="analysis_data_source").set_value("Paste Reviews").run()
        app.text_area(key="pasted_reviews").set_value("Sound is clear.").run()
        _button(app, "Refresh System Status").click().run()

        self.assertFalse(app.exception)
        self.assertEqual("en", app.selectbox(key="ui_language").value)
        self.assertEqual("Analyze New Dataset", app.radio(key="dataset_mode").value)
        self.assertEqual("Paste Reviews", app.radio(key="analysis_data_source").value)
        self.assertEqual("Sound is clear.", app.text_area(key="pasted_reviews").value)

    def test_defensive_execution_gate_never_creates_client(self) -> None:
        report = dashboard_app._offline_ui_health_profile("limited")
        fake_st = MagicMock()
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(
                dashboard_app.interactive,
                "create_checked_ollama_client",
            ) as create_client,
        ):
            result = dashboard_app._execute_interactive_analysis(
                pd.DataFrame({"content": ["Great picture."]}),
                "limited_test",
                health_result=report,
            )

        self.assertIsNone(result)
        create_client.assert_not_called()
        fake_st.warning.assert_called()

    def test_amazon_preview_is_saved_for_later_without_inference(self) -> None:
        report = dashboard_app._offline_ui_health_profile("limited")
        result = SimpleNamespace(
            reviews=[
                {
                    "review_id": "A1",
                    "rating": 5,
                    "title": "Great",
                    "content": "Great picture.",
                    "date": "2026-01-01",
                    "verified_purchase": True,
                }
            ],
            review_count=1,
            retrieved_review_count=1,
            availability_status="available",
            to_metadata=lambda: {"asin": "B012345678"},
        )
        fake_st = MagicMock()
        fake_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        with (
            patch.object(dashboard_app, "st", fake_st),
            patch.object(dashboard_app.amazon, "save_browser_amazon_import") as save,
            patch.object(dashboard_app, "_close_amazon_browser_session"),
            patch.object(dashboard_app, "_execute_amazon_analysis") as execute,
        ):
            dashboard_app._render_staged_browser_reviews(result, report)

        save.assert_called_once()
        execute.assert_not_called()


class Step16SafetyAndLocalizationTests(unittest.TestCase):
    def test_diagnostics_redact_paths_credentials_and_reviews(self) -> None:
        report = dashboard_app._offline_ui_health_profile("ready")
        report["checks"]["ollama"]["details"].update(
            {
                "host": "http://user:password@localhost:11434",
                "path": "/Users/private/project",
                "raw_review": "secret review content",
            }
        )
        text = dashboard_app._diagnostics_text(report)

        self.assertNotIn("/Users/", text)
        self.assertNotIn("user:password", text)
        self.assertNotIn("secret review content", text)
        self.assertNotIn("raw_review", text)
        self.assertIn("[redacted]", text)

    def test_step16_translation_keys_have_full_parity(self) -> None:
        required = {
            "health.system",
            "health.system_status",
            "health.refresh",
            "health.limited",
            "health.new_analysis_unavailable",
            "health.advanced_diagnostics",
            "first_run.title",
            "first_run.try_demo",
            "first_run.paste_reviews",
            "first_run.upload_csv",
            "first_run.amazon_quick_scan",
            "first_run.saved_analyses",
            "first_run.documentation",
            "first_run.continue_dashboard",
            "setup.required",
        }
        self.assertTrue(required.issubset(TRANSLATIONS["en"]))
        self.assertTrue(required.issubset(TRANSLATIONS["zh"]))
        self.assertEqual(set(TRANSLATIONS["en"]), set(TRANSLATIONS["zh"]))


if __name__ == "__main__":
    unittest.main()
