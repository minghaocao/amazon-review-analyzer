"""Offline Streamlit acceptance tests for Step 13 comparison mode."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from config import COMPARISON_CURRENT_DIR, DEMO_OUTPUT_PATHS  # noqa: E402


def _hashes(paths: object) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in paths
        if Path(path).is_file()
    }


def _comparison_output_hashes() -> dict[str, str]:
    return _hashes(
        COMPARISON_CURRENT_DIR / name
        for name in (
            "comparison_summary.json",
            "aspect_comparison.csv",
            "comparison_metadata.json",
        )
    )


class CompetitorComparisonAppTests(unittest.TestCase):
    """The comparison page must stay offline and display-only after creation."""

    def setUp(self) -> None:
        library_patcher = patch.object(
            dashboard_app.comparison_dashboard.library,
            "list_saved_analyses",
            return_value=[],
        )
        library_patcher.start()
        self.addCleanup(library_patcher.stop)
        targets = (
            (dashboard_app.interactive, "create_checked_ollama_client"),
            (dashboard_app.interactive, "run_interactive_pipeline"),
            (dashboard_app.interactive, "retry_failed_reviews"),
            (dashboard_app.amazon, "run_amazon_pipeline"),
            (dashboard_app.amazon, "retry_failed_amazon_reviews"),
        )
        self.analysis_mocks = []
        for owner, name in targets:
            patcher = patch.object(
                owner,
                name,
                side_effect=AssertionError(
                    f"{name} must never run in competitor comparison mode"
                ),
            )
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            self.analysis_mocks.append(mocked)

    def _app(self) -> AppTest:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        app.run()
        self.assertFalse(app.exception)
        return app

    def _open_comparison(self, app: AppTest) -> None:
        app.radio(key="dataset_mode").set_value("Compare Products").run()
        self.assertFalse(app.exception)

    def _create_fixture_comparison(self, app: AppTest) -> None:
        self._open_comparison(app)
        app.button(key="competitor_comparison_run").click().run()
        self.assertFalse(app.exception)

    def test_default_remains_demo_and_compare_mode_is_available(self) -> None:
        app = self._app()
        mode = app.radio(key="dataset_mode")
        self.assertEqual("Demo Dataset", mode.value)
        self.assertEqual(
            [
                "Demo Dataset",
                "Analyze New Dataset",
                "Saved Analyses",
                "Compare Products",
            ],
            mode.options,
        )

    def test_english_fixture_renders_five_views_and_writes_only_comparison(self) -> None:
        demo_before = _hashes(DEMO_OUTPUT_PATHS)
        app = self._app()
        self._create_fixture_comparison(app)

        expected_tabs = {
            "Comparison Overview",
            "Head-to-Head",
            "Rating Comparison",
            "Customer Language",
            "Evidence Comparison",
        }
        self.assertTrue(expected_tabs.issubset({tab.label for tab in app.tabs}))
        self.assertEqual(
            {
                "aspect_comparison.csv",
                "comparison_metadata.json",
                "comparison_summary.json",
            },
            {path.name for path in COMPARISON_CURRENT_DIR.iterdir() if path.is_file()},
        )
        self.assertEqual(demo_before, _hashes(DEMO_OUTPUT_PATHS))
        for mocked in self.analysis_mocks:
            mocked.assert_not_called()

    def test_language_switch_rerenders_without_recomputing_or_rewriting(self) -> None:
        app = self._app()
        self._create_fixture_comparison(app)
        before = _comparison_output_hashes()

        app.selectbox(key="ui_language").set_value("zh").run()
        self.assertFalse(app.exception)
        self.assertEqual(before, _comparison_output_hashes())
        expected_tabs = {"对比概览", "维度对比", "评分对比", "客户语言", "证据对比"}
        self.assertTrue(expected_tabs.issubset({tab.label for tab in app.tabs}))
        self.assertIn("竞品对比", {title.value for title in app.title})
        for mocked in self.analysis_mocks:
            mocked.assert_not_called()

    def test_customer_phrases_and_evidence_remain_verbatim_in_chinese(self) -> None:
        app = self._app()
        self._create_fixture_comparison(app)
        app.selectbox(key="ui_language").set_value("zh").run()
        self.assertFalse(app.exception)

        rendered = "\n".join(
            str(item.value)
            for collection in (app.markdown, app.caption, app.info)
            for item in collection
        )
        self.assertIn("sharp picture", rendered)
        self.assertIn("the deep blacks make movies look fantastic", rendered)
        self.assertNotIn("comparison.", rendered)


if __name__ == "__main__":
    unittest.main()
