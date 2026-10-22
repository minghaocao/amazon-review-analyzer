"""Offline Streamlit acceptance tests for the Step 12.5 localized dashboard."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from config import DEMO_OUTPUT_PATHS  # noqa: E402


ENGLISH_TABS = [
    "Overview",
    "Product Insights",
    "Customer Language",
    "Aspect Deep Dive",
    "Review Explorer",
]
CHINESE_TABS = [
    "概览",
    "产品洞察",
    "客户语言",
    "维度深度分析",
    "评论浏览器",
]


def _file_hashes() -> dict[Path, str]:
    """Hash the frozen demo inputs and outputs without changing them."""

    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DEMO_OUTPUT_PATHS
        if path.is_file()
    }


def _top_level_tab_labels(app: AppTest, expected: list[str]) -> list[str]:
    """Ignore the four nested Evidence tabs when inspecting primary navigation."""

    expected_set = set(expected)
    return [tab.label for tab in app.tabs if tab.label in expected_set]


class LocalizedDashboardAppTests(unittest.TestCase):
    """Exercise language switching against frozen outputs and no live model."""

    def setUp(self) -> None:
        # An accidental analysis call must fail locally instead of reaching Ollama.
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
                    f"{name} must not run while rendering or changing language"
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

    def _switch_to_chinese(self, app: AppTest) -> None:
        app.selectbox(key="ui_language").set_value("zh").run()
        self.assertFalse(app.exception)

    def test_default_language_is_english_with_five_tab_layout(self) -> None:
        app = self._app()

        language = app.selectbox(key="ui_language")
        self.assertEqual("en", language.value)
        self.assertEqual(["English", "中文"], language.options)
        self.assertEqual("en", app.session_state["ui_language"])
        self.assertEqual(ENGLISH_TABS, _top_level_tab_labels(app, ENGLISH_TABS))

        self.assertIn("Executive Summary", {item.value for item in app.header})
        self.assertTrue(
            {
                "Overall Sentiment",
                "Customer Priority Matrix",
                "Language by Aspect",
                "Customer Evidence",
            }.issubset({item.value for item in app.subheader})
        )
        dataset_mode = app.radio(key="dataset_mode")
        self.assertEqual("Demo Dataset", dataset_mode.value)
        self.assertEqual(
            [
                "Demo Dataset",
                "Analyze New Dataset",
                "Saved Analyses",
                "Compare Products",
            ],
            dataset_mode.options,
        )

    def test_switching_to_chinese_persists_in_session_and_localizes_layout(self) -> None:
        app = self._app()
        self._switch_to_chinese(app)

        language = app.selectbox(key="ui_language")
        self.assertEqual("zh", language.value)
        self.assertEqual("zh", app.session_state["ui_language"])
        self.assertEqual(["English", "中文"], language.options)
        self.assertEqual(CHINESE_TABS, _top_level_tab_labels(app, CHINESE_TABS))
        self.assertTrue(
            {
                "执行摘要",
                "客户语言",
                "维度深度分析",
                "评论浏览器",
            }.issubset({item.value for item in app.header})
        )
        self.assertTrue(
            {
                "整体情绪",
                "客户优先级矩阵",
                "各维度客户语言",
                "客户证据",
            }.issubset({item.value for item in app.subheader})
        )

    def test_chinese_widgets_keep_canonical_internal_selection_values(self) -> None:
        app = self._app()
        # Submit all changes in one event so this also exercises their shared rerun.
        app.selectbox(key="ui_language").set_value("zh")
        app.selectbox(key="demo_selected_aspect").set_value("Remote Control")
        app.selectbox(key="demo_review_sentiment").set_value("negative")
        app.radio(key="demo_review_quick_filter").set_value(
            "Negative Evidence"
        ).run()
        self.assertFalse(app.exception)

        aspect = app.selectbox(key="demo_selected_aspect")
        sentiment = app.selectbox(key="demo_review_sentiment")
        quick_filter = app.radio(key="demo_review_quick_filter")
        dataset_mode = app.radio(key="dataset_mode")

        self.assertEqual("Remote Control", aspect.value)
        self.assertIn("遥控器", aspect.options)
        self.assertNotIn("Remote Control", aspect.options)
        self.assertEqual("negative", sentiment.value)
        self.assertIn("负面", sentiment.options)
        self.assertEqual("Negative Evidence", quick_filter.value)
        self.assertIn("负面证据", quick_filter.options)
        self.assertEqual("Demo Dataset", dataset_mode.value)
        self.assertEqual(
            ["演示数据", "分析新数据", "已保存分析", "竞品对比"],
            dataset_mode.options,
        )
        self.assertEqual("Remote Control", app.session_state["demo_selected_aspect"])
        self.assertEqual("negative", app.session_state["demo_review_sentiment"])

    def test_switching_language_never_starts_analysis_or_ollama(self) -> None:
        app = self._app()
        self._switch_to_chinese(app)

        for mocked in self.analysis_mocks:
            mocked.assert_not_called()

    def test_chinese_view_keeps_raw_reviews_and_evidence_verbatim(self) -> None:
        reviews = pd.read_csv(PROJECT_ROOT / "outputs" / "analyzed_reviews.csv")
        raw_review = str(reviews.loc[0, "original_content"])
        evidence_source = json.loads(
            (PROJECT_ROOT / "outputs" / "aspect_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        raw_evidence = evidence_source["Picture Quality"]["positive"][0][
            "evidence"
        ]

        app = self._app()
        self.assertIn(raw_review, {item.value for item in app.markdown})
        self.assertIn(f"“{raw_evidence}”", {item.value for item in app.markdown})

        self._switch_to_chinese(app)
        self.assertIn(raw_review, {item.value for item in app.markdown})
        self.assertIn(f"“{raw_evidence}”", {item.value for item in app.markdown})

    def test_language_rendering_does_not_modify_frozen_demo_outputs(self) -> None:
        before = _file_hashes()
        self.assertEqual(len(DEMO_OUTPUT_PATHS), len(before))

        app = self._app()
        self._switch_to_chinese(app)

        self.assertEqual(before, _file_hashes())


if __name__ == "__main__":
    unittest.main()
