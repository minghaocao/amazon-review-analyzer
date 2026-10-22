"""Offline Streamlit tests for Step 13.1 Saved Analyses integration."""

from __future__ import annotations

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

import analysis_library as library  # noqa: E402
import app as dashboard_app  # noqa: E402
import comparison_ui  # noqa: E402


def _saved(analysis_id: str, name: str, run_id: str) -> library.SavedAnalysis:
    manifest = {
        "analysis_id": analysis_id,
        "product_name": name,
        "analysis_run_id": run_id,
        "source_type": "amazon",
        "imported_reviews": 12,
        "successful_reviews": 11,
        "coverage": 11 / 12,
        "saved_at": "2026-10-15T10:00:00Z",
        "analysis_candidate": "generalization_v2_candidate",
        "taxonomy_version": "generalization_v2_candidate",
        "schema_version": library.ANALYSIS_LIBRARY_SCHEMA_VERSION,
    }
    return library.SavedAnalysis(
        PROJECT_ROOT / "outputs" / "library" / analysis_id.casefold(),
        manifest,
    )


class SavedAnalysisAppTests(unittest.TestCase):
    """Rendering and persistence controls must stay completely offline."""

    def setUp(self) -> None:
        self.saved_a = _saved("AN_20261015_100000_a1b2", "Roku TV", "RUN_A")
        self.saved_b = _saved("AN_20261015_110000_c3d4", "Samsung TV", "RUN_B")
        list_patcher = patch.object(
            dashboard_app.library,
            "list_saved_analyses",
            return_value=[self.saved_b, self.saved_a],
        )
        list_patcher.start()
        self.addCleanup(list_patcher.stop)

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
                side_effect=AssertionError(f"{name} must not run in Saved Analyses UI"),
            )
            self.analysis_mocks.append(patcher.start())
            self.addCleanup(patcher.stop)

    def _app(self) -> AppTest:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_save_analysis_button_snapshots_outputs_without_pipeline(self) -> None:
        save_patcher = patch.object(
            dashboard_app.library,
            "save_analysis",
            return_value=self.saved_a,
        )
        save_mock = save_patcher.start()
        self.addCleanup(save_patcher.stop)
        app = self._app()

        app.text_input(key="demo_saved_analysis_product_name").set_value("Interview TV")
        app.button(key="demo_save_analysis").click().run()

        self.assertFalse(app.exception)
        save_mock.assert_called_once()
        self.assertEqual("Interview TV", save_mock.call_args.args[1])
        self.assertTrue(save_mock.call_args.kwargs["trusted_current_outputs"])
        for mocked in self.analysis_mocks:
            mocked.assert_not_called()

    def test_duplicate_save_uses_friendly_message(self) -> None:
        duplicate = library.DuplicateAnalysisError("RUN_A", self.saved_a.analysis_id)
        save_patcher = patch.object(
            dashboard_app.library,
            "save_analysis",
            side_effect=duplicate,
        )
        save_patcher.start()
        self.addCleanup(save_patcher.stop)
        app = self._app()
        app.button(key="demo_save_analysis").click().run()

        self.assertFalse(app.exception)
        self.assertIn(
            "This analysis has already been saved.",
            {item.value for item in app.info},
        )

    def test_saved_analyses_page_is_localized_and_lists_snapshots(self) -> None:
        app = self._app()
        app.radio(key="dataset_mode").set_value("Saved Analyses").run()
        self.assertFalse(app.exception)
        self.assertIn("Saved Analyses", {item.value for item in app.title})
        frame = next(item.value for item in app.dataframe if "Analysis ID" in item.value)
        self.assertEqual(["Samsung TV", "Roku TV"], frame["Product Name"].tolist())

        app.selectbox(key="ui_language").set_value("zh").run()
        self.assertFalse(app.exception)
        self.assertIn("已保存分析", {item.value for item in app.title})
        localized = next(item.value for item in app.dataframe if "分析 ID" in item.value)
        self.assertEqual(["Samsung TV", "Roku TV"], localized["产品名称"].tolist())

    def test_saved_a_b_selection_survives_language_change(self) -> None:
        app = self._app()
        app.radio(key="dataset_mode").set_value("Compare Products").run()
        self.assertFalse(app.exception)
        expected = (
            f"saved:{self.saved_b.analysis_id}",
            f"saved:{self.saved_a.analysis_id}",
        )
        self.assertEqual(
            expected,
            (
                app.session_state[comparison_ui.SOURCE_A_STATE_KEY],
                app.session_state[comparison_ui.SOURCE_B_STATE_KEY],
            ),
        )

        app.selectbox(key="ui_language").set_value("zh").run()
        self.assertEqual(
            expected,
            (
                app.session_state[comparison_ui.SOURCE_A_STATE_KEY],
                app.session_state[comparison_ui.SOURCE_B_STATE_KEY],
            ),
        )
        for mocked in self.analysis_mocks:
            mocked.assert_not_called()

    def test_saved_a_b_selection_survives_dataset_mode_return(self) -> None:
        app = self._app()
        app.radio(key="dataset_mode").set_value("Compare Products").run()
        expected = (
            app.session_state[comparison_ui.SOURCE_A_STATE_KEY],
            app.session_state[comparison_ui.SOURCE_B_STATE_KEY],
        )
        expected_names = (
            app.session_state[comparison_ui.PRODUCT_NAME_A_STATE_KEY],
            app.session_state[comparison_ui.PRODUCT_NAME_B_STATE_KEY],
        )
        app.radio(key="dataset_mode").set_value("Demo Dataset").run()
        app.radio(key="dataset_mode").set_value("Compare Products").run()
        self.assertFalse(app.exception)
        self.assertEqual(
            expected,
            (
                app.session_state[comparison_ui.SOURCE_A_STATE_KEY],
                app.session_state[comparison_ui.SOURCE_B_STATE_KEY],
            ),
        )
        self.assertEqual(
            expected_names,
            (
                app.session_state[comparison_ui.PRODUCT_NAME_A_STATE_KEY],
                app.session_state[comparison_ui.PRODUCT_NAME_B_STATE_KEY],
            ),
        )
        self.assertEqual(
            expected_names,
            (
                app.text_input(
                    key=f"{comparison_ui.PRODUCT_NAME_A_STATE_KEY}_widget"
                ).value,
                app.text_input(
                    key=f"{comparison_ui.PRODUCT_NAME_B_STATE_KEY}_widget"
                ).value,
            ),
        )
        for mocked in self.analysis_mocks:
            mocked.assert_not_called()

    def test_same_saved_analysis_is_rejected_before_comparison(self) -> None:
        app = self._app()
        app.radio(key="dataset_mode").set_value("Compare Products").run()
        source_a = app.session_state[comparison_ui.SOURCE_A_STATE_KEY]
        app.selectbox(
            key=f"{comparison_ui.SOURCE_B_STATE_KEY}_widget_en"
        ).set_value(source_a).run()

        self.assertFalse(app.exception)
        self.assertTrue(app.button(key="competitor_comparison_run").disabled)
        self.assertIn(
            "Please select two different saved analyses.",
            {item.value for item in app.warning},
        )


if __name__ == "__main__":
    unittest.main()
