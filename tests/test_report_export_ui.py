"""Offline Streamlit acceptance tests for Step 14 report export controls."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
import report_generator  # noqa: E402


def _document(report_id: str = "RPT_20261016_010000_ab12") -> SimpleNamespace:
    return SimpleNamespace(
        report_id=report_id,
        report_type="single_product",
        language="en",
        html="<!doctype html><html><body>offline report</body></html>",
        metadata={"report_id": report_id},
    )


def _saved(root: Path, analysis_id: str, name: str) -> library.SavedAnalysis:
    return library.SavedAnalysis(
        root,
        {
            "analysis_id": analysis_id,
            "product_name": name,
            "analysis_run_id": f"RUN_{analysis_id}",
            "source_type": "amazon",
            "imported_reviews": 8,
            "successful_reviews": 8,
            "coverage": 1.0,
            "saved_at": "2026-10-16T01:00:00Z",
            "analysis_candidate": "generalization_v2_candidate",
            "taxonomy_version": "generalization_v2_candidate",
            "schema_version": library.ANALYSIS_LIBRARY_SCHEMA_VERSION,
        },
    )


class SingleAndSavedReportUITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.saved = _saved(
            Path(self.temporary.name) / "saved",
            "AN_20261016_010000_ab12",
            "Saved Interview TV",
        )
        list_patcher = patch.object(
            dashboard_app.library,
            "list_saved_analyses",
            return_value=[self.saved],
        )
        list_patcher.start()
        self.addCleanup(list_patcher.stop)
        for owner, name in (
            (dashboard_app.interactive, "create_checked_ollama_client"),
            (dashboard_app.interactive, "run_interactive_pipeline"),
            (dashboard_app.amazon, "run_amazon_pipeline"),
        ):
            patcher = patch.object(
                owner,
                name,
                side_effect=AssertionError(f"{name} must not run while exporting"),
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def _app(self) -> AppTest:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_single_dashboard_generates_from_completed_outputs_only(self) -> None:
        document = _document()
        generator = patch.object(
            report_generator,
            "build_single_product_report",
            return_value=document,
        )
        generate_mock = generator.start()
        self.addCleanup(generator.stop)
        exporter = patch.object(
            report_generator,
            "export_report",
            return_value=SimpleNamespace(root=Path("reports")),
        )
        exporter.start()
        self.addCleanup(exporter.stop)
        app = self._app()

        app.button(key="demo_generate_report").click().run()

        self.assertFalse(app.exception)
        generate_mock.assert_called_once()
        self.assertEqual("en", generate_mock.call_args.kwargs["language"])
        self.assertFalse(generate_mock.call_args.kwargs["validate_integrity"])
        self.assertIn("Report generated successfully.", {item.value for item in app.success})
        self.assertEqual(document, app.session_state["demo_report_document"])

    def test_saved_analysis_generates_directly_with_integrity_guard(self) -> None:
        document = _document("RPT_20261016_010100_cd34")
        generator = patch.object(
            report_generator,
            "build_single_product_report",
            return_value=document,
        )
        generate_mock = generator.start()
        self.addCleanup(generator.stop)
        exporter = patch.object(
            report_generator,
            "export_report",
            return_value=SimpleNamespace(root=Path("reports")),
        )
        exporter.start()
        self.addCleanup(exporter.stop)
        app = self._app()
        app.radio(key="dataset_mode").set_value("Saved Analyses").run()
        button_key = "saved_library_an_20261016_010000_ab12_generate_report"

        app.button(key=button_key).click().run()

        self.assertFalse(app.exception)
        generate_mock.assert_called_once_with(
            self.saved.root,
            language="en",
            product_name="Saved Interview TV",
            validate_integrity=True,
        )
        self.assertIn(
            "Report generated for Saved Interview TV.",
            {item.value for item in app.success},
        )

    def test_saved_report_remains_available_in_limited_mode(self) -> None:
        """Report export must not depend on an available inference provider."""

        document = _document("RPT_20261016_010150_lm01")
        with (
            patch.dict(
                os.environ,
                {"ARA_UI_HEALTH_TEST_PROFILE": "limited"},
                clear=False,
            ),
            patch.object(
                report_generator,
                "build_single_product_report",
                return_value=document,
            ) as generate_mock,
            patch.object(
                report_generator,
                "export_report",
                return_value=SimpleNamespace(root=Path("reports")),
            ),
        ):
            app = self._app()
            app.radio(key="dataset_mode").set_value("Saved Analyses").run()
            app.button(
                key="saved_library_an_20261016_010000_ab12_generate_report"
            ).click().run()

        self.assertFalse(app.exception)
        generate_mock.assert_called_once_with(
            self.saved.root,
            language="en",
            product_name="Saved Interview TV",
            validate_integrity=True,
        )
        self.assertIn(
            "Report generated for Saved Interview TV.",
            {item.value for item in app.success},
        )

    def test_report_language_defaults_to_current_ui_and_is_independent(self) -> None:
        app = self._app()
        self.assertEqual("en", app.selectbox(key="demo_report_language_en").value)
        app.selectbox(key="demo_report_language_en").set_value("zh").run()
        self.assertEqual("zh", app.selectbox(key="demo_report_language_en").value)
        self.assertEqual("en", app.session_state[dashboard_app.LANGUAGE_STATE_KEY])

    def test_saved_integrity_failure_uses_localized_message(self) -> None:
        error = report_generator.ReportIntegrityError(
            "Saved analysis integrity check failed."
        )
        with patch.object(dashboard_app, "_language", return_value="zh"):
            self.assertEqual(
                "已保存分析的完整性校验失败。",
                dashboard_app._report_error_message(error),
            )


class ComparisonReportUITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.output_root = root / "comparison"
        self.source_a = root / "a"
        self.source_b = root / "b"

    def _app(self) -> AppTest:
        script = f"""
from pathlib import Path
from comparison_ui import ComparisonSource, _render_comparison_report_controls
source_a = ComparisonSource('a', 'a', 'A', 'saved', outputs_dir=Path({str(self.source_a)!r}), saved_analysis_id='AN_A')
source_b = ComparisonSource('b', 'b', 'B', 'saved', outputs_dir=Path({str(self.source_b)!r}), saved_analysis_id='AN_B')
_render_comparison_report_controls(output_root=Path({str(self.output_root)!r}), source_a=source_a, source_b=source_b, language='en')
"""
        app = AppTest.from_string(script, default_timeout=30)
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_comparison_export_uses_two_saved_snapshots(self) -> None:
        document = _document("CMP_RPT_20261016_010200_ef56")
        generator = patch.object(
            comparison_ui,
            "_generate_comparison_report",
            return_value=(document, SimpleNamespace(root=Path("reports"))),
        )
        generate_mock = generator.start()
        self.addCleanup(generator.stop)
        app = self._app()

        app.button(key="competitor_comparison_generate_report").click().run()

        self.assertFalse(app.exception)
        generate_mock.assert_called_once_with(
            self.output_root,
            product_a_root=self.source_a,
            product_b_root=self.source_b,
            language="en",
        )
        self.assertIn(
            "Comparison report generated successfully.",
            {item.value for item in app.success},
        )

    def test_non_saved_sources_cannot_export_without_integrity_roots(self) -> None:
        script = f"""
from pathlib import Path
from comparison_ui import ComparisonSource, _render_comparison_report_controls
source_a = ComparisonSource('a', 'a', 'A', 'fixture', fixture_key='fixture_a')
source_b = ComparisonSource('b', 'b', 'B', 'fixture', fixture_key='fixture_b')
_render_comparison_report_controls(output_root=Path({str(self.output_root)!r}), source_a=source_a, source_b=source_b, language='en')
"""
        app = AppTest.from_string(script, default_timeout=30)
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button(key="competitor_comparison_generate_report").disabled)
        self.assertIn(
            "Save both product analyses before exporting so their snapshot integrity can be verified.",
            {item.value for item in app.info},
        )


if __name__ == "__main__":
    unittest.main()
