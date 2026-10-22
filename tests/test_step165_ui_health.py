"""Offline checks for the Step 16.5 runtime-port and Limited Mode UI."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import environment_health as health
from first_run import mark_first_run_completed
from i18n import TRANSLATIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RuntimePortSettingsTests(unittest.TestCase):
    def test_valid_native_launcher_port_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = health.discover_health_settings(
                directory,
                environment={"ARA_STREAMLIT_PORT": "8504"},
            )

        self.assertEqual(8504, settings.streamlit_port)

    def test_invalid_native_launcher_ports_fall_back_safely(self) -> None:
        invalid_values = ("", "zero", "0", "65536", "-1", "8501;open bad")
        with tempfile.TemporaryDirectory() as directory:
            for value in invalid_values:
                with self.subTest(value=value):
                    settings = health.discover_health_settings(
                        directory,
                        environment={"ARA_STREAMLIT_PORT": value},
                    )
                    self.assertEqual(
                        health.DEFAULT_STREAMLIT_PORT,
                        settings.streamlit_port,
                    )


class LimitedModeBannerTests(unittest.TestCase):
    def _completed_app(self) -> AppTest:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_path = Path(temporary.name) / "first_run.json"
        mark_first_run_completed("1.0.0", state_path)
        environment = {
            "ARA_FIRST_RUN_STATE_PATH": str(state_path),
            "ARA_UI_HEALTH_TEST_PROFILE": "limited",
        }
        patcher = patch.dict(os.environ, environment, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        application = AppTest.from_file(
            str(PROJECT_ROOT / "app.py"),
            default_timeout=30,
        )
        application.run()
        self.assertFalse(application.exception)
        return application

    def test_limited_mode_has_exact_english_viewing_message(self) -> None:
        application = self._completed_app()

        self.assertIn(
            "Local AI unavailable — viewing features remain available.",
            {item.value for item in application.caption},
        )

    def test_limited_mode_has_exact_chinese_viewing_message(self) -> None:
        application = self._completed_app()
        application.selectbox(key="ui_language").set_value("zh").run()

        self.assertFalse(application.exception)
        self.assertIn(
            "本地 AI 当前不可用 — 已有结果与查看功能仍可使用。",
            {item.value for item in application.caption},
        )

    def test_localizations_remain_in_full_parity(self) -> None:
        self.assertEqual(
            "Local AI unavailable — viewing features remain available.",
            TRANSLATIONS["en"]["health.limited_viewing"],
        )
        self.assertEqual(
            "本地 AI 当前不可用 — 已有结果与查看功能仍可使用。",
            TRANSLATIONS["zh"]["health.limited_viewing"],
        )
        self.assertEqual(set(TRANSLATIONS["en"]), set(TRANSLATIONS["zh"]))


if __name__ == "__main__":
    unittest.main()
