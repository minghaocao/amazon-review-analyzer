"""Focused offline UI acceptance checks for the v1.0 release surface."""

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

import app as dashboard_app  # noqa: E402
from config import APP_VERSION  # noqa: E402


class ReleaseUIAcceptanceTests(unittest.TestCase):
    def test_version_is_config_sourced_and_localized_in_sidebar(self) -> None:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        app.run()
        self.assertFalse(app.exception)
        expected = f"Amazon Review Analyzer v{APP_VERSION}"
        self.assertIn(expected, {item.value for item in app.caption})

        app.selectbox(key="ui_language").set_value("zh").run()
        self.assertFalse(app.exception)
        self.assertIn(expected, {item.value for item in app.caption})

    def test_user_facing_failure_messages_do_not_echo_exception_text(self) -> None:
        marker = "Traceback: /private/internal/path"
        message_en = dashboard_app._report_error_message(RuntimeError(marker))
        with patch.object(dashboard_app, "_language", return_value="zh"):
            message_zh = dashboard_app._label(
                "error.operation_failed",
                "The operation could not be completed. Please try again.",
            )
        self.assertNotIn(marker, message_en)
        self.assertNotIn(marker, message_zh)
        self.assertEqual(
            "The report could not be generated. Please try again.",
            message_en,
        )

        with patch.object(dashboard_app, "_language", return_value="zh"):
            self.assertEqual(
                "无法生成报告，请重试。",
                dashboard_app._report_error_message(RuntimeError(marker)),
            )


if __name__ == "__main__":
    unittest.main()
