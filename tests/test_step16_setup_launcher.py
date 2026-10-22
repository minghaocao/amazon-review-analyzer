"""Offline safety tests for Step 16 setup, launcher and living docs."""

from __future__ import annotations

import hashlib
import os
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP = PROJECT_ROOT / "setup_amazon_review_analyzer.command"
LAUNCHER = PROJECT_ROOT / "launcher" / "Amazon Review Analyzer.command"
STOP_LAUNCHER = PROJECT_ROOT / "launcher" / "Stop Amazon Review Analyzer.command"
INSTALLER = PROJECT_ROOT / "Install to Applications.command"
STEP15_ZIP = PROJECT_ROOT / "release" / "Amazon_Review_Analyzer_v1.0.0.zip"
STEP15_ZIP_SHA256 = "8755f81b66fe677bd2babece50b7c69edcb70accf58d4b70256e616653d00d4e"


class SetupAndLauncherTests(unittest.TestCase):
    def test_setup_and_launcher_are_executable_valid_zsh(self) -> None:
        for path in (SETUP, LAUNCHER, STOP_LAUNCHER, INSTALLER):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                self.assertTrue(os.access(path, os.X_OK))
                completed = subprocess.run(
                    ["/bin/zsh", "-n", str(path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(0, completed.returncode, completed.stderr)

    def test_setup_has_bounded_safe_responsibilities(self) -> None:
        text = SETUP.read_text(encoding="utf-8")
        self.assertIn('python3', text)
        self.assertIn('-m venv "$VENV_DIR"', text)
        self.assertIn('-m pip install -r "$REQUIREMENTS_FILE"', text)
        self.assertIn('/api/tags', text)
        self.assertIn('ollama pull %s', text)  # printed guidance only
        self.assertNotIn('ollama pull "$MODEL_NAME"', text)
        self.assertNotIn('amazon.com', text.casefold())
        self.assertNotIn('chat/completions', text)
        self.assertNotIn('/api/generate', text)
        self.assertNotIn('git clean', text)
        self.assertNotIn('rm -rf', text)
        self.assertIn('/bin/cp -n', text)

    def test_launcher_reuses_existing_instance_and_routes_setup_failures(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('mac_launcher.py', text)
        self.assertIn('--foreground', text)
        self.assertIn('--project-root', text)
        self.assertIn('setup_amazon_review_analyzer.command', text)
        self.assertIn('8501', text)
        self.assertNotIn('/Users/', text)
        self.assertNotIn('kill -9', text)
        self.assertNotIn('killall', text)

    def test_stop_and_install_commands_have_bounded_safe_responsibilities(self) -> None:
        stop_text = STOP_LAUNCHER.read_text(encoding="utf-8")
        install_text = INSTALLER.read_text(encoding="utf-8")
        self.assertIn("--stop-dashboard", stop_text)
        self.assertIn("mac_launcher.py", stop_text)
        self.assertIn("--remember-project", install_text)
        self.assertIn("/Applications/Amazon Review Analyzer.app", install_text)
        self.assertIn("/usr/bin/ditto", install_text)
        combined = (stop_text + install_text).casefold()
        for forbidden in (
            "killall",
            "kill -9",
            "sudo ",
            "xattr ",
            "ollama pull",
            "pip install",
            "curl ",
            "wget ",
        ):
            self.assertNotIn(forbidden, combined)

    def test_health_cli_bootstrap_precedes_third_party_imports(self) -> None:
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index('if __name__ == "__main__" and sys.argv[1:]'),
            source.index('import pandas as pd'),
        )
        self.assertIn('["--health-check"]', source)
        self.assertIn('["--health-check-json"]', source)

    def test_living_quick_starts_and_release_notes_exist(self) -> None:
        chinese = (PROJECT_ROOT / "QUICK_START_CN.md").read_text(encoding="utf-8")
        english = (PROJECT_ROOT / "QUICK_START_EN.md").read_text(encoding="utf-8")
        notes = (PROJECT_ROOT / "release_step16_notes.md").read_text(encoding="utf-8")
        self.assertIn("第一次使用", chinese)
        self.assertIn("First-time setup", english)
        self.assertIn("--health-check-json", chinese)
        self.assertIn("--health-check-json", english)
        self.assertIn("v1.0.1", notes)
        self.assertIn("v1.0.0", notes)
        self.assertNotIn("/Users/", chinese + english + notes)

    def test_step15_release_zip_is_byte_identical(self) -> None:
        self.assertEqual(
            STEP15_ZIP_SHA256,
            hashlib.sha256(STEP15_ZIP.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
