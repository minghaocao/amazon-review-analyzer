"""Offline contract tests for the Step 16.5 native macOS wrapper."""

from __future__ import annotations

import json
import plistlib
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from build_macos_app import (
    APP_BUNDLE_NAME,
    APP_NAME,
    APP_VERSION,
    BUNDLE_IDENTIFIER,
    AppBuildError,
    build_app,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MacOSAppBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "Project With Spaces"
        (cls.root / "launcher" / "assets").mkdir(parents=True)
        shutil.copyfile(
            PROJECT_ROOT / "launcher" / "app_bootstrap.zsh",
            cls.root / "launcher" / "app_bootstrap.zsh",
        )
        shutil.copyfile(
            PROJECT_ROOT / "launcher" / "assets" / "app_icon.svg",
            cls.root / "launcher" / "assets" / "app_icon.svg",
        )
        (cls.root / "mac_launcher.py").write_text(
            "# bundled test helper\n",
            encoding="utf-8",
        )
        cls.bundle = build_app(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_bundle_path_and_small_wrapper_contents(self) -> None:
        self.assertEqual(APP_BUNDLE_NAME, self.bundle.name)
        contents = self.bundle / "Contents"
        self.assertTrue((contents / "Info.plist").is_file())
        self.assertTrue((contents / "MacOS" / APP_NAME).is_file())
        self.assertTrue((contents / "Resources" / "mac_launcher.py").is_file())
        self.assertTrue((contents / "Resources" / "AppIcon.icns").is_file())
        names = {path.name.casefold() for path in self.bundle.rglob("*")}
        self.assertNotIn(".venv", names)
        self.assertFalse(any("qwen3" in name or "model" == name for name in names))
        self.assertLess(
            sum(path.stat().st_size for path in self.bundle.rglob("*") if path.is_file()),
            5 * 1024 * 1024,
        )

    def test_info_plist_has_truthful_version_and_native_wrapper_contract(self) -> None:
        with (self.bundle / "Contents" / "Info.plist").open("rb") as handle:
            info = plistlib.load(handle)
        self.assertEqual(BUNDLE_IDENTIFIER, info["CFBundleIdentifier"])
        self.assertEqual(APP_VERSION, info["CFBundleShortVersionString"])
        self.assertEqual("101", info["CFBundleVersion"])
        self.assertEqual(APP_NAME, info["CFBundleExecutable"])
        self.assertEqual("APPL", info["CFBundlePackageType"])
        self.assertIs(True, info["LSUIElement"])
        completed = subprocess.run(
            ["/usr/bin/plutil", "-lint", str(self.bundle / "Contents" / "Info.plist")],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_executable_is_valid_zsh_and_delegates_to_shared_helper(self) -> None:
        executable = self.bundle / "Contents" / "MacOS" / APP_NAME
        self.assertTrue(executable.stat().st_mode & stat.S_IXUSR)
        completed = subprocess.run(
            ["/bin/zsh", "-n", str(executable)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        text = executable.read_text(encoding="utf-8")
        self.assertIn('"$LAUNCHER_HELPER"', text)
        self.assertIn("--native", text)
        self.assertIn("--app-bundle", text)
        self.assertNotIn("eval ", text)

    def test_bundle_uses_relative_locator_without_personal_path(self) -> None:
        locator_path = self.bundle / "Contents" / "Resources" / "project_locator.json"
        locator = json.loads(locator_path.read_text(encoding="utf-8"))
        self.assertEqual("../..", locator["relative_project_path"])
        for path in self.bundle.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"/Users/", path.read_bytes())

    def test_bootstrap_and_builder_contain_no_install_or_security_bypass(self) -> None:
        sources = "\n".join(
            (
                (PROJECT_ROOT / "build_macos_app.py").read_text(encoding="utf-8"),
                (PROJECT_ROOT / "launcher" / "app_bootstrap.zsh").read_text(
                    encoding="utf-8"
                ),
            )
        ).casefold()
        for forbidden in (
            "sudo ",
            "xattr ",
            "pip install",
            "ollama pull",
            "killall",
            "curl ",
            "wget ",
            "verify=false",
            "cert_none",
        ):
            self.assertNotIn(forbidden, sources)

    def test_icon_is_local_icns_and_not_an_external_asset(self) -> None:
        icon = self.bundle / "Contents" / "Resources" / "AppIcon.icns"
        self.assertEqual(b"icns", icon.read_bytes()[:4])
        source = (PROJECT_ROOT / "launcher" / "assets" / "app_icon.svg").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("href=", source.casefold())
        self.assertNotIn("amazon logo", source.casefold())
        self.assertNotIn("openai", source.casefold())

    def test_rebuild_replaces_only_the_scoped_generated_app(self) -> None:
        marker = self.bundle / "Contents" / "obsolete.txt"
        marker.write_text("obsolete", encoding="utf-8")
        rebuilt = build_app(self.root)
        self.assertEqual(self.bundle, rebuilt)
        self.assertFalse(marker.exists())

    def test_missing_helper_fails_without_creating_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "launcher" / "assets").mkdir(parents=True)
            shutil.copyfile(
                PROJECT_ROOT / "launcher" / "app_bootstrap.zsh",
                root / "launcher" / "app_bootstrap.zsh",
            )
            shutil.copyfile(
                PROJECT_ROOT / "launcher" / "assets" / "app_icon.svg",
                root / "launcher" / "assets" / "app_icon.svg",
            )
            with self.assertRaises(AppBuildError):
                build_app(root)
            self.assertFalse((root / "dist" / APP_BUNDLE_NAME).exists())


if __name__ == "__main__":
    unittest.main()

