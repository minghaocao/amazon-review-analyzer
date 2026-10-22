"""Offline acceptance contracts for the Step 16.6 v1.0.1 release package."""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath

import config
from build_release_v1_0_1 import (
    BUNDLE_BUILD_VERSION,
    BUNDLE_ID,
    BUNDLE_VERSION,
    COPY_SPECS,
    EXTERNAL_MANIFEST_NAME,
    INTERNAL_MANIFEST_NAME,
    MANIFEST_SCHEMA,
    PACKAGE_NAME,
    VERSION,
    ZIP_NAME,
    create_deterministic_zip,
    scan_release_tree,
    sha256_file,
    tree_hashes,
    validate_release_zip,
    verify_checksum_file,
)
from release_health_check import REQUIRED_MANIFEST_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT / "release"
PACKAGE_ROOT = RELEASE_ROOT / PACKAGE_NAME
ZIP_PATH = RELEASE_ROOT / ZIP_NAME
EXTERNAL_MANIFEST_PATH = PROJECT_ROOT / EXTERNAL_MANIFEST_NAME
LEGACY_RELEASE_ROOT = RELEASE_ROOT / "Amazon_Review_Analyzer_v1.0.0"
LEGACY_ZIP_PATH = RELEASE_ROOT / "Amazon_Review_Analyzer_v1.0.0.zip"
LEGACY_ZIP_SHA256 = "8755f81b66fe677bd2babece50b7c69edcb70accf58d4b70256e616653d00d4e"
LEGACY_MANIFEST_SHA256 = "95d2764ee580fb2fdadda907d7404ceb87ac075c88808faa322fc58c78cb4778"


def _manifest() -> dict[str, object]:
    return json.loads(EXTERNAL_MANIFEST_PATH.read_text(encoding="utf-8"))


def _zip_regular_members() -> set[str]:
    with zipfile.ZipFile(ZIP_PATH) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}


class Step166ReleasePackageTests(unittest.TestCase):
    """The package is a verified companion distribution, never a source dump."""

    def test_01_current_product_version_is_1_0_1(self) -> None:
        self.assertEqual("1.0.1", VERSION)
        self.assertEqual(VERSION, config.APP_VERSION)

    def test_02_historical_root_manifest_is_unchanged(self) -> None:
        self.assertEqual(
            LEGACY_MANIFEST_SHA256,
            sha256_file(PROJECT_ROOT / "release_manifest.json"),
        )

    def test_03_historical_v1_zip_is_unchanged(self) -> None:
        self.assertEqual(LEGACY_ZIP_SHA256, sha256_file(LEGACY_ZIP_PATH))

    def test_04_historical_release_still_identifies_v1_0_0(self) -> None:
        legacy = json.loads(
            (LEGACY_RELEASE_ROOT / "release_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual("1.0.0", legacy["version"])

    def test_05_app_bundle_display_version_is_1_0_1(self) -> None:
        with (
            PACKAGE_ROOT / "Amazon Review Analyzer.app/Contents/Info.plist"
        ).open("rb") as handle:
            info = plistlib.load(handle)
        self.assertEqual(BUNDLE_VERSION, info["CFBundleShortVersionString"])

    def test_06_app_bundle_build_version_is_101(self) -> None:
        with (
            PACKAGE_ROOT / "Amazon Review Analyzer.app/Contents/Info.plist"
        ).open("rb") as handle:
            info = plistlib.load(handle)
        self.assertEqual(BUNDLE_BUILD_VERSION, info["CFBundleVersion"])

    def test_07_bundle_identifier_is_unchanged(self) -> None:
        with (
            PACKAGE_ROOT / "Amazon Review Analyzer.app/Contents/Info.plist"
        ).open("rb") as handle:
            info = plistlib.load(handle)
        self.assertEqual(BUNDLE_ID, info["CFBundleIdentifier"])

    def test_08_readme_presents_current_stable_local_release(self) -> None:
        text = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Amazon Review Analyzer v1.0.1", text)
        self.assertIn("Stable Local Release", text)

    def test_09_release_notes_identify_maintenance_scope(self) -> None:
        text = (PACKAGE_ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8")
        self.assertIn("v1.0.1", text)
        self.assertIn("Maintenance / UX / Distribution", text)

    def test_10_changelog_contains_v1_0_1(self) -> None:
        text = (PACKAGE_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("## v1.0.1", text)
        self.assertIn("### Unchanged", text)

    def test_11_external_manifest_uses_step16_6_schema(self) -> None:
        self.assertEqual(MANIFEST_SCHEMA, _manifest()["manifest_schema_version"])

    def test_12_external_manifest_has_every_required_field(self) -> None:
        self.assertTrue(REQUIRED_MANIFEST_FIELDS.issubset(_manifest()))

    def test_13_external_manifest_records_zero_new_calls(self) -> None:
        manifest = _manifest()
        self.assertEqual(0, manifest["new_llm_calls"])
        self.assertEqual(0, manifest["amazon_network_calls"])

    def test_14_external_manifest_zip_hash_matches_release(self) -> None:
        self.assertEqual(sha256_file(ZIP_PATH), _manifest()["release_package_sha256"])

    def test_15_zip_integrity_check_passes(self) -> None:
        with zipfile.ZipFile(ZIP_PATH) as archive:
            self.assertIsNone(archive.testzip())

    def test_16_zip_member_paths_are_safe_and_relative(self) -> None:
        with zipfile.ZipFile(ZIP_PATH) as archive:
            for name in archive.namelist():
                path = PurePosixPath(name)
                self.assertFalse(path.is_absolute(), name)
                self.assertNotIn("..", path.parts, name)
                self.assertEqual(PACKAGE_NAME, path.parts[0])

    def test_17_release_tree_security_scan_is_clean(self) -> None:
        self.assertEqual([], scan_release_tree(PACKAGE_ROOT))

    def test_18_package_contains_no_env_file(self) -> None:
        names = {part.casefold() for path in PACKAGE_ROOT.rglob("*") for part in path.parts}
        self.assertNotIn(".env", names)

    def test_19_package_contains_no_virtual_environment(self) -> None:
        relative_parts = {
            part.casefold()
            for path in PACKAGE_ROOT.rglob("*")
            for part in path.relative_to(PACKAGE_ROOT).parts
        }
        self.assertNotIn(".venv", relative_parts)

    def test_20_package_contains_no_model_payload(self) -> None:
        forbidden = {".gguf", ".safetensors", ".onnx", ".pt", ".pth"}
        self.assertFalse(
            [path for path in PACKAGE_ROOT.rglob("*") if path.suffix.casefold() in forbidden]
        )

    def test_21_package_contains_no_cookie_or_session_artifact(self) -> None:
        suspicious = [
            path
            for path in PACKAGE_ROOT.rglob("*")
            if any(token in path.name.casefold() for token in ("cookie", "session"))
        ]
        self.assertEqual([], suspicious)

    def test_22_required_documentation_is_included(self) -> None:
        expected = {
            "README.md",
            "RELEASE_NOTES.md",
            "CHANGELOG.md",
            "SYSTEM_OVERVIEW.md",
            "PORTFOLIO_SUMMARY.md",
            "docs/Amazon_Review_Analyzer_v1.0.1_User_Guide_EN.md",
            "docs/Amazon_Review_Analyzer_v1.0.1_User_Guide_CN.md",
        }
        self.assertTrue(all((PACKAGE_ROOT / name).is_file() for name in expected))

    def test_23_bilingual_quick_starts_are_included(self) -> None:
        self.assertTrue((PACKAGE_ROOT / "QUICK_START_EN.md").is_file())
        self.assertTrue((PACKAGE_ROOT / "QUICK_START_CN.md").is_file())

    def test_24_app_bundle_is_included(self) -> None:
        executable = (
            PACKAGE_ROOT
            / "Amazon Review Analyzer.app/Contents/MacOS/Amazon Review Analyzer"
        )
        self.assertTrue(executable.is_file())
        self.assertTrue(os.access(executable, os.X_OK))

    def test_25_install_helper_is_included_and_executable(self) -> None:
        helper = PACKAGE_ROOT / "Install to Applications.command"
        self.assertTrue(helper.is_file())
        self.assertTrue(helper.stat().st_mode & stat.S_IXUSR)

    def test_26_safe_stop_helper_is_included_and_executable(self) -> None:
        helper = PACKAGE_ROOT / "Stop Amazon Review Analyzer.command"
        self.assertTrue(helper.is_file())
        self.assertTrue(helper.stat().st_mode & stat.S_IXUSR)

    def test_27_setup_helper_is_included_and_executable(self) -> None:
        helper = PACKAGE_ROOT / "setup_amazon_review_analyzer.command"
        self.assertTrue(helper.is_file())
        self.assertTrue(helper.stat().st_mode & stat.S_IXUSR)

    def test_28_sha256sums_covers_and_verifies_every_payload_file(self) -> None:
        verify_checksum_file(PACKAGE_ROOT)

    def test_29_zip_inventory_and_hashes_match_release_folder(self) -> None:
        validate_release_zip(ZIP_PATH, PACKAGE_ROOT)

    def test_30_internal_manifest_handles_zip_self_hash_honestly(self) -> None:
        internal = json.loads(
            (PACKAGE_ROOT / INTERNAL_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        self.assertIsNone(internal["release_package_sha256"])
        self.assertIn("self-hash", internal["release_package_sha256_scope"])

    def test_31_external_manifest_file_hashes_match_release_folder(self) -> None:
        self.assertEqual(tree_hashes(PACKAGE_ROOT), _manifest()["file_hashes"])

    def test_32_zip_file_inventory_matches_release_folder(self) -> None:
        expected = {
            f"{PACKAGE_NAME}/{relative}" for relative in tree_hashes(PACKAGE_ROOT)
        }
        self.assertEqual(expected, _zip_regular_members())

    def test_33_approved_sample_assets_are_copied_byte_identically(self) -> None:
        for name in ("screenshots", "sample_reports", "sample_data"):
            with self.subTest(asset=name):
                self.assertEqual(
                    tree_hashes(LEGACY_RELEASE_ROOT / name),
                    tree_hashes(PACKAGE_ROOT / name),
                )

    def test_34_documentation_discloses_local_runtime_limits(self) -> None:
        text = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8").casefold()
        self.assertIn("lightweight launcher", text)
        self.assertIn("not a fully self-contained", text)
        self.assertIn("notarize", text)
        self.assertIn("qwen3:8b", text)

    def test_35_documentation_discloses_amazon_access_limits(self) -> None:
        text = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8").casefold()
        self.assertIn("captcha", text)
        self.assertIn("paste reviews", text)
        self.assertIn("upload dataset", text)

    def test_36_blind_v2_metrics_are_scoped_not_marketed_as_universal(self) -> None:
        text = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("85.66%", text)
        self.assertIn(
            "These metrics apply only to the frozen Blind V2 evaluation set",
            text,
        )

    def test_37_release_copy_sources_are_bounded_and_never_runtime_outputs(self) -> None:
        sources = {spec.source for spec in COPY_SPECS}
        self.assertFalse(any(name.startswith("outputs/") for name in sources))
        self.assertFalse(any(name.startswith("evaluation/") for name in sources))
        self.assertFalse(any(".venv" in name or ".env" in name for name in sources))

    def test_38_zip_is_reproducible_from_the_final_release_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rebuilt = Path(temporary) / ZIP_NAME
            create_deterministic_zip(PACKAGE_ROOT, rebuilt)
            self.assertEqual(sha256_file(ZIP_PATH), sha256_file(rebuilt))


if __name__ == "__main__":
    unittest.main()
