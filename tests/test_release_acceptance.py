"""Cross-module v1.0.1 release contracts that must remain fully offline."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import main as main_module  # noqa: E402
from analysis_library import ANALYSIS_LIBRARY_SCHEMA_VERSION  # noqa: E402
from competitor_comparison import COMPARISON_SCHEMA_VERSION  # noqa: E402
from release_health_check import (  # noqa: E402
    REQUIRED_MANIFEST_FIELDS,
    REQUIRED_RELEASE_FILES,
    HealthCheckResult,
    HealthStatus,
    ReleaseHealthReport,
    format_health_report,
    load_release_manifest,
    run_release_health_check,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CoreReleaseAcceptanceTests(unittest.TestCase):
    def _manifest(self, frozen_path: str, frozen_hash: str) -> dict[str, object]:
        return {
            "manifest_schema_version": "step16_6.release.v1",
            "product_name": "Amazon Review Analyzer",
            "version": config.APP_VERSION,
            "release_type": "Maintenance / UX / Distribution Release",
            "created_at": "2026-10-22T12:00:00+08:00",
            "provider": config.DEFAULT_MODEL_PROVIDER,
            "model": config.DEFAULT_MODEL_NAME,
            "runtime_mode": config.DEFAULT_RUNTIME_MODE,
            "bundle_id": "com.caominghao.amazonreviewanalyzer",
            "app_bundle_version": config.APP_VERSION,
            "release_package_name": "Amazon-Review-Analyzer-v1.0.1.zip",
            "release_package_sha256": "0" * 64,
            "frozen_analysis_candidate": config.ANALYSIS_CANDIDATE_VERSION,
            "frozen_taxonomy_version": config.TAXONOMY_VERSION,
            "blind_v2_reference": "outputs/evaluation/blind_v2/blind_v2_evaluation.json",
            "automated_test_count": 1,
            "automated_test_passed": 1,
            "health_check_status": "PASS",
            "new_llm_calls": 0,
            "amazon_network_calls": 0,
            "frozen_assets_verified": True,
            "included_artifacts": ["README.md"],
            "file_hashes": {"README.md": "0" * 64},
            "app_version": config.APP_VERSION,
            "release_name": f"Amazon Review Analyzer v{config.APP_VERSION}",
            "released_at": "2026-10-16T12:00:00+08:00",
            "model_provider": config.DEFAULT_MODEL_PROVIDER,
            "model_name": config.DEFAULT_MODEL_NAME,
            "analysis_candidate": config.ANALYSIS_CANDIDATE_VERSION,
            "taxonomy_version": config.TAXONOMY_VERSION,
            "report_template_version": config.REPORT_TEMPLATE_VERSION,
            "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
            "saved_analysis_schema_version": ANALYSIS_LIBRARY_SCHEMA_VERSION,
            "default_runtime_mode": config.DEFAULT_RUNTIME_MODE,
            "default_workers": config.DEFAULT_WORKERS,
            "supported_languages": sorted(config.SUPPORTED_LANGUAGES.values()),
            "supported_input_modes": list(config.SUPPORTED_INPUT_MODES),
            "tests_passed": 1,
            "frozen_asset_hashes": {frozen_path: frozen_hash},
        }

    def _project_fixture(self, root: Path) -> tuple[Path, Path]:
        for relative_name in REQUIRED_RELEASE_FILES:
            path = root / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative_name == "report_generator.py":
                path.write_text(
                    "# report template\n# @media print\n# html.escape\n",
                    encoding="utf-8",
                )
            elif relative_name not in {
                "release_manifest.json",
                "release_manifest_v1.0.1.json",
            }:
                path.write_text("# release fixture\n", encoding="utf-8")
        (root / "release_manifest.json").write_text(
            json.dumps({"app_version": "1.0.0"}),
            encoding="utf-8",
        )
        frozen = root / "frozen.txt"
        frozen.write_text("frozen release asset\n", encoding="utf-8")
        manifest_path = root / "release_manifest_v1.0.1.json"
        manifest_path.write_text(
            json.dumps(self._manifest("frozen.txt", _sha256(frozen))),
            encoding="utf-8",
        )
        launcher = root / "Amazon Review Analyzer.command"
        launcher.write_text(
            f"#!/bin/zsh\ncd '{root.resolve()}'\n# Streamlit port 8501\nexit 0\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        return manifest_path, launcher

    def test_app_version_has_one_config_source_of_truth(self) -> None:
        self.assertEqual("1.0.1", config.APP_VERSION)
        source = (PROJECT_ROOT / "config.py").read_text(encoding="utf-8")
        self.assertEqual(1, source.count('APP_VERSION = "1.0.1"'))

    def test_real_release_manifest_has_all_required_contract_fields(self) -> None:
        manifest = load_release_manifest(PROJECT_ROOT / "release_manifest_v1.0.1.json")
        self.assertTrue(REQUIRED_MANIFEST_FIELDS.issubset(manifest))
        self.assertEqual(config.APP_VERSION, manifest["app_version"])
        self.assertEqual(1, manifest["default_workers"])
        self.assertEqual("stable", manifest["default_runtime_mode"])
        self.assertGreater(int(manifest["tests_passed"]), 0)

    def test_real_release_manifest_uses_well_formed_sha256_values(self) -> None:
        manifest = load_release_manifest(PROJECT_ROOT / "release_manifest_v1.0.1.json")
        hashes = manifest["frozen_asset_hashes"]
        self.assertIsInstance(hashes, dict)
        self.assertGreater(len(hashes), 40)
        for name, digest in hashes.items():
            with self.subTest(asset=name):
                self.assertIsInstance(name, str)
                self.assertRegex(str(digest), r"^[0-9a-f]{64}$")

    def test_health_check_passes_a_complete_offline_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, launcher = self._project_fixture(root)
            report = run_release_health_check(
                root,
                manifest_path=manifest,
                launcher_path=launcher,
                require_launcher=True,
            )

        self.assertTrue(report.passed, format_health_report(report))
        self.assertEqual(
            {
                "required_files",
                "release_manifest",
                "configuration",
                "localization",
                "frozen_assets",
                "report_template",
                "saved_analysis_library",
                "desktop_launcher",
            },
            {check.name for check in report.checks},
        )

    def test_health_check_detects_a_tampered_frozen_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, launcher = self._project_fixture(root)
            (root / "frozen.txt").write_text("tampered\n", encoding="utf-8")
            report = run_release_health_check(
                root,
                manifest_path=manifest,
                launcher_path=launcher,
                require_launcher=True,
            )

        self.assertFalse(report.passed)
        frozen = next(check for check in report.checks if check.name == "frozen_assets")
        self.assertEqual(HealthStatus.FAIL, frozen.status)
        self.assertIn("hash mismatch", frozen.detail)

    def test_health_check_rejects_frozen_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, launcher = self._project_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["frozen_asset_hashes"] = {"../outside.txt": "0" * 64}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = run_release_health_check(
                root,
                manifest_path=manifest_path,
                launcher_path=launcher,
                require_launcher=True,
            )

        self.assertFalse(report.passed)
        frozen = next(check for check in report.checks if check.name == "frozen_assets")
        self.assertIn("unsafe path", frozen.detail)

    def test_health_check_rejects_a_symlinked_frozen_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, launcher = self._project_fixture(root)
            outside = root / "outside.txt"
            outside.write_text("outside\n", encoding="utf-8")
            link = root / "frozen-link.txt"
            link.symlink_to(outside)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["frozen_asset_hashes"] = {
                "frozen-link.txt": _sha256(outside)
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = run_release_health_check(
                root,
                manifest_path=manifest_path,
                launcher_path=launcher,
                require_launcher=True,
            )

        self.assertFalse(report.passed)
        frozen = next(check for check in report.checks if check.name == "frozen_assets")
        self.assertIn("unsafe path", frozen.detail)

    def test_optional_missing_launcher_is_warning_not_release_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _launcher = self._project_fixture(root)
            report = run_release_health_check(
                root,
                manifest_path=manifest,
                launcher_path=root / "missing.command",
                require_launcher=False,
            )

        self.assertTrue(report.passed)
        launcher = next(check for check in report.checks if check.name == "desktop_launcher")
        self.assertEqual(HealthStatus.WARN, launcher.status)

    def test_required_missing_launcher_blocks_release_health(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _launcher = self._project_fixture(root)
            report = run_release_health_check(
                root,
                manifest_path=manifest,
                launcher_path=root / "missing.command",
                require_launcher=True,
            )

        self.assertFalse(report.passed)
        self.assertIn("desktop_launcher", {check.name for check in report.failed_checks})

    def test_manifest_config_mismatch_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, launcher = self._project_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["default_workers"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = run_release_health_check(
                root,
                manifest_path=manifest_path,
                launcher_path=launcher,
                require_launcher=True,
            )

        self.assertFalse(report.passed)
        configuration = next(
            check for check in report.checks if check.name == "configuration"
        )
        self.assertIn("default_workers", configuration.detail)

    def test_invalid_manifest_fails_closed_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_name in REQUIRED_RELEASE_FILES:
                path = root / relative_name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture\n", encoding="utf-8")
            (root / "release_manifest_v1.0.1.json").write_text(
                "{broken", encoding="utf-8"
            )
            report = run_release_health_check(root, require_launcher=False)

        self.assertFalse(report.passed)
        self.assertIn("unreadable or invalid", report.failed_checks[-1].detail)

    def test_invalid_manifest_types_fail_closed_without_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, launcher = self._project_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["supported_languages"] = [{"unexpected": "object"}]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = run_release_health_check(
                root,
                manifest_path=manifest_path,
                launcher_path=launcher,
                require_launcher=True,
            )

        self.assertFalse(report.passed)
        self.assertIn("supported_languages", report.failed_checks[-1].detail)

    def test_cli_routes_release_health_check_and_prints_pass(self) -> None:
        report = ReleaseHealthReport(
            app_version="1.0.0",
            checks=(
                HealthCheckResult("offline", HealthStatus.PASS, "No network used."),
            ),
        )
        output = io.StringIO()
        with (
            patch.object(main_module, "execute_release_health_check", return_value=report) as runner,
            redirect_stdout(output),
        ):
            exit_code = main_module.cli(["--release-health-check"])

        self.assertEqual(0, exit_code)
        runner.assert_called_once_with()
        self.assertIn("Release health check: PASS", output.getvalue())

    def test_cli_returns_failure_for_a_failed_health_check(self) -> None:
        report = ReleaseHealthReport(
            app_version="1.0.0",
            checks=(
                HealthCheckResult("frozen_assets", HealthStatus.FAIL, "Mismatch."),
            ),
        )
        output = io.StringIO()
        with (
            patch.object(main_module, "execute_release_health_check", return_value=report),
            redirect_stdout(output),
        ):
            exit_code = main_module.cli(["--release-health-check"])

        self.assertEqual(1, exit_code)
        self.assertIn("Release health check: FAIL", output.getvalue())


if __name__ == "__main__":
    unittest.main()
