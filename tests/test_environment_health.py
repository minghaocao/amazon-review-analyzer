"""Offline tests for the Step 16 environment-health boundary."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import Mock, patch

import environment_health as health


DiskUsage = namedtuple("DiskUsage", "total used free")


class EnvironmentHealthTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        (self.root / "requirements.txt").write_text(
            "\n".join(
                (
                    "pandas>=2.2",
                    "python-dotenv>=1.0",
                    "certifi>=2024.8.30",
                    "pytest>=8.0",
                    "openai>=2.0",
                    "ollama>=0.6.2,<1",
                    "pydantic>=2.9",
                    "streamlit>=1.50,<2",
                    "plotly>=6.0,<7",
                    "playwright>=1.49,<2",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        (self.root / "release_manifest.json").write_text(
            json.dumps(
                {
                    "app_version": "1.0.0",
                    "model_name": "qwen3:8b",
                    "default_runtime_mode": "stable",
                    "default_workers": 1,
                }
            ),
            encoding="utf-8",
        )
        venv_python = self.root / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("", encoding="utf-8")
        self.settings = health.HealthSettings(app_version="1.0.0")

    def _json_probe(self, url: str, _timeout: float):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b", "size": 5_000_000_000}]}
        if url.endswith("/api/version"):
            return {"version": "0.34.2"}
        raise AssertionError(f"unexpected endpoint: {url}")

    def _ready_probes(self, **overrides):
        values = {
            "package_version": lambda _name: "1.0.0",
            "which": lambda name: "/usr/local/bin/ollama" if name == "ollama" else None,
            "run_command": lambda _command, _timeout: (0, "ollama version is 0.34.2"),
            "http_get_json": self._json_probe,
            "port_probe": lambda _root, _port: "free",
            "disk_usage": lambda _path: DiskUsage(20 * 1024**3, 5 * 1024**3, 15 * 1024**3),
            "write_probe": health._default_write_probe,
        }
        values.update(overrides)
        return health.HealthProbes(**values)

    def _run(self, **kwargs):
        return health.run_environment_health(
            project_root=self.root,
            settings=kwargs.pop("settings", self.settings),
            probes=kwargs.pop("probes", self._ready_probes()),
            python_version_info=kwargs.pop("python_version_info", (3, 13, 4)),
            python_version_text=kwargs.pop("python_version_text", "3.13.4"),
            python_executable=kwargs.pop("python_executable", sys.executable),
            current_prefix=kwargs.pop("current_prefix", self.root / ".venv"),
            base_prefix=kwargs.pop("base_prefix", self.root / "base-python"),
            **kwargs,
        )

    def test_ready_environment_has_stable_shape_and_capabilities(self) -> None:
        report = self._run()
        payload = report.to_dict()

        self.assertEqual(health.OverallStatus.READY, report.overall_status)
        self.assertEqual("environment-health.v1", payload["schema_version"])
        self.assertEqual(
            {
                "python",
                "venv",
                "dependencies",
                "ollama",
                "model",
                "port",
                "filesystem",
                "disk",
            },
            set(payload["checks"]),
        )
        self.assertTrue(all(payload["capabilities"].values()))
        self.assertEqual({"mode": "stable", "workers": 1}, payload["runtime"])
        self.assertNotIn("checked_at", payload)

    def test_python_minimum_is_grounded_at_311(self) -> None:
        supported = health.check_python(
            self.settings,
            version_info=(3, 11, 0),
            version_text="3.11.0",
            executable=sys.executable,
        )
        unsupported = health.check_python(
            self.settings,
            version_info=(3, 10, 14),
            version_text="3.10.14",
            executable=sys.executable,
        )

        self.assertEqual(health.CheckStatus.READY, supported.status)
        self.assertEqual(health.CheckStatus.UNAVAILABLE, unsupported.status)
        self.assertEqual("3.11+", unsupported.details["required"])
        self.assertNotIn(str(sys.executable), json.dumps(unsupported.to_dict()))

    def test_unsupported_python_is_fatal_exit_two(self) -> None:
        report = self._run(python_version_info=(3, 10, 9), python_version_text="3.10.9")

        self.assertEqual(health.OverallStatus.UNAVAILABLE, report.overall_status)
        self.assertEqual(2, health.health_exit_code(report))

    def test_missing_virtual_environment_is_action_required_without_path(self) -> None:
        for path in (self.root / ".venv").rglob("*"):
            if path.is_file():
                path.unlink()
        (self.root / ".venv" / "bin").rmdir()
        (self.root / ".venv").rmdir()

        report = self._run(current_prefix=self.root / "other")
        check = report.checks["venv"]

        self.assertEqual("venv_missing", check.code)
        self.assertEqual(health.OverallStatus.ACTION_REQUIRED, report.overall_status)
        self.assertNotIn(str(self.root), json.dumps(report.to_dict()))

    def test_inactive_project_environment_is_detected(self) -> None:
        check = health.check_virtual_environment(
            self.root,
            current_prefix=self.root / "another-venv",
            base_prefix=self.root / "base-python",
        )
        self.assertEqual("venv_inactive", check.code)
        self.assertEqual(health.CheckStatus.ACTION_REQUIRED, check.status)

    def test_requirements_are_the_dependency_source(self) -> None:
        names = health.load_requirement_names(self.root / "requirements.txt")
        self.assertIn("python-dotenv", names)
        self.assertIn("streamlit", names)
        self.assertNotIn("requests", names)
        self.assertEqual(len(names), len(set(names)))

    def test_missing_required_dependency_is_fatal(self) -> None:
        def versions(name: str) -> str:
            if name == "pandas":
                raise health.importlib.metadata.PackageNotFoundError(name)
            return "1.0"

        report = self._run(probes=self._ready_probes(package_version=versions))
        check = report.checks["dependencies"]

        self.assertEqual("dependencies_missing", check.code)
        self.assertEqual(["pandas"], check.details["missing_required"])
        self.assertEqual(health.OverallStatus.UNAVAILABLE, report.overall_status)

    def test_optional_dependencies_warn_without_disabling_core_dashboard(self) -> None:
        optional = {"openai", "pytest", "playwright"}

        def versions(name: str) -> str:
            if name in optional:
                raise health.importlib.metadata.PackageNotFoundError(name)
            return "1.0"

        report = self._run(probes=self._ready_probes(package_version=versions))

        self.assertEqual(health.CheckStatus.WARNING, report.checks["dependencies"].status)
        self.assertTrue(report.capabilities["dashboard"])
        self.assertTrue(report.capabilities["new_analysis"])
        self.assertFalse(report.capabilities["amazon_import"])
        self.assertEqual(health.OverallStatus.LIMITED, report.overall_status)

    def test_dev_and_optional_provider_packages_do_not_create_limited_mode(self) -> None:
        optional = {"openai", "pytest"}

        def versions(name: str) -> str:
            if name in optional:
                raise health.importlib.metadata.PackageNotFoundError(name)
            return "1.0"

        report = self._run(probes=self._ready_probes(package_version=versions))

        self.assertEqual(health.CheckStatus.READY, report.checks["dependencies"].status)
        self.assertEqual(health.OverallStatus.READY, report.overall_status)
        self.assertTrue(report.capabilities["new_analysis"])
        self.assertEqual(sorted(optional), report.checks["dependencies"].details["missing_optional"])

    def test_missing_ollama_python_package_disables_only_new_analysis(self) -> None:
        def versions(name: str) -> str:
            if name == "ollama":
                raise health.importlib.metadata.PackageNotFoundError(name)
            return "1.0"

        report = self._run(probes=self._ready_probes(package_version=versions))

        self.assertEqual(health.OverallStatus.LIMITED, report.overall_status)
        self.assertTrue(report.capabilities["demo"])
        self.assertFalse(report.capabilities["new_analysis"])
        self.assertIn("ollama", report.checks["dependencies"].details["missing_features"])

    def test_missing_ollama_is_limited_and_never_calls_inference(self) -> None:
        endpoints: list[str] = []

        def unavailable(url: str, _timeout: float):
            endpoints.append(url)
            raise ConnectionRefusedError()

        report = self._run(
            probes=self._ready_probes(
                which=lambda _name: None,
                http_get_json=unavailable,
            )
        )

        self.assertEqual("ollama_not_installed", report.checks["ollama"].code)
        self.assertEqual(health.OverallStatus.LIMITED, report.overall_status)
        self.assertTrue(report.capabilities["demo"])
        self.assertFalse(report.capabilities["new_analysis"])
        self.assertTrue(all("/api/tags" in url for url in endpoints))
        self.assertFalse(any("chat" in url or "generate" in url for url in endpoints))

    def test_installed_but_stopped_ollama_is_distinct(self) -> None:
        report = self._run(
            probes=self._ready_probes(
                http_get_json=lambda _url, _timeout: (_ for _ in ()).throw(
                    ConnectionRefusedError()
                )
            )
        )
        self.assertEqual("ollama_not_running", report.checks["ollama"].code)
        self.assertTrue(report.checks["ollama"].details["installed"])
        self.assertFalse(report.checks["ollama"].details["running"])

    def test_model_missing_does_not_run_model(self) -> None:
        calls: list[str] = []

        def get_json(url: str, _timeout: float):
            calls.append(url)
            if url.endswith("/api/tags"):
                return {"models": [{"name": "another-model:latest"}]}
            return {"version": "0.34.2"}

        report = self._run(probes=self._ready_probes(http_get_json=get_json))

        self.assertEqual("model_missing", report.checks["model"].code)
        self.assertEqual(health.OverallStatus.LIMITED, report.overall_status)
        self.assertEqual(2, len(calls))
        self.assertTrue(all(url.endswith(("/api/tags", "/api/version")) for url in calls))

    def test_model_metadata_size_is_used_instead_of_hardcoded_size(self) -> None:
        report = self._run()
        self.assertEqual(5_000_000_000, report.checks["model"].details["size_bytes"])

    def test_credentialed_ollama_host_is_rejected_and_redacted(self) -> None:
        settings = health.HealthSettings(
            app_version="1.0.0",
            ollama_host="http://alice:topsecret@localhost:11434",
        )
        network = Mock(side_effect=AssertionError("network must not be called"))
        report = self._run(
            settings=settings,
            probes=self._ready_probes(http_get_json=network),
        )
        text = json.dumps(report.to_dict())

        self.assertEqual("ollama_host_invalid", report.checks["ollama"].code)
        network.assert_not_called()
        self.assertNotIn("alice", text)
        self.assertNotIn("topsecret", text)

    def test_port_free_and_existing_app_are_ready(self) -> None:
        free = health.check_port(
            self.root, self.settings, port_probe=lambda _root, _port: "free"
        )
        current = health.check_port(
            self.root, self.settings, port_probe=lambda _root, _port: "current_app"
        )
        self.assertEqual("port_ready", free.code)
        self.assertEqual("app_already_running", current.code)
        self.assertEqual(health.CheckStatus.READY, current.status)

    def test_other_port_owner_requires_action_and_is_never_killed(self) -> None:
        probe = Mock(return_value="other_process")
        report = self._run(probes=self._ready_probes(port_probe=probe))

        self.assertEqual("port_in_use", report.checks["port"].code)
        self.assertEqual(health.OverallStatus.ACTION_REQUIRED, report.overall_status)
        probe.assert_called_once_with(self.root, 8501)

    def test_default_port_probe_identifies_current_project_streamlit(self) -> None:
        with (
            patch.object(health, "_port_is_open", return_value=True),
            patch.object(health, "_listener_pids", return_value=("123",)),
            patch.object(health, "_pid_cwd", return_value=self.root.resolve()),
            patch.object(health, "_pid_is_streamlit_app", return_value=True),
        ):
            state = health._default_port_probe(self.root, 8501)
        self.assertEqual("current_app", state)

    def test_default_port_probe_does_not_claim_unknown_listener(self) -> None:
        with (
            patch.object(health, "_port_is_open", return_value=True),
            patch.object(health, "_listener_pids", return_value=("456",)),
            patch.object(health, "_pid_cwd", return_value=self.root.resolve()),
            patch.object(health, "_pid_is_streamlit_app", return_value=False),
        ):
            state = health._default_port_probe(self.root, 8501)
        self.assertEqual("other_process", state)

    def test_filesystem_creates_only_runtime_directories_and_removes_probe(self) -> None:
        check = health.check_filesystem(self.root)

        self.assertEqual(health.CheckStatus.READY, check.status)
        for relative_name in health.RUNTIME_DIRECTORY_NAMES:
            self.assertTrue((self.root / relative_name).is_dir())
        self.assertFalse(list((self.root / "outputs").glob(".environment-health-*")))
        self.assertFalse((self.root / "data" / "raw_reviews.csv").exists())

    def test_filesystem_write_failure_is_fatal_and_does_not_expose_error(self) -> None:
        def fail(_directory: Path) -> None:
            raise PermissionError(f"cannot write {self.root}/secret")

        report = self._run(probes=self._ready_probes(write_probe=fail))
        text = json.dumps(report.to_dict())

        self.assertEqual("filesystem_not_writable", report.checks["filesystem"].code)
        self.assertEqual(health.OverallStatus.UNAVAILABLE, report.overall_status)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("secret", text)

    def test_filesystem_refuses_output_symlink_outside_project(self) -> None:
        external = self.root.parent / f"{self.root.name}-external"
        external.mkdir()
        self.addCleanup(external.rmdir)
        (self.root / "outputs").symlink_to(external, target_is_directory=True)

        check = health.check_filesystem(self.root)
        self.assertEqual("filesystem_unsafe_path", check.code)
        self.assertEqual(health.CheckStatus.UNAVAILABLE, check.status)

    def test_low_disk_space_warns_but_demo_remains_available(self) -> None:
        low = DiskUsage(20 * 1024**3, 19_500_000_000, 512 * 1024**2)
        report = self._run(probes=self._ready_probes(disk_usage=lambda _path: low))

        self.assertEqual("disk_low", report.checks["disk"].code)
        self.assertEqual(health.OverallStatus.LIMITED, report.overall_status)
        self.assertTrue(report.capabilities["demo"])

    def test_diagnostics_sanitize_paths_credentials_and_sensitive_keys(self) -> None:
        payload = health.sanitize_diagnostics(
            {
                "api_key": "sk-secret-value",
                "detail": (
                    "File /Users/private-name/project/app.py and "
                    "https://person:password@example.test/path"
                ),
                "nested": {"cookie": "session-secret"},
            }
        )
        serialized = json.dumps(payload)
        self.assertNotIn("sk-secret-value", serialized)
        self.assertNotIn("private-name", serialized)
        self.assertNotIn("person:password", serialized)
        self.assertNotIn("session-secret", serialized)
        self.assertIn("[redacted]", serialized)

    def test_json_and_human_outputs_have_no_absolute_user_path(self) -> None:
        report = self._run()
        json_text = health.format_health_json(report)
        human_text = health.format_health_report(report)

        self.assertNotIn(str(self.root), json_text)
        self.assertNotIn(str(self.root), human_text)
        self.assertNotIn(str(sys.executable), json_text)
        self.assertIn("Status: READY", human_text)

    def test_health_cli_json_is_machine_readable_and_exit_zero(self) -> None:
        report = self._run()
        stream = io.StringIO()
        exit_code = health.run_health_cli(report=report, json_output=True, stream=stream)

        payload = json.loads(stream.getvalue())
        self.assertEqual(0, exit_code)
        self.assertEqual("ready", payload["overall_status"])

    def test_health_cli_exit_one_for_limited_and_action_required(self) -> None:
        limited = self._run(
            probes=self._ready_probes(
                http_get_json=lambda _url, _timeout: (_ for _ in ()).throw(
                    ConnectionRefusedError()
                )
            )
        )
        action = self._run(
            probes=self._ready_probes(port_probe=lambda _root, _port: "other_process")
        )
        self.assertEqual(1, health.health_exit_code(limited))
        self.assertEqual(1, health.health_exit_code(action))

    def test_health_cli_exit_two_for_fatal_environment(self) -> None:
        report = self._run(python_version_info=(3, 10, 0), python_version_text="3.10.0")
        stream = io.StringIO()
        exit_code = health.run_health_cli(report=report, stream=stream)
        self.assertEqual(2, exit_code)
        self.assertIn("UNAVAILABLE", stream.getvalue())

    def test_settings_discovery_uses_manifest_and_selected_env_keys_only(self) -> None:
        (self.root / ".env").write_text(
            "OLLAMA_HOST=http://localhost:9999\n"
            "OLLAMA_MODEL=model-from-file:latest\n"
            "OPENAI_API_KEY=must-not-be-read\n",
            encoding="utf-8",
        )
        settings = health.discover_health_settings(
            self.root,
            environment={"OLLAMA_MODEL": "qwen3:8b"},
        )
        serialized = json.dumps(settings.__dict__)

        self.assertEqual("1.0.0", settings.app_version)
        self.assertEqual("http://localhost:9999", settings.ollama_host)
        self.assertEqual("qwen3:8b", settings.model_name)
        self.assertEqual("stable", settings.runtime_mode)
        self.assertEqual(1, settings.runtime_workers)
        self.assertNotIn("must-not-be-read", serialized)

    def test_report_is_deterministic_for_identical_probes(self) -> None:
        first = self._run().to_dict()
        second = self._run().to_dict()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
