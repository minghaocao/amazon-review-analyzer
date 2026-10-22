"""Offline unit and security tests for the Step 16.5 macOS launcher."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mac_launcher as launcher


def _make_project(parent: Path, name: str = "amazon review analyzer") -> Path:
    root = parent / name
    root.mkdir(parents=True)
    for relative in launcher.REQUIRED_PROJECT_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n" if relative.endswith(".json") else "# test\n", encoding="utf-8")
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    return root


def _report(*, ai_ready: bool = True, ollama_installed: bool = True, model_missing: bool = False) -> dict:
    checks = {
        name: {"code": f"{name}_ready", "status": "ready", "details": {}}
        for name in ("python", "venv", "dependencies", "filesystem", "port", "disk")
    }
    checks["ollama"] = {
        "code": "ollama_ready" if ai_ready else "ollama_not_running",
        "status": "ready" if ai_ready else "action_required",
        "details": {"installed": ollama_installed, "running": ai_ready},
    }
    checks["model"] = {
        "code": "model_missing" if model_missing else ("model_ready" if ai_ready else "model_unavailable"),
        "status": "ready" if ai_ready else "action_required",
        "details": {"model": "qwen3:8b", "installed": ai_ready and not model_missing},
    }
    return {
        "overall_status": "ready" if ai_ready else "limited",
        "checks": checks,
        "capabilities": {
            "dashboard": True,
            "demo": True,
            "saved_analyses": True,
            "reports": True,
            "comparison": True,
            "new_analysis": ai_ready,
        },
    }


class FakeProcess:
    def __init__(self, pid: int = 43210, poll_values: list[int | None] | None = None):
        self.pid = pid
        self._poll_values = list(poll_values or [None])
        self.signals: list[int] = []
        self.waited = False

    def poll(self) -> int | None:
        if len(self._poll_values) > 1:
            return self._poll_values.pop(0)
        return self._poll_values[0]

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited = True
        return 0

    def send_signal(self, signal_number: int) -> None:
        self.signals.append(signal_number)


class ProjectDiscoveryTests(unittest.TestCase):
    def test_bundle_relative_discovery_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary), "project with spaces")
            bundle = root / "dist" / "Amazon Review Analyzer.app"
            resources = bundle / "Contents" / "Resources"
            resources.mkdir(parents=True)
            (resources / launcher.PROJECT_LOCATOR_FILENAME).write_text(
                json.dumps({"relative_project_path": "../.."}), encoding="utf-8"
            )
            found = launcher.discover_project_root(
                app_bundle=bundle,
                home=Path(temporary) / "unused-home",
            )
            self.assertEqual(root.resolve(), found)

    def test_saved_private_preference_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = _make_project(base)
            preference = base / "private" / "project.json"
            self.assertTrue(launcher.remember_project_root(root, preference))
            self.assertEqual(
                root.resolve(),
                launcher.discover_project_root(
                    preference_path=preference,
                    home=base / "home",
                ),
            )

    def test_known_home_fallback_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            root = _make_project(home / "Documents" / "Projects", "amazon_review_analyzer")
            self.assertEqual(root.resolve(), launcher.discover_project_root(home=home))

    def test_invalid_and_malformed_candidates_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            preference = base / "project.json"
            preference.write_text("not-json", encoding="utf-8")
            self.assertIsNone(
                launcher.discover_project_root(
                    preference_path=preference,
                    home=base / "empty-home",
                )
            )

    def test_signature_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = _make_project(base)
            outside = base / "outside.py"
            outside.write_text("# outside\n", encoding="utf-8")
            (root / "main.py").unlink()
            (root / "main.py").symlink_to(outside)
            self.assertFalse(launcher.is_valid_project_root(root))

    def test_absolute_locator_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = _make_project(base)
            bundle = base / "App.app"
            resources = bundle / "Contents" / "Resources"
            resources.mkdir(parents=True)
            (resources / launcher.PROJECT_LOCATOR_FILENAME).write_text(
                json.dumps({"relative_project_path": str(root)}), encoding="utf-8"
            )
            self.assertIsNone(
                launcher.discover_project_root(
                    app_bundle=bundle,
                    home=base / "empty-home",
                )
            )


class RuntimeMetadataTests(unittest.TestCase):
    def test_runtime_metadata_is_atomic_private_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            path = launcher.write_runtime_metadata(
                root,
                pid=43210,
                port=8501,
                launcher_mode="native",
                health_status="ready",
                started_at="2026-10-22T00:00:00+00:00",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn(str(root), path.read_text(encoding="utf-8"))
            self.assertNotIn("project_root", payload)
            self.assertEqual(0o600, path.stat().st_mode & 0o777)
            self.assertEqual(payload, launcher.load_runtime_metadata(root))

    def test_malformed_stale_and_wrong_signature_metadata_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            path = launcher.runtime_metadata_path(root)
            path.write_text("{broken", encoding="utf-8")
            self.assertIsNone(launcher.load_runtime_metadata(root))
            path.write_text(
                json.dumps(
                    {
                        "schema_version": launcher.RUNTIME_SCHEMA_VERSION,
                        "streamlit_pid": 123,
                        "port": 8501,
                        "project_signature": "wrong",
                        "launcher_mode": "native",
                        "health_status": "ready",
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(launcher.load_runtime_metadata(root))

    def test_runtime_metadata_rejects_invalid_pid_port_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            for kwargs in (
                {"pid": 1, "port": 8501, "launcher_mode": "native", "health_status": "ready"},
                {"pid": 10, "port": 70000, "launcher_mode": "native", "health_status": "ready"},
                {"pid": 10, "port": 8501, "launcher_mode": "unsafe", "health_status": "ready"},
            ):
                with self.assertRaises(ValueError):
                    launcher.write_runtime_metadata(root, **kwargs)

    def test_launcher_log_is_allowlisted_and_drops_sensitive_error_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            launcher.append_launcher_log(
                root,
                stage="streamlit_start",
                status="failed",
                error_code="API key=/Users/person/review evidence cookie",
                port=8501,
            )
            text = launcher.launcher_log_path(root).read_text(encoding="utf-8")
            self.assertNotIn("/Users/", text)
            self.assertNotIn("review", text.casefold())
            self.assertNotIn("cookie", text.casefold())
            self.assertNotIn("api key", text.casefold())
            payload = json.loads(text)
            self.assertEqual("redacted", payload["error_code"])
            self.assertEqual(
                {
                    "timestamp",
                    "app_version",
                    "launcher_version",
                    "stage",
                    "status",
                    "error_code",
                    "port",
                },
                set(payload),
            )

    def test_runtime_symlink_outside_project_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = _make_project(base)
            outside = base / "outside-runtime"
            outside.mkdir()
            (root / "outputs").mkdir()
            (root / "outputs" / "runtime").symlink_to(outside)
            with self.assertRaises(OSError):
                launcher.runtime_metadata_path(root)


class ExistingInstanceTests(unittest.TestCase):
    def test_live_project_streamlit_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            probes = launcher.LauncherProbes(
                listener_pids=lambda port: (321,),
                pid_alive=lambda pid: True,
                pid_command=lambda pid: "/venv/python -m streamlit run app.py",
                pid_cwd=lambda pid: root.resolve(),
                http_ready=lambda port: True,
            )
            self.assertEqual((321, 8501), launcher.find_existing_instance(root, probes=probes))

    def test_unrelated_process_is_not_misidentified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            probes = launcher.LauncherProbes(
                listener_pids=lambda port: (321,),
                pid_alive=lambda pid: True,
                pid_command=lambda pid: "python unrelated_server.py",
                pid_cwd=lambda pid: root.resolve(),
                http_ready=lambda port: True,
            )
            self.assertIsNone(launcher.find_existing_instance(root, probes=probes))

    def test_stale_runtime_pid_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            launcher.write_runtime_metadata(
                root,
                pid=321,
                port=8501,
                launcher_mode="native",
                health_status="ready",
            )
            probes = launcher.LauncherProbes(
                listener_pids=lambda port: (),
                pid_alive=lambda pid: False,
            )
            self.assertIsNone(launcher.find_existing_instance(root, probes=probes))


class LaunchFlowTests(unittest.TestCase):
    def _probes(
        self,
        root: Path,
        *,
        report: dict | None = None,
        listeners: tuple[int, ...] = (),
        ready: bool = True,
        dialog_action: bool = True,
        process: FakeProcess | None = None,
    ) -> tuple[launcher.LauncherProbes, dict[str, list]]:
        events: dict[str, list] = {
            "urls": [],
            "dialogs": [],
            "commands": [],
            "paths": [],
            "ollama": [],
        }
        resolved_process = process or FakeProcess()

        def dialog(title: str, message: str, cancel: str, action: str) -> str:
            events["dialogs"].append((title, message, cancel, action))
            return action if dialog_action else cancel

        def spawn(command, cwd, environment):
            events["commands"].append((list(command), Path(cwd), dict(environment)))
            return resolved_process

        probes = launcher.LauncherProbes(
            listener_pids=lambda port: listeners,
            pid_alive=lambda pid: True,
            pid_command=lambda pid: "/venv/python -m streamlit run app.py",
            pid_cwd=lambda pid: root.resolve(),
            http_ready=lambda port: ready,
            open_url=lambda url: events["urls"].append(url) is None or True,
            open_path=lambda path: events["paths"].append(path) is None or True,
            open_ollama=lambda: events["ollama"].append(True) is None or True,
            dialog=dialog,
            spawn=spawn,
            health_runner=lambda project, port: report or _report(),
            sleep=lambda seconds: None,
            monotonic=lambda: 1.0,
        )
        return probes, events

    def test_ready_native_launch_uses_safe_argv_and_opens_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary), "project ; with spaces")
            probes, events = self._probes(root)
            result = launcher.launch_dashboard(
                root,
                native=True,
                foreground=False,
                probes=probes,
            )
            self.assertEqual(0, result)
            self.assertEqual(["http://localhost:8501"], events["urls"])
            command, cwd, environment = events["commands"][0]
            self.assertIsInstance(command, list)
            self.assertEqual(root.resolve(), cwd)
            self.assertEqual(
                str((root / ".venv" / "bin" / "python").resolve()),
                command[0],
            )
            self.assertIn("app.py", command)
            self.assertEqual("8501", environment["ARA_STREAMLIT_PORT"])
            self.assertEqual("native", environment["ARA_LAUNCHER_MODE"])
            self.assertEqual([], events["dialogs"])

    def test_duplicate_launch_reopens_without_spawning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            probes, events = self._probes(root, listeners=(321,))
            result = launcher.launch_dashboard(
                root,
                native=True,
                foreground=False,
                probes=probes,
            )
            self.assertEqual(0, result)
            self.assertEqual([], events["commands"])
            self.assertEqual(["http://localhost:8501"], events["urls"])

    def test_unrelated_port_conflict_stops_without_killing_or_spawning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            probes, events = self._probes(root, listeners=(999,))
            probes = launcher.LauncherProbes(
                **{
                    **probes.__dict__,
                    "pid_command": lambda pid: "unrelated server",
                }
            )
            with mock.patch("os.kill") as kill:
                result = launcher.launch_dashboard(root, native=True, foreground=False, probes=probes)
            self.assertEqual(3, result)
            kill.assert_not_called()
            self.assertEqual([], events["commands"])
            self.assertIn("Port 8501", events["dialogs"][0][1])

    def test_limited_mode_can_open_without_ai(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            probes, events = self._probes(
                root,
                report=_report(ai_ready=False, ollama_installed=False),
            )
            result = launcher.launch_dashboard(root, native=True, foreground=False, probes=probes)
            self.assertEqual(0, result)
            self.assertEqual(1, len(events["dialogs"]))
            self.assertIn("Limited Mode", events["dialogs"][0][1])
            metadata = launcher.load_runtime_metadata(root)
            self.assertEqual("limited", metadata["health_status"])

    def test_limited_mode_cancel_does_not_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            probes, events = self._probes(
                root,
                report=_report(ai_ready=False, ollama_installed=False),
                dialog_action=False,
            )
            result = launcher.launch_dashboard(root, native=True, foreground=False, probes=probes)
            self.assertEqual(5, result)
            self.assertEqual([], events["commands"])

    def test_model_missing_gets_specific_dialog_and_never_pulls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            report = _report(ai_ready=False, ollama_installed=True, model_missing=True)
            report["checks"]["ollama"] = {
                "code": "ollama_ready",
                "status": "ready",
                "details": {"installed": True, "running": True},
            }
            probes, events = self._probes(root, report=report)
            self.assertEqual(0, launcher.launch_dashboard(root, native=True, foreground=False, probes=probes))
            self.assertIn("qwen3:8b", events["dialogs"][0][1])
            command_text = json.dumps(events["commands"], default=str)
            self.assertNotIn("pull", command_text)

    def test_ollama_start_is_bounded_then_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            reports = iter([_report(ai_ready=False), _report(ai_ready=True)])
            probes, events = self._probes(root)
            probes = launcher.LauncherProbes(
                **{**probes.__dict__, "health_runner": lambda project, port: next(reports)}
            )
            self.assertEqual(0, launcher.launch_dashboard(root, native=True, foreground=False, probes=probes))
            self.assertEqual([True], events["ollama"])
            self.assertEqual([], events["dialogs"])

    def test_failed_readiness_stops_only_spawned_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            process = FakeProcess()
            probes, events = self._probes(root, ready=False, process=process)
            result = launcher.launch_dashboard(root, native=True, foreground=False, probes=probes)
            self.assertEqual(7, result)
            self.assertEqual([signal.SIGINT], process.signals)
            self.assertTrue(process.waited)
            self.assertEqual([], events["urls"])

    def test_core_health_failure_routes_to_existing_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            report = _report()
            report["checks"]["dependencies"]["status"] = "unavailable"
            report["capabilities"]["dashboard"] = False
            probes, events = self._probes(root, report=report)
            result = launcher.launch_dashboard(root, native=True, foreground=False, probes=probes)
            self.assertEqual(4, result)
            self.assertEqual(
                [(root / "setup_amazon_review_analyzer.command").resolve()],
                events["paths"],
            )
            self.assertEqual([], events["commands"])


class StopAndSecurityTests(unittest.TestCase):
    def test_stop_uses_only_verified_runtime_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            launcher.write_runtime_metadata(
                root,
                pid=777,
                port=8501,
                launcher_mode="native",
                health_status="ready",
            )
            probes = launcher.LauncherProbes(
                listener_pids=lambda port: (777,),
                pid_alive=lambda pid: True,
                pid_command=lambda pid: "python -m streamlit run app.py",
                pid_cwd=lambda pid: root.resolve(),
                http_ready=lambda port: True,
            )
            with mock.patch("os.kill") as kill:
                self.assertEqual(0, launcher.stop_dashboard(root, native=False, probes=probes))
            kill.assert_called_once_with(777, signal.SIGINT)

    def test_stop_refuses_stale_or_foreign_runtime_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_project(Path(temporary))
            launcher.write_runtime_metadata(
                root,
                pid=777,
                port=8501,
                launcher_mode="native",
                health_status="ready",
            )
            probes = launcher.LauncherProbes(
                listener_pids=lambda port: (777,),
                pid_alive=lambda pid: True,
                pid_command=lambda pid: "python unrelated.py",
                pid_cwd=lambda pid: root.resolve(),
                http_ready=lambda port: True,
            )
            with mock.patch("os.kill") as kill:
                self.assertEqual(1, launcher.stop_dashboard(root, native=False, probes=probes))
            kill.assert_not_called()

    def test_source_has_no_unsafe_shell_or_global_kill(self) -> None:
        source = Path(launcher.__file__).read_text(encoding="utf-8")
        self.assertNotIn("shell=True", source)
        self.assertNotIn("killall", source.casefold())
        self.assertNotIn("xattr", source.casefold())
        self.assertNotIn("CERT_NONE", source)
        self.assertNotIn("verify=False", source)

    def test_browser_url_is_local_and_numeric_only(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch("subprocess.run", return_value=completed) as run:
            self.assertTrue(launcher._default_open_url("http://localhost:8501"))
            self.assertFalse(launcher._default_open_url("https://example.com"))
            self.assertFalse(launcher._default_open_url("http://localhost:8501;open evil"))
        run.assert_called_once()
        self.assertEqual(
            ["/usr/bin/open", "http://localhost:8501"],
            run.call_args.args[0],
        )

    def test_dialog_text_is_static_bilingual_and_html_safe(self) -> None:
        for code in (
            "project_not_found",
            "venv_missing",
            "ollama_missing",
            "model_missing",
            "port_conflict",
        ):
            title, message, cancel, action = launcher._dialog_text(code)
            self.assertEqual(launcher.APP_NAME, title)
            self.assertIn("取消", cancel)
            self.assertRegex(message, r"[\u4e00-\u9fff]")
            self.assertNotIn("<script", message.casefold())
            self.assertNotIn("traceback", message.casefold())
            self.assertTrue(action)


if __name__ == "__main__":
    unittest.main()

