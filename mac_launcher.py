"""Privacy-safe macOS launcher orchestration for Amazon Review Analyzer.

The native ``.app`` and the legacy ``.command`` entry points both delegate to
this standard-library-only module.  It deliberately does not import any review
analysis code.  Environment decisions continue to come from
``environment_health.py`` after a project directory has been located.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen


APP_NAME = "Amazon Review Analyzer"
APP_VERSION = "1.0.1"
LAUNCHER_VERSION = "16.5"
BUNDLE_IDENTIFIER = "com.caominghao.amazonreviewanalyzer"
DEFAULT_PORT = 8501
LOCAL_HOST = "127.0.0.1"
MODEL_NAME = "qwen3:8b"
RUNTIME_SCHEMA_VERSION = "amazon-review-analyzer.launcher-runtime.v1"
PREFERENCE_SCHEMA_VERSION = "amazon-review-analyzer.launcher-preference.v1"
PROJECT_LOCATOR_FILENAME = "project_locator.json"
MAX_JSON_BYTES = 32 * 1024
MAX_LOG_BYTES = 256 * 1024
READY_ATTEMPTS = 30
READY_DELAY_SECONDS = 0.5
OLLAMA_START_ATTEMPTS = 5
OLLAMA_START_DELAY_SECONDS = 1.0

REQUIRED_PROJECT_FILES = (
    "main.py",
    "app.py",
    "environment_health.py",
    "release_manifest_v1.0.1.json",
    "setup_amazon_review_analyzer.command",
)

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_SAFE_STAGES = frozenset(
    {
        "discover",
        "health",
        "ollama_start",
        "existing_instance",
        "streamlit_start",
        "streamlit_ready",
        "browser_open",
        "stop",
    }
)
_SAFE_STATUSES = frozenset(
    {"started", "ready", "limited", "reused", "failed", "stopped", "cancelled"}
)


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def send_signal(self, signal_number: int) -> None: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _inside(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_resolve(path: Path) -> Path | None:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def is_valid_project_root(path: str | Path) -> bool:
    """Validate one bounded project candidate without scanning elsewhere."""

    resolved = _safe_resolve(Path(path))
    if resolved is None or not resolved.is_dir():
        return False
    for relative_name in REQUIRED_PROJECT_FILES:
        candidate = resolved / relative_name
        try:
            if not candidate.is_file():
                return False
            if not _inside(resolved, candidate.resolve(strict=True)):
                return False
        except (OSError, RuntimeError):
            return False
    return True


def default_preference_path(home: str | Path | None = None) -> Path:
    base = Path(home).expanduser() if home is not None else Path.home()
    return (
        base
        / "Library"
        / "Application Support"
        / APP_NAME
        / "project.json"
    )


def _read_small_json(path: Path) -> Mapping[str, Any] | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _locator_relative_candidate(app_bundle: Path) -> Path | None:
    locator = app_bundle / "Contents" / "Resources" / PROJECT_LOCATOR_FILENAME
    payload = _read_small_json(locator)
    if payload is None:
        return None
    relative = payload.get("relative_project_path")
    if not isinstance(relative, str) or not relative.strip():
        return None
    candidate = Path(relative)
    if candidate.is_absolute():
        return None
    return app_bundle / candidate


def _preference_candidate(preference_path: Path) -> Path | None:
    payload = _read_small_json(preference_path)
    if payload is None or payload.get("schema_version") != PREFERENCE_SCHEMA_VERSION:
        return None
    value = payload.get("project_root")
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else None


def discover_project_root(
    *,
    app_bundle: str | Path | None = None,
    project_override: str | Path | None = None,
    preference_path: str | Path | None = None,
    home: str | Path | None = None,
) -> Path | None:
    """Find the project from a small ordered candidate set.

    No recursive filesystem search is performed.  The private preference may
    contain an absolute path, but that path is never copied into launcher logs,
    diagnostics, reports, or public documentation.
    """

    home_path = Path(home).expanduser() if home is not None else Path.home()
    preference = (
        Path(preference_path).expanduser()
        if preference_path is not None
        else default_preference_path(home_path)
    )
    candidates: list[Path] = []
    if project_override is not None:
        candidates.append(Path(project_override))
    bundle = _safe_resolve(Path(app_bundle)) if app_bundle is not None else None
    if bundle is not None:
        relative = _locator_relative_candidate(bundle)
        if relative is not None:
            candidates.append(relative)
        # Supports an app placed beside an ``amazon_review_analyzer`` folder.
        candidates.append(bundle.parent / "amazon_review_analyzer")
    saved = _preference_candidate(preference)
    if saved is not None:
        candidates.append(saved)
    candidates.append(home_path / "Documents" / "Projects" / "amazon_review_analyzer")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = _safe_resolve(candidate)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        if is_valid_project_root(resolved):
            return resolved
    return None


def remember_project_root(
    project_root: str | Path,
    preference_path: str | Path | None = None,
) -> bool:
    """Atomically save a private project location for an installed app."""

    root = _safe_resolve(Path(project_root))
    if root is None or not is_valid_project_root(root):
        return False
    destination = (
        Path(preference_path).expanduser()
        if preference_path is not None
        else default_preference_path()
    )
    payload = {
        "schema_version": PREFERENCE_SCHEMA_VERSION,
        "project_root": str(root),
    }
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(destination, payload)
    except OSError:
        return False
    return True


def project_signature(project_root: str | Path) -> str:
    """Return a stable, non-path project identity used by runtime metadata."""

    root = Path(project_root)
    digest = hashlib.sha256()
    for relative_name in ("release_manifest_v1.0.1.json", "main.py", "app.py"):
        digest.update(relative_name.encode("utf-8"))
        try:
            digest.update((root / relative_name).read_bytes())
        except OSError:
            digest.update(b"unavailable")
    return digest.hexdigest()


def _runtime_directory(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    destination = root / "outputs" / "runtime" / "launcher"
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    if not _inside(root, resolved):
        raise OSError("unsafe_runtime_directory")
    return resolved


def runtime_metadata_path(project_root: str | Path) -> Path:
    return _runtime_directory(project_root) / "runtime.json"


def launcher_log_path(project_root: str | Path) -> Path:
    return _runtime_directory(project_root) / "launcher.log"


def launch_lock_path(project_root: str | Path) -> Path:
    return _runtime_directory(project_root) / "launch.lock"


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def write_runtime_metadata(
    project_root: str | Path,
    *,
    pid: int,
    port: int,
    launcher_mode: str,
    health_status: str,
    started_at: str | None = None,
) -> Path:
    if pid <= 1 or not 1 <= port <= 65535:
        raise ValueError("invalid runtime process metadata")
    if launcher_mode not in {"native", "command"}:
        raise ValueError("invalid launcher mode")
    if health_status not in {"ready", "limited"}:
        raise ValueError("invalid health status")
    payload = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "launcher_version": LAUNCHER_VERSION,
        "started_at": started_at or _utc_now(),
        "streamlit_pid": pid,
        "port": port,
        "project_signature": project_signature(project_root),
        "launcher_mode": launcher_mode,
        "health_status": health_status,
    }
    destination = runtime_metadata_path(project_root)
    _atomic_json_write(destination, payload)
    return destination


def load_runtime_metadata(
    project_root: str | Path,
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    destination = Path(path) if path is not None else runtime_metadata_path(project_root)
    payload = _read_small_json(destination)
    if payload is None or payload.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        return None
    pid = payload.get("streamlit_pid")
    port = payload.get("port")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return None
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        return None
    if payload.get("project_signature") != project_signature(project_root):
        return None
    if payload.get("launcher_mode") not in {"native", "command"}:
        return None
    if payload.get("health_status") not in {"ready", "limited"}:
        return None
    allowed = {
        "schema_version",
        "app_version",
        "launcher_version",
        "started_at",
        "streamlit_pid",
        "port",
        "project_signature",
        "launcher_mode",
        "health_status",
    }
    return {key: payload[key] for key in allowed if key in payload}


def append_launcher_log(
    project_root: str | Path,
    *,
    stage: str,
    status: str,
    error_code: str = "none",
    pid: int | None = None,
    port: int | None = None,
    elapsed_ms: int | None = None,
) -> None:
    """Append one allowlisted JSON event; arbitrary text is never accepted."""

    if stage not in _SAFE_STAGES or status not in _SAFE_STATUSES:
        return
    if not _SAFE_TOKEN.fullmatch(error_code):
        error_code = "redacted"
    payload: dict[str, Any] = {
        "timestamp": _utc_now(),
        "app_version": APP_VERSION,
        "launcher_version": LAUNCHER_VERSION,
        "stage": stage,
        "status": status,
        "error_code": error_code,
    }
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1:
        payload["pid"] = pid
    if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535:
        payload["port"] = port
    if isinstance(elapsed_ms, int) and not isinstance(elapsed_ms, bool) and elapsed_ms >= 0:
        payload["elapsed_ms"] = elapsed_ms
    try:
        destination = launcher_log_path(project_root)
        if destination.exists() and destination.stat().st_size >= MAX_LOG_BYTES:
            rotated = destination.with_suffix(".log.1")
            os.replace(destination, rotated)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        os.chmod(destination, 0o600)
    except OSError:
        # Logging is best-effort and must never prevent the dashboard opening.
        return


def _listener_pids(port: int) -> tuple[int, ...]:
    try:
        completed = subprocess.run(
            ["/usr/sbin/lsof", "-nP", f"-tiTCP:{port}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    values: list[int] = []
    for line in completed.stdout.splitlines():
        if line.strip().isdigit():
            values.append(int(line.strip()))
    return tuple(values)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _pid_command(pid: int) -> str:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def _pid_cwd(pid: int) -> Path | None:
    try:
        completed = subprocess.run(
            ["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("n") and len(line) > 1:
            return _safe_resolve(Path(line[1:]))
    return None


def _http_ready(port: int) -> bool:
    if not 1 <= port <= 65535:
        return False
    request = Request(
        f"http://{LOCAL_HOST}:{port}/_stcore/health",
        headers={"User-Agent": "AmazonReviewAnalyzer-Launcher/16.5"},
    )
    try:
        with urlopen(request, timeout=1.0) as response:
            payload = response.read(64).decode("utf-8", errors="replace").casefold()
            return response.status == 200 and "ok" in payload
    except (OSError, URLError, TimeoutError, UnicodeError):
        return False


def _default_dialog(
    title: str,
    message: str,
    cancel_button: str,
    action_button: str,
) -> str:
    script = """
on run argv
    try
        set resultDialog to display dialog (item 2 of argv) with title (item 1 of argv) buttons {(item 3 of argv), (item 4 of argv)} default button (item 4 of argv) cancel button (item 3 of argv) with icon caution
        return button returned of resultDialog
    on error number -128
        return item 3 of argv
    end try
end run
"""
    try:
        completed = subprocess.run(
            [
                "/usr/bin/osascript",
                "-e",
                script,
                title,
                message,
                cancel_button,
                action_button,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return cancel_button
    return completed.stdout.strip() or cancel_button


def _default_open_url(url: str) -> bool:
    if not re.fullmatch(r"http://localhost:\d{1,5}", url):
        return False
    try:
        completed = subprocess.run(
            ["/usr/bin/open", url],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _default_open_path(path: Path) -> bool:
    try:
        completed = subprocess.run(
            ["/usr/bin/open", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _default_open_ollama() -> bool:
    try:
        completed = subprocess.run(
            ["/usr/bin/open", "-a", "Ollama"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _default_spawn(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
) -> ProcessHandle:
    return subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        shell=False,
    )


def _default_health_runner(project_root: Path, port: int) -> Mapping[str, Any]:
    # Imported only after the project and its virtual environment are known.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from environment_health import (  # pylint: disable=import-outside-toplevel
        discover_health_settings,
        run_environment_health,
    )

    settings = replace(discover_health_settings(project_root), streamlit_port=port)
    return run_environment_health(
        project_root=project_root,
        settings=settings,
    ).to_dict()


@dataclass(frozen=True)
class LauncherProbes:
    """Injectable operating-system boundary used by offline tests."""

    listener_pids: Callable[[int], tuple[int, ...]] = _listener_pids
    pid_alive: Callable[[int], bool] = _pid_alive
    pid_command: Callable[[int], str] = _pid_command
    pid_cwd: Callable[[int], Path | None] = _pid_cwd
    http_ready: Callable[[int], bool] = _http_ready
    open_url: Callable[[str], bool] = _default_open_url
    open_path: Callable[[Path], bool] = _default_open_path
    open_ollama: Callable[[], bool] = _default_open_ollama
    dialog: Callable[[str, str, str, str], str] = _default_dialog
    spawn: Callable[[Sequence[str], Path, Mapping[str, str]], ProcessHandle] = _default_spawn
    health_runner: Callable[[Path, int], Mapping[str, Any]] = _default_health_runner
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic


def _process_matches_project(
    pid: int,
    port: int,
    project_root: Path,
    probes: LauncherProbes,
) -> bool:
    if pid not in probes.listener_pids(port) or not probes.pid_alive(pid):
        return False
    command = probes.pid_command(pid).casefold()
    if "streamlit" not in command or "app.py" not in command:
        return False
    cwd = probes.pid_cwd(pid)
    if cwd is None or _safe_resolve(cwd) != project_root.resolve():
        return False
    return probes.http_ready(port)


def find_existing_instance(
    project_root: str | Path,
    *,
    port: int = DEFAULT_PORT,
    probes: LauncherProbes | None = None,
) -> tuple[int, int] | None:
    root = Path(project_root).resolve()
    system = probes or LauncherProbes()
    metadata = load_runtime_metadata(root)
    if metadata is not None:
        pid = int(metadata["streamlit_pid"])
        saved_port = int(metadata["port"])
        if _process_matches_project(pid, saved_port, root, system):
            return pid, saved_port
    for pid in system.listener_pids(port):
        if _process_matches_project(pid, port, root, system):
            return pid, port
    return None


def _safe_check(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    checks = report.get("checks", {})
    if not isinstance(checks, Mapping):
        return {}
    value = checks.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _core_environment_ready(report: Mapping[str, Any]) -> bool:
    allowed = {"ready", "warning"}
    for name in ("python", "venv", "dependencies", "filesystem"):
        if str(_safe_check(report, name).get("status") or "") not in allowed:
            return False
    capabilities = report.get("capabilities", {})
    return isinstance(capabilities, Mapping) and capabilities.get("dashboard") is True


def _new_analysis_ready(report: Mapping[str, Any]) -> bool:
    capabilities = report.get("capabilities", {})
    return isinstance(capabilities, Mapping) and capabilities.get("new_analysis") is True


def _ollama_can_be_started(report: Mapping[str, Any]) -> bool:
    details = _safe_check(report, "ollama").get("details", {})
    return bool(
        isinstance(details, Mapping)
        and details.get("installed") is True
        and details.get("running") is not True
    )


def _dialog_text(code: str) -> tuple[str, str, str, str]:
    title = APP_NAME
    dialogs = {
        "project_not_found": (
            "Amazon Review Analyzer could not locate its application files.\n\n"
            "Please keep the application with the project folder or run Setup again.\n\n"
            "Amazon Review Analyzer 无法找到应用程序文件。\n\n"
            "请确认应用与项目目录仍在原位置，或重新运行安装程序。",
            "Cancel / 取消",
            "OK / 确定",
        ),
        "venv_missing": (
            "Python environment is not ready.\n\n"
            "Run Setup to prepare Amazon Review Analyzer.\n\n"
            "Python 环境尚未就绪。\n\n"
            "请运行 Setup 以准备 Amazon Review Analyzer。",
            "Cancel / 取消",
            "Run Setup / 运行设置",
        ),
        "setup_required": (
            "The local application environment needs attention.\n\n"
            "Run Setup to prepare Amazon Review Analyzer.\n\n"
            "本地应用环境需要处理。\n\n"
            "请运行 Setup 以准备 Amazon Review Analyzer。",
            "Cancel / 取消",
            "Run Setup / 运行设置",
        ),
        "ollama_missing": (
            "Local AI engine is not available.\n\n"
            "You can still open Amazon Review Analyzer in Limited Mode.\n\n"
            "本地 AI 引擎当前不可用。\n\n"
            "你仍可使用有限模式打开 Amazon Review Analyzer。",
            "Cancel / 取消",
            "Open Limited Mode / 打开有限模式",
        ),
        "model_missing": (
            "The qwen3:8b model is not available.\n\n"
            "Existing analyses, Demo, reports and comparisons can still be viewed.\n\n"
            "qwen3:8b 模型当前不可用。\n\n"
            "已有分析、演示、报告和竞品对比仍可查看。",
            "Cancel / 取消",
            "Open Limited Mode / 打开有限模式",
        ),
        "port_conflict": (
            "Port 8501 is currently being used by another application.\n\n"
            "Amazon Review Analyzer did not stop or change that application.\n\n"
            "端口 8501 当前正被其他应用使用。\n\n"
            "Amazon Review Analyzer 未停止或更改该应用。",
            "Cancel / 取消",
            "OK / 确定",
        ),
        "start_failed": (
            "Amazon Review Analyzer could not start the local dashboard.\n\n"
            "Run Setup or check System Status, then try again.\n\n"
            "Amazon Review Analyzer 无法启动本地 Dashboard。\n\n"
            "请运行 Setup 或检查系统状态后重试。",
            "Cancel / 取消",
            "OK / 确定",
        ),
        "nothing_to_stop": (
            "No safely identified Amazon Review Analyzer dashboard is running.\n\n"
            "当前没有可安全识别并停止的 Amazon Review Analyzer Dashboard。",
            "Cancel / 取消",
            "OK / 确定",
        ),
    }
    message, cancel, action = dialogs[code]
    return title, message, cancel, action


def _ask(code: str, probes: LauncherProbes) -> bool:
    title, message, cancel, action = _dialog_text(code)
    return probes.dialog(title, message, cancel, action) == action


def _console(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def _spawn_command(project_root: Path, port: int) -> list[str]:
    python = project_root / ".venv" / "bin" / "python"
    return [
        str(python),
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address",
        LOCAL_HOST,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def _limited_dialog_code(report: Mapping[str, Any]) -> str:
    model = _safe_check(report, "model")
    if model.get("code") == "model_missing":
        return "model_missing"
    return "ollama_missing"


def launch_dashboard(
    project_root: str | Path,
    *,
    native: bool,
    foreground: bool,
    probes: LauncherProbes | None = None,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> int:
    """Launch or reopen exactly one project Streamlit instance."""

    root = Path(project_root).resolve()
    system = probes or LauncherProbes()
    console = not native
    start_clock = system.monotonic()
    runtime_dir = _runtime_directory(root)
    lock_path = runtime_dir / "launch.lock"

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = find_existing_instance(root, port=port, probes=system)
        if existing is not None:
            pid, actual_port = existing
            url = f"http://localhost:{actual_port}"
            if open_browser:
                system.open_url(url)
            append_launcher_log(
                root,
                stage="existing_instance",
                status="reused",
                pid=pid,
                port=actual_port,
                elapsed_ms=int((system.monotonic() - start_clock) * 1000),
            )
            _console(f"{APP_NAME} is already running at {url}", enabled=console)
            return 0

        listeners = system.listener_pids(port)
        if listeners:
            append_launcher_log(
                root,
                stage="streamlit_start",
                status="failed",
                error_code="port_in_use",
                port=port,
            )
            if native:
                _ask("port_conflict", system)
            else:
                _console("Port 8501 is currently in use by another application.", enabled=True)
            return 3

        try:
            report = system.health_runner(root, port)
        except Exception:
            report = {}
        if not _core_environment_ready(report):
            append_launcher_log(
                root,
                stage="health",
                status="failed",
                error_code="setup_required",
                port=port,
            )
            setup_path = root / "setup_amazon_review_analyzer.command"
            if native and _ask("setup_required", system):
                system.open_path(setup_path)
            else:
                _console("Run setup_amazon_review_analyzer.command, then try again.", enabled=console)
            return 4

        if not _new_analysis_ready(report) and _ollama_can_be_started(report):
            append_launcher_log(
                root,
                stage="ollama_start",
                status="started",
                port=port,
            )
            system.open_ollama()
            for _ in range(OLLAMA_START_ATTEMPTS):
                system.sleep(OLLAMA_START_DELAY_SECONDS)
                try:
                    report = system.health_runner(root, port)
                except Exception:
                    continue
                if _new_analysis_ready(report):
                    break

        health_status = "ready" if _new_analysis_ready(report) else "limited"
        if health_status == "limited":
            if native and not _ask(_limited_dialog_code(report), system):
                append_launcher_log(
                    root,
                    stage="health",
                    status="cancelled",
                    error_code="limited_mode_cancelled",
                    port=port,
                )
                return 5
            _console(
                "Local AI unavailable — viewing features remain available.",
                enabled=console,
            )

        environment = dict(os.environ)
        environment["ARA_STREAMLIT_PORT"] = str(port)
        environment["ARA_LAUNCHER_MODE"] = "native" if native else "command"
        command = _spawn_command(root, port)
        try:
            process = system.spawn(command, root, environment)
        except Exception:
            append_launcher_log(
                root,
                stage="streamlit_start",
                status="failed",
                error_code="spawn_failed",
                port=port,
            )
            if native:
                _ask("start_failed", system)
            else:
                _console("The dashboard could not be started.", enabled=True)
            return 6
        append_launcher_log(
            root,
            stage="streamlit_start",
            status="started",
            pid=process.pid,
            port=port,
        )

        ready = False
        for _ in range(READY_ATTEMPTS):
            if system.http_ready(port):
                ready = True
                break
            if process.poll() is not None:
                break
            system.sleep(READY_DELAY_SECONDS)
        if not ready:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
            append_launcher_log(
                root,
                stage="streamlit_ready",
                status="failed",
                error_code="readiness_timeout",
                pid=process.pid,
                port=port,
            )
            if native:
                _ask("start_failed", system)
            else:
                _console("The dashboard could not be started.", enabled=True)
            return 7

        mode = "native" if native else "command"
        write_runtime_metadata(
            root,
            pid=process.pid,
            port=port,
            launcher_mode=mode,
            health_status=health_status,
        )
        append_launcher_log(
            root,
            stage="streamlit_ready",
            status=health_status,
            pid=process.pid,
            port=port,
            elapsed_ms=int((system.monotonic() - start_clock) * 1000),
        )
        url = f"http://localhost:{port}"
        if open_browser:
            system.open_url(url)
            append_launcher_log(
                root,
                stage="browser_open",
                status="ready",
                pid=process.pid,
                port=port,
            )
        _console(f"Dashboard ready: {url}", enabled=console)

    if not foreground:
        return 0

    _console("Press Ctrl+C to stop the dashboard.", enabled=console)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except Exception:
            pass
        return_code = 0
    append_launcher_log(
        root,
        stage="stop",
        status="stopped",
        pid=process.pid,
        port=port,
    )
    _console(f"{APP_NAME} stopped.", enabled=console)
    return 0 if return_code in {0, 130, 143, -signal.SIGINT} else int(return_code)


def stop_dashboard(
    project_root: str | Path,
    *,
    native: bool,
    probes: LauncherProbes | None = None,
) -> int:
    """Stop only the PID verified by this project's runtime metadata."""

    root = Path(project_root).resolve()
    system = probes or LauncherProbes()
    metadata = load_runtime_metadata(root)
    if metadata is None:
        if native:
            _ask("nothing_to_stop", system)
        else:
            print("No safely identified dashboard is running.")
        return 1
    pid = int(metadata["streamlit_pid"])
    port = int(metadata["port"])
    if not _process_matches_project(pid, port, root, system):
        if native:
            _ask("nothing_to_stop", system)
        else:
            print("No safely identified dashboard is running.")
        return 1
    try:
        os.kill(pid, signal.SIGINT)
    except OSError:
        return 1
    append_launcher_log(root, stage="stop", status="stopped", pid=pid, port=port)
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Amazon Review Analyzer on macOS.")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--stop-dashboard", action="store_true")
    parser.add_argument("--remember-project", action="store_true")
    parser.add_argument("--app-bundle")
    parser.add_argument("--project-root")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def _show_preproject_dialog(code: str) -> None:
    title, message, cancel, action = _dialog_text(code)
    _default_dialog(title, message, cancel, action)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = discover_project_root(
        app_bundle=args.app_bundle,
        project_override=args.project_root,
    )
    if root is None:
        if args.native:
            _show_preproject_dialog("project_not_found")
        else:
            print("Amazon Review Analyzer project folder was not found.")
        return 2

    if args.remember_project:
        return 0 if remember_project_root(root) else 1

    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.is_file() or not os.access(venv_python, os.X_OK):
        setup = root / "setup_amazon_review_analyzer.command"
        if args.native:
            if _ask("venv_missing", LauncherProbes()):
                _default_open_path(setup)
        else:
            print("Python environment is not ready. Run Setup first.")
        return 2

    # The app bootstrap may initially use the system Python.  Re-exec exactly
    # once into the project environment so existing health checks report the
    # real virtual environment and package set.
    try:
        current = Path(sys.executable).resolve()
        expected = venv_python.resolve()
    except OSError:
        current = Path(sys.executable)
        expected = venv_python
    if current != expected and os.getenv("ARA_LAUNCHER_NO_REEXEC") != "1":
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *(argv or sys.argv[1:])])

    remember_project_root(root)
    if args.stop_dashboard:
        return stop_dashboard(root, native=args.native)
    return launch_dashboard(
        root,
        native=args.native,
        foreground=args.foreground,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    raise SystemExit(main())

