"""Fast, inference-free environment checks for first-run and support flows.

This module intentionally imports only the Python standard library.  It must
remain usable when a project dependency is missing, because explaining that
condition is one of its primary responsibilities.  The checks never start a
model, download software, access Amazon, or write business data.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


HEALTH_SCHEMA_VERSION = "environment-health.v1"
PROJECT_ROOT = Path(__file__).resolve().parent
MIN_PYTHON_VERSION = (3, 11)
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_STREAMLIT_PORT = 8501
DEFAULT_RUNTIME_MODE = "stable"
DEFAULT_RUNTIME_WORKERS = 1
LOW_DISK_BYTES = 1024**3
HTTP_TIMEOUT_SECONDS = 1.5
MAX_HTTP_RESPONSE_BYTES = 1024 * 1024

# These are feature classifications, not a second dependency list.  The
# dependency names themselves are always read from requirements.txt.
OPTIONAL_PROVIDER_DISTRIBUTIONS = frozenset({"openai"})
DEVELOPMENT_DISTRIBUTIONS = frozenset({"pytest"})
OPTIONAL_FEATURE_DISTRIBUTIONS = frozenset({"ollama", "playwright"})

RUNTIME_DIRECTORY_NAMES = (
    "outputs",
    "outputs/session",
    "outputs/session/current",
    "outputs/amazon",
    "outputs/amazon/current",
    "outputs/library",
    "outputs/reports",
    "outputs/comparison",
    "outputs/comparison/current",
    "outputs/app_state",
)

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")
_VERSION_NUMBER = re.compile(r"\b(\d+(?:\.\d+){1,3}(?:[-+._A-Za-z0-9]*)?)\b")
_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}$")
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential|password|secret|token)",
    re.IGNORECASE,
)
_MAC_USER_PATH = re.compile(r"/Users/[^\s,;:'\"]+")
_LINUX_USER_PATH = re.compile(r"/home/[^\s,;:'\"]+")
_WINDOWS_USER_PATH = re.compile(
    r"[A-Za-z]:\\Users\\[^\s,;:'\"]+", re.IGNORECASE
)
_URL_CREDENTIALS = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)


class CheckStatus(str, Enum):
    """Stable per-check states used by the CLI and Streamlit presentation."""

    READY = "ready"
    WARNING = "warning"
    ACTION_REQUIRED = "action_required"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class OverallStatus(str, Enum):
    """User-facing operating modes derived from individual checks."""

    READY = "ready"
    LIMITED = "limited"
    ACTION_REQUIRED = "action_required"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class HealthSettings:
    """Safe runtime settings needed by environment checks."""

    app_version: str = "unknown"
    minimum_python: tuple[int, int] = MIN_PYTHON_VERSION
    ollama_host: str = DEFAULT_OLLAMA_HOST
    model_name: str = DEFAULT_OLLAMA_MODEL
    streamlit_port: int = DEFAULT_STREAMLIT_PORT
    runtime_mode: str = DEFAULT_RUNTIME_MODE
    runtime_workers: int = DEFAULT_RUNTIME_WORKERS
    low_disk_bytes: int = LOW_DISK_BYTES
    http_timeout_seconds: float = HTTP_TIMEOUT_SECONDS


def _default_package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _default_which(command: str) -> str | None:
    return shutil.which(command)


def _default_run_command(
    command: Sequence[str], timeout_seconds: float
) -> tuple[int, str]:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return completed.returncode, completed.stdout or completed.stderr or ""


def _default_http_get_json(url: str, timeout_seconds: float) -> Mapping[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "AmazonReviewAnalyzer-EnvironmentHealth/1.0",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
    if len(raw) > MAX_HTTP_RESPONSE_BYTES:
        raise ValueError("response_too_large")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("response_not_object")
    return payload


def _default_disk_usage(path: Path) -> Any:
    return shutil.disk_usage(path)


def _default_write_probe(directory: Path) -> None:
    """Create and remove one private temporary file, never a business file."""

    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".environment-health-",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as handle:
            handle.write(b"ok")
            probe_path = Path(handle.name)
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.25)
        return client.connect_ex(("127.0.0.1", port)) == 0


def _listener_pids(port: int) -> tuple[str, ...]:
    lsof = "/usr/sbin/lsof" if Path("/usr/sbin/lsof").is_file() else shutil.which("lsof")
    if not lsof:
        return ()
    try:
        completed = subprocess.run(
            [lsof, "-nP", f"-tiTCP:{port}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip().isdigit())


def _pid_cwd(pid: str) -> Path | None:
    lsof = "/usr/sbin/lsof" if Path("/usr/sbin/lsof").is_file() else shutil.which("lsof")
    if not lsof:
        return None
    try:
        completed = subprocess.run(
            [lsof, "-a", "-p", pid, "-d", "cwd", "-Fn"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("n") and len(line) > 1:
            try:
                return Path(line[1:]).resolve()
            except OSError:
                return None
    return None


def _pid_is_streamlit_app(pid: str) -> bool:
    ps = "/bin/ps" if Path("/bin/ps").is_file() else shutil.which("ps")
    if not ps:
        return False
    try:
        completed = subprocess.run(
            [ps, "-p", pid, "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    command = completed.stdout.casefold()
    return "streamlit" in command and "app.py" in command


def _default_port_probe(project_root: Path, port: int) -> str:
    """Return free/current_app/other_process/unknown without killing anything."""

    try:
        if not _port_is_open(port):
            return "free"
    except OSError:
        return "unknown"

    try:
        expected_cwd = project_root.resolve()
    except OSError:
        return "other_process"
    pids = _listener_pids(port)
    if not pids:
        return "other_process"
    for pid in pids:
        if _pid_cwd(pid) == expected_cwd and _pid_is_streamlit_app(pid):
            return "current_app"
    return "other_process"


@dataclass(frozen=True)
class HealthProbes:
    """Injectable side-effect boundary used by deterministic offline tests."""

    package_version: Callable[[str], str] = _default_package_version
    which: Callable[[str], str | None] = _default_which
    run_command: Callable[[Sequence[str], float], tuple[int, str]] = _default_run_command
    http_get_json: Callable[[str, float], Mapping[str, Any]] = _default_http_get_json
    port_probe: Callable[[Path, int], str] = _default_port_probe
    disk_usage: Callable[[Path], Any] = _default_disk_usage
    write_probe: Callable[[Path], None] = _default_write_probe


def _sanitize_text(value: str) -> str:
    sanitized = _URL_CREDENTIALS.sub(r"\1[redacted]@", value)
    sanitized = _MAC_USER_PATH.sub("<path>", sanitized)
    sanitized = _LINUX_USER_PATH.sub("<path>", sanitized)
    sanitized = _WINDOWS_USER_PATH.sub("<path>", sanitized)
    return sanitized


def sanitize_diagnostics(value: Any, *, key: str = "") -> Any:
    """Recursively allow diagnostics while redacting common sensitive data."""

    if key and _SENSITIVE_KEY.search(key):
        return "[redacted]"
    if isinstance(value, Mapping):
        return {
            str(item_key): sanitize_diagnostics(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_diagnostics(item) for item in value]
    if isinstance(value, Path):
        return "<path>"
    if isinstance(value, str):
        return _sanitize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_text(str(value))


@dataclass(frozen=True)
class HealthCheck:
    """One concise check with stable machine and human representations."""

    code: str
    status: CheckStatus
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_diagnostics(
            {
                "code": self.code,
                "status": self.status.value,
                "message": self.message,
                "details": dict(self.details),
            }
        )


@dataclass(frozen=True)
class EnvironmentHealthReport:
    """Complete environment state consumed by CLI and UI presentation layers."""

    app_version: str
    overall_status: OverallStatus
    checks: Mapping[str, HealthCheck]
    capabilities: Mapping[str, bool]
    runtime_mode: str
    runtime_workers: int

    def to_dict(self) -> dict[str, Any]:
        return sanitize_diagnostics(
            {
                "schema_version": HEALTH_SCHEMA_VERSION,
                "app_version": self.app_version,
                "overall_status": self.overall_status.value,
                "checks": {
                    name: check.to_dict() for name, check in self.checks.items()
                },
                "capabilities": dict(self.capabilities),
                "runtime": {
                    "mode": self.runtime_mode,
                    "workers": self.runtime_workers,
                },
            }
        )


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def load_requirement_names(path: str | Path) -> tuple[str, ...]:
    """Read distribution names from the project's authoritative requirements."""

    requirement_path = Path(path)
    text = requirement_path.read_text(encoding="utf-8")
    names: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = _REQUIREMENT_NAME.match(line)
        if match is None:
            continue
        name = _canonical_distribution(match.group(1))
        if name not in seen:
            names.append(name)
            seen.add(name)
    return tuple(names)


def _read_env_settings(path: Path) -> dict[str, str]:
    """Read only the two Ollama keys without importing python-dotenv."""

    allowed = {"OLLAMA_HOST", "OLLAMA_MODEL"}
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if name not in allowed:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value:
            values[name] = value
    return values


def discover_health_settings(
    project_root: str | Path = PROJECT_ROOT,
    *,
    environment: Mapping[str, str] | None = None,
) -> HealthSettings:
    """Discover safe health settings without importing the application config."""

    root = Path(project_root)
    manifest: Mapping[str, Any] = {}
    for manifest_name in ("release_manifest_v1.0.1.json", "release_manifest.json"):
        try:
            payload = json.loads((root / manifest_name).read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                manifest = payload
                break
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue

    env_file = _read_env_settings(root / ".env")
    current_env = os.environ if environment is None else environment
    host = (
        str(current_env.get("OLLAMA_HOST", "")).strip()
        or env_file.get("OLLAMA_HOST", "")
        or DEFAULT_OLLAMA_HOST
    )
    model = (
        str(current_env.get("OLLAMA_MODEL", "")).strip()
        or env_file.get("OLLAMA_MODEL", "")
        or str(manifest.get("model_name", "")).strip()
        or DEFAULT_OLLAMA_MODEL
    )
    raw_streamlit_port = str(
        current_env.get("ARA_STREAMLIT_PORT", "")
    ).strip()
    try:
        streamlit_port = int(raw_streamlit_port, 10)
    except (TypeError, ValueError):
        streamlit_port = DEFAULT_STREAMLIT_PORT
    if not 1 <= streamlit_port <= 65_535:
        streamlit_port = DEFAULT_STREAMLIT_PORT
    workers = manifest.get("default_workers", DEFAULT_RUNTIME_WORKERS)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        workers = DEFAULT_RUNTIME_WORKERS
    mode = str(manifest.get("default_runtime_mode", DEFAULT_RUNTIME_MODE)).strip()
    return HealthSettings(
        app_version=str(manifest.get("app_version", "unknown")).strip() or "unknown",
        ollama_host=host,
        model_name=model,
        streamlit_port=streamlit_port,
        runtime_mode=mode or DEFAULT_RUNTIME_MODE,
        runtime_workers=workers,
    )


def _safe_model_name(model_name: str) -> str:
    return model_name if _SAFE_MODEL_NAME.fullmatch(model_name) else "configured model"


def _normalized_ollama_host(host: str) -> tuple[str, str]:
    """Return request/display hosts without credentials, query, or fragments."""

    parsed = urlsplit(host.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("invalid_host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentialed_host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid_port") from exc
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname + (f":{port}" if port is not None else "")
    clean = urlunsplit((parsed.scheme, netloc, "", "", "")).rstrip("/")
    return clean, clean


def check_python(
    settings: HealthSettings,
    *,
    version_info: Sequence[int] | None = None,
    version_text: str | None = None,
    executable: str | Path | None = None,
) -> HealthCheck:
    actual = tuple(version_info or sys.version_info[:3])
    displayed = version_text or ".".join(str(part) for part in actual[:3])
    required = f"{settings.minimum_python[0]}.{settings.minimum_python[1]}+"
    executable_value = Path(executable or sys.executable)
    executable_available = executable_value.is_file() and os.access(executable_value, os.X_OK)
    if not executable_available:
        return HealthCheck(
            "python_executable_unavailable",
            CheckStatus.UNAVAILABLE,
            "The Python executable is unavailable.",
            {"version": displayed, "required": required, "executable_available": False},
        )
    if actual[:2] < settings.minimum_python:
        return HealthCheck(
            "python_version_unsupported",
            CheckStatus.UNAVAILABLE,
            "Python version is not supported.",
            {"version": displayed, "required": required, "executable_available": True},
        )
    return HealthCheck(
        "python_ready",
        CheckStatus.READY,
        f"Python {displayed} is ready.",
        {"version": displayed, "required": required, "executable_available": True},
    )


def check_virtual_environment(
    project_root: str | Path,
    *,
    current_prefix: str | Path | None = None,
    base_prefix: str | Path | None = None,
) -> HealthCheck:
    root = Path(project_root)
    expected = root / ".venv"
    python_candidates = (expected / "bin" / "python", expected / "Scripts" / "python.exe")
    installed = expected.is_dir() and any(path.is_file() for path in python_candidates)
    prefix = Path(current_prefix or sys.prefix)
    base = Path(base_prefix or sys.base_prefix)
    try:
        active = installed and prefix.resolve() == expected.resolve() and prefix.resolve() != base.resolve()
    except OSError:
        active = False
    if not installed:
        return HealthCheck(
            "venv_missing",
            CheckStatus.ACTION_REQUIRED,
            "Virtual environment is not ready. Run Setup to create it.",
            {"installed": False, "active": False},
        )
    if not active:
        return HealthCheck(
            "venv_inactive",
            CheckStatus.ACTION_REQUIRED,
            "The project virtual environment is not active.",
            {"installed": True, "active": False},
        )
    return HealthCheck(
        "venv_ready",
        CheckStatus.READY,
        "Virtual environment is ready.",
        {"installed": True, "active": True},
    )


def check_dependencies(
    project_root: str | Path,
    *,
    package_version: Callable[[str], str] = _default_package_version,
) -> HealthCheck:
    try:
        requirements = load_requirement_names(Path(project_root) / "requirements.txt")
    except (OSError, UnicodeError):
        return HealthCheck(
            "requirements_unavailable",
            CheckStatus.UNAVAILABLE,
            "The project dependency definition is unavailable.",
            {"required_count": 0, "installed_count": 0},
        )
    if not requirements:
        return HealthCheck(
            "requirements_empty",
            CheckStatus.UNAVAILABLE,
            "The project dependency definition is empty.",
            {"required_count": 0, "installed_count": 0},
        )

    versions: dict[str, str] = {}
    missing: list[str] = []
    for distribution in requirements:
        try:
            versions[distribution] = str(package_version(distribution))
        except Exception:
            missing.append(distribution)

    optional_names = (
        OPTIONAL_PROVIDER_DISTRIBUTIONS
        | DEVELOPMENT_DISTRIBUTIONS
        | OPTIONAL_FEATURE_DISTRIBUTIONS
    )
    missing_optional = sorted(name for name in missing if name in optional_names)
    missing_required = sorted(name for name in missing if name not in optional_names)
    missing_features = sorted(
        name for name in missing if name in OPTIONAL_FEATURE_DISTRIBUTIONS
    )
    details = {
        "required_count": len(requirements),
        "installed_count": len(versions),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "missing_features": missing_features,
        "versions": dict(sorted(versions.items())),
    }
    if missing_required:
        return HealthCheck(
            "dependencies_missing",
            CheckStatus.UNAVAILABLE,
            "Some required Python packages are missing. Run Setup to install them.",
            details,
        )
    if missing_features:
        return HealthCheck(
            "optional_dependencies_missing",
            CheckStatus.WARNING,
            "Core dependencies are ready; some optional features are unavailable.",
            details,
        )
    return HealthCheck(
        "dependencies_ready",
        CheckStatus.READY,
        "Dependencies are ready.",
        details,
    )


def _extract_version(output: str) -> str:
    match = _VERSION_NUMBER.search(output)
    return match.group(1) if match else "unknown"


def _ollama_error_code(exc: BaseException) -> str:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "ollama_timeout"
    if isinstance(exc, ConnectionRefusedError):
        return "ollama_not_running"
    if isinstance(exc, URLError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "ollama_timeout"
        if isinstance(reason, ConnectionRefusedError):
            return "ollama_not_running"
    if isinstance(exc, (json.JSONDecodeError, UnicodeError, ValueError, TypeError)):
        return "ollama_response_invalid"
    return "ollama_unavailable"


def _model_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("models_missing")
    return [item for item in raw_models if isinstance(item, Mapping)]


def check_ollama_and_model(
    settings: HealthSettings,
    *,
    which: Callable[[str], str | None] = _default_which,
    run_command: Callable[[Sequence[str], float], tuple[int, str]] = _default_run_command,
    http_get_json: Callable[[str, float], Mapping[str, Any]] = _default_http_get_json,
) -> tuple[HealthCheck, HealthCheck]:
    executable = which("ollama")
    cli_version = "unknown"
    if executable:
        try:
            return_code, output = run_command((executable, "--version"), 2.0)
            if return_code == 0:
                cli_version = _extract_version(output)
        except Exception:
            pass

    try:
        request_host, display_host = _normalized_ollama_host(settings.ollama_host)
    except ValueError:
        ollama = HealthCheck(
            "ollama_host_invalid",
            CheckStatus.ACTION_REQUIRED,
            "The local AI host configuration is invalid.",
            {"installed": bool(executable), "running": False, "version": cli_version},
        )
        model = HealthCheck(
            "model_unavailable",
            CheckStatus.UNAVAILABLE,
            "The AI model cannot be checked until Ollama is available.",
            {"model": _safe_model_name(settings.model_name), "installed": False},
        )
        return ollama, model

    try:
        tags = http_get_json(
            f"{request_host}/api/tags", settings.http_timeout_seconds
        )
        models = _model_records(tags)
    except Exception as exc:
        error_code = _ollama_error_code(exc)
        if not executable:
            code = "ollama_not_installed"
            message = "Ollama is not installed or cannot be found."
        elif error_code == "ollama_timeout":
            code = error_code
            message = "Ollama is installed, but the local service timed out."
        elif error_code == "ollama_response_invalid":
            code = error_code
            message = "Ollama returned an invalid health response."
        else:
            code = "ollama_not_running" if error_code == "ollama_not_running" else error_code
            message = "Ollama is installed but not running."
        ollama = HealthCheck(
            code,
            CheckStatus.ACTION_REQUIRED,
            message,
            {
                "installed": bool(executable),
                "running": False,
                "version": cli_version,
                "host": display_host,
            },
        )
        model = HealthCheck(
            "model_unavailable",
            CheckStatus.UNAVAILABLE,
            "The AI model cannot be checked until Ollama is available.",
            {"model": _safe_model_name(settings.model_name), "installed": False},
        )
        return ollama, model

    server_version = cli_version
    try:
        version_payload = http_get_json(
            f"{request_host}/api/version", settings.http_timeout_seconds
        )
        candidate = version_payload.get("version")
        if isinstance(candidate, str) and candidate.strip():
            server_version = _extract_version(candidate) or "unknown"
    except Exception:
        pass

    ollama_status = CheckStatus.READY if executable else CheckStatus.WARNING
    ollama_code = "ollama_ready" if executable else "ollama_executable_missing"
    ollama_message = (
        "Ollama is running."
        if executable
        else "Ollama is running, but the Ollama command is unavailable."
    )
    ollama = HealthCheck(
        ollama_code,
        ollama_status,
        ollama_message,
        {
            "installed": bool(executable),
            "running": True,
            "version": server_version,
            "host": display_host,
        },
    )

    configured_model = settings.model_name
    matched: Mapping[str, Any] | None = None
    for record in models:
        candidate = record.get("name") or record.get("model")
        if candidate == configured_model:
            matched = record
            break
    model_details: dict[str, Any] = {
        "model": _safe_model_name(configured_model),
        "installed": matched is not None,
    }
    if matched is not None:
        size = matched.get("size")
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            model_details["size_bytes"] = size
        model = HealthCheck(
            "model_ready",
            CheckStatus.READY,
            f"AI model {_safe_model_name(configured_model)} is ready.",
            model_details,
        )
    else:
        model = HealthCheck(
            "model_missing",
            CheckStatus.ACTION_REQUIRED,
            f"AI model {_safe_model_name(configured_model)} is not installed.",
            model_details,
        )
    return ollama, model


def check_port(
    project_root: str | Path,
    settings: HealthSettings,
    *,
    port_probe: Callable[[Path, int], str] = _default_port_probe,
) -> HealthCheck:
    try:
        state = port_probe(Path(project_root), settings.streamlit_port)
    except Exception:
        state = "unknown"
    details = {"port": settings.streamlit_port, "state": state}
    if state == "free":
        return HealthCheck(
            "port_ready", CheckStatus.READY, "Dashboard port is available.", details
        )
    if state == "current_app":
        return HealthCheck(
            "app_already_running",
            CheckStatus.READY,
            "Amazon Review Analyzer is already running.",
            details,
        )
    if state == "other_process":
        return HealthCheck(
            "port_in_use",
            CheckStatus.ACTION_REQUIRED,
            f"Port {settings.streamlit_port} is currently in use by another application.",
            details,
        )
    return HealthCheck(
        "port_unknown",
        CheckStatus.UNKNOWN,
        "Dashboard port status could not be determined.",
        {"port": settings.streamlit_port, "state": "unknown"},
    )


def _inside_root(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


def check_filesystem(
    project_root: str | Path,
    *,
    write_probe: Callable[[Path], None] = _default_write_probe,
    create_directories: bool = True,
) -> HealthCheck:
    try:
        root = Path(project_root).resolve()
    except OSError:
        return HealthCheck(
            "filesystem_unavailable",
            CheckStatus.UNAVAILABLE,
            "Project storage is unavailable.",
            {"writable": False, "directory_count": len(RUNTIME_DIRECTORY_NAMES)},
        )

    targets: list[Path] = []
    for relative_name in RUNTIME_DIRECTORY_NAMES:
        try:
            target = (root / relative_name).resolve()
        except OSError:
            return HealthCheck(
                "filesystem_unavailable",
                CheckStatus.UNAVAILABLE,
                "Project storage is unavailable.",
                {"writable": False, "directory_count": len(RUNTIME_DIRECTORY_NAMES)},
            )
        if not _inside_root(root, target):
            return HealthCheck(
                "filesystem_unsafe_path",
                CheckStatus.UNAVAILABLE,
                "A runtime directory points outside the project.",
                {"writable": False, "directory_count": len(RUNTIME_DIRECTORY_NAMES)},
            )
        targets.append(target)

    try:
        for target in targets:
            if target.exists():
                if not target.is_dir():
                    raise OSError("runtime_target_not_directory")
                continue
            if not create_directories:
                return HealthCheck(
                    "runtime_directories_missing",
                    CheckStatus.ACTION_REQUIRED,
                    "Runtime directories are not ready.",
                    {
                        "writable": False,
                        "directory_count": len(targets),
                    },
                )
            target.mkdir(parents=True, exist_ok=True)
        write_probe(targets[0])
    except Exception:
        return HealthCheck(
            "filesystem_not_writable",
            CheckStatus.UNAVAILABLE,
            "Project runtime storage is not writable.",
            {
                "writable": False,
                "directory_count": len(targets),
            },
        )

    return HealthCheck(
        "filesystem_ready",
        CheckStatus.READY,
        "Project runtime storage is ready.",
        {
            "writable": True,
            "directory_count": len(targets),
        },
    )


def check_disk_space(
    project_root: str | Path,
    settings: HealthSettings,
    *,
    disk_usage: Callable[[Path], Any] = _default_disk_usage,
) -> HealthCheck:
    try:
        usage = disk_usage(Path(project_root))
        free_bytes = int(usage.free)
    except Exception:
        return HealthCheck(
            "disk_unknown",
            CheckStatus.UNKNOWN,
            "Available disk space could not be determined.",
            {"available_gb": None},
        )
    available_gb = round(free_bytes / (1024**3), 1)
    details = {"available_gb": available_gb}
    if free_bytes < settings.low_disk_bytes:
        return HealthCheck(
            "disk_low",
            CheckStatus.WARNING,
            "Available disk space may be too low for local AI analysis.",
            details,
        )
    return HealthCheck(
        "disk_ready",
        CheckStatus.READY,
        "Available disk space is sufficient.",
        details,
    )


def _dependency_capabilities(check: HealthCheck) -> tuple[bool, bool, bool]:
    core_ready = check.status in {CheckStatus.READY, CheckStatus.WARNING}
    missing_features = set(check.details.get("missing_features", []))
    browser_ready = "playwright" not in missing_features
    inference_ready = "ollama" not in missing_features
    return core_ready, browser_ready, inference_ready


def _derive_capabilities(checks: Mapping[str, HealthCheck]) -> dict[str, bool]:
    dependency_ready, browser_ready, inference_dependency_ready = _dependency_capabilities(
        checks["dependencies"]
    )
    python_ready = checks["python"].status is CheckStatus.READY
    venv_ready = checks["venv"].status is CheckStatus.READY
    filesystem_ready = checks["filesystem"].status is CheckStatus.READY
    dashboard = python_ready and venv_ready and dependency_ready
    ollama_ready = bool(checks["ollama"].details.get("running"))
    model_ready = checks["model"].status is CheckStatus.READY
    new_analysis = (
        dashboard
        and filesystem_ready
        and inference_dependency_ready
        and ollama_ready
        and model_ready
    )
    amazon_import = dashboard and filesystem_ready and browser_ready
    return {
        "dashboard": dashboard,
        "demo": dashboard,
        "saved_analyses": dashboard,
        "reports": dashboard,
        "comparison": dashboard,
        "new_analysis": new_analysis,
        "amazon_import": amazon_import,
        "amazon_analysis": amazon_import and new_analysis,
    }


def _derive_overall_status(
    checks: Mapping[str, HealthCheck], capabilities: Mapping[str, bool]
) -> OverallStatus:
    if checks["python"].status is CheckStatus.UNAVAILABLE:
        return OverallStatus.UNAVAILABLE
    if checks["dependencies"].status is CheckStatus.UNAVAILABLE:
        return OverallStatus.UNAVAILABLE
    if checks["filesystem"].status is CheckStatus.UNAVAILABLE:
        return OverallStatus.UNAVAILABLE
    if checks["venv"].status is CheckStatus.ACTION_REQUIRED:
        return OverallStatus.ACTION_REQUIRED
    if checks["port"].status is CheckStatus.ACTION_REQUIRED:
        return OverallStatus.ACTION_REQUIRED
    if not capabilities["new_analysis"]:
        return OverallStatus.LIMITED if capabilities["dashboard"] else OverallStatus.ACTION_REQUIRED
    if any(
        check.status in {CheckStatus.WARNING, CheckStatus.UNKNOWN}
        for check in checks.values()
    ):
        return OverallStatus.LIMITED
    return OverallStatus.READY


def run_environment_health(
    *,
    project_root: str | Path = PROJECT_ROOT,
    settings: HealthSettings | None = None,
    probes: HealthProbes | None = None,
    create_directories: bool = True,
    python_version_info: Sequence[int] | None = None,
    python_version_text: str | None = None,
    python_executable: str | Path | None = None,
    current_prefix: str | Path | None = None,
    base_prefix: str | Path | None = None,
) -> EnvironmentHealthReport:
    """Run every lightweight environment check without model inference."""

    root = Path(project_root)
    resolved_settings = settings or discover_health_settings(root)
    resolved_probes = probes or HealthProbes()
    python_check = check_python(
        resolved_settings,
        version_info=python_version_info,
        version_text=python_version_text,
        executable=python_executable,
    )
    venv_check = check_virtual_environment(
        root,
        current_prefix=current_prefix,
        base_prefix=base_prefix,
    )
    dependency_check = check_dependencies(
        root, package_version=resolved_probes.package_version
    )
    ollama_check, model_check = check_ollama_and_model(
        resolved_settings,
        which=resolved_probes.which,
        run_command=resolved_probes.run_command,
        http_get_json=resolved_probes.http_get_json,
    )
    port_check = check_port(
        root, resolved_settings, port_probe=resolved_probes.port_probe
    )
    filesystem_check = check_filesystem(
        root,
        write_probe=resolved_probes.write_probe,
        create_directories=create_directories,
    )
    disk_check = check_disk_space(
        root, resolved_settings, disk_usage=resolved_probes.disk_usage
    )
    checks = {
        "python": python_check,
        "venv": venv_check,
        "dependencies": dependency_check,
        "ollama": ollama_check,
        "model": model_check,
        "port": port_check,
        "filesystem": filesystem_check,
        "disk": disk_check,
    }
    capabilities = _derive_capabilities(checks)
    return EnvironmentHealthReport(
        app_version=resolved_settings.app_version,
        overall_status=_derive_overall_status(checks, capabilities),
        checks=checks,
        capabilities=capabilities,
        runtime_mode=resolved_settings.runtime_mode,
        runtime_workers=resolved_settings.runtime_workers,
    )


def health_exit_code(report: EnvironmentHealthReport) -> int:
    if report.overall_status is OverallStatus.READY:
        return 0
    if report.overall_status is OverallStatus.UNAVAILABLE:
        return 2
    return 1


def format_health_json(report: EnvironmentHealthReport) -> str:
    """Return stable, sanitized machine-readable diagnostics."""

    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def format_health_report(report: EnvironmentHealthReport) -> str:
    """Return a concise summary suitable for Terminal users."""

    labels = {
        "python": "Python",
        "venv": "Environment",
        "dependencies": "Dependencies",
        "ollama": "Ollama",
        "model": "AI Model",
        "port": "Dashboard Port",
        "filesystem": "Storage",
        "disk": "Disk Space",
    }
    lines = [
        "Amazon Review Analyzer Environment Health",
        "",
        f"Status: {report.overall_status.value.upper().replace('_', ' ')}",
        "",
    ]
    for name, check in report.checks.items():
        status = check.status.value.upper().replace("_", " ")
        lines.append(f"{labels[name]:16} {status:15} {check.message}")
    if report.overall_status is OverallStatus.LIMITED:
        lines.extend(
            [
                "",
                "Limited Mode: Demo and saved results remain available; new AI analysis may be unavailable.",
            ]
        )
    return _sanitize_text("\n".join(lines))


def run_health_cli(
    *,
    json_output: bool = False,
    project_root: str | Path = PROJECT_ROOT,
    settings: HealthSettings | None = None,
    probes: HealthProbes | None = None,
    report: EnvironmentHealthReport | None = None,
    stream: TextIO | None = None,
) -> int:
    """Print one health report and return the documented process exit code."""

    resolved_report = report or run_environment_health(
        project_root=project_root,
        settings=settings,
        probes=probes,
    )
    output = format_health_json(resolved_report) if json_output else format_health_report(resolved_report)
    print(output, file=stream or sys.stdout)
    return health_exit_code(resolved_report)


__all__ = [
    "CheckStatus",
    "EnvironmentHealthReport",
    "HEALTH_SCHEMA_VERSION",
    "HealthCheck",
    "HealthProbes",
    "HealthSettings",
    "OverallStatus",
    "check_dependencies",
    "check_disk_space",
    "check_filesystem",
    "check_ollama_and_model",
    "check_port",
    "check_python",
    "check_virtual_environment",
    "discover_health_settings",
    "format_health_json",
    "format_health_report",
    "health_exit_code",
    "load_requirement_names",
    "run_environment_health",
    "run_health_cli",
    "sanitize_diagnostics",
]
