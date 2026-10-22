"""Offline release-contract checks for Amazon Review Analyzer v1.0.1.

The health check deliberately performs no HTTP requests, browser automation,
model discovery, or inference.  It verifies only local release metadata,
configuration contracts, frozen file hashes, localization coverage, report
template markers, Saved Analysis structure, and the optional macOS launcher.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from string import Formatter
from typing import Any, Mapping

from analysis_library import AnalysisLibraryError, validate_analysis
from analysis_runtime import run_analysis_runtime
from competitor_comparison import COMPARISON_SCHEMA_VERSION
from config import (
    ANALYSIS_CANDIDATE_VERSION,
    APP_VERSION,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_RUNTIME_MODE,
    DEFAULT_WORKERS,
    PROJECT_ROOT,
    REPORT_TEMPLATE_VERSION,
    SUPPORTED_INPUT_MODES,
    SUPPORTED_LANGUAGES,
    TAXONOMY_VERSION,
)
from analysis_library import ANALYSIS_LIBRARY_SCHEMA_VERSION
from i18n import TRANSLATIONS
from report_generator import (
    REPORT_TEMPLATE_VERSION as RENDERER_TEMPLATE_VERSION,
    _REPORT_I18N_KEYS,
)


RELEASE_MANIFEST_PATH = PROJECT_ROOT / "release_manifest_v1.0.1.json"
RELEASE_MANIFEST_SCHEMA_VERSION = "step16_6.release.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_RELEASE_FILES = (
    "app.py",
    "main.py",
    "config.py",
    "input_loader.py",
    "cleaner.py",
    "review_analyzer.py",
    "batch_analyzer.py",
    "aggregator.py",
    "keyword_analyzer.py",
    "analysis_runtime.py",
    "analysis_library.py",
    "competitor_comparison.py",
    "comparison_ui.py",
    "report_generator.py",
    "release_metadata.py",
    "release_health_check.py",
    "release_manifest.json",
    "release_manifest_v1.0.1.json",
    "i18n.py",
    "README.md",
    "requirements.txt",
    ".env.example",
)

REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "manifest_schema_version",
        "product_name",
        "version",
        "release_type",
        "created_at",
        "provider",
        "model",
        "runtime_mode",
        "bundle_id",
        "app_bundle_version",
        "release_package_name",
        "release_package_sha256",
        "frozen_analysis_candidate",
        "frozen_taxonomy_version",
        "blind_v2_reference",
        "automated_test_count",
        "automated_test_passed",
        "health_check_status",
        "new_llm_calls",
        "amazon_network_calls",
        "frozen_assets_verified",
        "included_artifacts",
        "file_hashes",
        "app_version",
        "release_name",
        "released_at",
        "model_provider",
        "model_name",
        "analysis_candidate",
        "taxonomy_version",
        "report_template_version",
        "comparison_schema_version",
        "saved_analysis_schema_version",
        "default_runtime_mode",
        "default_workers",
        "supported_languages",
        "supported_input_modes",
        "tests_passed",
        "frozen_asset_hashes",
    }
)


class HealthStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


@dataclass(frozen=True)
class HealthCheckResult:
    """One concise, non-sensitive release check result."""

    name: str
    status: HealthStatus
    detail: str


@dataclass(frozen=True)
class ReleaseHealthReport:
    """Complete deterministic result of the offline release health check."""

    app_version: str
    checks: tuple[HealthCheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.status is not HealthStatus.FAIL for check in self.checks)

    @property
    def failed_checks(self) -> tuple[HealthCheckResult, ...]:
        return tuple(
            check for check in self.checks if check.status is HealthStatus.FAIL
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_version": self.app_version,
            "status": "PASS" if self.passed else "FAIL",
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "detail": check.detail,
                }
                for check in self.checks
            ],
        }


class ReleaseManifestError(ValueError):
    """The release manifest is absent, unreadable, or malformed."""


def _result(
    name: str,
    passed: bool,
    success: str,
    failure: str,
) -> HealthCheckResult:
    return HealthCheckResult(
        name=name,
        status=HealthStatus.PASS if passed else HealthStatus.FAIL,
        detail=success if passed else failure,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release_manifest(path: str | Path = RELEASE_MANIFEST_PATH) -> dict[str, Any]:
    """Load one local release manifest without mutating project state."""

    manifest_path = Path(path)
    if manifest_path.is_symlink():
        raise ReleaseManifestError("The release manifest cannot be a symbolic link.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseManifestError("The release manifest is missing.") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError("The release manifest is unreadable or invalid.") from exc
    if not isinstance(payload, dict):
        raise ReleaseManifestError("The release manifest must contain a JSON object.")
    return payload


def _check_required_files(root: Path) -> HealthCheckResult:
    missing = [name for name in REQUIRED_RELEASE_FILES if not (root / name).is_file()]
    return _result(
        "required_files",
        not missing,
        f"All {len(REQUIRED_RELEASE_FILES)} required release files are present.",
        "Missing required files: " + ", ".join(missing),
    )


def _manifest_contract_errors(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(manifest))
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
    if manifest.get("manifest_schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION:
        errors.append("unsupported manifest_schema_version")
    string_fields = (
        "product_name",
        "version",
        "release_type",
        "provider",
        "model",
        "runtime_mode",
        "bundle_id",
        "app_bundle_version",
        "release_package_name",
        "frozen_analysis_candidate",
        "frozen_taxonomy_version",
        "blind_v2_reference",
        "health_check_status",
        "app_version",
        "release_name",
        "model_provider",
        "model_name",
        "analysis_candidate",
        "taxonomy_version",
        "report_template_version",
        "comparison_schema_version",
        "saved_analysis_schema_version",
        "default_runtime_mode",
    )
    for field in string_fields:
        if not isinstance(manifest.get(field), str) or not str(manifest.get(field)).strip():
            errors.append(f"{field} must be a non-empty string")
    if isinstance(manifest.get("app_version"), str) and not re.fullmatch(
        r"\d+\.\d+\.\d+", manifest["app_version"]
    ):
        errors.append("app_version must use semantic version format")
    if isinstance(manifest.get("version"), str) and not re.fullmatch(
        r"\d+\.\d+\.\d+", manifest["version"]
    ):
        errors.append("version must use semantic version format")
    try:
        released_at = datetime.fromisoformat(str(manifest.get("released_at", "")))
        if released_at.tzinfo is None:
            errors.append("released_at must include a timezone")
    except ValueError:
        errors.append("released_at must be an ISO-8601 timestamp")
    try:
        created_at = datetime.fromisoformat(str(manifest.get("created_at", "")))
        if created_at.tzinfo is None:
            errors.append("created_at must include a timezone")
    except ValueError:
        errors.append("created_at must be an ISO-8601 timestamp")
    tests_passed = manifest.get("tests_passed")
    if isinstance(tests_passed, bool) or not isinstance(tests_passed, int):
        errors.append("tests_passed must be an integer")
    elif tests_passed <= 0:
        errors.append("tests_passed must be positive")
    for field in ("automated_test_count", "automated_test_passed"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{field} must be a positive integer")
    for field in ("new_llm_calls", "amazon_network_calls"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{field} must be a non-negative integer")
    if manifest.get("frozen_assets_verified") is not True:
        errors.append("frozen_assets_verified must be true")
    if not isinstance(manifest.get("included_artifacts"), list) or not manifest.get(
        "included_artifacts"
    ):
        errors.append("included_artifacts must be a non-empty list")
    package_hash = manifest.get("release_package_sha256")
    if not isinstance(package_hash, str) or not SHA256_RE.fullmatch(package_hash):
        errors.append("release_package_sha256 must be a SHA-256 digest")
    file_hashes = manifest.get("file_hashes")
    if not isinstance(file_hashes, dict) or not file_hashes:
        errors.append("file_hashes must be a non-empty object")
    default_workers = manifest.get("default_workers")
    if (
        isinstance(default_workers, bool)
        or not isinstance(default_workers, int)
        or default_workers <= 0
    ):
        errors.append("default_workers must be a positive integer")
    languages = manifest.get("supported_languages")
    if not isinstance(languages, list) or not all(
        isinstance(language, str) and language for language in languages
    ):
        errors.append("supported_languages must be a non-empty string list")
    input_modes = manifest.get("supported_input_modes")
    if not isinstance(input_modes, list) or not input_modes or not all(
        isinstance(mode, str) and mode for mode in input_modes
    ):
        errors.append("supported_input_modes must be a non-empty string list")
    elif len(input_modes) != len(set(input_modes)):
        errors.append("supported_input_modes contains duplicates")
    if (
        isinstance(languages, list)
        and all(isinstance(language, str) for language in languages)
        and len(languages) != len(set(languages))
    ):
        errors.append("supported_languages contains duplicates")
    hashes = manifest.get("frozen_asset_hashes")
    if not isinstance(hashes, dict) or not hashes:
        errors.append("frozen_asset_hashes must be a non-empty object")
    return errors


def _check_manifest_contract(manifest: Mapping[str, Any]) -> HealthCheckResult:
    errors = _manifest_contract_errors(manifest)
    return _result(
        "release_manifest",
        not errors,
        "Release manifest schema and required fields are valid.",
        "Release manifest errors: " + "; ".join(errors),
    )


def _check_config_contract(manifest: Mapping[str, Any]) -> HealthCheckResult:
    expected: dict[str, Any] = {
        "product_name": "Amazon Review Analyzer",
        "version": APP_VERSION,
        "release_type": "Maintenance / UX / Distribution Release",
        "provider": DEFAULT_MODEL_PROVIDER,
        "model": DEFAULT_MODEL_NAME,
        "runtime_mode": DEFAULT_RUNTIME_MODE,
        "bundle_id": "com.caominghao.amazonreviewanalyzer",
        "app_bundle_version": APP_VERSION,
        "frozen_analysis_candidate": ANALYSIS_CANDIDATE_VERSION,
        "frozen_taxonomy_version": TAXONOMY_VERSION,
        "new_llm_calls": 0,
        "amazon_network_calls": 0,
        "frozen_assets_verified": True,
        "app_version": APP_VERSION,
        "release_name": f"Amazon Review Analyzer v{APP_VERSION}",
        "model_provider": DEFAULT_MODEL_PROVIDER,
        "model_name": DEFAULT_MODEL_NAME,
        "analysis_candidate": ANALYSIS_CANDIDATE_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "report_template_version": REPORT_TEMPLATE_VERSION,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "saved_analysis_schema_version": ANALYSIS_LIBRARY_SCHEMA_VERSION,
        "default_runtime_mode": DEFAULT_RUNTIME_MODE,
        "default_workers": DEFAULT_WORKERS,
        "supported_languages": sorted(SUPPORTED_LANGUAGES.values()),
        "supported_input_modes": sorted(SUPPORTED_INPUT_MODES),
    }
    mismatches = [
        key
        for key, value in expected.items()
        if (
            sorted(manifest.get(key, []))
            if key in {"supported_languages", "supported_input_modes"}
            and isinstance(manifest.get(key), list)
            else manifest.get(key)
        )
        != value
    ]
    concurrency_default = inspect.signature(run_analysis_runtime).parameters[
        "concurrency"
    ].default
    if concurrency_default != DEFAULT_WORKERS or DEFAULT_WORKERS != 1:
        mismatches.append("run_analysis_runtime.concurrency")
    if DEFAULT_RUNTIME_MODE != "stable":
        mismatches.append("default_runtime_mode")
    return _result(
        "configuration",
        not mismatches,
        "Version, model, schema, language, and Stable runtime contracts match.",
        "Manifest/config mismatch: " + ", ".join(sorted(set(mismatches))),
    )


def _check_localization() -> HealthCheckResult:
    expected_languages = set(SUPPORTED_LANGUAGES.values())
    actual_languages = set(TRANSLATIONS)
    problems: list[str] = []
    if actual_languages != expected_languages:
        problems.append("language set mismatch")
    if "en" in TRANSLATIONS and "zh" in TRANSLATIONS:
        if set(TRANSLATIONS["en"]) != set(TRANSLATIONS["zh"]):
            problems.append("English/Chinese key coverage differs")
        else:
            formatter = Formatter()

            def placeholders(template: str) -> set[str]:
                return {
                    field_name
                    for _, field_name, _, _ in formatter.parse(template)
                    if field_name
                }

            for key, english in TRANSLATIONS["en"].items():
                if placeholders(english) != placeholders(TRANSLATIONS["zh"][key]):
                    problems.append(f"placeholder mismatch: {key}")
                    break
    for language, translations in TRANSLATIONS.items():
        if any(
            not isinstance(value, str) or not value.strip()
            for value in translations.values()
        ):
            problems.append(f"{language} contains empty translations")
    key_count = len(TRANSLATIONS.get("en", {}))
    return _result(
        "localization",
        not problems,
        f"English and Chinese cover the same {key_count} translation keys.",
        "Localization errors: " + "; ".join(problems),
    )


def _safe_asset_path(root: Path, relative_name: object) -> Path | None:
    if not isinstance(relative_name, str) or not relative_name.strip():
        return None
    relative = Path(relative_name)
    if relative.is_absolute():
        return None
    root_resolved = root.resolve()
    unresolved = root_resolved / relative
    if unresolved.is_symlink():
        return None
    candidate = unresolved.resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


def _check_frozen_assets(
    root: Path,
    manifest: Mapping[str, Any],
) -> HealthCheckResult:
    hashes = manifest.get("frozen_asset_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        return HealthCheckResult(
            "frozen_assets",
            HealthStatus.FAIL,
            "No frozen asset hashes are available.",
        )
    failures: list[str] = []
    checked = 0
    for relative_name, expected_hash in hashes.items():
        safe_path = _safe_asset_path(root, relative_name)
        if safe_path is None:
            failures.append(f"unsafe path: {relative_name!s}")
            continue
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            failures.append(f"invalid expected hash: {relative_name!s}")
            continue
        if not safe_path.is_file() or safe_path.is_symlink():
            failures.append(f"missing asset: {relative_name!s}")
            continue
        checked += 1
        try:
            actual_hash = _sha256(safe_path)
        except OSError:
            failures.append(f"unreadable asset: {relative_name!s}")
            continue
        if actual_hash != expected_hash:
            failures.append(f"hash mismatch: {relative_name!s}")
    if failures:
        preview = failures[:5]
        if len(failures) > len(preview):
            preview.append(f"and {len(failures) - len(preview)} more")
        return HealthCheckResult(
            "frozen_assets",
            HealthStatus.FAIL,
            "; ".join(preview),
        )
    return HealthCheckResult(
        "frozen_assets",
        HealthStatus.PASS,
        f"Verified {checked} frozen assets by SHA-256.",
    )


def _check_report_template(root: Path, manifest: Mapping[str, Any]) -> HealthCheckResult:
    source_path = root / "report_generator.py"
    problems: list[str] = []
    if manifest.get("report_template_version") != RENDERER_TEMPLATE_VERSION:
        problems.append("template version mismatch")
    try:
        source = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        problems.append("report generator is unreadable")
    else:
        if "@media print" not in source:
            problems.append("print stylesheet marker is missing")
        if "html.escape" not in source:
            problems.append("HTML escaping marker is missing")
    for language in ("en", "zh"):
        translations = TRANSLATIONS.get(language, {})
        missing = sorted(set(_REPORT_I18N_KEYS.values()).difference(translations))
        if missing:
            problems.append(f"{language} report translations are incomplete")
    return _result(
        "report_template",
        not problems,
        "Standalone report template version, print CSS, and escaping markers are present.",
        "Report template errors: " + "; ".join(problems),
    )


def _check_library_structure(root: Path) -> HealthCheckResult:
    library_root = root / "outputs" / "library"
    if not library_root.exists():
        return HealthCheckResult(
            "saved_analysis_library",
            HealthStatus.PASS,
            "Saved Analysis library is empty and can be created on first save.",
        )
    if not library_root.is_dir() or library_root.is_symlink():
        return HealthCheckResult(
            "saved_analysis_library",
            HealthStatus.FAIL,
            "Saved Analysis library path is not a safe directory.",
        )
    children = [
        child
        for child in library_root.iterdir()
        if child.is_dir() and not child.is_symlink() and not child.name.startswith(".tmp-")
    ]
    failures: list[str] = []
    for child in children:
        try:
            validate_analysis(child, library_root=library_root, verify_hashes=True)
        except AnalysisLibraryError:
            failures.append(child.name)
    return _result(
        "saved_analysis_library",
        not failures,
        f"Validated {len(children)} persisted Saved Analysis snapshots.",
        "Invalid saved snapshots: " + ", ".join(failures[:5]),
    )


def _desktop_launcher_candidates() -> tuple[Path, ...]:
    home = Path.home()
    return (
        home / "Desktop" / "Amazon Review Analyzer.command",
        home
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "Desktop"
        / "Amazon Review Analyzer.command",
    )


def _check_launcher(
    launcher_path: str | Path | None,
    *,
    project_root: Path,
    required: bool,
) -> HealthCheckResult:
    if launcher_path is None:
        candidates = _desktop_launcher_candidates()
        launcher = next(
            (path for path in candidates if path.is_file()), candidates[0]
        )
    else:
        launcher = Path(launcher_path).expanduser()
    if not launcher.is_file() or launcher.is_symlink():
        return HealthCheckResult(
            "desktop_launcher",
            HealthStatus.FAIL if required else HealthStatus.WARN,
            "Desktop launcher was not found; this does not affect CLI operation.",
        )
    problems: list[str] = []
    if not os.access(launcher, os.X_OK):
        problems.append("launcher is not executable")
    try:
        content = launcher.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        problems.append("launcher is unreadable")
    else:
        first_line = content.splitlines()[0].strip() if content.splitlines() else ""
        if first_line != "#!/bin/zsh":
            problems.append("launcher does not use #!/bin/zsh")
        shared_launcher = (
            "mac_launcher.py" in content
            and "--foreground" in content
            and "--project-root" in content
        )
        legacy_launcher = str(project_root.resolve()) in content
        if not (shared_launcher or legacy_launcher):
            problems.append("launcher does not target this project")
        if "8501" not in content and not shared_launcher:
            problems.append("launcher does not target Streamlit port 8501")
    return _result(
        "desktop_launcher",
        not problems,
        "Desktop launcher exists, is executable, and uses zsh.",
        "Desktop launcher errors: " + "; ".join(problems),
    )


def run_release_health_check(
    project_root: str | Path = PROJECT_ROOT,
    *,
    manifest_path: str | Path | None = None,
    launcher_path: str | Path | None = None,
    require_launcher: bool | None = None,
) -> ReleaseHealthReport:
    """Run deterministic local checks only; no service or network is touched."""

    root = Path(project_root).resolve()
    checks: list[HealthCheckResult] = [_check_required_files(root)]
    release_path = (
        Path(manifest_path)
        if manifest_path is not None
        else root / "release_manifest_v1.0.1.json"
    )
    try:
        manifest = load_release_manifest(release_path)
    except ReleaseManifestError as exc:
        checks.append(
            HealthCheckResult("release_manifest", HealthStatus.FAIL, str(exc))
        )
        return ReleaseHealthReport(APP_VERSION, tuple(checks))

    manifest_check = _check_manifest_contract(manifest)
    checks.append(manifest_check)
    if manifest_check.status is HealthStatus.FAIL:
        return ReleaseHealthReport(str(manifest.get("app_version", APP_VERSION)), tuple(checks))

    checks.extend(
        (
            _check_config_contract(manifest),
            _check_localization(),
            _check_frozen_assets(root, manifest),
            _check_report_template(root, manifest),
            _check_library_structure(root),
            _check_launcher(
                launcher_path,
                project_root=root,
                required=(
                    (sys.platform == "darwin")
                    if require_launcher is None
                    else require_launcher
                ),
            ),
        )
    )
    return ReleaseHealthReport(str(manifest["app_version"]), tuple(checks))


def format_health_report(report: ReleaseHealthReport) -> str:
    """Render a compact, terminal-friendly PASS/FAIL report."""

    lines = [f"Amazon Review Analyzer v{report.app_version} Release Health Check"]
    lines.extend(
        f"[{check.status.value}] {check.name}: {check.detail}"
        for check in report.checks
    )
    lines.append("")
    lines.append(f"Release health check: {'PASS' if report.passed else 'FAIL'}")
    return "\n".join(lines)


__all__ = [
    "HealthCheckResult",
    "HealthStatus",
    "RELEASE_MANIFEST_PATH",
    "RELEASE_MANIFEST_SCHEMA_VERSION",
    "ReleaseHealthReport",
    "ReleaseManifestError",
    "format_health_report",
    "load_release_manifest",
    "run_release_health_check",
]
