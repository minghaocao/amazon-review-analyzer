"""Build the allowlisted Amazon Review Analyzer v1.0.1 release package.

This builder creates a portfolio/local-distribution companion package.  It is
deliberately not a source archive and does not make the lightweight macOS app
standalone.  The package never includes a Python environment, model weights,
runtime outputs, credentials, browser state, or real review data.

The final ZIP digest is stored in the project-root companion manifest.  The
manifest inside the ZIP records a null ZIP digest because embedding a ZIP's
own digest in that ZIP would be self-referential and cannot be made truthful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PRODUCT_NAME = "Amazon Review Analyzer"
VERSION = "1.0.1"
RELEASE_TYPE = "Maintenance / UX / Distribution Release"
PACKAGE_NAME = "Amazon-Review-Analyzer-v1.0.1"
ZIP_NAME = f"{PACKAGE_NAME}.zip"
MANIFEST_SCHEMA = "step16_6.release.v1"
BUNDLE_ID = "com.caominghao.amazonreviewanalyzer"
BUNDLE_VERSION = "1.0.1"
BUNDLE_BUILD_VERSION = "101"
MODEL_PROVIDER = "ollama"
MODEL_NAME = "qwen3:8b"
RUNTIME_MODE = "stable"
RUNTIME_WORKERS = 1
FROZEN_CANDIDATE = "generalization_v2_candidate"
FROZEN_TAXONOMY = "generalization_v2_candidate"
REPORT_TEMPLATE_VERSION = "step14.report.v1"
COMPARISON_SCHEMA_VERSION = "step13.v1"
SAVED_ANALYSIS_SCHEMA_VERSION = "step13.1.analysis.v1"
SUPPORTED_LANGUAGES = ("en", "zh")
SUPPORTED_INPUT_MODES = ("demo", "amazon_quick_scan", "upload", "paste")
EXTERNAL_MANIFEST_NAME = "release_manifest_v1.0.1.json"
INTERNAL_MANIFEST_NAME = "release_manifest.json"
LEGACY_MANIFEST_NAME = "release_manifest.json"
CHECKSUMS_NAME = "SHA256SUMS.txt"

# A fixed member timestamp makes the ZIP byte-for-byte reproducible when the
# payload and explicitly supplied metadata are unchanged.
ZIP_TIMESTAMP = (2026, 10, 22, 0, 0, 0)


class ReleaseBuildError(RuntimeError):
    """Raised when the release cannot be built without weakening safeguards."""


@dataclass(frozen=True)
class ReleaseBuildResult:
    """Paths and verified metadata produced by :func:`build_release`."""

    package_directory: Path
    zip_path: Path
    external_manifest_path: Path
    zip_sha256: str
    zip_size_bytes: int
    included_file_count: int


@dataclass(frozen=True)
class CopySpec:
    source: str
    destination: str


# Every distributable file originates from this bounded whitelist.  Directories
# are copied recursively, but their contents are independently checked below.
COPY_SPECS: tuple[CopySpec, ...] = (
    CopySpec("dist/Amazon Review Analyzer.app", "Amazon Review Analyzer.app"),
    CopySpec("Install to Applications.command", "Install to Applications.command"),
    CopySpec(
        "launcher/Stop Amazon Review Analyzer.command",
        "Stop Amazon Review Analyzer.command",
    ),
    CopySpec("setup_amazon_review_analyzer.command", "setup_amazon_review_analyzer.command"),
    CopySpec("README.md", "README.md"),
    CopySpec("QUICK_START_EN.md", "QUICK_START_EN.md"),
    CopySpec("QUICK_START_CN.md", "QUICK_START_CN.md"),
    CopySpec("RELEASE_NOTES_v1.0.1.md", "RELEASE_NOTES.md"),
    CopySpec("CHANGELOG.md", "CHANGELOG.md"),
    CopySpec("SYSTEM_OVERVIEW.md", "SYSTEM_OVERVIEW.md"),
    CopySpec("PORTFOLIO_SUMMARY.md", "PORTFOLIO_SUMMARY.md"),
    CopySpec(
        "docs/Amazon_Review_Analyzer_v1.0.1_User_Guide_EN.md",
        "docs/Amazon_Review_Analyzer_v1.0.1_User_Guide_EN.md",
    ),
    CopySpec(
        "docs/Amazon_Review_Analyzer_v1.0_User_Guide_CN.md",
        "docs/Amazon_Review_Analyzer_v1.0.1_User_Guide_CN.md",
    ),
    CopySpec("release/Amazon_Review_Analyzer_v1.0.0/screenshots", "screenshots"),
    CopySpec("release/Amazon_Review_Analyzer_v1.0.0/sample_reports", "sample_reports"),
    CopySpec("release/Amazon_Review_Analyzer_v1.0.0/sample_data", "sample_data"),
)


FORBIDDEN_PATH_PARTS = {
    ".env",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "cookies",
    "browser_profile",
    "browser_profiles",
    "session_data",
}
FORBIDDEN_SUFFIXES = {
    ".gguf",
    ".safetensors",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".bin",
    ".onnx",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
}
TEXT_SUFFIXES = {
    ".command",
    ".csv",
    ".html",
    ".json",
    ".md",
    ".plist",
    ".py",
    ".txt",
    "",  # The bundle executable is a small shell script without a suffix.
}
PRIVATE_PATH_PATTERN = re.compile(r"/(?:Users|home)/[^/\s<>'\"]+")
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "assigned_secret",
        re.compile(
            r"(?im)^\s*(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)"
            r"\s*[:=]\s*['\"]?(?!optional\b|none\b|null\b|not set\b|your[_ -])"
            r"[^'\"\s#]{8,}"
        ),
    ),
    ("cookie_header", re.compile(r"(?im)^\s*(?:set-)?cookie\s*:\s*[^\s].+")),
    ("session_cookie", re.compile(r"(?i)\b(?:sessionid|session-token|ubid-main)=\S+")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def tree_hashes(root: Path) -> dict[str, str]:
    """Return stable relative-path SHA-256 values for all regular files."""

    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _regular_files(root)
    }


def _assert_no_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReleaseBuildError(
                f"Release inputs and payloads may not contain symbolic links: {path.name}"
            )


def _copy_allowlisted(project_root: Path, package_root: Path) -> None:
    for spec in COPY_SPECS:
        source = project_root / spec.source
        destination = package_root / spec.destination
        if not source.exists():
            raise ReleaseBuildError(f"Required release artifact is missing: {spec.source}")
        if source.is_symlink():
            raise ReleaseBuildError(f"Release artifact may not be a symbolic link: {spec.source}")
        if source.is_dir():
            _assert_no_symlinks(source)
            shutil.copytree(source, destination, copy_function=shutil.copy2)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        else:
            raise ReleaseBuildError(f"Unsupported release artifact: {spec.source}")


def _validate_app_bundle(package_root: Path) -> None:
    app_root = package_root / "Amazon Review Analyzer.app"
    plist_path = app_root / "Contents/Info.plist"
    executable = app_root / "Contents/MacOS/Amazon Review Analyzer"
    launcher = app_root / "Contents/Resources/mac_launcher.py"
    if not plist_path.is_file() or not executable.is_file() or not launcher.is_file():
        raise ReleaseBuildError("The macOS app bundle is incomplete.")
    try:
        with plist_path.open("rb") as handle:
            info = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException) as exc:
        raise ReleaseBuildError("The macOS app Info.plist is invalid.") from exc
    expected = {
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleShortVersionString": BUNDLE_VERSION,
        "CFBundleVersion": BUNDLE_BUILD_VERSION,
    }
    for key, value in expected.items():
        if str(info.get(key, "")) != value:
            raise ReleaseBuildError(f"Unexpected app bundle metadata: {key}")
    if not os.access(executable, os.X_OK):
        raise ReleaseBuildError("The macOS app launcher is not executable.")


def scan_release_tree(root: Path) -> list[str]:
    """Return privacy/security violations found in an unpacked release tree."""

    violations: list[str] = []
    if not root.is_dir():
        return ["release_root_missing"]
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        parts_lower = {part.lower() for part in relative.parts}
        if path.is_symlink():
            violations.append(f"symbolic_link:{relative.as_posix()}")
            continue
        forbidden_parts = parts_lower & FORBIDDEN_PATH_PARTS
        if forbidden_parts:
            violations.append(f"forbidden_path:{relative.as_posix()}")
        for part in parts_lower:
            if part.startswith(".env") or any(
                marker in part
                for marker in ("cookie", "session", "browser_profile", "browser-profile")
            ):
                violations.append(f"forbidden_path:{relative.as_posix()}")
            if part in {"models", "ollama_models", "ollama-models", "blobs"}:
                violations.append(f"model_payload:{relative.as_posix()}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden_file_type:{relative.as_posix()}")
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Binary files with no suffix are permitted only as the known app
            # launcher; all copied inputs remain constrained by COPY_SPECS.
            continue
        if PRIVATE_PATH_PATTERN.search(text):
            violations.append(f"private_absolute_path:{relative.as_posix()}")
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                violations.append(f"{label}:{relative.as_posix()}")
    return sorted(set(violations))


def _blind_v2_metrics() -> dict[str, Any]:
    return {
        "evaluation_set": "frozen Blind V2",
        "case_count": 150,
        "aspect_sentiment_f1": 0.8566,
        "aspect_only_f1": 0.8711,
        "sentiment_accuracy": 0.9833,
        "evidence_validity": 1.0,
        "evidence_match": 1.0,
        "scope_note": (
            "Metrics apply only to the frozen Blind V2 evaluation set and do not "
            "represent universal accuracy."
        ),
    }


def _manifest_payload(
    *,
    created_at: str,
    automated_test_count: int,
    automated_test_passed: int,
    health_check_status: str,
    frozen_assets_verified: bool,
    frozen_asset_hashes: Mapping[str, str],
    included_artifacts: Sequence[str],
    file_hashes: Mapping[str, str],
    release_package_sha256: str | None,
    release_package_size_bytes: int | None,
    manifest_scope: str,
) -> dict[str, Any]:
    return {
        "manifest_schema_version": MANIFEST_SCHEMA,
        "product_name": PRODUCT_NAME,
        "version": VERSION,
        "app_version": VERSION,
        "release_name": f"{PRODUCT_NAME} v{VERSION}",
        "release_type": RELEASE_TYPE,
        "created_at": created_at,
        "released_at": created_at,
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "model_provider": MODEL_PROVIDER,
        "model_name": MODEL_NAME,
        "runtime_mode": RUNTIME_MODE,
        "runtime_workers": RUNTIME_WORKERS,
        "default_runtime_mode": RUNTIME_MODE,
        "default_workers": RUNTIME_WORKERS,
        "bundle_id": BUNDLE_ID,
        "app_bundle_version": BUNDLE_VERSION,
        "app_bundle_build_version": BUNDLE_BUILD_VERSION,
        "release_package_name": ZIP_NAME,
        "release_package_sha256": release_package_sha256,
        "release_package_size_bytes": release_package_size_bytes,
        "release_package_sha256_scope": (
            "The final ZIP SHA-256 is authoritative only in the external companion "
            f"manifest {EXTERNAL_MANIFEST_NAME}; an embedded self-hash is intentionally "
            "null."
            if release_package_sha256 is None
            else "SHA-256 of the finalized release ZIP beside this companion manifest."
        ),
        "frozen_analysis_candidate": FROZEN_CANDIDATE,
        "frozen_taxonomy_version": FROZEN_TAXONOMY,
        "analysis_candidate": FROZEN_CANDIDATE,
        "taxonomy_version": FROZEN_TAXONOMY,
        "report_template_version": REPORT_TEMPLATE_VERSION,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "saved_analysis_schema_version": SAVED_ANALYSIS_SCHEMA_VERSION,
        "supported_languages": list(SUPPORTED_LANGUAGES),
        "supported_input_modes": list(SUPPORTED_INPUT_MODES),
        "blind_v2_reference": "outputs/evaluation/blind_v2/blind_v2_evaluation.json",
        "blind_v2_metrics": _blind_v2_metrics(),
        "automated_test_count": automated_test_count,
        "automated_test_passed": automated_test_passed,
        "tests_passed": automated_test_passed,
        "health_check_status": health_check_status,
        "new_llm_calls": 0,
        "amazon_network_calls": 0,
        "frozen_assets_verified": frozen_assets_verified,
        "frozen_asset_hashes": dict(frozen_asset_hashes),
        "included_artifacts": list(included_artifacts),
        "file_hashes": dict(file_hashes),
        "manifest_scope": manifest_scope,
        "distribution_scope": "Portfolio / local distribution package",
        "standalone": False,
        "required_external_runtime": [
            "source project",
            "Python environment and dependencies",
            "Ollama for new AI analysis",
            "qwen3:8b for new AI analysis",
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_internal_metadata(
    package_root: Path,
    *,
    created_at: str,
    automated_test_count: int,
    automated_test_passed: int,
    health_check_status: str,
    frozen_assets_verified: bool,
    frozen_asset_hashes: Mapping[str, str],
) -> None:
    payload_hashes = tree_hashes(package_root)
    manifest = _manifest_payload(
        created_at=created_at,
        automated_test_count=automated_test_count,
        automated_test_passed=automated_test_passed,
        health_check_status=health_check_status,
        frozen_assets_verified=frozen_assets_verified,
        frozen_asset_hashes=frozen_asset_hashes,
        included_artifacts=sorted(payload_hashes),
        file_hashes=payload_hashes,
        release_package_sha256=None,
        release_package_size_bytes=None,
        manifest_scope=(
            "Hashes cover the allowlisted package payload before release_manifest.json "
            "and SHA256SUMS.txt are added."
        ),
    )
    _write_json(package_root / INTERNAL_MANIFEST_NAME, manifest)

    checksummed_files = {
        path.relative_to(package_root).as_posix(): sha256_file(path)
        for path in _regular_files(package_root)
        if path.name != CHECKSUMS_NAME
    }
    checksum_text = "".join(
        f"{digest}  {relative}\n"
        for relative, digest in sorted(checksummed_files.items())
    )
    (package_root / CHECKSUMS_NAME).write_text(checksum_text, encoding="utf-8")


def verify_checksum_file(package_root: Path) -> None:
    checksum_path = package_root / CHECKSUMS_NAME
    if not checksum_path.is_file():
        raise ReleaseBuildError("SHA256SUMS.txt is missing.")
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    expected_paths: set[str] = set()
    for line in lines:
        if "  " not in line:
            raise ReleaseBuildError("SHA256SUMS.txt contains an invalid line.")
        expected_digest, relative = line.split("  ", 1)
        safe_relative = PurePosixPath(relative)
        if safe_relative.is_absolute() or ".." in safe_relative.parts:
            raise ReleaseBuildError("SHA256SUMS.txt contains an unsafe path.")
        target = package_root.joinpath(*safe_relative.parts)
        if not target.is_file() or target.is_symlink():
            raise ReleaseBuildError(f"Checksummed file is unavailable: {relative}")
        if sha256_file(target) != expected_digest:
            raise ReleaseBuildError(f"Checksum mismatch: {relative}")
        expected_paths.add(relative)
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in _regular_files(package_root)
        if path.name != CHECKSUMS_NAME
    }
    if expected_paths != actual_paths:
        raise ReleaseBuildError("SHA256SUMS.txt does not cover the complete payload.")


def _zip_info(archive_name: str, mode: int, *, is_directory: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(archive_name, ZIP_TIMESTAMP)
    info.create_system = 3
    normalized_mode = mode & 0o777
    if is_directory:
        normalized_mode |= stat.S_IFDIR
        info.external_attr = (normalized_mode << 16) | 0x10
    else:
        normalized_mode |= stat.S_IFREG
        info.external_attr = normalized_mode << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def create_deterministic_zip(package_root: Path, zip_path: Path) -> None:
    """Create a sorted, reproducible ZIP with executable modes preserved."""

    all_paths = sorted(
        package_root.rglob("*"), key=lambda path: path.relative_to(package_root).as_posix()
    )
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        root_prefix = f"{PACKAGE_NAME}/"
        archive.writestr(_zip_info(root_prefix, 0o755, is_directory=True), b"")
        for path in all_paths:
            relative = path.relative_to(package_root).as_posix()
            archive_name = f"{root_prefix}{relative}"
            mode = stat.S_IMODE(path.stat().st_mode)
            if path.is_dir():
                archive.writestr(
                    _zip_info(f"{archive_name}/", mode or 0o755, is_directory=True), b""
                )
            elif path.is_file() and not path.is_symlink():
                archive.writestr(
                    _zip_info(archive_name, mode or 0o644, is_directory=False),
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
            else:
                raise ReleaseBuildError(f"Unsupported ZIP payload: {relative}")


def validate_release_zip(zip_path: Path, package_root: Path) -> None:
    """Validate paths, contents, hashes, and file inventory in the final ZIP."""

    if not zip_path.is_file():
        raise ReleaseBuildError("Release ZIP was not created.")
    expected = {
        f"{PACKAGE_NAME}/{path.relative_to(package_root).as_posix()}": sha256_file(path)
        for path in _regular_files(package_root)
    }
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ReleaseBuildError(f"Release ZIP integrity failed: {bad_member}")
        actual: dict[str, str] = {}
        for info in archive.infolist():
            candidate = PurePosixPath(info.filename)
            if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
                raise ReleaseBuildError("Release ZIP contains an unsafe member path.")
            if candidate.parts[0] != PACKAGE_NAME:
                raise ReleaseBuildError("Release ZIP has an unexpected top-level path.")
            if info.is_dir():
                continue
            digest = hashlib.sha256(archive.read(info)).hexdigest()
            actual[info.filename] = digest
        if actual != expected:
            raise ReleaseBuildError("Release ZIP contents differ from the release folder.")


def _safe_replace_directory(staged: Path, destination: Path, release_root: Path) -> None:
    if destination.name != PACKAGE_NAME or destination.parent.resolve() != release_root.resolve():
        raise ReleaseBuildError("Refusing to replace an unexpected release directory.")
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ReleaseBuildError("Refusing to replace an unsafe release target.")
        shutil.rmtree(destination)
    os.replace(staged, destination)


def _iso_timestamp_now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _load_frozen_asset_hashes(project_root: Path) -> dict[str, str]:
    """Read, but never rewrite, the historical release's frozen hash contract."""

    path = project_root / LEGACY_MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError("The historical frozen-asset manifest is unavailable.") from exc
    hashes = payload.get("frozen_asset_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ReleaseBuildError("The historical frozen-asset hash contract is invalid.")
    normalized: dict[str, str] = {}
    for relative, digest in hashes.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ReleaseBuildError("The frozen-asset hash contract is invalid.")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ReleaseBuildError("The frozen-asset hash contract contains an invalid digest.")
        target = project_root / relative
        if not target.is_file() or target.is_symlink() or sha256_file(target) != digest:
            raise ReleaseBuildError(f"Frozen asset verification failed: {relative}")
        normalized[relative] = digest
    return dict(sorted(normalized.items()))


def build_release(
    project_root: str | Path,
    *,
    automated_test_count: int,
    automated_test_passed: int,
    health_check_status: str = "PASS",
    frozen_assets_verified: bool = True,
    created_at: str | None = None,
) -> ReleaseBuildResult:
    """Build and verify the complete v1.0.1 companion package."""

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise ReleaseBuildError("Project root is unavailable.")
    if automated_test_count < 1 or automated_test_passed != automated_test_count:
        raise ReleaseBuildError("A release requires a complete passing automated test run.")
    if health_check_status != "PASS" or not frozen_assets_verified:
        raise ReleaseBuildError("Health and frozen-asset verification must pass before release.")

    release_root = root / "release"
    release_root.mkdir(parents=True, exist_ok=True)
    old_package = release_root / "Amazon_Review_Analyzer_v1.0.0"
    old_zip = release_root / "Amazon_Review_Analyzer_v1.0.0.zip"
    old_package_before = tree_hashes(old_package)
    old_zip_before = sha256_file(old_zip) if old_zip.is_file() else None
    frozen_asset_hashes = _load_frozen_asset_hashes(root)

    timestamp = created_at or _iso_timestamp_now()
    temp_root = Path(tempfile.mkdtemp(prefix=".step16_6-", dir=release_root))
    staged_package = temp_root / PACKAGE_NAME
    staged_zip = temp_root / ZIP_NAME
    final_package = release_root / PACKAGE_NAME
    final_zip = release_root / ZIP_NAME
    external_manifest = root / EXTERNAL_MANIFEST_NAME
    try:
        staged_package.mkdir(parents=True)
        _copy_allowlisted(root, staged_package)
        _validate_app_bundle(staged_package)
        _write_internal_metadata(
            staged_package,
            created_at=timestamp,
            automated_test_count=automated_test_count,
            automated_test_passed=automated_test_passed,
            health_check_status=health_check_status,
            frozen_assets_verified=frozen_assets_verified,
            frozen_asset_hashes=frozen_asset_hashes,
        )
        verify_checksum_file(staged_package)
        violations = scan_release_tree(staged_package)
        if violations:
            raise ReleaseBuildError(
                "Release privacy/security scan failed: " + ", ".join(violations)
            )
        create_deterministic_zip(staged_package, staged_zip)
        validate_release_zip(staged_zip, staged_package)

        _safe_replace_directory(staged_package, final_package, release_root)
        if final_zip.exists() and (final_zip.is_symlink() or not final_zip.is_file()):
            raise ReleaseBuildError("Refusing to replace an unsafe release ZIP target.")
        os.replace(staged_zip, final_zip)
        validate_release_zip(final_zip, final_package)
        verify_checksum_file(final_package)

        final_hashes = tree_hashes(final_package)
        zip_digest = sha256_file(final_zip)
        zip_size = final_zip.stat().st_size
        external_payload = _manifest_payload(
            created_at=timestamp,
            automated_test_count=automated_test_count,
            automated_test_passed=automated_test_passed,
            health_check_status=health_check_status,
            frozen_assets_verified=frozen_assets_verified,
            frozen_asset_hashes=frozen_asset_hashes,
            included_artifacts=sorted(final_hashes),
            file_hashes=final_hashes,
            release_package_sha256=zip_digest,
            release_package_size_bytes=zip_size,
            manifest_scope=(
                "Authoritative external companion manifest. File hashes cover every "
                "regular file in the release folder, including its internal manifest "
                "and SHA256SUMS.txt."
            ),
        )
        _write_json(external_manifest, external_payload)

        if tree_hashes(old_package) != old_package_before:
            raise ReleaseBuildError("Historical v1.0.0 release folder changed during build.")
        old_zip_after = sha256_file(old_zip) if old_zip.is_file() else None
        if old_zip_after != old_zip_before:
            raise ReleaseBuildError("Historical v1.0.0 release ZIP changed during build.")

        return ReleaseBuildResult(
            package_directory=final_package,
            zip_path=final_zip,
            external_manifest_path=external_manifest,
            zip_sha256=zip_digest,
            zip_size_bytes=zip_size,
            included_file_count=len(final_hashes),
        )
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the verified Amazon Review Analyzer v1.0.1 release package."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project directory (defaults to the directory containing this script).",
    )
    parser.add_argument(
        "--automated-test-count",
        "--test-count",
        dest="automated_test_count",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--automated-test-passed",
        "--test-passed",
        dest="automated_test_passed",
        type=int,
        required=True,
    )
    parser.add_argument("--health-check-status", default="PASS")
    parser.add_argument(
        "--frozen-assets-verified",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--created-at",
        help="Optional ISO-8601 timestamp for reproducible rebuilds.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = build_release(
            args.project_root,
            automated_test_count=args.automated_test_count,
            automated_test_passed=args.automated_test_passed,
            health_check_status=args.health_check_status,
            frozen_assets_verified=args.frozen_assets_verified,
            created_at=args.created_at,
        )
    except ReleaseBuildError as exc:
        print(f"Release build failed: {exc}")
        return 1
    print(f"Release folder: {result.package_directory}")
    print(f"Release ZIP: {result.zip_path}")
    print(f"ZIP size: {result.zip_size_bytes} bytes")
    print(f"ZIP SHA-256: {result.zip_sha256}")
    print(f"Included files: {result.included_file_count}")
    print(f"Manifest: {result.external_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
