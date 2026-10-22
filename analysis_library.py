"""Local, immutable snapshots of completed product analyses.

The Analysis Library is a persistence layer only.  It copies the seven
already-produced dashboard artifacts into a uniquely named directory and
records enough metadata and hashes to load that snapshot later.  This module
does not import or invoke the model, scraper, batch analyzer, aggregator, or
keyword analyzer.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from config import (
    AMAZON_CURRENT_DIR,
    ANALYSIS_CANDIDATE_VERSION,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OUTPUTS_DIR,
    SESSION_CURRENT_DIR,
)


ANALYSIS_LIBRARY_SCHEMA_VERSION = "step13.1.analysis.v1"
DEFAULT_ANALYSIS_LIBRARY_DIR = OUTPUTS_DIR / "library"

# These are the compact, already-generated files required by the comparison
# and dashboard layers.  Runtime logs, checkpoints, and raw imports are not
# copied into the library.
REQUIRED_ANALYSIS_FILES = (
    "analyzed_reviews.csv",
    "aspect_summary.csv",
    "aspect_evidence.json",
    "business_summary.json",
    "keyword_summary.csv",
    "sentiment_keywords.json",
    "aspect_keywords.json",
)
HASH_PROTECTED_FILES = (
    "analyzed_reviews.csv",
    "aspect_summary.csv",
    "business_summary.json",
)

_ANALYSIS_ID_RE = re.compile(r"^AN_\d{8}_\d{6}_[a-f0-9]{4}$")
_SAFE_DIRECTORY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,159}$")
_TRUSTED_CURRENT_OUTPUT_ROOTS = frozenset(
    path.resolve()
    for path in (OUTPUTS_DIR, SESSION_CURRENT_DIR, AMAZON_CURRENT_DIR)
)
_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "analysis_id",
        "product_name",
        "product_id",
        "asin",
        "source_type",
        "source_url",
        "analysis_run_id",
        "analysis_candidate",
        "taxonomy_version",
        "model_provider",
        "model_name",
        "imported_reviews",
        "successful_reviews",
        "failed_reviews",
        "skipped_reviews",
        "coverage",
        "sample_level",
        "has_rating",
        "saved_at",
        "source_output_path",
        "schema_version",
        "content_hash",
        "content_hashes",
    }
)


class AnalysisLibraryError(ValueError):
    """Base class for a safe, user-displayable library failure."""


class MissingAnalysisOutputError(AnalysisLibraryError):
    """A completed source is missing one or more required artifacts."""


class DuplicateAnalysisError(AnalysisLibraryError):
    """The same stable analysis run has already been saved."""

    def __init__(self, analysis_run_id: str, existing_analysis_id: str) -> None:
        self.analysis_run_id = analysis_run_id
        self.existing_analysis_id = existing_analysis_id
        super().__init__("This analysis has already been saved.")


class ManifestValidationError(AnalysisLibraryError):
    """A saved manifest is missing or does not follow the frozen schema."""


class SnapshotIntegrityError(AnalysisLibraryError):
    """A saved artifact is absent or no longer matches its recorded hash."""


class LibraryPathError(AnalysisLibraryError):
    """A requested path is outside the configured local library."""


@dataclass(frozen=True)
class SavedAnalysis:
    """One validated local snapshot and its language-independent manifest."""

    root: Path
    manifest: dict[str, Any]

    @property
    def analysis_id(self) -> str:
        return str(self.manifest["analysis_id"])

    @property
    def product_name(self) -> str:
        return str(self.manifest["product_name"])

    @property
    def analysis_run_id(self) -> str:
        return str(self.manifest["analysis_run_id"])

    @property
    def source_type(self) -> str:
        return str(self.manifest["source_type"])

    def artifact(self, filename: str) -> Path:
        """Return a required snapshot path without permitting path traversal."""

        if filename not in REQUIRED_ANALYSIS_FILES:
            raise LibraryPathError(f"Unknown saved-analysis artifact: {filename}")
        return self.root / filename


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _safe_slug(value: object, *, default: str = "untitled-product") -> str:
    normalized = str(value or "").strip().casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return (slug[:64].rstrip("-") or default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SnapshotIntegrityError(f"Unable to hash saved artifact {path.name}: {exc}") from exc
    return digest.hexdigest()


def _combined_hash(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for filename in sorted(hashes):
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(hashes[filename]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"Unable to read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{path.name} must contain a JSON object.")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        path.write_text(serialized + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise AnalysisLibraryError(f"Unable to write analysis manifest: {exc}") from exc


def _coerce_nonnegative_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return max(0, default)


def _coerce_coverage(value: object, default: float) -> float:
    try:
        coverage = float(value)
    except (TypeError, ValueError):
        coverage = default
    if not math.isfinite(coverage):
        coverage = default
    return min(1.0, max(0.0, coverage))


def _truthy_rating(value: object) -> bool:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(rating) and 1.0 <= rating <= 5.0


def _csv_facts(path: Path) -> dict[str, Any]:
    statuses = {"success": 0, "failed": 0, "skipped": 0}
    rows = 0
    has_rating = False
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for record in csv.DictReader(handle):
                rows += 1
                status = str(record.get("analysis_status", "")).strip().casefold()
                if status in statuses:
                    statuses[status] += 1
                has_rating = has_rating or _truthy_rating(record.get("rating"))
    except (OSError, csv.Error, UnicodeError) as exc:
        raise MissingAnalysisOutputError(
            f"Unable to inspect analyzed_reviews.csv: {exc}"
        ) from exc
    return {"rows": rows, "has_rating": has_rating, **statuses}


def _sample_level(successful: int) -> str:
    if successful <= 9:
        return "early_signals"
    if successful < 30:
        return "limited_sample"
    return "full_insight"


def _source_facts(source_root: Path) -> dict[str, Any]:
    facts = _csv_facts(source_root / "analyzed_reviews.csv")
    business = _read_json_object(source_root / "business_summary.json")
    quality = business.get("data_quality", {})
    if not isinstance(quality, Mapping):
        quality = {}
    sample = business.get("sample_level", {})
    if isinstance(sample, Mapping):
        sample_value = str(sample.get("key") or "").strip()
    else:
        sample_value = str(sample or "").strip()
    imported = _coerce_nonnegative_int(quality.get("total_input_rows"), facts["rows"])
    successful = _coerce_nonnegative_int(
        quality.get("successful_reviews"), facts["success"]
    )
    failed = _coerce_nonnegative_int(quality.get("failed_reviews"), facts["failed"])
    skipped = _coerce_nonnegative_int(quality.get("skipped_reviews"), facts["skipped"])
    imported = max(imported, successful + failed + skipped, facts["rows"])
    default_coverage = successful / imported if imported else 0.0
    return {
        "imported_reviews": imported,
        "successful_reviews": successful,
        "failed_reviews": failed,
        "skipped_reviews": skipped,
        "coverage": _coerce_coverage(quality.get("analysis_coverage"), default_coverage),
        "sample_level": sample_value or _sample_level(successful),
        "has_rating": facts["has_rating"],
    }


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _assert_required_source_files(source_root: Path) -> None:
    if not source_root.is_dir():
        raise MissingAnalysisOutputError("The completed analysis folder was not found.")
    missing: list[str] = []
    unsafe: list[str] = []
    for filename in REQUIRED_ANALYSIS_FILES:
        path = source_root / filename
        if not path.is_file():
            missing.append(filename)
        elif path.is_symlink():
            unsafe.append(filename)
    if missing:
        raise MissingAnalysisOutputError(
            "Completed analysis is missing required output(s): " + ", ".join(missing)
        )
    if unsafe:
        raise LibraryPathError(
            "Saved-analysis source artifacts cannot be symbolic links: "
            + ", ".join(unsafe)
        )


def _is_explicitly_trusted_current_output(source_root: Path) -> bool:
    return source_root.resolve() in _TRUSTED_CURRENT_OUTPUT_ROOTS


def _analysis_id(
    timestamp: datetime,
    token_factory: Callable[[], str],
) -> str:
    raw_token = re.sub(r"[^a-f0-9]", "", str(token_factory()).casefold())
    token = (raw_token + secrets.token_hex(2))[:4]
    return f"AN_{timestamp.astimezone(UTC):%Y%m%d_%H%M%S}_{token}"


def _safe_library_root(library_root: str | Path) -> Path:
    root = Path(library_root).expanduser().resolve(strict=False)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LibraryPathError(f"Unable to access the local Analysis Library: {exc}") from exc
    if not root.is_dir():
        raise LibraryPathError("Analysis Library path is not a directory.")
    return root


def _is_direct_child(path: Path, parent: Path) -> bool:
    try:
        return path.resolve().parent == parent.resolve()
    except OSError:
        return False


def _manifest_errors(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(_MANIFEST_REQUIRED_FIELDS.difference(manifest))
    if missing:
        errors.append("missing field(s): " + ", ".join(missing))
        return errors
    if manifest.get("schema_version") != ANALYSIS_LIBRARY_SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if not _ANALYSIS_ID_RE.fullmatch(str(manifest.get("analysis_id", ""))):
        errors.append("invalid analysis_id")
    for field in (
        "product_name",
        "source_type",
        "analysis_run_id",
        "analysis_candidate",
        "taxonomy_version",
        "model_provider",
        "model_name",
        "sample_level",
        "saved_at",
        "source_output_path",
        "content_hash",
    ):
        if not str(manifest.get(field) or "").strip():
            errors.append(f"{field} must not be empty")
    for field in (
        "imported_reviews",
        "successful_reviews",
        "failed_reviews",
        "skipped_reviews",
    ):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{field} must be a non-negative integer")
    coverage = manifest.get("coverage")
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not math.isfinite(float(coverage))
        or not 0.0 <= float(coverage) <= 1.0
    ):
        errors.append("coverage must be between 0 and 1")
    if not isinstance(manifest.get("has_rating"), bool):
        errors.append("has_rating must be boolean")
    hashes = manifest.get("content_hashes")
    if not isinstance(hashes, Mapping):
        errors.append("content_hashes must be an object")
    else:
        for filename in HASH_PROTECTED_FILES:
            value = hashes.get(filename)
            if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
                errors.append(f"missing valid SHA-256 for {filename}")
    return errors


def _resolve_saved_root(
    analysis: str | Path | SavedAnalysis,
    library_root: str | Path,
) -> Path:
    root = _safe_library_root(library_root)
    if isinstance(analysis, SavedAnalysis):
        candidate = analysis.root
    else:
        raw = Path(analysis)
        if raw.is_absolute() or len(raw.parts) > 1:
            candidate = raw
        else:
            direct = root / raw.name
            if direct.is_dir():
                candidate = direct
            elif _ANALYSIS_ID_RE.fullmatch(raw.name):
                matches: list[Path] = []
                for child in root.iterdir():
                    if not child.is_dir() or child.is_symlink():
                        continue
                    manifest = _read_json_object(child / "analysis_manifest.json")
                    if manifest.get("analysis_id") == raw.name:
                        matches.append(child)
                if len(matches) != 1:
                    raise LibraryPathError("Saved analysis was not found.")
                candidate = matches[0]
            else:
                candidate = direct
    resolved = candidate.expanduser().resolve(strict=False)
    if not _is_direct_child(resolved, root) or resolved.is_symlink():
        raise LibraryPathError("Saved analysis must be a direct child of the library.")
    if not resolved.is_dir():
        raise LibraryPathError("Saved analysis was not found.")
    return resolved


def validate_analysis(
    analysis: str | Path | SavedAnalysis,
    *,
    library_root: str | Path = DEFAULT_ANALYSIS_LIBRARY_DIR,
    verify_hashes: bool = True,
) -> SavedAnalysis:
    """Validate one saved snapshot and optionally verify its recorded hashes."""

    saved_root = _resolve_saved_root(analysis, library_root)
    manifest_path = saved_root / "analysis_manifest.json"
    if manifest_path.is_symlink():
        raise LibraryPathError("Saved analysis manifest cannot be a symbolic link.")
    manifest = _read_json_object(manifest_path)
    errors = _manifest_errors(manifest)
    if errors:
        raise ManifestValidationError("Invalid analysis manifest: " + "; ".join(errors))

    for filename in REQUIRED_ANALYSIS_FILES:
        artifact = saved_root / filename
        if not artifact.is_file():
            raise SnapshotIntegrityError(f"Saved artifact is missing: {filename}")
        if artifact.is_symlink():
            raise LibraryPathError(f"Saved artifact cannot be a symbolic link: {filename}")

    hashes = manifest["content_hashes"]
    if verify_hashes:
        for filename, expected in hashes.items():
            if filename not in REQUIRED_ANALYSIS_FILES:
                raise ManifestValidationError(
                    f"Manifest contains an unknown hashed artifact: {filename}"
                )
            actual = _sha256(saved_root / filename)
            if actual != expected:
                raise SnapshotIntegrityError(
                    f"Saved artifact failed integrity verification: {filename}"
                )
        if _combined_hash(hashes) != manifest["content_hash"]:
            raise SnapshotIntegrityError("Saved analysis content hash does not match.")
    return SavedAnalysis(saved_root, copy.deepcopy(manifest))


def load_analysis(
    analysis: str | Path | SavedAnalysis,
    *,
    library_root: str | Path = DEFAULT_ANALYSIS_LIBRARY_DIR,
    verify_hashes: bool = True,
) -> SavedAnalysis:
    """Load one Analysis Library snapshot without rebuilding any output."""

    return validate_analysis(
        analysis,
        library_root=library_root,
        verify_hashes=verify_hashes,
    )


def get_manifest(
    analysis: str | Path | SavedAnalysis,
    *,
    library_root: str | Path = DEFAULT_ANALYSIS_LIBRARY_DIR,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Return a defensive copy of one validated canonical manifest."""

    return copy.deepcopy(
        load_analysis(
            analysis,
            library_root=library_root,
            verify_hashes=verify_hashes,
        ).manifest
    )


def list_saved_analyses(
    *,
    library_root: str | Path = DEFAULT_ANALYSIS_LIBRARY_DIR,
    verify_hashes: bool = True,
) -> list[SavedAnalysis]:
    """List valid local snapshots newest first, ignoring unrelated/corrupt dirs."""

    root = _safe_library_root(library_root)
    saved: list[SavedAnalysis] = []
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink() or child.name.startswith(".tmp-"):
            continue
        try:
            saved.append(
                validate_analysis(
                    child,
                    library_root=root,
                    verify_hashes=verify_hashes,
                )
            )
        except AnalysisLibraryError:
            continue
    return sorted(
        saved,
        key=lambda item: (str(item.manifest["saved_at"]), item.analysis_id),
        reverse=True,
    )


def _find_duplicate_run_id(run_id: str, library_root: Path) -> SavedAnalysis | None:
    for saved in list_saved_analyses(library_root=library_root, verify_hashes=False):
        if saved.analysis_run_id == run_id:
            return saved
    return None


def save_analysis(
    source_dir: str | Path,
    product_name: str,
    source_type: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    library_root: str | Path = DEFAULT_ANALYSIS_LIBRARY_DIR,
    trusted_current_outputs: bool = False,
    now: datetime | None = None,
    token_factory: Callable[[], str] | None = None,
) -> SavedAnalysis:
    """Atomically snapshot one completed seven-output analysis.

    ``trusted_current_outputs`` is deliberately explicit and only succeeds for
    the project's three known current output roots.  Arbitrary directories
    must supply ``analysis_candidate`` and ``taxonomy_version`` in ``metadata``;
    they are never silently labelled as current-compatible.
    """

    source_root = Path(source_dir).expanduser().resolve(strict=False)
    _assert_required_source_files(source_root)
    destination_root = _safe_library_root(library_root)
    if source_root == destination_root or destination_root in source_root.parents:
        raise LibraryPathError("The source analysis cannot be the Analysis Library itself.")

    name = str(product_name or "").strip() or "Untitled Product"
    source = str(source_type or "").strip().casefold()
    if not source:
        raise ManifestValidationError("source_type must not be empty.")
    overrides = dict(metadata or {})
    run_metadata = _read_json_object(source_root / "analysis_run_metadata.json")
    is_trusted = trusted_current_outputs and _is_explicitly_trusted_current_output(source_root)
    if trusted_current_outputs and not is_trusted:
        raise LibraryPathError(
            "Only the project's current Demo, Session, or Amazon outputs can "
            "be marked as trusted current outputs."
        )

    analysis_candidate = _normalize_optional_text(overrides.get("analysis_candidate"))
    taxonomy_version = _normalize_optional_text(overrides.get("taxonomy_version"))
    if is_trusted:
        analysis_candidate = analysis_candidate or _normalize_optional_text(
            run_metadata.get("candidate")
        ) or ANALYSIS_CANDIDATE_VERSION
        taxonomy_version = taxonomy_version or _normalize_optional_text(
            run_metadata.get("taxonomy_version")
        ) or ANALYSIS_CANDIDATE_VERSION
    if not analysis_candidate or not taxonomy_version:
        raise ManifestValidationError(
            "analysis_candidate and taxonomy_version are required for an "
            "untrusted analysis source."
        )

    hashes = {
        filename: _sha256(source_root / filename)
        for filename in REQUIRED_ANALYSIS_FILES
    }
    aggregate_hash = _combined_hash(hashes)
    run_id = _normalize_optional_text(overrides.get("analysis_run_id"))
    if not run_id and is_trusted:
        run_id = _normalize_optional_text(run_metadata.get("run_id"))
    run_id = run_id or f"RUN_{aggregate_hash[:16]}"
    duplicate = _find_duplicate_run_id(run_id, destination_root)
    if duplicate is not None:
        raise DuplicateAnalysisError(run_id, duplicate.analysis_id)

    facts = _source_facts(source_root)
    for field in (
        "imported_reviews",
        "successful_reviews",
        "failed_reviews",
        "skipped_reviews",
    ):
        if field in overrides:
            facts[field] = _coerce_nonnegative_int(overrides[field], facts[field])
    facts["coverage"] = _coerce_coverage(
        overrides.get("coverage"),
        facts["successful_reviews"] / facts["imported_reviews"]
        if facts["imported_reviews"]
        else 0.0,
    )
    if "has_rating" in overrides:
        facts["has_rating"] = bool(overrides["has_rating"])
    if _normalize_optional_text(overrides.get("sample_level")):
        facts["sample_level"] = str(overrides["sample_level"]).strip()

    saved_at = now or _utc_now()
    factory = token_factory or (lambda: secrets.token_hex(2))
    analysis_id = ""
    final_root: Path | None = None
    for _ in range(32):
        candidate_id = _analysis_id(saved_at, factory)
        candidate_root = destination_root / f"{_safe_slug(name)}_{candidate_id.casefold()}"
        if not candidate_root.exists() and _SAFE_DIRECTORY_RE.fullmatch(candidate_root.name):
            analysis_id = candidate_id
            final_root = candidate_root
            break
    if final_root is None:
        raise AnalysisLibraryError("Unable to allocate a unique saved-analysis ID.")

    manifest: dict[str, Any] = {
        "analysis_id": analysis_id,
        "product_name": name,
        "product_id": _normalize_optional_text(overrides.get("product_id")),
        "asin": _normalize_optional_text(overrides.get("asin") or overrides.get("ASIN")),
        "source_type": source,
        "source_url": _normalize_optional_text(overrides.get("source_url")),
        "analysis_run_id": run_id,
        "analysis_candidate": analysis_candidate,
        "taxonomy_version": taxonomy_version,
        "model_provider": _normalize_optional_text(overrides.get("model_provider"))
        or (LLM_PROVIDER if is_trusted else "unknown"),
        "model_name": _normalize_optional_text(overrides.get("model_name"))
        or _normalize_optional_text(run_metadata.get("model") if is_trusted else None)
        or (OLLAMA_MODEL if is_trusted else "unknown"),
        "imported_reviews": facts["imported_reviews"],
        "successful_reviews": facts["successful_reviews"],
        "failed_reviews": facts["failed_reviews"],
        "skipped_reviews": facts["skipped_reviews"],
        "coverage": facts["coverage"],
        "sample_level": facts["sample_level"],
        "has_rating": facts["has_rating"],
        "saved_at": _iso_utc(saved_at),
        "source_output_path": str(source_root),
        "schema_version": ANALYSIS_LIBRARY_SCHEMA_VERSION,
        "content_hash": aggregate_hash,
        "content_hashes": hashes,
    }

    temporary_root = destination_root / f".tmp-{analysis_id.casefold()}-{secrets.token_hex(4)}"
    if not _is_direct_child(temporary_root, destination_root):
        raise LibraryPathError("Unsafe Analysis Library destination.")
    try:
        temporary_root.mkdir(mode=0o700)
        for filename in REQUIRED_ANALYSIS_FILES:
            shutil.copyfile(source_root / filename, temporary_root / filename)
        _write_json(temporary_root / "analysis_manifest.json", manifest)
        # Verify the invisible staging snapshot before the atomic directory move.
        for filename, expected in hashes.items():
            if _sha256(temporary_root / filename) != expected:
                raise SnapshotIntegrityError(
                    f"Snapshot copy failed integrity verification: {filename}"
                )
        os.replace(temporary_root, final_root)
    except Exception:
        if temporary_root.is_dir() and _is_direct_child(temporary_root, destination_root):
            shutil.rmtree(temporary_root)
        raise

    return validate_analysis(final_root, library_root=destination_root)


__all__ = [
    "ANALYSIS_LIBRARY_SCHEMA_VERSION",
    "DEFAULT_ANALYSIS_LIBRARY_DIR",
    "REQUIRED_ANALYSIS_FILES",
    "HASH_PROTECTED_FILES",
    "AnalysisLibraryError",
    "DuplicateAnalysisError",
    "LibraryPathError",
    "ManifestValidationError",
    "MissingAnalysisOutputError",
    "SnapshotIntegrityError",
    "SavedAnalysis",
    "get_manifest",
    "list_saved_analyses",
    "load_analysis",
    "save_analysis",
    "validate_analysis",
]
