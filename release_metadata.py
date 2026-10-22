"""Read-only release metadata for presentation-layer consumers.

The dashboard deliberately does not import the analysis/runtime configuration
module.  Release health checks guarantee that the manifest version matches the
single authoritative version declared in ``config.py``.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RELEASE_MANIFEST_PATH = PROJECT_ROOT / "release_manifest_v1.0.1.json"
LEGACY_RELEASE_MANIFEST_PATH = PROJECT_ROOT / "release_manifest.json"


def get_app_version(manifest_path: Path = RELEASE_MANIFEST_PATH) -> str:
    """Return the validated-looking manifest version, or ``unknown`` safely."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        if manifest_path != RELEASE_MANIFEST_PATH:
            return "unknown"
        try:
            payload = json.loads(LEGACY_RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return "unknown"
    version = payload.get("app_version") if isinstance(payload, dict) else None
    if not isinstance(version, str) or not version.strip():
        return "unknown"
    return version.strip()
