"""Offline tests for presentation-safe release metadata loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from release_metadata import get_app_version


class ReleaseMetadataTests(unittest.TestCase):
    def test_reads_version_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "release_manifest.json"
            path.write_text(json.dumps({"app_version": "1.0.0"}), encoding="utf-8")
            self.assertEqual("1.0.0", get_app_version(path))

    def test_invalid_or_missing_manifest_fails_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "release_manifest.json"
            self.assertEqual("unknown", get_app_version(path))
            path.write_text("not json", encoding="utf-8")
            self.assertEqual("unknown", get_app_version(path))


if __name__ == "__main__":
    unittest.main()
