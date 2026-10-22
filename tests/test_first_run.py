"""Offline tests for privacy-safe first-run state persistence."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from first_run import (
    FirstRunState,
    is_first_run_completed,
    load_first_run_state,
    mark_first_run_completed,
    reset_first_run_state_for_tests,
)


class FirstRunStateTests(unittest.TestCase):
    def test_missing_state_is_safely_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "app_state" / "first_run.json"
            self.assertEqual(FirstRunState(), load_first_run_state(path))
            self.assertFalse(is_first_run_completed(path))

    def test_completion_creates_parent_and_writes_only_allowed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "first_run.json"
            result = mark_first_run_completed(" 1.0.0 ", path)

            self.assertEqual(FirstRunState(True, "1.0.0"), result)
            self.assertTrue(is_first_run_completed(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                {"first_run_completed": True, "app_version": "1.0.0"},
                payload,
            )
            serialized = path.read_text(encoding="utf-8")
            for forbidden in ("review", "identity", "account", "path", "user"):
                self.assertNotIn(forbidden, serialized.lower())

    def test_malformed_or_schema_invalid_state_is_safely_incomplete(self) -> None:
        invalid_payloads = (
            "not json",
            json.dumps([]),
            json.dumps({"first_run_completed": "yes", "app_version": "1.0.0"}),
            json.dumps({"first_run_completed": True, "app_version": ""}),
            json.dumps({"first_run_completed": True, "app_version": 100}),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "first_run.json"
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    self.assertEqual(FirstRunState(), load_first_run_state(path))

    def test_safe_reader_ignores_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "first_run.json"
            path.write_text(
                json.dumps(
                    {
                        "first_run_completed": True,
                        "app_version": "1.0.0",
                        "review_text": "must not escape through the API",
                        "absolute_path": "/private/example",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                FirstRunState(True, "1.0.0"),
                load_first_run_state(path),
            )

    def test_invalid_app_version_is_rejected_without_creating_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "first_run.json"
            for version in ("", "   ", None):
                with self.subTest(version=version):
                    with self.assertRaises(ValueError):
                        mark_first_run_completed(version, path)  # type: ignore[arg-type]
            self.assertFalse(path.exists())

    def test_atomic_replace_failure_preserves_existing_state_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "first_run.json"
            original = {"first_run_completed": True, "app_version": "0.9.0"}
            path.write_text(json.dumps(original), encoding="utf-8")

            with patch("first_run.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    mark_first_run_completed("1.0.0", path)

            self.assertEqual(original, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual([], list(directory.glob(".first_run.json.*.tmp")))

    def test_successful_write_uses_atomic_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "first_run.json"
            real_replace = os.replace
            calls: list[tuple[Path, Path]] = []

            def replace(source: str | Path, destination: str | Path) -> None:
                calls.append((Path(source), Path(destination)))
                real_replace(source, destination)

            with patch("first_run.os.replace", side_effect=replace):
                mark_first_run_completed("1.0.0", path)

            self.assertEqual(1, len(calls))
            self.assertEqual(path, calls[0][1])
            self.assertEqual(path.parent, calls[0][0].parent)

    def test_test_only_reset_is_idempotent_and_requires_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "first_run.json"
            mark_first_run_completed("1.0.0", path)
            reset_first_run_state_for_tests(path)
            self.assertFalse(path.exists())
            reset_first_run_state_for_tests(path)
            self.assertFalse(path.exists())

        with self.assertRaises(TypeError):
            reset_first_run_state_for_tests()  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
