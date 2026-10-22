"""Offline tests for the Step 11 performance benchmark runner."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module  # noqa: E402
from config import OLLAMA_MODEL  # noqa: E402
from llm_client import LLMConnectionError  # noqa: E402
from performance_benchmark import (  # noqa: E402
    PERFORMANCE_DATASET_PATH,
    PerformanceBenchmarkError,
    load_performance_reviews,
    run_performance_benchmark,
    run_performance_comparison,
    select_runtime_strategy,
)


class _FakeBenchmarkClient:
    """Thread-safe provider fake with optional latency and transient failures."""

    def __init__(
        self,
        *,
        fail_counts: dict[str, int] | None = None,
        delays: dict[str, float] | None = None,
    ) -> None:
        self.fail_counts = dict(fail_counts or {})
        self.delays = dict(delays or {})
        self.calls: Counter[str] = Counter()
        self._lock = threading.Lock()

    def check_connection(self) -> dict[str, str]:
        return {
            "host": "http://localhost:11434",
            "model": OLLAMA_MODEL,
        }

    def analyze_review(
        self,
        review_text: str,
        allowed_aspects: Sequence[str],
    ) -> dict[str, Any]:
        del allowed_aspects
        with self._lock:
            self.calls[review_text] += 1
            attempt = self.calls[review_text]
        delay = self.delays.get(review_text, 0.0)
        if delay:
            time.sleep(delay)
        if attempt <= self.fail_counts.get(review_text, 0):
            raise LLMConnectionError("temporary Ollama connection failure")
        return {"summary": "Offline benchmark result.", "aspects": []}

    def repair_evidence(
        self,
        review_text: str,
        invalid_aspects: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        del review_text, invalid_aspects
        return {"repairs": []}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PerformanceBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_performance_reviews(review_count=20)
        cls.first_review = str(cls.source.iloc[0]["content"])

    def _run(
        self,
        directory: Path,
        *,
        workers: int,
        fake: _FakeBenchmarkClient | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], _FakeBenchmarkClient]:
        client = fake or _FakeBenchmarkClient()
        report, timings = run_performance_benchmark(
            review_count=10,
            workers=workers,
            output_dir=directory,
            client_factory=lambda: client,
            sleeper=lambda _: None,
        )
        return report, timings, client

    def test_workers_one_processes_exactly_ten_offline_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report, timings, _ = self._run(Path(temporary), workers=1)

            self.assertEqual(10, report["total_reviews"])
            self.assertEqual(10, report["successful"])
            self.assertEqual(0, report["failed"])
            self.assertEqual(0, report["skipped"])
            self.assertEqual(10, len(timings))
            self.assertTrue(report["schema_valid"])
            self.assertTrue(report["evidence_valid"])

    def test_workers_two_processes_ten_and_preserves_source_order(self) -> None:
        delayed = _FakeBenchmarkClient(delays={self.first_review: 0.03})
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            report, timings, _ = self._run(
                destination,
                workers=2,
                fake=delayed,
            )

            analyzed = pd.read_csv(destination / "workers_2" / "analyzed_reviews.csv")
            self.assertEqual(list(range(10)), analyzed["source_row_id"].tolist())
            self.assertEqual(list(range(10)), [row["source_row_id"] for row in timings])
            self.assertTrue(report["output_order_preserved"])
            self.assertEqual(10, report["successful"])

    def test_transient_provider_failure_retries_once_then_succeeds(self) -> None:
        fake = _FakeBenchmarkClient(fail_counts={self.first_review: 1})
        with tempfile.TemporaryDirectory() as temporary:
            report, timings, _ = self._run(Path(temporary), workers=1, fake=fake)

        first = next(row for row in timings if row["source_row_id"] == 0)
        self.assertEqual(2, fake.calls[self.first_review])
        self.assertEqual(1, first["retry_count"])
        self.assertEqual("success", first["status"])
        self.assertEqual(1, report["retries"])
        self.assertEqual(0, report["failed"])

    def test_second_provider_failure_is_recorded_without_infinite_retry(self) -> None:
        fake = _FakeBenchmarkClient(fail_counts={self.first_review: 99})
        with tempfile.TemporaryDirectory() as temporary:
            report, timings, _ = self._run(Path(temporary), workers=1, fake=fake)

        first = next(row for row in timings if row["source_row_id"] == 0)
        self.assertEqual(2, fake.calls[self.first_review])
        self.assertEqual(1, first["retry_count"])
        self.assertEqual("failed", first["status"])
        self.assertEqual("LLMConnectionError", first["error_type"])
        self.assertEqual(1, report["failed"])
        self.assertEqual(9, report["successful"])

    def test_run_writes_metrics_and_request_metadata_only_under_output_dir(self) -> None:
        protected = [
            PROJECT_ROOT / "data" / "raw_reviews.csv",
            PROJECT_ROOT / "outputs" / "analyzed_reviews.csv",
            PROJECT_ROOT / "outputs" / "session" / "current" / "raw_reviews.csv",
            PROJECT_ROOT / "outputs" / "amazon" / "current" / "raw_reviews.csv",
        ]
        before = {path: _sha256(path) for path in protected}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            comparison = run_performance_comparison(
                review_count=10,
                output_dir=destination,
                client_factory=lambda: _FakeBenchmarkClient(),
                sleeper=lambda _: None,
            )
            baseline = destination / "performance_baseline_10.json"
            timings_path = destination / "performance_request_timings_10.csv"
            self.assertTrue(baseline.is_file())
            self.assertTrue(timings_path.is_file())
            self.assertTrue(
                (destination / "reviews_10" / "workers_1" / "performance_run.json").is_file()
            )
            self.assertTrue(
                (destination / "reviews_10" / "workers_2" / "performance_run.json").is_file()
            )
            payload = json.loads(baseline.read_text(encoding="utf-8"))
            timings = pd.read_csv(timings_path)
            self.assertEqual(10, comparison["review_count"])
            self.assertEqual(2, len(payload["runs"]))
            self.assertEqual(20, len(timings))
            self.assertTrue(
                {
                    "source_row_id",
                    "review_id",
                    "workers",
                    "status",
                    "latency_seconds",
                    "retry_count",
                    "schema_valid",
                    "evidence_valid",
                }.issubset(timings.columns)
            )
        after = {path: _sha256(path) for path in protected}
        self.assertEqual(before, after)

    def test_baseline_cannot_be_overwritten_implicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            (destination / "performance_baseline_10.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                PerformanceBenchmarkError,
                "already exists",
            ):
                run_performance_comparison(
                    review_count=10,
                    output_dir=destination,
                    client_factory=lambda: _FakeBenchmarkClient(),
                    sleeper=lambda _: None,
                )

    def test_partial_output_prevents_an_implicit_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            partial = destination / "workers_1"
            partial.mkdir(parents=True)
            (partial / "performance_run.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                PerformanceBenchmarkError,
                "earlier run was interrupted",
            ):
                run_performance_comparison(
                    review_count=20,
                    output_dir=destination,
                    client_factory=lambda: _FakeBenchmarkClient(),
                    sleeper=lambda _: None,
                )

    def test_selection_gates_allow_optional_faster_mode_but_keep_default_one(self) -> None:
        serial = {
            "throughput_reviews_per_minute": 10.0,
            "failed": 0,
            "retries": 0,
            "P90_latency": 5.0,
        }
        parallel = {
            "throughput_reviews_per_minute": 12.5,
            "failed": 0,
            "retries": 1,
            "P90_latency": 7.0,
            "oom_or_crash": False,
            "schema_valid": True,
            "evidence_valid": True,
            "output_order_preserved": True,
        }

        decision = select_runtime_strategy(serial, parallel)

        self.assertTrue(decision["optional_faster_mode_allowed"])
        self.assertEqual(1, decision["production_default_workers"])
        self.assertTrue(all(decision["gates"].values()))

    def test_selection_rejects_faster_mode_when_any_safety_gate_fails(self) -> None:
        serial = {
            "throughput_reviews_per_minute": 10.0,
            "failed": 0,
            "retries": 0,
            "P90_latency": 5.0,
        }
        parallel = {
            "throughput_reviews_per_minute": 14.0,
            "failed": 1,
            "retries": 0,
            "P90_latency": 6.0,
            "oom_or_crash": False,
            "schema_valid": True,
            "evidence_valid": True,
            "output_order_preserved": True,
        }

        decision = select_runtime_strategy(serial, parallel)

        self.assertFalse(decision["optional_faster_mode_allowed"])
        self.assertFalse(decision["gates"]["failures_not_increased"])
        self.assertEqual("Stable mode (1 worker)", decision["recommended_strategy"])

    def test_selection_cannot_pass_when_both_modes_fail_every_review(self) -> None:
        serial = {
            "total_reviews": 20,
            "successful": 0,
            "failed": 20,
            "skipped": 0,
            "throughput_reviews_per_minute": 0.0,
            "retries": 20,
            "P90_latency": 1.0,
        }
        parallel = {
            **serial,
            "schema_valid": True,
            "evidence_valid": True,
            "oom_or_crash": False,
            "output_order_preserved": True,
        }

        decision = select_runtime_strategy(serial, parallel)

        self.assertFalse(decision["optional_faster_mode_allowed"])
        self.assertFalse(decision["gates"]["all_attempted_reviews_succeeded"])

    def test_review_count_and_worker_arguments_are_restricted(self) -> None:
        with self.assertRaisesRegex(ValueError, "review_count must be 10 or 20"):
            load_performance_reviews(PERFORMANCE_DATASET_PATH, review_count=11)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "workers must be 1 or 2"):
                run_performance_benchmark(
                    review_count=10,
                    workers=3,
                    output_dir=temporary,
                    client_factory=lambda: _FakeBenchmarkClient(),
                )

    def test_cli_routes_ten_review_performance_benchmark(self) -> None:
        with patch.object(
            main_module,
            "run_large_dataset_performance_benchmark",
            return_value=0,
        ) as runner:
            exit_code = main_module.cli(
                ["--benchmark-performance", "--reviews", "10"]
            )

        self.assertEqual(0, exit_code)
        runner.assert_called_once_with(10)

    def test_cli_rejects_reviews_without_performance_mode(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main_module.cli(["--reviews", "10"])

        self.assertEqual(2, exit_code)
        self.assertIn(
            "--reviews can only be used with --benchmark-performance.",
            output.getvalue(),
        )

    def test_cli_rejects_unsupported_review_count(self) -> None:
        with (
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main_module.cli(["--benchmark-performance", "--reviews", "15"])

        self.assertEqual(2, raised.exception.code)


if __name__ == "__main__":
    unittest.main()
