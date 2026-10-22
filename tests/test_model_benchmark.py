"""Offline tests for the serial model benchmark calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import LLMClientError, LLMConnectionError  # noqa: E402
from model_benchmark import benchmark_models  # noqa: E402


def _case(case_id: str, review: str, aspects: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "review_text": review,
        "expected_aspects": [
            {"aspect": aspect, "sentiment": sentiment}
            for aspect, sentiment in aspects
        ],
    }


class _TickClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class _BenchmarkClient:
    def __init__(
        self,
        model: str,
        responses: dict[str, list[dict[str, str]] | Exception],
        events: list[tuple[str, str]],
    ) -> None:
        self.model = model
        self.responses = responses
        self.events = events

    def analyze_review(self, review_text: str, allowed_aspects: Any) -> dict[str, Any]:
        self.events.append((self.model, review_text))
        response = self.responses[review_text]
        if isinstance(response, Exception):
            raise response
        return {"summary": "Result.", "aspects": response}


class _TransientClient:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def analyze_review(self, review_text: str, allowed_aspects: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.failures:
            raise LLMConnectionError("temporarily unavailable")
        return {"summary": "Result.", "aspects": []}


class ModelBenchmarkTests(unittest.TestCase):
    def test_exact_match_is_order_independent_and_empty_case_matches(self) -> None:
        cases = [
            _case("A", "picture sound", [("Picture Quality", "positive"), ("Sound Quality", "negative")]),
            _case("B", "generic", []),
        ]
        responses = {
            "picture sound": [
                {"raw_aspect": "sound", "aspect": "Sound Quality", "sentiment": "negative", "evidence": "sound"},
                {"raw_aspect": "picture", "aspect": "Picture Quality", "sentiment": "positive", "evidence": "picture"},
            ],
            "generic": [],
        }
        events: list[tuple[str, str]] = []

        results = benchmark_models(
            cases,
            ["model-a"],
            client_factory=lambda model: _BenchmarkClient(model, responses, events),
            clock=_TickClock(),
        )

        self.assertEqual(2, results[0]["exact_matches"])
        self.assertEqual(0, results[0]["errors"])

    def test_duplicate_expected_mentions_are_not_collapsed(self) -> None:
        cases = [
            _case(
                "M005",
                "menu apps",
                [
                    ("System Performance", "negative"),
                    ("System Performance", "negative"),
                ],
            )
        ]
        responses = {
            "menu apps": [
                {"raw_aspect": "menu", "aspect": "System Performance", "sentiment": "negative", "evidence": "menu"}
            ]
        }

        results = benchmark_models(
            cases,
            ["model-a"],
            client_factory=lambda model: _BenchmarkClient(model, responses, []),
            clock=_TickClock(),
        )

        self.assertEqual(0, results[0]["exact_matches"])
        self.assertEqual(0, results[0]["errors"])

    def test_models_and_cases_run_strictly_serially(self) -> None:
        cases = [_case("A", "one", []), _case("B", "two", [])]
        responses: dict[str, list[dict[str, str]] | Exception] = {
            "one": [],
            "two": [],
        }
        events: list[tuple[str, str]] = []

        benchmark_models(
            cases,
            ["model-a", "model-b"],
            client_factory=lambda model: _BenchmarkClient(model, responses, events),
            clock=_TickClock(),
        )

        self.assertEqual(
            [
                ("model-a", "one"),
                ("model-a", "two"),
                ("model-b", "one"),
                ("model-b", "two"),
            ],
            events,
        )

    def test_case_error_is_counted_and_next_case_continues(self) -> None:
        cases = [_case("A", "bad", []), _case("B", "good", [])]
        responses: dict[str, list[dict[str, str]] | Exception] = {
            "bad": LLMClientError("inference failed"),
            "good": [],
        }

        results = benchmark_models(
            cases,
            ["model-a"],
            client_factory=lambda model: _BenchmarkClient(model, responses, []),
            clock=_TickClock(),
        )

        self.assertEqual(1, results[0]["errors"])
        self.assertEqual(1, results[0]["exact_matches"])
        self.assertEqual(5.0, results[0]["total_runtime"])
        self.assertEqual(2.5, results[0]["average_runtime"])

    def test_one_transient_connection_failure_is_retried_and_reported(self) -> None:
        client = _TransientClient(failures=1)

        results = benchmark_models(
            [_case("A", "review", [])],
            ["model-a"],
            client_factory=lambda model: client,
            clock=_TickClock(),
        )

        self.assertEqual(2, client.calls)
        self.assertEqual(1, results[0]["transient_retries"])
        self.assertEqual(0, results[0]["errors"])
        self.assertEqual(1, results[0]["exact_matches"])

    def test_transient_connection_failure_is_retried_only_once(self) -> None:
        client = _TransientClient(failures=2)

        results = benchmark_models(
            [_case("A", "review", [])],
            ["model-a"],
            client_factory=lambda model: client,
            clock=_TickClock(),
        )

        self.assertEqual(2, client.calls)
        self.assertEqual(1, results[0]["transient_retries"])
        self.assertEqual(1, results[0]["errors"])
        self.assertEqual(0, results[0]["exact_matches"])


if __name__ == "__main__":
    unittest.main()
