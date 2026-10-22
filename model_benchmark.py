"""Serial, real-model benchmark for the curated single-review cases."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from config import OLLAMA_BENCHMARK_MODELS, TESTS_DIR
from llm_client import (
    LLMClientError,
    LLMConnectionError,
    OllamaReviewClient,
    ReviewAnalysisClient,
)
from review_analyzer import ReviewAnalysisError, analyze_review


MANUAL_CASES_PATH = TESTS_DIR / "manual_review_cases.json"


def load_manual_cases(path: Path = MANUAL_CASES_PATH) -> list[dict[str, Any]]:
    """Load the fixed human-labelled cases used by both accuracy and timing."""

    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manual benchmark cases must be a non-empty list.")
    return cases


def _aspect_sentiment_counter(
    aspects: Sequence[Mapping[str, Any]],
) -> Counter[tuple[str, str]]:
    """Preserve duplicate mentions such as the two M005 performance findings."""

    return Counter(
        (str(aspect["aspect"]), str(aspect["sentiment"]))
        for aspect in aspects
    )


def _counter_items(counter: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {"aspect": aspect, "sentiment": sentiment, "count": count}
        for (aspect, sentiment), count in sorted(counter.items())
    ]


def benchmark_models(
    cases: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    *,
    client_factory: Callable[[str], ReviewAnalysisClient] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict[str, Any]]:
    """Run every model and case serially; no threads or concurrent inference."""

    factory = client_factory or (lambda model: OllamaReviewClient(model=model))
    results: list[dict[str, Any]] = []

    for model in models:
        client = factory(model)
        exact_matches = 0
        errors = 0
        transient_retries = 0
        case_results: list[dict[str, Any]] = []
        model_started = clock()

        for case in cases:
            case_started = clock()
            expected = _aspect_sentiment_counter(case.get("expected_aspects", []))
            case_retries = 0
            actual: Counter[tuple[str, str]] = Counter()
            error = ""
            is_exact = False
            while True:
                try:
                    analysis = analyze_review(
                        str(case["review_text"]),
                        llm_client=client,
                    )
                except LLMConnectionError as exc:
                    if case_retries == 0:
                        case_retries = 1
                        transient_retries += 1
                        continue
                    errors += 1
                    error = str(exc)
                    break
                except (LLMClientError, ReviewAnalysisError) as exc:
                    errors += 1
                    error = str(exc)
                    break
                else:
                    actual = _aspect_sentiment_counter(analysis["aspects"])
                    is_exact = actual == expected
                    if is_exact:
                        exact_matches += 1
                    break
            case_runtime = clock() - case_started
            case_results.append(
                {
                    "case_id": str(case["case_id"]),
                    "is_exact": is_exact,
                    "error": error,
                    "transient_retries": case_retries,
                    "runtime": case_runtime,
                    "expected": _counter_items(expected),
                    "actual": _counter_items(actual),
                }
            )

        total_runtime = clock() - model_started
        total_cases = len(cases)
        warm_runtimes = [item["runtime"] for item in case_results[1:]]
        results.append(
            {
                "model": model,
                "total_cases": total_cases,
                "exact_matches": exact_matches,
                "errors": errors,
                "transient_retries": transient_retries,
                "total_runtime": total_runtime,
                "average_runtime": total_runtime / total_cases if total_cases else 0.0,
                "first_review_runtime": (
                    case_results[0]["runtime"] if case_results else 0.0
                ),
                "warm_average_runtime": (
                    sum(warm_runtimes) / len(warm_runtimes)
                    if warm_runtimes
                    else 0.0
                ),
                "case_results": case_results,
            }
        )

    return results


def _format_pairs(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "(none)"
    rendered: list[str] = []
    for item in items:
        label = f"{item['aspect']} -> {item['sentiment']}"
        count = int(item.get("count", 1))
        rendered.append(f"{label} x{count}" if count > 1 else label)
    return "; ".join(rendered)


def print_benchmark_results(results: Sequence[Mapping[str, Any]]) -> None:
    print("Amazon Review Analyzer - Model Benchmark")
    print("Execution: serial")
    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"Cases: {result['total_cases']}")
        print(
            f"Exact matches: {result['exact_matches']}/{result['total_cases']}"
        )
        print(f"Cases with errors: {result['errors']}")
        print(f"Transient connection retries: {result['transient_retries']}")
        print(f"Total runtime: {result['total_runtime']:.2f} sec")
        print(f"Average: {result['average_runtime']:.2f} sec/review")
        print(
            "Warm average (reviews 2-10): "
            f"{result['warm_average_runtime']:.2f} sec/review"
        )

        mismatches = [
            case for case in result["case_results"] if not case["is_exact"]
        ]
        if not mismatches:
            print("Mismatched cases: none")
            continue
        print("Mismatched cases:")
        for case in mismatches:
            print(f"  {case['case_id']}")
            print(f"    Expected: {_format_pairs(case['expected'])}")
            if case["error"]:
                print(f"    Error: {case['error']}")
            else:
                print(f"    Actual: {_format_pairs(case['actual'])}")


def run_model_benchmarks(
    models: Sequence[str] = OLLAMA_BENCHMARK_MODELS,
) -> int:
    """CLI entry: load the fixed cases, execute, and print real measurements."""

    try:
        cases = load_manual_cases()
        results = benchmark_models(cases, models)
    except (OSError, json.JSONDecodeError, ValueError, LLMClientError) as exc:
        print(f"Cannot run model benchmark: {exc}")
        return 1

    print_benchmark_results(results)
    return 1 if any(result["errors"] for result in results) else 0
