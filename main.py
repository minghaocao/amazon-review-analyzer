"""Amazon Review Analyzer 的命令行入口。"""

from __future__ import annotations

import sys


# Health diagnostics must remain usable even when a third-party dependency is
# missing.  Dispatch these two exact CLI modes before importing pandas or any
# application module that depends on the project environment.
if __name__ == "__main__" and sys.argv[1:] in (
    ["--health-check"],
    ["--health-check-json"],
):
    from environment_health import run_health_cli

    raise SystemExit(
        run_health_cli(json_output=sys.argv[1:] == ["--health-check-json"])
    )

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from amazon_access import (
    AccessStatus,
    AmazonAccessReport,
    DEFAULT_AMAZON_DIAGNOSTIC_URL,
    run_amazon_access_diagnostic,
)
from amazon_url import AmazonURLValidationError
from aggregator import (
    AggregationError,
    aggregate_reviews,
    load_analyzed_reviews,
    save_aggregation_outputs,
)
from batch_analyzer import CheckpointError, run_checkpointed_batch
from cleaner import clean_reviews, summarize_cleaning
from config import (
    ANALYZED_REVIEWS_PATH,
    ANALYSIS_CHECKPOINT_META_PATH,
    ANALYSIS_CHECKPOINT_PATH,
    ASPECT_EVIDENCE_PATH,
    ASPECT_KEYWORDS_PATH,
    ASPECT_SUMMARY_PATH,
    BATCH_CHECKPOINT_INTERVAL,
    BUSINESS_SUMMARY_PATH,
    KEYWORD_MIN_REVIEW_COUNT,
    KEYWORD_SUMMARY_PATH,
    LLM_PROVIDER,
    MAX_CONSECUTIVE_LLM_FAILURES,
    MIN_ASPECT_REVIEWS_FOR_RANKING,
    OLLAMA_MODEL,
    PROJECT_ROOT,
    RAW_REVIEWS_PATH,
    REQUIRED_DIRECTORIES,
    TESTS_DIR,
    SENTIMENT_KEYWORDS_PATH,
)
from evaluator import (
    EVALUATION_DIR,
    RESULTS_PATH as EVALUATION_RESULTS_PATH,
    check_regression,
    evaluate_models,
    format_evaluation_report,
    format_model_comparison,
    run_evaluation,
)
from input_loader import load_reviews, validate_reviews
from holdout_evaluator import (
    HoldoutIntegrityError,
    format_holdout_report,
    run_holdout_evaluation,
    save_holdout_outputs,
)
from generalization_evaluator import (
    GeneralizationIntegrityError,
    format_generalization_report,
    run_generalization_evaluation,
    save_generalization_evaluation,
)
from final_blind_evaluator import (
    FinalBlindIntegrityError,
    final_outputs_exist,
    format_final_blind_report,
    run_final_blind_evaluation,
    save_final_blind_outputs,
)
from environment_health import run_health_cli
from generalization_error_analysis import (
    GeneralizationStudyError,
    format_generalization_error_study,
    run_generalization_error_study,
)
from generalization_dev_v2_evaluator import (
    GeneralizationDevV2IntegrityError,
    baseline_outputs_exist as generalization_dev_v2_outputs_exist,
    format_generalization_dev_v2_report,
    run_generalization_dev_v2_baseline,
    save_generalization_dev_v2_outputs,
)
from generalization_dev_v2_optimizer import (
    BEST_REPORT_PATH as GENERALIZATION_DEV_V2_BEST_PATH,
    format_optimization_round_report,
    load_optimization_round_reports,
    optimization_round_outputs_exist,
    run_generalization_dev_v2_optimization_round,
    save_best_round,
    save_optimization_round_outputs,
    verify_frozen_step_811_inputs,
)
from generalization_dev_v2_repair import (
    REGRESSION_ANALYSIS_PATH as GENERALIZATION_V2_REGRESSION_ANALYSIS_PATH,
    REGRESSION_CASES_PATH as GENERALIZATION_V2_REGRESSION_CASES_PATH,
    format_repair_candidate_report,
    generate_and_save_round1_regression_analysis,
    repair_outputs_exist,
    run_repair_candidate_evaluation,
    save_repair_candidate_outputs,
)
from blind_v2_evaluator import (
    BlindV2IntegrityError,
    evaluation_outputs_exist as blind_v2_outputs_exist,
    format_blind_v2_report,
    run_blind_v2_evaluation,
    save_blind_v2_outputs,
)
from keyword_analyzer import (
    KeywordAnalysisError,
    analyze_keywords,
    load_keyword_reviews,
    save_keyword_outputs,
)
from llm_client import (
    LLMClientError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    OllamaReviewClient,
    ReviewAnalysisClient,
    check_ollama_connection,
    create_review_client,
)
from model_benchmark import run_model_benchmarks
from performance_benchmark import (
    PERFORMANCE_BASELINE_PATH,
    PERFORMANCE_TIMINGS_PATH,
    PerformanceBenchmarkError,
    format_performance_report,
    run_performance_comparison,
)
from review_analyzer import ReviewAnalysisError, analyze_review
from release_health_check import (
    format_health_report,
    run_release_health_check as execute_release_health_check,
)


MANUAL_CASES_PATH = TESTS_DIR / "manual_review_cases.json"
EVALUATION_MODELS = ("qwen3:8b", "qwen3:4b-instruct")
EVALUATION_BASELINE_PATH = EVALUATION_DIR / "baseline_qwen3_8b.json"


def run_release_health_check_command() -> int:
    """Run the deterministic offline v1.0 release checks."""

    report = execute_release_health_check()
    print(format_health_report(report))
    return 0 if report.passed else 1


def check_project_setup() -> tuple[bool, list[str]]:
    """检查 Step 1 必需的目录是否存在。"""

    missing = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_DIRECTORIES if not path.is_dir()]
    return not missing, missing


def _display_text(value: object, fallback: str) -> str:
    """为命令行预览安全显示可能为空的文本。"""

    if pd.isna(value):
        return fallback
    text = str(value)
    return text if text.strip() else fallback


def _print_validation_summary(result: dict[str, Any]) -> None:
    print("\nValidation Summary")
    print(f"Total reviews: {result['total_reviews']}")
    print(f"Missing content: {result['missing_content']}")
    print(f"Missing titles: {result['missing_titles']}")
    print(f"Missing review IDs: {result['missing_review_ids']}")
    print(f"Duplicate review IDs: {result['duplicate_review_ids']}")
    print(f"Invalid ratings: {result['invalid_ratings']}")
    print(f"Non-numeric ratings: {result['non_numeric_ratings']}")
    print(f"Ratings below 1: {result['ratings_below_range']}")
    print(f"Ratings above 5: {result['ratings_above_range']}")


def _print_cleaning_summary(result: dict[str, Any]) -> None:
    print("\nCleaning Summary")
    print(f"Total reviews: {result['total_reviews']}")
    print(f"Analyzable reviews: {result['analyzable_reviews']}")
    print(f"Skipped reviews: {result['skipped_reviews']}")
    print("Skip reasons:")
    for reason, count in result["skip_reasons"].items():
        print(f"  {reason}: {count}")


def _print_review_analysis(result: dict[str, Any]) -> None:
    print(f"Summary:\n{result['summary']}")
    print("\nAspects:")

    if not result["aspects"]:
        print("(none)")

    for aspect in result["aspects"]:
        print(f"\nRaw Aspect: {aspect['raw_aspect']}")
        print(f"Normalized Aspect: {aspect['aspect']}")
        print(f"Sentiment: {aspect['sentiment']}")
        print(f"Evidence: {aspect['evidence']}")

    print("\nPositive points:")
    for point in result["positive_points"]:
        print(f"- {point}")

    print("\nNegative points:")
    for point in result["negative_points"]:
        print(f"- {point}")


def _print_expected_aspects(case: dict[str, Any]) -> None:
    aspects = case.get("expected_aspects", [])
    if not aspects:
        print("  (no specific aspects)")
        return
    for aspect in aspects:
        print(f"  {aspect['aspect']} -> {aspect['sentiment']}")


def run_ollama_check(client: OllamaReviewClient | None = None) -> int:
    """Run an explicit local-service/model check without generating text."""

    print("LLM provider: ollama")
    try:
        status = check_ollama_connection(client)
    except LLMClientError as exc:
        print(exc)
        return 1

    print("Ollama running.")
    print(f"Host: {status['host']}")
    print(f"Model found: {status['model']}")
    return 0


def _print_amazon_access_step(report: AmazonAccessReport, name: str) -> None:
    step = report.get_step(name)
    labels = {
        "dns_resolution": "DNS resolution",
        "amazon_home": "Amazon marketplace homepage",
        "product_url": "Canonical product URL",
        "product_reviews_url": "Product reviews URL",
        "external_https": "External HTTPS sanity check",
    }
    print(f"\n{labels[name]}")
    print(f"URL: {step.url}")
    print(f"Attempted: {'yes' if step.attempted else 'no'}")
    print(f"Status: {step.status}")
    if step.resolved_addresses is not None:
        print(f"Resolved addresses: {step.resolved_addresses}")
    if step.attempted and name != "dns_resolution":
        connection_established = step.http_status is not None
        print(
            "TCP/HTTPS connection: "
            f"{'established' if connection_established else 'failed'}"
        )
        print(f"HTTP status: {step.http_status if step.http_status is not None else 'n/a'}")
        print(f"Final URL: {step.final_url or 'n/a'}")
        print(f"Redirected: {'yes' if step.redirected else 'no'}")
        print(f"Redirect target: {step.final_url if step.redirected else 'none'}")
        print(f"Content type: {step.content_type or 'n/a'}")
        print(f"Response size: {step.response_size} bytes")
        print(
            "Amazon challenge detected: "
            f"{'yes' if step.challenge_detected else 'no'}"
        )
        if name == "product_reviews_url":
            print(
                "Review container detected: "
                f"{'yes' if step.review_container_detected else 'no'}"
            )
    if step.exception_type:
        print(f"Exception type: {step.exception_type}")
    if step.error_message:
        print(f"Detail: {step.error_message}")


def run_amazon_access_check(
    product_url: str = DEFAULT_AMAZON_DIAGNOSTIC_URL,
) -> int:
    """Print a read-only diagnosis using the exact scraper HTTP stack."""

    print("Amazon Access Diagnostics")
    try:
        report = run_amazon_access_diagnostic(product_url)
    except AmazonURLValidationError as exc:
        print(f"Invalid Amazon URL: {exc}")
        return 2

    print(f"HTTP client: {report.http_library}")
    print(f"Timeout: {report.timeout_seconds:g} seconds")
    print(f"Redirects enabled: {'yes' if report.redirects_enabled else 'no'}")
    print(
        "TLS certificate verification: "
        f"{'enabled' if report.tls_verification_enabled else 'disabled'}"
    )
    print("Proxy environment (values are intentionally hidden):")
    for name, present in report.proxy_environment.items():
        print(f"  {name}: {'present' if present else 'not set'}")

    for name in (
        "external_https",
        "dns_resolution",
        "amazon_home",
        "product_url",
        "product_reviews_url",
    ):
        _print_amazon_access_step(report, name)

    print(f"\nOverall status: {report.overall_status}")
    print("No Amazon or Demo output files were written.")
    return 0 if report.overall_status == AccessStatus.SUCCESS else 1


def run_manual_tests(llm_client: ReviewAnalysisClient | None = None) -> int:
    """Show Expected/Actual for the ten curated cases without scoring them."""

    try:
        cases = json.loads(MANUAL_CASES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Cannot load manual review cases: {exc}")
        return 1

    try:
        client = llm_client or create_review_client()
    except LLMClientError as exc:
        print(exc)
        return 1

    print("Amazon Review Analyzer - Manual Validation")
    print(f"LLM provider: {LLM_PROVIDER}")
    errors = 0

    for case in cases:
        case_id = case["case_id"]
        review_text = case["review_text"]
        print(f"\nCase ID: {case_id}")
        print(f"Original Review: {review_text}")
        print("Expected:")
        _print_expected_aspects(case)
        print("Actual:")

        try:
            result = analyze_review(review_text, llm_client=client)
        except (LLMConfigurationError, LLMConnectionError, LLMModelNotFoundError) as exc:
            print(f"  ERROR: {exc}")
            return 1
        except (LLMClientError, ReviewAnalysisError) as exc:
            print(f"  ERROR: {exc}")
            errors += 1
            continue

        print(f"  Summary: {result['summary']}")
        if not result["aspects"]:
            print("  (no specific aspects)")
        for aspect in result["aspects"]:
            print(f"  Raw Aspect: {aspect['raw_aspect']}")
            print(f"  Normalized Aspect: {aspect['aspect']}")
            print(f"  Sentiment: {aspect['sentiment']}")
            print(f"  Evidence: {aspect['evidence']}")

    print(f"\nManual cases processed: {len(cases)}")
    print(f"Cases with errors: {errors}")
    return 1 if errors else 0


def run_quality_evaluation() -> int:
    """Run the configured provider/model against the isolated gold dataset."""

    print("Amazon Review Analyzer — Evaluation")
    try:
        report = run_evaluation()
    except (LLMClientError, OSError, TypeError, ValueError) as exc:
        print(f"Cannot complete evaluation: {exc}")
        return 1

    print()
    print(format_evaluation_report(report))
    return 0


def run_quality_model_comparison() -> int:
    """Evaluate the two interview models serially without changing the baseline."""

    print("Amazon Review Analyzer — Model Evaluation")
    try:
        reports = evaluate_models(models=EVALUATION_MODELS)
    except (LLMClientError, OSError, TypeError, ValueError) as exc:
        print(f"Cannot complete model comparison: {exc}")
        return 1

    print()
    print(format_model_comparison(reports))
    return 0


def run_evaluation_regression_check() -> int:
    """Compare the latest evaluation with baseline; warnings are non-blocking."""

    print("Amazon Review Analyzer — Regression Check")
    try:
        current_report = json.loads(
            EVALUATION_RESULTS_PATH.read_text(encoding="utf-8")
        )
        result = check_regression(
            current_report,
            baseline_path=EVALUATION_BASELINE_PATH,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"Cannot check regression: {exc}")
        return 1

    if result["has_regression"]:
        print("\nREGRESSION WARNING")
        for warning in result["warnings"]:
            print(f"- {warning}")
    else:
        print("\nNo regression detected.")
    return 0


def run_blind_holdout_evaluation() -> int:
    """Measure the frozen v1 analyzer on the SHA-locked blind holdout set."""

    print("Amazon Review Analyzer — Blind Holdout Evaluation")
    if LLM_PROVIDER != "ollama" or OLLAMA_MODEL != "qwen3:8b":
        print(
            "Step 8.7 is locked to the local Ollama qwen3:8b baseline. "
            f"Current configuration: {LLM_PROVIDER} / {OLLAMA_MODEL}."
        )
        return 1
    try:
        report = run_holdout_evaluation(
            provider=LLM_PROVIDER,
            model=OLLAMA_MODEL,
        )
        paths = save_holdout_outputs(report)
    except (
        HoldoutIntegrityError,
        LLMClientError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cannot complete blind holdout evaluation: {exc}")
        return 1

    print(format_holdout_report(report))
    print("\nHoldout outputs")
    print(f"Report: {paths['report']}")
    print(f"Errors: {paths['errors']}")
    print(f"Predictions: {paths['predictions']}")
    return 0


def run_generalization_dev_evaluation(stage: str = "optimized") -> int:
    """Evaluate the frozen Step 8.8 development expansion set."""

    print("Amazon Review Analyzer — Generalization Development Evaluation")
    print(f"Stage: {stage}")
    if LLM_PROVIDER != "ollama" or OLLAMA_MODEL != "qwen3:8b":
        print(
            "Step 8.8 evaluation is locked to local Ollama qwen3:8b. "
            f"Current configuration: {LLM_PROVIDER} / {OLLAMA_MODEL}."
        )
        return 1
    try:
        report = run_generalization_evaluation(
            provider=LLM_PROVIDER,
            model=OLLAMA_MODEL,
        )
        paths = save_generalization_evaluation(report, stage=stage)
    except (
        GeneralizationIntegrityError,
        LLMClientError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cannot complete generalization evaluation: {exc}")
        return 1

    print()
    print(format_generalization_report(report))
    print("\nEvaluation outputs")
    print(f"Results: {paths['results']}")
    print(f"Errors: {paths['errors']}")
    return 0


def run_final_blind_test() -> int:
    """Run the SHA-locked Step 8.9 Final Blind Test exactly once."""

    print("Amazon Review Analyzer — Final Blind Test")
    print(
        "Evaluation-only: results will not be used to change the model, "
        "taxonomy, Prompt, or post-processing."
    )
    if LLM_PROVIDER != "ollama" or OLLAMA_MODEL != "qwen3:8b":
        print(
            "Step 8.9 is locked to local Ollama qwen3:8b. "
            f"Current configuration: {LLM_PROVIDER} / {OLLAMA_MODEL}."
        )
        return 1
    if final_outputs_exist():
        print(
            "Final Blind outputs already exist. The one-time Final Test will "
            "not be run or overwritten."
        )
        return 1
    try:
        report = run_final_blind_evaluation(
            provider=LLM_PROVIDER,
            model=OLLAMA_MODEL,
        )
        paths = save_final_blind_outputs(report)
    except (
        FinalBlindIntegrityError,
        LLMClientError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cannot complete Final Blind Test: {exc}")
        return 1

    print()
    print(format_final_blind_report(report))
    print("\nFinal Blind outputs")
    print(f"Report: {paths['report']}")
    print(f"Errors: {paths['errors']}")
    print(f"Predictions: {paths['predictions']}")
    return 0


def run_generalization_error_analysis() -> int:
    """Analyze frozen Step 8.9 artifacts without inference or network access."""

    print("Step 8.10 Generalization Error Study")
    print("Offline diagnostics only; frozen Final Blind artifacts will not be changed.")
    try:
        report = run_generalization_error_study()
    except (GeneralizationStudyError, OSError, TypeError, ValueError) as exc:
        print(f"Cannot complete generalization error study: {exc}")
        return 1
    print()
    print(format_generalization_error_study(report))
    print("\nStudy outputs")
    for label, path in report.get("outputs", {}).items():
        print(f"{label}: {path}")
    return 0


def run_generalization_dev_v2_evaluation() -> int:
    """Run the immutable Step 8.11 qwen3:8b baseline exactly once."""

    print("Amazon Review Analyzer — Generalization Dev V2 Baseline")
    print("Dataset creation and untouched baseline measurement only.")
    if generalization_dev_v2_outputs_exist():
        print(
            "Generalization Dev V2 baseline outputs already exist. The "
            "one-time baseline will not be run or overwritten."
        )
        return 1
    if LLM_PROVIDER != "ollama" or OLLAMA_MODEL != "qwen3:8b":
        print(
            "Step 8.11 is locked to local Ollama qwen3:8b. "
            f"Current configuration: {LLM_PROVIDER} / {OLLAMA_MODEL}."
        )
        return 1
    try:
        # The runner verifies an already-created manifest before constructing
        # a provider client.  The CLI intentionally never creates that manifest.
        report = run_generalization_dev_v2_baseline(
            provider=LLM_PROVIDER,
            model=OLLAMA_MODEL,
        )
        paths = save_generalization_dev_v2_outputs(report)
    except (
        GeneralizationDevV2IntegrityError,
        LLMClientError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cannot complete Generalization Dev V2 baseline: {exc}")
        return 1

    print()
    print(format_generalization_dev_v2_report(report))
    print("\nGeneralization Dev V2 outputs")
    print(f"Baseline: {paths['baseline']}")
    print(f"Errors: {paths['errors']}")
    print(f"Predictions: {paths['predictions']}")
    return 0


def run_generalization_dev_v2_optimization(round_number: int) -> int:
    """Run one bounded Step 8.12 round without touching frozen evaluations."""

    print(
        "Amazon Review Analyzer — Generalization Dev V2 "
        f"Optimization Round {round_number}"
    )
    print("Step 8.12 optimized only against Generalization Dev V2.")
    print("Final Blind was not rerun.")
    if LLM_PROVIDER != "ollama" or OLLAMA_MODEL != "qwen3:8b":
        print(
            "Step 8.12 is locked to local Ollama qwen3:8b. "
            f"Current configuration: {LLM_PROVIDER} / {OLLAMA_MODEL}."
        )
        return 1
    if GENERALIZATION_DEV_V2_BEST_PATH.exists():
        print("Step 8.12 already has a finalized best-round report.")
        return 1
    if optimization_round_outputs_exist(round_number):
        print(
            f"Optimization round {round_number} outputs already exist and will "
            "not be run or overwritten."
        )
        return 1

    try:
        prior_reports = load_optimization_round_reports(round_number - 1)
        baseline, _cases, _hashes = verify_frozen_step_811_inputs()
        report = run_generalization_dev_v2_optimization_round(
            round_number,
            provider=LLM_PROVIDER,
            model=OLLAMA_MODEL,
            prior_round_reports=prior_reports,
        )
        paths = save_optimization_round_outputs(report)
        completed_reports = [*prior_reports, report]
        stop = report["stop_evaluation"]
        best_path: str | None = None
        if stop["should_stop"]:
            save_best_round(completed_reports, baseline=baseline)
            best_path = str(GENERALIZATION_DEV_V2_BEST_PATH)
    except (
        GeneralizationDevV2IntegrityError,
        LLMClientError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cannot complete Generalization Dev V2 optimization: {exc}")
        return 1

    print()
    print(format_optimization_round_report(report))
    print("\nOptimization outputs")
    print(f"Report: {paths['report']}")
    print(f"Errors: {paths['errors']}")
    print(f"Predictions: {paths['predictions']}")
    if best_path is not None:
        print(f"Best round: {best_path}")
    else:
        print("Stop condition not reached; another bounded round may be evaluated.")
    return 0


def run_generalization_dev_v2_regression_analysis() -> int:
    """Create the required frozen Round 1 diff without an LLM call."""

    print("Amazon Review Analyzer — Step 8.12.1 Regression Isolation")
    print("Offline comparison only; no model will be called.")
    if (
        GENERALIZATION_V2_REGRESSION_ANALYSIS_PATH.exists()
        or GENERALIZATION_V2_REGRESSION_CASES_PATH.exists()
    ):
        print("Regression analysis outputs already exist and will not be overwritten.")
        print(f"Analysis: {GENERALIZATION_V2_REGRESSION_ANALYSIS_PATH}")
        print(f"Cases: {GENERALIZATION_V2_REGRESSION_CASES_PATH}")
        return 0
    try:
        paths = generate_and_save_round1_regression_analysis()
    except (GeneralizationDevV2IntegrityError, OSError, TypeError, ValueError) as exc:
        print(f"Cannot complete Step 8.12.1 regression isolation: {exc}")
        return 1
    print("Regression isolation completed.")
    print(f"Analysis: {paths['analysis']}")
    print(f"Cases: {paths['cases']}")
    return 0


def run_generalization_dev_v2_repair_evaluation() -> int:
    """Run and save the single permitted Step 8.12.1 qwen3:8b repair."""

    print("Amazon Review Analyzer — Step 8.12.1 Repair Candidate")
    print("Exactly one Generalization Dev V2 repair evaluation is permitted.")
    print("Final Blind and Holdout will not be run.")
    if repair_outputs_exist():
        print(
            "A Step 8.12.1 repair evaluation was already started. "
            "A second run is forbidden."
        )
        return 1

    def progress(position: int, total: int, case_id: str, status: str, elapsed: float) -> None:
        print(f"[{position}/{total}] Repair {case_id} {status} | {elapsed:.2f}s")

    try:
        client = create_review_client(provider="ollama")
        report, runtime = run_repair_candidate_evaluation(
            client=client,
            progress=progress,
        )
        paths = save_repair_candidate_outputs(report, runtime)
    except (
        GeneralizationDevV2IntegrityError,
        LLMClientError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cannot complete Step 8.12.1 repair evaluation: {exc}")
        return 1

    print()
    print(format_repair_candidate_report(report))
    print("\nRepair outputs")
    print(f"Report: {paths['report']}")
    print(f"Errors: {paths['errors']}")
    print(f"Predictions: {paths['predictions']}")
    print(f"Runtime: {paths['runtime']}")
    return 0


def run_blind_v2_test() -> int:
    """Run the single permitted Step 8.13 Blind V2 evaluation."""

    print("Amazon Review Analyzer — Step 8.13 Blind Test V2")
    print("Evaluation-only; the frozen candidate will not be modified.")
    if blind_v2_outputs_exist():
        print(
            "Blind V2 was already started. Its one-time evaluation cannot be "
            "run or overwritten."
        )
        return 1
    if LLM_PROVIDER != "ollama" or OLLAMA_MODEL != "qwen3:8b":
        print(
            "Blind V2 is locked to local Ollama qwen3:8b. "
            f"Current configuration: {LLM_PROVIDER} / {OLLAMA_MODEL}."
        )
        return 1

    def progress(
        position: int,
        total: int,
        case_id: str,
        status: str,
        elapsed: float,
    ) -> None:
        print(f"[{position}/{total}] {case_id} {status} | {elapsed:.2f}s")

    try:
        report = run_blind_v2_evaluation(
            provider=LLM_PROVIDER,
            model=OLLAMA_MODEL,
            progress=progress,
        )
        paths = save_blind_v2_outputs(report)
    except (
        BlindV2IntegrityError,
        LLMClientError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cannot complete Blind V2: {exc}")
        return 1

    print()
    print(format_blind_v2_report(report))
    print("\nBlind V2 frozen outputs")
    print(f"Report: {paths['report']}")
    print(f"Errors: {paths['errors']}")
    print(f"Predictions: {paths['predictions']}")
    print(f"Freeze manifest: {paths['freeze_manifest']}")
    return 0


def run_batch_analysis(
    limit: int | None,
    resume: bool = False,
    retry_failed: bool = False,
    llm_client: ReviewAnalysisClient | None = None,
) -> int:
    """Load, clean, and run one checkpoint-protected serial batch."""

    print("Amazon Review Analyzer - Batch Analysis")
    print(f"LLM provider: {LLM_PROVIDER}")
    print(
        "Analyzable review limit: "
        f"{limit if limit is not None else 'all remaining'}"
    )

    try:
        reviews = load_reviews(RAW_REVIEWS_PATH)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Cannot continue: {exc}")
        return 1

    validation = validate_reviews(reviews)
    if not validation["is_valid"]:
        print("Cannot continue because of fatal validation errors:")
        for error in validation["errors"]:
            print(f"- {error}")
        return 1

    try:
        cleaned_reviews = clean_reviews(reviews)
        result = run_checkpointed_batch(
            cleaned_reviews,
            limit=limit,
            resume=resume,
            retry_failed=retry_failed,
            llm_client=llm_client,
            checkpoint_path=ANALYSIS_CHECKPOINT_PATH,
            meta_path=ANALYSIS_CHECKPOINT_META_PATH,
            final_output_path=ANALYZED_REVIEWS_PATH,
            checkpoint_interval=BATCH_CHECKPOINT_INTERVAL,
            max_consecutive_failures=MAX_CONSECUTIVE_LLM_FAILURES,
        )
    except (CheckpointError, OSError, TypeError, ValueError) as exc:
        print(f"Cannot complete batch analysis: {exc}")
        return 1

    status_counts = result.reviews["analysis_status"].value_counts()
    attempted_mask = result.reviews["analysis_status"].isin(
        ["success", "failed"]
    )
    attempted_times = pd.to_numeric(
        result.reviews.loc[attempted_mask, "analysis_time_seconds"],
        errors="coerce",
    ).dropna()
    average_runtime = float(attempted_times.mean()) if not attempted_times.empty else 0.0
    print("\nBatch Analysis Summary")
    print(f"Total input rows: {result.total_input_rows}")
    print(f"Previously completed: {result.previously_completed}")
    print(f"Newly processed: {result.newly_processed}")
    print(f"Successful: {int(status_counts.get('success', 0))}")
    print(f"Failed: {int(status_counts.get('failed', 0))}")
    print(f"Skipped: {int(status_counts.get('skipped', 0))}")
    print(f"Average analysis time: {average_runtime:.2f}s")
    print(f"Checkpoint: {result.checkpoint_path}")
    if result.final_output_path is None:
        print("Final output: not written because the batch is not complete.")
    else:
        print(f"Final output: {result.final_output_path}")

    if result.interrupted:
        print("Batch status: interrupted; checkpoint is safe to resume.")
        return 130
    if result.stopped_for_failures:
        print("Batch status: stopped after repeated LLM availability failures.")
        return 1
    if result.is_complete:
        print("Batch status: complete.")
    else:
        print("Batch status: partial; run with --resume to continue.")
    return 0


def run_large_dataset_performance_benchmark(review_count: int = 20) -> int:
    """Run the isolated one-time workers=1/workers=2 production benchmark."""

    print("Amazon Review Analyzer - Performance Benchmark")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Independent performance reviews: {review_count}")
    print("Accuracy is not evaluated in this benchmark.")
    try:
        report = run_performance_comparison(review_count=review_count)
    except (
        PerformanceBenchmarkError,
        LLMClientError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cannot complete performance benchmark: {exc}")
        return 1
    print()
    print(format_performance_report(report))
    print("\nPerformance outputs")
    if review_count == 20:
        baseline_path = PERFORMANCE_BASELINE_PATH
        timings_path = PERFORMANCE_TIMINGS_PATH
    else:
        baseline_path = PERFORMANCE_BASELINE_PATH.with_name(
            f"performance_baseline_{review_count}.json"
        )
        timings_path = PERFORMANCE_TIMINGS_PATH.with_name(
            f"performance_request_timings_{review_count}.csv"
        )
    print(f"Baseline: {baseline_path}")
    print(f"Request timings: {timings_path}")
    return 0


def _print_rankings(
    title: str,
    records: list[dict[str, Any]],
    *,
    sentiment: str | None = None,
) -> None:
    print(f"\n{title}")
    if not records:
        print("(none)")
        return
    for position, record in enumerate(records, start=1):
        print(
            f"{position}. {record['aspect']} "
            f"[{record['business_category']}]"
        )
        review_count = int(record["aspect_review_count"])
        if sentiment is None:
            share = float(record["aspect_share_of_reviews"])
            print(f"   {review_count} reviews ({share:.1f}%)")
            continue
        positive = int(record["positive_review_count"])
        negative = int(record["negative_review_count"])
        neutral = int(record["neutral_review_count"])
        mixed = int(record["mixed_review_count"])
        print(
            f"   Counts: {positive} positive, {negative} negative, "
            f"{neutral} neutral, {mixed} mixed / "
            f"{review_count} reviews"
        )
        print(
            "   Rates: "
            f"{float(record['positive_rate']):.1f}% positive, "
            f"{float(record['negative_rate']):.1f}% negative, "
            f"{float(record['neutral_rate']):.1f}% neutral, "
            f"{float(record['mixed_rate']):.1f}% mixed"
        )


def run_aggregation(
    *,
    input_path: str | Path = ANALYZED_REVIEWS_PATH,
    min_aspect_reviews: int = MIN_ASPECT_REVIEWS_FOR_RANKING,
    top_n: int = 5,
    aspect_summary_path: str | Path = ASPECT_SUMMARY_PATH,
    evidence_path: str | Path = ASPECT_EVIDENCE_PATH,
    business_summary_path: str | Path = BUSINESS_SUMMARY_PATH,
) -> int:
    """Aggregate the completed formal batch output without calling an LLM."""

    print("Amazon Review Aggregation")
    try:
        reviews = load_analyzed_reviews(input_path)
        result = aggregate_reviews(
            reviews,
            min_aspect_reviews=min_aspect_reviews,
            top_n=top_n,
        )
    except (AggregationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"Cannot aggregate reviews: {exc}")
        print("Complete the batch before running --aggregate.")
        return 1

    data_quality = result.business_summary["data_quality"]
    print(f"\nTotal rows: {data_quality['total_input_rows']}")
    print(f"Successful reviews: {data_quality['successful_reviews']}")
    print(f"Failed: {data_quality['failed_reviews']}")
    print(f"Skipped: {data_quality['skipped_reviews']}")
    print(f"Analysis coverage: {data_quality['analysis_coverage']:.1%}")
    print(f"Invalid JSON rows: {data_quality['invalid_json_rows']}")

    if data_quality["successful_reviews"] == 0:
        print("No successfully analyzed reviews are available for aggregation.")
        return 1

    print(
        "Minimum reviews for directional rankings: "
        f"{min_aspect_reviews}"
    )
    _print_rankings(
        "Top Customer Concerns",
        result.business_summary["top_customer_concerns"],
    )
    _print_rankings(
        "Product Selling Points",
        result.business_summary["product_selling_points"],
        sentiment="positive",
    )
    _print_rankings(
        "Product Pain Points",
        result.business_summary["product_pain_points"],
        sentiment="negative",
    )
    print("\nShopping Experience")
    _print_rankings(
        "Positive",
        result.business_summary["shopping_experience"]["positive"],
        sentiment="positive",
    )
    _print_rankings(
        "Negative",
        result.business_summary["shopping_experience"]["negative"],
        sentiment="negative",
    )

    try:
        paths = save_aggregation_outputs(
            result,
            aspect_summary_path=aspect_summary_path,
            evidence_path=evidence_path,
            business_summary_path=business_summary_path,
        )
    except OSError as exc:
        print(f"Cannot save aggregation outputs: {exc}")
        return 1

    print("\nAggregation outputs")
    print(f"Aspect summary: {paths['aspect_summary']}")
    print(f"Aspect evidence: {paths['aspect_evidence']}")
    print(f"Business summary: {paths['business_summary']}")
    return 0


def _print_keyword_records(
    title: str,
    records: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> None:
    print(f"\n{title}")
    if not records:
        print("(none)")
        return
    for position, record in enumerate(records[:limit], start=1):
        print(
            f"{position}. {record['term']} | "
            f"{record['review_count']} reviews | "
            f"{record['count']} occurrences"
        )


def run_keyword_analysis(
    *,
    input_path: str | Path = ANALYZED_REVIEWS_PATH,
    min_keyword_reviews: int = KEYWORD_MIN_REVIEW_COUNT,
    keyword_summary_path: str | Path = KEYWORD_SUMMARY_PATH,
    sentiment_keywords_path: str | Path = SENTIMENT_KEYWORDS_PATH,
    aspect_keywords_path: str | Path = ASPECT_KEYWORDS_PATH,
) -> int:
    """Build Step 7 statistics from the existing formal batch output."""

    print("Keyword Analysis")
    try:
        reviews = load_keyword_reviews(input_path)
        result = analyze_keywords(
            reviews,
            min_review_count=min_keyword_reviews,
        )
    except (
        KeywordAnalysisError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cannot analyze keywords: {exc}")
        return 1

    print(f"\nSuccessful reviews: {result.successful_reviews}")
    print(f"Minimum review count: {min_keyword_reviews}")
    print(f"Evidence issues: {len(result.issues)}")
    if result.successful_reviews == 0:
        print("No successfully analyzed reviews are available for keyword analysis.")
        return 1

    records = result.keyword_summary.to_dict(orient="records")
    _print_keyword_records(
        "Top Keywords",
        [record for record in records if record["term_type"] == "unigram"],
    )
    _print_keyword_records(
        "Top Phrases",
        [record for record in records if record["term_type"] in {"bigram", "trigram"}],
    )
    _print_keyword_records(
        "Positive Phrases",
        [
            record
            for record in result.sentiment_keywords["positive"]
            if record["term_type"] in {"bigram", "trigram"}
        ],
    )
    _print_keyword_records(
        "Negative Phrases",
        [
            record
            for record in result.sentiment_keywords["negative"]
            if record["term_type"] in {"bigram", "trigram"}
        ],
    )

    try:
        paths = save_keyword_outputs(
            result,
            keyword_summary_path=keyword_summary_path,
            sentiment_keywords_path=sentiment_keywords_path,
            aspect_keywords_path=aspect_keywords_path,
        )
    except (KeywordAnalysisError, OSError) as exc:
        print(f"Cannot save keyword outputs: {exc}")
        return 1

    print("\nKeyword outputs")
    print(f"Keyword summary: {paths['keyword_summary']}")
    print(f"Sentiment keywords: {paths['sentiment_keywords']}")
    print(f"Aspect keywords: {paths['aspect_keywords']}")
    return 0


def main(llm_client: ReviewAnalysisClient | None = None) -> int:
    """运行数据准备，并只分析 R003 这一条评论。"""

    is_ready, missing = check_project_setup()

    print("Amazon Review Analyzer")

    if not is_ready:
        print(f"Setup check failed. Missing directories: {', '.join(missing)}")
        return 1

    print("\nLoading reviews...")
    try:
        reviews = load_reviews(RAW_REVIEWS_PATH)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Cannot continue: {exc}")
        return 1

    print(f"Loaded {len(reviews)} reviews.")
    validation = validate_reviews(reviews)
    _print_validation_summary(validation)

    if not validation["is_valid"]:
        print("\nCannot continue because of fatal validation errors:")
        for error in validation["errors"]:
            print(f"- {error}")
        return 1

    cleaned_reviews = clean_reviews(reviews)
    cleaning_summary = summarize_cleaning(cleaned_reviews)
    _print_cleaning_summary(cleaning_summary)

    selected_reviews = cleaned_reviews.loc[cleaned_reviews["review_id"].eq("R003")]
    if selected_reviews.empty:
        print("\nCannot analyze R003: review was not found.")
        return 1

    review = selected_reviews.iloc[0]
    if not bool(review["is_analyzable"]):
        print(f"\nCannot analyze R003: {review['skip_reason']}.")
        return 1

    review_text = str(review["cleaned_content"])
    print("\nSingle Review Analysis: R003")
    print(f"LLM provider: {LLM_PROVIDER}")
    print(f'Review: "{review_text}"')

    try:
        result = analyze_review(review_text, llm_client=llm_client)
    except (LLMClientError, ReviewAnalysisError) as exc:
        print(f"Cannot analyze R003: {exc}")
        return 1

    print()
    _print_review_analysis(result)

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Amazon product reviews.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--release-health-check",
        action="store_true",
        help="run offline v1.0 release integrity and configuration checks",
    )
    modes.add_argument(
        "--health-check",
        action="store_true",
        help="run lightweight environment readiness diagnostics",
    )
    modes.add_argument(
        "--health-check-json",
        action="store_true",
        help="print lightweight environment readiness diagnostics as JSON",
    )
    modes.add_argument(
        "--check-ollama",
        "--check-llm",
        dest="check_ollama",
        action="store_true",
        help="check the local Ollama service and configured model only",
    )
    modes.add_argument(
        "--check-amazon-access",
        action="store_true",
        help="diagnose public Amazon HTTPS access without saving any data",
    )
    modes.add_argument(
        "--manual-test",
        action="store_true",
        help="analyze the ten manually labelled review cases",
    )
    modes.add_argument(
        "--benchmark-models",
        action="store_true",
        help="serially benchmark the configured Ollama models on the ten manual cases",
    )
    modes.add_argument(
        "--benchmark-performance",
        action="store_true",
        help="benchmark the frozen qwen3:8b runtime with 1 and 2 workers",
    )
    modes.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the configured provider/model against the gold dataset",
    )
    modes.add_argument(
        "--evaluate-models",
        action="store_true",
        help="compare qwen3:8b and qwen3:4b-instruct serially",
    )
    modes.add_argument(
        "--check-regression",
        action="store_true",
        help="compare the latest evaluation with the qwen3:8b baseline",
    )
    modes.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="evaluate frozen qwen3:8b on the SHA-locked blind holdout set",
    )
    modes.add_argument(
        "--evaluate-generalization-dev",
        action="store_true",
        help="evaluate qwen3:8b on the Step 8.8 development expansion set",
    )
    modes.add_argument(
        "--evaluate-final-blind",
        action="store_true",
        help="run the one-time SHA-locked Step 8.9 Final Blind Test",
    )
    modes.add_argument(
        "--analyze-generalization-errors",
        action="store_true",
        help="run the offline Step 8.10 study over frozen evaluation artifacts",
    )
    modes.add_argument(
        "--evaluate-generalization-dev-v2",
        action="store_true",
        help="run the one-time untouched Step 8.11 qwen3:8b baseline",
    )
    modes.add_argument(
        "--optimize-generalization-dev-v2-round",
        type=int,
        choices=(1, 2, 3),
        metavar="ROUND",
        help="run one bounded Step 8.12 qwen3:8b optimization round",
    )
    modes.add_argument(
        "--analyze-generalization-dev-v2-regressions",
        action="store_true",
        help="compare frozen Step 8.11 and Step 8.12 Round 1 without an LLM",
    )
    modes.add_argument(
        "--evaluate-generalization-dev-v2-repair",
        action="store_true",
        help="run the single permitted Step 8.12.1 qwen3:8b repair evaluation",
    )
    modes.add_argument(
        "--evaluate-blind-v2",
        action="store_true",
        help="run the one-time SHA-locked Step 8.13 Blind Test V2",
    )
    modes.add_argument(
        "--batch",
        action="store_true",
        help="run a serial CSV batch analysis",
    )
    modes.add_argument(
        "--aggregate",
        action="store_true",
        help="aggregate the completed analyzed_reviews.csv without an LLM",
    )
    modes.add_argument(
        "--keywords",
        action="store_true",
        help="extract offline keyword and phrase statistics",
    )
    parser.add_argument(
        "--amazon-url",
        default=None,
        help="Amazon product URL used only with --check-amazon-access",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="maximum new LLM attempts for this --batch invocation",
    )
    parser.add_argument(
        "--reviews",
        type=int,
        choices=(10, 20),
        default=None,
        help="independent review count for --benchmark-performance (default: 20)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue rows missing from a valid checkpoint",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry only rows marked failed in a valid checkpoint",
    )
    parser.add_argument(
        "--min-aspect-reviews",
        type=int,
        help="minimum review sample for selling and pain point rankings",
    )
    parser.add_argument(
        "--min-keyword-reviews",
        type=int,
        help="minimum distinct successful reviews for keyword output",
    )
    parser.add_argument(
        "--generalization-stage",
        choices=("baseline", "optimized"),
        default="optimized",
        help="output stage used only with --evaluate-generalization-dev",
    )
    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.amazon_url is not None and not args.check_amazon_access:
        print("--amazon-url can only be used with --check-amazon-access.")
        return 2
    if (
        not args.evaluate_generalization_dev
        and args.generalization_stage != "optimized"
    ):
        print(
            "--generalization-stage can only be changed with "
            "--evaluate-generalization-dev."
        )
        return 2
    if not args.batch and args.limit is not None:
        print("--limit can only be used with --batch.")
        return 2
    if args.reviews is not None and not args.benchmark_performance:
        print("--reviews can only be used with --benchmark-performance.")
        return 2
    if not args.batch and (args.resume or args.retry_failed):
        print("--resume and --retry-failed can only be used with --batch.")
        return 2
    if not args.aggregate and args.min_aspect_reviews is not None:
        print("--min-aspect-reviews can only be used with --aggregate.")
        return 2
    if args.min_aspect_reviews is not None and args.min_aspect_reviews <= 0:
        print("--min-aspect-reviews must be a positive integer.")
        return 2
    if not args.keywords and args.min_keyword_reviews is not None:
        print("--min-keyword-reviews can only be used with --keywords.")
        return 2
    if args.min_keyword_reviews is not None and args.min_keyword_reviews <= 0:
        print("--min-keyword-reviews must be a positive integer.")
        return 2
    if args.release_health_check:
        return run_release_health_check_command()
    if args.health_check:
        return run_health_cli(json_output=False)
    if args.health_check_json:
        return run_health_cli(json_output=True)
    if args.check_ollama:
        return run_ollama_check()
    if args.check_amazon_access:
        return run_amazon_access_check(
            args.amazon_url or DEFAULT_AMAZON_DIAGNOSTIC_URL
        )
    if args.manual_test:
        return run_manual_tests()
    if args.benchmark_models:
        return run_model_benchmarks()
    if args.benchmark_performance:
        return run_large_dataset_performance_benchmark(args.reviews or 20)
    if args.evaluate:
        return run_quality_evaluation()
    if args.evaluate_models:
        return run_quality_model_comparison()
    if args.check_regression:
        return run_evaluation_regression_check()
    if args.evaluate_holdout:
        return run_blind_holdout_evaluation()
    if args.evaluate_generalization_dev:
        return run_generalization_dev_evaluation(args.generalization_stage)
    if args.evaluate_final_blind:
        return run_final_blind_test()
    if args.analyze_generalization_errors:
        return run_generalization_error_analysis()
    if args.evaluate_generalization_dev_v2:
        return run_generalization_dev_v2_evaluation()
    if args.optimize_generalization_dev_v2_round is not None:
        return run_generalization_dev_v2_optimization(
            args.optimize_generalization_dev_v2_round
        )
    if args.analyze_generalization_dev_v2_regressions:
        return run_generalization_dev_v2_regression_analysis()
    if args.evaluate_generalization_dev_v2_repair:
        return run_generalization_dev_v2_repair_evaluation()
    if args.evaluate_blind_v2:
        return run_blind_v2_test()
    if args.aggregate:
        return run_aggregation(
            min_aspect_reviews=(
                args.min_aspect_reviews
                if args.min_aspect_reviews is not None
                else MIN_ASPECT_REVIEWS_FOR_RANKING
            )
        )
    if args.keywords:
        return run_keyword_analysis(
            min_keyword_reviews=(
                args.min_keyword_reviews
                if args.min_keyword_reviews is not None
                else KEYWORD_MIN_REVIEW_COUNT
            )
        )
    if args.batch:
        if args.limit is not None and args.limit <= 0:
            print("--limit must be a positive integer.")
            return 2
        if args.limit is None and not (args.resume or args.retry_failed):
            print(
                "A fresh --batch requires --limit with a positive integer. "
                "Resume and retry modes may omit --limit."
            )
            return 2
        return run_batch_analysis(
            limit=args.limit,
            resume=args.resume,
            retry_failed=args.retry_failed,
        )
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
