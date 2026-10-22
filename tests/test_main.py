"""Offline tests for the real-model diagnostic command paths."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module  # noqa: E402
from main import (  # noqa: E402
    main,
    run_aggregation,
    run_keyword_analysis,
    run_manual_tests,
    run_ollama_check,
)


class _FakeConnectionClient:
    def __init__(self) -> None:
        self.calls = 0

    def check_connection(self) -> dict[str, str]:
        self.calls += 1
        return {"host": "http://localhost:11434", "model": "qwen3:8b"}


class _FakeAnalysisClient:
    def __init__(self, r003_response: bool = False) -> None:
        self.reviews: list[str] = []
        self.r003_response = r003_response

    def analyze_review(self, review_text: str, allowed_aspects: Any) -> dict[str, Any]:
        self.reviews.append(review_text)
        if self.r003_response:
            return {
                "summary": "Excellent picture quality, with a cheap remote and slow menu.",
                "aspects": [
                    {
                        "raw_aspect": "picture quality",
                        "aspect": "Picture Quality",
                        "sentiment": "positive",
                        "evidence": "The picture quality is excellent",
                    },
                    {
                        "raw_aspect": "remote",
                        "aspect": "Remote Control",
                        "sentiment": "negative",
                        "evidence": "the remote feels cheap",
                    },
                    {
                        "raw_aspect": "menu",
                        "aspect": "System Performance",
                        "sentiment": "negative",
                        "evidence": "the menu is slow",
                    },
                ],
                "positive_points": ["Excellent picture quality"],
                "negative_points": ["Cheap remote", "Slow menu"],
            }
        return {
            "summary": "Manual review result.",
            "aspects": [],
            "positive_points": [],
            "negative_points": [],
        }


class MainCommandTests(unittest.TestCase):
    def test_amazon_access_command_routes_to_read_only_diagnostic(self) -> None:
        with patch.object(
            main_module,
            "run_amazon_access_check",
            return_value=0,
        ) as diagnostic:
            exit_code = main_module.cli(["--check-amazon-access"])

        self.assertEqual(0, exit_code)
        diagnostic.assert_called_once_with(
            main_module.DEFAULT_AMAZON_DIAGNOSTIC_URL
        )

    def test_amazon_access_command_accepts_an_explicit_product_url(self) -> None:
        product_url = "https://www.amazon.de/dp/B012345678"
        with patch.object(
            main_module,
            "run_amazon_access_check",
            return_value=1,
        ) as diagnostic:
            exit_code = main_module.cli(
                ["--check-amazon-access", "--amazon-url", product_url]
            )

        self.assertEqual(1, exit_code)
        diagnostic.assert_called_once_with(product_url)

    def test_amazon_url_option_is_rejected_outside_diagnostic_mode(self) -> None:
        with patch.object(main_module, "run_amazon_access_check") as diagnostic:
            exit_code = main_module.cli(
                ["--amazon-url", "https://www.amazon.com/dp/B012345678"]
            )

        self.assertEqual(2, exit_code)
        diagnostic.assert_not_called()

    def test_connection_command_only_checks_service_and_model(self) -> None:
        fake = _FakeConnectionClient()
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = run_ollama_check(fake)  # type: ignore[arg-type]

        self.assertEqual(0, exit_code)
        self.assertEqual(1, fake.calls)
        self.assertIn("Ollama running.", output.getvalue())
        self.assertIn("Model found: qwen3:8b", output.getvalue())

    def test_main_analyzes_only_r003_and_prints_required_labels(self) -> None:
        fake = _FakeAnalysisClient(r003_response=True)
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(llm_client=fake)

        self.assertEqual(0, exit_code)
        self.assertEqual(
            [
                "The picture quality is excellent, but the remote feels cheap and the menu is slow."
            ],
            fake.reviews,
        )
        rendered = output.getvalue()
        self.assertIn("Summary:", rendered)
        self.assertIn("Raw Aspect:", rendered)
        self.assertIn("Normalized Aspect:", rendered)
        self.assertIn("Sentiment:", rendered)
        self.assertIn("Evidence:", rendered)

    def test_manual_command_processes_exactly_ten_cases(self) -> None:
        fake = _FakeAnalysisClient()
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = run_manual_tests(llm_client=fake)

        self.assertEqual(0, exit_code)
        self.assertEqual(10, len(fake.reviews))
        rendered = output.getvalue()
        self.assertIn("Case ID: M001", rendered)
        self.assertIn("Case ID: M010", rendered)
        self.assertIn("Expected:", rendered)
        self.assertIn("Actual:", rendered)
        self.assertIn("Manual cases processed: 10", rendered)

    def test_benchmark_command_routes_to_independent_runner(self) -> None:
        with patch.object(
            main_module, "run_model_benchmarks", return_value=0
        ) as benchmark:
            exit_code = main_module.cli(["--benchmark-models"])

        self.assertEqual(0, exit_code)
        benchmark.assert_called_once_with()

    def test_batch_command_passes_limit_to_batch_runner(self) -> None:
        with patch.object(
            main_module, "run_batch_analysis", return_value=0
        ) as batch:
            exit_code = main_module.cli(["--batch", "--limit", "5"])

        self.assertEqual(0, exit_code)
        batch.assert_called_once_with(
            limit=5,
            resume=False,
            retry_failed=False,
        )

    def test_batch_command_refuses_to_run_without_limit(self) -> None:
        with patch.object(main_module, "run_batch_analysis") as batch:
            exit_code = main_module.cli(["--batch"])

        self.assertEqual(2, exit_code)
        batch.assert_not_called()

    def test_limit_is_rejected_without_batch_mode(self) -> None:
        with patch.object(main_module, "run_batch_analysis") as batch:
            exit_code = main_module.cli(["--limit", "5"])

        self.assertEqual(2, exit_code)
        batch.assert_not_called()

    def test_resume_without_limit_routes_to_batch_runner(self) -> None:
        with patch.object(
            main_module, "run_batch_analysis", return_value=0
        ) as batch:
            exit_code = main_module.cli(["--batch", "--resume"])

        self.assertEqual(0, exit_code)
        batch.assert_called_once_with(
            limit=None,
            resume=True,
            retry_failed=False,
        )

    def test_retry_failed_routes_to_batch_runner(self) -> None:
        with patch.object(
            main_module, "run_batch_analysis", return_value=0
        ) as batch:
            exit_code = main_module.cli(
                ["--batch", "--retry-failed", "--limit", "2"]
            )

        self.assertEqual(0, exit_code)
        batch.assert_called_once_with(
            limit=2,
            resume=False,
            retry_failed=True,
        )

    def test_resume_and_retry_can_be_combined(self) -> None:
        with patch.object(
            main_module, "run_batch_analysis", return_value=0
        ) as batch:
            exit_code = main_module.cli(
                ["--batch", "--resume", "--retry-failed", "--limit", "4"]
            )

        self.assertEqual(0, exit_code)
        batch.assert_called_once_with(
            limit=4,
            resume=True,
            retry_failed=True,
        )

    def test_resume_is_rejected_without_batch_mode(self) -> None:
        with patch.object(main_module, "run_batch_analysis") as batch:
            exit_code = main_module.cli(["--resume"])

        self.assertEqual(2, exit_code)
        batch.assert_not_called()

    def test_aggregate_command_routes_to_independent_runner(self) -> None:
        with patch.object(main_module, "run_aggregation", return_value=0) as aggregate:
            exit_code = main_module.cli(["--aggregate"])

        self.assertEqual(0, exit_code)
        aggregate.assert_called_once_with(
            min_aspect_reviews=main_module.MIN_ASPECT_REVIEWS_FOR_RANKING
        )

    def test_aggregate_command_accepts_visible_minimum_sample(self) -> None:
        with patch.object(main_module, "run_aggregation", return_value=0) as aggregate:
            exit_code = main_module.cli(
                ["--aggregate", "--min-aspect-reviews", "1"]
            )

        self.assertEqual(0, exit_code)
        aggregate.assert_called_once_with(min_aspect_reviews=1)

    def test_minimum_sample_is_rejected_without_aggregate(self) -> None:
        with patch.object(main_module, "run_aggregation") as aggregate:
            exit_code = main_module.cli(["--min-aspect-reviews", "1"])

        self.assertEqual(2, exit_code)
        aggregate.assert_not_called()

    def test_nonpositive_minimum_sample_is_rejected(self) -> None:
        with patch.object(main_module, "run_aggregation") as aggregate:
            exit_code = main_module.cli(
                ["--aggregate", "--min-aspect-reviews", "0"]
            )

        self.assertEqual(2, exit_code)
        aggregate.assert_not_called()

    def test_aggregation_missing_formal_file_is_controlled(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output):
            exit_code = run_aggregation(
                input_path=Path(directory) / "missing.csv",
                aspect_summary_path=Path(directory) / "aspect.csv",
                evidence_path=Path(directory) / "evidence.json",
                business_summary_path=Path(directory) / "business.json",
            )

        self.assertEqual(1, exit_code)
        self.assertIn("Cannot aggregate reviews:", output.getvalue())

    def test_aggregation_with_zero_success_prints_clear_message(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                [
                    {
                        "source_row_id": 0,
                        "review_id": "R001",
                        "analysis_status": "failed",
                        "aspects": "[]",
                        "unique_aspects": "[]",
                    }
                ]
            ).to_csv(root / "analyzed.csv", index=False)
            with redirect_stdout(output):
                exit_code = run_aggregation(
                    input_path=root / "analyzed.csv",
                    aspect_summary_path=root / "aspect.csv",
                    evidence_path=root / "evidence.json",
                    business_summary_path=root / "business.json",
                )
            self.assertFalse((root / "aspect.csv").exists())

        self.assertEqual(1, exit_code)
        self.assertIn(
            "No successfully analyzed reviews are available for aggregation.",
            output.getvalue(),
        )

    def test_aggregation_writes_outputs_without_calling_an_llm(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unique = [
                {
                    "aspect": "Picture Quality",
                    "sentiment": "positive",
                    "mention_count": 1,
                    "include_in_business_metrics": True,
                }
            ]
            aspects = [
                {
                    "raw_aspect": "picture",
                    "aspect": "Picture Quality",
                    "sentiment": "positive",
                    "evidence": "picture is excellent",
                }
            ]
            pd.DataFrame(
                [
                    {
                        "source_row_id": 0,
                        "review_id": "R001",
                        "analysis_status": "success",
                        "aspects": json.dumps(aspects),
                        "unique_aspects": json.dumps(unique),
                    }
                ]
            ).to_csv(root / "analyzed.csv", index=False)
            with (
                patch.object(main_module, "create_review_client") as create_client,
                redirect_stdout(output),
            ):
                exit_code = run_aggregation(
                    input_path=root / "analyzed.csv",
                    min_aspect_reviews=1,
                    aspect_summary_path=root / "aspect.csv",
                    evidence_path=root / "evidence.json",
                    business_summary_path=root / "business.json",
                )
            create_client.assert_not_called()
            self.assertTrue((root / "aspect.csv").is_file())
            self.assertTrue((root / "evidence.json").is_file())
            self.assertTrue((root / "business.json").is_file())

        self.assertEqual(0, exit_code)
        rendered = output.getvalue()
        self.assertIn("Top Customer Concerns", rendered)
        self.assertIn("Product Selling Points", rendered)
        self.assertIn("Product Pain Points", rendered)
        self.assertIn("Shopping Experience", rendered)
        self.assertNotIn("\nTop Selling Points", rendered)

    def test_keywords_command_routes_to_offline_runner(self) -> None:
        with patch.object(
            main_module, "run_keyword_analysis", return_value=0
        ) as keywords:
            exit_code = main_module.cli(["--keywords"])

        self.assertEqual(0, exit_code)
        keywords.assert_called_once_with(
            min_keyword_reviews=main_module.KEYWORD_MIN_REVIEW_COUNT
        )

    def test_keywords_command_accepts_visible_minimum(self) -> None:
        with patch.object(
            main_module, "run_keyword_analysis", return_value=0
        ) as keywords:
            exit_code = main_module.cli(
                ["--keywords", "--min-keyword-reviews", "1"]
            )

        self.assertEqual(0, exit_code)
        keywords.assert_called_once_with(min_keyword_reviews=1)

    def test_keyword_minimum_is_rejected_without_keywords_mode(self) -> None:
        with patch.object(main_module, "run_keyword_analysis") as keywords:
            exit_code = main_module.cli(["--min-keyword-reviews", "1"])

        self.assertEqual(2, exit_code)
        keywords.assert_not_called()

    def test_nonpositive_keyword_minimum_is_rejected(self) -> None:
        with patch.object(main_module, "run_keyword_analysis") as keywords:
            exit_code = main_module.cli(
                ["--keywords", "--min-keyword-reviews", "0"]
            )

        self.assertEqual(2, exit_code)
        keywords.assert_not_called()

    def test_keyword_runner_writes_three_outputs_without_an_llm(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aspects = [
                {
                    "raw_aspect": "picture quality",
                    "aspect": "Picture Quality",
                    "sentiment": "positive",
                    "evidence": "picture quality is excellent",
                }
            ]
            pd.DataFrame(
                [
                    {
                        "source_row_id": 0,
                        "review_id": "R001",
                        "analysis_status": "success",
                        "cleaned_content": "Picture quality is excellent.",
                        "aspects": json.dumps(aspects),
                    }
                ]
            ).to_csv(root / "analyzed.csv", index=False)
            with (
                patch.object(main_module, "create_review_client") as create_client,
                redirect_stdout(output),
            ):
                exit_code = run_keyword_analysis(
                    input_path=root / "analyzed.csv",
                    min_keyword_reviews=1,
                    keyword_summary_path=root / "keywords.csv",
                    sentiment_keywords_path=root / "sentiment.json",
                    aspect_keywords_path=root / "aspects.json",
                )

            create_client.assert_not_called()
            self.assertTrue((root / "keywords.csv").is_file())
            self.assertTrue((root / "sentiment.json").is_file())
            self.assertTrue((root / "aspects.json").is_file())

        self.assertEqual(0, exit_code)
        rendered = output.getvalue()
        self.assertIn("Successful reviews: 1", rendered)
        self.assertIn("Top Keywords", rendered)
        self.assertIn("Top Phrases", rendered)
        self.assertIn("Positive Phrases", rendered)
        self.assertIn("Negative Phrases", rendered)

    def test_keyword_runner_missing_input_is_controlled(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output):
            exit_code = run_keyword_analysis(
                input_path=Path(directory) / "missing.csv",
                min_keyword_reviews=1,
                keyword_summary_path=Path(directory) / "keywords.csv",
                sentiment_keywords_path=Path(directory) / "sentiment.json",
                aspect_keywords_path=Path(directory) / "aspects.json",
            )

        self.assertEqual(1, exit_code)
        self.assertIn("Cannot analyze keywords:", output.getvalue())

    def test_keyword_runner_zero_success_does_not_replace_outputs(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                [
                    {
                        "source_row_id": 0,
                        "review_id": "R001",
                        "analysis_status": "failed",
                        "cleaned_content": "Picture quality is excellent.",
                        "aspects": "[]",
                    }
                ]
            ).to_csv(root / "analyzed.csv", index=False)
            with redirect_stdout(output):
                exit_code = run_keyword_analysis(
                    input_path=root / "analyzed.csv",
                    min_keyword_reviews=1,
                    keyword_summary_path=root / "keywords.csv",
                    sentiment_keywords_path=root / "sentiment.json",
                    aspect_keywords_path=root / "aspects.json",
                )
            self.assertFalse((root / "keywords.csv").exists())

        self.assertEqual(1, exit_code)
        self.assertIn("No successfully analyzed reviews", output.getvalue())

    def test_health_check_cli_dispatches_human_output(self) -> None:
        with patch.object(main_module, "run_health_cli", return_value=1) as health:
            exit_code = main_module.cli(["--health-check"])

        self.assertEqual(1, exit_code)
        health.assert_called_once_with(json_output=False)

    def test_health_check_cli_dispatches_json_output(self) -> None:
        with patch.object(main_module, "run_health_cli", return_value=0) as health:
            exit_code = main_module.cli(["--health-check-json"])

        self.assertEqual(0, exit_code)
        health.assert_called_once_with(json_output=True)


if __name__ == "__main__":
    unittest.main()
