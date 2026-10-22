"""Offline unit tests for the Step 8 evaluation and QA system.

Every model-facing test injects a fake client.  This module must never contact
Ollama, OpenAI, Amazon, or write to the frozen Demo outputs.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluator  # noqa: E402
import main as main_module  # noqa: E402
from config import DEMO_OUTPUT_PATHS  # noqa: E402


def _aspect(
    name: str,
    sentiment: str,
    evidence: str,
    *,
    raw_aspect: str | None = None,
) -> dict[str, str]:
    return {
        "raw_aspect": raw_aspect or name,
        "aspect": name,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _case(
    case_id: str,
    review: str,
    expected: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "review": review,
        "expected_aspects": expected,
    }


def _analysis(aspects: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "summary": "Fake offline result.",
        "aspects": aspects,
        "positive_points": [],
        "negative_points": [],
    }


def _summary(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": "qwen3:8b",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "total_cases": 2,
        "successful_cases": 2,
        "failed_cases": 0,
        "aspect_precision": 0.8,
        "aspect_recall": 0.75,
        "aspect_f1": 0.774,
        "aspect_only_precision": 0.9,
        "aspect_only_recall": 0.9,
        "aspect_only_f1": 0.9,
        "sentiment_accuracy": 0.8,
        "evidence_validity_rate": 1.0,
        "evidence_match_rate": 0.75,
        "exact_case_accuracy": 0.5,
        "exact_matches": 1,
        "cases_with_errors": 1,
        "average_analysis_time": 2.0,
        "median_analysis_time": 2.0,
        "total_analysis_time": 4.0,
        "confusion_matrix": {},
    }
    result.update(overrides)
    return result


class _FakeClient:
    def __init__(
        self,
        model: str,
        responses: dict[str, dict[str, Any] | Exception],
        events: list[tuple[str, str]] | None = None,
    ) -> None:
        self.model = model
        self.responses = responses
        self.events = events if events is not None else []

    def analyze_review(self, review_text: str, allowed_aspects: Any) -> dict[str, Any]:
        self.events.append((self.model, review_text))
        response = self.responses[review_text]
        if isinstance(response, Exception):
            raise response
        return response


class _TickClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class EvaluatorMetricTests(unittest.TestCase):
    def test_exact_and_aspect_only_metrics_distinguish_sentiment_errors(self) -> None:
        review = "The picture is excellent, the sound is harsh, and the design is sleek."
        gold = _case(
            "E001",
            review,
            [
                _aspect("Picture Quality", "positive", "picture is excellent"),
                _aspect("Sound Quality", "negative", "sound is harsh"),
            ],
        )
        actual = _analysis(
            [
                _aspect("Picture Quality", "positive", "picture is excellent"),
                _aspect("Sound Quality", "positive", "sound is harsh"),
                _aspect("Design", "positive", "design is sleek"),
            ]
        )

        case_result = evaluator.evaluate_case(gold, actual)
        metrics = evaluator.calculate_metrics([case_result], model="fake-model")

        self.assertAlmostEqual(1 / 3, metrics["aspect_precision"])
        self.assertAlmostEqual(1 / 2, metrics["aspect_recall"])
        self.assertAlmostEqual(0.4, metrics["aspect_f1"])
        self.assertAlmostEqual(2 / 3, metrics["aspect_only_precision"])
        self.assertAlmostEqual(1.0, metrics["aspect_only_recall"])
        self.assertAlmostEqual(0.8, metrics["aspect_only_f1"])
        self.assertAlmostEqual(0.5, metrics["sentiment_accuracy"])
        self.assertEqual(["Design"], case_result["extra_aspects"])
        self.assertEqual("Sound Quality", case_result["wrong_sentiments"][0]["aspect"])

    def test_empty_expected_and_prediction_is_an_exact_case(self) -> None:
        result = evaluator.evaluate_case(
            _case("E002", "Great!", []),
            _analysis([]),
        )
        metrics = evaluator.calculate_metrics([result], model="fake-model")

        self.assertTrue(result["exact_match"])
        self.assertEqual([], result["errors"])
        self.assertEqual(1.0, metrics["exact_case_accuracy"])
        self.assertEqual(0.0, metrics["aspect_precision"])
        self.assertEqual(0.0, metrics["aspect_recall"])

    def test_empty_prediction_records_every_missing_aspect(self) -> None:
        result = evaluator.evaluate_case(
            _case(
                "E003",
                "Picture is clear and sound is rich.",
                [
                    _aspect("Picture Quality", "positive", "Picture is clear"),
                    _aspect("Sound Quality", "positive", "sound is rich"),
                ],
            ),
            _analysis([]),
        )

        self.assertFalse(result["exact_match"])
        self.assertCountEqual(
            ["Picture Quality", "Sound Quality"],
            result["missing_aspects"],
        )
        self.assertEqual(2, result["counts"]["exact_fn"])
        self.assertEqual(
            2,
            sum(error["error_type"] == "MISS_ASPECT" for error in result["errors"]),
        )

    def test_duplicate_expected_pairs_are_counted_not_collapsed(self) -> None:
        expected = _aspect("System Performance", "negative", "menu is slow")
        result = evaluator.evaluate_case(
            _case("E004", "The menu is slow and apps are slow.", [expected, expected.copy()]),
            _analysis([expected.copy()]),
        )

        self.assertEqual(1, result["counts"]["exact_tp"])
        self.assertEqual(1, result["counts"]["exact_fn"])
        self.assertFalse(result["exact_match"])

    def test_mixed_gold_accepts_positive_and_negative_projection(self) -> None:
        result = evaluator.evaluate_case(
            _case(
                "E005",
                "The picture is vivid but sometimes washed out.",
                [_aspect("Picture Quality", "mixed", "vivid but sometimes washed out")],
            ),
            _analysis(
                [
                    _aspect("Picture Quality", "positive", "picture is vivid"),
                    _aspect("Picture Quality", "negative", "sometimes washed out"),
                ]
            ),
        )

        self.assertTrue(result["exact_match"])
        self.assertEqual("mixed", result["evaluation_aspects"][0]["sentiment"])

    def test_confusion_matrix_records_actual_and_predicted_sentiment(self) -> None:
        result = evaluator.evaluate_case(
            _case(
                "E006",
                "The sound is not good.",
                [_aspect("Sound Quality", "negative", "sound is not good")],
            ),
            _analysis([_aspect("Sound Quality", "positive", "sound is not good")]),
        )
        metrics = evaluator.calculate_metrics([result], model="fake-model")

        self.assertEqual(1, result["confusion_matrix"]["negative"]["positive"])
        self.assertEqual(1, metrics["confusion_matrix"]["negative"]["positive"])
        self.assertEqual(0.0, metrics["sentiment_accuracy"])

    def test_exact_case_accuracy_and_runtime_statistics(self) -> None:
        exact = evaluator.evaluate_case(
            _case("E007", "Great!", []),
            _analysis([]),
            analysis_time_seconds=1.0,
        )
        inexact = evaluator.evaluate_case(
            _case(
                "E008",
                "Delivery was late.",
                [_aspect("Delivery", "negative", "Delivery was late")],
            ),
            _analysis([]),
            analysis_time_seconds=3.0,
        )
        third = evaluator.evaluate_case(
            _case("E009", "Exactly what I expected.", []),
            _analysis([]),
            analysis_time_seconds=2.0,
        )

        metrics = evaluator.calculate_metrics([exact, inexact, third], model="fake-model")

        self.assertAlmostEqual(2 / 3, metrics["exact_case_accuracy"])
        self.assertEqual(2.0, metrics["average_analysis_time"])
        self.assertEqual(2.0, metrics["median_analysis_time"])
        self.assertEqual(6.0, metrics["total_analysis_time"])


class EvidenceAndErrorTests(unittest.TestCase):
    def test_evidence_validity_requires_a_contiguous_review_substring(self) -> None:
        review = "The picture is excellent, but the remote feels cheap."
        result = evaluator.evaluate_case(
            _case(
                "E010",
                review,
                [_aspect("Picture Quality", "positive", "picture is excellent")],
            ),
            _analysis(
                [
                    _aspect("Picture Quality", "positive", "picture is excellent"),
                    _aspect("Remote Control", "negative", "invented remote claim"),
                ]
            ),
        )

        self.assertEqual(0.5, result["evidence_validity_rate"])
        self.assertTrue(result["evidence_evaluations"][0]["valid"])
        self.assertFalse(result["evidence_evaluations"][1]["valid"])
        self.assertIn(
            "INVALID_EVIDENCE",
            {error["error_type"] for error in result["errors"]},
        )

    def test_evidence_match_normalizes_case_punctuation_and_containment(self) -> None:
        result = evaluator.evaluate_case(
            _case(
                "E011",
                "The picture quality is excellent.",
                [_aspect("Picture Quality", "positive", "picture quality is excellent")],
            ),
            _analysis(
                [
                    _aspect(
                        "Picture Quality",
                        "positive",
                        "The picture quality is excellent.",
                    )
                ]
            ),
        )

        self.assertEqual(1.0, result["evidence_validity_rate"])
        self.assertEqual(1.0, result["evidence_match_rate"])
        self.assertTrue(result["evidence_matches"][0]["matched"])

    def test_weak_evidence_match_is_reported_separately(self) -> None:
        result = evaluator.evaluate_case(
            _case(
                "E012",
                "The battery lasts all week.",
                [_aspect("Reliability", "positive", "battery lasts all week")],
            ),
            _analysis([_aspect("Reliability", "positive", "battery")]),
        )

        self.assertEqual(0.0, result["evidence_match_rate"])
        self.assertIn(
            "WEAK_EVIDENCE_MATCH",
            {error["error_type"] for error in result["errors"]},
        )

    def test_strong_shared_evidence_can_identify_wrong_taxonomy_classification(self) -> None:
        result = evaluator.evaluate_case(
            _case(
                "E013",
                "The TV works perfectly.",
                [_aspect("Reliability", "positive", "TV works perfectly")],
            ),
            _analysis(
                [_aspect("System Performance", "positive", "TV works perfectly")]
            ),
        )

        self.assertEqual("Reliability", result["wrong_aspects"][0]["expected"])
        self.assertEqual("System Performance", result["wrong_aspects"][0]["actual"])
        self.assertIn(
            "WRONG_ASPECT",
            {error["error_type"] for error in result["errors"]},
        )

    def test_model_and_schema_errors_have_structured_taxonomy(self) -> None:
        model_error = evaluator.evaluate_case(
            _case("E014", "Review", []),
            _analysis([]),
            model_error="offline fake failure",
        )
        schema_error = evaluator.evaluate_case(
            _case("E015", "Picture is good.", []),
            {"summary": "bad", "aspects": [{"aspect": "Picture Quality"}]},
        )

        self.assertIn(
            "MODEL_ERROR",
            {error["error_type"] for error in model_error["errors"]},
        )
        self.assertIn(
            "SCHEMA_ERROR",
            {error["error_type"] for error in schema_error["errors"]},
        )


class SerializationAndRegressionTests(unittest.TestCase):
    def test_report_json_and_error_csv_are_exported_to_requested_paths(self) -> None:
        case_result = evaluator.evaluate_case(
            _case(
                "E016",
                "Delivery was late.",
                [_aspect("Delivery", "negative", "Delivery was late")],
            ),
            _analysis([]),
        )
        report = {
            "summary": evaluator.calculate_metrics([case_result], model="fake-model"),
            "cases": [case_result],
            "error_counts": {"MISS_ASPECT": 1},
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "evaluation_results.json"
            errors_path = root / "evaluation_errors.csv"
            paths = evaluator.save_evaluation_report(
                report,
                results_path=results_path,
                errors_path=errors_path,
            )

            loaded = json.loads(results_path.read_text(encoding="utf-8"))
            with errors_path.open(encoding="utf-8", newline="") as handle:
                errors = list(csv.DictReader(handle))

        self.assertEqual("fake-model", loaded["summary"]["model"])
        self.assertEqual("E016", loaded["cases"][0]["case_id"])
        self.assertEqual("MISS_ASPECT", errors[0]["error_type"])
        self.assertEqual("E016", errors[0]["case_id"])
        self.assertEqual(results_path, Path(paths["results"]))
        self.assertEqual(errors_path, Path(paths["errors"]))

    def test_regression_threshold_is_strictly_more_than_five_points(self) -> None:
        baseline_report = {"summary": _summary(aspect_f1=0.80)}

        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            evaluator.save_baseline(baseline_report, path=baseline_path)
            at_threshold = evaluator.check_regression(
                {"summary": _summary(aspect_f1=0.75)},
                baseline_path=baseline_path,
            )
            below_threshold = evaluator.check_regression(
                {"summary": _summary(aspect_f1=0.749)},
                baseline_path=baseline_path,
            )

        self.assertFalse(at_threshold["has_regression"])
        self.assertTrue(below_threshold["has_regression"])
        self.assertTrue(
            any("Aspect F1" in warning for warning in below_threshold["warnings"])
        )

    def test_evidence_validity_below_ninety_five_percent_warns(self) -> None:
        baseline_report = {"summary": _summary(evidence_validity_rate=1.0)}

        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            evaluator.save_baseline(baseline_report, path=baseline_path)
            result = evaluator.check_regression(
                {"summary": _summary(evidence_validity_rate=0.949)},
                baseline_path=baseline_path,
            )

        self.assertTrue(result["has_regression"])
        self.assertTrue(
            any("Evidence Validity" in warning for warning in result["warnings"])
        )

    def test_model_comparison_format_contains_metrics_without_a_winner(self) -> None:
        reports = [
            {"summary": _summary(model="qwen3:8b")},
            {
                "summary": _summary(
                    model="qwen3:4b-instruct",
                    aspect_f1=0.7,
                    sentiment_accuracy=0.72,
                    exact_case_accuracy=0.4,
                    average_analysis_time=1.5,
                )
            },
        ]

        rendered = evaluator.format_model_comparison(reports)

        self.assertIn("Model Comparison", rendered)
        self.assertLess(rendered.index("qwen3:8b"), rendered.index("qwen3:4b-instruct"))
        self.assertIn("77.4%", rendered)
        self.assertIn("1.50s", rendered)
        self.assertNotIn("winner", rendered.casefold())
        self.assertNotIn("best model", rendered.casefold())

    def test_evaluation_report_includes_transparent_error_examples(self) -> None:
        case_result = evaluator.evaluate_case(
            _case(
                "E016B",
                "Delivery was late.",
                [_aspect("Delivery", "negative", "Delivery was late")],
            ),
            _analysis([]),
        )
        report = {
            "summary": evaluator.calculate_metrics(
                [case_result], model="fake-model"
            ),
            "cases": [case_result],
            "error_counts": {"MISS_ASPECT": 1},
        }

        rendered = evaluator.format_evaluation_report(report)

        self.assertIn("Error Examples:", rendered)
        self.assertIn("Case E016B", rendered)
        self.assertIn("Expected: Delivery → negative", rendered)
        self.assertIn("Actual: (no specific aspect)", rendered)
        self.assertIn("Error: MISS_ASPECT", rendered)


class OfflineExecutionTests(unittest.TestCase):
    def test_model_comparison_runs_models_and_cases_strictly_serially(self) -> None:
        cases = [
            _case("E017", "First review.", []),
            _case("E018", "Second review.", []),
        ]
        responses = {
            "First review.": _analysis([]),
            "Second review.": _analysis([]),
        }
        events: list[tuple[str, str]] = []

        reports = evaluator.evaluate_models(
            models=("qwen3:8b", "qwen3:4b-instruct"),
            cases=cases,
            client_factory=lambda model: _FakeClient(model, responses, events),
            clock=_TickClock(),
        )

        self.assertEqual(
            [
                ("qwen3:8b", "First review."),
                ("qwen3:8b", "Second review."),
                ("qwen3:4b-instruct", "First review."),
                ("qwen3:4b-instruct", "Second review."),
            ],
            events,
        )
        self.assertEqual(
            ["qwen3:8b", "qwen3:4b-instruct"],
            [report["summary"]["model"] for report in reports],
        )

    def test_fake_evaluation_does_not_modify_demo_files(self) -> None:
        before = {
            path: path.read_bytes()
            for path in DEMO_OUTPUT_PATHS
            if path.exists()
        }
        case = _case("E019", "Great!", [])
        fake = _FakeClient("fake-model", {"Great!": _analysis([])})

        report = evaluator.evaluate_model(
            cases=[case],
            model="fake-model",
            client=fake,
            clock=_TickClock(),
        )

        after = {path: path.read_bytes() for path in before}
        self.assertEqual(before, after)
        self.assertEqual(1, report["summary"]["total_cases"])


class EvaluationCliTests(unittest.TestCase):
    def test_evaluate_routes_to_configured_evaluation_runner(self) -> None:
        with patch.object(main_module, "run_quality_evaluation", return_value=0) as run:
            exit_code = main_module.cli(["--evaluate"])

        self.assertEqual(0, exit_code)
        run.assert_called_once_with()

    def test_evaluate_models_routes_to_serial_comparison_runner(self) -> None:
        with patch.object(
            main_module,
            "run_quality_model_comparison",
            return_value=0,
        ) as run:
            exit_code = main_module.cli(["--evaluate-models"])

        self.assertEqual(0, exit_code)
        run.assert_called_once_with()

    def test_check_regression_routes_to_nonblocking_runner(self) -> None:
        with patch.object(
            main_module,
            "run_evaluation_regression_check",
            return_value=0,
        ) as run:
            exit_code = main_module.cli(["--check-regression"])

        self.assertEqual(0, exit_code)
        run.assert_called_once_with()

    def test_evaluation_modes_are_mutually_exclusive(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main_module.cli(["--evaluate", "--evaluate-models"])

        self.assertEqual(2, raised.exception.code)

    def test_evaluate_uses_default_configured_provider_and_model(self) -> None:
        report = {"summary": _summary(), "cases": [], "error_counts": {}}
        output = io.StringIO()

        with (
            patch.object(main_module, "run_evaluation", return_value=report) as run,
            patch.object(main_module, "format_evaluation_report", return_value="report"),
            redirect_stdout(output),
        ):
            exit_code = main_module.run_quality_evaluation()

        self.assertEqual(0, exit_code)
        run.assert_called_once_with()
        self.assertIn("report", output.getvalue())

    def test_comparison_uses_only_the_two_required_models_in_order(self) -> None:
        output = io.StringIO()
        reports = [{"summary": _summary()}]

        with (
            patch.object(main_module, "evaluate_models", return_value=reports) as run,
            patch.object(main_module, "format_model_comparison", return_value="comparison"),
            redirect_stdout(output),
        ):
            exit_code = main_module.run_quality_model_comparison()

        self.assertEqual(0, exit_code)
        run.assert_called_once_with(
            models=("qwen3:8b", "qwen3:4b-instruct")
        )
        self.assertIn("comparison", output.getvalue())

    def test_regression_warning_is_reported_but_returns_success(self) -> None:
        output = io.StringIO()
        current = {"summary": _summary()}

        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "results.json"
            baseline_path = Path(directory) / "baseline.json"
            results_path.write_text(json.dumps(current), encoding="utf-8")
            with (
                patch.object(main_module, "EVALUATION_RESULTS_PATH", results_path),
                patch.object(main_module, "EVALUATION_BASELINE_PATH", baseline_path),
                patch.object(
                    main_module,
                    "check_regression",
                    return_value={
                        "has_regression": True,
                        "warnings": ["Aspect F1 decreased by 6.0 percentage points."],
                        "deltas": {},
                        "baseline_path": str(baseline_path),
                    },
                ) as check,
                redirect_stdout(output),
            ):
                exit_code = main_module.run_evaluation_regression_check()

        self.assertEqual(0, exit_code)
        self.assertIn("REGRESSION WARNING", output.getvalue())
        check.assert_called_once_with(current, baseline_path=baseline_path)


if __name__ == "__main__":
    unittest.main()
