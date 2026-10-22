"""Offline tests for the Step 8.12 bounded optimization evaluator."""

from __future__ import annotations

import json
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import generalization_dev_v2_optimizer as opt
import main as main_module
from evaluator import calculate_metrics, evaluate_case
from final_blind_evaluator import strict_exact_case
from holdout_evaluator import file_sha256
from llm_client import LLMConnectionError, LLMResponseError


def _mention(aspect: str, sentiment: str, evidence: str) -> dict[str, str]:
    return {"aspect": aspect, "sentiment": sentiment, "evidence": evidence}


def _case(
    index: int,
    review: str,
    expected: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "case_id": f"V2D{index:03d}",
        "review": review,
        "expected_aspects": expected,
        "tags": ["offline_fixture"],
        "notes": f"Synthetic offline case {index}.",
    }


def _analysis(aspects: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "summary": "Mock analysis.",
        "aspects": aspects or [],
        "positive_points": [],
        "negative_points": [],
    }


def _strict_result(case: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_case(case, analysis)
    result["base_exact_match"] = result["exact_match"]
    result["exact_match"] = strict_exact_case(result)
    return result


def _priority_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return opt.calculate_aspect_metrics(results, opt.PRIORITY_ASPECTS)


def _report(
    round_number: int,
    *,
    f1: float,
    aspect_only_f1: float | None = None,
    fp: int = 10,
    fn: int = 10,
    exact: float = 0.5,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "round_number": round_number,
        "stage": f"optimization_round{round_number}",
        "provider": "ollama",
        "model": "qwen3:8b",
        "dataset_sha256": "a" * 64,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "case_count": 1,
        "summary": {
            "total_cases": 1,
            "successful_cases": 1,
            "failed_cases": 0,
            "aspect_precision": f1,
            "aspect_recall": f1,
            "aspect_f1": f1,
            "aspect_only_precision": aspect_only_f1 if aspect_only_f1 is not None else f1,
            "aspect_only_recall": aspect_only_f1 if aspect_only_f1 is not None else f1,
            "aspect_only_f1": aspect_only_f1 if aspect_only_f1 is not None else f1,
            "sentiment_accuracy": 1.0,
            "evidence_validity_rate": 1.0,
            "evidence_match_rate": 1.0,
            "exact_matches": int(exact == 1.0),
            "exact_case_accuracy": exact,
            "aspect_true_positives": 1,
            "aspect_false_positives": fp,
            "aspect_false_negatives": fn,
        },
        "error_counts": {error_type: 0 for error_type in opt.ERROR_TYPES},
        "priority_aspect_metrics": {
            aspect: {"precision": 0.5, "recall": 0.5, "f1": 0.5}
            for aspect in opt.PRIORITY_ASPECTS
        },
        "regression_warnings": warnings or [],
        "runtime_statistics": {"total_seconds": 1.0},
        "cases": [],
        "predictions": [],
    }


def _fixture(root: Path) -> dict[str, Any]:
    cases = [
        _case(
            1,
            "The screen is crisp.",
            [_mention("Picture Quality", "positive", "screen is crisp")],
        ),
        _case(2, "It arrived yesterday.", []),
        _case(
            3,
            "Wi-Fi keeps dropping.",
            [_mention("Connectivity", "negative", "Wi-Fi keeps dropping")],
        ),
    ]
    dataset = root / "generalization_dev_v2_cases.json"
    dataset.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")

    perfect_analyses = [
        _analysis([_mention("Picture Quality", "positive", "screen is crisp")]),
        _analysis(),
        _analysis([_mention("Connectivity", "negative", "Wi-Fi keeps dropping")]),
    ]
    baseline_results = [
        _strict_result(case, analysis)
        for case, analysis in zip(cases, perfect_analyses, strict=True)
    ]
    baseline_summary = calculate_metrics(baseline_results, model="qwen3:8b")
    baseline = {
        "stage": "baseline",
        "provider": "ollama",
        "model": "qwen3:8b",
        "dataset_sha256": file_sha256(dataset),
        "case_count": len(cases),
        "summary": baseline_summary,
        "error_counts": {error_type: 0 for error_type in opt.ERROR_TYPES},
        "priority_aspect_metrics": _priority_metrics(baseline_results),
        "cases": baseline_results,
    }
    manifest = root / "generalization_dev_v2_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sha256": file_sha256(dataset),
                "case_count": len(cases),
                # A deliberately old production fingerprint: optimization
                # verification must not compare it with current core files.
                "core_model_file_sha256": {"old.py": "0" * 64},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_path = root / "baseline.json"
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    errors_path = root / "baseline_errors.csv"
    errors_path.write_text("case_id,error_type\n", encoding="utf-8")
    predictions_path = root / "baseline_predictions.json"
    predictions_path.write_text('{"predictions": []}\n', encoding="utf-8")
    logic = root / "current_logic.py"
    logic.write_text("CURRENT = True\n", encoding="utf-8")
    paths = {
        "dataset": dataset,
        "manifest": manifest,
        "baseline": baseline_path,
        "baseline_errors": errors_path,
        "baseline_predictions": predictions_path,
    }
    hashes = {label: file_sha256(path) for label, path in paths.items()}
    return {
        "cases": cases,
        "analyses": perfect_analyses,
        "baseline": baseline,
        "paths": paths,
        "hashes": hashes,
        "logic": logic,
    }


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


def _run(
    fixture: dict[str, Any], analyzer: Any, *, round_number: int = 1
) -> dict[str, Any]:
    return opt.run_generalization_dev_v2_optimization_round(
        round_number,
        frozen_paths=fixture["paths"],
        frozen_sha256=fixture["hashes"],
        expected_count=3,
        provider="ollama",
        model="qwen3:8b",
        core_model_paths=(fixture["logic"],),
        client=object(),
        analyzer=analyzer,
        clock=_Clock(),
        sleeper=lambda _seconds: None,
        progress=lambda *_args: None,
    )


class FrozenInputTests(unittest.TestCase):
    def test_real_dataset_hash_constant_is_step_811_value(self) -> None:
        self.assertEqual(
            "ea6d264db2dea92fa6e0e2e469e2de572cac56654e916fcc3502bfaf23f650f2",
            opt.EXPECTED_V2_DATASET_SHA256,
        )

    def test_frozen_inputs_validate_without_requiring_current_core_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            baseline, cases, hashes = opt.verify_frozen_step_811_inputs(
                paths=fixture["paths"],
                expected_sha256=fixture["hashes"],
                expected_count=3,
            )
        self.assertEqual(3, len(cases))
        self.assertEqual("baseline", baseline["stage"])
        self.assertEqual(fixture["hashes"], hashes)

    def test_mutation_fails_before_provider_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            fixture["paths"]["manifest"].write_text("{}\n", encoding="utf-8")
            factory = Mock()
            with self.assertRaisesRegex(
                opt.GeneralizationDevV2IntegrityError, "manifest SHA-256 changed"
            ):
                opt.run_generalization_dev_v2_optimization_round(
                    1,
                    frozen_paths=fixture["paths"],
                    frozen_sha256=fixture["hashes"],
                    expected_count=3,
                    provider="ollama",
                    model="qwen3:8b",
                    core_model_paths=(fixture["logic"],),
                    client_factory=factory,
                )
            factory.assert_not_called()

    def test_runner_detects_core_mutation_during_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))

            def mutate(_review: str, **_kwargs: Any) -> dict[str, Any]:
                fixture["logic"].write_text("CHANGED = True\n", encoding="utf-8")
                return _analysis()

            with self.assertRaisesRegex(
                opt.GeneralizationDevV2IntegrityError, "hashes changed"
            ):
                _run(fixture, mutate)


class RoundRunnerTests(unittest.TestCase):
    def test_runner_is_serial_and_reuses_existing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            analyzer = Mock(side_effect=fixture["analyses"])
            report = _run(fixture, analyzer)
        self.assertEqual(3, analyzer.call_count)
        self.assertEqual(1.0, report["summary"]["aspect_f1"])
        self.assertEqual(3, report["summary"]["exact_matches"])
        self.assertEqual(fixture["hashes"]["dataset"], report["dataset_sha256"])
        self.assertIn("baseline_deltas", report)
        self.assertIn("stop_evaluation", report)
        self.assertIn("Connectivity", report["stable_aspect_metrics"])

    def test_transient_failure_retries_once_then_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            analyzer = Mock(
                side_effect=[
                    LLMConnectionError("temporary"),
                    fixture["analyses"][0],
                    fixture["analyses"][1],
                    fixture["analyses"][2],
                ]
            )
            report = _run(fixture, analyzer)
        self.assertEqual(4, analyzer.call_count)
        self.assertEqual(1, report["runtime_statistics"]["retry_count"])
        self.assertEqual(0, report["summary"]["failed_cases"])

    def test_second_transient_failure_is_failed_and_round_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            analyzer = Mock(
                side_effect=[
                    LLMConnectionError("temporary"),
                    LLMConnectionError("still unavailable"),
                    fixture["analyses"][1],
                    fixture["analyses"][2],
                ]
            )
            report = _run(fixture, analyzer)
        self.assertEqual(4, analyzer.call_count)
        self.assertEqual(1, report["summary"]["failed_cases"])
        self.assertEqual(1, report["error_counts"]["MODEL_ERROR"])

    def test_quality_error_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            analyzer = Mock(
                side_effect=[
                    LLMResponseError("bad schema"),
                    fixture["analyses"][1],
                    fixture["analyses"][2],
                ]
            )
            report = _run(fixture, analyzer)
        self.assertEqual(3, analyzer.call_count)
        self.assertEqual(0, report["runtime_statistics"]["retry_count"])
        self.assertEqual(1, report["error_counts"]["SCHEMA_ERROR"])

    def test_injected_runner_does_not_access_network_or_default_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            with patch.object(
                socket, "create_connection", side_effect=AssertionError("network")
            ), patch.object(
                opt, "create_review_client", side_effect=AssertionError("real client")
            ):
                report = _run(fixture, Mock(side_effect=fixture["analyses"]))
        self.assertEqual(3, report["summary"]["successful_cases"])


class MetricsAndPolicyTests(unittest.TestCase):
    def test_aspect_metrics_preserve_same_aspect_multiplicity(self) -> None:
        case = _case(
            1,
            "The speakers are clear, but the bass is weak.",
            [
                _mention("Sound Quality", "positive", "speakers are clear"),
                _mention("Sound Quality", "negative", "bass is weak"),
            ],
        )
        result = _strict_result(
            case,
            _analysis([_mention("Sound Quality", "positive", "speakers are clear")]),
        )
        sound = opt.calculate_aspect_metrics([result], ["Sound Quality"])[
            "Sound Quality"
        ]
        self.assertEqual((2, 1, 0, 1), (sound["support"], sound["tp"], sound["fp"], sound["fn"]))

    def test_stable_regression_warning_requires_more_than_five_points(self) -> None:
        baseline = {aspect: {"f1": 0.9} for aspect in opt.STABLE_ASPECTS}
        exactly_five = {aspect: {"f1": 0.85} for aspect in opt.STABLE_ASPECTS}
        self.assertEqual([], opt.stable_aspect_regressions(exactly_five, baseline))
        current = dict(exactly_five)
        current["Delivery"] = {"f1": 0.849}
        warnings = opt.stable_aspect_regressions(current, baseline)
        self.assertEqual("Delivery", warnings[0]["aspect"])
        self.assertEqual("REGRESSION_WARNING", warnings[0]["warning"])

    def test_stop_condition_a(self) -> None:
        baseline = _report(1, f1=0.74)
        round_one = _report(1, f1=0.85)
        result = opt.evaluate_stop_conditions(baseline, [round_one])
        self.assertTrue(result["should_stop"])
        self.assertIn("A", [item["condition"] for item in result["conditions"]])

    def test_stop_condition_b_uses_two_consecutive_round_gains(self) -> None:
        baseline = _report(1, f1=0.74)
        round_one = _report(1, f1=0.745)
        round_two = _report(2, f1=0.754)
        result = opt.evaluate_stop_conditions(baseline, [round_one, round_two])
        self.assertIn("B", [item["condition"] for item in result["conditions"]])

    def test_stop_conditions_c_and_d(self) -> None:
        baseline = _report(1, f1=0.74)
        warning = [{"warning": "REGRESSION_WARNING", "aspect": "Delivery"}]
        reports = [
            _report(1, f1=0.76),
            _report(2, f1=0.78),
            _report(3, f1=0.80, warnings=warning),
        ]
        conditions = opt.evaluate_stop_conditions(baseline, reports)["conditions"]
        self.assertEqual({"C", "D"}, {item["condition"] for item in conditions})

    def test_best_round_selection_uses_required_metric_order(self) -> None:
        lower_f1 = _report(1, f1=0.80, aspect_only_f1=0.99)
        higher_f1 = _report(2, f1=0.81, aspect_only_f1=0.80)
        self.assertEqual(2, opt.select_best_round([lower_f1, higher_f1])["round_number"])
        same_f1_better_aspect_only = _report(3, f1=0.81, aspect_only_f1=0.82)
        self.assertEqual(
            3,
            opt.select_best_round(
                [lower_f1, higher_f1, same_f1_better_aspect_only]
            )[
                "round_number"
            ],
        )


class PersistenceTests(unittest.TestCase):
    def test_baseline_diagnostic_counts_confusions_and_never_overwrites(self) -> None:
        baseline = {
            "case_count": 1,
            "dataset_sha256": "a" * 64,
            "error_counts": {"WRONG_ASPECT": 1, "EXTRA_ASPECT": 1},
            "cases": [
                {
                    "expected_aspects": [],
                    "actual_aspects": [
                        _mention("Design", "neutral", "A factual label")
                    ],
                    "errors": [
                        {
                            "error_type": "EXTRA_ASPECT",
                            "actual_aspect": "Design",
                        },
                        {
                            "error_type": "WRONG_ASPECT",
                            "expected_aspect": "Other",
                            "actual_aspect": "Design",
                        },
                    ],
                }
            ],
        }
        diagnostic = opt.build_baseline_optimization_diagnostic(baseline)
        self.assertEqual(1, diagnostic["extra_aspects_by_label"]["Design"])
        self.assertEqual(1, diagnostic["wrong_aspect_directions"]["Other -> Design"])
        self.assertEqual(1, diagnostic["mention_granularity"]["extra_mentions_in_no_aspect_cases"])
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "diagnostic.json"
            opt.save_baseline_optimization_diagnostic(
                diagnostic, destination=destination
            )
            with self.assertRaisesRegex(
                opt.GeneralizationDevV2IntegrityError, "will not be overwritten"
            ):
                opt.save_baseline_optimization_diagnostic(
                    diagnostic, destination=destination
                )

    def test_round_paths_are_independent_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            one = opt.optimization_round_paths(1, output_dir=temporary)
            two = opt.optimization_round_paths(2, output_dir=temporary)
            self.assertFalse(
                opt.optimization_round_outputs_exist(1, output_dir=temporary)
            )
            one["report"].touch()
            self.assertTrue(
                opt.optimization_round_outputs_exist(1, output_dir=temporary)
            )
        self.assertTrue(set(one.values()).isdisjoint(two.values()))
        self.assertTrue(str(one["report"]).endswith("optimization_round1_generalization_dev_v2.json"))
        with self.assertRaises(ValueError):
            opt.optimization_round_paths(4)

    def test_round_outputs_include_report_errors_predictions_and_no_overwrite(self) -> None:
        report = _report(1, f1=0.8)
        with tempfile.TemporaryDirectory() as temporary:
            saved = opt.save_optimization_round_outputs(report, output_dir=temporary)
            self.assertEqual({"report", "errors", "predictions"}, set(saved))
            self.assertTrue(all(Path(path).is_file() for path in saved.values()))
            with self.assertRaisesRegex(
                opt.GeneralizationDevV2IntegrityError, "will not be overwritten"
            ):
                opt.save_optimization_round_outputs(report, output_dir=temporary)

    def test_saved_round_reports_load_only_as_consecutive_qwen_v2_results(self) -> None:
        one = _report(1, f1=0.8)
        two = _report(2, f1=0.82)
        one["dataset_sha256"] = opt.EXPECTED_V2_DATASET_SHA256
        two["dataset_sha256"] = opt.EXPECTED_V2_DATASET_SHA256
        with tempfile.TemporaryDirectory() as temporary:
            opt.save_optimization_round_outputs(one, output_dir=temporary)
            opt.save_optimization_round_outputs(two, output_dir=temporary)
            loaded = opt.load_optimization_round_reports(
                2, output_dir=temporary
            )
        self.assertEqual([1, 2], [report["round_number"] for report in loaded])

    def test_best_round_file_is_final_and_not_overwritten(self) -> None:
        baseline = _report(1, f1=0.74)
        rounds = [_report(1, f1=0.80), _report(2, f1=0.82)]
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "best.json"
            payload = opt.save_best_round(
                rounds, baseline=baseline, destination=destination
            )
            self.assertEqual(2, payload["selected_round"])
            self.assertTrue(destination.is_file())
            with self.assertRaisesRegex(
                opt.GeneralizationDevV2IntegrityError, "will not be overwritten"
            ):
                opt.save_best_round(rounds, baseline=baseline, destination=destination)

    def test_module_has_no_amazon_openai_or_final_blind_execution(self) -> None:
        source = Path(opt.__file__).read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\s*(?:from|import)\s+amazon")
        self.assertNotIn("run_final_blind", source)
        self.assertNotRegex(source, r"(?m)^\s*(?:from|import)\s+openai")


class OptimizationCliTests(unittest.TestCase):
    def test_cli_dispatches_requested_round(self) -> None:
        with patch.object(
            main_module,
            "run_generalization_dev_v2_optimization",
            return_value=9,
        ) as runner:
            result = main_module.cli(
                ["--optimize-generalization-dev-v2-round", "2"]
            )
        self.assertEqual(9, result)
        runner.assert_called_once_with(2)

    def test_existing_round_outputs_block_real_inference(self) -> None:
        output = StringIO()
        with patch.object(
            main_module,
            "GENERALIZATION_DEV_V2_BEST_PATH",
            Mock(exists=Mock(return_value=False)),
        ), patch.object(
            main_module,
            "optimization_round_outputs_exist",
            return_value=True,
        ), patch.object(
            main_module, "run_generalization_dev_v2_optimization_round"
        ) as runner, redirect_stdout(output):
            result = main_module.run_generalization_dev_v2_optimization(1)
        self.assertEqual(1, result)
        runner.assert_not_called()
        self.assertIn("will not be run or overwritten", output.getvalue())


if __name__ == "__main__":
    unittest.main()
