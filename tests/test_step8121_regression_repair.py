"""Offline guards for Step 8.12.1 regression isolation and repair."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generalization_dev_v2_repair as repair
import main as main_module
from aspect_taxonomy import STANDARD_ASPECTS
from generalization_dev_v2_evaluator import GeneralizationDevV2IntegrityError
from llm_client import _analysis_instructions
from review_analyzer import analyze_review


class _StaticClient:
    def __init__(self, aspects: list[dict[str, str]]) -> None:
        self.response = {"summary": "Synthetic offline analysis.", "aspects": aspects}

    def analyze_review(self, review_text: str, allowed_aspects):
        return self.response


def _mention(
    raw_aspect: str,
    aspect: str,
    sentiment: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "raw_aspect": raw_aspect,
        "aspect": aspect,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _analyze(review: str, aspects: list[dict[str, str]]):
    return analyze_review(review, llm_client=_StaticClient(aspects))


class Step8121PromptAndBoundaryTests(unittest.TestCase):
    def test_repair_prompt_fits_below_baseline_and_context_proxy(self) -> None:
        profile = repair.current_prompt_profile()
        self.assertLess(
            profile["ollama_system_chars_with_schema"],
            repair.BASELINE_PROMPT_PROFILE["ollama_system_chars_with_schema"],
        )
        self.assertLessEqual(
            profile["ollama_system_approx_tokens_chars_div_4"],
            repair.EXPECTED_PROVIDER_CONFIGURATION["num_ctx"],
        )
        prompt = _analysis_instructions(STANDARD_ASPECTS)
        self.assertNotRegex(prompt, r"\bV2D\d{3}\b")

    def test_delivery_route_fact_is_not_an_opinion(self) -> None:
        review = "Tracking listed the parcel at three transfer depots."
        result = _analyze(
            review,
            [_mention("parcel route", "Delivery", "neutral", review)],
        )
        self.assertEqual([], result["aspects"])

    def test_delivery_timing_is_kept_but_packaging_stays_separate(self) -> None:
        review = "The courier arrived two days late, while the foam stayed intact."
        result = _analyze(
            review,
            [
                _mention(
                    "courier arrival",
                    "Delivery",
                    "negative",
                    "The courier arrived two days late",
                ),
                _mention(
                    "foam",
                    "Packaging",
                    "positive",
                    "the foam stayed intact",
                ),
            ],
        )
        self.assertEqual(
            ["Delivery", "Packaging"],
            [item["aspect"] for item in result["aspects"]],
        )

    def test_delivery_alert_complaint_is_kept(self) -> None:
        review = "No delivery alert was sent."
        result = _analyze(
            review,
            [_mention("delivery alert", "Delivery", "negative", review)],
        )
        self.assertEqual("Delivery", result["aspects"][0]["aspect"])

    def test_initial_setup_clarity_is_setup_not_ease(self) -> None:
        review = "The first-time account wizard never explained the privacy choice."
        result = _analyze(
            review,
            [
                _mention(
                    "first-time account wizard",
                    "Ease of Use",
                    "negative",
                    review,
                )
            ],
        )
        self.assertEqual("Setup / Installation", result["aspects"][0]["aspect"])

    def test_daily_navigation_is_ease_and_pairing_is_connectivity(self) -> None:
        navigation = "The daily settings menu is confusing."
        navigation_result = _analyze(
            navigation,
            [_mention("settings menu", "Setup / Installation", "negative", navigation)],
        )
        self.assertEqual("Ease of Use", navigation_result["aspects"][0]["aspect"])

        pairing = "Bluetooth pairing was painless."
        pairing_result = _analyze(
            pairing,
            [_mention("Bluetooth pairing", "Setup / Installation", "positive", pairing)],
        )
        self.assertEqual("Connectivity", pairing_result["aspects"][0]["aspect"])

    def test_not_hard_setup_is_positive(self) -> None:
        review = "Bolting the stand on was not hard."
        result = _analyze(
            review,
            [_mention("bolting the stand", "Setup / Installation", "neutral", review)],
        )
        self.assertEqual("positive", result["aspects"][0]["sentiment"])

    def test_neutral_duplicate_loses_to_exact_polar_judgment(self) -> None:
        review = "Mounting the base was easy."
        result = _analyze(
            review,
            [
                _mention("mounting", "Setup / Installation", "positive", review),
                _mention("mounting", "Setup / Installation", "neutral", review),
            ],
        )
        self.assertEqual(1, len(result["aspects"]))
        self.assertEqual("positive", result["aspects"][0]["sentiment"])

    def test_opposite_polarities_are_not_merged(self) -> None:
        review = "The first setup was easy, but reinstalling later was difficult."
        result = _analyze(
            review,
            [
                _mention("first setup", "Setup / Installation", "positive", "The first setup was easy"),
                _mention("reinstalling", "Setup / Installation", "negative", "reinstalling later was difficult"),
            ],
        )
        self.assertEqual(["positive", "negative"], [a["sentiment"] for a in result["aspects"]])

    def test_price_design_packaging_and_comparison_guards_remain(self) -> None:
        value = "The benefits justify the higher cost."
        self.assertEqual(
            "Price / Value",
            _analyze(
                value,
                [_mention("higher cost", "Price / Value", "positive", value)],
            )["aspects"][0]["aspect"],
        )

        design = "The narrow silver bezel looks elegant."
        self.assertEqual(
            "Design",
            _analyze(
                design,
                [_mention("silver bezel", "Picture Quality", "positive", design)],
            )["aspects"][0]["aspect"],
        )

        packaging = "The carton padding protected every corner."
        self.assertEqual(
            "Packaging",
            _analyze(
                packaging,
                [_mention("carton padding", "Build Quality", "positive", packaging)],
            )["aspects"][0]["aspect"],
        )

        comparison = "My previous screen froze, but this one opens apps quickly."
        result = _analyze(
            comparison,
            [
                _mention("previous screen", "System Performance", "negative", "My previous screen froze"),
                _mention("this one", "System Performance", "positive", "this one opens apps quickly"),
            ],
        )
        self.assertEqual(["this one opens apps quickly"], [a["evidence"] for a in result["aspects"]])


class Step8121InfrastructureTests(unittest.TestCase):
    def test_frozen_regression_diff_has_expected_extra_partition(self) -> None:
        if not all(Path(path).is_file() for path in repair.ROUND1_PATHS.values()):
            self.skipTest("Frozen Round 1 artifacts are not present in this checkout.")
        baseline, _cases, _hashes = repair.verify_frozen_step_811_inputs()
        repair.verify_frozen_round1_files()
        round1 = repair._load_json_object(
            repair.ROUND1_PATHS["report"], label="Round 1 report"
        )
        analysis, _rows = repair.build_round1_regression_analysis(baseline, round1)
        extras = analysis["extra_aspect_diff"]
        self.assertEqual(
            (18, 8, 9),
            (
                extras["persistent_count"],
                extras["fixed_count"],
                extras["new_count"],
            ),
        )

    def test_runtime_instrumentation_counts_provider_work(self) -> None:
        class Delegate:
            def analyze_review(self, review_text, allowed_aspects):
                return {"summary": "ok", "aspects": [_mention("screen", "Picture Quality", "positive", "screen is clear")]}

            def repair_evidence(self, review_text, invalid_aspects):
                return {"repairs": []}

        ticks = iter((1.0, 3.0, 4.0, 5.5))
        client = repair.InstrumentedReviewClient(Delegate(), clock=lambda: next(ticks))
        client.analyze_review("screen is clear", ("Picture Quality",))
        client.repair_evidence("screen is clear", [])
        timing = client.by_review["screen is clear"]
        self.assertEqual(2.0, timing["llm_analysis_seconds"])
        self.assertEqual(1.5, timing["llm_repair_seconds"])
        self.assertEqual(1, timing["generated_mentions"])

    def test_one_run_reservation_is_permanent_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / repair.REPAIR_STARTED_PATH.name
            repair.reserve_repair_evaluation(marker)
            self.assertTrue(marker.is_file())
            with self.assertRaises(GeneralizationDevV2IntegrityError):
                repair.reserve_repair_evaluation(marker)
            self.assertTrue(repair.repair_outputs_exist(directory))

    def test_acceptance_protects_runtime_and_priority_aspects(self) -> None:
        round1 = {
            "summary": {"aspect_f1": 0.7674},
            "priority_aspect_metrics": {
                "Price / Value": {"f1": 0.7692},
                "Design": {"f1": 0.7083},
                "Packaging": {"f1": 0.7742},
            },
            "runtime_statistics": {"failed_cases": 2},
        }
        report = {
            "summary": {"aspect_f1": 0.78, "evidence_validity_rate": 1.0},
            "error_counts": {"EXTRA_ASPECT": 20, "WRONG_ASPECT": 20, "MISS_ASPECT": 10},
            "priority_aspect_metrics": {
                "Price / Value": {"f1": 0.75},
                "Design": {"f1": 0.70},
                "Packaging": {"f1": 0.76},
            },
            "stable_aspect_metrics": {
                "Delivery": {"f1": 0.72},
                "Setup / Installation": {"f1": 0.85},
            },
            "runtime_statistics": {
                "average_seconds_per_review": 12.0,
                "failed_cases": 0,
            },
        }
        result = repair.repair_acceptance(report, round1)
        self.assertTrue(result["all_required_quality_checks_passed"])
        self.assertTrue(result["checks"]["zero_model_failures"])

        report["runtime_statistics"]["average_seconds_per_review"] = 16.0
        result = repair.repair_acceptance(report, round1)
        self.assertFalse(result["all_required_quality_checks_passed"])

    def test_cli_routes_both_repair_commands_without_real_inference(self) -> None:
        with patch.object(
            main_module,
            "run_generalization_dev_v2_regression_analysis",
            return_value=7,
        ) as offline:
            self.assertEqual(
                7,
                main_module.cli(["--analyze-generalization-dev-v2-regressions"]),
            )
            offline.assert_called_once_with()

        with patch.object(
            main_module,
            "run_generalization_dev_v2_repair_evaluation",
            return_value=8,
        ) as model:
            self.assertEqual(
                8,
                main_module.cli(["--evaluate-generalization-dev-v2-repair"]),
            )
            model.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
