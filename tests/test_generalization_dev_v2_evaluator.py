"""Offline tests for Step 8.11 Generalization Dev V2 infrastructure."""

from __future__ import annotations

import csv
import json
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import generalization_dev_v2_evaluator as v2
import main as main_module
from llm_client import LLMConnectionError, LLMResponseError


EXPECTED_V2_DATASET_SHA256 = (
    "ea6d264db2dea92fa6e0e2e469e2de572cac56654e916fcc3502bfaf23f650f2"
)

# Step 8.12 is authorized to improve llm_client.py, aspect_taxonomy.py, and
# review_analyzer.py.  Step 11 is separately authorized to improve the
# Dashboard and the shared Amazon/Upload/Paste runtime, so app.py and
# amazon_analysis.py are no longer source-code freeze targets.  Their data,
# old evaluation assets, evaluator.py, and Amazon scraping logic still fail
# loudly on mutation.
PROTECTED_PRE_STEP_8_11_SHA256 = {
    "evaluator.py": "310b29fa1952cb61e1d246c842e94c0ba1cad097c6d6e95541063c9cbd51cad2",
    "evaluation/gold_cases.json": "be361d8d3cb0bfaef6f0a4c70b9cb1c5fdac07d23b1a6f67456cfa01ab36f120",
    "evaluation/holdout_cases.json": "a368c34b48bf5e6dfcbdb3afa521e2bd7fbd3ab08a85a53c0f1b2b9837d5862a",
    "evaluation/generalization_dev_cases.json": "d1864655cc95d5fa82546ed29bb4af11626b7723c57a4b705e690614e330ac87",
    "evaluation/final_blind_cases.json": "3bfced46011b2fe0e8c4d1ce4ae271e2cd8f0916e9ef9e92eaca4299c16d66f9",
    "evaluation/final_blind_manifest.json": "4341945c9cbb58f54dfd06c564e7c47efdcc2c29c5b29aaa0b1deec3234a7c45",
    "outputs/evaluation/final_blind/final_blind_predictions.json": "f164ddb9161c7793d01bc347b7cb35aa15126725c8d3249db6a47fdb4eeb5a8b",
    "outputs/evaluation/final_blind/final_blind_evaluation.json": "4615e13458be85d094c6653aff58b72694382135e2436e12e924d54d2ec21e95",
    "outputs/evaluation/final_blind/final_blind_errors.csv": "6f02f8bd75b4b05188fdc3a7630acc2002b87bca4fa360e5a58c123024540296",
    "outputs/evaluation/generalization_error_study/generalization_dev_v2_spec.json": "06cdcba95b0bad3b9c285c4bad37bf5e513686f680437561ab1eec584b38d12f",
    "outputs/evaluation/generalization_error_study/generalization_error_report.json": "f7e9dbc342476d4e6ef97f49fbbe3686950b071b4383fc5456b83e57a0e5a9e9",
    "outputs/evaluation/generalization_error_study/generalization_case_analysis.csv": "1e48592cddccb7e1ac091d62d902e26784c44f70b0fe4b460d48edbf47bf7d3c",
    "data/raw_reviews.csv": "26e810e73c8f9175c95de086d8fd8892ea3ae9af3d6ff703e259c31315dd0ec8",
    "outputs/analyzed_reviews.csv": "ecf697503620a14c3e3ededd5ac4f25d921e4095b8c0d6ec4760f41a762afb73",
    "outputs/aspect_summary.csv": "aaacdd2f13ab7e4ff9558afd54f4f204ba2fc0e06ad643489eac968592ce7148",
    "outputs/business_summary.json": "e2137bb630c1f67e78c0067d829825fe3204f6127aac65dbda84022455de26f7",
    "outputs/aspect_evidence.json": "8f445c1f1bf2f6de08f80874f673adcf9eccf1805b5a4f207ee079901259310f",
    "outputs/keyword_summary.csv": "796f1aaa5aa52f065c851f689476d8f47312bbe1df1f59e3dbb05578b4a86031",
    "outputs/sentiment_keywords.json": "5d5150655dc686ed563e4d85bf38f6a9b0066e403650c27014c6a4f0fa621cfc",
    "outputs/aspect_keywords.json": "a334174f13c9424c451c2b308b1d112156182ba373fa363222a70891b2b94009",
    "browser_amazon_import.py": "3529f98ccd2f4848fa0f15f54f397302e215ce61b08bc753e116a4b7c6b42de4",
    "amazon_url.py": "f0f66283c831b9579d580251f2978aa0422c19ed52a49d53e90215b0cf64b6c8",
    "scraper.py": "b89a9f5586bd68d50369aaeae567d263dc2ba5802c5e159930985869f1cb5ed1",
}


def _case(
    index: int,
    *,
    review: str | None = None,
    expected: list[dict[str, str]] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": f"V2D{index:03d}",
        "review": review or f"Unique factual review number {index} has no opinion.",
        "expected_aspects": expected or [],
        "tags": tags or ["offline_fixture"],
        "notes": f"Offline annotation {index}.",
    }


def _mention(aspect: str, sentiment: str, evidence: str) -> dict[str, str]:
    return {
        "raw_aspect": aspect,
        "aspect": aspect,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _analysis(aspects: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "summary": "Offline mock.",
        "aspects": aspects or [],
        "positive_points": [],
        "negative_points": [],
    }


def _write_cases(root: Path, cases: list[dict[str, Any]]) -> Path:
    path = root / "generalization_dev_v2_cases.json"
    path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    return path


def _distribution_spec(cases: list[dict[str, Any]]) -> dict[str, Any]:
    distribution = v2.dataset_distribution(cases)
    return {
        "target_cases": len(cases),
        "target_mentions": distribution["gold_mention_count"],
        "target_aspect_distribution": distribution["aspect_distribution"],
        "case_cardinality_distribution": distribution[
            "case_cardinality_distribution"
        ],
        "boundary_case_distribution": {"taxonomy_boundary_cases": 0},
        "overlapping_language_and_structure_targets": {
            "experience_vs_product": 0,
            "same_aspect_multi_mention": 0,
            "multi_aspect_sentences": 0,
            "negation": 0,
            "contrast": 0,
            "comparison": 0,
            "temporal_change": 0,
            "weak_sentiment": 0,
            "long_reviews": 0,
            "no_aspect": 0,
        },
    }


def _fixture(
    root: Path, cases: list[dict[str, Any]]
) -> tuple[Path, Path, Path, Path]:
    dataset = _write_cases(root, cases)
    spec = root / "generalization_dev_v2_spec.json"
    spec.write_text(
        json.dumps(_distribution_spec(cases), indent=2) + "\n", encoding="utf-8"
    )
    logic = root / "frozen_logic.py"
    logic.write_text("FROZEN = True\n", encoding="utf-8")
    manifest = root / "generalization_dev_v2_manifest.json"
    v2.create_generalization_dev_v2_manifest(
        dataset,
        manifest,
        spec_path=spec,
        expected_count=len(cases),
        comparison_dataset_paths=(),
        demo_paths=(),
        core_model_paths=(logic,),
    )
    return dataset, manifest, spec, logic


def _run(
    dataset: Path,
    manifest: Path,
    spec: Path,
    logic: Path,
    *,
    count: int,
    analyzer: Any,
) -> dict[str, Any]:
    return v2.run_generalization_dev_v2_baseline(
        dataset_path=dataset,
        manifest_path=manifest,
        spec_path=spec,
        expected_count=count,
        provider="ollama",
        model="qwen3:8b",
        comparison_dataset_paths=(),
        demo_paths=(),
        core_model_paths=(logic,),
        client=object(),
        analyzer=analyzer,
        sleeper=lambda _seconds: None,
        progress=lambda *_args: None,
    )


class V2DatasetValidationTests(unittest.TestCase):
    def test_schema_ids_and_case_sensitive_contiguous_evidence(self) -> None:
        cases = [
            _case(
                1,
                review="The thin bezel looks elegant.",
                expected=[
                    {
                        "aspect": "Design",
                        "sentiment": "positive",
                        "evidence": "thin bezel looks elegant",
                    }
                ],
            )
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_cases(Path(temporary), cases)
            validated = v2.validate_generalization_dev_v2_cases(
                path, expected_count=1
            )
            self.assertEqual("V2D001", validated[0]["case_id"])
            cases[0]["expected_aspects"][0]["evidence"] = "Thin bezel looks elegant"
            _write_cases(Path(temporary), cases)
            with self.assertRaisesRegex(v2.GeneralizationDevV2IntegrityError, "substring"):
                v2.validate_generalization_dev_v2_cases(path, expected_count=1)

    def test_duplicate_after_normalization_is_rejected(self) -> None:
        cases = [
            _case(1, review="Looks GOOD!"),
            _case(2, review=" looks good "),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_cases(Path(temporary), cases)
            with self.assertRaisesRegex(v2.GeneralizationDevV2IntegrityError, "unique"):
                v2.validate_generalization_dev_v2_cases(path, expected_count=2)

    def test_unknown_taxonomy_aspect_is_rejected(self) -> None:
        cases = [
            _case(
                1,
                review="The battery is excellent.",
                expected=[
                    {
                        "aspect": "Battery",
                        "sentiment": "positive",
                        "evidence": "battery is excellent",
                    }
                ],
            )
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_cases(Path(temporary), cases)
            with self.assertRaisesRegex(
                v2.GeneralizationDevV2IntegrityError, "fixed taxonomy|unknown Aspect"
            ):
                v2.validate_generalization_dev_v2_cases(path, expected_count=1)

    def test_distribution_checks_spec_and_hard_tag_minimum(self) -> None:
        cases = [
            _case(
                1,
                review="The screen is crisp.",
                expected=[
                    {
                        "aspect": "Picture Quality",
                        "sentiment": "positive",
                        "evidence": "screen is crisp",
                    }
                ],
                tags=["contrast"],
            )
        ]
        spec = _distribution_spec(cases)
        spec["overlapping_language_and_structure_targets"]["contrast"] = 1
        result = v2.validate_distribution_against_spec(cases, spec)
        self.assertEqual(1, result["minimum_tag_targets"]["contrast"])
        cases[0]["tags"] = ["offline_fixture"]
        with self.assertRaisesRegex(v2.GeneralizationDevV2IntegrityError, "contrast"):
            v2.validate_distribution_against_spec(cases, spec)

    def test_distribution_reports_gold_sentiment_aspect_and_tags(self) -> None:
        cases = [
            _case(
                1,
                review="The box protected it well.",
                expected=[
                    {
                        "aspect": "Packaging",
                        "sentiment": "positive",
                        "evidence": "box protected it well",
                    }
                ],
                tags=["taxonomy_boundary", "long_review"],
            )
        ]
        result = v2.dataset_distribution(cases)
        self.assertEqual(1, result["gold_mention_count"])
        self.assertEqual({"Packaging": 1}, result["aspect_distribution"])
        self.assertEqual({"positive": 1}, result["sentiment_distribution"])
        self.assertEqual(1, result["hard_case_tag_distribution"]["taxonomy_boundary"])


class V2FrozenProjectArtifactTests(unittest.TestCase):
    def test_project_dataset_matches_frozen_spec_and_required_coverage(self) -> None:
        cases = v2.validate_generalization_dev_v2_cases()
        spec = json.loads(v2.V2_SPEC_PATH.read_text(encoding="utf-8"))
        result = v2.validate_distribution_against_spec(cases, spec)

        self.assertEqual(110, result["case_count"])
        self.assertEqual(210, result["gold_mention_count"])
        self.assertEqual(EXPECTED_V2_DATASET_SHA256, v2.file_sha256(v2.V2_CASES_PATH))
        self.assertEqual(set(v2.STANDARD_ASPECTS), set(result["aspect_distribution"]))
        self.assertTrue(all(result["aspect_distribution"].values()))
        long_cases = [case for case in cases if "long_review" in case["tags"]]
        self.assertEqual(20, len(long_cases))
        self.assertTrue(all(len(case["review"].split()) >= 35 for case in long_cases))

    def test_project_dataset_has_no_old_data_leakage(self) -> None:
        result = v2.check_generalization_dev_v2_leakage()
        self.assertTrue(result["passed"])
        self.assertEqual(0, result["exact_duplicate_count"])
        self.assertEqual(0, result["normalized_duplicate_count"])
        self.assertLess(result["maximum_similarity"], v2.SIMILARITY_THRESHOLD)

    def test_project_manifest_preserves_step_8_11_core_logic_fingerprint(self) -> None:
        manifest = json.loads(v2.V2_MANIFEST_PATH.read_text(encoding="utf-8"))
        cases = v2.validate_generalization_dev_v2_cases()
        self.assertEqual(110, len(cases))
        self.assertTrue(manifest["created_before_evaluation"])
        self.assertEqual(EXPECTED_V2_DATASET_SHA256, manifest["sha256"])
        self.assertEqual(EXPECTED_V2_DATASET_SHA256, v2.file_sha256(v2.V2_CASES_PATH))
        self.assertTrue(manifest["leakage_check"]["passed"])
        self.assertEqual(
            {
                "llm_client.py": "158d9ecfa66847807f9bb48f1b7db1ee0a5f6c8fda3537c820a7a8543f6f365f",
                "aspect_taxonomy.py": "e3d5577df11c4b3087672bae5dffe1339a216d44e0faece1b6b90b96d9cea778",
                "review_analyzer.py": "7b9f457b70ccb97291a7b408ca988995531b1ae16d92707ef8912879d2263fff",
                "evaluator.py": "310b29fa1952cb61e1d246c842e94c0ba1cad097c6d6e95541063c9cbd51cad2",
            },
            manifest["core_model_file_sha256"],
        )

    def test_pre_step_8_11_assets_remain_byte_identical(self) -> None:
        for relative_path, expected_hash in PROTECTED_PRE_STEP_8_11_SHA256.items():
            with self.subTest(path=relative_path):
                path = v2.PROJECT_ROOT / relative_path
                self.assertTrue(path.is_file(), relative_path)
                self.assertEqual(expected_hash, v2.file_sha256(path))


class V2LeakageAndManifestTests(unittest.TestCase):
    def test_leakage_detects_exact_normalized_and_similarity(self) -> None:
        cases = [_case(1, review="The panel looks incredibly refined today.")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _write_cases(root, cases)
            old = root / "old.json"
            old.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "OLD1",
                            "review": "the panel looks incredibly refined today",
                            "expected_aspects": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            result = v2.check_generalization_dev_v2_leakage(
                dataset,
                expected_count=1,
                comparison_dataset_paths=(old,),
                demo_paths=(),
            )
        self.assertEqual(0, result["exact_duplicate_count"])
        self.assertEqual(1, result["normalized_duplicate_count"])
        self.assertGreaterEqual(result["maximum_similarity"], 0.75)
        self.assertFalse(result["passed"])

    def test_demo_csv_is_included_in_leakage_corpus(self) -> None:
        cases = [_case(1, review="An unmistakably fresh V2 sentence.")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _write_cases(root, cases)
            demo = root / "demo.csv"
            with demo.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("review_id", "content"))
                writer.writeheader()
                writer.writerow({"review_id": "D1", "content": cases[0]["review"]})
            result = v2.check_generalization_dev_v2_leakage(
                dataset,
                expected_count=1,
                comparison_dataset_paths=(),
                demo_paths=(demo,),
            )
        self.assertEqual(1, result["exact_duplicate_count"])
        self.assertEqual("demo.csv", result["closest_source"]["source"])

    def test_manifest_records_hash_distribution_timestamp_and_core_hashes(self) -> None:
        cases = [_case(1)]
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, spec, logic = _fixture(Path(temporary), cases)
            payload, validated = v2.verify_generalization_dev_v2_manifest(
                dataset,
                manifest,
                spec_path=spec,
                expected_count=1,
                comparison_dataset_paths=(),
                demo_paths=(),
                core_model_paths=(logic,),
            )
        self.assertEqual(1, len(validated))
        self.assertTrue(payload["created_before_evaluation"])
        self.assertEqual(64, len(payload["sha256"]))
        self.assertIn("created_at", payload)
        self.assertIn("core_model_file_sha256", payload)
        self.assertEqual(0, payload["gold_mention_count"])

    def test_manifest_and_dataset_are_immutable(self) -> None:
        cases = [_case(1)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, manifest, spec, logic = _fixture(root, cases)
            with self.assertRaisesRegex(v2.GeneralizationDevV2IntegrityError, "immutable"):
                v2.create_generalization_dev_v2_manifest(
                    dataset,
                    manifest,
                    spec_path=spec,
                    expected_count=1,
                    comparison_dataset_paths=(),
                    demo_paths=(),
                    core_model_paths=(logic,),
                )
            dataset.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(v2.GeneralizationDevV2IntegrityError, "changed"):
                v2.verify_generalization_dev_v2_manifest(
                    dataset,
                    manifest,
                    spec_path=spec,
                    expected_count=1,
                    comparison_dataset_paths=(),
                    demo_paths=(),
                    core_model_paths=(logic,),
                )

    def test_core_model_hash_change_fails_closed(self) -> None:
        cases = [_case(1)]
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, spec, logic = _fixture(Path(temporary), cases)
            logic.write_text("FROZEN = False\n", encoding="utf-8")
            with self.assertRaisesRegex(v2.GeneralizationDevV2IntegrityError, "hashes changed"):
                v2.verify_generalization_dev_v2_manifest(
                    dataset,
                    manifest,
                    spec_path=spec,
                    expected_count=1,
                    comparison_dataset_paths=(),
                    demo_paths=(),
                    core_model_paths=(logic,),
                )

    def test_invalid_manifest_blocks_client_creation(self) -> None:
        cases = [_case(1)]
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, spec, logic = _fixture(Path(temporary), cases)
            dataset.write_text("[]\n", encoding="utf-8")
            factory = Mock()
            with self.assertRaises(v2.GeneralizationDevV2IntegrityError):
                v2.run_generalization_dev_v2_baseline(
                    dataset_path=dataset,
                    manifest_path=manifest,
                    spec_path=spec,
                    expected_count=1,
                    comparison_dataset_paths=(),
                    demo_paths=(),
                    core_model_paths=(logic,),
                    client_factory=factory,
                )
            factory.assert_not_called()


class V2RunnerAndMetricsTests(unittest.TestCase):
    def test_runner_reuses_metrics_and_reports_priority_aspect(self) -> None:
        review = "Worth every dollar."
        cases = [
            _case(
                1,
                review=review,
                expected=[
                    {
                        "aspect": "Price / Value",
                        "sentiment": "positive",
                        "evidence": "Worth every dollar",
                    }
                ],
            )
        ]
        analyzer = Mock(
            return_value=_analysis(
                [_mention("Price / Value", "positive", "Worth every dollar")]
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, spec, logic = _fixture(Path(temporary), cases)
            report = _run(
                dataset, manifest, spec, logic, count=1, analyzer=analyzer
            )
        self.assertEqual(1, report["summary"]["aspect_true_positives"])
        self.assertEqual(1.0, report["summary"]["aspect_f1"])
        self.assertEqual(1, report["summary"]["exact_matches"])
        price = report["priority_aspect_metrics"]["Price / Value"]
        self.assertEqual((1, 1, 0, 0), (price["support"], price["tp"], price["fp"], price["fn"]))
        self.assertEqual(1.0, price["f1"])
        self.assertTrue(report["core_model_hashes_unchanged"])

    def test_transient_connection_failure_retries_once_then_succeeds(self) -> None:
        cases = [_case(1)]
        analyzer = Mock(
            side_effect=[LLMConnectionError("temporary"), _analysis()]
        )
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, spec, logic = _fixture(Path(temporary), cases)
            report = _run(
                dataset, manifest, spec, logic, count=1, analyzer=analyzer
            )
        self.assertEqual(2, analyzer.call_count)
        self.assertEqual(1, report["runtime_statistics"]["retry_count"])
        self.assertEqual(1, report["summary"]["successful_cases"])

    def test_second_connection_failure_is_recorded_without_more_retries(self) -> None:
        cases = [_case(1)]
        analyzer = Mock(side_effect=LLMConnectionError("still unavailable"))
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, spec, logic = _fixture(Path(temporary), cases)
            report = _run(
                dataset, manifest, spec, logic, count=1, analyzer=analyzer
            )
        self.assertEqual(2, analyzer.call_count)
        self.assertEqual(1, report["error_counts"]["MODEL_ERROR"])
        self.assertEqual(1, report["summary"]["failed_cases"])

    def test_schema_error_does_not_retry(self) -> None:
        cases = [_case(1)]
        analyzer = Mock(side_effect=LLMResponseError("bad schema"))
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, spec, logic = _fixture(Path(temporary), cases)
            report = _run(
                dataset, manifest, spec, logic, count=1, analyzer=analyzer
            )
        self.assertEqual(1, analyzer.call_count)
        self.assertEqual(1, report["error_counts"]["SCHEMA_ERROR"])
        self.assertEqual(0, report["runtime_statistics"]["retry_count"])

    def test_all_error_types_are_present_even_when_zero(self) -> None:
        cases = [_case(1)]
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, spec, logic = _fixture(Path(temporary), cases)
            report = _run(
                dataset,
                manifest,
                spec,
                logic,
                count=1,
                analyzer=Mock(return_value=_analysis()),
            )
        self.assertEqual(set(v2.ERROR_TYPES), set(report["error_counts"]))
        self.assertIn("p90_seconds_per_review", report["runtime_statistics"])
        self.assertIn("cold_start_first_case_seconds", report["runtime_statistics"])

    def test_model_logic_mutation_during_run_is_detected(self) -> None:
        cases = [_case(1)]
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, spec, logic = _fixture(Path(temporary), cases)

            def mutating_analyzer(_review: str, **_kwargs: Any) -> dict[str, Any]:
                logic.write_text("CHANGED = True\n", encoding="utf-8")
                return _analysis()

            with self.assertRaisesRegex(v2.GeneralizationDevV2IntegrityError, "hashes changed"):
                _run(
                    dataset,
                    manifest,
                    spec,
                    logic,
                    count=1,
                    analyzer=mutating_analyzer,
                )

    def test_injected_runner_makes_no_network_or_default_client_call(self) -> None:
        cases = [_case(1)]
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, spec, logic = _fixture(Path(temporary), cases)
            with patch.object(socket, "create_connection", side_effect=AssertionError("network")), patch.object(
                v2, "create_review_client", side_effect=AssertionError("real client")
            ):
                report = _run(
                    dataset,
                    manifest,
                    spec,
                    logic,
                    count=1,
                    analyzer=Mock(return_value=_analysis()),
                )
        self.assertEqual(1, report["summary"]["successful_cases"])


class V2PersistenceAndFormattingTests(unittest.TestCase):
    def _report(self) -> dict[str, Any]:
        return {
            "evaluation_policy": f"{v2.STEP_NOTICE} {v2.OPTIMIZATION_NOTICE}",
            "provider": "ollama",
            "model": "qwen3:8b",
            "dataset_sha256": "a" * 64,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "case_count": 1,
            "summary": {
                "total_cases": 1,
                "successful_cases": 1,
                "failed_cases": 0,
                "aspect_precision": 1.0,
                "aspect_recall": 1.0,
                "aspect_f1": 1.0,
                "aspect_only_precision": 1.0,
                "aspect_only_recall": 1.0,
                "aspect_only_f1": 1.0,
                "sentiment_accuracy": 1.0,
                "evidence_validity_rate": 1.0,
                "evidence_match_rate": 1.0,
                "exact_matches": 1,
                "aspect_true_positives": 1,
                "aspect_false_positives": 0,
                "aspect_false_negatives": 0,
            },
            "runtime_statistics": {},
            "error_counts": {key: 0 for key in v2.ERROR_TYPES},
            "priority_aspect_metrics": {
                aspect: {
                    "support": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
                for aspect in v2.PRIORITY_ASPECTS
            },
            "cases": [],
            "predictions": [],
            "core_model_hashes_unchanged": True,
        }

    def test_three_outputs_are_written_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "baseline_path": root / "baseline.json",
                "errors_path": root / "errors.csv",
                "predictions_path": root / "predictions.json",
            }
            result = v2.save_generalization_dev_v2_outputs(self._report(), **paths)
            self.assertEqual(set(result), {"baseline", "errors", "predictions"})
            self.assertTrue(all(Path(path).is_file() for path in result.values()))
            with self.assertRaisesRegex(v2.GeneralizationDevV2IntegrityError, "already exist"):
                v2.save_generalization_dev_v2_outputs(self._report(), **paths)

    def test_formatter_includes_metrics_priorities_and_required_notices(self) -> None:
        text = v2.format_generalization_dev_v2_report(self._report())
        self.assertIn(v2.STEP_NOTICE, text)
        self.assertIn(v2.OPTIMIZATION_NOTICE, text)
        self.assertIn("Price / Value", text)
        self.assertIn("Aspect F1: 100.00%", text)
        self.assertIn("Core model hashes unchanged: True", text)

    def test_module_has_no_amazon_or_openai_dependency(self) -> None:
        source = Path(v2.__file__).read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\s*(?:from|import)\s+amazon")
        self.assertNotRegex(source, r"(?m)^\s*(?:from|import)\s+openai")


class V2CliTests(unittest.TestCase):
    def test_cli_flag_dispatches_to_v2_entrypoint(self) -> None:
        with patch.object(
            main_module, "run_generalization_dev_v2_evaluation", return_value=7
        ) as runner:
            result = main_module.cli(["--evaluate-generalization-dev-v2"])
        self.assertEqual(7, result)
        runner.assert_called_once_with()

    def test_existing_outputs_block_runner_and_model_work(self) -> None:
        output = StringIO()
        with patch.object(
            main_module, "generalization_dev_v2_outputs_exist", return_value=True
        ), patch.object(
            main_module, "run_generalization_dev_v2_baseline"
        ) as runner, redirect_stdout(output):
            result = main_module.run_generalization_dev_v2_evaluation()
        self.assertEqual(1, result)
        runner.assert_not_called()
        self.assertIn("will not be run or overwritten", output.getvalue())

    def test_missing_or_invalid_manifest_is_reported_without_save(self) -> None:
        output = StringIO()
        with patch.object(
            main_module, "generalization_dev_v2_outputs_exist", return_value=False
        ), patch.object(
            main_module,
            "run_generalization_dev_v2_baseline",
            side_effect=v2.GeneralizationDevV2IntegrityError(
                "V2 manifest was not found"
            ),
        ) as runner, patch.object(
            main_module, "save_generalization_dev_v2_outputs"
        ) as saver, redirect_stdout(output):
            result = main_module.run_generalization_dev_v2_evaluation()
        self.assertEqual(1, result)
        runner.assert_called_once_with(provider="ollama", model="qwen3:8b")
        saver.assert_not_called()
        self.assertIn("manifest was not found", output.getvalue())

    def test_successful_cli_saves_and_formats_baseline(self) -> None:
        report = {"summary": {"total_cases": 110}}
        paths = {
            "baseline": "/tmp/baseline.json",
            "errors": "/tmp/errors.csv",
            "predictions": "/tmp/predictions.json",
        }
        output = StringIO()
        with patch.object(
            main_module, "generalization_dev_v2_outputs_exist", return_value=False
        ), patch.object(
            main_module, "run_generalization_dev_v2_baseline", return_value=report
        ) as runner, patch.object(
            main_module, "save_generalization_dev_v2_outputs", return_value=paths
        ) as saver, patch.object(
            main_module, "format_generalization_dev_v2_report", return_value="V2 report"
        ), redirect_stdout(output):
            result = main_module.run_generalization_dev_v2_evaluation()
        self.assertEqual(0, result)
        runner.assert_called_once_with(provider="ollama", model="qwen3:8b")
        saver.assert_called_once_with(report)
        rendered = output.getvalue()
        self.assertIn("V2 report", rendered)
        self.assertIn(paths["predictions"], rendered)


if __name__ == "__main__":
    unittest.main()
