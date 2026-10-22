from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import blind_v2_evaluator as blind
import main
from aspect_taxonomy import STANDARD_ASPECTS
from llm_client import LLMConnectionError, LLMResponseError


def _case(case_id: str, review: str = "A plain review with no rated feature.") -> dict:
    return {
        "case_id": case_id,
        "review": review,
        "expected_aspects": [],
        "tags": ["no_aspect"],
        "notes": "Synthetic infrastructure fixture without a product opinion.",
    }


def _analysis(aspects: list[dict] | None = None) -> dict:
    return {
        "summary": "Test summary.",
        "aspects": aspects or [],
        "positive_points": [],
        "negative_points": [],
        "keywords": [],
        "warnings": [],
    }


class BlindV2DatasetTests(unittest.TestCase):
    def test_real_dataset_has_150_ordered_cases(self) -> None:
        cases = blind.validate_blind_v2_cases()
        self.assertEqual(len(cases), 150)
        self.assertEqual(cases[0]["case_id"], "B2-001")
        self.assertEqual(cases[-1]["case_id"], "B2-150")

    def test_real_dataset_distribution_is_valid(self) -> None:
        distribution = blind.validate_blind_v2_distribution(
            blind.validate_blind_v2_cases()
        )
        self.assertEqual(
            distribution["case_cardinality_distribution"],
            {
                "no_aspect": 15,
                "one_mention": 40,
                "three_to_five_mentions": 35,
                "two_mentions": 60,
            },
        )
        self.assertGreater(distribution["gold_mention_count"], 250)

    def test_real_dataset_covers_every_taxonomy_label(self) -> None:
        distribution = blind.dataset_distribution(blind.validate_blind_v2_cases())
        self.assertEqual(set(distribution["aspect_distribution"]), set(STANDARD_ASPECTS))
        self.assertTrue(
            all(count >= 8 for count in distribution["aspect_distribution"].values())
        )

    def test_real_dataset_evidence_is_exact_substring(self) -> None:
        for case in blind.validate_blind_v2_cases():
            for mention in case["expected_aspects"]:
                values = (
                    mention["evidence"]
                    if isinstance(mention["evidence"], list)
                    else [mention["evidence"]]
                )
                for evidence in values:
                    self.assertIn(evidence, case["review"])

    def test_duplicate_review_is_rejected(self) -> None:
        cases = [_case(f"B2-{index:03d}", f"Review {index} is factual.") for index in range(1, 151)]
        cases[1]["review"] = cases[0]["review"].upper()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            with self.assertRaises(blind.BlindV2IntegrityError):
                blind.validate_blind_v2_cases(path)

    def test_leakage_checker_detects_high_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "new.json"
            source = root / "old.json"
            demo = root / "demo.csv"
            review = "The unusually narrow silver frame looks elegant beside my shelf."
            dataset.write_text(json.dumps([_case("B2-001", review)]), encoding="utf-8")
            source.write_text(
                json.dumps([{"case_id": "OLD", "review": review}]), encoding="utf-8"
            )
            demo.write_text("review_id,content\n", encoding="utf-8")
            result = blind.check_blind_v2_leakage(
                dataset,
                expected_count=1,
                comparison_dataset_paths=(source,),
                demo_paths=(demo,),
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["exact_duplicate_count"], 1)
        self.assertEqual(result["similarity_violation_count"], 1)

    def test_manifest_matches_real_dataset_hash(self) -> None:
        manifest = json.loads(blind.BLIND_V2_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["sha256"], blind.file_sha256(blind.BLIND_V2_CASES_PATH))
        self.assertTrue(manifest["created_before_evaluation"])


class BlindV2MetricTests(unittest.TestCase):
    def test_per_aspect_metrics_include_all_labels(self) -> None:
        case_result = {
            "expected_aspects": [
                {
                    "aspect": "Design",
                    "sentiment": "positive",
                    "evidence": "looks polished",
                }
            ],
            "actual_aspects": [
                {
                    "aspect": "Design",
                    "sentiment": "positive",
                    "evidence": "looks polished",
                }
            ],
        }
        metrics = blind.calculate_per_aspect_metrics([case_result])
        self.assertEqual(set(metrics), set(STANDARD_ASPECTS))
        self.assertEqual(metrics["Design"]["support"], 1)
        self.assertEqual(metrics["Design"]["f1"], 1.0)

    def _run_fixture(self, analyzer, directory: Path) -> dict:
        core = directory / "core.py"
        protected = directory / "protected.txt"
        dataset = directory / "dataset.json"
        manifest = directory / "manifest.json"
        started = directory / "started.json"
        comparison = directory / "comparison.json"
        core.write_text("frozen", encoding="utf-8")
        protected.write_text("frozen", encoding="utf-8")
        dataset.write_text(json.dumps([_case("B2-001")]), encoding="utf-8")
        comparison.write_text(json.dumps({"summary": {}}), encoding="utf-8")
        fake_manifest = {
            "sha256": blind.file_sha256(dataset),
            "created_at": "2026-01-01T00:00:00+00:00",
            "core_model_file_sha256": {str(core): blind.file_sha256(core)},
            "leakage_check": {"passed": True, "maximum_similarity": 0.1},
        }
        reports = {
            "step8.9_final_blind_v1": comparison,
            "step8.11_dev_v2_baseline": comparison,
            "step8.12.1_dev_v2_candidate": comparison,
        }
        with patch.object(
            blind,
            "verify_blind_v2_manifest",
            return_value=(fake_manifest, [_case("B2-001")]),
        ):
            return blind.run_blind_v2_evaluation(
                dataset_path=dataset,
                manifest_path=manifest,
                started_path=started,
                expected_count=1,
                core_model_paths=(core,),
                protected_paths=(protected,),
                comparison_report_paths=reports,
                client=object(),
                analyzer=analyzer,
                sleeper=lambda _seconds: None,
                clock=iter([0.0, 1.0]).__next__,
                progress=lambda *_args: None,
            )

    def test_transient_connection_error_retries_once(self) -> None:
        calls = 0

        def analyzer(_review: str, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise LLMConnectionError("temporary connection reset")
            return _analysis()

        with tempfile.TemporaryDirectory() as directory:
            report = self._run_fixture(analyzer, Path(directory))
        self.assertEqual(calls, 2)
        self.assertEqual(report["runtime_statistics"]["retry_count"], 1)
        self.assertEqual(report["summary"]["failed_cases"], 0)

    def test_connection_failure_twice_is_failed(self) -> None:
        calls = 0

        def analyzer(_review: str, **_kwargs):
            nonlocal calls
            calls += 1
            raise LLMConnectionError("still unavailable")

        with tempfile.TemporaryDirectory() as directory:
            report = self._run_fixture(analyzer, Path(directory))
        self.assertEqual(calls, 2)
        self.assertEqual(report["summary"]["failed_cases"], 1)
        self.assertEqual(report["error_counts"]["MODEL_ERROR"], 1)

    def test_schema_error_is_not_retried(self) -> None:
        calls = 0

        def analyzer(_review: str, **_kwargs):
            nonlocal calls
            calls += 1
            raise LLMResponseError("invalid schema")

        with tempfile.TemporaryDirectory() as directory:
            report = self._run_fixture(analyzer, Path(directory))
        self.assertEqual(calls, 1)
        self.assertEqual(report["error_counts"]["SCHEMA_ERROR"], 1)

    def test_started_sentinel_prevents_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "started.json"
            blind._write_started_sentinel(path, dataset_sha256="abc", core_hashes={})
            with self.assertRaises(blind.BlindV2IntegrityError):
                blind._write_started_sentinel(path, dataset_sha256="abc", core_hashes={})

    def test_save_outputs_creates_freeze_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "cases.json"
            manifest = root / "manifest.json"
            started = root / "started.json"
            dataset.write_text("[]", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            started.write_text("{}", encoding="utf-8")
            paths = blind.save_blind_v2_outputs(
                {
                    "provider": "ollama",
                    "model": "qwen3:8b",
                    "dataset_sha256": blind.file_sha256(dataset),
                    "case_count": 0,
                    "timestamp": "now",
                    "predictions": [],
                    "cases": [],
                    "core_model_file_sha256_after": {},
                    "protected_asset_sha256_after": {},
                },
                report_path=root / "report.json",
                errors_path=root / "errors.csv",
                predictions_path=root / "predictions.json",
                freeze_path=root / "freeze.json",
                dataset_path=dataset,
                manifest_path=manifest,
                started_path=started,
            )
            freeze = json.loads(Path(paths["freeze_manifest"]).read_text())
        self.assertTrue(freeze["no_post_blind_optimization"])
        self.assertEqual(
            set(freeze["frozen_file_sha256"]),
            {
                "dataset",
                "manifest",
                "predictions",
                "evaluation_report",
                "errors_csv",
                "started_sentinel",
            },
        )


class BlindV2CliTests(unittest.TestCase):
    def test_parser_accepts_blind_v2_flag(self) -> None:
        args = main._parse_args(["--evaluate-blind-v2"])
        self.assertTrue(args.evaluate_blind_v2)

    @patch.object(main, "run_blind_v2_test", return_value=0)
    def test_cli_dispatches_blind_v2(self, runner) -> None:
        self.assertEqual(main.cli(["--evaluate-blind-v2"]), 0)
        runner.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
