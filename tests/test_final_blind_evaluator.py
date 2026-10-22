"""Offline tests for the one-time Step 8.9 Final Blind infrastructure."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import final_blind_evaluator as final_eval
import main as main_module
from aspect_taxonomy import STANDARD_ASPECTS
from evaluator import evaluate_case
from llm_client import LLMClientError, LLMConnectionError, LLMResponseError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _case(
    index: int,
    *,
    review: str | None = None,
    expected: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": f"F{index:03d}",
        "review": review or f"Final review number {index} has no explicit opinion.",
        "expected_aspects": expected or [],
        "notes": f"Independent Final annotation {index}.",
        "tags": ["offline_fixture"],
    }


def _analysis(aspects: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "summary": "Offline fake analysis.",
        "aspects": aspects or [],
        "positive_points": [],
        "negative_points": [],
    }


def _mention(aspect: str, sentiment: str, evidence: str) -> dict[str, str]:
    return {
        "raw_aspect": aspect,
        "aspect": aspect,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _write_dataset(root: Path, cases: list[dict[str, Any]]) -> Path:
    path = root / "final_blind_cases.json"
    path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _fixture(
    root: Path,
    cases: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    dataset = _write_dataset(root, cases)
    manifest = root / "final_blind_manifest.json"
    logic = root / "frozen_logic.py"
    logic.write_text("FROZEN = True\n", encoding="utf-8")
    final_eval.create_final_blind_manifest(
        dataset,
        manifest,
        expected_count=len(cases),
        comparison_dataset_paths=(),
        demo_paths=(),
        analysis_paths=(logic,),
    )
    return dataset, manifest, logic


def _run(
    dataset: Path,
    manifest: Path,
    logic: Path,
    *,
    count: int,
    analyzer: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    return final_eval.run_final_blind_evaluation(
        dataset_path=dataset,
        manifest_path=manifest,
        expected_count=count,
        provider="ollama",
        model="qwen3:8b",
        analysis_paths=(logic,),
        comparison_dataset_paths=(),
        demo_paths=(),
        development_report_paths={},
        client=object(),
        analyzer=analyzer,
        sleeper=lambda _seconds: None,
        progress=lambda *_args: None,
        **kwargs,
    )


class FinalDatasetValidationTests(unittest.TestCase):
    def test_schema_count_ids_and_contiguous_evidence(self) -> None:
        cases = [
            _case(
                1,
                review="The image stays crisp in dark scenes.",
                expected=[
                    {
                        "aspect": "Picture Quality",
                        "sentiment": "positive",
                        "evidence": "image stays crisp",
                    }
                ],
            ),
            _case(2),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _write_dataset(Path(temporary), cases)
            validated = final_eval.validate_final_blind_cases(
                dataset,
                expected_count=2,
            )
        self.assertEqual(["F001", "F002"], [case["case_id"] for case in validated])

    def test_invalid_evidence_and_normalized_duplicate_are_rejected(self) -> None:
        bad_evidence = [
            _case(
                1,
                review="The image stays crisp.",
                expected=[
                    {
                        "aspect": "Picture Quality",
                        "sentiment": "positive",
                        "evidence": "image looks crisp",
                    }
                ],
            )
        ]
        duplicates = [
            _case(1, review="Same words!"),
            _case(2, review="same words"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                final_eval.FinalBlindIntegrityError,
                "contiguous review substring",
            ):
                final_eval.validate_final_blind_cases(
                    _write_dataset(root, bad_evidence),
                    expected_count=1,
                )
            with self.assertRaisesRegex(
                final_eval.FinalBlindIntegrityError,
                "unique after normalization",
            ):
                final_eval.validate_final_blind_cases(
                    _write_dataset(root, duplicates),
                    expected_count=2,
                )

    def test_real_dataset_has_100_cases_and_required_distribution(self) -> None:
        path = PROJECT_ROOT / "evaluation" / "final_blind_cases.json"
        if not path.is_file():
            raise unittest.SkipTest("Final Blind dataset is assembled by this task.")
        cases = final_eval.validate_final_blind_cases(path, expected_count=100)
        mentions = [
            mention for case in cases for mention in case["expected_aspects"]
        ]
        self.assertEqual(set(STANDARD_ASPECTS), {item["aspect"] for item in mentions})
        self.assertEqual(
            {"positive", "negative", "neutral"},
            {item["sentiment"] for item in mentions},
        )
        cardinalities = {len(case["expected_aspects"]) for case in cases}
        self.assertTrue({0, 1, 2}.issubset(cardinalities))
        self.assertTrue(any(3 <= count <= 5 for count in cardinalities))
        self.assertTrue(any(
            len([item["aspect"] for item in case["expected_aspects"]])
            != len({item["aspect"] for item in case["expected_aspects"]})
            for case in cases
        ))
        self.assertTrue(any(
            any(
                {"positive", "negative"}.issubset({
                    item["sentiment"]
                    for item in case["expected_aspects"]
                    if item["aspect"] == aspect
                })
                for aspect in STANDARD_ASPECTS
            )
            for case in cases
        ))
        lengths = [len(case["review"].split()) for case in cases]
        self.assertLessEqual(min(lengths), 8)
        self.assertGreaterEqual(max(lengths), 40)


class FinalLeakageAndManifestTests(unittest.TestCase):
    def test_leakage_checker_reports_exact_normalized_and_similar_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.json"
            old.write_text(
                json.dumps([
                    {"case_id": "OLD1", "review": "A distinct original sentence."},
                    {
                        "case_id": "OLD2",
                        "review": "The menu responds quickly whenever I select an app.",
                    },
                ]),
                encoding="utf-8",
            )
            demo = root / "demo.csv"
            with demo.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("review_id", "content"))
                writer.writeheader()
                writer.writerow({"review_id": "R1", "content": "A distinct original sentence."})

            exact_dataset = _write_dataset(
                root,
                [_case(1, review="A distinct original sentence.")],
            )
            exact = final_eval.check_final_blind_leakage(
                exact_dataset,
                expected_count=1,
                comparison_dataset_paths=(old,),
                demo_paths=(demo,),
            )
            self.assertFalse(exact["passed"])
            self.assertGreaterEqual(exact["exact_duplicate_count"], 1)
            self.assertGreaterEqual(exact["normalized_duplicate_count"], 1)

            similar_dataset = _write_dataset(
                root,
                [_case(1, review="The menu responds quickly whenever I choose an app.")],
            )
            similar = final_eval.check_final_blind_leakage(
                similar_dataset,
                expected_count=1,
                comparison_dataset_paths=(old,),
                demo_paths=(),
            )
            self.assertGreaterEqual(
                similar["maximum_similarity"],
                final_eval.SIMILARITY_THRESHOLD,
            )
            self.assertFalse(similar["passed"])
            self.assertEqual("OLD2", similar["closest_source"]["source_id"])

    def test_manifest_records_dataset_logic_config_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, manifest_path, logic = _fixture(root, [_case(1)])
            manifest, cases = final_eval.verify_final_blind_manifest(
                dataset,
                manifest_path,
                expected_count=1,
                analysis_paths=(logic,),
                comparison_dataset_paths=(),
                demo_paths=(),
            )
            dataset_hash = final_eval.file_sha256(dataset)
        self.assertEqual(1, manifest["case_count"])
        self.assertEqual(dataset_hash, manifest["sha256"])
        self.assertTrue(manifest["created_at"])
        self.assertEqual(final_eval.TAXONOMY_VERSION, manifest["taxonomy_version"])
        self.assertEqual("qwen3:8b", manifest["runtime_config"]["model"])
        self.assertTrue(manifest["analysis_fingerprint"])
        self.assertTrue(manifest["leakage_check"]["passed"])
        self.assertEqual(1, len(cases))

    def test_dataset_or_analysis_tampering_aborts_before_client_creation(self) -> None:
        for tamper_target in ("dataset", "logic"):
            with self.subTest(tamper_target=tamper_target):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    dataset, manifest, logic = _fixture(root, [_case(1)])
                    target = dataset if tamper_target == "dataset" else logic
                    target.write_text(
                        target.read_text(encoding="utf-8") + " ",
                        encoding="utf-8",
                    )
                    factory = Mock(return_value=object())
                    analyzer = Mock(return_value=_analysis())
                    with self.assertRaises(final_eval.FinalBlindIntegrityError):
                        final_eval.run_final_blind_evaluation(
                            dataset_path=dataset,
                            manifest_path=manifest,
                            expected_count=1,
                            analysis_paths=(logic,),
                            comparison_dataset_paths=(),
                            demo_paths=(),
                            development_report_paths={},
                            client_factory=factory,
                            analyzer=analyzer,
                        )
                    factory.assert_not_called()
                    analyzer.assert_not_called()

    def test_comparison_corpus_tampering_aborts_before_client_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _write_dataset(root, [_case(1)])
            logic = root / "frozen_logic.py"
            logic.write_text("FROZEN = True\n", encoding="utf-8")
            old = root / "old.json"
            old.write_text(
                json.dumps([{"case_id": "OLD1", "review": "An unrelated old sentence."}]),
                encoding="utf-8",
            )
            manifest = root / "final_blind_manifest.json"
            final_eval.create_final_blind_manifest(
                dataset,
                manifest,
                expected_count=1,
                comparison_dataset_paths=(old,),
                demo_paths=(),
                analysis_paths=(logic,),
            )
            old.write_text(
                json.dumps([{"case_id": "OLD1", "review": "The old corpus changed."}]),
                encoding="utf-8",
            )
            factory = Mock(return_value=object())
            with self.assertRaisesRegex(
                final_eval.FinalBlindIntegrityError,
                "comparison corpus changed",
            ):
                final_eval.run_final_blind_evaluation(
                    dataset_path=dataset,
                    manifest_path=manifest,
                    expected_count=1,
                    analysis_paths=(logic,),
                    comparison_dataset_paths=(old,),
                    demo_paths=(),
                    development_report_paths={},
                    client_factory=factory,
                    analyzer=Mock(return_value=_analysis()),
                )
            factory.assert_not_called()

    def test_real_manifest_preserves_final_blind_dataset_and_original_analysis(self) -> None:
        dataset = PROJECT_ROOT / "evaluation" / "final_blind_cases.json"
        manifest = PROJECT_ROOT / "evaluation" / "final_blind_manifest.json"
        if not dataset.is_file() or not manifest.is_file():
            raise unittest.SkipTest("Final Blind manifest is created by this task.")
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        cases = final_eval.validate_final_blind_cases(dataset)
        self.assertEqual(100, len(cases))
        self.assertEqual(final_eval.file_sha256(dataset), recorded["sha256"])
        self.assertTrue(recorded["leakage_check"]["passed"])
        # Step 8.12 is allowed to improve current production analysis logic, but
        # the Final Blind manifest must continue to record the exact logic used
        # for its one-time frozen evaluation. It must never be rewritten to the
        # optimized hashes or used to rerun that test.
        self.assertEqual(
            "95aeddc1f1d04d38a7adf44a49bd399a0ece7d1d5917415ef692c70dc7b9249f",
            recorded["analysis_fingerprint"],
        )
        self.assertEqual(
            "158d9ecfa66847807f9bb48f1b7db1ee0a5f6c8fda3537c820a7a8543f6f365f",
            recorded["analysis_file_sha256"]["llm_client.py"],
        )
        self.assertEqual(
            "e3d5577df11c4b3087672bae5dffe1339a216d44e0faece1b6b90b96d9cea778",
            recorded["analysis_file_sha256"]["aspect_taxonomy.py"],
        )
        self.assertEqual(
            "7b9f457b70ccb97291a7b408ca988995531b1ae16d92707ef8912879d2263fff",
            recorded["analysis_file_sha256"]["review_analyzer.py"],
        )


class FinalMetricAndExecutionTests(unittest.TestCase):
    def test_final_summary_uses_established_metric_calculation(self) -> None:
        positive_review = "The picture is crisp."
        cases = [
            _case(
                1,
                review=positive_review,
                expected=[
                    {
                        "aspect": "Picture Quality",
                        "sentiment": "positive",
                        "evidence": "picture is crisp",
                    }
                ],
            ),
            _case(
                2,
                review="The speakers buzz.",
                expected=[
                    {
                        "aspect": "Sound Quality",
                        "sentiment": "negative",
                        "evidence": "speakers buzz",
                    }
                ],
            ),
        ]
        analyzer = Mock(side_effect=[
            _analysis([_mention("Picture Quality", "positive", "picture is crisp")]),
            _analysis(),
        ])
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, logic = _fixture(Path(temporary), cases)
            report = _run(
                dataset,
                manifest,
                logic,
                count=2,
                analyzer=analyzer,
            )
        summary = report["summary"]
        self.assertEqual(1, summary["aspect_true_positives"])
        self.assertEqual(0, summary["aspect_false_positives"])
        self.assertEqual(1, summary["aspect_false_negatives"])
        self.assertEqual(1.0, summary["aspect_precision"])
        self.assertEqual(0.5, summary["aspect_recall"])
        self.assertAlmostEqual(2.0 / 3.0, summary["aspect_f1"])
        self.assertEqual(1, summary["exact_matches"])

    def test_strict_exact_is_order_independent_and_requires_evidence(self) -> None:
        review = "The picture is crisp, but the speakers buzz."
        case = _case(
            1,
            review=review,
            expected=[
                {"aspect": "Picture Quality", "sentiment": "positive", "evidence": "picture is crisp"},
                {"aspect": "Sound Quality", "sentiment": "negative", "evidence": "speakers buzz"},
            ],
        )
        reversed_actual = _analysis([
            _mention("Sound Quality", "negative", "speakers buzz"),
            _mention("Picture Quality", "positive", "picture is crisp"),
        ])
        result = evaluate_case(case, reversed_actual)
        self.assertTrue(final_eval.strict_exact_case(result))

        weak = _analysis([
            _mention("Sound Quality", "negative", "speakers"),
            _mention("Picture Quality", "positive", "picture"),
        ])
        self.assertFalse(final_eval.strict_exact_case(evaluate_case(case, weak)))

    def test_error_classification_reuses_established_evaluator(self) -> None:
        case = _case(
            1,
            review="The picture is crisp and the speakers buzz.",
            expected=[
                {"aspect": "Picture Quality", "sentiment": "positive", "evidence": "picture is crisp"},
                {"aspect": "Sound Quality", "sentiment": "negative", "evidence": "speakers buzz"},
            ],
        )
        actual = _analysis([
            _mention("Design", "positive", "picture is crisp"),
            _mention("Sound Quality", "positive", "speakers buzz"),
            _mention("Remote Control", "negative", "speakers"),
        ])
        errors = {item["error_type"] for item in evaluate_case(case, actual)["errors"]}
        self.assertTrue({"WRONG_ASPECT", "WRONG_SENTIMENT", "EXTRA_ASPECT"}.issubset(errors))

    def test_transient_connection_failure_retries_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, logic = _fixture(Path(temporary), [_case(1)])
            analyzer = Mock(
                side_effect=[LLMConnectionError("reset"), _analysis()]
            )
            report = _run(
                dataset,
                manifest,
                logic,
                count=1,
                analyzer=analyzer,
            )
        self.assertEqual(2, analyzer.call_count)
        self.assertEqual(1, report["runtime_statistics"]["retry_count"])
        self.assertEqual("success", report["predictions"][0]["status"])

    def test_quality_or_nonconnection_errors_are_not_retried(self) -> None:
        for error in (
            LLMResponseError("bad schema"),
            LLMClientError("non-transient"),
        ):
            with self.subTest(error=type(error).__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    dataset, manifest, logic = _fixture(Path(temporary), [_case(1)])
                    analyzer = Mock(side_effect=error)
                    report = _run(
                        dataset,
                        manifest,
                        logic,
                        count=1,
                        analyzer=analyzer,
                    )
                self.assertEqual(1, analyzer.call_count)
                self.assertEqual(0, report["runtime_statistics"]["retry_count"])

    def test_runtime_includes_mean_median_p90_and_first_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cases = [_case(index) for index in range(1, 4)]
            dataset, manifest, logic = _fixture(Path(temporary), cases)
            ticks = iter((0.0, 1.0, 1.0, 3.0, 3.0, 8.0))
            report = _run(
                dataset,
                manifest,
                logic,
                count=3,
                analyzer=lambda *_args, **_kwargs: _analysis(),
                clock=lambda: next(ticks),
            )
        runtime = report["runtime_statistics"]
        self.assertEqual(8.0, runtime["total_seconds"])
        self.assertAlmostEqual(8.0 / 3.0, runtime["mean_seconds_per_review"])
        self.assertEqual(2.0, runtime["median_seconds_per_review"])
        self.assertEqual(5.0, runtime["p90_seconds_per_review"])
        self.assertEqual(1.0, runtime["cold_start_first_case_seconds"])

    def test_report_contains_all_error_buckets_and_evaluation_only_notice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, logic = _fixture(Path(temporary), [_case(1)])
            report = _run(
                dataset,
                manifest,
                logic,
                count=1,
                analyzer=lambda *_args, **_kwargs: _analysis(),
            )
        self.assertEqual(set(final_eval.FINAL_ERROR_TYPES), set(report["error_counts"]))
        self.assertEqual(final_eval.EVALUATION_ONLY_NOTICE, report["evaluation_policy"])
        self.assertIn("not pooled", report["development_comparison"]["policy"])


class FinalOutputAndCliTests(unittest.TestCase):
    def test_outputs_use_isolated_paths_and_refuse_overwrite(self) -> None:
        report = {
            "evaluation_policy": final_eval.EVALUATION_ONLY_NOTICE,
            "provider": "ollama",
            "model": "qwen3:8b",
            "dataset_sha256": "a" * 64,
            "analysis_fingerprint": "b" * 64,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "case_count": 1,
            "cases": [],
            "predictions": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "report_path": root / "final_blind_evaluation.json",
                "errors_path": root / "final_blind_errors.csv",
                "predictions_path": root / "final_blind_predictions.json",
            }
            paths = final_eval.save_final_blind_outputs(report, **kwargs)
            self.assertEqual({"report", "errors", "predictions"}, set(paths))
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            payload = json.loads(Path(paths["predictions"]).read_text(encoding="utf-8"))
            self.assertEqual(final_eval.EVALUATION_ONLY_NOTICE, payload["evaluation_policy"])
            with self.assertRaisesRegex(
                final_eval.FinalBlindIntegrityError,
                "will not be overwritten",
            ):
                final_eval.save_final_blind_outputs(report, **kwargs)

    def test_cli_dispatches_final_mode_without_real_provider(self) -> None:
        with patch.object(main_module, "run_final_blind_test", return_value=0) as run:
            self.assertEqual(0, main_module.cli(["--evaluate-final-blind"]))
        run.assert_called_once_with()

    def test_cli_locks_provider_model_and_blocks_second_run(self) -> None:
        with (
            patch.object(main_module, "LLM_PROVIDER", "openai"),
            patch.object(main_module, "OLLAMA_MODEL", "qwen3:8b"),
            patch.object(main_module, "run_final_blind_evaluation") as run,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(1, main_module.run_final_blind_test())
            run.assert_not_called()

        with (
            patch.object(main_module, "LLM_PROVIDER", "ollama"),
            patch.object(main_module, "OLLAMA_MODEL", "qwen3:8b"),
            patch.object(main_module, "final_outputs_exist", return_value=True),
            patch.object(main_module, "run_final_blind_evaluation") as run,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(1, main_module.run_final_blind_test())
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
