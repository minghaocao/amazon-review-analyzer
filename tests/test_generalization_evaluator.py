"""Offline guards for the Step 8.8 generalization development evaluator.

No test in this module may contact Ollama, OpenAI, Amazon, or write to the
frozen Demo outputs.  Model-facing paths always receive an injected analyzer
and client.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import re
import sys
import tempfile
import unittest
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generalization_evaluator  # noqa: E402
import main as main_module  # noqa: E402
from aspect_taxonomy import STANDARD_ASPECTS  # noqa: E402
from config import DEMO_OUTPUT_PATHS  # noqa: E402
from llm_client import LLMClientError, LLMConnectionError, LLMResponseError  # noqa: E402


def _mention(
    aspect: str,
    sentiment: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "raw_aspect": aspect,
        "aspect": aspect,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _analysis(aspects: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "summary": "Offline fake analysis.",
        "aspects": aspects or [],
        "positive_points": [],
        "negative_points": [],
    }


def _cases(count: int, *, first_expected: bool = False) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        review = f"Development sentence number {index} states no product opinion."
        expected: list[dict[str, str]] = []
        if first_expected and index == 1:
            review = "The image is crisp, while the stand looks ordinary."
            expected = [
                {
                    "aspect": "Picture Quality",
                    "sentiment": "positive",
                    "evidence": "image is crisp",
                }
            ]
        cases.append(
            {
                "case_id": f"G{index:03d}",
                "review": review,
                "expected_aspects": expected,
                "notes": f"Independent development annotation {index}.",
                "tags": ["no_aspect"] if not expected else ["single_aspect"],
            }
        )
    return cases


def _write_fixture(
    directory: Path,
    *,
    count: int,
    cases: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    dataset = directory / "generalization_dev_cases.json"
    manifest = directory / "generalization_dev_manifest.json"
    dataset.write_text(
        json.dumps(cases or _cases(count), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generalization_evaluator.create_generalization_manifest(
        dataset,
        manifest,
        expected_count=count,
    )
    return dataset, manifest


def _run(
    dataset: Path,
    manifest: Path,
    *,
    count: int,
    analyzer: Callable[..., Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    return generalization_evaluator.run_generalization_evaluation(
        dataset_path=dataset,
        manifest_path=manifest,
        expected_count=count,
        provider="ollama",
        model="qwen3:8b",
        client=object(),
        analyzer=analyzer,
        sleeper=lambda _seconds: None,
        progress=lambda *_args: None,
        **kwargs,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_text(value: str) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9 ]", " ", value.casefold()).split()
    )


class GeneralizationIntegrityTests(unittest.TestCase):
    def test_manifest_records_hash_and_validates_schema_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _write_fixture(
                Path(temporary), count=2, cases=_cases(2, first_expected=True)
            )

            recorded, cases = generalization_evaluator.verify_generalization_manifest(
                dataset,
                manifest,
                expected_count=2,
            )

            self.assertEqual("generalization_dev_cases.json", recorded["dataset"])
            self.assertEqual(2, recorded["case_count"])
            self.assertTrue(recorded["created_before_baseline"])
            self.assertEqual(_sha256(dataset), recorded["sha256"])
            self.assertEqual(["G001", "G002"], [case["case_id"] for case in cases])
            self.assertEqual(
                "image is crisp",
                cases[0]["expected_aspects"][0]["evidence"],
            )

    def test_non_contiguous_gold_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = _cases(1, first_expected=True)
            cases[0]["expected_aspects"][0]["evidence"] = "image looks crisp"
            dataset = root / "generalization_dev_cases.json"
            dataset.write_text(json.dumps(cases), encoding="utf-8")

            with self.assertRaisesRegex(
                generalization_evaluator.GeneralizationIntegrityError,
                "contiguous review substring",
            ):
                generalization_evaluator.create_generalization_manifest(
                    dataset,
                    root / "manifest.json",
                    expected_count=1,
                )

    def test_tampering_aborts_before_client_creation_or_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _write_fixture(Path(temporary), count=1)
            dataset.write_text(
                dataset.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            client_factory = Mock(return_value=object())
            analyzer = Mock(return_value=_analysis())

            with self.assertRaisesRegex(
                generalization_evaluator.GeneralizationIntegrityError,
                "SHA-256",
            ):
                generalization_evaluator.run_generalization_evaluation(
                    dataset_path=dataset,
                    manifest_path=manifest,
                    expected_count=1,
                    client_factory=client_factory,
                    analyzer=analyzer,
                    progress=lambda *_args: None,
                )

            client_factory.assert_not_called()
            analyzer.assert_not_called()

    def test_manifest_is_one_time_and_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, manifest = _write_fixture(root, count=1)

            with self.assertRaisesRegex(
                generalization_evaluator.GeneralizationIntegrityError,
                "will not be overwritten",
            ):
                generalization_evaluator.create_generalization_manifest(
                    dataset,
                    manifest,
                    expected_count=1,
                )


class GeneralizationExecutionTests(unittest.TestCase):
    def test_transient_connection_failure_retries_once_then_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _write_fixture(Path(temporary), count=2)
            calls: list[str] = []
            sleeps: list[float] = []

            def analyzer(review: str, **_kwargs: Any) -> Mapping[str, Any]:
                calls.append(review)
                if len(calls) == 1:
                    raise LLMConnectionError("temporary connection reset")
                return _analysis()

            report = generalization_evaluator.run_generalization_evaluation(
                dataset_path=dataset,
                manifest_path=manifest,
                expected_count=2,
                client=object(),
                analyzer=analyzer,
                sleeper=sleeps.append,
                retry_delay_seconds=0.25,
                progress=lambda *_args: None,
            )

            self.assertEqual(3, len(calls))
            self.assertEqual([0.25], sleeps)
            self.assertEqual("success", report["predictions"][0]["status"])
            self.assertEqual(2, report["predictions"][0]["attempts"])
            self.assertEqual(1, report["runtime_statistics"]["retry_count"])

    def test_two_connection_failures_record_failure_and_continue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _write_fixture(Path(temporary), count=2)
            fixture_cases = _cases(2)
            calls: list[str] = []

            def analyzer(review: str, **_kwargs: Any) -> Mapping[str, Any]:
                calls.append(review)
                if review == fixture_cases[0]["review"]:
                    raise LLMConnectionError("provider temporarily unreachable")
                return _analysis()

            report = _run(
                dataset,
                manifest,
                count=2,
                analyzer=analyzer,
            )

            self.assertEqual(
                [
                    fixture_cases[0]["review"],
                    fixture_cases[0]["review"],
                    fixture_cases[1]["review"],
                ],
                calls,
            )
            self.assertEqual(
                ["failed", "success"],
                [item["status"] for item in report["predictions"]],
            )
            self.assertEqual(1, report["summary"]["failed_cases"])
            self.assertEqual(1, report["runtime_statistics"]["model_failure_count"])

    def test_schema_and_non_connection_errors_are_not_retried(self) -> None:
        error_types = (
            (LLMResponseError("bad structure"), "SCHEMA_ERROR"),
            (LLMClientError("non-transient provider error"), "MODEL_ERROR"),
        )
        for error, expected_type in error_types:
            with self.subTest(expected_type=expected_type):
                with tempfile.TemporaryDirectory() as temporary:
                    dataset, manifest = _write_fixture(Path(temporary), count=2)
                    first_review = _cases(2)[0]["review"]
                    calls: list[str] = []

                    def analyzer(review: str, **_kwargs: Any) -> Mapping[str, Any]:
                        calls.append(review)
                        if review == first_review:
                            raise error
                        return _analysis()

                    report = _run(
                        dataset,
                        manifest,
                        count=2,
                        analyzer=analyzer,
                    )

                    self.assertEqual(2, len(calls))
                    self.assertEqual(1, report["predictions"][0]["attempts"])
                    self.assertEqual(
                        expected_type,
                        report["predictions"][0]["error_type"],
                    )
                    self.assertEqual(0, report["runtime_statistics"]["retry_count"])

    def test_strict_exact_requires_matching_evidence(self) -> None:
        cases = _cases(1, first_expected=True)
        cases[0]["review"] = "The image is crisp, while colors look vivid indoors."
        cases[0]["expected_aspects"][0]["evidence"] = "image is crisp"
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _write_fixture(
                Path(temporary), count=1, cases=cases
            )
            report = _run(
                dataset,
                manifest,
                count=1,
                analyzer=lambda *_args, **_kwargs: _analysis(
                    [_mention("Picture Quality", "positive", "colors look vivid indoors")]
                ),
            )

            result = report["cases"][0]
            self.assertTrue(result["base_exact_match"])
            self.assertFalse(result["exact_match"])
            self.assertEqual(0, report["summary"]["exact_matches"])
            self.assertIn(
                "WEAK_EVIDENCE_MATCH",
                {error["error_type"] for error in result["errors"]},
            )

    def test_prediction_keeps_production_analysis_schema(self) -> None:
        cases = _cases(1, first_expected=True)
        fake = _analysis(
            [_mention("Picture Quality", "positive", "image is crisp")]
        )
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _write_fixture(
                Path(temporary), count=1, cases=cases
            )
            report = _run(
                dataset,
                manifest,
                count=1,
                analyzer=lambda *_args, **_kwargs: fake,
            )

            saved = report["predictions"][0]["analysis"]
            self.assertEqual(
                {"summary", "aspects", "positive_points", "negative_points"},
                set(saved),
            )
            self.assertEqual(
                {"raw_aspect", "aspect", "sentiment", "evidence"},
                set(saved["aspects"][0]),
            )


class GeneralizationOutputTests(unittest.TestCase):
    @staticmethod
    def _report(dataset_hash: str, *, f1: float = 0.50) -> dict[str, Any]:
        return {
            "dataset_sha256": dataset_hash,
            "summary": {
                "aspect_precision": f1,
                "aspect_recall": f1,
                "aspect_f1": f1,
                "aspect_only_precision": f1,
                "aspect_only_recall": f1,
                "aspect_only_f1": f1,
                "sentiment_accuracy": f1,
                "evidence_validity_rate": 1.0,
                "evidence_match_rate": 1.0,
                "exact_case_accuracy": f1,
            },
            "error_counts": {"EXTRA_ASPECT": 2},
            "cases": [],
            "predictions": [],
        }

    def test_baseline_outputs_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "baseline_path": root / "baseline.json",
                "baseline_errors_path": root / "baseline_errors.csv",
                "results_path": root / "optimized.json",
                "errors_path": root / "optimized_errors.csv",
            }
            report = self._report("a" * 64)
            generalization_evaluator.save_generalization_evaluation(
                report,
                stage="baseline",
                **kwargs,
            )
            original = (root / "baseline.json").read_bytes()

            with self.assertRaisesRegex(
                generalization_evaluator.GeneralizationIntegrityError,
                "will not be overwritten",
            ):
                generalization_evaluator.save_generalization_evaluation(
                    report,
                    stage="baseline",
                    **kwargs,
                )

            self.assertEqual(original, (root / "baseline.json").read_bytes())

    def test_optimized_requires_matching_baseline_and_records_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "baseline_path": root / "baseline.json",
                "baseline_errors_path": root / "baseline_errors.csv",
                "results_path": root / "optimized.json",
                "errors_path": root / "optimized_errors.csv",
            }
            baseline = self._report("a" * 64, f1=0.50)
            optimized = self._report("a" * 64, f1=0.75)

            with self.assertRaisesRegex(
                generalization_evaluator.GeneralizationIntegrityError,
                "baseline before optimization",
            ):
                generalization_evaluator.save_generalization_evaluation(
                    optimized,
                    stage="optimized",
                    **kwargs,
                )

            generalization_evaluator.save_generalization_evaluation(
                baseline,
                stage="baseline",
                **kwargs,
            )
            mismatched = self._report("b" * 64, f1=0.75)
            with self.assertRaisesRegex(
                generalization_evaluator.GeneralizationIntegrityError,
                "does not match",
            ):
                generalization_evaluator.save_generalization_evaluation(
                    mismatched,
                    stage="optimized",
                    **kwargs,
                )

            paths = generalization_evaluator.save_generalization_evaluation(
                optimized,
                stage="optimized",
                **kwargs,
            )
            payload = json.loads(Path(paths["results"]).read_text(encoding="utf-8"))
            comparison = payload["baseline_comparison"]["metrics"]["aspect_f1"]
            self.assertEqual(0.50, comparison["baseline"])
            self.assertEqual(0.75, comparison["optimized"])
            self.assertEqual(0.25, comparison["difference"])
            with Path(paths["errors"]).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(
                    list(generalization_evaluator.ERROR_CSV_FIELDS),
                    list(next(csv.reader(handle))),
                )

    def test_offline_run_and_save_do_not_modify_demo_outputs(self) -> None:
        before = {
            path: _sha256(path)
            for path in DEMO_OUTPUT_PATHS
            if path.is_file()
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, manifest = _write_fixture(root, count=1)
            report = _run(
                dataset,
                manifest,
                count=1,
                analyzer=lambda *_args, **_kwargs: _analysis(),
            )
            generalization_evaluator.save_generalization_evaluation(
                report,
                stage="baseline",
                baseline_path=root / "baseline.json",
                baseline_errors_path=root / "baseline_errors.csv",
                results_path=root / "optimized.json",
                errors_path=root / "optimized_errors.csv",
            )

        self.assertEqual(
            before,
            {path: _sha256(path) for path in before},
        )


class GeneralizationCliTests(unittest.TestCase):
    def test_cli_dispatches_default_and_explicit_stages(self) -> None:
        with patch.object(
            main_module,
            "run_generalization_dev_evaluation",
            return_value=0,
        ) as run:
            self.assertEqual(0, main_module.cli(["--evaluate-generalization-dev"]))
            self.assertEqual(
                0,
                main_module.cli(
                    [
                        "--evaluate-generalization-dev",
                        "--generalization-stage",
                        "baseline",
                    ]
                ),
            )

        self.assertEqual(
            [unittest.mock.call("optimized"), unittest.mock.call("baseline")],
            run.call_args_list,
        )

    def test_stage_flag_is_rejected_outside_generalization_mode(self) -> None:
        stream = StringIO()
        with redirect_stdout(stream):
            result = main_module.cli(["--generalization-stage", "baseline"])

        self.assertEqual(2, result)
        self.assertIn("only be changed", stream.getvalue())

    def test_cli_runner_is_locked_to_local_qwen3_8b(self) -> None:
        with (
            patch.object(main_module, "LLM_PROVIDER", "openai"),
            patch.object(main_module, "OLLAMA_MODEL", "qwen3:8b"),
            patch.object(main_module, "run_generalization_evaluation") as run,
        ):
            self.assertEqual(1, main_module.run_generalization_dev_evaluation())
            run.assert_not_called()

        report = {"summary": {}, "runtime_statistics": {}, "error_counts": {}}
        paths = {"results": "/tmp/result.json", "errors": "/tmp/errors.csv"}
        with (
            patch.object(main_module, "LLM_PROVIDER", "ollama"),
            patch.object(main_module, "OLLAMA_MODEL", "qwen3:8b"),
            patch.object(
                main_module,
                "run_generalization_evaluation",
                return_value=report,
            ) as run,
            patch.object(
                main_module,
                "save_generalization_evaluation",
                return_value=paths,
            ) as save,
            patch.object(
                main_module,
                "format_generalization_report",
                return_value="offline report",
            ),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(
                0,
                main_module.run_generalization_dev_evaluation("baseline"),
            )

        run.assert_called_once_with(provider="ollama", model="qwen3:8b")
        save.assert_called_once_with(report, stage="baseline")


class GeneralizationDatasetGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = PROJECT_ROOT / "evaluation" / "generalization_dev_cases.json"
        if not cls.dataset.is_file():
            raise unittest.SkipTest(
                "Generalization dataset is assembled by the coordinating task."
            )
        cls.cases = json.loads(cls.dataset.read_text(encoding="utf-8"))

    def test_real_dataset_has_required_ids_schema_and_coverage(self) -> None:
        validated = generalization_evaluator.validate_generalization_cases(
            self.dataset,
            expected_count=80,
        )
        self.assertEqual(
            [f"G{index:03d}" for index in range(1, 81)],
            [case["case_id"] for case in validated],
        )
        mentions = [
            mention
            for case in validated
            for mention in case["expected_aspects"]
        ]
        self.assertEqual(set(STANDARD_ASPECTS), {item["aspect"] for item in mentions})
        self.assertEqual(
            {"positive", "negative", "neutral"},
            {item["sentiment"] for item in mentions},
        )
        cardinalities = {len(case["expected_aspects"]) for case in validated}
        self.assertTrue({0, 1, 2}.issubset(cardinalities))
        self.assertTrue(any(3 <= count <= 5 for count in cardinalities))
        self.assertTrue(any(
            len([item["aspect"] for item in case["expected_aspects"]])
            != len({item["aspect"] for item in case["expected_aspects"]})
            for case in validated
        ))
        self.assertTrue(any(
            {"positive", "negative"}.issubset(
                {item["sentiment"] for item in case["expected_aspects"]}
            )
            for case in validated
        ))
        for case in validated:
            self.assertTrue(case["notes"])
            self.assertTrue(case["tags"])
            for mention in case["expected_aspects"]:
                evidence = mention["evidence"]
                values = evidence if isinstance(evidence, list) else [evidence]
                self.assertTrue(all(value in case["review"] for value in values))

    def test_real_dataset_covers_required_hard_case_tag_families(self) -> None:
        normalized_tags = {
            tag.casefold().replace("-", "_").replace("/", "_")
            for case in self.cases
            for tag in case.get("tags", [])
        }
        required_families = {
            "packaging_build": {
                "packaging_vs_build_quality",
                "packaging_build_boundary",
            },
            "delivery_packaging": {
                "delivery_vs_packaging",
                "delivery_packaging_boundary",
            },
            "system_picture": {
                "system_performance_vs_picture_quality",
                "system_picture_boundary",
            },
            "system_ease": {
                "ease_of_use_vs_system_performance",
                "system_ease_boundary",
            },
            "remote_sound": {
                "remote_control_vs_sound_quality",
                "remote_sound_boundary",
            },
            "reliability_delivery": {
                "reliability_vs_delivery",
                "reliability_delivery_boundary",
            },
            "multi_mention": {"multi_mention", "same_aspect_multiple_mentions"},
            "overprediction": {
                "extra_aspect_prevention",
                "no_aspect",
            },
            "price_value": {"price_value", "price_fact"},
            "sentiment_hard": {
                "sentiment_hard",
                "sentiment_boundary",
                "weak_sentiment",
            },
        }
        for family, aliases in required_families.items():
            with self.subTest(family=family):
                self.assertTrue(aliases & normalized_tags)

    def test_no_exact_or_obvious_paraphrase_of_old_gold_holdout_or_demo(self) -> None:
        old_reviews: list[str] = []
        for name in ("gold_cases.json", "holdout_cases.json"):
            path = PROJECT_ROOT / "evaluation" / name
            if path.is_file():
                old_reviews.extend(
                    case["review"]
                    for case in json.loads(path.read_text(encoding="utf-8"))
                )
        for path in (
            PROJECT_ROOT / "data" / "raw_reviews.csv",
            PROJECT_ROOT / "outputs" / "analyzed_reviews.csv",
        ):
            if not path.is_file():
                continue
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    review = row.get("content") or row.get("original_content") or ""
                    if review.strip():
                        old_reviews.append(review)

        new_normalized = [_normalized_text(case["review"]) for case in self.cases]
        old_normalized = [_normalized_text(review) for review in old_reviews]
        self.assertEqual(len(new_normalized), len(set(new_normalized)))
        self.assertTrue(set(new_normalized).isdisjoint(old_normalized))
        maximum_similarity = max(
            difflib.SequenceMatcher(None, new, old).ratio()
            for new in new_normalized
            for old in old_normalized
        )
        self.assertLess(
            maximum_similarity,
            0.75,
            "A Step 8.8 review is too similar to an earlier evaluation/Demo review.",
        )

    def test_production_runtime_contains_no_evaluation_case_ids_or_reviews(self) -> None:
        runtime_paths = (
            PROJECT_ROOT / "llm_client.py",
            PROJECT_ROOT / "aspect_taxonomy.py",
            PROJECT_ROOT / "review_analyzer.py",
        )
        runtime_text = "\n".join(
            path.read_text(encoding="utf-8") for path in runtime_paths
        )
        evaluation_cases: list[dict[str, Any]] = []
        for name in (
            "gold_cases.json",
            "holdout_cases.json",
            "generalization_dev_cases.json",
        ):
            path = PROJECT_ROOT / "evaluation" / name
            if path.is_file():
                evaluation_cases.extend(json.loads(path.read_text(encoding="utf-8")))

        for case in evaluation_cases:
            self.assertNotIn(case["case_id"], runtime_text)
            self.assertNotIn(case["review"], runtime_text)


if __name__ == "__main__":
    unittest.main()
