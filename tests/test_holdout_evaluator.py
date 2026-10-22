"""Offline tests for the SHA-locked Step 8.7 blind holdout evaluator.

These tests use injected analyzers and clients only.  They must never contact
Ollama, OpenAI, Amazon, or mutate any pre-Step-8.7 evaluation/Demo artifact.
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

import holdout_evaluator  # noqa: E402
import main as main_module  # noqa: E402
from aspect_taxonomy import STANDARD_ASPECTS  # noqa: E402
from llm_client import LLMClientError, LLMConnectionError, LLMResponseError  # noqa: E402


def _mention(
    aspect: str,
    sentiment: str,
    evidence: str,
) -> dict[str, str]:
    return {
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
        review = f"Holdout sentence number {index} has no product opinion."
        expected: list[dict[str, str]] = []
        if first_expected and index == 1:
            review = "The picture is crisp, although the frame looks plain."
            expected = [_mention("Picture Quality", "positive", "picture is crisp")]
        cases.append(
            {
                "case_id": f"H{index:03d}",
                "review": review,
                "expected_aspects": expected,
                "notes": f"Independent annotation note {index}.",
                "tags": ["no_aspect"] if not expected else ["single_aspect"],
            }
        )
    return cases


def _write_fixture(
    directory: Path,
    *,
    count: int,
    cases: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path]:
    dataset = directory / "holdout_cases.json"
    manifest = directory / "holdout_manifest.json"
    development = directory / "development.json"
    dataset.write_text(
        json.dumps(cases or _cases(count), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    holdout_evaluator.create_holdout_manifest(
        dataset,
        manifest,
        expected_count=count,
    )
    development.write_text(
        json.dumps(
            {
                "summary": {
                    "total_cases": 30,
                    "aspect_precision": 0.90,
                    "aspect_recall": 0.80,
                    "aspect_f1": 0.85,
                    "sentiment_accuracy": 0.75,
                    "exact_case_accuracy": 0.70,
                    "exact_matches": 21,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset, manifest, development


def _run(
    dataset: Path,
    manifest: Path,
    development: Path,
    *,
    count: int,
    analyzer: Callable[..., Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    return holdout_evaluator.run_holdout_evaluation(
        dataset_path=dataset,
        manifest_path=manifest,
        development_report_path=development,
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


class HoldoutManifestTests(unittest.TestCase):
    def test_manifest_locks_sha_count_ids_notes_and_contiguous_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, manifest, _development = _write_fixture(
                root,
                count=50,
                cases=_cases(50, first_expected=True),
            )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            recorded, validated = holdout_evaluator.verify_holdout_manifest(
                dataset,
                manifest,
                expected_count=50,
            )

            self.assertEqual("holdout_cases.json", payload["dataset"])
            self.assertEqual(50, payload["case_count"])
            self.assertTrue(payload["created_before_evaluation"])
            self.assertEqual(_sha256(dataset), payload["sha256"])
            self.assertEqual(payload, recorded)
            self.assertEqual(
                [f"H{index:03d}" for index in range(1, 51)],
                [case["case_id"] for case in validated],
            )
            self.assertEqual("picture is crisp", validated[0]["expected_aspects"][0]["evidence"])

    def test_non_contiguous_gold_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = _cases(1, first_expected=True)
            cases[0]["expected_aspects"][0]["evidence"] = "picture looks crisp"
            dataset = root / "holdout_cases.json"
            dataset.write_text(json.dumps(cases), encoding="utf-8")

            with self.assertRaisesRegex(
                holdout_evaluator.HoldoutIntegrityError,
                "contiguous review substring",
            ):
                holdout_evaluator.create_holdout_manifest(
                    dataset,
                    root / "manifest.json",
                    expected_count=1,
                )

    def test_tampering_aborts_before_client_factory_or_analyzer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, development = _write_fixture(
                Path(temporary), count=1
            )
            dataset.write_text(
                dataset.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            client_factory = Mock(return_value=object())
            analyzer = Mock(return_value=_analysis())

            with self.assertRaisesRegex(
                holdout_evaluator.HoldoutIntegrityError,
                "SHA-256",
            ):
                holdout_evaluator.run_holdout_evaluation(
                    dataset_path=dataset,
                    manifest_path=manifest,
                    development_report_path=development,
                    expected_count=1,
                    client_factory=client_factory,
                    analyzer=analyzer,
                    progress=lambda *_args: None,
                )

            client_factory.assert_not_called()
            analyzer.assert_not_called()


class HoldoutExecutionTests(unittest.TestCase):
    def test_connection_failure_retries_once_then_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, development = _write_fixture(
                Path(temporary), count=2
            )
            calls: list[str] = []
            sleeps: list[float] = []

            def analyzer(review: str, **_kwargs: Any) -> Mapping[str, Any]:
                calls.append(review)
                if len(calls) == 1:
                    raise LLMConnectionError("temporary connection reset")
                return _analysis()

            report = holdout_evaluator.run_holdout_evaluation(
                dataset_path=dataset,
                manifest_path=manifest,
                development_report_path=development,
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
            self.assertTrue(report["predictions"][0]["retried"])
            self.assertEqual(1, report["runtime_statistics"]["retry_count"])

    def test_two_connection_failures_record_failed_and_continue_serially(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, development = _write_fixture(
                Path(temporary), count=2
            )
            first_review = _cases(2)[0]["review"]
            calls: list[str] = []
            progress: list[tuple[int, int, str, str]] = []

            def analyzer(review: str, **_kwargs: Any) -> Mapping[str, Any]:
                calls.append(review)
                if review == first_review:
                    raise LLMConnectionError("provider temporarily unreachable")
                return _analysis()

            report = holdout_evaluator.run_holdout_evaluation(
                dataset_path=dataset,
                manifest_path=manifest,
                development_report_path=development,
                expected_count=2,
                client=object(),
                analyzer=analyzer,
                sleeper=lambda _seconds: None,
                progress=lambda current, total, case_id, status, _elapsed: progress.append(
                    (current, total, case_id, status)
                ),
            )

            self.assertEqual([first_review, first_review, _cases(2)[1]["review"]], calls)
            self.assertEqual(["failed", "success"], [item["status"] for item in report["predictions"]])
            self.assertEqual(1, report["summary"]["failed_cases"])
            self.assertEqual(1, report["runtime_statistics"]["model_failure_count"])
            self.assertEqual(
                [(1, 2, "H001", "failed"), (2, 2, "H002", "success")],
                progress,
            )

    def test_schema_error_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, development = _write_fixture(
                Path(temporary), count=2
            )
            calls: list[str] = []

            def analyzer(review: str, **_kwargs: Any) -> Mapping[str, Any]:
                calls.append(review)
                if review == _cases(2)[0]["review"]:
                    raise LLMResponseError("invalid structured response")
                return _analysis()

            report = _run(
                dataset,
                manifest,
                development,
                count=2,
                analyzer=analyzer,
            )

            self.assertEqual(2, len(calls))
            self.assertEqual(1, report["predictions"][0]["attempts"])
            self.assertEqual("SCHEMA_ERROR", report["predictions"][0]["error_type"])
            self.assertEqual(0, report["runtime_statistics"]["retry_count"])
            self.assertEqual(1, report["runtime_statistics"]["schema_failure_count"])

    def test_non_connection_provider_error_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, development = _write_fixture(
                Path(temporary), count=2
            )
            calls: list[str] = []

            def analyzer(review: str, **_kwargs: Any) -> Mapping[str, Any]:
                calls.append(review)
                if review == _cases(2)[0]["review"]:
                    raise LLMClientError("non-transient provider failure")
                return _analysis()

            report = _run(
                dataset,
                manifest,
                development,
                count=2,
                analyzer=analyzer,
            )

            self.assertEqual(2, len(calls))
            self.assertEqual(1, report["predictions"][0]["attempts"])
            self.assertEqual("MODEL_ERROR", report["predictions"][0]["error_type"])
            self.assertEqual(0, report["runtime_statistics"]["retry_count"])

    def test_exact_case_requires_matching_evidence_not_only_pair(self) -> None:
        cases = _cases(1, first_expected=True)
        cases[0]["review"] = "The picture is crisp, while the display dazzles indoors."
        cases[0]["expected_aspects"][0]["evidence"] = "picture is crisp"
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, development = _write_fixture(
                Path(temporary), count=1, cases=cases
            )

            report = _run(
                dataset,
                manifest,
                development,
                count=1,
                analyzer=lambda *_args, **_kwargs: _analysis(
                    [_mention("Picture Quality", "positive", "display dazzles indoors")]
                ),
            )

            case = report["cases"][0]
            self.assertTrue(case["base_exact_match"])
            self.assertFalse(case["exact_match"])
            self.assertEqual(0, report["summary"]["exact_matches"])
            self.assertIn(
                "WEAK_EVIDENCE_MATCH",
                [error["error_type"] for error in case["errors"]],
            )

    def test_runtime_p90_slowest_five_and_complete_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, development = _write_fixture(
                Path(temporary), count=6
            )
            clock_values = iter(
                [
                    0.0,
                    1.0,
                    10.0,
                    12.0,
                    20.0,
                    23.0,
                    30.0,
                    34.0,
                    40.0,
                    45.0,
                    50.0,
                    56.0,
                ]
            )
            report = _run(
                dataset,
                manifest,
                development,
                count=6,
                analyzer=lambda *_args, **_kwargs: _analysis(),
                clock=lambda: next(clock_values),
            )

            runtime = report["runtime_statistics"]
            self.assertEqual(21.0, runtime["total_seconds"])
            self.assertEqual(3.5, runtime["average_seconds_per_review"])
            self.assertEqual(3.5, runtime["median_seconds_per_review"])
            self.assertEqual(6.0, runtime["p90_seconds_per_review"])
            self.assertEqual(
                ["H006", "H005", "H004", "H003", "H002"],
                [item["case_id"] for item in runtime["slowest_five"]],
            )
            self.assertEqual(6, len(report["predictions"]))
            self.assertTrue(all(item["analysis"] == _analysis() for item in report["predictions"]))

    def test_generalization_gap_is_holdout_minus_development(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, development = _write_fixture(
                Path(temporary), count=1
            )
            report = _run(
                dataset,
                manifest,
                development,
                count=1,
                analyzer=lambda *_args, **_kwargs: _analysis(),
            )

            gap = report["generalization_gap"]["exact_case_accuracy"]
            self.assertEqual(0.70, gap["development"])
            self.assertEqual(1.0, gap["holdout"])
            self.assertAlmostEqual(0.30, gap["difference"])
            self.assertAlmostEqual(30.0, gap["difference_percentage_points"])

    def test_all_fifty_predictions_keep_complete_structured_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest, development = _write_fixture(
                Path(temporary), count=50
            )
            fake_analysis = {
                "summary": "Complete fake result.",
                "aspects": [],
                "positive_points": ["kept for compatibility"],
                "negative_points": [],
                "diagnostic_extension": {"source": "offline-test"},
            }
            report = _run(
                dataset,
                manifest,
                development,
                count=50,
                analyzer=lambda *_args, **_kwargs: fake_analysis,
            )

            self.assertEqual(50, report["case_count"])
            self.assertEqual(
                [f"H{index:03d}" for index in range(1, 51)],
                [item["case_id"] for item in report["predictions"]],
            )
            self.assertTrue(
                all(item["analysis"] == fake_analysis for item in report["predictions"])
            )


class HoldoutOutputTests(unittest.TestCase):
    def test_three_dedicated_outputs_and_required_csv_fields(self) -> None:
        report = {
            "provider": "ollama",
            "model": "qwen3:8b",
            "dataset_sha256": "a" * 64,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "case_count": 1,
            "predictions": [
                {
                    "case_id": "H001",
                    "review": "The sound is harsh.",
                    "analysis": _analysis(
                        [_mention("Sound Quality", "negative", "sound is harsh")]
                    ),
                }
            ],
            "cases": [
                {
                    "case_id": "H001",
                    "review": "The sound is harsh.",
                    "expected_aspects": [
                        _mention("Sound Quality", "positive", "sound is harsh")
                    ],
                    "actual_aspects": [
                        _mention("Sound Quality", "negative", "sound is harsh")
                    ],
                    "notes": "Deliberate fixture.",
                    "errors": [
                        {
                            "error_type": "WRONG_SENTIMENT",
                            "message": "Wrong polarity.",
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = holdout_evaluator.save_holdout_outputs(
                report,
                report_path=root / "holdout_evaluation_qwen3_8b.json",
                errors_path=root / "holdout_errors.csv",
                predictions_path=root / "holdout_predictions.json",
            )

            self.assertEqual(
                {"report", "errors", "predictions"}, set(paths)
            )
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            with Path(paths["errors"]).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(list(holdout_evaluator.ERROR_CSV_FIELDS), list(rows[0]))
                self.assertEqual("H001", rows[0]["case_id"])
                self.assertEqual("WRONG_SENTIMENT", rows[0]["error_type"])
            predictions = json.loads(Path(paths["predictions"]).read_text(encoding="utf-8"))
            self.assertEqual(report["predictions"], predictions["predictions"])
            self.assertEqual("Offline fake analysis.", predictions["predictions"][0]["analysis"]["summary"])

    def test_saving_holdout_outputs_does_not_write_legacy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = {
                root / "evaluation_results.json": b"legacy result\n",
                root / "evaluation_errors.csv": b"legacy,error\n",
                root / "gold_cases.json": b"[]\n",
                root / "analyzed_reviews.csv": b"demo frozen\n",
            }
            for path, content in legacy.items():
                path.write_bytes(content)
            before = {path: _sha256(path) for path in legacy}

            holdout_evaluator.save_holdout_outputs(
                {"predictions": [], "cases": []},
                report_path=root / "holdout" / "report.json",
                errors_path=root / "holdout" / "errors.csv",
                predictions_path=root / "holdout" / "predictions.json",
            )

            self.assertEqual(before, {path: _sha256(path) for path in legacy})

    def test_top_ten_uses_documented_business_priority(self) -> None:
        order = [
            "INVALID_EVIDENCE",
            "WRONG_SENTIMENT",
            "EXTRA_ASPECT",
            "MISS_ASPECT",
            "WRONG_ASPECT",
        ]
        cases = [
            {
                "case_id": f"H{index:03d}",
                "review": f"Review {index}",
                "expected_aspects": [],
                "actual_aspects": [],
                "errors": [{"error_type": error_type}],
            }
            for index, error_type in enumerate(order, start=1)
        ]

        selected = holdout_evaluator.select_top_errors(cases, limit=10)

        self.assertEqual(
            [
                "WRONG_ASPECT",
                "MISS_ASPECT",
                "EXTRA_ASPECT",
                "WRONG_SENTIMENT",
                "INVALID_EVIDENCE",
            ],
            [item["error_type"] for item in selected],
        )


class HoldoutFreezeAndCliTests(unittest.TestCase):
    def test_real_holdout_has_required_coverage_and_is_distinct_from_gold(self) -> None:
        dataset = PROJECT_ROOT / "evaluation" / "holdout_cases.json"
        gold_path = PROJECT_ROOT / "evaluation" / "gold_cases.json"
        if not dataset.is_file():
            self.skipTest("Holdout dataset is assembled by the coordinating task.")
        cases = json.loads(dataset.read_text(encoding="utf-8"))
        gold_cases = json.loads(gold_path.read_text(encoding="utf-8"))

        mentions = [
            mention
            for case in cases
            for mention in case["expected_aspects"]
        ]
        self.assertEqual(set(STANDARD_ASPECTS), {item["aspect"] for item in mentions})
        self.assertEqual(
            {"positive", "negative", "neutral"},
            {item["sentiment"] for item in mentions},
        )
        cardinalities = [len(case["expected_aspects"]) for case in cases]
        self.assertTrue(any(count == 0 for count in cardinalities))
        self.assertTrue(any(count == 1 for count in cardinalities))
        self.assertTrue(any(count == 2 for count in cardinalities))
        self.assertTrue(any(3 <= count <= 5 for count in cardinalities))
        self.assertTrue(any(
            {"positive", "negative"}.issubset(
                {item["sentiment"] for item in case["expected_aspects"]}
            )
            for case in cases
        ))
        all_tags = {tag for case in cases for tag in case.get("tags", [])}
        self.assertTrue(
            {
                "same_aspect_multiple_mentions",
                "same_sentence_multiple_aspects",
                "negation",
                "contrast",
                "weak_sentiment",
                "comparison",
                "price_boundary",
                "packaging_reliability_boundary",
                "remote_sound_boundary",
                "system_ease_boundary",
                "connectivity",
                "product_vs_shopping_experience",
                "no_explicit_aspect",
            }.issubset(all_tags)
        )

        def normalize(value: str) -> str:
            return " ".join(re.sub(r"[^a-z0-9 ]", " ", value.casefold()).split())

        maximum_similarity = max(
            difflib.SequenceMatcher(
                None,
                normalize(holdout["review"]),
                normalize(gold["review"]),
            ).ratio()
            for holdout in cases
            for gold in gold_cases
        )
        self.assertLess(maximum_similarity, 0.75)

    def test_frozen_runtime_contains_no_holdout_ids_or_complete_reviews(self) -> None:
        dataset = PROJECT_ROOT / "evaluation" / "holdout_cases.json"
        if not dataset.is_file():
            self.skipTest("Holdout dataset is assembled by the coordinating task.")
        cases = json.loads(dataset.read_text(encoding="utf-8"))
        runtime_paths = [
            PROJECT_ROOT / "llm_client.py",
            PROJECT_ROOT / "aspect_taxonomy.py",
            PROJECT_ROOT / "review_analyzer.py",
        ]
        runtime_text = "\n".join(path.read_text(encoding="utf-8") for path in runtime_paths)
        for case in cases:
            self.assertNotIn(case["case_id"], runtime_text)
            self.assertNotIn(case["review"], runtime_text)

    def test_real_holdout_manifest_is_valid_when_dataset_is_present(self) -> None:
        dataset = PROJECT_ROOT / "evaluation" / "holdout_cases.json"
        manifest = PROJECT_ROOT / "evaluation" / "holdout_manifest.json"
        if not dataset.is_file() or not manifest.is_file():
            self.skipTest("Holdout artifacts are assembled by the coordinating task.")
        frozen, cases = holdout_evaluator.verify_holdout_manifest(
            dataset,
            manifest,
            expected_count=50,
        )
        self.assertEqual(50, len(cases))
        self.assertEqual(_sha256(dataset), frozen["sha256"])

    def test_cli_dispatches_evaluate_holdout_mode(self) -> None:
        with patch.object(
            main_module,
            "run_blind_holdout_evaluation",
            return_value=0,
        ) as run:
            result = main_module.cli(["--evaluate-holdout"])

        self.assertEqual(0, result)
        run.assert_called_once_with()

    def test_cli_runner_uses_current_provider_and_model_and_saves_outputs(self) -> None:
        report = {
            "summary": {},
            "development_set": {"summary": {}},
            "runtime_statistics": {},
            "generalization_gap": {},
            "top_errors": [],
        }
        output_paths = {
            "report": "/tmp/report.json",
            "errors": "/tmp/errors.csv",
            "predictions": "/tmp/predictions.json",
        }
        stream = StringIO()
        with (
            patch.object(main_module, "LLM_PROVIDER", "ollama"),
            patch.object(main_module, "OLLAMA_MODEL", "qwen3:8b"),
            patch.object(
                main_module,
                "run_holdout_evaluation",
                return_value=report,
            ) as run,
            patch.object(
                main_module,
                "save_holdout_outputs",
                return_value=output_paths,
            ) as save,
            patch.object(main_module, "format_holdout_report", return_value="report"),
            redirect_stdout(stream),
        ):
            result = main_module.run_blind_holdout_evaluation()

        self.assertEqual(0, result)
        run.assert_called_once_with(provider="ollama", model="qwen3:8b")
        save.assert_called_once_with(report)
        self.assertIn("Blind Holdout Evaluation", stream.getvalue())
        self.assertIn("/tmp/predictions.json", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
