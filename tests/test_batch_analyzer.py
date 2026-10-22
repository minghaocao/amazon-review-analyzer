"""Offline tests for the Step 5A serial batch analyzer."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch_analyzer import (  # noqa: E402
    CheckpointInvalidError,
    CheckpointMismatchError,
    add_source_row_ids,
    analyze_reviews,
    build_unique_aspects,
    calculate_input_signature,
    load_analysis_checkpoint,
    run_checkpointed_batch,
    save_analysis_checkpoint,
    save_analyzed_reviews,
)
from llm_client import LLMConnectionError  # noqa: E402
from review_analyzer import ReviewAnalysisError  # noqa: E402


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    defaults: dict[str, Any] = {
        "review_id": "R000",
        "rating": 5,
        "title": "Title",
        "content": "Original review.",
        "date": "2026-09-01",
        "verified_purchase": "true",
        "original_content": "Original review.",
        "cleaned_content": "Original review.",
        "is_analyzable": True,
        "skip_reason": "",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _analysis(
    summary: str = "Summary.",
    *,
    aspect: str = "Picture Quality",
    sentiment: str = "positive",
    evidence: str = "picture is excellent",
) -> dict[str, Any]:
    aspect_item = {
        "raw_aspect": "picture",
        "aspect": aspect,
        "sentiment": sentiment,
        "evidence": evidence,
    }
    return {
        "summary": summary,
        "aspects": [aspect_item],
        "positive_points": [evidence] if sentiment == "positive" else [],
        "negative_points": [evidence] if sentiment == "negative" else [],
    }


def _repeated_aspect_analysis() -> dict[str, Any]:
    aspects = [
        {
            "raw_aspect": "picture",
            "aspect": "Picture Quality",
            "sentiment": "positive",
            "evidence": "The picture is sharp",
        },
        {
            "raw_aspect": "colors",
            "aspect": "Picture Quality",
            "sentiment": "positive",
            "evidence": "the colors are accurate",
        },
        {
            "raw_aspect": "screen",
            "aspect": "Picture Quality",
            "sentiment": "positive",
            "evidence": "the screen looks excellent",
        },
    ]
    return {
        "summary": "The picture is consistently strong.",
        "aspects": aspects,
        "positive_points": [item["evidence"] for item in aspects],
        "negative_points": [],
    }


def _checkpoint_record(row: pd.Series, status: str) -> dict[str, Any]:
    record = row.to_dict()
    record.update(
        {
            "analysis_status": status,
            "summary": "Saved summary." if status == "success" else "",
            "aspects": json.dumps(_analysis()["aspects"]) if status == "success" else "[]",
            "unique_aspects": json.dumps(
                build_unique_aspects(_analysis()["aspects"])
            )
            if status == "success"
            else "[]",
            "positive_points": json.dumps(["picture is excellent"])
            if status == "success"
            else "[]",
            "negative_points": "[]",
            "error_message": "saved failure" if status == "failed" else "",
            "analysis_time_seconds": 1.25 if status != "skipped" else 0.0,
        }
    )
    return record


class BatchAnalyzerTests(unittest.TestCase):
    def test_multiple_normal_reviews_are_analyzed_in_order(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R001", "cleaned_content": "first"},
                {"review_id": "R002", "cleaned_content": "second"},
                {"review_id": "R003", "cleaned_content": "third"},
            ]
        )
        client = object()

        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[_analysis("one"), _analysis("two"), _analysis("three")],
        ) as analyze:
            result = analyze_reviews(
                reviews,
                llm_client=client,  # type: ignore[arg-type]
                show_progress=False,
            )

        self.assertEqual(["success", "success", "success"], result["analysis_status"].tolist())
        self.assertEqual(["one", "two", "three"], result["summary"].tolist())
        self.assertEqual(["first", "second", "third"], [call.args[0] for call in analyze.call_args_list])
        self.assertTrue(all(call.kwargs["llm_client"] is client for call in analyze.call_args_list))

    def test_skipped_review_does_not_create_client_or_call_analyzer(self) -> None:
        reviews = _frame(
            [
                {
                    "review_id": "R008",
                    "content": "",
                    "original_content": "",
                    "cleaned_content": "",
                    "is_analyzable": False,
                    "skip_reason": "missing_content",
                }
            ]
        )

        with (
            patch("batch_analyzer.create_review_client") as create_client,
            patch("batch_analyzer.analyze_review") as analyze,
        ):
            result = analyze_reviews(reviews, show_progress=False)

        create_client.assert_not_called()
        analyze.assert_not_called()
        self.assertEqual("skipped", result.loc[0, "analysis_status"])
        self.assertEqual("missing_content", result.loc[0, "skip_reason"])
        self.assertEqual("", result.loc[0, "error_message"])
        self.assertEqual("[]", result.loc[0, "unique_aspects"])

    def test_failed_review_does_not_stop_later_reviews(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R001", "cleaned_content": "first"},
                {"review_id": "R002", "cleaned_content": "bad"},
                {"review_id": "R003", "cleaned_content": "third"},
            ]
        )

        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[_analysis("one"), RuntimeError("model exploded"), _analysis("three")],
        ) as analyze:
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        self.assertEqual(3, analyze.call_count)
        self.assertEqual(["success", "failed", "success"], result["analysis_status"].tolist())
        self.assertEqual("three", result.loc[2, "summary"])

    def test_transient_provider_failure_retries_once_then_succeeds(self) -> None:
        reviews = _frame([{"review_id": "R001", "cleaned_content": "retry once"}])
        sleeper = Mock()

        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[LLMConnectionError("connection reset"), _analysis("recovered")],
        ) as analyze:
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
                sleeper=sleeper,
            )

        self.assertEqual(2, analyze.call_count)
        sleeper.assert_called_once_with(1.0)
        self.assertEqual("success", result.loc[0, "analysis_status"])
        self.assertEqual("recovered", result.loc[0, "summary"])
        self.assertEqual("", result.loc[0, "error_message"])

    def test_provider_failure_twice_stops_after_one_retry(self) -> None:
        reviews = _frame([{"review_id": "R001", "cleaned_content": "retry once"}])
        sleeper = Mock()

        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[
                LLMConnectionError("temporary unavailable"),
                LLMConnectionError("still unavailable"),
            ],
        ) as analyze:
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
                sleeper=sleeper,
            )

        self.assertEqual(2, analyze.call_count)
        sleeper.assert_called_once_with(1.0)
        self.assertEqual("failed", result.loc[0, "analysis_status"])
        self.assertEqual("still unavailable", result.loc[0, "error_message"])

    def test_non_provider_failure_is_not_retried(self) -> None:
        reviews = _frame([{"review_id": "R001", "cleaned_content": "bad evidence"}])
        sleeper = Mock()

        with patch(
            "batch_analyzer.analyze_review",
            side_effect=ReviewAnalysisError("evidence invalid"),
        ) as analyze:
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
                sleeper=sleeper,
            )

        analyze.assert_called_once()
        sleeper.assert_not_called()
        self.assertEqual("failed", result.loc[0, "analysis_status"])
        self.assertEqual("evidence invalid", result.loc[0, "error_message"])

    def test_analysis_status_uses_only_allowed_values(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R001"},
                {"review_id": "R002"},
                {
                    "review_id": "R008",
                    "cleaned_content": "",
                    "is_analyzable": False,
                    "skip_reason": "missing_content",
                },
            ]
        )

        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[_analysis(), RuntimeError("failed")],
        ):
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        self.assertEqual({"success", "failed", "skipped"}, set(result["analysis_status"]))

    def test_aspects_are_valid_json_strings(self) -> None:
        reviews = _frame([{"review_id": "R001"}])
        expected = _analysis()["aspects"]

        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        serialized = result.loc[0, "aspects"]
        self.assertIsInstance(serialized, str)
        self.assertEqual(expected, json.loads(serialized))

    def test_failed_error_message_is_preserved_exactly(self) -> None:
        reviews = _frame([{"review_id": "R001"}])

        with patch(
            "batch_analyzer.analyze_review",
            side_effect=RuntimeError("actual provider error"),
        ):
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        self.assertEqual("failed", result.loc[0, "analysis_status"])
        self.assertEqual("actual provider error", result.loc[0, "error_message"])
        self.assertEqual("[]", result.loc[0, "aspects"])
        self.assertEqual("[]", result.loc[0, "unique_aspects"])

    def test_analysis_time_exists_and_is_non_negative(self) -> None:
        reviews = _frame([{"review_id": "R001"}, {"review_id": "R002"}])

        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[_analysis(), RuntimeError("failed")],
        ):
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        self.assertIn("analysis_time_seconds", result.columns)
        self.assertTrue(result["analysis_time_seconds"].ge(0).all())

    def test_limit_counts_only_analyzable_reviews(self) -> None:
        reviews = _frame(
            [
                {"review_id": "S001", "cleaned_content": "", "is_analyzable": False, "skip_reason": "missing_content"},
                {"review_id": "R001", "cleaned_content": "first"},
                {"review_id": "S002", "cleaned_content": "", "is_analyzable": False, "skip_reason": "empty_after_cleaning"},
                {"review_id": "R002", "cleaned_content": "second"},
                {"review_id": "R003", "cleaned_content": "not in this run"},
            ]
        )

        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[_analysis("first"), _analysis("second")],
        ) as analyze:
            result = analyze_reviews(
                reviews,
                limit=2,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        self.assertEqual(2, analyze.call_count)
        self.assertEqual(["S001", "R001", "S002", "R002"], result["review_id"].tolist())
        self.assertNotIn("R003", result["review_id"].tolist())
        self.assertEqual(["skipped", "success", "skipped", "success"], result["analysis_status"].tolist())

    def test_original_fields_are_preserved(self) -> None:
        reviews = _frame(
            [
                {
                    "review_id": "R001",
                    "rating": 4,
                    "title": pd.NA,
                    "content": "  Original text.  ",
                    "original_content": "  Original text.  ",
                    "cleaned_content": "Original text.",
                    "date": "2026-09-22",
                    "verified_purchase": "false",
                }
            ]
        )

        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        for column in reviews.columns:
            self.assertIn(column, result.columns)
            if column == "title":
                self.assertTrue(pd.isna(result.loc[0, column]))
            else:
                self.assertEqual(reviews.loc[0, column], result.loc[0, column])

    def test_input_dataframe_is_not_modified(self) -> None:
        reviews = _frame([{"review_id": "R001"}, {"review_id": "R002"}])
        original = reviews.copy(deep=True)

        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        pd.testing.assert_frame_equal(original, reviews)

    def test_save_round_trip_keeps_json_and_does_not_touch_another_file(self) -> None:
        results = pd.DataFrame(
            [{"review_id": "R001", "analysis_status": "success", "aspects": "[]"}]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw_reviews.csv"
            raw_path.write_text("unchanged", encoding="utf-8")
            output_path = temp_path / "outputs" / "analyzed_reviews.csv"

            saved_path = save_analyzed_reviews(results, output_path)
            loaded = pd.read_csv(saved_path, keep_default_na=False)

            self.assertEqual("[]", loaded.loc[0, "aspects"])
            self.assertEqual("unchanged", raw_path.read_text(encoding="utf-8"))

    def test_progress_displays_position_status_and_elapsed_time(self) -> None:
        reviews = _frame([{"review_id": "R001"}])
        output = io.StringIO()

        with (
            patch("batch_analyzer.analyze_review", return_value=_analysis()),
            redirect_stdout(output),
        ):
            analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
            )

        rendered = output.getvalue()
        self.assertIn("[1/1] R001 success |", rendered)
        self.assertIn("s", rendered)


class UniqueAspectTests(unittest.TestCase):
    def test_same_aspect_and_sentiment_collapses_to_one_record(self) -> None:
        unique = build_unique_aspects(_repeated_aspect_analysis()["aspects"])

        self.assertEqual(1, len(unique))
        self.assertEqual("Picture Quality", unique[0]["aspect"])
        self.assertEqual("positive", unique[0]["sentiment"])

    def test_repeated_mentions_have_count_three(self) -> None:
        unique = build_unique_aspects(_repeated_aspect_analysis()["aspects"])

        self.assertEqual(3, unique[0]["mention_count"])

    def test_same_aspect_with_different_sentiments_is_not_merged(self) -> None:
        aspects = [
            {"aspect": "Picture Quality", "sentiment": "positive"},
            {"aspect": "Picture Quality", "sentiment": "negative"},
        ]

        unique = build_unique_aspects(aspects)

        self.assertEqual(
            [("Picture Quality", "positive"), ("Picture Quality", "negative")],
            [(item["aspect"], item["sentiment"]) for item in unique],
        )

    def test_different_aspects_are_not_merged(self) -> None:
        aspects = [
            {"aspect": "Picture Quality", "sentiment": "positive"},
            {"aspect": "Sound Quality", "sentiment": "positive"},
        ]

        unique = build_unique_aspects(aspects)

        self.assertEqual(2, len(unique))
        self.assertEqual(
            ["Picture Quality", "Sound Quality"],
            [item["aspect"] for item in unique],
        )

    def test_batch_output_keeps_all_original_aspect_mentions(self) -> None:
        reviews = _frame([{"review_id": "R001"}])
        analysis = _repeated_aspect_analysis()
        original_aspects = deepcopy(analysis["aspects"])

        with patch("batch_analyzer.analyze_review", return_value=analysis):
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        self.assertEqual(original_aspects, json.loads(result.loc[0, "aspects"]))

    def test_evidence_and_mock_analysis_are_not_modified(self) -> None:
        reviews = _frame([{"review_id": "R001"}])
        analysis = _repeated_aspect_analysis()
        original_analysis = deepcopy(analysis)

        with patch("batch_analyzer.analyze_review", return_value=analysis):
            analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        self.assertEqual(original_analysis, analysis)
        self.assertEqual(
            [item["evidence"] for item in original_analysis["aspects"]],
            [item["evidence"] for item in analysis["aspects"]],
        )

    def test_other_is_retained_but_excluded_from_default_business_metrics(self) -> None:
        unique = build_unique_aspects(
            [{"aspect": "Other", "sentiment": "positive"}]
        )

        self.assertEqual("Other", unique[0]["aspect"])
        self.assertEqual(1, unique[0]["mention_count"])
        self.assertFalse(unique[0]["include_in_business_metrics"])

    def test_unique_aspects_is_a_valid_json_string(self) -> None:
        reviews = _frame([{"review_id": "R001"}])

        with patch(
            "batch_analyzer.analyze_review",
            return_value=_repeated_aspect_analysis(),
        ):
            result = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        serialized = result.loc[0, "unique_aspects"]
        self.assertIsInstance(serialized, str)
        self.assertEqual(3, json.loads(serialized)[0]["mention_count"])

    def test_unique_aspect_processing_does_not_modify_input_dataframe(self) -> None:
        reviews = _frame([{"review_id": "R001"}, {"review_id": "R002"}])
        original = reviews.copy(deep=True)

        with patch(
            "batch_analyzer.analyze_review",
            return_value=_repeated_aspect_analysis(),
        ):
            analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )

        pd.testing.assert_frame_equal(original, reviews)


class CheckpointBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        root = Path(self._temporary_directory.name)
        self.checkpoint_path = root / "analyzed_reviews_checkpoint.csv"
        self.meta_path = root / "analyzed_reviews_checkpoint.meta.json"
        self.final_path = root / "analyzed_reviews.csv"

    def _run(
        self,
        reviews: pd.DataFrame,
        *,
        limit: int | None = None,
        resume: bool = False,
        retry_failed: bool = False,
        checkpoint_interval: int = 50,
        max_consecutive_failures: int = 3,
        sleeper: Any | None = None,
    ):
        return run_checkpointed_batch(
            reviews,
            limit=limit,
            resume=resume,
            retry_failed=retry_failed,
            llm_client=object(),  # type: ignore[arg-type]
            checkpoint_path=self.checkpoint_path,
            meta_path=self.meta_path,
            final_output_path=self.final_path,
            checkpoint_interval=checkpoint_interval,
            max_consecutive_failures=max_consecutive_failures,
            show_progress=False,
            sleeper=sleeper if sleeper is not None else (lambda _seconds: None),
        )

    def test_source_row_id_is_stable_and_positional(self) -> None:
        reviews = _frame(
            [{"review_id": "R001"}, {"review_id": "R001"}, {"review_id": "R003"}]
        )
        reviews.index = [20, 20, 5]

        first = add_source_row_ids(reviews)
        second = add_source_row_ids(reviews.copy(deep=True))

        self.assertEqual([0, 1, 2], first["source_row_id"].tolist())
        self.assertEqual(first["source_row_id"].tolist(), second["source_row_id"].tolist())

    def test_duplicate_review_ids_resume_as_independent_source_rows(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R001", "cleaned_content": "first duplicate"},
                {"review_id": "R001", "cleaned_content": "second duplicate"},
            ]
        )
        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            self._run(reviews, limit=1)

        with patch("batch_analyzer.analyze_review", return_value=_analysis()) as analyze:
            result = self._run(reviews, resume=True, limit=1)

        analyze.assert_called_once()
        self.assertEqual("second duplicate", analyze.call_args.args[0])
        self.assertEqual([0, 1], result.reviews["source_row_id"].tolist())
        self.assertEqual(["R001", "R001"], result.reviews["review_id"].tolist())

    def test_checkpoint_round_trip_contains_required_columns(self) -> None:
        reviews = _frame([{"review_id": "R001"}])
        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            self._run(reviews, limit=1)

        loaded = pd.read_csv(self.checkpoint_path, keep_default_na=False)

        required = {
            "source_row_id",
            *reviews.columns,
            "analysis_status",
            "summary",
            "aspects",
            "unique_aspects",
            "positive_points",
            "negative_points",
            "error_message",
            "analysis_time_seconds",
        }
        self.assertTrue(required.issubset(loaded.columns))
        self.assertEqual("success", loaded.loc[0, "analysis_status"])
        self.assertTrue(self.meta_path.is_file())

    def test_checkpoint_save_uses_atomic_replace_and_preserves_old_file_on_write_error(self) -> None:
        reviews = _frame([{"review_id": "R001"}])
        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            results = analyze_reviews(
                reviews,
                llm_client=object(),  # type: ignore[arg-type]
                show_progress=False,
            )
        signature = calculate_input_signature(add_source_row_ids(reviews))
        real_replace = os.replace
        replace_calls: list[tuple[Path, Path]] = []

        def replace_spy(source: object, destination: object) -> None:
            source_path = Path(source)
            destination_path = Path(destination)
            self.assertTrue(source_path.is_file())
            self.assertEqual(source_path.parent, destination_path.parent)
            replace_calls.append((source_path, destination_path))
            real_replace(source_path, destination_path)

        with patch("batch_analyzer.os.replace", side_effect=replace_spy):
            save_analysis_checkpoint(
                results,
                signature,
                len(reviews),
                self.checkpoint_path,
                self.meta_path,
            )

        self.assertEqual(2, len(replace_calls))
        self.assertEqual(self.checkpoint_path, replace_calls[0][1])
        self.assertFalse(self.checkpoint_path.with_suffix(".tmp").exists())
        previous_bytes = self.checkpoint_path.read_bytes()

        with (
            patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")),
            patch("batch_analyzer.os.replace") as replace,
        ):
            with self.assertRaises(OSError):
                save_analysis_checkpoint(
                    results,
                    signature,
                    len(reviews),
                    self.checkpoint_path,
                    self.meta_path,
                )

        replace.assert_not_called()
        self.assertEqual(previous_bytes, self.checkpoint_path.read_bytes())

    def test_resume_skips_existing_success(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R001", "cleaned_content": "already done"},
                {"review_id": "R002", "cleaned_content": "new row"},
            ]
        )
        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            self._run(reviews, limit=1)

        with patch("batch_analyzer.analyze_review", return_value=_analysis()) as analyze:
            self._run(reviews, resume=True, limit=1)

        analyze.assert_called_once()
        self.assertEqual("new row", analyze.call_args.args[0])

    def test_resume_preserves_existing_row_text_exactly(self) -> None:
        reviews = _frame(
            [
                {
                    "review_id": "R001",
                    "cleaned_content": "already done",
                    "verified_purchase": "true",
                },
                {"review_id": "R002", "cleaned_content": "new row"},
            ]
        )
        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            self._run(reviews, limit=1)
        before = pd.read_csv(
            self.checkpoint_path,
            dtype="string",
            keep_default_na=False,
        ).iloc[0].copy()

        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            self._run(reviews, resume=True, limit=1)

        after = pd.read_csv(
            self.checkpoint_path,
            dtype="string",
            keep_default_na=False,
        ).iloc[0]
        pd.testing.assert_series_equal(before, after)

    def test_resume_skips_existing_skipped_row(self) -> None:
        reviews = _frame(
            [
                {
                    "review_id": "S001",
                    "cleaned_content": "",
                    "is_analyzable": False,
                    "skip_reason": "missing_content",
                },
                {"review_id": "R001", "cleaned_content": "already done"},
                {"review_id": "R002", "cleaned_content": "new row"},
            ]
        )
        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            self._run(reviews, limit=1)

        with patch("batch_analyzer.analyze_review", return_value=_analysis()) as analyze:
            result = self._run(reviews, resume=True, limit=1)

        analyze.assert_called_once()
        self.assertEqual("new row", analyze.call_args.args[0])
        skipped = result.reviews.loc[result.reviews["source_row_id"].eq(0)].iloc[0]
        self.assertEqual("skipped", skipped["analysis_status"])
        self.assertEqual("missing_content", skipped["skip_reason"])

    def test_resume_preserves_failed_without_retrying_it(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R001", "cleaned_content": "failed before"},
                {"review_id": "R002", "cleaned_content": "new row"},
            ]
        )
        with patch(
            "batch_analyzer.analyze_review",
            side_effect=RuntimeError("first failure"),
        ):
            self._run(reviews, limit=1)
        before = pd.read_csv(
            self.checkpoint_path,
            dtype="string",
            keep_default_na=False,
        )
        old_time = before.loc[0, "analysis_time_seconds"]

        with patch("batch_analyzer.analyze_review", return_value=_analysis()) as analyze:
            resumed = self._run(reviews, resume=True, limit=1)

        analyze.assert_called_once()
        self.assertEqual("new row", analyze.call_args.args[0])
        failed = resumed.reviews.loc[resumed.reviews["source_row_id"].eq(0)].iloc[0]
        self.assertEqual("failed", failed["analysis_status"])
        self.assertEqual("first failure", failed["error_message"])
        after = pd.read_csv(
            self.checkpoint_path,
            dtype="string",
            keep_default_na=False,
        )
        self.assertEqual(old_time, after.loc[0, "analysis_time_seconds"])

    def test_retry_failed_calls_only_rows_failed_at_start(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R001", "cleaned_content": "success row"},
                {"review_id": "S001", "cleaned_content": "", "is_analyzable": False},
                {"review_id": "R002", "cleaned_content": "failed row"},
                {"review_id": "R003", "cleaned_content": "unseen row"},
            ]
        )
        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[_analysis(), RuntimeError("failed")],
        ):
            self._run(reviews, limit=2)

        with patch("batch_analyzer.analyze_review", return_value=_analysis()) as analyze:
            retried = self._run(reviews, retry_failed=True)

        analyze.assert_called_once()
        self.assertEqual("failed row", analyze.call_args.args[0])
        self.assertNotIn(3, retried.reviews["source_row_id"].tolist())

    def test_successful_retry_replaces_failed_row_without_duplicate(self) -> None:
        reviews = _frame([{"review_id": "R001", "cleaned_content": "retry me"}])
        with patch(
            "batch_analyzer.analyze_review",
            side_effect=RuntimeError("temporary failure"),
        ):
            self._run(reviews, limit=1)

        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            retried = self._run(reviews, retry_failed=True)

        self.assertEqual(1, len(retried.reviews))
        self.assertEqual("success", retried.reviews.loc[0, "analysis_status"])
        self.assertEqual("", retried.reviews.loc[0, "error_message"])

    def test_limit_after_resume_counts_only_new_llm_attempts(self) -> None:
        reviews = _frame(
            [
                {"review_id": f"R{index:03d}", "cleaned_content": f"review {index}"}
                for index in range(5)
            ]
        )
        with patch("batch_analyzer.analyze_review", return_value=_analysis()) as first:
            self._run(reviews, limit=3)
        self.assertEqual(3, first.call_count)

        with patch("batch_analyzer.analyze_review", return_value=_analysis()) as second:
            result = self._run(reviews, resume=True, limit=2)

        self.assertEqual(2, second.call_count)
        self.assertEqual(
            ["review 3", "review 4"],
            [call.args[0] for call in second.call_args_list],
        )
        self.assertEqual([0, 1, 2, 3, 4], result.reviews["source_row_id"].tolist())

    def test_input_signature_is_stable_for_identical_rows(self) -> None:
        reviews = _frame([{"review_id": "R001"}, {"review_id": "R001"}])

        first = calculate_input_signature(add_source_row_ids(reviews))
        second = calculate_input_signature(add_source_row_ids(reviews.copy(deep=True)))

        self.assertEqual(first, second)
        self.assertEqual(64, len(first))

    def test_signature_mismatch_rejects_resume_before_llm_call(self) -> None:
        reviews = _frame([{"review_id": "R001", "cleaned_content": "original"}])
        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            self._run(reviews, limit=1)

        changed = reviews.copy(deep=True)
        changed.loc[0, "cleaned_content"] = "changed"
        with patch("batch_analyzer.analyze_review") as analyze:
            with self.assertRaisesRegex(
                CheckpointMismatchError,
                "Checkpoint does not match the current input dataset",
            ):
                self._run(changed, resume=True)

        analyze.assert_not_called()

    def test_three_consecutive_connection_failures_stop_and_save(self) -> None:
        reviews = _frame(
            [
                {"review_id": f"R{index:03d}", "cleaned_content": f"review {index}"}
                for index in range(5)
            ]
        )
        failures = [LLMConnectionError(f"offline {index}") for index in range(6)]
        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[*failures, _analysis()],
        ) as analyze:
            result = self._run(reviews, limit=5, max_consecutive_failures=3)

        self.assertEqual(6, analyze.call_count)
        self.assertTrue(result.stopped_for_failures)
        self.assertFalse(result.is_complete)
        self.assertEqual(3, len(result.reviews))
        self.assertTrue(self.checkpoint_path.is_file())
        self.assertFalse(self.final_path.exists())

    def test_success_resets_consecutive_connection_failure_count(self) -> None:
        reviews = _frame(
            [
                {"review_id": f"R{index:03d}", "cleaned_content": f"review {index}"}
                for index in range(6)
            ]
        )
        side_effects: list[object] = [
            LLMConnectionError("offline 1"),
            LLMConnectionError("offline 2"),
            LLMConnectionError("offline 3"),
            LLMConnectionError("offline 4"),
            _analysis(),
            LLMConnectionError("offline 5"),
            LLMConnectionError("offline 6"),
            LLMConnectionError("offline 7"),
            LLMConnectionError("offline 8"),
            LLMConnectionError("offline 9"),
            LLMConnectionError("offline 10"),
        ]
        with patch(
            "batch_analyzer.analyze_review",
            side_effect=side_effects,
        ) as analyze:
            result = self._run(reviews, limit=6, max_consecutive_failures=3)

        self.assertEqual(11, analyze.call_count)
        self.assertEqual("success", result.reviews.loc[2, "analysis_status"])
        self.assertTrue(result.stopped_for_failures)

    def test_evidence_validation_failure_does_not_count_as_provider_outage(self) -> None:
        reviews = _frame(
            [
                {"review_id": f"R{index:03d}", "cleaned_content": f"review {index}"}
                for index in range(4)
            ]
        )
        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[
                LLMConnectionError("offline 1"),
                LLMConnectionError("offline 1 retry"),
                ReviewAnalysisError("evidence invalid"),
                LLMConnectionError("offline 2"),
                LLMConnectionError("offline 2 retry"),
                _analysis(),
            ],
        ) as analyze:
            result = self._run(reviews, limit=4, max_consecutive_failures=2)

        self.assertEqual(6, analyze.call_count)
        self.assertFalse(result.stopped_for_failures)
        self.assertTrue(result.is_complete)

    def test_keyboard_interrupt_saves_completed_rows(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R001", "cleaned_content": "first"},
                {"review_id": "R002", "cleaned_content": "interrupt here"},
                {"review_id": "R003", "cleaned_content": "never called"},
            ]
        )
        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[_analysis(), KeyboardInterrupt()],
        ) as analyze:
            result = self._run(reviews, limit=3, checkpoint_interval=20)

        self.assertEqual(2, analyze.call_count)
        self.assertTrue(result.interrupted)
        self.assertEqual([0], result.reviews["source_row_id"].tolist())
        self.assertTrue(self.checkpoint_path.is_file())
        self.assertFalse(self.final_path.exists())

    def test_final_output_is_sorted_by_source_row_id(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R000", "cleaned_content": "zero"},
                {"review_id": "R001", "cleaned_content": "one"},
                {"review_id": "R002", "cleaned_content": "two"},
            ]
        )
        prepared = add_source_row_ids(reviews)
        seeded = pd.DataFrame(
            [
                _checkpoint_record(prepared.iloc[2], "success"),
                _checkpoint_record(prepared.iloc[0], "success"),
            ]
        )
        save_analysis_checkpoint(
            seeded,
            calculate_input_signature(prepared),
            len(prepared),
            self.checkpoint_path,
            self.meta_path,
        )

        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            result = self._run(reviews, resume=True)

        final = pd.read_csv(self.final_path, keep_default_na=False)
        self.assertTrue(result.is_complete)
        self.assertEqual([0, 1, 2], final["source_row_id"].tolist())

    def test_checkpoint_json_fields_are_valid_for_every_status(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R001"},
                {"review_id": "R002"},
                {"review_id": "S001", "cleaned_content": "", "is_analyzable": False},
            ]
        )
        with patch(
            "batch_analyzer.analyze_review",
            side_effect=[_analysis(), RuntimeError("bad row")],
        ):
            self._run(reviews, limit=2)

        loaded = pd.read_csv(self.checkpoint_path, keep_default_na=False)
        self.assertEqual({"success", "failed", "skipped"}, set(loaded["analysis_status"]))
        for column in ("aspects", "unique_aspects", "positive_points", "negative_points"):
            for serialized in loaded[column]:
                self.assertIsInstance(json.loads(serialized), list)

    def test_unique_aspects_survive_checkpoint_round_trip(self) -> None:
        reviews = _frame([{"review_id": "R001"}])
        with patch(
            "batch_analyzer.analyze_review",
            return_value=_repeated_aspect_analysis(),
        ):
            self._run(reviews, limit=1)

        loaded = pd.read_csv(self.checkpoint_path, keep_default_na=False)
        unique = json.loads(loaded.loc[0, "unique_aspects"])
        self.assertEqual(3, unique[0]["mention_count"])
        self.assertTrue(unique[0]["include_in_business_metrics"])

    def test_original_aspects_and_evidence_survive_checkpoint_round_trip(self) -> None:
        reviews = _frame([{"review_id": "R001"}])
        analysis = _repeated_aspect_analysis()
        analysis["aspects"][0]["evidence"] = "The picture is sharp — even at 4K!"
        original = deepcopy(analysis)
        with patch("batch_analyzer.analyze_review", return_value=analysis):
            self._run(reviews, limit=1)

        loaded = pd.read_csv(self.checkpoint_path, keep_default_na=False)
        self.assertEqual(original["aspects"], json.loads(loaded.loc[0, "aspects"]))
        self.assertEqual(original, analysis)

    def test_checkpointed_resume_does_not_modify_input_dataframe(self) -> None:
        reviews = _frame([{"review_id": "R001"}, {"review_id": "R002"}])
        original = reviews.copy(deep=True)
        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            self._run(reviews, limit=1)
            self._run(reviews, resume=True, limit=1)

        pd.testing.assert_frame_equal(original, reviews)

    def test_corrupted_checkpoint_is_rejected_without_llm_call(self) -> None:
        reviews = _frame([{"review_id": "R001"}])
        with patch("batch_analyzer.analyze_review", return_value=_analysis()):
            self._run(reviews, limit=1)
        corrupted = pd.read_csv(self.checkpoint_path, keep_default_na=False)
        corrupted.loc[0, "aspects"] = "{not valid json"
        corrupted.to_csv(self.checkpoint_path, index=False)

        with patch("batch_analyzer.analyze_review") as analyze:
            with self.assertRaisesRegex(
                CheckpointInvalidError,
                "Checkpoint is invalid or corrupted",
            ):
                self._run(reviews, resume=True)

        analyze.assert_not_called()

    def test_checkpoint_interval_counts_success_and_failed_but_not_skipped(self) -> None:
        reviews = _frame(
            [
                {"review_id": "R001"},
                {"review_id": "S001", "cleaned_content": "", "is_analyzable": False},
                {"review_id": "R002"},
                {"review_id": "R003"},
            ]
        )
        with (
            patch(
                "batch_analyzer.analyze_review",
                side_effect=[_analysis(), RuntimeError("failed"), _analysis()],
            ),
            patch(
                "batch_analyzer.save_analysis_checkpoint",
                wraps=save_analysis_checkpoint,
            ) as save,
        ):
            self._run(reviews, limit=3, checkpoint_interval=2)

        self.assertEqual(2, save.call_count)

    def test_failed_retry_failure_updates_same_row(self) -> None:
        reviews = _frame([{"review_id": "R001", "cleaned_content": "retry me"}])
        with patch(
            "batch_analyzer.analyze_review",
            side_effect=RuntimeError("first error"),
        ):
            self._run(reviews, limit=1)
        with patch(
            "batch_analyzer.analyze_review",
            side_effect=RuntimeError("second error"),
        ):
            retried = self._run(reviews, retry_failed=True)

        self.assertEqual(1, len(retried.reviews))
        self.assertEqual("failed", retried.reviews.loc[0, "analysis_status"])
        self.assertEqual("second error", retried.reviews.loc[0, "error_message"])


if __name__ == "__main__":
    unittest.main()
