"""Offline tests for the shared Step 11 production runtime utilities."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_runtime import (  # noqa: E402
    AnalysisRunBusyError,
    AnalysisState,
    ProgressTracker,
    RuntimeExecutionResult,
    RuntimePaths,
    StructuredEventLog,
    analysis_run_lock,
    format_duration,
    inspect_checkpoint,
    load_checkpoint_statuses,
    load_run_metadata,
    make_run_id,
    run_analysis_runtime,
)
from batch_analyzer import (  # noqa: E402
    BatchRunResult,
    CheckpointMismatchError,
    load_analysis_checkpoint,
    save_analysis_checkpoint,
)
import amazon_analysis  # noqa: E402
import interactive_analysis  # noqa: E402


class ManualClock:
    """Controllable monotonic clock used to make ETA tests deterministic."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _runtime_paths(root: Path) -> RuntimePaths:
    return RuntimePaths(
        root=root,
        checkpoint=root / "analysis_checkpoint.csv",
        checkpoint_meta=root / "analysis_checkpoint.meta.json",
        analyzed_reviews=root / "analyzed_reviews.csv",
        aspect_summary=root / "aspect_summary.csv",
        aspect_evidence=root / "aspect_evidence.json",
        business_summary=root / "business_summary.json",
        keyword_summary=root / "keyword_summary.csv",
        sentiment_keywords=root / "sentiment_keywords.json",
        aspect_keywords=root / "aspect_keywords.json",
        run_metadata=root / "analysis_run_metadata.json",
        event_log=root / "analysis_run_events.jsonl",
    )


class AnalysisStateTests(unittest.TestCase):
    def test_state_set_is_explicit_and_complete(self) -> None:
        self.assertEqual(
            {
                "idle",
                "validating",
                "preparing",
                "analyzing",
                "paused",
                "retrying",
                "aggregating",
                "generating_keywords",
                "completed",
                "completed_with_failures",
                "interrupted",
                "failed",
            },
            {state.value for state in AnalysisState},
        )

    def test_format_duration_handles_pending_seconds_minutes_and_hours(self) -> None:
        self.assertEqual("Estimating remaining time...", format_duration(None))
        self.assertEqual("0s", format_duration(-5))
        self.assertEqual("42s", format_duration(42))
        self.assertEqual("2m 5s", format_duration(125))
        self.assertEqual("1h 2m", format_duration(3725))


class ProgressTrackerTests(unittest.TestCase):
    def test_eta_stays_hidden_until_three_successful_reviews(self) -> None:
        clock = ManualClock()
        tracker = ProgressTracker(5, clock=clock)

        for position, latency in enumerate((2.0, 4.0), start=1):
            tracker.handle(position, 5, f"R{position:03d}", "analyzing")
            clock.advance(latency)
            snapshot = tracker.handle(position, 5, f"R{position:03d}", "success")
            self.assertIsNone(snapshot.eta_seconds)

        tracker.handle(3, 5, "R003", "analyzing")
        clock.advance(6.0)
        snapshot = tracker.handle(3, 5, "R003", "success")

        # Rolling median is 4 seconds and two operation rows remain.
        self.assertEqual(8.0, snapshot.eta_seconds)
        self.assertEqual(12.0, snapshot.elapsed_seconds)
        self.assertEqual(3, snapshot.successful)
        self.assertEqual(3, snapshot.completed_reviews)
        self.assertEqual("R003", snapshot.current_review_id)
        self.assertAlmostEqual(0.6, snapshot.percent_complete)

    def test_eta_accounts_for_two_worker_throughput_without_changing_counts(self) -> None:
        clock = ManualClock()
        tracker = ProgressTracker(5, concurrency=2, clock=clock)
        for position in range(1, 4):
            tracker.handle(position, 5, f"R{position:03d}", "analyzing")
            clock.advance(4)
            snapshot = tracker.handle(position, 5, f"R{position:03d}", "success")

        self.assertEqual(4.0, snapshot.eta_seconds)
        self.assertEqual(3, snapshot.operation_completed)
        self.assertEqual(5, snapshot.operation_total)

    def test_large_batch_progress_has_no_artificial_100_review_cap(self) -> None:
        initial = {position: "success" for position in range(300)}
        pending = set(range(300, 500))
        tracker = ProgressTracker(
            500,
            initial_statuses=initial,
            operation_positions=pending,
            clock=ManualClock(),
        )

        initial_snapshot = tracker.snapshot()
        self.assertEqual(500, initial_snapshot.total_reviews)
        self.assertEqual(300, initial_snapshot.completed_reviews)
        self.assertEqual(200, initial_snapshot.operation_total)
        self.assertEqual(0, initial_snapshot.operation_completed)
        self.assertEqual(0.0, initial_snapshot.percent_complete)

        tracker.handle(301, 500, "R0301", "analyzing")
        snapshot = tracker.handle(301, 500, "R0301", "success")
        self.assertEqual(301, snapshot.completed_reviews)
        self.assertEqual(1, snapshot.operation_completed)
        self.assertAlmostEqual(1 / 200, snapshot.percent_complete)

    def test_initial_failed_and_skipped_counts_are_kept_for_resume(self) -> None:
        tracker = ProgressTracker(
            6,
            initial_statuses={0: "success", 1: "failed", 2: "skipped"},
            operation_positions={3, 4, 5},
            clock=ManualClock(),
        )
        snapshot = tracker.snapshot()
        self.assertEqual(3, snapshot.completed_reviews)
        self.assertEqual(1, snapshot.successful)
        self.assertEqual(1, snapshot.failed)
        self.assertEqual(1, snapshot.skipped)

    def test_retry_counter_and_non_analysis_state_are_reported(self) -> None:
        tracker = ProgressTracker(1, clock=ManualClock())
        snapshot = tracker.add_retry()
        self.assertEqual(1, snapshot.retry_count)
        snapshot = tracker.set_state(AnalysisState.AGGREGATING)
        self.assertEqual(AnalysisState.AGGREGATING, snapshot.state)
        self.assertIsNone(snapshot.current_review_id)

    def test_invalid_total_or_concurrency_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProgressTracker(-1)
        for concurrency in (0, 3):
            with self.subTest(concurrency=concurrency), self.assertRaises(ValueError):
                ProgressTracker(1, concurrency=concurrency)


class CheckpointSummaryTests(unittest.TestCase):
    def test_checkpoint_summary_separates_completed_failed_and_remaining(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.csv"
            pd.DataFrame(
                [
                    {"source_row_id": 0, "analysis_status": "success"},
                    {"source_row_id": 1, "analysis_status": "failed"},
                    {"source_row_id": 2, "analysis_status": "skipped"},
                ]
            ).to_csv(path, index=False)

            self.assertEqual(
                {0: "success", 1: "failed", 2: "skipped"},
                load_checkpoint_statuses(path),
            )
            summary = inspect_checkpoint(path, total_reviews=5)

        self.assertEqual(5, summary.total_reviews)
        self.assertEqual(2, summary.already_completed)
        self.assertEqual(2, summary.remaining)
        self.assertEqual(1, summary.successful)
        self.assertEqual(1, summary.failed)
        self.assertEqual(1, summary.skipped)

    def test_missing_or_invalid_checkpoint_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.csv"
            self.assertEqual({}, load_checkpoint_statuses(missing))
            self.assertEqual(7, inspect_checkpoint(missing, total_reviews=7).remaining)

            malformed = root / "malformed.csv"
            malformed.write_text("not,the,required,columns\n1,2,3,4\n", encoding="utf-8")
            self.assertEqual({}, load_checkpoint_statuses(malformed))


class RuntimeMetadataAndPrivacyTests(unittest.TestCase):
    def _completed_batch(self, paths: RuntimePaths) -> BatchRunResult:
        analyzed = pd.DataFrame(
            {
                "source_row_id": [0],
                "review_id": ["R001"],
                "analysis_status": ["success"],
                "analysis_time_seconds": [1.0],
            }
        )
        return BatchRunResult(
            reviews=analyzed,
            total_input_rows=1,
            previously_completed=0,
            newly_processed=1,
            attempted_reviews=1,
            is_complete=True,
            interrupted=False,
            stopped_for_failures=False,
            checkpoint_path=paths.checkpoint,
            final_output_path=paths.analyzed_reviews,
        )

    def _run_id_survives_follow_up_mode(self, mode: str) -> None:
        reviews = pd.DataFrame(
            {
                "source_row_id": [0],
                "review_id": ["R001"],
                "cleaned_content": ["Review text is deliberately not logged."],
                "is_analyzable": [True],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = _runtime_paths(Path(temporary))
            batch = self._completed_batch(paths)
            with (
                patch("analysis_runtime.run_checkpointed_batch", return_value=batch),
                patch("analysis_runtime.aggregate_reviews", return_value=object()),
                patch("analysis_runtime.save_aggregation_outputs"),
                patch("analysis_runtime.analyze_keywords", return_value=object()),
                patch("analysis_runtime.save_keyword_outputs"),
            ):
                first = run_analysis_runtime(
                    reviews,
                    source_type="upload",
                    llm_client=object(),  # type: ignore[arg-type]
                    paths=paths,
                )
                follow_up = run_analysis_runtime(
                    reviews,
                    source_type="ignored-on-follow-up",
                    llm_client=object(),  # type: ignore[arg-type]
                    paths=paths,
                    resume=mode == "resume",
                    retry_failed=mode == "retry_failed",
                )

        self.assertEqual(first.run_metadata["run_id"], follow_up.run_metadata["run_id"])
        self.assertEqual("upload", follow_up.run_metadata["source_type"])

    def test_run_id_and_source_type_stay_stable_when_resuming(self) -> None:
        self._run_id_survives_follow_up_mode("resume")

    def test_run_id_and_source_type_stay_stable_when_retrying_failures(self) -> None:
        self._run_id_survives_follow_up_mode("retry_failed")

    def test_run_id_contains_utc_timestamp_and_random_suffix(self) -> None:
        fixed = datetime(2026, 10, 8, 7, 6, 5, tzinfo=timezone.utc)
        with patch("analysis_runtime.secrets.token_hex", return_value="a1b2c3"):
            run_id = make_run_id(fixed)
        self.assertEqual("20261008T070605Z-a1b2c3", run_id)
        self.assertRegex(run_id, r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")

    def test_structured_event_log_allowlist_excludes_review_text_and_error_message(self) -> None:
        secret_review = "PRIVATE REVIEW BODY 9821"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            logger = StructuredEventLog(path, "run-1", append=False)
            logger(
                {
                    "source_row_id": 7,
                    "review_id": "R008",
                    "status": "failed",
                    "latency_seconds": 1.25,
                    "retry_count": 1,
                    "error_type": "LLMConnectionError",
                    "original_content": secret_review,
                    "cleaned_content": secret_review,
                    "error_message": f"failure while processing {secret_review}",
                    "summary": secret_review,
                }
            )
            raw = path.read_text(encoding="utf-8")
            event = json.loads(raw)

        self.assertNotIn(secret_review, raw)
        self.assertNotIn("error_message", event)
        self.assertEqual(
            {
                "timestamp",
                "run_id",
                "source_row_id",
                "review_id",
                "status",
                "latency_seconds",
                "retry_count",
                "error_type",
            },
            set(event),
        )

    def test_load_run_metadata_handles_valid_missing_and_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "metadata.json"
            self.assertIsNone(load_run_metadata(path))
            path.write_text('{"run_id": "run-1", "status": "completed"}', encoding="utf-8")
            self.assertEqual("run-1", load_run_metadata(path)["run_id"])
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(load_run_metadata(path))

    def test_workspace_lock_blocks_a_second_run_and_releases_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "session" / "current"
            with analysis_run_lock(root):
                with self.assertRaisesRegex(AnalysisRunBusyError, "already"):
                    with analysis_run_lock(root):
                        self.fail("the second lock must not be acquired")
            with analysis_run_lock(root) as path:
                self.assertTrue(path.is_file())
            self.assertFalse(path.exists())

    def test_checkpoint_context_rejects_a_different_model_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.csv"
            metadata = root / "checkpoint.meta.json"
            save_analysis_checkpoint(
                pd.DataFrame(),
                "input-signature",
                0,
                checkpoint,
                metadata,
                checkpoint_context={"model": "qwen3:8b", "candidate": "v2"},
            )
            with self.assertRaisesRegex(CheckpointMismatchError, "model"):
                load_analysis_checkpoint(
                    "input-signature",
                    0,
                    checkpoint_path=checkpoint,
                    meta_path=metadata,
                    expected_checkpoint_context={
                        "model": "different-model",
                        "candidate": "v2",
                    },
                )

    def test_resume_accumulates_duration_and_phase_timings(self) -> None:
        reviews = pd.DataFrame(
            {
                "source_row_id": [0],
                "review_id": ["R001"],
                "cleaned_content": ["Works well."],
                "is_analyzable": [True],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = _runtime_paths(Path(temporary))
            paths.run_metadata.write_text(
                json.dumps(
                    {
                        "run_id": "stable-run",
                        "source_type": "upload",
                        "started_at": "2026-10-08T00:00:00+00:00",
                        "duration_seconds": 10.0,
                        "timings": {
                            "validation_time": 2.0,
                            "preparing_time": 1.0,
                            "llm_time": 5.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            batch = self._completed_batch(paths)
            with (
                patch("analysis_runtime.run_checkpointed_batch", return_value=batch),
                patch("analysis_runtime.aggregate_reviews", return_value=object()),
                patch("analysis_runtime.save_aggregation_outputs"),
                patch("analysis_runtime.analyze_keywords", return_value=object()),
                patch("analysis_runtime.save_keyword_outputs"),
            ):
                result = run_analysis_runtime(
                    reviews,
                    source_type="ignored",
                    llm_client=object(),  # type: ignore[arg-type]
                    paths=paths,
                    resume=True,
                    validation_time_seconds=0.5,
                    preparing_time_seconds=0.25,
                    pre_runtime_elapsed_seconds=1.0,
                )

        self.assertGreaterEqual(result.run_metadata["duration_seconds"], 11.0)
        self.assertEqual(2.5, result.run_metadata["timings"]["validation_time"])
        self.assertEqual(1.25, result.run_metadata["timings"]["preparing_time"])
        self.assertEqual(
            0.5,
            result.run_metadata["last_invocation_timings"]["validation_time"],
        )

    def test_failed_resume_preserves_previous_duration_and_timings(self) -> None:
        reviews = pd.DataFrame(
            {
                "review_id": ["R001"],
                "cleaned_content": ["Works well."],
                "is_analyzable": [True],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = _runtime_paths(Path(temporary))
            paths.run_metadata.write_text(
                json.dumps(
                    {
                        "run_id": "stable-run",
                        "source_type": "upload",
                        "started_at": "2026-10-08T00:00:00+00:00",
                        "duration_seconds": 10.0,
                        "timings": {"llm_time": 5.0, "checkpoint_time": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "analysis_runtime.run_checkpointed_batch",
                side_effect=RuntimeError("offline failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "offline failure"):
                    run_analysis_runtime(
                        reviews,
                        source_type="ignored",
                        llm_client=object(),  # type: ignore[arg-type]
                        paths=paths,
                        resume=True,
                        validation_time_seconds=0.25,
                        pre_runtime_elapsed_seconds=1.0,
                    )
            saved = json.loads(paths.run_metadata.read_text(encoding="utf-8"))

        self.assertGreaterEqual(saved["duration_seconds"], 11.0)
        self.assertEqual(5.0, saved["timings"]["llm_time"])
        self.assertEqual(0.5, saved["timings"]["checkpoint_time"])
        self.assertEqual(0.25, saved["timings"]["validation_time"])
        self.assertEqual("failed", saved["status"])

    def test_runtime_writes_operational_metadata_and_never_review_body(self) -> None:
        private_body = "PRIVATE AMAZON REVIEW BODY 4319"
        reviews = pd.DataFrame(
            [
                {
                    "source_row_id": 0,
                    "review_id": "R001",
                    "original_content": private_body,
                    "cleaned_content": private_body,
                    "is_analyzable": True,
                },
                {
                    "source_row_id": 1,
                    "review_id": "R002",
                    "original_content": "",
                    "cleaned_content": "",
                    "is_analyzable": False,
                },
            ]
        )
        analyzed = reviews.copy(deep=True)
        analyzed["analysis_status"] = ["success", "skipped"]
        analyzed["analysis_time_seconds"] = [2.0, 0.0]

        with tempfile.TemporaryDirectory() as temporary:
            paths = _runtime_paths(Path(temporary))
            batch = BatchRunResult(
                reviews=analyzed,
                total_input_rows=2,
                previously_completed=0,
                newly_processed=2,
                attempted_reviews=1,
                is_complete=True,
                interrupted=False,
                stopped_for_failures=False,
                checkpoint_path=paths.checkpoint,
                final_output_path=paths.analyzed_reviews,
                retry_count=1,
                model_failures=0,
                llm_time_seconds=1.7,
                postprocessing_time_seconds=0.2,
                checkpoint_time_seconds=0.1,
            )

            def fake_batch(*args: object, **kwargs: object) -> BatchRunResult:
                self.assertEqual(
                    {
                        "model": "qwen3:8b",
                        "candidate": "generalization_v2_candidate",
                    },
                    kwargs["checkpoint_context"],
                )
                progress = kwargs["progress_callback"]
                event = kwargs["runtime_event_callback"]
                progress(1, 2, "R001", "analyzing")
                event(
                    {
                        "source_row_id": 0,
                        "review_id": "R001",
                        "status": "success",
                        "latency_seconds": 2.0,
                        "retry_count": 1,
                        "error_type": None,
                        "original_content": private_body,
                    }
                )
                progress(1, 2, "R001", "success")
                progress(2, 2, "R002", "skipped")
                return batch

            with (
                patch("analysis_runtime.run_checkpointed_batch", side_effect=fake_batch),
                patch("analysis_runtime.aggregate_reviews", return_value=object()),
                patch("analysis_runtime.save_aggregation_outputs"),
                patch("analysis_runtime.analyze_keywords", return_value=object()),
                patch("analysis_runtime.save_keyword_outputs"),
            ):
                result = run_analysis_runtime(
                    reviews,
                    source_type="upload",
                    llm_client=object(),  # type: ignore[arg-type]
                    paths=paths,
                    validation_time_seconds=0.03,
                    preparing_time_seconds=0.04,
                )

            metadata_text = paths.run_metadata.read_text(encoding="utf-8")
            event_text = paths.event_log.read_text(encoding="utf-8")
            metadata = json.loads(metadata_text)

        self.assertEqual(AnalysisState.COMPLETED, result.state)
        self.assertEqual(1, result.successful_reviews)
        self.assertEqual(0, result.failed_reviews)
        self.assertEqual(1, result.skipped_reviews)
        self.assertEqual("upload", metadata["source_type"])
        self.assertEqual(2, metadata["total_reviews"])
        self.assertEqual(1, metadata["retry_count"])
        self.assertEqual(1, metadata["concurrency"])
        self.assertEqual("completed", metadata["status"])
        self.assertEqual(0.03, metadata["timings"]["validation_time"])
        self.assertEqual(1.7, metadata["timings"]["llm_time"])
        self.assertEqual(0.2, metadata["timings"]["postprocessing_time"])
        self.assertEqual(0.1, metadata["timings"]["checkpoint_time"])
        self.assertNotIn(private_body, metadata_text)
        self.assertNotIn(private_body, event_text)
        self.assertTrue(re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", metadata["run_id"]))

    def test_production_runtime_rejects_unselected_parallel_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _runtime_paths(Path(temporary))
            with self.assertRaisesRegex(ValueError, "Stable mode"):
                run_analysis_runtime(
                    pd.DataFrame(),
                    source_type="paste",
                    llm_client=object(),  # type: ignore[arg-type]
                    paths=paths,
                    concurrency=2,
                )

    def test_interrupted_run_persists_interrupted_metadata_without_downstream_work(self) -> None:
        reviews = pd.DataFrame(
            {
                "source_row_id": [0],
                "review_id": ["R001"],
                "cleaned_content": ["A review body."],
                "is_analyzable": [True],
            }
        )
        analyzed = pd.DataFrame(
            {
                "source_row_id": [0],
                "review_id": ["R001"],
                "analysis_status": ["success"],
                "analysis_time_seconds": [1.0],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = _runtime_paths(Path(temporary))
            batch = BatchRunResult(
                reviews=analyzed,
                total_input_rows=2,
                previously_completed=0,
                newly_processed=1,
                attempted_reviews=1,
                is_complete=False,
                interrupted=True,
                stopped_for_failures=False,
                checkpoint_path=paths.checkpoint,
                final_output_path=None,
            )
            with (
                patch("analysis_runtime.run_checkpointed_batch", return_value=batch),
                patch("analysis_runtime.aggregate_reviews") as aggregate,
                patch("analysis_runtime.analyze_keywords") as keywords,
            ):
                result = run_analysis_runtime(
                    reviews,
                    source_type="paste",
                    llm_client=object(),  # type: ignore[arg-type]
                    paths=paths,
                )
            saved = json.loads(paths.run_metadata.read_text(encoding="utf-8"))

        self.assertEqual(AnalysisState.INTERRUPTED, result.state)
        self.assertFalse(result.downstream_completed)
        self.assertEqual("interrupted", saved["status"])
        self.assertIsNotNone(saved["completed_at"])
        aggregate.assert_not_called()
        keywords.assert_not_called()


class ProductionAdapterTests(unittest.TestCase):
    def test_large_dataset_messages_cover_100_over_100_and_500_plus(self) -> None:
        at_100 = interactive_analysis.large_dataset_message(100)
        over_100 = interactive_analysis.large_dataset_message(101)
        at_500 = interactive_analysis.large_dataset_message(500)
        over_500 = interactive_analysis.large_dataset_message(800)

        self.assertIn("100 reviews", at_100)
        self.assertIn("Large dataset", over_100)
        self.assertIn("checkpointed", over_100)
        self.assertIn("Large local analysis", at_500)
        self.assertIn("resumed", at_500)
        self.assertEqual(at_500, over_500)

    def test_upload_paste_and_amazon_adapters_call_the_same_shared_runtime(self) -> None:
        reviews = pd.DataFrame(
            {
                "review_id": ["R001"],
                "rating": [5],
                "title": ["Good"],
                "content": ["It works."],
                "date": ["2026-10-08"],
                "verified_purchase": [True],
                "cleaned_content": ["It works."],
                "is_analyzable": [True],
            }
        )

        def fake_execution(*args: object, **kwargs: object) -> RuntimeExecutionResult:
            paths = kwargs["paths"]
            batch = BatchRunResult(
                reviews=pd.DataFrame(
                    {
                        "review_id": ["R001"],
                        "analysis_status": ["success"],
                        "analysis_time_seconds": [1.0],
                    }
                ),
                total_input_rows=1,
                previously_completed=0,
                newly_processed=1,
                attempted_reviews=1,
                is_complete=True,
                interrupted=False,
                stopped_for_failures=False,
                checkpoint_path=paths.checkpoint,
                final_output_path=paths.analyzed_reviews,
            )
            return RuntimeExecutionResult(
                batch=batch,
                successful_reviews=1,
                failed_reviews=0,
                skipped_reviews=0,
                coverage=1.0,
                elapsed_seconds=1.0,
                state=AnalysisState.COMPLETED,
                run_metadata={"run_id": "shared-runtime"},
                downstream_completed=True,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upload_paths = interactive_analysis.SessionPaths.from_root(
                root / "upload" / "session" / "current"
            )
            paste_paths = interactive_analysis.SessionPaths.from_root(
                root / "paste" / "session" / "current"
            )
            amazon_paths = amazon_analysis.AmazonSessionPaths.from_root(
                root / "amazon" / "current"
            )
            with (
                patch(
                    "analysis_runtime.run_analysis_runtime",
                    side_effect=fake_execution,
                ) as shared_runtime,
                patch(
                    "interactive_analysis.validate_interactive_reviews",
                    return_value={"is_valid": True, "errors": []},
                ),
                patch("interactive_analysis.clean_reviews", return_value=reviews),
                patch("interactive_analysis.current_analysis_available", return_value=True),
                patch("amazon_analysis.amazon_import_available", return_value=True),
                patch("amazon_analysis._cleaned_import", return_value=reviews),
                patch("amazon_analysis._clear_analysis_artifacts"),
                patch("amazon_analysis.amazon_analysis_available", return_value=True),
            ):
                interactive_analysis.run_interactive_pipeline(
                    reviews,
                    llm_client=object(),  # type: ignore[arg-type]
                    paths=upload_paths,
                    source_type="upload",
                )
                interactive_analysis.run_interactive_pipeline(
                    reviews,
                    llm_client=object(),  # type: ignore[arg-type]
                    paths=paste_paths,
                    source_type="paste",
                )
                amazon_analysis.run_amazon_pipeline(
                    llm_client=object(),  # type: ignore[arg-type]
                    paths=amazon_paths,
                )

        self.assertEqual(3, shared_runtime.call_count)
        self.assertEqual(
            ["upload", "paste", "amazon"],
            [call.kwargs["source_type"] for call in shared_runtime.call_args_list],
        )
        self.assertIs(
            interactive_analysis.runtime.run_analysis_runtime,
            amazon_analysis.runtime.run_analysis_runtime,
        )


if __name__ == "__main__":
    unittest.main()
