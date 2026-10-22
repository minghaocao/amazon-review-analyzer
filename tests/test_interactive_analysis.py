"""Offline tests for Step 10A input normalization and session orchestration."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import interactive_analysis as interactive  # noqa: E402
from input_loader import STANDARD_COLUMNS  # noqa: E402
from llm_client import LLMConnectionError, LLMModelNotFoundError  # noqa: E402
from visualization import load_dashboard_data  # noqa: E402


class _FakeReviewClient:
    def __init__(self, *, fail_token: str = "FAIL") -> None:
        self.fail_token = fail_token
        self.calls: list[str] = []

    def check_connection(self) -> dict[str, str]:
        return {"host": "http://localhost:11434", "model": "qwen3:8b"}

    def analyze_review(
        self,
        review_text: str,
        allowed_aspects: object,
    ) -> dict[str, object]:
        del allowed_aspects
        self.calls.append(review_text)
        if self.fail_token and self.fail_token in review_text:
            raise RuntimeError("synthetic row failure")
        if "Picture quality is excellent" in review_text:
            return {
                "summary": "The picture quality is excellent.",
                "aspects": [
                    {
                        "raw_aspect": "Picture quality",
                        "aspect": "Picture Quality",
                        "sentiment": "positive",
                        "evidence": "Picture quality is excellent",
                    }
                ],
            }
        return {"summary": "A short review.", "aspects": []}

    def repair_evidence(
        self,
        review_text: str,
        invalid_aspects: object,
    ) -> dict[str, object]:
        del review_text, invalid_aspects
        return {"repairs": []}


class _InterruptingReviewClient(_FakeReviewClient):
    def __init__(self) -> None:
        super().__init__(fail_token="")
        self.attempts = 0

    def analyze_review(
        self,
        review_text: str,
        allowed_aspects: object,
    ) -> dict[str, object]:
        self.attempts += 1
        if self.attempts == 2:
            raise KeyboardInterrupt
        return super().analyze_review(review_text, allowed_aspects)


def _session_paths(base: Path) -> interactive.SessionPaths:
    return interactive.SessionPaths.from_root(base / "session" / "current")


def _pipeline_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["P001", "P002", "P003"],
            "rating": [5, 1, pd.NA],
            "title": ["Great picture", "Synthetic failure", pd.NA],
            "content": [
                "Picture quality is excellent.",
                "FAIL review",
                "",
            ],
            "date": [pd.NA, pd.NA, pd.NA],
            "verified_purchase": [True, False, pd.NA],
        },
        columns=list(STANDARD_COLUMNS),
    )


class UploadParsingTests(unittest.TestCase):
    def test_utf8_csv_is_read_without_changing_text(self) -> None:
        result = interactive.read_uploaded_csv(
            "content\nThe picture is excellent.\n".encode("utf-8")
        )
        self.assertEqual("The picture is excellent.", result.loc[0, "content"])

    def test_utf8_sig_csv_is_supported(self) -> None:
        result = interactive.read_uploaded_csv(
            "content\nGreat!\n".encode("utf-8-sig")
        )
        self.assertEqual(["content"], result.columns.tolist())

    def test_cp1252_is_the_single_bounded_fallback(self) -> None:
        result = interactive.read_uploaded_csv("content\nCafé quality\n".encode("cp1252"))
        self.assertEqual("Café quality", result.loc[0, "content"])

    def test_empty_upload_is_rejected(self) -> None:
        with self.assertRaisesRegex(interactive.InteractiveInputError, "empty"):
            interactive.read_uploaded_csv(b"")

    def test_header_only_upload_is_rejected(self) -> None:
        with self.assertRaisesRegex(interactive.InteractiveInputError, "at least one"):
            interactive.read_uploaded_csv(b"content\n")

    def test_malformed_csv_has_a_friendly_error(self) -> None:
        with self.assertRaisesRegex(
            interactive.InteractiveInputError,
            "Unable to read this CSV file",
        ):
            interactive.read_uploaded_csv(b'content,title\n"unterminated,value\n')

    def test_oversized_upload_is_rejected_before_pandas_parsing(self) -> None:
        with patch("interactive_analysis.pd.read_csv") as read_csv:
            with self.assertRaisesRegex(interactive.InteractiveInputError, "10 MB"):
                interactive.read_uploaded_csv(
                    b"x" * (interactive.MAX_UPLOAD_BYTES + 1)
                )
        read_csv.assert_not_called()

    def test_reasonable_column_names_receive_overridable_suggestions(self) -> None:
        result = interactive.suggest_column_mapping(
            ["review_text", "stars", "review_title", "verified"]
        )
        self.assertEqual("review_text", result["content"])
        self.assertEqual("stars", result["rating"])
        self.assertEqual("review_title", result["title"])
        self.assertEqual("verified", result["verified_purchase"])


class NormalizationAndPasteTests(unittest.TestCase):
    def test_schema_normalization_returns_only_six_standard_columns(self) -> None:
        source = pd.DataFrame({"body": ["Good"], "extra": ["discard me"]})
        result = interactive.normalize_uploaded_reviews(source, {"content": "body"})
        self.assertEqual(list(STANDARD_COLUMNS), result.columns.tolist())
        self.assertNotIn("extra", result.columns)

    def test_content_only_upload_is_valid_after_normalization(self) -> None:
        source = pd.DataFrame({"content": ["Good", "Bad"]})
        result = interactive.normalize_uploaded_reviews(
            source,
            {"content": "content"},
        )
        self.assertTrue(interactive.validate_interactive_reviews(result)["is_valid"])

    def test_missing_review_ids_are_generated(self) -> None:
        source = pd.DataFrame({"content": ["A", "B"]})
        result = interactive.normalize_uploaded_reviews(source, {"content": "content"})
        self.assertEqual(["UPLOAD_0001", "UPLOAD_0002"], result["review_id"].tolist())

    def test_blank_ids_are_filled_without_replacing_existing_ids(self) -> None:
        source = pd.DataFrame(
            {
                "rid": ["UPLOAD_0001", "", "0007"],
                "body": ["A", "B", "C"],
            }
        )
        result = interactive.normalize_uploaded_reviews(
            source,
            {"review_id": "rid", "content": "body"},
        )
        self.assertEqual(
            ["UPLOAD_0001", "UPLOAD_0002", "0007"],
            result["review_id"].astype(str).tolist(),
        )

    def test_optional_columns_are_added_as_missing_values(self) -> None:
        result = interactive.normalize_uploaded_reviews(
            pd.DataFrame({"body": ["A"]}),
            {"content": "body"},
        )
        for column in ("rating", "title", "date", "verified_purchase"):
            self.assertTrue(pd.isna(result.loc[0, column]))

    def test_custom_column_mapping_copies_the_selected_fields(self) -> None:
        source = pd.DataFrame(
            {
                "review_text": ["Solid picture"],
                "stars": [4],
                "headline": ["Solid"],
            }
        )
        result = interactive.normalize_uploaded_reviews(
            source,
            {
                "content": "review_text",
                "rating": "stars",
                "title": "headline",
            },
        )
        self.assertEqual("Solid picture", result.loc[0, "content"])
        self.assertEqual(4, result.loc[0, "rating"])
        self.assertEqual("Solid", result.loc[0, "title"])

    def test_content_mapping_is_required(self) -> None:
        with self.assertRaisesRegex(interactive.InteractiveInputError, "Review Content"):
            interactive.normalize_uploaded_reviews(
                pd.DataFrame({"body": ["A"]}),
                {"content": None},
            )

    def test_normalization_does_not_modify_the_uploaded_dataframe(self) -> None:
        source = pd.DataFrame({"body": [" A "], "stars": [5]})
        before = source.copy(deep=True)
        interactive.normalize_uploaded_reviews(
            source,
            {"content": "body", "rating": "stars"},
        )
        assert_frame_equal(before, source)

    def test_paste_uses_each_non_empty_line_as_one_review(self) -> None:
        result = interactive.parse_pasted_reviews("First\n\n  Second  \nThird")
        self.assertEqual(["First", "  Second  ", "Third"], result["content"].tolist())

    def test_paste_ids_are_stable_and_sequential(self) -> None:
        result = interactive.parse_pasted_reviews("A\nB")
        self.assertEqual(["PASTE_0001", "PASTE_0002"], result["review_id"].tolist())

    def test_empty_paste_is_rejected(self) -> None:
        with self.assertRaisesRegex(interactive.InteractiveInputError, "at least one"):
            interactive.parse_pasted_reviews(" \n\t\n")

    def test_all_empty_content_is_rejected_before_model_use(self) -> None:
        frame = pd.DataFrame(
            {
                "review_id": ["A", "B"],
                "rating": [pd.NA, pd.NA],
                "title": [pd.NA, pd.NA],
                "content": ["", "  "],
                "date": [pd.NA, pd.NA],
                "verified_purchase": [pd.NA, pd.NA],
            }
        )
        result = interactive.validate_interactive_reviews(frame)
        self.assertFalse(result["is_valid"])
        self.assertIn("All review content is empty.", result["errors"])

    def test_one_non_empty_content_allows_row_level_skipping_later(self) -> None:
        frame = pd.DataFrame(
            {
                "review_id": ["A", "B"],
                "rating": [pd.NA, pd.NA],
                "title": [pd.NA, pd.NA],
                "content": ["Good", ""],
                "date": [pd.NA, pd.NA],
                "verified_purchase": [pd.NA, pd.NA],
            }
        )
        self.assertTrue(interactive.validate_interactive_reviews(frame)["is_valid"])


class LimitAndDisplayHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "review_id": [f"R{index:03d}" for index in range(60)],
                "content": [f"Review {index}" for index in range(60)],
            }
        )

    def test_limit_does_not_change_a_small_input(self) -> None:
        source = self.frame.head(5)
        result, truncated = interactive.apply_interactive_limit(source)
        self.assertFalse(truncated)
        assert_frame_equal(source, result)

    def test_over_limit_input_requires_explicit_confirmation(self) -> None:
        with self.assertRaises(interactive.InteractiveLimitConfirmationRequired):
            interactive.apply_interactive_limit(self.frame)

    def test_confirmed_limit_keeps_the_first_fifty_in_order(self) -> None:
        result, truncated = interactive.apply_interactive_limit(
            self.frame,
            allow_truncate=True,
        )
        self.assertTrue(truncated)
        self.assertEqual(50, len(result))
        self.assertEqual("R049", result.iloc[-1]["review_id"])

    def test_dataset_directory_switching_is_explicit(self) -> None:
        demo = Path("/tmp/example-demo")
        current = Path("/tmp/example-current")
        self.assertEqual(
            demo,
            interactive.resolve_dashboard_directory(
                "Demo Dataset",
                demo_directory=demo,
                current_directory=current,
            ),
        )
        self.assertEqual(
            current,
            interactive.resolve_dashboard_directory(
                "Current Analysis",
                demo_directory=demo,
                current_directory=current,
            ),
        )

    def test_small_sample_warning_stops_at_thirty_successes(self) -> None:
        self.assertIsNotNone(interactive.small_sample_message(29))
        self.assertIsNone(interactive.small_sample_message(30))


class LocalModelHelperTests(unittest.TestCase):
    def test_ollama_unavailable_has_the_requested_friendly_message(self) -> None:
        message = interactive.friendly_ollama_error(
            LLMConnectionError("socket refused")
        )
        self.assertEqual(
            "Local AI is currently unavailable.\n\n"
            "Please start Ollama and try again.",
            message,
        )

    def test_missing_model_has_the_requested_friendly_message(self) -> None:
        message = interactive.friendly_ollama_error(
            LLMModelNotFoundError("missing")
        )
        self.assertEqual("Model qwen3:8b is not installed.", message)

    def test_health_check_forces_ollama_provider(self) -> None:
        fake = _FakeReviewClient()
        with patch(
            "interactive_analysis.create_review_client",
            return_value=fake,
        ) as factory:
            client, status = interactive.create_checked_ollama_client()
        factory.assert_called_once_with(provider="ollama")
        self.assertIs(fake, client)
        self.assertEqual("qwen3:8b", status["model"])

    def test_failed_health_check_propagates_before_any_pipeline_call(self) -> None:
        fake = Mock()
        fake.check_connection.side_effect = LLMConnectionError("offline")
        with patch("interactive_analysis.create_review_client", return_value=fake):
            with self.assertRaises(LLMConnectionError):
                interactive.create_checked_ollama_client()


class SessionPipelineTests(unittest.TestCase):
    def test_session_paths_keep_all_outputs_below_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _session_paths(Path(directory))
            for value in paths.__dict__.values():
                if isinstance(value, Path) and value != paths.root:
                    self.assertEqual(paths.root, value.parent)
            self.assertEqual("analyzed_reviews_checkpoint.csv", paths.checkpoint.name)
            self.assertEqual(
                "analyzed_reviews_checkpoint.meta.json",
                paths.checkpoint_meta.name,
            )

    def test_no_current_analysis_is_a_normal_false_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(
                interactive.current_analysis_available(
                    Path(directory) / "session" / "current"
                )
            )

    def test_current_analysis_pipeline_rejects_demo_output_directory(self) -> None:
        demo_paths = interactive.SessionPaths.from_root(PROJECT_ROOT / "outputs")
        reviews = interactive.parse_pasted_reviews("Picture quality is excellent.")

        with self.assertRaises(interactive.SessionSafetyError):
            interactive.run_interactive_pipeline(
                reviews,
                llm_client=_FakeReviewClient(),
                paths=demo_paths,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )

    def test_current_analysis_rejects_demo_filename_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _session_paths(root)
            frozen_demo = root / "outputs" / "analyzed_reviews.csv"
            malicious = replace(paths, analyzed_reviews=frozen_demo)
            reviews = interactive.parse_pasted_reviews(
                "Picture quality is excellent."
            )

            with self.assertRaises(interactive.SessionSafetyError):
                interactive.run_interactive_pipeline(
                    reviews,
                    llm_client=_FakeReviewClient(),
                    paths=malicious,
                    min_aspect_reviews=1,
                    min_keyword_reviews=1,
                )

            self.assertFalse(frozen_demo.exists())
            self.assertFalse(paths.root.exists())

    def test_interrupted_checkpoint_can_resume_without_reprocessing_success(self) -> None:
        reviews = interactive.parse_pasted_reviews(
            "Picture quality is excellent.\nSecond review.\nThird review."
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = _session_paths(Path(directory))
            with self.assertRaises(interactive.InteractivePipelineError):
                interactive.run_interactive_pipeline(
                    reviews,
                    llm_client=_InterruptingReviewClient(),
                    paths=paths,
                    checkpoint_interval=10,
                    min_aspect_reviews=1,
                    min_keyword_reviews=1,
                )
            self.assertTrue(interactive.interrupted_analysis_available(paths.root))
            resume_client = _FakeReviewClient(fail_token="")
            result = interactive.run_interactive_pipeline(
                None,
                llm_client=resume_client,
                paths=paths,
                resume=True,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            self.assertEqual(3, result.successful_reviews)
            self.assertEqual(
                ["Second review.", "Third review."],
                resume_client.calls,
            )
            self.assertTrue(interactive.current_analysis_available(paths.root))
            self.assertFalse(interactive.interrupted_analysis_available(paths.root))

    def test_reset_is_idempotent_and_does_not_delete_demo_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo = root / "outputs" / "business_summary.json"
            demo.parent.mkdir()
            demo.write_text("demo", encoding="utf-8")
            current = root / "session" / "current"
            current.mkdir(parents=True)
            (current / "raw_reviews.csv").write_text("session", encoding="utf-8")
            interactive.reset_current_session(current)
            interactive.reset_current_session(current)
            self.assertFalse(current.exists())
            self.assertEqual("demo", demo.read_text(encoding="utf-8"))

    def test_reset_refuses_a_broad_or_demo_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(interactive.SessionSafetyError):
                interactive.reset_current_session(root)
            with self.assertRaises(interactive.SessionSafetyError):
                interactive.reset_current_session(root / "outputs")
            with self.assertRaises(interactive.SessionSafetyError):
                interactive.reset_current_session(root / "session")

    def test_unanalyzable_html_input_does_not_erase_existing_current_results(self) -> None:
        reviews = interactive.parse_pasted_reviews("<br><br>")
        with tempfile.TemporaryDirectory() as directory:
            paths = _session_paths(Path(directory))
            paths.root.mkdir(parents=True)
            sentinel = paths.root / "existing.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(
                interactive.InteractiveInputError,
                "after cleaning",
            ):
                interactive.run_interactive_pipeline(
                    reviews,
                    llm_client=_FakeReviewClient(),
                    paths=paths,
                )
            self.assertEqual("keep", sentinel.read_text(encoding="utf-8"))

    def test_pipeline_generates_raw_checkpoint_and_all_seven_dashboard_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _session_paths(Path(directory))
            result = interactive.run_interactive_pipeline(
                _pipeline_reviews(),
                llm_client=_FakeReviewClient(),
                paths=paths,
                checkpoint_interval=1,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            self.assertTrue(paths.raw_reviews.is_file())
            self.assertTrue(paths.checkpoint.is_file())
            self.assertTrue(paths.checkpoint_meta.is_file())
            for name in interactive.DASHBOARD_OUTPUT_NAMES:
                self.assertTrue((paths.root / name).is_file(), name)
            self.assertEqual(3, result.submitted_reviews)
            self.assertEqual(1, result.successful_reviews)
            self.assertEqual(1, result.failed_reviews)
            self.assertEqual(1, result.skipped_reviews)
            self.assertAlmostEqual(1 / 3, result.coverage)

    def test_progress_callback_reports_analyzing_terminal_and_skipped_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[tuple[int, int, str, str]] = []
            interactive.run_interactive_pipeline(
                _pipeline_reviews(),
                llm_client=_FakeReviewClient(),
                paths=_session_paths(Path(directory)),
                progress_callback=lambda *event: events.append(event),
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            self.assertEqual(
                [
                    (1, 3, "P001", "analyzing"),
                    (1, 3, "P001", "success"),
                    (2, 3, "P002", "analyzing"),
                    (2, 3, "P002", "failed"),
                    (3, 3, "P003", "skipped"),
                ],
                events,
            )

    def test_progress_callback_failure_does_not_change_analysis_results(self) -> None:
        def broken_callback(*event: object) -> None:
            del event
            raise RuntimeError("UI disconnected")

        with tempfile.TemporaryDirectory() as directory:
            result = interactive.run_interactive_pipeline(
                interactive.parse_pasted_reviews("Picture quality is excellent."),
                llm_client=_FakeReviewClient(),
                paths=_session_paths(Path(directory)),
                progress_callback=broken_callback,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
        self.assertEqual(1, result.successful_reviews)
        self.assertEqual(0, result.failed_reviews)

    def test_one_failed_review_does_not_block_later_success(self) -> None:
        reviews = interactive.parse_pasted_reviews(
            "FAIL review\nPicture quality is excellent."
        )
        with tempfile.TemporaryDirectory() as directory:
            result = interactive.run_interactive_pipeline(
                reviews,
                llm_client=_FakeReviewClient(),
                paths=_session_paths(Path(directory)),
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
        self.assertEqual(1, result.failed_reviews)
        self.assertEqual(1, result.successful_reviews)

    def test_retry_failed_button_path_only_retries_failed_and_refreshes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _session_paths(Path(directory))
            initial = interactive.run_interactive_pipeline(
                _pipeline_reviews(),
                llm_client=_FakeReviewClient(),
                paths=paths,
                checkpoint_interval=1,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            self.assertEqual(1, initial.failed_reviews)
            self.assertEqual(1, interactive.failed_review_count(paths.root))

            before = pd.read_csv(
                paths.analyzed_reviews,
                dtype="string",
                keep_default_na=False,
            ).set_index("review_id")
            preserved_columns = [
                "analysis_status",
                "summary",
                "aspects",
                "unique_aspects",
                "positive_points",
                "negative_points",
                "analysis_time_seconds",
            ]
            successful_before = before.loc["P001", preserved_columns].copy()
            skipped_before = before.loc["P003", preserved_columns].copy()
            business_before = paths.business_summary.read_bytes()

            retry_client = _FakeReviewClient(fail_token="")
            retried = interactive.retry_failed_reviews(
                llm_client=retry_client,
                paths=paths,
                checkpoint_interval=1,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )

            self.assertEqual(["FAIL review"], retry_client.calls)
            self.assertEqual(1, retried.batch.newly_processed)
            self.assertEqual(1, retried.batch.attempted_reviews)
            self.assertEqual(2, retried.successful_reviews)
            self.assertEqual(0, retried.failed_reviews)
            self.assertEqual(1, retried.skipped_reviews)
            self.assertEqual(0, interactive.failed_review_count(paths.root))

            after = pd.read_csv(
                paths.analyzed_reviews,
                dtype="string",
                keep_default_na=False,
            ).set_index("review_id")
            pd.testing.assert_series_equal(
                successful_before,
                after.loc["P001", preserved_columns],
            )
            pd.testing.assert_series_equal(
                skipped_before,
                after.loc["P003", preserved_columns],
            )
            self.assertEqual("success", after.loc["P002", "analysis_status"])

            business_after = paths.business_summary.read_bytes()
            self.assertNotEqual(business_before, business_after)
            business = json.loads(business_after.decode("utf-8"))
            self.assertEqual(2, business["data_quality"]["successful_reviews"])
            self.assertEqual(0, business["data_quality"]["failed_reviews"])
            for name in interactive.DASHBOARD_OUTPUT_NAMES:
                self.assertTrue((paths.root / name).is_file(), name)
            self.assertFalse(load_dashboard_data(paths.root).issues)

    def test_pipeline_does_not_modify_input_or_an_unrelated_demo_file(self) -> None:
        reviews = _pipeline_reviews()
        before = reviews.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo = root / "demo" / "analyzed_reviews.csv"
            demo.parent.mkdir()
            demo.write_bytes(b"unchanged demo bytes")
            raw = root / "data" / "raw_reviews.csv"
            raw.parent.mkdir()
            raw.write_bytes(b"unchanged production raw bytes")
            interactive.run_interactive_pipeline(
                reviews,
                llm_client=_FakeReviewClient(),
                paths=_session_paths(root),
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            self.assertEqual(b"unchanged demo bytes", demo.read_bytes())
            self.assertEqual(b"unchanged production raw bytes", raw.read_bytes())
        assert_frame_equal(before, reviews)

    def test_current_dashboard_loader_reads_the_generated_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _session_paths(Path(directory))
            interactive.run_interactive_pipeline(
                _pipeline_reviews(),
                llm_client=_FakeReviewClient(),
                paths=paths,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            dashboard = load_dashboard_data(paths.root)
            self.assertIsNotNone(dashboard.analyzed_reviews)
            self.assertIsNotNone(dashboard.business_summary)
            self.assertIsNotNone(dashboard.keyword_summary)
            self.assertFalse(dashboard.issues)

    def test_demo_loader_remains_independent_from_current_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _session_paths(root)
            interactive.run_interactive_pipeline(
                _pipeline_reviews(),
                llm_client=_FakeReviewClient(),
                paths=paths,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            demo = root / "demo"
            demo.mkdir()
            for name in interactive.DASHBOARD_OUTPUT_NAMES:
                shutil.copy2(paths.root / name, demo / name)
            demo_data = load_dashboard_data(
                interactive.resolve_dashboard_directory(
                    "Demo Dataset",
                    demo_directory=demo,
                    current_directory=paths.root,
                )
            )
            current_data = load_dashboard_data(
                interactive.resolve_dashboard_directory(
                    "Current Analysis",
                    demo_directory=demo,
                    current_directory=paths.root,
                )
            )
            self.assertEqual(
                len(demo_data.analyzed_reviews),
                len(current_data.analyzed_reviews),
            )

    def test_keyword_and_aggregation_outputs_contain_real_offline_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _session_paths(Path(directory))
            interactive.run_interactive_pipeline(
                interactive.parse_pasted_reviews("Picture quality is excellent."),
                llm_client=_FakeReviewClient(),
                paths=paths,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            aspect_summary = pd.read_csv(paths.aspect_summary)
            business = json.loads(paths.business_summary.read_text(encoding="utf-8"))
            keyword_summary = pd.read_csv(paths.keyword_summary)
            self.assertIn("Picture Quality", aspect_summary["aspect"].tolist())
            self.assertEqual(1, business["data_quality"]["successful_reviews"])
            self.assertFalse(keyword_summary.empty)


if __name__ == "__main__":
    unittest.main()
