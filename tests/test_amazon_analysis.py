"""Offline tests for isolated Amazon import persistence and analysis."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import amazon_analysis as amazon  # noqa: E402
from input_loader import STANDARD_COLUMNS  # noqa: E402


class _FakeReviewClient:
    def __init__(self, *, fail_token: str = "FAIL") -> None:
        self.fail_token = fail_token
        self.calls: list[str] = []

    def analyze_review(
        self,
        review_text: str,
        allowed_aspects: object,
    ) -> dict[str, object]:
        del allowed_aspects
        self.calls.append(review_text)
        if self.fail_token and self.fail_token in review_text:
            raise RuntimeError("synthetic Amazon analysis failure")
        if "Picture quality" in review_text:
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
        return {"summary": "A short Amazon review.", "aspects": []}

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


def _paths(base: Path) -> amazon.AmazonSessionPaths:
    return amazon.AmazonSessionPaths.from_root(
        base / "outputs" / "amazon" / "current"
    )


def _reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["AMZ-1", "AMZ-2", "AMZ-3"],
            "rating": [5.0, 2.0, pd.NA],
            "title": ["Great picture", "Failure", pd.NA],
            "content": [
                "Picture quality is excellent.",
                "FAIL review",
                "",
            ],
            "date": ["2026-01-01", "2026-01-02", pd.NA],
            "verified_purchase": [True, False, pd.NA],
            "asin": ["B012345678"] * 3,
            "marketplace": ["amazon.com"] * 3,
        },
        columns=[*STANDARD_COLUMNS, "asin", "marketplace"],
    )


def _metadata() -> dict[str, object]:
    return {
        "asin": "B012345678",
        "marketplace": "amazon.com",
        "source_url": "https://www.amazon.com/dp/B012345678",
        "requested_limit": 5,
        "reviews_collected": 999,
        "scrape_status": "success",
        "reason": None,
        "pages_fetched": 1,
        "cookies": "must-not-be-saved",
        "authorization": "must-not-be-saved",
    }


class AmazonPathIsolationTests(unittest.TestCase):
    def test_current_paths_have_expected_isolated_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _paths(Path(directory))
            self.assertEqual("amazon", paths.root.parent.name)
            self.assertEqual("current", paths.root.name)
            self.assertEqual("scrape_metadata.json", paths.scrape_metadata.name)
            for field_name, value in paths.__dict__.items():
                if field_name != "root":
                    self.assertEqual(paths.root, value.parent)

    def test_demo_and_session_roots_are_refused(self) -> None:
        with self.assertRaises(amazon.AmazonPathSafetyError):
            amazon.AmazonSessionPaths.from_root(amazon.OUTPUTS_DIR)
        with self.assertRaises(amazon.AmazonPathSafetyError):
            amazon.AmazonSessionPaths.from_root(amazon.CURRENT_SESSION_DIR)

    def test_noncanonical_file_override_is_refused_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _paths(root)
            demo = root / "outputs" / "analyzed_reviews.csv"
            malicious = replace(paths, raw_reviews=demo)
            with self.assertRaises(amazon.AmazonPathSafetyError):
                amazon.save_amazon_import(_reviews(), _metadata(), paths=malicious)
            self.assertFalse(demo.exists())
            self.assertFalse(paths.root.exists())

    def test_reset_removes_only_amazon_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _paths(root)
            paths.root.mkdir(parents=True)
            paths.raw_reviews.write_text("amazon", encoding="utf-8")
            demo = root / "outputs" / "business_summary.json"
            demo.write_text("demo", encoding="utf-8")
            current = root / "outputs" / "session" / "current" / "raw_reviews.csv"
            current.parent.mkdir(parents=True)
            current.write_text("session", encoding="utf-8")

            amazon.reset_amazon_analysis(paths.root)
            amazon.reset_amazon_analysis(paths.root)

            self.assertFalse(paths.root.exists())
            self.assertEqual("demo", demo.read_text(encoding="utf-8"))
            self.assertEqual("session", current.read_text(encoding="utf-8"))

    def test_reset_rejects_broad_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(amazon.AmazonPathSafetyError):
                amazon.reset_amazon_analysis(Path(directory) / "outputs")

    def test_protected_demo_manifest_contains_raw_and_seven_outputs(self) -> None:
        self.assertEqual(8, len(amazon.DEMO_PROTECTED_PATHS))
        self.assertIn(amazon.RAW_REVIEWS_PATH.resolve(), amazon.DEMO_PROTECTED_PATHS)
        self.assertIn(
            amazon.ANALYZED_REVIEWS_PATH.resolve(),
            amazon.DEMO_PROTECTED_PATHS,
        )


class AmazonImportTests(unittest.TestCase):
    def test_save_and_load_round_trip_preserves_review_fields(self) -> None:
        reviews = _reviews()
        before = reviews.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            paths = _paths(Path(directory))
            saved = amazon.save_amazon_import(reviews, _metadata(), paths=paths)
            loaded = amazon.load_amazon_import(paths)
            self.assertTrue(amazon.amazon_import_available(paths.root))
            self.assertEqual(3, saved.metadata["reviews_collected"])
            self.assertEqual(3, loaded.metadata["reviews_collected"])
            self.assertEqual(
                ["AMZ-1", "AMZ-2", "AMZ-3"],
                loaded.reviews["review_id"].astype(str).tolist(),
            )
            self.assertIn("asin", loaded.reviews.columns)
            self.assertIn("marketplace", loaded.reviews.columns)
        assert_frame_equal(before, reviews)

    def test_metadata_uses_allowlist_and_never_saves_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _paths(Path(directory))
            amazon.save_amazon_import(_reviews(), _metadata(), paths=paths)
            raw = json.loads(paths.scrape_metadata.read_text(encoding="utf-8"))
            self.assertEqual("B012345678", raw["asin"])
            self.assertEqual(3, raw["reviews_collected"])
            self.assertNotIn("cookies", raw)
            self.assertNotIn("authorization", raw)

    def test_new_import_clears_stale_amazon_analysis_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _paths(root)
            paths.root.mkdir(parents=True)
            paths.business_summary.write_text("stale", encoding="utf-8")
            demo = root / "outputs" / "business_summary.json"
            demo.write_text("demo", encoding="utf-8")
            session = root / "outputs" / "session" / "current" / "keep.txt"
            session.parent.mkdir(parents=True)
            session.write_text("session", encoding="utf-8")

            amazon.save_amazon_import(_reviews(), _metadata(), paths=paths)

            self.assertFalse(paths.business_summary.exists())
            self.assertTrue(paths.raw_reviews.is_file())
            self.assertTrue(paths.scrape_metadata.is_file())
            self.assertEqual("demo", demo.read_text(encoding="utf-8"))
            self.assertEqual("session", session.read_text(encoding="utf-8"))

    def test_invalid_import_does_not_erase_existing_amazon_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _paths(Path(directory))
            paths.root.mkdir(parents=True)
            sentinel = paths.root / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(amazon.AmazonImportError):
                amazon.save_amazon_import(
                    pd.DataFrame({"content": ["missing required columns"]}),
                    _metadata(),
                    paths=paths,
                )
            self.assertEqual("keep", sentinel.read_text(encoding="utf-8"))

    def test_missing_or_corrupt_import_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _paths(Path(directory))
            with self.assertRaisesRegex(amazon.AmazonImportError, "No saved"):
                amazon.load_amazon_import(paths)
            paths.root.mkdir(parents=True)
            _reviews().to_csv(paths.raw_reviews, index=False)
            paths.scrape_metadata.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(amazon.AmazonImportError, "JSON object"):
                amazon.load_amazon_import(paths)

    def test_browser_import_writes_safe_separate_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _paths(Path(directory))
            browser_metadata = {
                "marketplace": "amazon.com",
                "asin": "B012345678",
                "product_url": "https://www.amazon.com/dp/B012345678",
                "reviews_url": (
                    "https://www.amazon.com/product-reviews/B012345678"
                ),
                "review_count": 999,
                "retrieved_review_count": 5,
                "unique_review_count": 999,
                "target_review_count": 10,
                "availability_status": "access_limited",
                "availability_reason": "Limited current session.",
                "stop_reason": "amazon_access_limited",
                "pages_visited": 2,
                "challenge_encountered": True,
                "login_required": False,
                "status": "success",
                "started_at": "2026-09-28T01:00:00+00:00",
                "completed_at": "2026-09-28T01:01:00+00:00",
                "cookies": "must-not-be-written",
                "password": "must-not-be-written",
                "token": "must-not-be-written",
            }

            amazon.save_browser_amazon_import(
                _reviews(),
                browser_metadata,
                paths=paths,
            )

            self.assertTrue(paths.raw_reviews.is_file())
            self.assertTrue(paths.scrape_metadata.is_file())
            self.assertTrue(paths.browser_import_metadata.is_file())
            saved = json.loads(
                paths.browser_import_metadata.read_text(encoding="utf-8")
            )
            self.assertEqual(3, saved["review_count"])
            self.assertEqual(5, saved["retrieved_review_count"])
            self.assertEqual(3, saved["unique_review_count"])
            self.assertIsNone(saved["target_review_count"])
            self.assertEqual("access_limited", saved["availability_status"])
            self.assertEqual("amazon_access_limited", saved["stop_reason"])
            self.assertEqual(2, saved["pages_visited"])
            self.assertTrue(saved["challenge_encountered"])
            self.assertFalse(saved["login_required"])
            self.assertNotIn("cookies", saved)
            self.assertNotIn("password", saved)
            self.assertNotIn("token", saved)
            self.assertEqual(3, len(amazon.load_amazon_import(paths).reviews))
            self.assertEqual(saved, amazon.load_browser_import_metadata(paths))

            compatibility = amazon.load_amazon_import(paths).metadata
            self.assertIsNone(compatibility["requested_limit"])
            self.assertIsNone(compatibility["target_review_count"])
            self.assertEqual(3, compatibility["unique_review_count"])
            self.assertEqual(5, compatibility["retrieved_review_count"])

    def test_invalid_browser_metadata_does_not_erase_existing_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _paths(Path(directory))
            amazon.save_amazon_import(_reviews(), _metadata(), paths=paths)
            before = paths.raw_reviews.read_bytes()

            with self.assertRaisesRegex(
                amazon.AmazonImportError,
                "missing reviews_url",
            ):
                amazon.save_browser_amazon_import(
                    _reviews(),
                    {
                        "marketplace": "amazon.com",
                        "asin": "B012345678",
                        "product_url": "https://www.amazon.com/dp/B012345678",
                        "status": "success",
                    },
                    paths=paths,
                )

            self.assertEqual(before, paths.raw_reviews.read_bytes())


class AmazonPipelineTests(unittest.TestCase):
    def _save(self, base: Path) -> amazon.AmazonSessionPaths:
        paths = _paths(base)
        amazon.save_amazon_import(_reviews(), _metadata(), paths=paths)
        return paths

    def test_pipeline_reuses_checkpoint_aggregation_keywords_and_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._save(Path(directory))
            result = amazon.run_amazon_pipeline(
                llm_client=_FakeReviewClient(),
                paths=paths,
                checkpoint_interval=1,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            self.assertEqual(3, result.submitted_reviews)
            self.assertEqual(1, result.successful_reviews)
            self.assertEqual(1, result.failed_reviews)
            self.assertEqual(1, result.skipped_reviews)
            self.assertTrue(paths.checkpoint.is_file())
            self.assertTrue(paths.checkpoint_meta.is_file())
            self.assertTrue(amazon.amazon_analysis_available(paths.root))
            dashboard = amazon.load_amazon_dashboard(paths.root)
            self.assertFalse(dashboard.issues)
            self.assertEqual(3, len(dashboard.analyzed_reviews))

    def test_interrupted_pipeline_resumes_without_reprocessing_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._save(Path(directory))
            with self.assertRaises(amazon.AmazonPipelineError):
                amazon.run_amazon_pipeline(
                    llm_client=_InterruptingReviewClient(),
                    paths=paths,
                    checkpoint_interval=10,
                    min_aspect_reviews=1,
                    min_keyword_reviews=1,
                )
            self.assertTrue(
                amazon.interrupted_amazon_analysis_available(paths.root)
            )
            client = _FakeReviewClient(fail_token="")
            result = amazon.run_amazon_pipeline(
                llm_client=client,
                paths=paths,
                resume=True,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            self.assertEqual(["FAIL review"], client.calls)
            self.assertEqual(2, result.successful_reviews)
            self.assertEqual(1, result.skipped_reviews)
            self.assertTrue(amazon.amazon_analysis_available(paths.root))

    def test_retry_failed_only_reprocesses_failed_and_rebuilds_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._save(Path(directory))
            initial = amazon.run_amazon_pipeline(
                llm_client=_FakeReviewClient(),
                paths=paths,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )
            self.assertEqual(1, initial.failed_reviews)
            before = pd.read_csv(
                paths.analyzed_reviews,
                dtype="string",
                keep_default_na=False,
            ).set_index("review_id")
            success_before = before.loc["AMZ-1"].copy()
            skipped_before = before.loc["AMZ-3"].copy()
            business_before = paths.business_summary.read_bytes()

            client = _FakeReviewClient(fail_token="")
            result = amazon.retry_failed_amazon_reviews(
                llm_client=client,
                paths=paths,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )

            self.assertEqual(["FAIL review"], client.calls)
            self.assertEqual(2, result.successful_reviews)
            self.assertEqual(0, result.failed_reviews)
            self.assertEqual(0, amazon.failed_amazon_review_count(paths.root))
            after = pd.read_csv(
                paths.analyzed_reviews,
                dtype="string",
                keep_default_na=False,
            ).set_index("review_id")
            pd.testing.assert_series_equal(success_before, after.loc["AMZ-1"])
            pd.testing.assert_series_equal(skipped_before, after.loc["AMZ-3"])
            self.assertNotEqual(business_before, paths.business_summary.read_bytes())
            summary = json.loads(paths.business_summary.read_text(encoding="utf-8"))
            self.assertEqual(2, summary["data_quality"]["successful_reviews"])

    def test_pipeline_never_modifies_demo_or_upload_session_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._save(root)
            demo = root / "outputs" / "analyzed_reviews.csv"
            demo.write_bytes(b"frozen-demo")
            current = root / "outputs" / "session" / "current" / "raw_reviews.csv"
            current.parent.mkdir(parents=True)
            current.write_bytes(b"upload-session")

            amazon.run_amazon_pipeline(
                llm_client=_FakeReviewClient(),
                paths=paths,
                min_aspect_reviews=1,
                min_keyword_reviews=1,
            )

            self.assertEqual(b"frozen-demo", demo.read_bytes())
            self.assertEqual(b"upload-session", current.read_bytes())

    def test_pipeline_requires_saved_import_and_never_calls_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = _FakeReviewClient()
            with self.assertRaisesRegex(amazon.AmazonImportError, "Import Amazon"):
                amazon.run_amazon_pipeline(
                    llm_client=client,
                    paths=_paths(Path(directory)),
                )
            self.assertEqual([], client.calls)


if __name__ == "__main__":
    unittest.main()
