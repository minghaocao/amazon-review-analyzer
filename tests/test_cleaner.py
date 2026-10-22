"""Tests for basic review text cleaning and analyzability rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleaner import clean_reviews, clean_text, summarize_cleaning  # noqa: E402


class CleanTextTests(unittest.TestCase):
    def test_normal_english_review_is_unchanged(self) -> None:
        text = "Delivery was terrible but the TV picture is excellent."
        self.assertEqual(text, clean_text(text))

    def test_surrounding_spaces_are_removed(self) -> None:
        self.assertEqual(
            "The picture quality is AMAZING!!!",
            clean_text("  The picture quality is AMAZING!!!   "),
        )

    def test_consecutive_spaces_are_collapsed(self) -> None:
        self.assertEqual("Great picture!", clean_text("Great   picture!"))

    def test_tabs_are_replaced_with_single_spaces(self) -> None:
        self.assertEqual("Great picture!", clean_text("Great\t\tpicture!"))

    def test_newlines_are_replaced_with_single_spaces(self) -> None:
        self.assertEqual(
            "Great picture! Very good.",
            clean_text("  Great picture!\n\nVery good. "),
        )

    def test_html_tags_are_removed_without_joining_blocks(self) -> None:
        self.assertEqual(
            "Great picture Good sound",
            clean_text("<p>Great picture</p><br>Good sound"),
        )

    def test_html_entities_are_decoded(self) -> None:
        self.assertEqual(
            "Tom & Jerry <3 TV > old model",
            clean_text("Tom &amp; Jerry &lt;3 TV &gt; old model"),
        )

    def test_unicode_spaces_are_normalized(self) -> None:
        self.assertEqual(
            "Great picture quality",
            clean_text("Great\u00a0\u2003picture\u200bquality"),
        )

    def test_none_returns_empty_string(self) -> None:
        self.assertEqual("", clean_text(None))

    def test_nan_returns_empty_string(self) -> None:
        self.assertEqual("", clean_text(float("nan")))

    def test_not_is_preserved(self) -> None:
        self.assertEqual("NOT good", clean_text("NOT good"))

    def test_numbers_are_preserved(self) -> None:
        self.assertEqual(
            "Battery lasts only 2 hours!!!",
            clean_text("Battery lasts only 2 hours!!!"),
        )

    def test_complex_review_keeps_all_semantic_details(self) -> None:
        text = (
            "The picture quality is AMAZING!!! But the menu is NOT fast, "
            "and delivery took 7 days."
        )
        cleaned = clean_text(text)

        self.assertEqual(text, cleaned)
        for detail in ("AMAZING", "NOT", "7 days", "picture quality", "menu", "delivery"):
            with self.subTest(detail=detail):
                self.assertIn(detail, cleaned)


class CleanReviewsTests(unittest.TestCase):
    @staticmethod
    def _reviews(content: object, title: object = "Title") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "review_id": ["R001"],
                "rating": [5],
                "title": [title],
                "content": [content],
            }
        )

    def test_empty_content_is_not_analyzable(self) -> None:
        cleaned = clean_reviews(self._reviews(""))
        self.assertFalse(cleaned.loc[0, "is_analyzable"])
        self.assertEqual("missing_content", cleaned.loc[0, "skip_reason"])

    def test_whitespace_only_content_is_not_analyzable(self) -> None:
        cleaned = clean_reviews(self._reviews("     "))
        self.assertFalse(cleaned.loc[0, "is_analyzable"])
        self.assertEqual("empty_after_cleaning", cleaned.loc[0, "skip_reason"])

    def test_html_only_content_is_not_analyzable(self) -> None:
        cleaned = clean_reviews(self._reviews("<br><br>"))
        self.assertFalse(cleaned.loc[0, "is_analyzable"])
        self.assertEqual("empty_after_cleaning", cleaned.loc[0, "skip_reason"])

    def test_short_content_is_still_analyzable(self) -> None:
        cleaned = clean_reviews(self._reviews("Great!"))
        self.assertTrue(cleaned.loc[0, "is_analyzable"])
        self.assertEqual("", cleaned.loc[0, "skip_reason"])

    def test_none_content_uses_missing_content_reason(self) -> None:
        cleaned = clean_reviews(self._reviews(None))
        self.assertFalse(cleaned.loc[0, "is_analyzable"])
        self.assertEqual("missing_content", cleaned.loc[0, "skip_reason"])

    def test_empty_title_does_not_skip_review(self) -> None:
        cleaned = clean_reviews(self._reviews("Great!", title=None))
        self.assertEqual("", cleaned.loc[0, "cleaned_title"])
        self.assertTrue(cleaned.loc[0, "is_analyzable"])

    def test_original_content_is_preserved_exactly(self) -> None:
        original = "  <p>The picture is AMAZING!!!</p>   "
        cleaned = clean_reviews(self._reviews(original))
        self.assertEqual(original, cleaned.loc[0, "original_content"])
        self.assertEqual("The picture is AMAZING!!!", cleaned.loc[0, "cleaned_content"])

    def test_clean_reviews_does_not_modify_input_dataframe(self) -> None:
        reviews = self._reviews("  NOT bad after 2 days!  ", title="  Model X  ")
        original = reviews.copy(deep=True)
        clean_reviews(reviews)
        pd.testing.assert_frame_equal(original, reviews)

    def test_cleaning_summary_counts_skip_reasons(self) -> None:
        reviews = pd.DataFrame(
            {
                "review_id": ["R001", "R002", "R003"],
                "rating": [5, 3, 4],
                "title": ["A", "B", "C"],
                "content": [None, "   ", "Great!"],
            }
        )
        summary = summarize_cleaning(clean_reviews(reviews))

        self.assertEqual(3, summary["total_reviews"])
        self.assertEqual(1, summary["analyzable_reviews"])
        self.assertEqual(2, summary["skipped_reviews"])
        self.assertEqual(1, summary["skip_reasons"]["missing_content"])
        self.assertEqual(1, summary["skip_reasons"]["empty_after_cleaning"])


if __name__ == "__main__":
    unittest.main()
