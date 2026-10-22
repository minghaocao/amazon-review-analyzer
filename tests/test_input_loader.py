"""Tests for CSV loading and Step 2 data validation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input_loader import STANDARD_COLUMNS, load_reviews, validate_reviews  # noqa: E402


class InputLoaderTests(unittest.TestCase):
    def _write_csv(self, data: pd.DataFrame, directory: str) -> Path:
        csv_path = Path(directory) / "reviews.csv"
        data.to_csv(csv_path, index=False)
        return csv_path

    def test_normal_csv_loads_and_uses_standard_columns(self) -> None:
        source = pd.DataFrame(
            {
                "review_id": ["001"],
                "rating": [5],
                "content": ["  Original review text stays unchanged.  "],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            reviews = load_reviews(self._write_csv(source, temporary_directory))

        self.assertIsInstance(reviews, pd.DataFrame)
        self.assertEqual(list(STANDARD_COLUMNS), list(reviews.columns))
        self.assertEqual("001", reviews.loc[0, "review_id"])
        self.assertEqual(5, reviews.loc[0, "rating"])
        self.assertEqual(source.loc[0, "content"], reviews.loc[0, "content"])

    def test_missing_required_columns_are_fatal(self) -> None:
        complete = pd.DataFrame(
            {"review_id": ["R001"], "rating": [5], "content": ["Great"]}
        )
        for column in ("review_id", "rating", "content"):
            with self.subTest(column=column):
                result = validate_reviews(complete.drop(columns=column))
                self.assertFalse(result["is_valid"])
                self.assertIn(column, " ".join(result["errors"]))

    def test_empty_dataframe_is_fatal(self) -> None:
        result = validate_reviews(pd.DataFrame(columns=STANDARD_COLUMNS))
        self.assertFalse(result["is_valid"])
        self.assertTrue(any("no review rows" in error for error in result["errors"]))

    def test_empty_content_is_detected(self) -> None:
        reviews = pd.DataFrame(
            {
                "review_id": ["R001", "R002", "R003"],
                "rating": [5, 4, 3],
                "title": ["A", "B", "C"],
                "content": [None, "", "   "],
            }
        )
        result = validate_reviews(reviews)
        self.assertEqual(3, result["missing_content"])
        self.assertTrue(result["is_valid"])

    def test_duplicate_review_id_is_detected(self) -> None:
        reviews = pd.DataFrame(
            {"review_id": ["R001", "R001"], "rating": [5, 4], "content": ["A", "B"]}
        )
        result = validate_reviews(reviews)
        self.assertEqual(1, result["duplicate_review_ids"])

    def test_rating_above_five_is_detected(self) -> None:
        reviews = pd.DataFrame(
            {"review_id": ["R001"], "rating": [6], "content": ["A"]}
        )
        result = validate_reviews(reviews)
        self.assertEqual(1, result["ratings_above_range"])
        self.assertEqual(1, result["invalid_ratings"])

    def test_rating_below_one_is_detected(self) -> None:
        reviews = pd.DataFrame(
            {"review_id": ["R001"], "rating": [0], "content": ["A"]}
        )
        result = validate_reviews(reviews)
        self.assertEqual(1, result["ratings_below_range"])
        self.assertEqual(1, result["invalid_ratings"])

    def test_non_numeric_rating_is_preserved_and_detected(self) -> None:
        source = pd.DataFrame(
            {"review_id": ["R001"], "rating": ["five"], "content": ["A"]}
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            reviews = load_reviews(self._write_csv(source, temporary_directory))

        self.assertEqual("five", reviews.loc[0, "rating"])
        result = validate_reviews(reviews)
        self.assertEqual(1, result["non_numeric_ratings"])
        self.assertEqual(1, result["invalid_ratings"])

    def test_empty_title_is_warning_not_fatal(self) -> None:
        reviews = pd.DataFrame(
            {"review_id": ["R001"], "rating": [5], "title": [None], "content": ["A"]}
        )
        result = validate_reviews(reviews)
        self.assertEqual(1, result["missing_titles"])
        self.assertTrue(result["is_valid"])

    def test_validation_does_not_modify_dataframe(self) -> None:
        reviews = pd.DataFrame(
            {
                "review_id": [" R001 "],
                "rating": [5],
                "title": ["  Title  "],
                "content": ["  Content  "],
            }
        )
        original = reviews.copy(deep=True)
        validate_reviews(reviews)
        pd.testing.assert_frame_equal(original, reviews)


if __name__ == "__main__":
    unittest.main()
