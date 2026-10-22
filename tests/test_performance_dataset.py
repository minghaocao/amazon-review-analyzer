"""Offline integrity tests for the Step 11 performance-only dataset."""

from __future__ import annotations

import csv
import json
import re
import unittest
from datetime import date
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "performance_benchmark_reviews.json"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
DEMO_REVIEWS_PATH = PROJECT_ROOT / "data" / "raw_reviews.csv"

REQUIRED_FIELDS = {
    "review_id",
    "rating",
    "title",
    "content",
    "date",
    "verified_purchase",
}
FORBIDDEN_EVALUATION_FIELDS = {
    "answer",
    "expected",
    "expected_aspects",
    "gold",
    "gold_aspects",
    "label",
    "labels",
    "notes",
    "tags",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _load_dataset() -> list[dict[str, Any]]:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise AssertionError("Performance dataset must be a JSON list.")
    return payload


def _old_evaluation_reviews() -> Iterable[tuple[str, str]]:
    for path in sorted(EVALUATION_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for index, item in enumerate(payload, start=1):
            if isinstance(item, dict) and isinstance(item.get("review"), str):
                yield f"{path.name}:{index}", item["review"]

    if DEMO_REVIEWS_PATH.is_file():
        with DEMO_REVIEWS_PATH.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=2):
                content = row.get("content")
                if content:
                    yield f"{DEMO_REVIEWS_PATH.name}:{index}", content


class PerformanceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reviews = _load_dataset()

    def test_contains_exactly_twenty_reviews(self) -> None:
        self.assertEqual(20, len(self.reviews))

    def test_schema_is_pipeline_compatible_and_has_no_gold_labels(self) -> None:
        for row in self.reviews:
            with self.subTest(review_id=row.get("review_id")):
                self.assertEqual(REQUIRED_FIELDS, set(row))
                self.assertFalse(FORBIDDEN_EVALUATION_FIELDS & set(row))
                self.assertRegex(row["review_id"], r"^PB\d{3}$")
                self.assertIsInstance(row["rating"], int)
                self.assertIn(row["rating"], range(1, 6))
                self.assertIsInstance(row["title"], str)
                self.assertTrue(row["title"].strip())
                self.assertIsInstance(row["content"], str)
                self.assertTrue(row["content"].strip())
                date.fromisoformat(row["date"])
                self.assertIsInstance(row["verified_purchase"], bool)

    def test_review_ids_and_content_are_unique(self) -> None:
        ids = [row["review_id"] for row in self.reviews]
        normalized = [_normalize_text(row["content"]) for row in self.reviews]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_contains_short_medium_and_long_natural_reviews(self) -> None:
        word_counts = [len(row["content"].split()) for row in self.reviews]
        self.assertTrue(any(count <= 30 for count in word_counts))
        self.assertTrue(any(31 <= count <= 40 for count in word_counts))
        self.assertTrue(any(count >= 41 for count in word_counts))

    def test_does_not_duplicate_frozen_evaluation_or_demo_reviews(self) -> None:
        benchmark_texts = {
            _normalize_text(row["content"]): row["review_id"]
            for row in self.reviews
        }
        old_texts = {
            _normalize_text(review): source
            for source, review in _old_evaluation_reviews()
        }
        overlap = sorted(set(benchmark_texts) & set(old_texts))
        self.assertEqual([], overlap)


if __name__ == "__main__":
    unittest.main()
