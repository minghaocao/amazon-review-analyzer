"""Offline Step 12 business-intelligence tests (no network or model calls)."""

from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

import pandas as pd

from business_intelligence import (
    build_business_intelligence,
    build_rating_insights,
    enrich_aspect_summary,
    evidence_level,
    opportunity_level,
    sample_level,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "business_intelligence"
    / "large_summary.json"
)


def _large_fixture() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for index in range(payload["dataset_size"]):
        rows.append(
            {
                "source_row_id": index,
                "review_id": f"S{index:03d}",
                "analysis_status": "success",
                "rating": (index % 5) + 1,
                "verified_purchase": index % 3 != 0,
                "aspects": [],
                "unique_aspects": [],
            }
        )
    cursor = 0
    for spec in payload["aspect_summary"]:
        assignments = (
            ("positive", spec["positive_review_count"]),
            ("negative", spec["negative_review_count"]),
            ("neutral", spec["neutral_review_count"]),
            ("mixed", spec["mixed_review_count"]),
        )
        local = 0
        for sentiment, count in assignments:
            for _ in range(count):
                row = rows[(cursor + local) % len(rows)]
                sentiments = (
                    ("positive", "negative")
                    if sentiment == "mixed"
                    else (sentiment,)
                )
                for direction in sentiments:
                    row["aspects"].append(
                        {
                            "aspect": spec["aspect"],
                            "sentiment": direction,
                            "evidence": f"{spec['aspect']} {direction}",
                        }
                    )
                    row["unique_aspects"].append(
                        {
                            "aspect": spec["aspect"],
                            "sentiment": direction,
                            "mention_count": 1,
                        }
                    )
                local += 1
        cursor = (cursor + 13) % len(rows)
    return pd.DataFrame(rows), pd.DataFrame(payload["aspect_summary"]), payload


def _rating_frame() -> pd.DataFrame:
    def encoded(entries: list[dict]) -> str:
        return json.dumps(entries)

    return pd.DataFrame(
        [
            {
                "source_row_id": "a",
                "review_id": "DUP",
                "analysis_status": "success",
                "rating": 5,
                "unique_aspects": encoded(
                    [
                        {
                            "aspect": "Remote Control",
                            "sentiment": "negative",
                            "mention_count": 3,
                        },
                        {
                            "aspect": "Picture Quality",
                            "sentiment": "positive",
                            "mention_count": 1,
                        },
                    ]
                ),
                "aspects": "[]",
            },
            {
                "source_row_id": "b",
                "review_id": "DUP",
                "analysis_status": "success",
                "rating": 1,
                "unique_aspects": encoded(
                    [
                        {
                            "aspect": "Picture Quality",
                            "sentiment": "positive",
                            "mention_count": 2,
                        }
                    ]
                ),
                "aspects": "[]",
            },
            {
                "source_row_id": "c",
                "review_id": "C",
                "analysis_status": "success",
                "rating": 4,
                "unique_aspects": encoded(
                    [
                        {
                            "aspect": "Remote Control",
                            "sentiment": "positive",
                            "mention_count": 1,
                        },
                        {
                            "aspect": "Remote Control",
                            "sentiment": "negative",
                            "mention_count": 1,
                        },
                    ]
                ),
                "aspects": "[]",
            },
            {
                "source_row_id": "invalid",
                "review_id": "I",
                "analysis_status": "success",
                "rating": 6,
                "unique_aspects": encoded(
                    [{"aspect": "Delivery", "sentiment": "negative"}]
                ),
                "aspects": "[]",
            },
        ]
    )


class BusinessIntelligenceRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reviews, self.summary, self.fixture = _large_fixture()
        self.result = build_business_intelligence(
            self.reviews,
            self.summary,
            minimum_support=self.fixture["minimum_support"],
            generated_at="2026-01-01T00:00:00+00:00",
        )

    def test_large_fixture_contains_one_hundred_reviews(self) -> None:
        self.assertEqual(100, len(self.reviews))
        self.assertEqual("full_insight", self.result["sample_level"]["key"])

    def test_selling_point_ranking_balances_share_and_positive_rate(self) -> None:
        self.assertEqual(
            ["Picture Quality", "Ease of Use"],
            [record["aspect"] for record in self.result["selling_points"][:2]],
        )
        scores = [record["selling_score"] for record in self.result["selling_points"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_pain_point_ranking_is_product_only(self) -> None:
        names = [record["aspect"] for record in self.result["pain_points"]]
        self.assertEqual(["Remote Control", "System Performance"], names[:2])
        self.assertNotIn("Delivery", names)

    def test_opportunity_score_formula_and_level(self) -> None:
        remote = next(
            record
            for record in self.result["opportunity_ranking"]
            if record["aspect"] == "Remote Control"
        )
        self.assertAlmostEqual(30.0, remote["opportunity_score"], places=3)
        self.assertEqual("High", remote["opportunity_level"])
        self.assertEqual("Medium", opportunity_level(8))

    def test_minimum_support_excludes_small_aspect(self) -> None:
        small = self.summary.iloc[[0]].copy(deep=True)
        small.loc[:, "aspect_review_count"] = 2
        result = build_business_intelligence(
            self.reviews.iloc[:2], small, minimum_support=3
        )
        self.assertEqual([], result["selling_points"])
        self.assertEqual([], result["pain_points"])

    def test_other_is_preserved_but_excluded_from_formal_views(self) -> None:
        metric_names = {record["aspect"] for record in self.result["aspect_business_metrics"]}
        formal_names = {
            record["aspect"]
            for field in ("selling_points", "pain_points", "opportunity_ranking")
            for record in self.result[field]
        }
        self.assertIn("Other", metric_names)
        self.assertNotIn("Other", formal_names)

    def test_product_and_shopping_summaries_have_explicit_scope(self) -> None:
        self.assertEqual("PRODUCT", self.result["product_experience_summary"]["scope"])
        self.assertEqual(
            "EXPERIENCE", self.result["shopping_experience_summary"]["scope"]
        )
        self.assertEqual(
            "ALL_ASPECTS_INCLUDING_OTHER", self.result["overall_sentiment"]["scope"]
        )

    def test_review_share_and_sentiment_rates_are_preserved(self) -> None:
        picture = next(
            row for row in self.result["aspect_business_metrics"]
            if row["aspect"] == "Picture Quality"
        )
        self.assertEqual(60.0, picture["review_share"])
        self.assertEqual(80.0, picture["positive_rate"])

    def test_priority_matrix_has_four_directions_and_balanced(self) -> None:
        summary = self.summary.copy(deep=True)
        balanced = summary.iloc[0].copy()
        balanced["aspect"] = "Reliability"
        balanced["positive_rate"] = 50.0
        balanced["negative_rate"] = 50.0
        balanced["positive_review_count"] = 30
        balanced["negative_review_count"] = 30
        summary = pd.concat([summary, balanced.to_frame().T], ignore_index=True)
        result = build_business_intelligence(self.reviews, summary, minimum_support=3)
        reliability = next(
            row for row in result["priority_matrix"] if row["aspect"] == "Reliability"
        )
        self.assertEqual("Balanced / Monitor", reliability["quadrant"])

    def test_stable_alphabetical_tie_break(self) -> None:
        base = self.summary.iloc[0].copy()
        rows = []
        for aspect in ("Reliability", "Build Quality"):
            row = base.copy()
            row["aspect"] = aspect
            rows.append(row)
        result = build_business_intelligence(
            self.reviews,
            pd.DataFrame(rows),
            minimum_support=3,
        )
        self.assertEqual(
            ["Build Quality", "Reliability"],
            [record["aspect"] for record in result["selling_points"]],
        )

    def test_recommendation_types_are_restrained_templates(self) -> None:
        allowed = {
            "Marketing Opportunity",
            "Product Improvement Priority",
            "Monitor",
            "Insufficient Evidence",
        }
        self.assertTrue(
            all(record["type"] in allowed for record in self.result["recommendations"])
        )


class BusinessIntelligenceBoundaryTests(unittest.TestCase):
    def test_evidence_level_boundaries(self) -> None:
        self.assertEqual(["Low", "Medium", "Medium", "High"], [
            evidence_level(value) for value in (2, 3, 9, 10)
        ])

    def test_sample_level_boundaries(self) -> None:
        self.assertEqual(
            ["early_signals", "limited_sample", "limited_sample", "full_insight"],
            [sample_level(value)["key"] for value in (9, 10, 29, 30)],
        )

    def test_invalid_public_parameters_are_rejected(self) -> None:
        reviews, summary, _ = _large_fixture()
        for value in (True, 0, 1.5, "3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_business_intelligence(
                        reviews, summary, minimum_support=value  # type: ignore[arg-type]
                    )

    def test_input_frames_are_not_modified(self) -> None:
        reviews, summary, _ = _large_fixture()
        original_reviews = copy.deepcopy(reviews)
        original_summary = copy.deepcopy(summary)
        build_business_intelligence(reviews, summary, generated_at="fixed")
        pd.testing.assert_frame_equal(original_reviews, reviews)
        pd.testing.assert_frame_equal(original_summary, summary)

    def test_strict_json_and_fixed_timestamp_are_deterministic(self) -> None:
        reviews, summary, _ = _large_fixture()
        summary.loc[0, "avg_rating"] = math.nan
        first = build_business_intelligence(
            reviews, summary, generated_at="2026-01-01T00:00:00Z"
        )
        second = build_business_intelligence(
            reviews, summary, generated_at="2026-01-01T00:00:00Z"
        )
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)

    def test_string_false_and_nan_category_are_safe(self) -> None:
        reviews, summary, _ = _large_fixture()
        summary["include_in_business_metrics"] = summary[
            "include_in_business_metrics"
        ].astype(object)
        summary.loc[0, "include_in_business_metrics"] = "False"
        summary.loc[1, "business_category"] = math.nan
        result = build_business_intelligence(reviews, summary, minimum_support=3)
        selling = {record["aspect"] for record in result["selling_points"]}
        self.assertNotIn("Picture Quality", selling)
        remote = next(
            row for row in result["aspect_business_metrics"]
            if row["aspect"] == "Remote Control"
        )
        self.assertEqual("PRODUCT", remote["business_category"])


class RatingInsightTests(unittest.TestCase):
    def test_rating_matrix_uses_one_review_per_aspect(self) -> None:
        result = build_rating_insights(_rating_frame())
        remote = next(row for row in result["rating_matrix"] if row["aspect"] == "Remote Control")
        self.assertEqual(2, remote["review_count"])
        self.assertEqual(1, remote["rating_4"])
        self.assertEqual(1, remote["rating_5"])

    def test_hidden_complaints_track_reviews_and_mentions(self) -> None:
        result = build_rating_insights(_rating_frame())
        remote = next(row for row in result["hidden_complaints"] if row["aspect"] == "Remote Control")
        self.assertEqual(2, remote["review_count"])
        self.assertEqual(4, remote["mention_count"])

    def test_positive_signal_inside_low_rating_review(self) -> None:
        result = build_rating_insights(_rating_frame())
        picture = next(
            row for row in result["strengths_in_negative_reviews"]
            if row["aspect"] == "Picture Quality"
        )
        self.assertEqual(1, picture["review_count"])
        self.assertEqual(2, picture["mention_count"])

    def test_invalid_and_missing_ratings_are_ignored(self) -> None:
        frame = _rating_frame()
        frame.loc[len(frame)] = {
            "source_row_id": "missing",
            "review_id": "M",
            "analysis_status": "success",
            "rating": None,
            "unique_aspects": "[]",
            "aspects": "[]",
        }
        result = build_rating_insights(frame)
        self.assertEqual(3, result["rated_reviews"])
        self.assertNotIn("Delivery", result["average_rating_by_aspect"])

    def test_rating_absence_hides_rating_insights(self) -> None:
        frame = _rating_frame()
        frame["rating"] = None
        result = build_rating_insights(frame)
        self.assertFalse(result["available"])
        self.assertEqual([], result["rating_matrix"])

    def test_enriched_summary_adds_step12_columns(self) -> None:
        reviews, summary, _ = _large_fixture()
        enriched = enrich_aspect_summary(summary, reviews)
        self.assertTrue(
            {"review_share", "avg_rating", "opportunity_score", "evidence_level"}
            .issubset(enriched.columns)
        )


if __name__ == "__main__":
    unittest.main()
