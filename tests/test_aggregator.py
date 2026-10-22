"""Offline synthetic tests for Step 6 product-level aggregation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregator import (  # noqa: E402
    ASPECT_SUMMARY_COLUMNS,
    AggregationError,
    aggregate_reviews,
    build_aspect_summary,
    build_business_summary,
    build_review_aspect_table,
    collect_aspect_evidence,
    load_analyzed_reviews,
    save_aggregation_outputs,
)
from aspect_taxonomy import business_category_for_aspect  # noqa: E402


def _unique(
    aspect: str,
    sentiment: str,
    mention_count: int = 1,
    include: bool = True,
) -> dict[str, Any]:
    return {
        "aspect": aspect,
        "sentiment": sentiment,
        "mention_count": mention_count,
        "include_in_business_metrics": include,
    }


def _aspect(
    aspect: str,
    sentiment: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "raw_aspect": aspect,
        "aspect": aspect,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _row(
    source_row_id: int,
    review_id: str,
    *,
    status: str = "success",
    unique_aspects: object | None = None,
    aspects: object | None = None,
) -> dict[str, Any]:
    unique_value = [] if unique_aspects is None else unique_aspects
    aspect_value = [] if aspects is None else aspects
    return {
        "source_row_id": source_row_id,
        "review_id": review_id,
        "analysis_status": status,
        "unique_aspects": (
            json.dumps(unique_value)
            if not isinstance(unique_value, str)
            else unique_value
        ),
        "aspects": (
            json.dumps(aspect_value)
            if not isinstance(aspect_value, str)
            else aspect_value
        ),
    }


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(
                0,
                "R001",
                unique_aspects=[
                    _unique("Picture Quality", "positive", 3),
                    _unique("Sound Quality", "negative"),
                ],
                aspects=[
                    _aspect("Picture Quality", "positive", "The picture is sharp"),
                    _aspect("Picture Quality", "positive", "The picture is sharp"),
                    _aspect("Picture Quality", "positive", "colors look accurate"),
                    _aspect("Sound Quality", "negative", "speakers distort"),
                ],
            ),
            _row(
                1,
                "R002",
                unique_aspects=[
                    _unique("Picture Quality", "negative", 2),
                    _unique("Remote Control", "negative"),
                    _unique("Other", "positive", include=False),
                ],
                aspects=[
                    _aspect("Picture Quality", "negative", "The picture is sharp"),
                    _aspect("Picture Quality", "negative", "screen flickers"),
                    _aspect("Remote Control", "negative", "remote stopped working"),
                    _aspect("Other", "positive", "works well"),
                ],
            ),
            _row(
                2,
                "R003",
                unique_aspects=[
                    _unique("Picture Quality", "positive"),
                    _unique("Picture Quality", "negative"),
                    _unique("Sound Quality", "neutral"),
                ],
                aspects=[
                    _aspect("Picture Quality", "positive", "picture is good"),
                    _aspect("Picture Quality", "negative", "not perfect"),
                    _aspect("Sound Quality", "neutral", "sound is average"),
                ],
            ),
            _row(
                3,
                "R004",
                unique_aspects=[
                    _unique("Picture Quality", "neutral"),
                    _unique("Sound Quality", "positive", 2),
                ],
                aspects=[
                    _aspect("Picture Quality", "neutral", "picture is acceptable"),
                    _aspect("Sound Quality", "positive", "clear dialogue"),
                    _aspect("Sound Quality", "positive", "good bass"),
                ],
            ),
            _row(
                4,
                "R001",
                unique_aspects=[_unique("Price / Value", "positive")],
                aspects=[_aspect("Price / Value", "positive", "worth the price")],
            ),
            _row(
                5,
                "R005",
                status="failed",
                unique_aspects=[_unique("Picture Quality", "positive", 99)],
                aspects=[_aspect("Picture Quality", "positive", "fabricated failed")],
            ),
            _row(
                6,
                "R006",
                status="skipped",
                unique_aspects=[_unique("Picture Quality", "negative", 99)],
                aspects=[_aspect("Picture Quality", "negative", "fabricated skipped")],
            ),
        ]
    )


def _summary_row(
    aspect: str,
    reviews: int,
    *,
    positive: int = 0,
    negative: int = 0,
    neutral: int = 0,
    mixed: int | None = None,
    include: bool = True,
) -> dict[str, Any]:
    if mixed is None:
        mixed = reviews - positive - negative - neutral
    return {
        "aspect": aspect,
        "business_category": business_category_for_aspect(aspect),
        "aspect_review_count": reviews,
        "mention_count": reviews,
        "aspect_share_of_reviews": float(reviews),
        "positive_review_count": positive,
        "negative_review_count": negative,
        "neutral_review_count": neutral,
        "mixed_review_count": mixed,
        "positive_rate": positive / reviews * 100,
        "negative_rate": negative / reviews * 100,
        "neutral_rate": neutral / reviews * 100,
        "mixed_rate": mixed / reviews * 100,
        "include_in_business_metrics": include,
    }


def _business_from_rows(
    rows: list[dict[str, Any]],
    *,
    minimum_reviews: int = 1,
) -> dict[str, Any]:
    summary = pd.DataFrame(rows, columns=ASPECT_SUMMARY_COLUMNS)
    total_reviews = max(
        (int(row["aspect_review_count"]) for row in rows),
        default=0,
    )
    return build_business_summary(
        pd.DataFrame({"analysis_status": ["success"] * total_reviews}),
        summary,
        [],
        min_aspect_reviews=minimum_reviews,
        top_n=10,
    )


class ReviewAspectTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reviews = _fixture()
        self.original = self.reviews.copy(deep=True)
        self.result = aggregate_reviews(self.reviews, min_aspect_reviews=1)
        self.table = self.result.review_aspect_table
        self.summary = self.result.aspect_summary.set_index("aspect")

    def test_unique_aspects_expand_to_review_aspect_rows(self) -> None:
        self.assertEqual(10, len(self.table))

    def test_same_review_aspect_and_sentiment_is_collapsed(self) -> None:
        reviews = pd.DataFrame(
            [
                _row(
                    0,
                    "R001",
                    unique_aspects=[
                        _unique("Picture Quality", "positive", 1),
                        _unique("Picture Quality", "positive", 2),
                    ],
                )
            ]
        )
        table = build_review_aspect_table(reviews)
        self.assertEqual(1, len(table))
        self.assertEqual(3, table.iloc[0]["mention_count"])

    def test_different_sentiments_derive_mixed(self) -> None:
        row = self.table.loc[
            self.table["source_row_id"].eq(2)
            & self.table["aspect"].eq("Picture Quality")
        ].iloc[0]
        self.assertEqual("mixed", row["review_sentiment"])
        self.assertEqual(2, row["mention_count"])

    def test_positive_and_neutral_also_derive_mixed(self) -> None:
        reviews = pd.DataFrame(
            [
                _row(
                    0,
                    "R001",
                    unique_aspects=[
                        _unique("Design", "positive"),
                        _unique("Design", "neutral"),
                    ],
                )
            ]
        )
        self.assertEqual(
            "mixed",
            build_review_aspect_table(reviews).iloc[0]["review_sentiment"],
        )

    def test_single_base_sentiments_are_preserved(self) -> None:
        sentiments = set(self.table["review_sentiment"])
        self.assertTrue({"positive", "negative", "neutral", "mixed"}.issubset(sentiments))

    def test_duplicate_review_id_uses_distinct_source_rows(self) -> None:
        rows = self.table.loc[self.table["review_id"].eq("R001")]
        self.assertEqual({0, 4}, set(rows["source_row_id"]))

    def test_mention_count_accumulates(self) -> None:
        self.assertEqual(8, self.summary.loc["Picture Quality", "mention_count"])

    def test_aspect_review_count_counts_reviews(self) -> None:
        self.assertEqual(4, self.summary.loc["Picture Quality", "aspect_review_count"])

    def test_positive_review_count_is_correct(self) -> None:
        self.assertEqual(1, self.summary.loc["Picture Quality", "positive_review_count"])

    def test_negative_review_count_is_correct(self) -> None:
        self.assertEqual(1, self.summary.loc["Picture Quality", "negative_review_count"])

    def test_neutral_review_count_is_correct(self) -> None:
        self.assertEqual(1, self.summary.loc["Picture Quality", "neutral_review_count"])

    def test_mixed_review_count_is_correct(self) -> None:
        self.assertEqual(1, self.summary.loc["Picture Quality", "mixed_review_count"])

    def test_sentiment_counts_reconcile_to_aspect_review_count(self) -> None:
        for _, row in self.result.aspect_summary.iterrows():
            total = sum(
                int(row[f"{sentiment}_review_count"])
                for sentiment in ("positive", "negative", "neutral", "mixed")
            )
            self.assertEqual(int(row["aspect_review_count"]), total)

    def test_sentiment_rates_use_aspect_review_count(self) -> None:
        for sentiment in ("positive", "negative", "neutral", "mixed"):
            self.assertEqual(25.0, self.summary.loc["Picture Quality", f"{sentiment}_rate"])

    def test_sentiment_rates_sum_to_one_hundred(self) -> None:
        sound = self.summary.loc["Sound Quality"]
        total = sum(float(sound[f"{sentiment}_rate"]) for sentiment in ("positive", "negative", "neutral", "mixed"))
        self.assertAlmostEqual(100.0, total)

    def test_aspect_share_uses_all_successful_reviews(self) -> None:
        self.assertEqual(80.0, self.summary.loc["Picture Quality", "aspect_share_of_reviews"])

    def test_failed_review_does_not_enter_statistics(self) -> None:
        self.assertEqual(8, self.summary.loc["Picture Quality", "mention_count"])
        self.assertNotIn("fabricated failed", json.dumps(self.result.evidence))

    def test_skipped_review_does_not_enter_statistics(self) -> None:
        self.assertNotIn("fabricated skipped", json.dumps(self.result.evidence))

    def test_other_remains_in_full_summary(self) -> None:
        self.assertIn("Other", self.summary.index)
        self.assertFalse(bool(self.summary.loc["Other", "include_in_business_metrics"]))
        self.assertEqual("OTHER", self.summary.loc["Other", "business_category"])

    def test_business_category_is_added_without_losing_aspect_rows(self) -> None:
        self.assertEqual(
            set(self.table["aspect"]),
            set(self.result.aspect_summary["aspect"]),
        )
        picture = self.table.loc[self.table["aspect"].eq("Picture Quality")]
        self.assertEqual({"PRODUCT"}, set(picture["business_category"]))

    def test_other_is_excluded_from_all_business_rankings(self) -> None:
        business = self.result.business_summary
        for key in (
            "top_customer_concerns",
            "product_selling_points",
            "product_pain_points",
        ):
            self.assertNotIn("Other", {item["aspect"] for item in business[key]})
        for records in business["shopping_experience"].values():
            self.assertNotIn("Other", {item["aspect"] for item in records})

    def test_customer_concerns_are_sorted_by_review_count(self) -> None:
        concerns = self.result.business_summary["top_customer_concerns"]
        counts = [item["aspect_review_count"] for item in concerns]
        self.assertEqual(sorted(counts, reverse=True), counts)
        self.assertEqual("Picture Quality", concerns[0]["aspect"])

    def test_mixed_is_not_counted_as_positive_or_negative(self) -> None:
        picture = self.summary.loc["Picture Quality"]
        self.assertEqual(1, picture["mixed_review_count"])
        self.assertEqual(1, picture["positive_review_count"])
        self.assertEqual(1, picture["negative_review_count"])

    def test_input_dataframe_is_not_modified(self) -> None:
        pd.testing.assert_frame_equal(self.original, self.reviews)


class RankingTests(unittest.TestCase):
    def test_selling_points_are_not_sorted_only_by_positive_rate(self) -> None:
        summary = pd.DataFrame(
            [
                _summary_row("Picture Quality", 5, positive=4, negative=1),
                _summary_row("Design", 1, positive=1),
            ],
            columns=ASPECT_SUMMARY_COLUMNS,
        )
        business = build_business_summary(
            pd.DataFrame({"analysis_status": ["success"] * 6}),
            summary,
            [],
            min_aspect_reviews=1,
            top_n=5,
        )
        self.assertEqual(
            "Picture Quality",
            business["product_selling_points"][0]["aspect"],
        )

    def test_pain_points_are_not_sorted_only_by_negative_rate(self) -> None:
        summary = pd.DataFrame(
            [
                _summary_row("Build Quality", 5, positive=2, negative=3),
                _summary_row("Remote Control", 1, negative=1),
            ],
            columns=ASPECT_SUMMARY_COLUMNS,
        )
        business = build_business_summary(
            pd.DataFrame({"analysis_status": ["success"] * 6}),
            summary,
            [],
            min_aspect_reviews=1,
            top_n=5,
        )
        self.assertEqual(
            "Build Quality",
            business["product_pain_points"][0]["aspect"],
        )

    def test_minimum_sample_threshold_applies_to_selling_and_pain(self) -> None:
        result = aggregate_reviews(_fixture(), min_aspect_reviews=3)
        selling = {
            item["aspect"]
            for item in result.business_summary["product_selling_points"]
        }
        pain = {
            item["aspect"]
            for item in result.business_summary["product_pain_points"]
        }
        self.assertNotIn("Price / Value", selling)
        self.assertNotIn("Remote Control", pain)

    def test_minimum_sample_threshold_can_be_lowered(self) -> None:
        result = aggregate_reviews(_fixture(), min_aspect_reviews=1)
        selling = result.business_summary["product_selling_points"]
        self.assertIn("Price / Value", {item["aspect"] for item in selling})
        price = next(item for item in selling if item["aspect"] == "Price / Value")
        self.assertEqual(1, price["aspect_review_count"])


class BusinessCategoryRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.business = _business_from_rows(
            [
                _summary_row("Picture Quality", 5, positive=4, negative=1),
                _summary_row("Build Quality", 4, positive=1, negative=3),
                _summary_row("Sound Quality", 4, positive=2, negative=2),
                _summary_row("Delivery", 3, positive=2, negative=1),
                _summary_row("Customer Service", 3, positive=1, negative=2),
                _summary_row("Packaging", 2, positive=1, negative=1),
                _summary_row("Other", 8, positive=8, include=False),
            ],
            minimum_reviews=3,
        )

    def test_experience_does_not_enter_product_selling_points(self) -> None:
        aspects = {
            item["aspect"] for item in self.business["product_selling_points"]
        }
        self.assertEqual({"Picture Quality"}, aspects)
        self.assertNotIn("Delivery", aspects)

    def test_experience_does_not_enter_product_pain_points(self) -> None:
        aspects = {
            item["aspect"] for item in self.business["product_pain_points"]
        }
        self.assertEqual({"Build Quality"}, aspects)
        self.assertNotIn("Customer Service", aspects)

    def test_experience_remains_in_customer_concerns(self) -> None:
        concerns = {
            item["aspect"]: item["business_category"]
            for item in self.business["top_customer_concerns"]
        }
        self.assertEqual("EXPERIENCE", concerns["Delivery"])
        self.assertEqual("EXPERIENCE", concerns["Customer Service"])

    def test_positive_shopping_experience_is_separate(self) -> None:
        positive = self.business["shopping_experience"]["positive"]
        self.assertEqual(["Delivery"], [item["aspect"] for item in positive])
        self.assertEqual("EXPERIENCE", positive[0]["business_category"])

    def test_negative_shopping_experience_is_separate(self) -> None:
        negative = self.business["shopping_experience"]["negative"]
        self.assertEqual(
            ["Customer Service"],
            [item["aspect"] for item in negative],
        )

    def test_positive_must_strictly_exceed_negative_for_selling(self) -> None:
        aspects = {
            item["aspect"] for item in self.business["product_selling_points"]
        }
        self.assertNotIn("Sound Quality", aspects)
        self.assertNotIn("Build Quality", aspects)

    def test_negative_must_strictly_exceed_positive_for_pain(self) -> None:
        aspects = {
            item["aspect"] for item in self.business["product_pain_points"]
        }
        self.assertNotIn("Sound Quality", aspects)
        self.assertNotIn("Picture Quality", aspects)

    def test_minimum_sample_applies_to_product_and_experience(self) -> None:
        rows = [
            _summary_row("Remote Control", 2, negative=2),
            _summary_row("Delivery", 2, positive=2),
        ]
        strict = _business_from_rows(rows, minimum_reviews=3)
        relaxed = _business_from_rows(rows, minimum_reviews=2)
        self.assertEqual([], strict["product_pain_points"])
        self.assertEqual([], strict["shopping_experience"]["positive"])
        self.assertEqual(
            ["Remote Control"],
            [item["aspect"] for item in relaxed["product_pain_points"]],
        )
        self.assertEqual(
            ["Delivery"],
            [item["aspect"] for item in relaxed["shopping_experience"]["positive"]],
        )

    def test_price_setup_and_ease_of_use_are_product_selling_candidates(self) -> None:
        business = _business_from_rows(
            [
                _summary_row("Price / Value", 1, positive=1),
                _summary_row("Setup / Installation", 1, positive=1),
                _summary_row("Ease of Use", 1, positive=1),
            ]
        )
        selling = {
            item["aspect"] for item in business["product_selling_points"]
        }
        self.assertEqual(
            {"Price / Value", "Setup / Installation", "Ease of Use"},
            selling,
        )

    def test_other_is_excluded_from_every_formal_ranking(self) -> None:
        self.assertNotIn(
            "Other",
            {item["aspect"] for item in self.business["top_customer_concerns"]},
        )
        self.assertNotIn(
            "Other",
            {item["aspect"] for item in self.business["product_selling_points"]},
        )
        self.assertNotIn(
            "Other",
            {item["aspect"] for item in self.business["product_pain_points"]},
        )

    def test_ranking_records_keep_transparent_counts_and_rates(self) -> None:
        required = {
            "aspect",
            "business_category",
            "aspect_review_count",
            "mention_count",
            "positive_review_count",
            "negative_review_count",
            "neutral_review_count",
            "mixed_review_count",
            "positive_rate",
            "negative_rate",
            "neutral_rate",
            "mixed_rate",
            "aspect_share_of_reviews",
        }
        record = self.business["product_selling_points"][0]
        self.assertEqual(required, set(record))

    def test_neutral_and_mixed_are_reported_but_do_not_block_majority(self) -> None:
        business = _business_from_rows(
            [
                _summary_row(
                    "Design",
                    5,
                    positive=2,
                    negative=1,
                    neutral=1,
                    mixed=1,
                )
            ]
        )
        record = business["product_selling_points"][0]
        self.assertEqual(1, record["neutral_review_count"])
        self.assertEqual(1, record["mixed_review_count"])
        self.assertEqual(20.0, record["neutral_rate"])
        self.assertEqual(20.0, record["mixed_rate"])

    def test_legacy_product_ranking_aliases_match_new_fields(self) -> None:
        self.assertEqual(
            self.business["product_selling_points"],
            self.business["top_selling_points"],
        )
        self.assertEqual(
            self.business["product_pain_points"],
            self.business["top_pain_points"],
        )


class EvidenceTests(unittest.TestCase):
    def test_evidence_comes_from_original_aspects_verbatim(self) -> None:
        evidence = collect_aspect_evidence(_fixture())
        values = [item["evidence"] for item in evidence["Picture Quality"]["positive"]]
        self.assertIn("The picture is sharp", values)
        self.assertIn("colors look accurate", values)

    def test_same_evidence_is_deduplicated_within_review(self) -> None:
        evidence = collect_aspect_evidence(_fixture())
        values = [item for item in evidence["Picture Quality"]["positive"] if item["source_row_id"] == 0]
        self.assertEqual(2, len(values))

    def test_same_evidence_is_retained_across_reviews(self) -> None:
        evidence = collect_aspect_evidence(_fixture())
        occurrences = [
            item
            for sentiment in ("positive", "negative")
            for item in evidence["Picture Quality"][sentiment]
            if item["evidence"] == "The picture is sharp"
        ]
        self.assertEqual({0, 1}, {item["source_row_id"] for item in occurrences})


class InvalidDataAndOutputTests(unittest.TestCase):
    def test_invalid_unique_json_is_reported_but_valid_evidence_survives(self) -> None:
        reviews = pd.DataFrame(
            [
                _row(
                    0,
                    "R001",
                    unique_aspects="{bad",
                    aspects=[_aspect("Design", "positive", "looks beautiful")],
                )
            ]
        )
        result = aggregate_reviews(reviews, min_aspect_reviews=1)
        self.assertTrue(result.aspect_summary.empty)
        self.assertIn("Design", result.evidence)
        self.assertEqual(1, result.business_summary["data_quality"]["invalid_json_rows"])

    def test_invalid_aspects_json_does_not_discard_valid_metrics(self) -> None:
        reviews = pd.DataFrame(
            [
                _row(
                    0,
                    "R001",
                    unique_aspects=[_unique("Design", "positive")],
                    aspects="{bad",
                )
            ]
        )
        result = aggregate_reviews(reviews, min_aspect_reviews=1)
        self.assertEqual("Design", result.aspect_summary.iloc[0]["aspect"])
        self.assertEqual({}, result.evidence)
        self.assertEqual(1, result.business_summary["data_quality"]["invalid_json_rows"])

    def test_two_bad_json_fields_in_one_row_count_as_one_invalid_row(self) -> None:
        reviews = pd.DataFrame([_row(0, "R001", unique_aspects="{bad", aspects="{bad")])
        result = aggregate_reviews(reviews, min_aspect_reviews=1)
        quality = result.business_summary["data_quality"]
        self.assertEqual(1, quality["invalid_json_rows"])
        self.assertEqual(2, len(quality["invalid_json_details"]))

    def test_parseable_non_list_json_is_invalid(self) -> None:
        reviews = pd.DataFrame([_row(0, "R001", unique_aspects="{}", aspects="[]")])
        result = aggregate_reviews(reviews, min_aspect_reviews=1)
        self.assertEqual(1, result.business_summary["data_quality"]["invalid_json_rows"])

    def test_empty_json_lists_are_valid(self) -> None:
        result = aggregate_reviews(pd.DataFrame([_row(0, "R001")]))
        self.assertTrue(result.aspect_summary.empty)
        self.assertEqual(0, result.business_summary["data_quality"]["invalid_json_rows"])

    def test_zero_success_is_safe_and_has_zero_coverage(self) -> None:
        reviews = pd.DataFrame([_row(0, "R001", status="failed")])
        result = aggregate_reviews(reviews)
        quality = result.business_summary["data_quality"]
        self.assertEqual(0.0, quality["analysis_coverage"])
        self.assertTrue(result.aspect_summary.empty)
        self.assertEqual([], result.business_summary["top_customer_concerns"])

    def test_analysis_coverage_is_success_over_total_rows(self) -> None:
        quality = aggregate_reviews(_fixture()).business_summary["data_quality"]
        self.assertAlmostEqual(5 / 7, quality["analysis_coverage"])

    def test_business_summary_is_json_serializable(self) -> None:
        payload = aggregate_reviews(_fixture()).business_summary
        self.assertIsInstance(json.dumps(payload, ensure_ascii=False), str)

    def test_aspect_summary_csv_and_json_outputs_are_saved(self) -> None:
        result = aggregate_reviews(_fixture(), min_aspect_reviews=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = save_aggregation_outputs(
                result,
                aspect_summary_path=root / "aspect_summary.csv",
                evidence_path=root / "aspect_evidence.json",
                business_summary_path=root / "business_summary.json",
            )
            saved = pd.read_csv(paths["aspect_summary"])
            self.assertEqual(list(ASPECT_SUMMARY_COLUMNS), list(saved.columns))
            self.assertEqual(
                sorted(saved["aspect_review_count"], reverse=True),
                saved["aspect_review_count"].tolist(),
            )
            self.assertEqual(result.evidence, json.loads(paths["aspect_evidence"].read_text()))
            self.assertEqual(
                result.business_summary,
                json.loads(paths["business_summary"].read_text()),
            )

    def test_load_analyzed_reviews_preserves_text_and_does_not_use_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analyzed_reviews.csv"
            _fixture().to_csv(path, index=False)
            loaded = load_analyzed_reviews(path)
            self.assertEqual(7, len(loaded))
            self.assertEqual("R001", loaded.iloc[0]["review_id"])

    def test_missing_required_column_has_clear_error(self) -> None:
        with self.assertRaisesRegex(AggregationError, "Missing analyzed review columns"):
            aggregate_reviews(pd.DataFrame({"source_row_id": [0]}))

    def test_duplicate_source_row_id_is_rejected(self) -> None:
        reviews = _fixture().iloc[:2].copy()
        reviews["source_row_id"] = [0, 0]
        with self.assertRaisesRegex(AggregationError, "source_row_id must be unique"):
            aggregate_reviews(reviews)


if __name__ == "__main__":
    unittest.main()
