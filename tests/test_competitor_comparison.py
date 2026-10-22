"""Offline tests for the deterministic Step 13 comparison engine."""

from __future__ import annotations

import ast
import copy
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

import competitor_comparison as comparison
from config import ANALYSIS_CANDIDATE_VERSION


class CompetitorComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.product_a = comparison.load_fixture_product("fixture_a")
        self.product_b = comparison.load_fixture_product("fixture_b")

    @staticmethod
    def _record(payload: dict, aspect: str) -> dict:
        return next(
            row for row in payload["aspect_comparisons"] if row["aspect"] == aspect
        )

    @staticmethod
    def _write_source(root: Path, *, include_metadata: bool = True) -> None:
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"analysis_status": "success", "rating": 5},
                {"analysis_status": "success", "rating": 3},
                {"analysis_status": "skipped", "rating": ""},
            ]
        ).to_csv(root / "analyzed_reviews.csv", index=False)
        pd.DataFrame(
            [
                {
                    "aspect": "Picture Quality",
                    "business_category": "PRODUCT",
                    "aspect_review_count": 2,
                    "aspect_share_of_reviews": 100.0,
                    "positive_rate": 50.0,
                    "negative_rate": 50.0,
                    "neutral_rate": 0.0,
                    "mixed_rate": 0.0,
                }
            ]
        ).to_csv(root / "aspect_summary.csv", index=False)
        (root / "business_summary.json").write_text(
            json.dumps(
                {
                    "data_quality": {
                        "total_input_rows": 3,
                        "successful_reviews": 2,
                    },
                    "sample_level": {"key": "early_signals"},
                }
            ),
            encoding="utf-8",
        )
        (root / "aspect_evidence.json").write_text(
            json.dumps(
                {
                    "Picture Quality": {
                        "positive": [
                            {
                                "source_row_id": 1,
                                "review_id": "A1",
                                "evidence": "sharp picture",
                            }
                        ],
                        "negative": [],
                    }
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [{"term": "sharp picture", "review_count": 1, "review_share": 50.0}]
        ).to_csv(root / "keyword_summary.csv", index=False)
        (root / "sentiment_keywords.json").write_text("{}", encoding="utf-8")
        (root / "aspect_keywords.json").write_text(
            json.dumps(
                {
                    "Picture Quality": [
                        {
                            "term": "sharp picture",
                            "term_type": "bigram",
                            "count": 1,
                            "review_count": 1,
                            "review_share": 50.0,
                            "review_refs": [{"source_row_id": 1}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        if include_metadata:
            (root / "analysis_run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-one",
                        "source_type": "upload",
                        "candidate": ANALYSIS_CANDIDATE_VERSION,
                        "taxonomy_version": ANALYSIS_CANDIDATE_VERSION,
                    }
                ),
                encoding="utf-8",
            )

    def test_equal_sample_comparison_uses_rates(self) -> None:
        b = replace(self.product_b, successful_reviews=100, total_reviews=100)
        result = comparison.build_comparison(self.product_a, b)
        picture = self._record(result, "Picture Quality")
        self.assertEqual(30.0, picture["net_sentiment_gap"])
        self.assertEqual("clear_advantage", picture["comparison_status"])

    def test_unequal_sample_comparison_warns_above_ratio(self) -> None:
        b = replace(self.product_b, successful_reviews=20, total_reviews=20)
        result = comparison.build_comparison(self.product_a, b)
        warning = next(w for w in result["sample_warnings"] if w["code"] == "uneven_sample_sizes")
        self.assertEqual(5.0, warning["sample_size_ratio"])

    def test_early_comparison_uses_directional_signal(self) -> None:
        a = replace(self.product_a, successful_reviews=8, total_reviews=8)
        result = comparison.build_comparison(a, self.product_b)
        self.assertEqual("early_comparison", result["comparison_mode"])
        self.assertEqual(
            "directional_signal_a",
            self._record(result, "Picture Quality")["comparison_status"],
        )
        self.assertTrue(result["recommendations"])
        self.assertTrue(
            all(item["type"] in {"monitor", "market_opportunity"} for item in result["recommendations"])
        )

    def test_zero_sample_warning_is_valid_json(self) -> None:
        a = replace(self.product_a, successful_reviews=0, total_reviews=0)
        result = comparison.build_comparison(a, self.product_b)
        warning = next(w for w in result["sample_warnings"] if w["code"] == "uneven_sample_sizes")
        self.assertIsNone(warning["sample_size_ratio"])
        json.dumps(result, allow_nan=False)

    def test_review_share_gap_is_normalized_percentage_gap(self) -> None:
        record = self._record(
            comparison.build_comparison(self.product_a, self.product_b),
            "Picture Quality",
        )
        self.assertEqual(7.0, record["review_share_gap"])
        self.assertEqual(62.0, record["a_review_share"])
        self.assertEqual(55.0, record["b_review_share"])

    def test_raw_count_does_not_determine_advantage(self) -> None:
        a_metrics = copy.deepcopy(list(self.product_a.aspect_metrics))
        b_metrics = copy.deepcopy(list(self.product_b.aspect_metrics))
        for row in a_metrics:
            if row["aspect"] == "Picture Quality":
                row.update(review_count=90, positive_rate=55.0, negative_rate=35.0)
        for row in b_metrics:
            if row["aspect"] == "Picture Quality":
                row.update(review_count=10, positive_rate=80.0, negative_rate=10.0)
        a = replace(self.product_a, aspect_metrics=tuple(a_metrics))
        b = replace(self.product_b, aspect_metrics=tuple(b_metrics))
        record = self._record(comparison.build_comparison(a, b), "Picture Quality")
        self.assertEqual("clear_weakness", record["comparison_status"])

    def test_net_sentiment_is_positive_minus_negative(self) -> None:
        record = self._record(
            comparison.build_comparison(self.product_a, self.product_b),
            "Reliability",
        )
        self.assertEqual(64.0, record["a_net_sentiment"])
        self.assertEqual(36.0, record["b_net_sentiment"])
        self.assertEqual(28.0, record["net_sentiment_gap"])

    def test_status_thresholds(self) -> None:
        self.assertEqual(
            "similar",
            comparison.classify_comparison(
                a_review_count=10, b_review_count=10, net_sentiment_gap=7.9
            ),
        )
        self.assertEqual(
            "slight_advantage",
            comparison.classify_comparison(
                a_review_count=10, b_review_count=10, net_sentiment_gap=8.0
            ),
        )
        self.assertEqual(
            "clear_advantage",
            comparison.classify_comparison(
                a_review_count=10, b_review_count=10, net_sentiment_gap=20.0
            ),
        )
        self.assertEqual(
            "clear_weakness",
            comparison.classify_comparison(
                a_review_count=10, b_review_count=10, net_sentiment_gap=-20.0
            ),
        )

    def test_medium_evidence_cannot_be_called_clear(self) -> None:
        self.assertEqual(
            "slight_advantage",
            comparison.classify_comparison(
                a_review_count=3, b_review_count=3, net_sentiment_gap=80.0
            ),
        )
        self.assertEqual(
            "slight_weakness",
            comparison.classify_comparison(
                a_review_count=3, b_review_count=3, net_sentiment_gap=-80.0
            ),
        )

    def test_evidence_thresholds_use_both_sides(self) -> None:
        self.assertEqual("Low", comparison.evidence_level(10, 2))
        self.assertEqual("Medium", comparison.evidence_level(10, 3))
        self.assertEqual("High", comparison.evidence_level(10, 10))

    def test_insufficient_evidence_below_minimum(self) -> None:
        self.assertEqual(
            "insufficient_evidence",
            comparison.classify_comparison(
                a_review_count=50, b_review_count=2, net_sentiment_gap=60
            ),
        )

    def test_missing_aspect_does_not_become_zero_satisfaction(self) -> None:
        b = replace(
            self.product_b,
            aspect_metrics=tuple(
                row for row in self.product_b.aspect_metrics if row["aspect"] != "Design"
            ),
        )
        record = self._record(comparison.build_comparison(self.product_a, b), "Design")
        self.assertIsNone(record["b_positive_rate"])
        self.assertIsNone(record["b_negative_rate"])
        self.assertEqual("insufficient_evidence", record["comparison_status"])

    def test_primary_schema_always_contains_all_canonical_aspects_in_order(self) -> None:
        result = comparison.build_comparison(self.product_a, self.product_b)
        self.assertEqual(15, len(result["aspect_comparisons"]))
        self.assertEqual(
            list(comparison.STANDARD_ASPECTS),
            [record["aspect"] for record in result["aspect_comparisons"]],
        )

    def test_aspect_missing_from_both_products_is_explicit_no_evidence(self) -> None:
        record = self._record(
            comparison.build_comparison(self.product_a, self.product_b),
            "Connectivity",
        )
        self.assertEqual(0, record["a_review_count"])
        self.assertEqual(0, record["b_review_count"])
        for key in (
            "a_positive_rate",
            "b_positive_rate",
            "a_negative_rate",
            "b_negative_rate",
            "a_net_sentiment",
            "b_net_sentiment",
            "net_sentiment_gap",
            "a_avg_rating",
            "b_avg_rating",
        ):
            self.assertIsNone(record[key], key)
        self.assertEqual("no_evidence", record["a_evidence_level"])
        self.assertEqual("no_evidence", record["b_evidence_level"])
        self.assertEqual("no_evidence", record["comparison_evidence"])
        self.assertEqual("insufficient_evidence", record["comparison_status"])
        self.assertEqual("none", record["leader"])

    def test_one_sided_evidence_never_creates_an_advantage(self) -> None:
        b = replace(
            self.product_b,
            aspect_metrics=tuple(
                row for row in self.product_b.aspect_metrics if row["aspect"] != "Design"
            ),
        )
        result = comparison.build_comparison(self.product_a, b)
        record = self._record(result, "Design")
        self.assertGreater(record["a_review_count"], 0)
        self.assertEqual(0, record["b_review_count"])
        self.assertIsNone(record["b_positive_rate"])
        self.assertIsNone(record["net_sentiment_gap"])
        self.assertEqual("insufficient_evidence", record["comparison_status"])
        self.assertNotIn(
            "Design",
            {row["aspect"] for row in result["competitive_advantages"]},
        )

    def test_zero_mention_placeholders_are_loaded_as_null_rates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root)
            frame = pd.read_csv(root / "aspect_summary.csv")
            frame.loc[0, "aspect_review_count"] = 0
            frame.loc[0, "aspect_share_of_reviews"] = 0
            frame.loc[0, "positive_rate"] = 0
            frame.loc[0, "negative_rate"] = 0
            frame.loc[0, "neutral_rate"] = 0
            frame.loc[0, "mixed_rate"] = 0
            frame["avg_rating"] = 0
            frame.to_csv(root / "aspect_summary.csv", index=False)
            product = comparison.load_product_analysis(root, product_name="Zero")
        metric = product.aspect_metrics[0]
        self.assertEqual(0, metric["review_count"])
        self.assertEqual(0.0, metric["review_share"])
        self.assertIsNone(metric["positive_rate"])
        self.assertIsNone(metric["negative_rate"])
        self.assertIsNone(metric["neutral_rate"])
        self.assertIsNone(metric["mixed_rate"])
        self.assertIsNone(metric["avg_rating"])
        self.assertEqual("no_evidence", metric["evidence_level"])

    def test_advantages_are_product_only_and_ranked(self) -> None:
        result = comparison.build_comparison(self.product_a, self.product_b)
        aspects = [row["aspect"] for row in result["competitive_advantages"]]
        self.assertEqual(["Picture Quality", "Reliability", "Design"], aspects[:3])
        self.assertNotIn("Delivery", aspects)
        self.assertNotIn("Other", aspects)

    def test_weaknesses_are_product_only_and_ranked(self) -> None:
        result = comparison.build_comparison(self.product_a, self.product_b)
        aspects = [row["aspect"] for row in result["competitive_weaknesses"]]
        self.assertEqual(["Price / Value", "Remote Control", "Sound Quality"], aspects[:3])
        self.assertNotIn("Delivery", aspects)

    def test_white_space_requires_both_products_negative(self) -> None:
        result = comparison.build_comparison(self.product_a, self.product_b)
        self.assertEqual(
            ["System Performance"],
            [row["aspect"] for row in result["white_space_opportunities"]],
        )

    def test_shopping_experience_is_separate(self) -> None:
        result = comparison.build_comparison(self.product_a, self.product_b)
        aspects = {row["aspect"] for row in result["shopping_experience_comparison"]}
        self.assertEqual({"Delivery", "Packaging", "Customer Service"}, aspects)

    def test_other_remains_in_full_table_only(self) -> None:
        result = comparison.build_comparison(self.product_a, self.product_b)
        self.assertIsNotNone(self._record(result, "Other"))
        ranked = result["competitive_advantages"] + result["competitive_weaknesses"]
        self.assertNotIn("Other", {row["aspect"] for row in ranked})

    def test_no_overall_winner_is_created(self) -> None:
        result = comparison.build_comparison(self.product_a, self.product_b)
        self.assertNotIn("overall_winner", result["head_to_head_summary"])

    def test_rating_comparison_available(self) -> None:
        result = comparison.build_comparison(self.product_a, self.product_b)
        ratings = result["rating_comparison"]
        self.assertTrue(ratings["available"])
        self.assertAlmostEqual(0.2, ratings["average_rating_gap"])

    def test_missing_rating_is_graceful(self) -> None:
        b = replace(
            self.product_b,
            rating_summary={"available": False, "rated_reviews": 0, "distribution": []},
        )
        ratings = comparison.build_comparison(self.product_a, b)["rating_comparison"]
        self.assertFalse(ratings["available"])
        self.assertEqual("missing_rating", ratings["reason"])

    def test_phrase_overlap_is_normalized_exact_only(self) -> None:
        result = comparison.compare_phrase_records(
            [{"term": "Easy Setup!", "review_count": 3, "review_share": 10}],
            [{"term": "easy setup", "review_count": 2, "review_share": 8}],
        )
        self.assertEqual("easy setup", result["shared"][0]["normalized_term"])

    def test_distinctive_phrase_requires_support_and_share_gap(self) -> None:
        result = comparison.compare_phrase_records(
            [
                {"term": "deep blacks", "review_count": 3, "review_share": 10},
                {"term": "single", "review_count": 1, "review_share": 30},
            ],
            [],
        )
        self.assertEqual(
            ["deep blacks"],
            [row["term"] for row in result["product_a_distinctive"]],
        )

    def test_evidence_side_by_side_is_verbatim_and_copied(self) -> None:
        records = comparison.evidence_for_product(
            self.product_a, "Picture Quality", "positive", limit=1
        )
        self.assertEqual("the deep blacks make movies look fantastic", records[0]["evidence"])
        records[0]["evidence"] = "changed"
        self.assertEqual(
            "the deep blacks make movies look fantastic",
            comparison.evidence_for_product(
                self.product_a, "Picture Quality", "positive", limit=1
            )[0]["evidence"],
        )

    def test_versions_require_current_candidate_and_taxonomy(self) -> None:
        self.assertTrue(
            comparison.analysis_versions_compatible(self.product_a, self.product_b)
        )
        old = replace(self.product_b, taxonomy_version="old-taxonomy")
        self.assertFalse(comparison.analysis_versions_compatible(self.product_a, old))
        with self.assertRaises(comparison.AnalysisVersionMismatch):
            comparison.build_comparison(self.product_a, old)
        unknown_a = replace(self.product_a, taxonomy_version="unknown")
        unknown_b = replace(self.product_b, taxonomy_version="unknown")
        self.assertFalse(comparison.analysis_versions_compatible(unknown_a, unknown_b))
        named_a = replace(self.product_a, taxonomy_version="step8.8-frozen")
        named_b = replace(self.product_b, taxonomy_version="step8.8-frozen")
        self.assertTrue(comparison.analysis_versions_compatible(named_a, named_b))

    def test_same_product_run_is_rejected(self) -> None:
        with self.assertRaises(comparison.ComparisonError):
            comparison.build_comparison(self.product_a, self.product_a)

    def test_same_saved_analysis_id_is_rejected_even_if_other_fields_differ(self) -> None:
        a = replace(self.product_a, analysis_id="AN_20261015_120000_aaaa")
        b = replace(
            self.product_b,
            analysis_id="AN_20261015_120000_aaaa",
            product_id="different-product",
            analysis_run_id="different-run",
        )
        with self.assertRaises(comparison.ComparisonError):
            comparison.build_comparison(a, b)

    def test_sorting_supported_then_gap_share_and_aspect(self) -> None:
        records = [
            {"aspect": "Z", "comparison_status": "similar", "net_sentiment_gap": 5, "average_review_share": 20},
            {"aspect": "B", "comparison_status": "similar", "net_sentiment_gap": 10, "average_review_share": 10},
            {"aspect": "A", "comparison_status": "similar", "net_sentiment_gap": 10, "average_review_share": 10},
            {"aspect": "X", "comparison_status": "insufficient_evidence", "net_sentiment_gap": 90, "average_review_share": 90},
        ]
        self.assertEqual(
            ["A", "B", "Z", "X"],
            [row["aspect"] for row in sorted(records, key=comparison.comparison_sort_key)],
        )

    def test_build_does_not_mutate_products(self) -> None:
        before_a = copy.deepcopy(self.product_a)
        before_b = copy.deepcopy(self.product_b)
        comparison.build_comparison(self.product_a, self.product_b)
        self.assertEqual(before_a, self.product_a)
        self.assertEqual(before_b, self.product_b)

    def test_loader_reads_seven_persisted_outputs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root)
            product = comparison.load_product_analysis(root, product_name="Saved A")
        self.assertEqual(3, product.total_reviews)
        self.assertEqual(2, product.successful_reviews)
        self.assertAlmostEqual(2 / 3, product.coverage)
        self.assertEqual(4.0, product.rating_summary["average_rating"])
        phrase = product.language_by_aspect["Picture Quality"]["positive"][0]
        self.assertEqual("sharp picture", phrase["term"])

    def test_loader_reads_saved_manifest_identity_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root)
            (root / "analysis_manifest.json").write_text(
                json.dumps(
                    {
                        "analysis_id": "AN_20261015_120000_abcd",
                        "product_id": "saved-product-id",
                        "analysis_run_id": "saved-run-id",
                        "source_type": "amazon",
                        "analysis_candidate": ANALYSIS_CANDIDATE_VERSION,
                        "taxonomy_version": ANALYSIS_CANDIDATE_VERSION,
                        "schema_version": "step13.1.analysis.v1",
                    }
                ),
                encoding="utf-8",
            )
            product = comparison.load_product_analysis(root, product_name="Saved")
        self.assertEqual("AN_20261015_120000_abcd", product.analysis_id)
        self.assertEqual("saved-product-id", product.product_id)
        self.assertEqual("saved-run-id", product.analysis_run_id)
        self.assertEqual("amazon", product.source_type)
        self.assertEqual("step13.1.analysis.v1", product.analysis_schema_version)
        self.assertEqual(
            "AN_20261015_120000_abcd",
            product.public_metadata()["analysis_id"],
        )

    def test_loader_rejects_unsupported_saved_manifest_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root)
            (root / "analysis_manifest.json").write_text(
                json.dumps(
                    {
                        "analysis_id": "AN_20261015_120000_abcd",
                        "analysis_candidate": ANALYSIS_CANDIDATE_VERSION,
                        "taxonomy_version": ANALYSIS_CANDIDATE_VERSION,
                        "schema_version": "future.schema.v99",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(comparison.AnalysisVersionMismatch):
                comparison.load_product_analysis(root, product_name="Future")

    def test_direct_products_with_unsupported_saved_schema_are_incompatible(self) -> None:
        incompatible = replace(
            self.product_a,
            analysis_schema_version="future.schema.v99",
        )
        self.assertFalse(
            comparison.analysis_versions_compatible(incompatible, self.product_b)
        )

    def test_loader_fails_closed_without_version_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root, include_metadata=False)
            with self.assertRaises(comparison.AnalysisVersionMismatch):
                comparison.load_product_analysis(root, product_name="Unknown")

    def test_loader_allows_explicit_current_version_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root, include_metadata=False)
            product = comparison.load_product_analysis(
                root, product_name="Attested", assume_current_version=True
            )
        self.assertEqual(ANALYSIS_CANDIDATE_VERSION, product.analysis_candidate)
        self.assertEqual(ANALYSIS_CANDIDATE_VERSION, product.taxonomy_version)

    def test_loader_requires_every_saved_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root)
            (root / "aspect_keywords.json").unlink()
            with self.assertRaisesRegex(comparison.ComparisonError, "aspect_keywords"):
                comparison.load_product_analysis(root, product_name="Incomplete")

    def test_loader_supports_missing_rating_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root)
            pd.DataFrame([{"analysis_status": "success"}]).to_csv(
                root / "analyzed_reviews.csv", index=False
            )
            product = comparison.load_product_analysis(root, product_name="No Rating")
        self.assertFalse(product.rating_summary["available"])

    def test_save_outputs_writes_compact_schema_and_loads(self) -> None:
        result = comparison.build_comparison(
            self.product_a,
            self.product_b,
            generated_at="2026-01-01T00:00:00+00:00",
            comparison_id="cmp-stable",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "comparison" / "current"
            paths = comparison.save_comparison_outputs(result, root=root)
            loaded, frame, metadata = comparison.load_saved_comparison(root)
            self.assertEqual(
                {"comparison_summary.json", "aspect_comparison.csv", "comparison_metadata.json"},
                {path.name for path in root.iterdir()},
            )
            self.assertEqual(list(comparison.COMPARISON_OUTPUT_COLUMNS), list(frame.columns))
            self.assertEqual("cmp-stable", loaded["comparison_id"])
            self.assertEqual("fixture-a-v1", metadata["product_a_run_id"])
            self.assertEqual(paths.root, root.resolve())
            self.assertEqual(15, len(frame))
            self.assertEqual(
                list(comparison.STANDARD_ASPECTS), frame["aspect"].tolist()
            )

    def test_metadata_records_saved_analysis_ids_and_aliases(self) -> None:
        a = replace(self.product_a, analysis_id="AN_20261015_120000_aaaa")
        b = replace(self.product_b, analysis_id="AN_20261015_120100_bbbb")
        result = comparison.build_comparison(
            a,
            b,
            generated_at="2026-10-15T12:02:00+00:00",
            comparison_id="CMP_20261015_120200_cccc",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "comparison" / "current"
            comparison.save_comparison_outputs(result, root=root)
            _, _, metadata = comparison.load_saved_comparison(root)
        self.assertEqual("AN_20261015_120000_aaaa", metadata["product_a_analysis_id"])
        self.assertEqual("AN_20261015_120100_bbbb", metadata["product_b_analysis_id"])
        self.assertEqual("AN_20261015_120000_aaaa", metadata["analysis_a_id"])
        self.assertEqual("AN_20261015_120100_bbbb", metadata["analysis_b_id"])

    def test_comparison_reproduction_is_deterministic_with_fixed_identity(self) -> None:
        kwargs = {
            "generated_at": "2026-10-15T12:02:00+00:00",
            "comparison_id": "CMP_20261015_120200_cccc",
        }
        first = comparison.build_comparison(self.product_a, self.product_b, **kwargs)
        second = comparison.build_comparison(self.product_a, self.product_b, **kwargs)
        self.assertEqual(first, second)

    def test_generated_comparison_id_uses_stable_readable_format(self) -> None:
        result = comparison.build_comparison(
            self.product_a,
            self.product_b,
            generated_at="2026-10-15T12:02:00+00:00",
        )
        self.assertRegex(
            result["comparison_id"],
            re.compile(r"^CMP_20261015_120200_[a-f0-9]{4}$"),
        )

    def test_save_path_must_be_comparison_current(self) -> None:
        result = comparison.build_comparison(self.product_a, self.product_b)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(comparison.ComparisonOutputSafetyError):
                comparison.save_comparison_outputs(
                    result, root=Path(temporary) / "session" / "current"
                )

    def test_saved_json_rejects_nan(self) -> None:
        result = comparison.build_comparison(self.product_a, self.product_b)
        result["aspect_comparisons"][0]["a_positive_rate"] = float("nan")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "comparison" / "current"
            with self.assertRaises(ValueError):
                comparison.save_comparison_outputs(result, root=root)

    def test_engine_has_no_model_or_pipeline_import(self) -> None:
        tree = ast.parse(Path(comparison.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(
            imported.isdisjoint(
                {
                    "llm_client",
                    "review_analyzer",
                    "batch_analyzer",
                    "aggregator",
                    "keyword_analyzer",
                    "business_intelligence",
                    "visualization",
                    "scraper",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
