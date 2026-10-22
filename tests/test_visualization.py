"""Offline tests for the Step 9 dashboard data and filtering layer."""

from __future__ import annotations

import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization import (  # noqa: E402
    DashboardData,
    apply_review_quick_filter,
    aspect_language_by_sentiment,
    calculate_kpis,
    count_validated_evidence,
    deterministic_business_implications,
    evidence_for_aspect,
    evidence_with_review_metadata,
    filter_aspect_summary,
    filter_reviews,
    load_dashboard_data,
    output_fingerprint,
    prepare_review_data,
    review_filter_options,
    safe_parse_json_list,
    top_term_records,
)


def _aspect(aspect: str, sentiment: str, evidence: str) -> dict[str, str]:
    return {
        "raw_aspect": aspect,
        "aspect": aspect,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _reviews() -> pd.DataFrame:
    picture_positive = _aspect(
        "Picture Quality", "positive", "picture quality is excellent"
    )
    remote_negative = _aspect(
        "Remote Control", "negative", "remote feels cheap"
    )
    picture_negative = _aspect(
        "Picture Quality", "negative", "screen flickers"
    )
    return pd.DataFrame(
        [
            {
                "source_row_id": "001",
                "review_id": "R001",
                "rating": 5,
                "title": "Great [TV]",
                "content": "The picture is great but the remote feels cheap.",
                "original_content": "The picture is great but the remote feels cheap.",
                "cleaned_content": "The picture is great but the remote feels cheap.",
                "verified_purchase": "True",
                "analysis_status": "success",
                "summary": "Strong picture, weak remote.",
                "aspects": json.dumps([picture_positive, remote_negative]),
                "unique_aspects": "[]",
                "positive_points": "[]",
                "negative_points": "[]",
            },
            {
                "source_row_id": "002",
                "review_id": "R009",
                "rating": 4,
                "title": "Flicker issue",
                "content": "Screen flickers sometimes.",
                "original_content": "Screen flickers sometimes.",
                "cleaned_content": "Screen flickers sometimes.",
                "verified_purchase": "False",
                "analysis_status": "success",
                "summary": "Screen flicker reported.",
                "aspects": json.dumps([picture_negative]),
                "unique_aspects": "[]",
                "positive_points": "[]",
                "negative_points": "[]",
            },
            {
                "source_row_id": "003",
                "review_id": "R009",
                "rating": 6,
                "title": "Invalid score fixture",
                "content": "Fixture row.",
                "original_content": "Fixture row.",
                "cleaned_content": "Fixture row.",
                "verified_purchase": "False",
                "analysis_status": "failed",
                "summary": "",
                "aspects": "[]",
                "unique_aspects": "[]",
                "positive_points": "[]",
                "negative_points": "[]",
            },
            {
                "source_row_id": "004",
                "review_id": "R004",
                "rating": 0,
                "title": "",
                "content": "",
                "original_content": "",
                "cleaned_content": "",
                "verified_purchase": "",
                "analysis_status": "skipped",
                "summary": "",
                "aspects": "[]",
                "unique_aspects": "[]",
                "positive_points": "[]",
                "negative_points": "[]",
            },
        ]
    )


def _aspect_summary() -> pd.DataFrame:
    rows = [
        ("Picture Quality", "PRODUCT", 2, 2, 50.0, 1, 1, 0, 0, 50.0, 50.0, 0.0, 0.0),
        ("Remote Control", "PRODUCT", 1, 1, 25.0, 0, 1, 0, 0, 0.0, 100.0, 0.0, 0.0),
        ("Delivery", "EXPERIENCE", 1, 1, 25.0, 1, 0, 0, 0, 100.0, 0.0, 0.0, 0.0),
    ]
    columns = [
        "aspect",
        "business_category",
        "aspect_review_count",
        "mention_count",
        "aspect_share_of_reviews",
        "positive_review_count",
        "negative_review_count",
        "neutral_review_count",
        "mixed_review_count",
        "positive_rate",
        "negative_rate",
        "neutral_rate",
        "mixed_rate",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["include_in_business_metrics"] = True
    return frame


def _business_summary() -> dict[str, object]:
    return {
        "data_quality": {
            "total_input_rows": 4,
            "successful_reviews": 2,
            "failed_reviews": 1,
            "skipped_reviews": 1,
            "analysis_coverage": 0.5,
        },
        "ranking_rules": {"minimum_aspect_reviews": 1},
        "top_customer_concerns": [
            {
                "aspect": "Picture Quality",
                "business_category": "PRODUCT",
                "aspect_review_count": 2,
                "positive_review_count": 1,
                "negative_review_count": 0,
            }
        ],
        "product_selling_points": [],
        "product_pain_points": [
            {
                "aspect": "Remote Control",
                "business_category": "PRODUCT",
                "aspect_review_count": 1,
                "positive_review_count": 0,
                "negative_review_count": 1,
                "negative_rate": 100.0,
            }
        ],
        "shopping_experience": {"positive": [], "negative": []},
    }


def _evidence() -> dict[str, object]:
    return {
        "Picture Quality": {
            "positive": [
                {
                    "source_row_id": 1,
                    "review_id": "R001",
                    "evidence": "picture quality is excellent",
                }
            ],
            "negative": [
                {
                    "source_row_id": 2,
                    "review_id": "R009",
                    "evidence": "screen flickers",
                }
            ],
            "neutral": [],
        },
        "Remote Control": {
            "positive": [],
            "negative": [
                {
                    "source_row_id": 1,
                    "review_id": "R001",
                    "evidence": "remote feels cheap",
                }
            ],
            "neutral": [],
        },
    }


def _write_fixture(root: Path) -> None:
    _reviews().to_csv(root / "analyzed_reviews.csv", index=False)
    _aspect_summary().to_csv(root / "aspect_summary.csv", index=False)
    (root / "business_summary.json").write_text(
        json.dumps(_business_summary()), encoding="utf-8"
    )
    (root / "aspect_evidence.json").write_text(
        json.dumps(_evidence()), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "term": "picture quality",
                "term_type": "bigram",
                "count": 2,
                "review_count": 2,
                "review_share": 100.0,
            }
        ]
    ).to_csv(root / "keyword_summary.csv", index=False)
    (root / "sentiment_keywords.json").write_text(
        json.dumps(
            {
                "positive": [
                    {
                        "term": "picture quality",
                        "term_type": "bigram",
                        "count": 1,
                        "review_count": 1,
                    }
                ],
                "negative": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "aspect_keywords.json").write_text(
        json.dumps(
            {
                "Picture Quality": [
                    {
                        "term": "picture quality",
                        "term_type": "bigram",
                        "count": 1,
                        "review_count": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class SafeJsonTests(unittest.TestCase):
    def test_valid_json_list_is_parsed(self) -> None:
        value, error = safe_parse_json_list('[{"aspect":"Design"}]')
        self.assertEqual([{"aspect": "Design"}], value)
        self.assertIsNone(error)

    def test_empty_none_and_nan_are_safe(self) -> None:
        for value in ("", None, float("nan")):
            with self.subTest(value=value):
                parsed, error = safe_parse_json_list(value)
                self.assertEqual([], parsed)
                self.assertIsNone(error)

    def test_malformed_json_returns_warning(self) -> None:
        parsed, error = safe_parse_json_list("{bad")
        self.assertEqual([], parsed)
        self.assertIn("invalid JSON", str(error))

    def test_non_list_json_returns_warning(self) -> None:
        parsed, error = safe_parse_json_list('{"aspect":"Design"}')
        self.assertEqual([], parsed)
        self.assertEqual("JSON value must be an array", error)

    def test_prepare_review_data_parses_all_four_json_columns(self) -> None:
        prepared, warnings = prepare_review_data(_reviews().iloc[:1])
        self.assertFalse(warnings)
        for column in (
            "parsed_aspects",
            "parsed_unique_aspects",
            "parsed_positive_points",
            "parsed_negative_points",
        ):
            self.assertIn(column, prepared.columns)

    def test_one_malformed_row_does_not_break_other_rows(self) -> None:
        frame = _reviews().iloc[:2].copy()
        frame.loc[frame.index[0], "aspects"] = "{bad"
        prepared, warnings = prepare_review_data(frame)
        self.assertEqual(2, len(prepared))
        self.assertEqual([], prepared.iloc[0]["parsed_aspects"])
        self.assertEqual(1, len(prepared.iloc[1]["parsed_aspects"]))
        self.assertEqual(1, len(warnings))
        self.assertEqual("001", warnings[0]["source_row_id"])

    def test_prepare_review_data_does_not_modify_input(self) -> None:
        frame = _reviews()
        before = frame.copy(deep=True)
        prepare_review_data(frame)
        assert_frame_equal(before, frame)


class DashboardLoaderTests(unittest.TestCase):
    def test_all_seven_outputs_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            data = load_dashboard_data(root)
        self.assertFalse(data.issues)
        self.assertIsNotNone(data.analyzed_reviews)
        self.assertIsNotNone(data.business_summary)
        self.assertIsNotNone(data.keyword_summary)

    def test_source_row_id_and_review_id_are_loaded_as_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            data = load_dashboard_data(root)
        assert data.analyzed_reviews is not None
        self.assertEqual("001", data.analyzed_reviews.iloc[0]["source_row_id"])
        self.assertEqual("R001", data.analyzed_reviews.iloc[0]["review_id"])

    def test_missing_files_return_issues_not_exceptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = load_dashboard_data(directory)
        self.assertEqual(7, len(data.issues))
        self.assertIsNone(data.analyzed_reviews)

    def test_missing_aggregate_file_has_correct_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            (root / "business_summary.json").unlink()
            data = load_dashboard_data(root)
        issue = data.issue_for("business_summary")
        self.assertIsNotNone(issue)
        self.assertEqual("python main.py --aggregate", issue.command)

    def test_missing_keyword_file_does_not_remove_core_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            (root / "keyword_summary.csv").unlink()
            data = load_dashboard_data(root)
        self.assertIsNotNone(data.analyzed_reviews)
        self.assertIsNotNone(data.business_summary)
        self.assertIsNone(data.keyword_summary)
        self.assertEqual(
            "python main.py --keywords",
            data.issue_for("keyword_summary").command,
        )

    def test_malformed_json_only_degrades_that_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            (root / "aspect_evidence.json").write_text("{bad", encoding="utf-8")
            data = load_dashboard_data(root)
        self.assertIsNone(data.aspect_evidence)
        self.assertIsNotNone(data.business_summary)
        self.assertIsNotNone(data.analyzed_reviews)

    def test_valid_json_with_invalid_business_shape_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            malformed = _business_summary()
            malformed["data_quality"] = []
            (root / "business_summary.json").write_text(
                json.dumps(malformed), encoding="utf-8"
            )
            data = load_dashboard_data(root)
        self.assertIsNone(data.business_summary)
        self.assertIsNotNone(data.analyzed_reviews)
        self.assertIn("invalid structure", data.issue_for("business_summary").message)

    def test_output_fingerprint_reflects_all_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            fingerprint = output_fingerprint(root)
        self.assertEqual(7, len(fingerprint))
        self.assertTrue(all(size > 0 for _, _, size in fingerprint))

    def test_loading_and_helpers_do_not_rewrite_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            data = load_dashboard_data(root)
            calculate_kpis(data)
            assert data.analyzed_reviews is not None
            filter_reviews(data.analyzed_reviews, query="picture")
            after = {path.name: path.read_bytes() for path in root.iterdir()}
        self.assertEqual(before, after)


class DashboardKpiTests(unittest.TestCase):
    def _data(self) -> DashboardData:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            return load_dashboard_data(root)

    def test_kpis_use_persisted_quality_and_existing_outputs(self) -> None:
        kpis = calculate_kpis(self._data())
        self.assertEqual(4, kpis["total_reviews"])
        self.assertEqual(2, kpis["successful_reviews"])
        self.assertEqual(1, kpis["failed_reviews"])
        self.assertEqual(1, kpis["skipped_reviews"])
        self.assertEqual(0.5, kpis["analysis_coverage"])
        self.assertEqual(2, kpis["product_aspects_identified"])
        self.assertEqual(3, kpis["validated_evidence_count"])

    def test_evidence_count_uses_nested_sentiment_lists(self) -> None:
        self.assertEqual(3, count_validated_evidence(_evidence()))
        self.assertEqual(0, count_validated_evidence(None))

    def test_zero_data_has_safe_zero_kpis(self) -> None:
        data = DashboardData(
            analyzed_reviews=None,
            aspect_summary=None,
            business_summary=None,
            aspect_evidence=None,
            keyword_summary=None,
            sentiment_keywords=None,
            aspect_keywords=None,
            issues=(),
            row_json_warnings=(),
        )
        kpis = calculate_kpis(data)
        self.assertEqual(0, kpis["total_reviews"])
        self.assertEqual(0.0, kpis["analysis_coverage"])

    def test_quick_scan_coverage_uses_imported_rows_not_external_total(self) -> None:
        frame = pd.DataFrame(
            {"analysis_status": ["success"] * 8}
        )
        data = DashboardData(
            analyzed_reviews=frame,
            aspect_summary=None,
            business_summary={
                "data_quality": {
                    "total_input_rows": 8,
                    "successful_reviews": 8,
                    "failed_reviews": 0,
                    "skipped_reviews": 0,
                    "analysis_coverage": 1.0,
                }
            },
            aspect_evidence=None,
            keyword_summary=None,
            sentiment_keywords=None,
            aspect_keywords=None,
            issues=(),
            row_json_warnings=(),
        )

        kpis = calculate_kpis(data)

        self.assertEqual(8, kpis["total_reviews"])
        self.assertEqual(8, kpis["successful_reviews"])
        self.assertEqual(1.0, kpis["analysis_coverage"])

    def test_kpi_helper_does_not_modify_nested_inputs(self) -> None:
        data = self._data()
        before = copy.deepcopy(data.business_summary)
        calculate_kpis(data)
        self.assertEqual(before, data.business_summary)


class DashboardFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = _reviews()
        self.frame, self.warnings = prepare_review_data(self.raw)

    def test_filter_options_include_invalid_fixture_ratings(self) -> None:
        options = review_filter_options(self.frame)
        self.assertEqual([0, 4, 5, 6], options["ratings"])

    def test_rating_filter(self) -> None:
        result = filter_reviews(self.frame, ratings=[5])
        self.assertEqual(["R001"], result["review_id"].tolist())

    def test_status_filter(self) -> None:
        result = filter_reviews(self.frame, statuses=["skipped"])
        self.assertEqual(["R004"], result["review_id"].tolist())

    def test_false_string_is_not_treated_as_true(self) -> None:
        result = filter_reviews(self.frame, verified_purchase="Not verified")
        self.assertEqual(["002", "003"], result["source_row_id"].tolist())

    def test_literal_case_insensitive_search(self) -> None:
        result = filter_reviews(self.frame, query="[tv]")
        self.assertEqual(["R001"], result["review_id"].tolist())

    def test_duplicate_review_ids_are_not_collapsed(self) -> None:
        result = filter_reviews(self.frame, query="R009")
        self.assertEqual(2, len(result))
        self.assertEqual(["002", "003"], result["source_row_id"].tolist())

    def test_aspect_and_sentiment_must_match_same_mention(self) -> None:
        incorrect = filter_reviews(
            self.frame,
            aspects=["Picture Quality"],
            sentiments=["negative"],
        )
        self.assertNotIn("001", incorrect["source_row_id"].tolist())
        self.assertEqual(["002"], incorrect["source_row_id"].tolist())
        correct = filter_reviews(
            self.frame,
            aspects=["Remote Control"],
            sentiments=["negative"],
        )
        self.assertEqual(["001"], correct["source_row_id"].tolist())

    def test_combined_filters_use_and_semantics(self) -> None:
        result = filter_reviews(
            self.frame,
            ratings=[5],
            statuses=["success"],
            verified_purchase="Verified",
            query="remote",
        )
        self.assertEqual(["001"], result["source_row_id"].tolist())

    def test_filter_does_not_modify_input_or_share_nested_objects(self) -> None:
        before = copy.deepcopy(self.frame)
        result = filter_reviews(self.frame, statuses=["success"])
        result.iloc[0]["parsed_aspects"][0]["aspect"] = "Changed"
        self.assertEqual(
            "Picture Quality",
            self.frame.iloc[0]["parsed_aspects"][0]["aspect"],
        )
        assert_frame_equal(before, self.frame)

    def test_concern_category_filter_returns_copy(self) -> None:
        source = _aspect_summary()
        before = source.copy(deep=True)
        product = filter_aspect_summary(source, "Product")
        self.assertEqual({"PRODUCT"}, set(product["business_category"]))
        assert_frame_equal(before, source)

    def test_formal_concern_view_excludes_other(self) -> None:
        source = _aspect_summary()
        source.loc[len(source)] = {
            **source.iloc[0].to_dict(),
            "aspect": "Other",
            "business_category": "OTHER",
        }
        result = filter_aspect_summary(source, "All")
        self.assertNotIn("Other", result["aspect"].tolist())


class DashboardDetailAndSafetyTests(unittest.TestCase):
    def test_evidence_metadata_join_keeps_verbatim_evidence(self) -> None:
        entries = copy.deepcopy(_evidence()["Picture Quality"]["positive"])
        entries[0]["source_row_id"] = "001"
        result = evidence_with_review_metadata(entries, _reviews())
        self.assertEqual(entries[0]["evidence"], result[0]["evidence"])
        self.assertEqual("5", result[0]["rating"])
        self.assertEqual("Verified", result[0]["verified_purchase"])

    def test_language_by_aspect_uses_matching_sentiment_evidence(self) -> None:
        keywords = {
            "Remote Control": [
                {
                    "term": "cheap remote",
                    "term_type": "bigram",
                    "count": 1,
                    "review_count": 1,
                    "review_refs": [{"source_row_id": "001"}],
                }
            ]
        }
        evidence = {
            "Remote Control": {
                "positive": [
                    {"source_row_id": "001", "evidence": "buttons are responsive"}
                ],
                "negative": [
                    {"source_row_id": "001", "evidence": "cheap remote"}
                ],
                "neutral": [],
                "mixed": [],
            }
        }
        grouped = aspect_language_by_sentiment(
            keywords, evidence, "Remote Control"
        )
        self.assertEqual([], grouped["positive"])
        self.assertEqual("cheap remote", grouped["negative"][0]["term"])
        self.assertEqual(1, grouped["negative"][0]["sentiment_review_count"])

    def test_quick_filter_with_absent_category_returns_empty(self) -> None:
        prepared, _ = prepare_review_data(_reviews())
        summary = _aspect_summary().loc[
            lambda frame: frame["business_category"].eq("PRODUCT")
        ]
        result = apply_review_quick_filter(
            prepared, summary, "Shopping Experience"
        )
        self.assertTrue(result.empty)

    def test_product_issue_quick_filter_requires_negative_product_evidence(self) -> None:
        prepared, _ = prepare_review_data(_reviews())
        result = apply_review_quick_filter(
            prepared, _aspect_summary(), "Product Issues"
        )
        self.assertEqual(["001", "002"], result["source_row_id"].tolist())

    def test_evidence_lookup_returns_verbatim_copies(self) -> None:
        source = _evidence()
        before = copy.deepcopy(source)
        result = evidence_for_aspect(source, "Picture Quality")
        self.assertEqual("picture quality is excellent", result["positive"][0]["evidence"])
        result["positive"][0]["evidence"] = "changed"
        self.assertEqual(before, source)

    def test_unknown_aspect_returns_empty_evidence_buckets(self) -> None:
        result = evidence_for_aspect(_evidence(), "Unknown")
        self.assertTrue(all(not values for values in result.values()))

    def test_top_term_records_filter_type_and_do_not_share_nested_data(self) -> None:
        source = [
            {
                "term": "picture",
                "term_type": "unigram",
                "review_refs": [{"source_row_id": 1}],
            },
            {
                "term": "picture quality",
                "term_type": "bigram",
                "review_refs": [{"source_row_id": 2}],
            },
        ]
        result = top_term_records(source, term_types=("bigram",), limit=1)
        self.assertEqual("picture quality", result[0]["term"])
        result[0]["review_refs"][0]["source_row_id"] = 99
        self.assertEqual(2, source[1]["review_refs"][0]["source_row_id"])

    def test_business_implications_use_transparent_direction_rules(self) -> None:
        messages = deterministic_business_implications(_business_summary())
        self.assertIn("potential listing strength", messages[0])
        self.assertTrue(any("Remote Control" in message for message in messages))

    def test_dashboard_modules_do_not_import_analysis_or_llm_modules(self) -> None:
        forbidden = {
            "main",
            "llm_client",
            "batch_analyzer",
            "review_analyzer",
            "aggregator",
            "keyword_analyzer",
            "ollama",
            "openai",
            "config",
        }
        for filename in ("app.py", "visualization.py"):
            tree = ast.parse((PROJECT_ROOT / filename).read_text(encoding="utf-8"))
            imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
            self.assertTrue(forbidden.isdisjoint(imports), (filename, imports & forbidden))


if __name__ == "__main__":
    unittest.main()
