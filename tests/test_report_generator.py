"""Offline acceptance tests for Step 14 deterministic HTML reports."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import aggregator
import keyword_analyzer
import llm_client
import scraper

import analysis_library as library
import report_generator as reports
from aspect_taxonomy import STANDARD_ASPECTS, business_category_for_aspect
from i18n import t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = datetime(2026, 10, 16, 1, 30, 0, tzinfo=UTC)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in library.REQUIRED_ANALYSIS_FILES
    }


class ReportGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.library_root = self.root / "library"
        self.source_a = self._copy_demo_source("source-a")
        self.source_b = self._copy_demo_source("source-b")
        self.saved_a = self._save(
            self.source_a,
            "Amazon <Interview> TV",
            "amazon",
            "RUN_REPORT_A",
            "a001",
            imported_reviews=8,
            successful_reviews=8,
            sample_level="early_signals",
        )
        self.saved_b = self._save(
            self.source_b,
            "Demo Television Dataset",
            "demo",
            "RUN_REPORT_B",
            "b002",
            imported_reviews=15,
            successful_reviews=14,
            skipped_reviews=1,
            sample_level="limited_sample",
        )
        self.comparison_root = self._write_comparison()

    def _copy_demo_source(self, name: str) -> Path:
        destination = self.root / name
        destination.mkdir()
        for filename in library.REQUIRED_ANALYSIS_FILES:
            shutil.copy2(PROJECT_ROOT / "outputs" / filename, destination / filename)
        return destination

    def _save(
        self,
        source: Path,
        product_name: str,
        source_type: str,
        run_id: str,
        token: str,
        **facts: object,
    ) -> library.SavedAnalysis:
        metadata = {
            "analysis_run_id": run_id,
            "analysis_candidate": "generalization_v2_candidate",
            "taxonomy_version": "generalization_v2_candidate",
            "model_provider": "ollama",
            "model_name": "qwen3:8b",
            "source_url": "https://www.amazon.com/dp/B012345678" if source_type == "amazon" else None,
            **facts,
        }
        return library.save_analysis(
            source,
            product_name,
            source_type,
            metadata,
            library_root=self.library_root,
            now=FIXED_TIME,
            token_factory=lambda: token,
        )

    @staticmethod
    def _comparison_record(aspect: str) -> dict[str, object]:
        category = business_category_for_aspect(aspect)
        base: dict[str, object] = {
            "aspect": aspect,
            "business_category": category,
            "a_review_count": 0,
            "b_review_count": 0,
            "a_review_share": 0.0,
            "b_review_share": 0.0,
            "review_share_gap": 0.0,
            "a_positive_rate": None,
            "b_positive_rate": None,
            "a_negative_rate": None,
            "b_negative_rate": None,
            "positive_rate_gap": None,
            "negative_rate_gap": None,
            "a_net_sentiment": None,
            "b_net_sentiment": None,
            "net_sentiment_gap": None,
            "a_avg_rating": None,
            "b_avg_rating": None,
            "comparison_status": "insufficient_evidence",
            "comparison_evidence": "no_evidence",
            "average_review_share": 0.0,
        }
        if aspect == "Picture Quality":
            base.update(
                a_review_count=4,
                b_review_count=3,
                a_review_share=50.0,
                b_review_share=21.4,
                review_share_gap=28.6,
                a_positive_rate=90.0,
                b_positive_rate=60.0,
                a_negative_rate=10.0,
                b_negative_rate=40.0,
                positive_rate_gap=30.0,
                negative_rate_gap=-30.0,
                a_net_sentiment=80.0,
                b_net_sentiment=20.0,
                net_sentiment_gap=60.0,
                a_avg_rating=4.5,
                b_avg_rating=3.8,
                comparison_status="clear_advantage",
                comparison_evidence="Medium",
                average_review_share=35.7,
            )
        elif aspect == "Build Quality":
            base.update(
                a_review_count=2,
                b_review_count=4,
                a_review_share=25.0,
                b_review_share=28.6,
                a_positive_rate=25.0,
                b_positive_rate=75.0,
                a_negative_rate=75.0,
                b_negative_rate=25.0,
                positive_rate_gap=-50.0,
                negative_rate_gap=50.0,
                comparison_status="clear_weakness",
                comparison_evidence="Low",
                average_review_share=26.8,
            )
        elif aspect == "Remote Control":
            base.update(
                a_review_count=2,
                b_review_count=2,
                a_review_share=25.0,
                b_review_share=14.3,
                a_positive_rate=20.0,
                b_positive_rate=20.0,
                a_negative_rate=80.0,
                b_negative_rate=80.0,
                positive_rate_gap=0.0,
                negative_rate_gap=0.0,
                comparison_status="similar",
                comparison_evidence="Low",
                average_review_share=19.65,
            )
        elif aspect == "Sound Quality":
            base.update(
                b_review_count=2,
                b_review_share=14.3,
                b_positive_rate=50.0,
                b_negative_rate=50.0,
                comparison_status="insufficient_evidence",
                comparison_evidence="no_evidence",
                average_review_share=7.15,
            )
        return base

    def _write_comparison(self) -> Path:
        root = self.root / "comparison"
        root.mkdir()
        records = [self._comparison_record(aspect) for aspect in STANDARD_ASPECTS]
        by_aspect = {
            "Picture Quality": {
                "positive": {
                    "shared": [{"term": "picture quality"}],
                    "product_a_distinctive": [{"term": "bright screen"}],
                    "product_b_distinctive": [{"term": "natural color"}],
                }
            }
        }
        summary = {
            "schema_version": "step13.v1",
            "comparison_id": "CMP_20261016_012500_ab12",
            "product_a": {
                "product_name": self.saved_a.product_name,
                "source_type": "amazon",
                "successful_reviews": 8,
                "coverage": 1.0,
            },
            "product_b": {
                "product_name": self.saved_b.product_name,
                "source_type": "demo",
                "successful_reviews": 14,
                "coverage": 14 / 15,
            },
            "analysis_compatibility": {
                "compatible": True,
                "required_candidate": "generalization_v2_candidate",
                "taxonomy_version": "generalization_v2_candidate",
            },
            "sample_warnings": [{"code": "early_comparison"}],
            "head_to_head_summary": {
                "product_a_leads": 1,
                "product_b_leads": 1,
                "similar_aspects": 1,
                "insufficient_evidence_aspects": 12,
            },
            "executive_summary": {
                "product_a_advantages": ["Picture Quality"],
                "product_b_advantages": ["Build Quality"],
                "shared_pain_points": ["Remote Control"],
            },
            "aspect_comparisons": records,
            "competitive_advantages": [records[0]],
            "competitive_weaknesses": [next(row for row in records if row["aspect"] == "Build Quality")],
            "white_space_opportunities": [next(row for row in records if row["aspect"] == "Remote Control")],
            "shopping_experience_comparison": [row for row in records if row["business_category"] == "EXPERIENCE"],
            "rating_comparison": {},
            "language_comparison": {"by_aspect": by_aspect},
            "recommendations": [],
        }
        metadata = {
            "comparison_id": summary["comparison_id"],
            "product_a_analysis_id": self.saved_a.analysis_id,
            "product_b_analysis_id": self.saved_b.analysis_id,
            "analysis_a_id": self.saved_a.analysis_id,
            "analysis_b_id": self.saved_b.analysis_id,
        }
        (root / "comparison_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False), encoding="utf-8"
        )
        (root / "comparison_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        with (root / "aspect_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return root

    def _single(self, saved: library.SavedAnalysis | None = None, language: str = "en") -> reports.ReportDocument:
        return reports.build_single_product_report(
            saved or self.saved_b,
            language=language,
            generated_at=FIXED_TIME,
            report_id="RPT_20261016_013000_c003",
        )

    def _comparison(self, language: str = "en") -> reports.ReportDocument:
        return reports.build_comparison_report(
            self.comparison_root,
            product_a_root=self.saved_a,
            product_b_root=self.saved_b,
            language=language,
            generated_at=FIXED_TIME,
            report_id="CMP_RPT_20261016_013000_d004",
        )

    def test_single_report_generation(self) -> None:
        document = self._single()
        self.assertEqual("single_product", document.report_type)
        self.assertIn("Single Product Business Report", document.html)

    def test_comparison_report_generation(self) -> None:
        document = self._comparison()
        self.assertEqual("competitor_comparison", document.report_type)
        self.assertIn("Competitor Comparison Report", document.html)

    def test_english_report(self) -> None:
        html = self._single(language="en").html
        self.assertIn("Executive Summary", html)
        self.assertIn("Top Selling Points", html)

    def test_chinese_report_and_aspect_display(self) -> None:
        html = self._single(language="zh").html
        self.assertIn("单产品商业分析报告", html)
        self.assertIn("画质", html)
        self.assertNotIn(">report.", html)

    def test_source_outputs_are_not_mutated(self) -> None:
        before = _file_hashes(self.saved_b.root)
        self._single()
        self.assertEqual(before, _file_hashes(self.saved_b.root))

    def test_no_model_aggregator_keyword_or_scraper_call(self) -> None:
        forbidden = [
            patch.object(llm_client, "create_review_client", side_effect=AssertionError),
            patch.object(aggregator, "aggregate_reviews", side_effect=AssertionError),
            patch.object(keyword_analyzer, "analyze_keywords", side_effect=AssertionError),
            patch.object(scraper, "scrape_reviews", side_effect=AssertionError),
        ]
        for item in forbidden:
            item.start()
            self.addCleanup(item.stop)
        self._single()
        self._comparison()

    def test_deterministic_business_content(self) -> None:
        first = self._single()
        second = self._single()
        self.assertEqual(first.html, second.html)
        self.assertEqual(first.metadata, second.metadata)

    def test_product_name_is_html_escaped(self) -> None:
        html = self._single(self.saved_a).html
        self.assertIn("Amazon &lt;Interview&gt; TV", html)
        self.assertNotIn("Amazon <Interview> TV", html)

    def test_evidence_is_html_escaped_and_kept_verbatim(self) -> None:
        source = self._copy_demo_source("unsafe-evidence")
        payload = json.loads((source / "aspect_evidence.json").read_text(encoding="utf-8"))
        original = '<script>alert("x")</script> & exact words'
        payload["Picture Quality"]["positive"][0]["evidence"] = original
        (source / "aspect_evidence.json").write_text(json.dumps(payload), encoding="utf-8")
        saved = self._save(source, "Safe Product", "upload", "RUN_UNSAFE", "e005")
        html = self._single(saved).html
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; exact words", html)

    def test_no_local_path_leakage(self) -> None:
        document = self._single()
        combined = document.html + json.dumps(document.metadata)
        self.assertNotIn(str(self.root), combined)
        self.assertNotRegex(combined, r"/(?:Users|home|private|tmp)/")

    def test_small_sample_warning(self) -> None:
        self.assertIn(
            "This report is based on a small sample",
            self._single(self.saved_a).html,
        )

    def test_quick_scan_disclaimer(self) -> None:
        self.assertIn(
            "Quick Scan may include only reviews accessible",
            self._single(self.saved_a).html,
        )

    def test_rating_section_present(self) -> None:
        self.assertIn("Rating Insights", self._single().html)

    def test_rating_section_absent_without_ratings(self) -> None:
        source = self._copy_demo_source("no-rating")
        path = source / "analyzed_reviews.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        fields = [field for field in rows[0] if field != "rating"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{key: row[key] for key in fields} for row in rows])
        saved = self._save(source, "No Rating Product", "upload", "RUN_NO_RATING", "f006")
        self.assertNotIn("Rating Insights", self._single(saved).html)

    def test_selling_points_render(self) -> None:
        html = self._single().html
        self.assertIn("Top Selling Points", html)
        self.assertIn("Picture Quality", html)

    def test_pain_points_render(self) -> None:
        html = self._single().html
        self.assertIn("Top Customer Pain Points", html)
        self.assertIn("Build Quality", html)

    def test_comparison_table_contains_all_15_aspects(self) -> None:
        html = self._comparison().html
        for aspect in STANDARD_ASPECTS:
            self.assertIn(f"<td>{aspect}</td>", html)

    def test_one_sided_missing_evidence_is_not_false_zero(self) -> None:
        html = self._comparison().html
        row = re.search(r"<tr><td>Sound Quality</td>(.*?)</tr>", html)
        self.assertIsNotNone(row)
        self.assertIn("No evidence", row.group(1))
        self.assertIn("50.0%", row.group(1))
        self.assertNotIn("<td>0.0%</td>", row.group(1))

    def test_customer_evidence_remains_original_language(self) -> None:
        evidence = json.loads((self.saved_b.root / "aspect_evidence.json").read_text())
        phrase = evidence["Picture Quality"]["positive"][0]["evidence"]
        self.assertIn(phrase, self._single(language="zh").html)

    def test_report_metadata_schema(self) -> None:
        metadata = self._single().metadata
        expected = {
            "report_id",
            "report_type",
            "schema_version",
            "template_version",
            "language",
            "source_analysis_id",
            "comparison_id",
            "generated_at",
            "source_snapshot_hash",
            "file_hash",
        }
        self.assertEqual(expected, set(metadata))

    def test_report_id_formats(self) -> None:
        single = reports.build_single_product_report(self.saved_b)
        comparison = reports.build_comparison_report(
            self.comparison_root,
            product_a_root=self.saved_a,
            product_b_root=self.saved_b,
        )
        self.assertRegex(single.report_id, r"^RPT_\d{8}_\d{6}_[a-f0-9]{4}$")
        self.assertRegex(comparison.report_id, r"^CMP_RPT_\d{8}_\d{6}_[a-f0-9]{4}$")

    def test_template_version(self) -> None:
        self.assertEqual("step14.report.v1", self._single().metadata["template_version"])

    def test_snapshot_integrity_guard(self) -> None:
        (self.saved_b.root / "business_summary.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(reports.ReportIntegrityError):
            self._single()

    def test_comparison_integrity_guard(self) -> None:
        (self.saved_a.root / "aspect_summary.csv").write_text("changed", encoding="utf-8")
        with self.assertRaises(reports.ReportIntegrityError):
            self._comparison()

    def test_comparison_identity_mismatch_is_blocked(self) -> None:
        path = self.comparison_root / "comparison_metadata.json"
        payload = json.loads(path.read_text())
        payload["product_a_analysis_id"] = "AN_20261016_000000_dead"
        payload["analysis_a_id"] = "AN_20261016_000000_dead"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(reports.ReportIntegrityError):
            self._comparison()

    def test_report_i18n_keys_are_complete(self) -> None:
        for translation_key in reports._REPORT_I18N_KEYS.values():
            for language in ("en", "zh"):
                self.assertNotEqual(translation_key, t(translation_key, language))

    def test_print_css_present(self) -> None:
        html = self._single().html
        self.assertIn("@media print", html)
        self.assertIn("page-break-inside:avoid", html)
        self.assertIn("thead{display:table-header-group}", html)

    def test_long_metric_values_wrap_inside_cover_cards(self) -> None:
        html = self._single().html
        self.assertIn("overflow-wrap:anywhere", html)
        self.assertIn("word-break:break-word", html)

    def test_report_size_guard(self) -> None:
        with patch.object(reports, "MAX_REPORT_BYTES", 100):
            with self.assertRaises(reports.ReportSizeError):
                self._single()

    def test_standalone_html_has_no_external_resources(self) -> None:
        html = self._single().html
        self.assertIn("<style>", html)
        self.assertNotRegex(html, r"<(?:script|iframe|object|embed)\b")
        self.assertNotRegex(html, r"\b(?:src|href)\s*=")
        self.assertNotIn("localhost", html)

    def test_export_report_writes_html_and_metadata(self) -> None:
        document = self._single()
        exported = reports.export_report(document, output_root=self.root / "reports")
        self.assertEqual(document.html, exported.html_path.read_text(encoding="utf-8"))
        metadata = json.loads(exported.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(document.metadata, metadata)

    def test_export_does_not_mutate_snapshot(self) -> None:
        before = _file_hashes(self.saved_b.root)
        reports.export_report(self._single(), output_root=self.root / "reports")
        self.assertEqual(before, _file_hashes(self.saved_b.root))

    def test_comparison_declares_no_overall_winner(self) -> None:
        html = self._comparison().html
        self.assertIn("No overall winner is declared", html)
        self.assertNotIn("Winner: Product", html)
        self.assertNotIn("better overall", html)

    def test_comparison_metadata_references_both_analysis_ids(self) -> None:
        metadata = self._comparison().metadata
        self.assertEqual(
            {
                "product_a": self.saved_a.analysis_id,
                "product_b": self.saved_b.analysis_id,
            },
            metadata["source_analysis_id"],
        )

    def test_current_amazon_workspace_detects_quick_scan_without_manifest(self) -> None:
        current = self._copy_demo_source("current-amazon")
        (current / "scrape_metadata.json").write_text("{}", encoding="utf-8")
        document = reports.build_single_product_report(
            current,
            validate_integrity=False,
            product_name="Current Amazon",
            generated_at=FIXED_TIME,
            report_id="RPT_20261016_013000_e005",
        )
        self.assertIn("Quick Scan may include only reviews accessible", document.html)
        self.assertIn("generalization_v2_candidate", document.html)
        self.assertNotIn(">unknown<", document.html)


if __name__ == "__main__":
    unittest.main()
