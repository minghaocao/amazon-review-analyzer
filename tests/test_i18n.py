"""Tests for the display-only Step 12.5 localization layer."""

from __future__ import annotations

import copy
import string
import unittest

from config import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
from i18n import (
    ASPECT_DISPLAY_KEYS,
    TRANSLATIONS,
    display_aspect,
    display_analysis_source_type,
    display_business_category,
    display_comparison_signal,
    display_comparison_status,
    display_duration,
    display_evidence_level,
    display_quadrant,
    display_recommendation,
    display_sample_level,
    display_sentiment,
    display_status,
    localized_executive_summary,
    localized_comparison_recommendation,
    localized_comparison_summary,
    localized_error_message,
    localized_recommendation,
    normalize_language,
    t,
    t_count,
)


class TranslationLookupTests(unittest.TestCase):
    def test_default_language_is_english(self) -> None:
        self.assertEqual(DEFAULT_LANGUAGE, "en")
        self.assertEqual(t("tab.overview"), "Overview")

    def test_supported_languages_are_centralized(self) -> None:
        self.assertEqual(SUPPORTED_LANGUAGES, {"English": "en", "中文": "zh"})

    def test_language_label_and_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_language("English"), "en")
        self.assertEqual(normalize_language("中文"), "zh")
        self.assertEqual(normalize_language("zh-CN"), "zh")
        self.assertEqual(normalize_language("unsupported"), "en")

    def test_switching_language_changes_display_text(self) -> None:
        self.assertEqual(t("tab.product_insights", "en"), "Product Insights")
        self.assertEqual(t("tab.product_insights", "zh"), "产品洞察")

    def test_missing_chinese_key_falls_back_to_english(self) -> None:
        original = TRANSLATIONS["zh"].pop("app.title")
        try:
            self.assertEqual(t("app.title", "zh"), "Amazon Review Intelligence")
        finally:
            TRANSLATIONS["zh"]["app.title"] = original

    def test_unknown_key_falls_back_to_key(self) -> None:
        self.assertEqual(t("future.untranslated.key", "zh"), "future.untranslated.key")

    def test_dynamic_placeholders_follow_language_order(self) -> None:
        self.assertEqual(
            t("runtime.review_progress", "en", completed=2, total=5),
            "2 of 5 reviews processed",
        )
        self.assertEqual(
            t("runtime.review_progress", "zh", completed=2, total=5),
            "已处理 2 / 5 条评论",
        )

    def test_count_helper_handles_english_plural_and_chinese(self) -> None:
        self.assertEqual(
            t_count("runtime.success_count", "runtime.success_count_plural", 1, "en"),
            "1 review analyzed successfully.",
        )
        self.assertEqual(
            t_count("runtime.success_count", "runtime.success_count_plural", 2, "en"),
            "2 reviews analyzed successfully.",
        )
        self.assertEqual(
            t_count("runtime.success_count", "runtime.success_count_plural", 2, "zh"),
            "已成功分析 2 条评论。",
        )

    def test_all_english_keys_have_chinese_translations(self) -> None:
        self.assertEqual(set(TRANSLATIONS["en"]), set(TRANSLATIONS["zh"]))

    def test_translation_placeholders_match(self) -> None:
        formatter = string.Formatter()

        def placeholders(template: str) -> set[str]:
            return {
                field_name
                for _, field_name, _, _ in formatter.parse(template)
                if field_name
            }

        for key, english in TRANSLATIONS["en"].items():
            with self.subTest(key=key):
                self.assertEqual(
                    placeholders(english),
                    placeholders(TRANSLATIONS["zh"][key]),
                )

    def test_required_dashboard_and_runtime_keys_exist(self) -> None:
        required = {
            "tab.overview",
            "tab.product_insights",
            "tab.customer_language",
            "tab.aspect_deep_dive",
            "tab.review_explorer",
            "runtime.analysis_progress",
            "runtime.retry_failed",
            "amazon.quick_scan_disclaimer",
            "overview.executive_summary",
            "bi.priority_matrix",
            "language.by_aspect",
            "deep_dive.customer_evidence",
            "explorer.original_review",
        }
        self.assertTrue(required.issubset(TRANSLATIONS["en"]))

    def test_duration_is_localized_without_changing_numeric_input(self) -> None:
        self.assertEqual(display_duration(125, "en"), "2m 5s")
        self.assertEqual(display_duration(125, "zh"), "2 分钟 5 秒")
        self.assertEqual(display_duration(None, "zh"), "正在估算剩余时间……")

    def test_localized_error_keeps_english_diagnostic_and_uses_chinese_copy(self) -> None:
        error = ValueError("original diagnostic")
        self.assertEqual(
            localized_error_message(error, "amazon.invalid_url", "en"),
            "original diagnostic",
        )
        self.assertEqual(
            localized_error_message(error, "amazon.invalid_url", "zh"),
            "请输入有效且受支持的 Amazon 商品链接。",
        )


class CanonicalDisplayMappingTests(unittest.TestCase):
    def test_all_required_aspects_have_chinese_display_names(self) -> None:
        expected = {
            "Picture Quality": "画质",
            "Sound Quality": "音质",
            "Price / Value": "价格 / 性价比",
            "Remote Control": "遥控器",
            "System Performance": "系统性能",
            "Ease of Use": "易用性",
            "Design": "外观设计",
            "Build Quality": "做工质量",
            "Reliability": "可靠性",
            "Connectivity": "连接性",
            "Setup / Installation": "设置 / 安装",
            "Delivery": "配送",
            "Packaging": "包装",
            "Customer Service": "客户服务",
            "Other": "其他",
        }
        self.assertEqual(set(ASPECT_DISPLAY_KEYS), set(expected))
        for canonical, localized in expected.items():
            with self.subTest(aspect=canonical):
                self.assertEqual(display_aspect(canonical, "en"), canonical)
                self.assertEqual(display_aspect(canonical, "zh"), localized)

    def test_unknown_aspect_is_preserved(self) -> None:
        self.assertEqual(display_aspect("Future Aspect", "zh"), "Future Aspect")

    def test_sentiment_mapping(self) -> None:
        self.assertEqual(display_sentiment("positive", "en"), "Positive")
        self.assertEqual(display_sentiment("positive", "zh"), "正面")
        self.assertEqual(display_sentiment("negative", "zh"), "负面")
        self.assertEqual(display_sentiment("neutral", "zh"), "中性")

    def test_business_category_mapping(self) -> None:
        self.assertEqual(display_business_category("PRODUCT", "en"), "Product Experience")
        self.assertEqual(display_business_category("PRODUCT", "zh"), "产品体验")
        self.assertEqual(display_business_category("EXPERIENCE", "zh"), "购物体验")
        self.assertEqual(display_business_category("OTHER", "zh"), "其他")

    def test_evidence_recommendation_quadrant_and_status_mapping(self) -> None:
        self.assertEqual(display_evidence_level("High", "zh"), "高")
        self.assertEqual(display_recommendation("Marketing Opportunity", "zh"), "营销机会")
        self.assertEqual(display_quadrant("Improvement Priority", "zh"), "重点改进")
        self.assertEqual(display_status("success", "zh"), "成功")
        self.assertEqual(display_sample_level("limited_sample", "zh"), "有限样本")

    def test_display_helpers_do_not_change_canonical_values(self) -> None:
        record = {
            "aspect": "Picture Quality",
            "sentiment": "positive",
            "business_category": "PRODUCT",
            "evidence_level": "High",
        }
        original = copy.deepcopy(record)
        display_aspect(record["aspect"], "zh")
        display_sentiment(record["sentiment"], "zh")
        display_business_category(record["business_category"], "zh")
        display_evidence_level(record["evidence_level"], "zh")
        self.assertEqual(record, original)

    def test_comparison_status_and_signal_are_localized_only_for_display(self) -> None:
        self.assertEqual(display_comparison_status("clear_advantage", "en"), "Clear Advantage")
        self.assertEqual(display_comparison_status("clear_advantage", "zh"), "明显优势")
        self.assertEqual(display_comparison_signal("competitive_risk", "zh"), "竞争风险")

    def test_saved_analysis_source_types_are_localized_for_display_only(self) -> None:
        self.assertEqual(display_analysis_source_type("amazon", "en"), "Amazon Quick Scan")
        self.assertEqual(display_analysis_source_type("amazon", "zh"), "Amazon 快速扫描")
        self.assertEqual(display_analysis_source_type("paste", "zh"), "粘贴评论")
        self.assertEqual(display_analysis_source_type("upload", "zh"), "上传数据集")
        self.assertEqual(display_analysis_source_type("demo", "zh"), "演示数据")

    def test_no_evidence_is_explicit_and_localized(self) -> None:
        self.assertEqual(display_evidence_level("no_evidence", "en"), "No Evidence")
        self.assertEqual(display_evidence_level("no_evidence", "zh"), "暂无证据")


class LocalizedBusinessTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intelligence = {
            "sample_level": {"key": "limited_sample"},
            "selling_points": [{"aspect": "Picture Quality"}],
            "pain_points": [
                {"aspect": "Remote Control"},
                {"aspect": "System Performance"},
            ],
        }

    def test_executive_summary_has_independent_templates(self) -> None:
        english = localized_executive_summary(self.intelligence, "en")
        chinese = localized_executive_summary(self.intelligence, "zh")
        self.assertEqual(
            english,
            "Within this limited sample, customers most frequently praise Picture Quality, while the strongest negative feedback concerns Remote Control and System Performance.",
        )
        self.assertEqual(
            chinese,
            "在当前有限样本中，客户最常认可画质，而遥控器、系统性能产生了最强的负面反馈。",
        )

    def test_empty_executive_summary_is_localized(self) -> None:
        self.assertIn("does not provide enough", localized_executive_summary({}, "en"))
        self.assertIn("尚未提供足够", localized_executive_summary({}, "zh"))

    def test_recommendation_has_independent_templates(self) -> None:
        record = {
            "aspect": "Picture Quality",
            "type": "Marketing Opportunity",
            "message": "canonical English persisted message",
        }
        original = copy.deepcopy(record)
        self.assertEqual(
            localized_recommendation(record, "zh"),
            {
                "type": "营销机会",
                "message": "可考虑在商品详情文案和广告素材中重点突出画质优势。",
            },
        )
        self.assertEqual(record, original)

    def test_comparison_summary_and_recommendation_are_independently_localized(self) -> None:
        payload = {
            "product_a": {"product_name": "Aurora"},
            "product_b": {"product_name": "Nimbus"},
            "competitive_advantages": [{"aspect": "Picture Quality"}],
            "competitive_weaknesses": [{"aspect": "Price / Value"}],
            "white_space_opportunities": [{"aspect": "System Performance"}],
            "executive_summary": {
                "product_a_advantages": ["Picture Quality"],
                "product_b_advantages": ["Price / Value"],
                "shared_pain_points": ["System Performance"],
            },
        }
        english = localized_comparison_summary(payload, "en")
        chinese = localized_comparison_summary(payload, "zh")
        self.assertIn("Picture Quality", english)
        self.assertIn("画质", chinese)
        self.assertIn("系统性能", chinese)
        recommendation = localized_comparison_recommendation(
            {"type": "protect_advantage", "aspect": "Picture Quality"},
            "zh",
            product_a_name="Aurora",
            product_b_name="Nimbus",
        )
        self.assertEqual("巩固优势", recommendation["type"])
        self.assertIn("画质", recommendation["message"])


if __name__ == "__main__":
    unittest.main()
