"""Offline regression tests for Step 8.6 multi-opinion recall rules."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from aspect_taxonomy import STANDARD_ASPECTS, normalize_aspect, taxonomy_definition_text
from llm_client import _analysis_instructions, _response_schema
from review_analyzer import analyze_review


class _StaticClient:
    def __init__(self, aspects: list[dict[str, str]]) -> None:
        self._response = {"summary": "Several opinions are present.", "aspects": aspects}

    def analyze_review(self, review_text: str, allowed_aspects: tuple[str, ...]):
        return self._response


class _RepairingClient(_StaticClient):
    def repair_evidence(self, review_text: str, invalid_aspects):
        return {
            "repairs": [
                {
                    "aspect_index": invalid_aspects[0]["aspect_index"],
                    "evidence": "Wi-Fi drops nightly",
                }
            ]
        }


class Step86MultiAspectTests(unittest.TestCase):
    def test_prompt_requires_atomic_same_aspect_mentions_and_completeness_scan(self) -> None:
        prompt = " ".join(_analysis_instructions(("Connectivity",)).split())
        self.assertIn("One output item represents one atomic customer judgment", prompt)
        self.assertIn("same normalized Aspect may therefore appear more than once", prompt)
        self.assertIn("one mention per independently supportable target-predicate pair", prompt)
        self.assertIn("Final cardinality rule", prompt)
        self.assertIn("Do not stop after the first item for a category", prompt)
        self.assertIn("silently perform a completeness check", prompt)
        self.assertIn("Do not output this checklist or reasoning", prompt)

    def test_json_schema_reinforces_atomic_cardinality_and_evidence(self) -> None:
        schema = _response_schema(STANDARD_ASPECTS)
        aspects = schema["properties"]["aspects"]["description"]
        evidence = schema["$defs"]["ReviewAspectOutput"]["properties"][
            "evidence"
        ]["description"]
        self.assertIn("Repeat a taxonomy label", aspects)
        self.assertIn("distinct evidence", aspects)
        self.assertIn("Shortest informative contiguous verbatim review span", evidence)

    def test_prompt_preserves_multi_aspect_clause_and_mixed_compatibility(self) -> None:
        prompt = " ".join(
            _analysis_instructions(("Setup / Installation", "Ease of Use")).split()
        )
        self.assertIn("support more than one different Aspect", prompt)
        self.assertIn("Never output mixed", prompt)
        self.assertIn("one positive and one negative", prompt)

    def test_prompt_distinguishes_packaging_from_operational_context(self) -> None:
        prompt = " ".join(
            _analysis_instructions(("Packaging", "Reliability")).split()
        )
        self.assertIn("by itself it is not a Packaging opinion", prompt)
        self.assertIn("physical package, packing materials", prompt)

    def test_prompt_requires_cost_benefit_semantics_for_implicit_value(self) -> None:
        prompt = " ".join(_analysis_instructions(("Price / Value",)).split())
        self.assertIn("benefits received relative to cost", prompt)
        self.assertIn("Generic praise", prompt)
        self.assertIn("Premium, high-end, or top-tier quality wording alone", prompt)

    def test_prompt_excludes_comparison_items_and_attribute_free_other(self) -> None:
        prompt = " ".join(
            _analysis_instructions(("Connectivity", "Reliability", "Other")).split()
        )
        self.assertIn("never for an old, alternative, or comparison product", prompt)
        self.assertIn("Other is not a catch-all for attribute-free sentiment", prompt)
        self.assertIn("do not create Reliability", prompt)
        self.assertIn("Final precision guard", prompt)

    def test_distinct_same_aspect_same_sentiment_mentions_are_preserved(self) -> None:
        review = "Audio is muffled at low levels, and the bass rattles when loud."
        aspects = [
            {
                "raw_aspect": "Audio",
                "aspect": "Sound Quality",
                "sentiment": "negative",
                "evidence": "Audio is muffled at low levels",
            },
            {
                "raw_aspect": "bass",
                "aspect": "Sound Quality",
                "sentiment": "negative",
                "evidence": "the bass rattles when loud",
            },
        ]
        result = analyze_review(review, llm_client=_StaticClient(aspects))
        self.assertEqual(2, len(result["aspects"]))
        self.assertEqual(
            ["Audio is muffled at low levels", "the bass rattles when loud"],
            [item["evidence"] for item in result["aspects"]],
        )

    def test_exact_duplicate_mention_is_removed(self) -> None:
        review = "The wireless link drops each evening."
        item = {
            "raw_aspect": "wireless link",
            "aspect": "Connectivity",
            "sentiment": "negative",
            "evidence": "The wireless link drops each evening",
        }
        duplicate = {**item, "raw_aspect": "wireless connection"}
        result = analyze_review(review, llm_client=_StaticClient([item, duplicate]))
        self.assertEqual([item], result["aspects"])

    def test_same_evidence_with_distinct_aspects_is_not_blindly_collapsed(self) -> None:
        review = "Initial setup was straightforward."
        aspects = [
            {
                "raw_aspect": "Initial setup",
                "aspect": "Setup / Installation",
                "sentiment": "positive",
                "evidence": review,
            },
            {
                "raw_aspect": "straightforward",
                "aspect": "Ease of Use",
                "sentiment": "positive",
                "evidence": review,
            },
        ]
        result = analyze_review(review, llm_client=_StaticClient(aspects))
        self.assertEqual(2, len(result["aspects"]))

    def test_repaired_mention_is_deduplicated_after_evidence_validation(self) -> None:
        review = "Wi-Fi drops nightly."
        aspects = [
            {
                "raw_aspect": "Wi-Fi",
                "aspect": "Connectivity",
                "sentiment": "negative",
                "evidence": "Wi-Fi drops nightly",
            },
            {
                "raw_aspect": "wireless connection",
                "aspect": "Connectivity",
                "sentiment": "negative",
                "evidence": "The wireless connection drops every night",
            },
        ]
        result = analyze_review(review, llm_client=_RepairingClient(aspects))
        self.assertEqual(1, len(result["aspects"]))
        self.assertEqual("Wi-Fi drops nightly", result["aspects"][0]["evidence"])

    def test_operational_out_of_box_phrase_can_follow_reliability_suggestion(self) -> None:
        self.assertEqual(
            "Reliability",
            normalize_aspect("unit was defective out of the box", "Packaging"),
        )
        self.assertNotEqual("Packaging", normalize_aspect("out of the box"))

    def test_real_package_condition_still_normalizes_to_packaging(self) -> None:
        self.assertEqual("Packaging", normalize_aspect("damaged box"))
        self.assertEqual("Packaging", normalize_aspect("protective foam was poor"))

    def test_implicit_value_alias_is_narrow_and_generic_praise_remains_other(self) -> None:
        self.assertEqual("Price / Value", normalize_aspect("for what you get"))
        self.assertEqual("Other", normalize_aspect("great product"))

    def test_taxonomy_guidance_documents_packaging_boundary(self) -> None:
        guidance = taxonomy_definition_text()
        self.assertIn("idiom 'out of the box' is not Packaging", guidance)

    def test_comparison_only_product_opinion_is_removed(self) -> None:
        review = (
            "A former device stayed online, whereas this unit disconnects daily."
        )
        aspects = [
            {
                "raw_aspect": "former device",
                "aspect": "Connectivity",
                "sentiment": "positive",
                "evidence": "A former device stayed online",
            },
            {
                "raw_aspect": "this unit",
                "aspect": "Connectivity",
                "sentiment": "negative",
                "evidence": "this unit disconnects daily",
            },
        ]
        result = analyze_review(review, llm_client=_StaticClient(aspects))
        self.assertEqual(1, len(result["aspects"]))
        self.assertEqual("this unit disconnects daily", result["aspects"][0]["evidence"])

    def test_price_value_without_cost_benefit_support_is_removed(self) -> None:
        review = "The image is crisp, but it is not elite enough to call premium."
        aspects = [
            {
                "raw_aspect": "image",
                "aspect": "Picture Quality",
                "sentiment": "positive",
                "evidence": "The image is crisp",
            },
            {
                "raw_aspect": "not elite enough to call premium",
                "aspect": "Price / Value",
                "sentiment": "negative",
                "evidence": "not elite enough to call premium",
            },
        ]
        result = analyze_review(review, llm_client=_StaticClient(aspects))
        self.assertEqual(["Picture Quality"], [item["aspect"] for item in result["aspects"]])

    def test_implicit_cost_benefit_price_value_is_kept(self) -> None:
        review = "For what you get, this monitor is an excellent purchase."
        aspect = {
            "raw_aspect": "For what you get",
            "aspect": "Price / Value",
            "sentiment": "positive",
            "evidence": review,
        }
        result = analyze_review(review, llm_client=_StaticClient([aspect]))
        self.assertEqual(["Price / Value"], [item["aspect"] for item in result["aspects"]])

    def test_minimum_acceptability_does_not_become_separate_praise(self) -> None:
        review = "The interface is slow, although it remains usable for simple tasks."
        aspects = [
            {
                "raw_aspect": "interface",
                "aspect": "System Performance",
                "sentiment": "negative",
                "evidence": "The interface is slow",
            },
            {
                "raw_aspect": "remains usable",
                "aspect": "Ease of Use",
                "sentiment": "positive",
                "evidence": "it remains usable for simple tasks",
            },
        ]
        result = analyze_review(review, llm_client=_StaticClient(aspects))
        self.assertEqual(1, len(result["aspects"]))
        self.assertEqual("negative", result["aspects"][0]["sentiment"])

    def test_unused_customer_service_reference_is_not_an_opinion(self) -> None:
        review = "Assembly was quick without calling customer support."
        aspects = [
            {
                "raw_aspect": "Assembly",
                "aspect": "Setup / Installation",
                "sentiment": "positive",
                "evidence": "Assembly was quick",
            },
            {
                "raw_aspect": "customer support",
                "aspect": "Customer Service",
                "sentiment": "positive",
                "evidence": "without calling customer support",
            },
        ]
        result = analyze_review(review, llm_client=_StaticClient(aspects))
        self.assertEqual(["Setup / Installation"], [item["aspect"] for item in result["aspects"]])

    def test_attribute_free_other_with_no_target_is_removed(self) -> None:
        review = "Pleasantly as anticipated."
        aspect = {
            "raw_aspect": "Pleasantly as anticipated",
            "aspect": "Other",
            "sentiment": "positive",
            "evidence": review,
        }
        result = analyze_review(review, llm_client=_StaticClient([aspect]))
        self.assertEqual([], result["aspects"])

    def test_factual_price_match_without_value_judgment_is_removed(self) -> None:
        review = "It was $250, exactly the amount advertised at checkout."
        aspect = {
            "raw_aspect": "amount",
            "aspect": "Price / Value",
            "sentiment": "positive",
            "evidence": review,
        }
        result = analyze_review(review, llm_client=_StaticClient([aspect]))
        self.assertEqual([], result["aspects"])

    def test_with_clause_elaboration_is_merged(self) -> None:
        review = "The image is vivid, with colors that remain accurate."
        aspects = [
            {
                "raw_aspect": "image",
                "aspect": "Picture Quality",
                "sentiment": "positive",
                "evidence": "The image is vivid",
            },
            {
                "raw_aspect": "colors",
                "aspect": "Picture Quality",
                "sentiment": "positive",
                "evidence": "colors that remain accurate",
            },
        ]
        result = analyze_review(review, llm_client=_StaticClient(aspects))
        self.assertEqual(1, len(result["aspects"]))
        self.assertEqual(
            "The image is vivid, with colors that remain accurate",
            result["aspects"][0]["evidence"],
        )

    def test_one_failure_state_is_merged_but_time_change_remains_separate(self) -> None:
        immediate_review = "The unit was defective and would not start."
        immediate = [
            {
                "raw_aspect": "defective",
                "aspect": "Reliability",
                "sentiment": "negative",
                "evidence": "The unit was defective",
            },
            {
                "raw_aspect": "would not start",
                "aspect": "Reliability",
                "sentiment": "negative",
                "evidence": "would not start",
            },
        ]
        merged = analyze_review(immediate_review, llm_client=_StaticClient(immediate))
        self.assertEqual(1, len(merged["aspects"]))

        time_review = "It worked well for months, then stopped working."
        time_aspects = [
            {
                "raw_aspect": "worked well",
                "aspect": "Reliability",
                "sentiment": "positive",
                "evidence": "It worked well for months",
            },
            {
                "raw_aspect": "stopped working",
                "aspect": "Reliability",
                "sentiment": "negative",
                "evidence": "stopped working",
            },
        ]
        separate = analyze_review(time_review, llm_client=_StaticClient(time_aspects))
        self.assertEqual(["positive", "negative"], [item["sentiment"] for item in separate["aspects"]])

    def test_explicit_price_framed_feature_comparison_derives_value(self) -> None:
        review = "Compared with rival monitors at this price, the display is brighter."
        aspect = {
            "raw_aspect": "display",
            "aspect": "Picture Quality",
            "sentiment": "positive",
            "evidence": "the display is brighter",
        }
        result = analyze_review(review, llm_client=_StaticClient([aspect]))
        self.assertEqual(
            ["Price / Value", "Picture Quality"],
            [item["aspect"] for item in result["aspects"]],
        )
        self.assertIn(result["aspects"][0]["evidence"], review)

    def test_runtime_does_not_copy_long_gold_review_fragments(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runtime = "\n".join(
            (root / name).read_text(encoding="utf-8")
            for name in ("llm_client.py", "aspect_taxonomy.py", "review_analyzer.py")
        )
        runtime_words = " ".join(re.findall(r"[a-z0-9']+", runtime.casefold()))
        cases = json.loads(
            (root / "evaluation" / "gold_cases.json").read_text(encoding="utf-8")
        )

        for case in cases:
            words = re.findall(r"[a-z0-9']+", case["review"].casefold())
            for index in range(max(0, len(words) - 7)):
                fragment = " ".join(words[index : index + 8])
                with self.subTest(case_id=case["case_id"], fragment=fragment):
                    self.assertNotIn(fragment, runtime_words)


if __name__ == "__main__":
    unittest.main()
