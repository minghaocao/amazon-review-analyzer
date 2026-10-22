"""Offline behavior guards for Step 8.12 Generalization Optimization V2."""

from __future__ import annotations

import re
import unittest

from aspect_taxonomy import (
    ASPECT_DEFINITIONS,
    STANDARD_ASPECTS,
    normalize_aspect,
    taxonomy_definition_text,
)
from llm_client import _analysis_instructions
from review_analyzer import analyze_review


class _StaticClient:
    def __init__(self, aspects: list[dict[str, str]]) -> None:
        self.response = {"summary": "Synthetic offline analysis.", "aspects": aspects}

    def analyze_review(self, review_text: str, allowed_aspects):
        return self.response


def _mention(
    raw_aspect: str,
    aspect: str,
    sentiment: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "raw_aspect": raw_aspect,
        "aspect": aspect,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _analyze(review: str, aspects: list[dict[str, str]]):
    return analyze_review(review, llm_client=_StaticClient(aspects))


class Step812PromptAndTaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt = " ".join(_analysis_instructions(STANDARD_ASPECTS).split())
        self.guidance = " ".join(taxonomy_definition_text().split())

    def test_opinion_unit_rejects_presence_inventory_and_specification_facts(self) -> None:
        self.assertIn("one concrete opinion target and one evaluative predicate", self.prompt)
        self.assertIn("Mere presence, inventory, specification", self.prompt)
        self.assertIn("Neutral is still an evaluative judgment", self.prompt)
        self.assertIn("fact into an Aspect", self.prompt)

    def test_precision_and_completeness_guards_are_balanced(self) -> None:
        self.assertIn("Does the evidence actually evaluate this exact Aspect?", self.prompt)
        self.assertIn("Would another taxonomy label describe", self.prompt)
        self.assertIn("only because a related word appeared", self.prompt)
        self.assertIn("After the precision guard", self.prompt)
        self.assertIn("without re-adding facts", self.prompt)

    def test_taxonomy_defines_customer_service_and_other(self) -> None:
        self.assertIn("Customer Service", ASPECT_DEFINITIONS)
        self.assertIn("Other", ASPECT_DEFINITIONS)
        self.assertIn("promised replies", self.guidance)
        self.assertIn("no more direct standard taxonomy label", self.guidance)
        self.assertIn("never a fallback", self.guidance)

    def test_boundaries_cover_design_packaging_build_and_price(self) -> None:
        self.assertIn("functional layout", self.guidance)
        self.assertIn("commercial bundle or total package", self.guidance)
        self.assertIn("flex, bowing, warping", self.guidance)
        self.assertIn("relationship between cost and benefits", self.guidance)

    def test_setup_ease_and_connectivity_use_task_lifecycle(self) -> None:
        self.assertIn("Judge Setup / Installation versus Ease of Use by the task lifecycle", self.prompt)
        self.assertIn("routine daily use is Ease of Use", self.prompt)
        self.assertIn("pairing is Connectivity", self.prompt)

    def test_comparison_guard_resolves_each_clause_referent(self) -> None:
        self.assertIn("Resolve the subject of each clause", self.prompt)
        self.assertIn("never for an old, alternative, or comparison product", self.prompt)
        self.assertIn("Keep an evaluation of the current item", self.prompt)

    def test_granularity_preserves_independent_claims_without_adjective_splitting(self) -> None:
        self.assertIn("Do not over-split one coherent opinion", self.prompt)
        self.assertIn("Two adjectives do not automatically mean two mentions", self.prompt)
        self.assertIn("different time periods, or opposite polarities", self.prompt)

    def test_prompt_contains_no_v2_case_identifier(self) -> None:
        self.assertIsNone(re.search(r"\bV2D\d{3}\b", self.prompt))


class Step812NormalizationAndPostprocessingTests(unittest.TestCase):
    def test_weak_screen_word_does_not_override_valid_setup_label(self) -> None:
        self.assertEqual(
            "Setup / Installation",
            normalize_aspect("first-time account screen", "Setup / Installation"),
        )

    def test_weak_speaker_word_does_not_override_build_label(self) -> None:
        self.assertEqual(
            "Build Quality",
            normalize_aspect("speaker housing", "Build Quality"),
        )

    def test_commercial_package_does_not_override_price_label(self) -> None:
        self.assertEqual(
            "Price / Value",
            normalize_aspect("total package", "Price / Value"),
        )

    def test_bluetooth_pairing_outranks_incorrect_setup_suggestion(self) -> None:
        self.assertEqual(
            "Connectivity",
            normalize_aspect("Bluetooth pairing", "Setup / Installation"),
        )

    def test_strong_remote_and_picture_aliases_still_correct_bad_suggestions(self) -> None:
        self.assertEqual(
            "Remote Control",
            normalize_aspect("remote volume button", "Design"),
        )
        self.assertEqual(
            "Picture Quality",
            normalize_aspect("screen quality", "Design"),
        )

    def test_price_fact_is_rejected_but_open_ended_value_judgment_is_kept(self) -> None:
        fact = "The invoice records a payment of $389."
        fact_result = _analyze(
            fact,
            [_mention("payment", "Price / Value", "neutral", fact)],
        )
        self.assertEqual([], fact_result["aspects"])

        opinion = "The benefits easily justify the higher cost."
        opinion_result = _analyze(
            opinion,
            [_mention("higher cost", "Price / Value", "positive", opinion)],
        )
        self.assertEqual("Price / Value", opinion_result["aspects"][0]["aspect"])

    def test_value_bundle_is_price_not_physical_packaging(self) -> None:
        review = "The bundle feels overpriced for its limited functions."
        result = _analyze(
            review,
            [_mention("bundle", "Price / Value", "negative", review)],
        )
        self.assertEqual("Price / Value", result["aspects"][0]["aspect"])

    def test_picture_build_and_sound_are_classified_by_predicate(self) -> None:
        picture = "Movie motion remains crisp during fast pans."
        picture_result = _analyze(
            picture,
            [_mention("Movie motion", "System Performance", "positive", picture)],
        )
        self.assertEqual("Picture Quality", picture_result["aspects"][0]["aspect"])

        build = "The speaker housing flexes under a finger."
        build_result = _analyze(
            build,
            [_mention("speaker housing", "Sound Quality", "negative", build)],
        )
        self.assertEqual("Build Quality", build_result["aspects"][0]["aspect"])

        sound = "Voices from the speakers are muffled."
        sound_result = _analyze(
            sound,
            [_mention("speakers", "Sound Quality", "negative", sound)],
        )
        self.assertEqual("Sound Quality", sound_result["aspects"][0]["aspect"])

    def test_packaging_relabel_does_not_hijack_another_clause(self) -> None:
        review = "The parcel arrived late, while thick foam protected every corner."
        result = _analyze(
            review,
            [
                _mention("parcel", "Delivery", "negative", review),
                _mention("thick foam", "Packaging", "positive", "thick foam protected every corner"),
            ],
        )
        self.assertEqual(
            ["Delivery", "Packaging"],
            [item["aspect"] for item in result["aspects"]],
        )

    def test_concrete_other_is_kept_and_attribute_free_other_is_removed(self) -> None:
        review = "The status light is distractingly bright at night."
        result = _analyze(
            review,
            [_mention("status light", "Other", "negative", review)],
        )
        self.assertEqual("Other", result["aspects"][0]["aspect"])

        vague = "This product is fine."
        vague_result = _analyze(
            vague,
            [_mention("product", "Other", "positive", vague)],
        )
        self.assertEqual([], vague_result["aspects"])

    def test_comparison_product_opinion_is_removed_but_current_product_is_kept(self) -> None:
        review = "My old monitor froze, but this one is straightforward to navigate."
        result = _analyze(
            review,
            [
                _mention("old monitor", "System Performance", "negative", "My old monitor froze"),
                _mention("this one", "Ease of Use", "positive", "this one is straightforward to navigate"),
            ],
        )
        self.assertEqual(["Ease of Use"], [item["aspect"] for item in result["aspects"]])

    def test_same_aspect_independent_mentions_remain_separate(self) -> None:
        review = "Wi-Fi stays steady, and Bluetooth reconnects quickly."
        result = _analyze(
            review,
            [
                _mention("Wi-Fi", "Connectivity", "positive", "Wi-Fi stays steady"),
                _mention("Bluetooth", "Connectivity", "positive", "Bluetooth reconnects quickly"),
            ],
        )
        self.assertEqual(2, len(result["aspects"]))
        self.assertEqual(
            ["Wi-Fi stays steady", "Bluetooth reconnects quickly"],
            [item["evidence"] for item in result["aspects"]],
        )


if __name__ == "__main__":
    unittest.main()
