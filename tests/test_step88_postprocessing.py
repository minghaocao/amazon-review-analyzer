"""Offline regression tests for Step 8.8 deterministic boundary guards."""

from __future__ import annotations

import unittest
from copy import deepcopy

from review_analyzer import analyze_review


class _StaticClient:
    def __init__(self, aspects: list[dict[str, str]]) -> None:
        self.response = {"summary": "The review contains specific opinions.", "aspects": aspects}

    def analyze_review(self, review_text: str, allowed_aspects: tuple[str, ...]):
        return deepcopy(self.response)


def _aspect(
    raw_aspect: str,
    suggested: str,
    sentiment: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "raw_aspect": raw_aspect,
        "aspect": suggested,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _analyze(review: str, aspects: list[dict[str, str]]):
    return analyze_review(review, llm_client=_StaticClient(aspects))


class Step88PostprocessingTests(unittest.TestCase):
    def test_physical_packing_material_condition_overrides_build_label(self) -> None:
        review = "The corner guards were crushed, although the television was unharmed."
        result = _analyze(
            review,
            [_aspect("corner guards", "Build Quality", "negative", "The corner guards were crushed")],
        )
        self.assertEqual("Packaging", result["aspects"][0]["aspect"])

    def test_shipping_timing_is_delivery_not_packaging(self) -> None:
        review = "The parcel arrived three days late."
        result = _analyze(
            review,
            [_aspect("parcel arrival", "Packaging", "negative", review)],
        )
        self.assertEqual("Delivery", result["aspects"][0]["aspect"])

    def test_product_construction_is_build_not_design(self) -> None:
        review = "The monitor stand wobbles whenever the desk moves."
        result = _analyze(
            review,
            [_aspect("monitor stand", "Design", "negative", "The monitor stand wobbles")],
        )
        self.assertEqual("Build Quality", result["aspects"][0]["aspect"])

    def test_software_crash_is_system_performance(self) -> None:
        review = "The streaming app crashes whenever I open the guide."
        result = _analyze(
            review,
            [_aspect("streaming app", "Reliability", "positive", "The streaming app crashes")],
        )
        self.assertEqual("System Performance", result["aspects"][0]["aspect"])
        self.assertEqual("negative", result["aspects"][0]["sentiment"])

    def test_settings_findability_is_ease_of_use(self) -> None:
        review = "The settings panel is clear, and every option is easy to find."
        result = _analyze(
            review,
            [_aspect("settings panel", "System Performance", "positive", review)],
        )
        self.assertEqual("Ease of Use", result["aspects"][0]["aspect"])

    def test_settings_latency_is_system_performance(self) -> None:
        review = "The settings panel takes several seconds to open."
        result = _analyze(
            review,
            [_aspect("settings panel", "Ease of Use", "negative", review)],
        )
        self.assertEqual("System Performance", result["aspects"][0]["aspect"])

    def test_nearby_remote_anaphor_grounds_directional_pad(self) -> None:
        review = "The handset is light, but its directional pad feels stiff."
        evidence = "its directional pad feels stiff"
        result = _analyze(
            review,
            [_aspect("directional pad", "Build Quality", "negative", evidence)],
        )
        self.assertEqual("Remote Control", result["aspects"][0]["aspect"])
        self.assertEqual(evidence, result["aspects"][0]["evidence"])

    def test_ungrounded_generic_button_is_not_forced_to_remote(self) -> None:
        review = "The button on the television casing is loose."
        result = _analyze(
            review,
            [_aspect("button", "Build Quality", "negative", review)],
        )
        self.assertEqual("Build Quality", result["aspects"][0]["aspect"])

    def test_rendered_skin_tones_are_picture_quality(self) -> None:
        review = "Skin tones look natural in daylight scenes."
        result = _analyze(
            review,
            [_aspect("skin tones", "Design", "positive", "Skin tones look natural")],
        )
        self.assertEqual("Picture Quality", result["aspects"][0]["aspect"])

    def test_exterior_aesthetic_is_design_not_picture(self) -> None:
        review = "The television bezel looks refined beside our shelves."
        result = _analyze(
            review,
            [_aspect("television bezel", "Picture Quality", "positive", "The television bezel looks refined")],
        )
        self.assertEqual("Design", result["aspects"][0]["aspect"])

    def test_power_cord_reach_is_other_not_connectivity(self) -> None:
        review = "The power cord is too short to reach the wall outlet."
        result = _analyze(
            review,
            [_aspect("power cord", "Connectivity", "negative", review)],
        )
        self.assertEqual("Other", result["aspects"][0]["aspect"])

    def test_plain_receipt_amount_is_not_price_value_opinion(self) -> None:
        review = "My receipt lists a charge of $319."
        result = _analyze(
            review,
            [_aspect("charge", "Price / Value", "neutral", review)],
        )
        self.assertEqual([], result["aspects"])

    def test_coupon_fact_without_evaluation_is_not_price_value_opinion(self) -> None:
        review = "A coupon code appeared on the receipt."
        result = _analyze(
            review,
            [_aspect("coupon", "Price / Value", "positive", review)],
        )
        self.assertEqual([], result["aspects"])

    def test_explicit_overall_deal_evaluation_is_kept(self) -> None:
        review = "The overall deal feels reasonable for the features included."
        result = _analyze(
            review,
            [_aspect("overall deal", "Price / Value", "positive", review)],
        )
        self.assertEqual("Price / Value", result["aspects"][0]["aspect"])

    def test_neither_fair_nor_unfair_price_is_neutral(self) -> None:
        review = "The price is neither fair nor unfair for this size."
        result = _analyze(
            review,
            [_aspect("price", "Price / Value", "negative", review)],
        )
        self.assertEqual("neutral", result["aspects"][0]["sentiment"])

    def test_negated_flaw_is_positive_without_changing_evidence(self) -> None:
        review = "The frame does not flex when lifted."
        evidence = "The frame does not flex"
        result = _analyze(
            review,
            [_aspect("frame", "Build Quality", "negative", evidence)],
        )
        self.assertEqual("positive", result["aspects"][0]["sentiment"])
        self.assertEqual(evidence, result["aspects"][0]["evidence"])

    def test_no_wobble_is_positive(self) -> None:
        review = "There is no wobble in the base."
        result = _analyze(
            review,
            [_aspect("base", "Build Quality", "negative", review)],
        )
        self.assertEqual("positive", result["aspects"][0]["sentiment"])

    def test_lower_price_comparison_is_positive(self) -> None:
        review = "The price is lower than comparable models."
        result = _analyze(
            review,
            [_aspect("price", "Price / Value", "neutral", review)],
        )
        self.assertEqual("positive", result["aspects"][0]["sentiment"])

    def test_qualified_sale_price_comparison_is_positive(self) -> None:
        review = "Its introductory sale price is lower than competing models."
        result = _analyze(
            review,
            [_aspect("sale price", "Price / Value", "negative", review)],
        )
        self.assertEqual("positive", result["aspects"][0]["sentiment"])

    def test_clear_value_language_is_not_dropped_by_price_fact_guard(self) -> None:
        reviews = (
            ("This is a good buy for a guest room.", "positive"),
            ("The asking price is unreasonable for the limited features.", "negative"),
            ("Paying a little extra still feels worthwhile.", "positive"),
        )
        for review, sentiment in reviews:
            with self.subTest(review=review):
                result = _analyze(
                    review,
                    [_aspect("price value", "Price / Value", sentiment, review)],
                )
                self.assertEqual("Price / Value", result["aspects"][0]["aspect"])

    def test_neutral_deal_language_is_kept_and_normalized(self) -> None:
        review = "The deal feels neither compelling nor unfair."
        result = _analyze(
            review,
            [_aspect("deal", "Price / Value", "negative", review)],
        )
        self.assertEqual("neutral", result["aspects"][0]["sentiment"])

    def test_not_good_is_not_mistaken_for_negated_negative(self) -> None:
        review = "The speaker output is not good."
        result = _analyze(
            review,
            [_aspect("speaker output", "Sound Quality", "negative", review)],
        )
        self.assertEqual("negative", result["aspects"][0]["sentiment"])

    def test_attribute_free_other_is_dropped(self) -> None:
        review = "This product is great."
        result = _analyze(
            review,
            [_aspect("product", "Other", "positive", review)],
        )
        self.assertEqual([], result["aspects"])

    def test_vague_expectation_is_not_reliability(self) -> None:
        review = "This item met expectations."
        result = _analyze(
            review,
            [_aspect("item", "Reliability", "positive", review)],
        )
        self.assertEqual([], result["aspects"])

    def test_assumption_language_is_not_reliability(self) -> None:
        review = "It behaves as we assumed it would."
        result = _analyze(
            review,
            [_aspect("behavior", "Reliability", "positive", review)],
        )
        self.assertEqual([], result["aspects"])

    def test_vague_phrase_does_not_hide_concrete_long_term_reliability(self) -> None:
        review = "It has worked as expected for seven months without a fault."
        result = _analyze(
            review,
            [_aspect("operation", "Reliability", "positive", review)],
        )
        self.assertEqual("Reliability", result["aspects"][0]["aspect"])

    def test_specific_steady_operation_remains_reliability(self) -> None:
        review = "The unit powers on every morning and runs steadily all day."
        result = _analyze(
            review,
            [_aspect("unit operation", "Reliability", "positive", review)],
        )
        self.assertEqual("Reliability", result["aspects"][0]["aspect"])

    def test_cosmetic_arrival_condition_is_not_reliability(self) -> None:
        review = "The monitor frame arrived with a deep scratch."
        result = _analyze(
            review,
            [_aspect("monitor frame", "Reliability", "negative", review)],
        )
        self.assertEqual("Build Quality", result["aspects"][0]["aspect"])

    def test_cosmetic_arrive_form_is_not_reliability(self) -> None:
        review = "It happened to arrive without a scratch."
        result = _analyze(
            review,
            [_aspect("condition", "Reliability", "positive", review)],
        )
        self.assertEqual([], result["aspects"])

    def test_next_day_delivery_is_positive(self) -> None:
        review = "The order reached us the next day."
        result = _analyze(
            review,
            [_aspect("delivery", "Delivery", "neutral", review)],
        )
        self.assertEqual("positive", result["aspects"][0]["sentiment"])

    def test_instant_software_opening_is_system_performance(self) -> None:
        review = "The options panel opens the moment I tap its icon."
        result = _analyze(
            review,
            [_aspect("options panel", "Ease of Use", "positive", review)],
        )
        self.assertEqual("System Performance", result["aspects"][0]["aspect"])

    def test_remote_part_can_ground_following_remote_anaphor(self) -> None:
        review = "The D-pad clicks cleanly. The key next to it occasionally sticks."
        aspects = [
            _aspect("D-pad", "Design", "positive", "The D-pad clicks cleanly"),
            _aspect(
                "key",
                "Build Quality",
                "negative",
                "The key next to it occasionally sticks",
            ),
        ]
        result = _analyze(review, aspects)
        self.assertEqual(
            ["Remote Control", "Remote Control"],
            [item["aspect"] for item in result["aspects"]],
        )

    def test_modified_negated_flaw_is_positive(self) -> None:
        review = "The base is not remotely as flimsy as it first appears."
        result = _analyze(
            review,
            [_aspect("base", "Build Quality", "negative", review)],
        )
        self.assertEqual("positive", result["aspects"][0]["sentiment"])

    def test_generic_unremarkable_other_claim_is_dropped(self) -> None:
        review = "Nothing about it is remarkable."
        result = _analyze(
            review,
            [_aspect("nothing remarkable", "Other", "negative", review)],
        )
        self.assertEqual([], result["aspects"])

    def test_distinct_mentions_remain_separate_and_evidence_is_unchanged(self) -> None:
        review = "The menu hangs during launch, and the overlay crashes during playback."
        aspects = [
            _aspect("menu", "Ease of Use", "positive", "The menu hangs during launch"),
            _aspect("overlay", "Reliability", "positive", "the overlay crashes during playback"),
        ]
        result = _analyze(review, aspects)
        self.assertEqual(2, len(result["aspects"]))
        self.assertEqual(
            ["The menu hangs during launch", "the overlay crashes during playback"],
            [item["evidence"] for item in result["aspects"]],
        )
        self.assertEqual(
            ["System Performance", "System Performance"],
            [item["aspect"] for item in result["aspects"]],
        )
        self.assertEqual(["negative", "negative"], [item["sentiment"] for item in result["aspects"]])


if __name__ == "__main__":
    unittest.main()
