"""Tests for the small Step 4 aspect taxonomy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspect_taxonomy import (  # noqa: E402
    BUSINESS_CATEGORY_BY_ASPECT,
    EXPERIENCE_ASPECTS,
    PRODUCT_ASPECTS,
    STANDARD_ASPECTS,
    business_category_for_aspect,
    normalize_aspect,
    taxonomy_definition_text,
)


class AspectTaxonomyTests(unittest.TestCase):
    def test_required_standard_aspects_are_present(self) -> None:
        expected = {
            "Picture Quality",
            "Sound Quality",
            "Price / Value",
            "Remote Control",
            "System Performance",
            "Ease of Use",
            "Design",
            "Build Quality",
            "Reliability",
            "Connectivity",
            "Setup / Installation",
            "Customer Service",
            "Delivery",
            "Packaging",
            "Other",
        }
        self.assertEqual(expected, set(STANDARD_ASPECTS))

    def test_common_aliases_are_normalized(self) -> None:
        examples = {
            "screen": "Picture Quality",
            "speakers": "Sound Quality",
            "remote": "Remote Control",
            "slow menu response": "System Performance",
            "Wi-Fi": "Connectivity",
            "arrived late": "Delivery",
        }
        for raw_aspect, expected in examples.items():
            with self.subTest(raw_aspect=raw_aspect):
                self.assertEqual(expected, normalize_aspect(raw_aspect))

    def test_unknown_aspect_falls_back_to_other(self) -> None:
        self.assertEqual("Other", normalize_aspect("coffee temperature", "Battery Life"))

    def test_raw_aspect_takes_priority_over_incorrect_suggestion(self) -> None:
        self.assertEqual("Picture Quality", normalize_aspect("screen", "Sound Quality"))

    def test_product_operation_phrases_normalize_to_reliability(self) -> None:
        self.assertEqual(
            "Reliability",
            normalize_aspect("the TV itself works perfectly", "System Performance"),
        )
        self.assertEqual("Reliability", normalize_aspect("stopped working"))

    def test_generic_works_does_not_override_explicit_software_performance(self) -> None:
        self.assertEqual(
            "System Performance",
            normalize_aspect("menu works slowly", "System Performance"),
        )

    def test_opinion_target_aliases_resolve_common_category_boundaries(self) -> None:
        examples = {
            "volume button": "Remote Control",
            "home screen lags": "System Performance",
            "dark scenes look washed out": "Picture Quality",
            "responds instantly": "System Performance",
            "worked perfectly after delivery": "Reliability",
            "not difficult": "Ease of Use",
            "at this price": "Price / Value",
        }
        for raw_aspect, expected in examples.items():
            with self.subTest(raw_aspect=raw_aspect):
                self.assertEqual(expected, normalize_aspect(raw_aspect))

    def test_suggested_category_disambiguates_a_multi_category_raw_phrase(self) -> None:
        self.assertEqual(
            "Remote Control",
            normalize_aspect(
                "the remote volume button is difficult to press",
                "Remote Control",
            ),
        )
        self.assertEqual(
            "System Performance",
            normalize_aspect(
                "the home screen lags when apps open",
                "System Performance",
            ),
        )
        self.assertEqual(
            "Reliability",
            normalize_aspect(
                "the unit worked perfectly after it arrived on time",
                "Reliability",
            ),
        )

    def test_category_guidance_describes_target_boundaries(self) -> None:
        guidance = taxonomy_definition_text()
        self.assertIn("volume button on the remote", guidance)
        self.assertIn("not the product's exterior", guidance)
        self.assertIn("not menu or app speed", guidance)
        self.assertIn("product operation after arrival", guidance)

    def test_every_standard_aspect_has_a_business_category(self) -> None:
        self.assertEqual(set(STANDARD_ASPECTS), set(BUSINESS_CATEGORY_BY_ASPECT))

    def test_product_business_category_contains_expected_aspects(self) -> None:
        self.assertEqual(
            {
                "Picture Quality",
                "Sound Quality",
                "Price / Value",
                "Remote Control",
                "System Performance",
                "Ease of Use",
                "Design",
                "Build Quality",
                "Reliability",
                "Connectivity",
                "Setup / Installation",
            },
            set(PRODUCT_ASPECTS),
        )
        for aspect in PRODUCT_ASPECTS:
            self.assertEqual("PRODUCT", business_category_for_aspect(aspect))

    def test_experience_and_other_business_categories(self) -> None:
        self.assertEqual(
            {"Delivery", "Packaging", "Customer Service"},
            set(EXPERIENCE_ASPECTS),
        )
        for aspect in EXPERIENCE_ASPECTS:
            self.assertEqual("EXPERIENCE", business_category_for_aspect(aspect))
        self.assertEqual("OTHER", business_category_for_aspect("Other"))
        self.assertEqual("OTHER", business_category_for_aspect("Unknown Aspect"))


if __name__ == "__main__":
    unittest.main()
