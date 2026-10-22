"""Offline regression tests for Step 8.8 general category guidance."""

from __future__ import annotations

import re
import unittest

from aspect_taxonomy import (
    ASPECT_ALIASES,
    STANDARD_ASPECTS,
    normalize_aspect,
    taxonomy_definition_text,
)
from llm_client import _analysis_instructions, _response_schema


class Step88PromptTaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt = " ".join(_analysis_instructions(STANDARD_ASPECTS).split())
        self.definitions = " ".join(taxonomy_definition_text().split())

    def test_software_timing_is_distinct_from_comprehensibility(self) -> None:
        self.assertIn("app-launcher speed", self.prompt)
        self.assertIn("are System Performance", self.prompt)
        self.assertIn("understandable, intuitive, or confusing is Ease of Use", self.prompt)
        self.assertIn("a confusing menu is not software lag", self.prompt)

    def test_remote_owned_controls_remain_remote_control(self) -> None:
        self.assertIn("controls owned by the remote handset", self.prompt)
        self.assertIn("directional pad", self.prompt)
        self.assertEqual("Remote Control", normalize_aspect("directional pad"))
        self.assertEqual("Remote Control", normalize_aspect("handset"))

    def test_package_materials_are_separated_from_product_body_and_delivery(self) -> None:
        self.assertIn("including cartons, sleeves, and padding", self.prompt)
        self.assertIn("Product-body materials are Build Quality", self.prompt)
        self.assertIn("transit timing is Delivery", self.prompt)
        for target in (
            "carton",
            "padding",
            "sleeve",
            "protective tape",
            "corner guard",
            "cushioning",
        ):
            with self.subTest(target=target):
                self.assertEqual("Packaging", normalize_aspect(target))

    def test_rendered_image_is_separated_from_exterior_design(self) -> None:
        self.assertIn("The rendered picture", self.prompt)
        self.assertIn("Physical exterior appearance or styling is Design", self.prompt)
        self.assertIn("displayed or rendered image itself", self.definitions)

    def test_power_cable_is_not_data_connectivity(self) -> None:
        self.assertIn("power cord or power cable", self.prompt)
        self.assertIn("is not Connectivity", self.prompt)
        self.assertEqual("Other", normalize_aspect("power cable"))
        self.assertIn("data, network, audio, or video links", self.definitions)

    def test_reliability_requires_actual_operation(self) -> None:
        self.assertIn("Reliability requires an assertion about actual product operation", self.prompt)
        self.assertIn("Feature presence, appearance", self.prompt)
        self.assertIn("do not create Reliability", self.prompt)
        self.assertIn("feature presence, or a general quality adjective", self.definitions)

    def test_price_rules_exclude_transaction_facts_and_reward_lower_comparison(self) -> None:
        self.assertIn("transaction fact alone is not an opinion", self.prompt)
        self.assertIn("clearly lower comparative price is positive Price / Value", self.prompt)
        self.assertIn("pure payment or transaction fact", self.prompt)
        self.assertIn("comparative price", self.definitions)

    def test_sentiment_rules_require_evaluative_neutral_and_handle_negated_defects(self) -> None:
        self.assertIn("absence or negation of a defect is positive", self.prompt)
        self.assertIn("Neutral requires an explicit assessment", self.prompt)
        self.assertIn("specification without evaluation produces no Aspect", self.prompt)
        self.assertNotIn("capability or port stated without praise or complaint may be neutral", self.prompt)
        self.assertIn("quick, early, or on-time arrival is positive", self.prompt)

    def test_app_launcher_alias_is_narrowly_normalized(self) -> None:
        for target in ("app", "app launcher", "software overlay", "program guide"):
            with self.subTest(target=target):
                self.assertEqual("System Performance", normalize_aspect(target))

    def test_alias_keys_are_unique_after_normalization(self) -> None:
        seen: dict[str, str] = {}
        for aspect, aliases in ASPECT_ALIASES.items():
            for alias in aliases:
                key = re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip()
                self.assertNotIn(key, seen, msg=f"duplicate alias {alias!r}")
                seen[key] = aspect

    def test_public_schema_and_taxonomy_labels_are_unchanged(self) -> None:
        schema = _response_schema(STANDARD_ASPECTS)
        self.assertEqual({"summary", "aspects"}, set(schema["properties"]))
        self.assertEqual(
            list(STANDARD_ASPECTS),
            schema["$defs"]["ReviewAspectOutput"]["properties"]["aspect"]["enum"],
        )

    def test_prompt_contains_no_development_case_identifier(self) -> None:
        self.assertIsNone(re.search(r"\bG\d{3}\b", self.prompt))


if __name__ == "__main__":
    unittest.main()
