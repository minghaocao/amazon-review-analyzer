"""Offline guards for the narrow Step 8.5 prompt/taxonomy optimization."""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

from aspect_taxonomy import ASPECT_ALIASES, STANDARD_ASPECTS
from llm_client import _response_schema
from review_analyzer import ALLOWED_SENTIMENTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    PROJECT_ROOT / "review_analyzer.py",
    PROJECT_ROOT / "llm_client.py",
    PROJECT_ROOT / "aspect_taxonomy.py",
)


class Step85SafetyGuardsTests(unittest.TestCase):
    def test_model_schema_still_uses_the_existing_base_sentiments(self) -> None:
        schema = _response_schema(STANDARD_ASPECTS)
        sentiment_schema = schema["$defs"]["ReviewAspectOutput"]["properties"][
            "sentiment"
        ]
        self.assertEqual(set(ALLOWED_SENTIMENTS), set(sentiment_schema["enum"]))
        self.assertEqual({"positive", "negative", "neutral"}, set(ALLOWED_SENTIMENTS))

    def test_runtime_does_not_embed_complete_gold_reviews_or_case_ids(self) -> None:
        cases = json.loads(
            (PROJECT_ROOT / "evaluation" / "gold_cases.json").read_text(
                encoding="utf-8"
            )
        )
        runtime = "\n".join(path.read_text(encoding="utf-8") for path in RUNTIME_FILES)

        self.assertIsNone(re.search(r"\bE\d{3}\b", runtime))
        self.assertNotIn("case_id", runtime)
        for case in cases:
            with self.subTest(case_id=case["case_id"]):
                self.assertNotIn(case["review"], runtime)

    def test_no_review_text_condition_contains_a_case_specific_literal(self) -> None:
        for path in RUNTIME_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.If, ast.IfExp)):
                    continue
                test = node.test
                names = {
                    item.id
                    for item in ast.walk(test)
                    if isinstance(item, ast.Name)
                }
                if "review_text" not in names:
                    continue
                long_literals = [
                    item.value
                    for item in ast.walk(test)
                    if isinstance(item, ast.Constant)
                    and isinstance(item.value, str)
                    and len(item.value.split()) >= 4
                ]
                self.assertEqual([], long_literals, f"case-specific guard in {path.name}")

    def test_aliases_are_unique_after_normalization(self) -> None:
        owners: dict[str, str] = {}
        for aspect, aliases in ASPECT_ALIASES.items():
            for alias in aliases:
                key = re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip()
                self.assertNotIn(
                    key,
                    owners,
                    f"alias {alias!r} belongs to both {owners.get(key)!r} and {aspect!r}",
                )
                owners[key] = aspect


if __name__ == "__main__":
    unittest.main()
