"""Integrity checks for the first ten manually labelled review cases."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspect_taxonomy import STANDARD_ASPECTS  # noqa: E402
from review_analyzer import ALLOWED_SENTIMENTS  # noqa: E402


CASES_PATH = Path(__file__).with_name("manual_review_cases.json")


class ManualReviewCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    def test_exactly_ten_unique_cases_exist(self) -> None:
        self.assertEqual(10, len(self.cases))
        case_ids = [case["case_id"] for case in self.cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_expected_aspects_use_allowed_values(self) -> None:
        for case in self.cases:
            for aspect in case["expected_aspects"]:
                with self.subTest(case=case["case_id"], aspect=aspect["aspect"]):
                    self.assertIn(aspect["aspect"], STANDARD_ASPECTS)
                    self.assertIn(aspect["sentiment"], ALLOWED_SENTIMENTS)

    def test_expected_evidence_is_verbatim(self) -> None:
        for case in self.cases:
            for aspect in case["expected_aspects"]:
                with self.subTest(case=case["case_id"], evidence=aspect["evidence"]):
                    self.assertIn(aspect["evidence"], case["review_text"])


if __name__ == "__main__":
    unittest.main()
