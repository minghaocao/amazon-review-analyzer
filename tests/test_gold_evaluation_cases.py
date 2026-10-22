"""Offline integrity checks for the fixed Step 8 gold dataset."""

from __future__ import annotations

import unittest

from aspect_taxonomy import STANDARD_ASPECTS
from evaluator import GOLD_CASES_PATH, load_gold_cases


class GoldEvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_gold_cases(GOLD_CASES_PATH)

    def test_has_exactly_thirty_unique_human_labelled_cases(self) -> None:
        self.assertEqual(30, len(self.cases))
        self.assertEqual(30, len({case["case_id"] for case in self.cases}))

    def test_every_gold_evidence_is_a_contiguous_source_substring(self) -> None:
        for case in self.cases:
            for mention in case["expected_aspects"]:
                evidence = mention["evidence"]
                values = evidence if isinstance(evidence, list) else [evidence]
                for value in values:
                    with self.subTest(case_id=case["case_id"], evidence=value):
                        self.assertIn(value, case["review"])

    def test_required_aspects_and_all_four_sentiments_are_covered(self) -> None:
        mentions = [
            mention
            for case in self.cases
            for mention in case["expected_aspects"]
        ]
        aspects = {mention["aspect"] for mention in mentions}
        sentiments = {mention["sentiment"] for mention in mentions}

        self.assertTrue(set(STANDARD_ASPECTS) - {"Other"} <= aspects)
        self.assertEqual({"positive", "negative", "neutral", "mixed"}, sentiments)

    def test_minimum_diversity_counts_are_preserved(self) -> None:
        multi_aspect = 0
        mixed_polarity = 0
        neutral_mentions = 0
        no_aspect = 0
        shopping_experience = 0
        experience_aspects = {"Delivery", "Packaging", "Customer Service"}

        for case in self.cases:
            mentions = case["expected_aspects"]
            distinct_aspects = {mention["aspect"] for mention in mentions}
            sentiments = {mention["sentiment"] for mention in mentions}
            multi_aspect += int(len(distinct_aspects) >= 2)
            mixed_polarity += int(
                "mixed" in sentiments
                or {"positive", "negative"} <= sentiments
            )
            neutral_mentions += sum(
                mention["sentiment"] == "neutral" for mention in mentions
            )
            no_aspect += int(not mentions)
            shopping_experience += int(bool(distinct_aspects & experience_aspects))

        self.assertGreaterEqual(multi_aspect, 5)
        self.assertGreaterEqual(mixed_polarity, 4)
        self.assertGreaterEqual(neutral_mentions, 3)
        self.assertGreaterEqual(no_aspect, 2)
        self.assertGreaterEqual(shopping_experience, 2)


if __name__ == "__main__":
    unittest.main()
