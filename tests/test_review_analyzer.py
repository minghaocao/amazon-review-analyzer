"""Tests for the single-review analysis and validation pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import LLMClientError  # noqa: E402
from review_analyzer import (  # noqa: E402
    ReviewAnalysisError,
    analyze_review,
    validate_analysis_result,
)


class _FakeLLMClient:
    def __init__(
        self,
        response: str | dict[str, Any],
        repair_response: str | dict[str, Any] | None = None,
        repair_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.repair_response = repair_response or {"repairs": []}
        self.repair_error = repair_error
        self.received_review: str | None = None
        self.received_aspects: tuple[str, ...] = ()
        self.repair_calls: list[tuple[str, list[dict[str, Any]]]] = []

    def analyze_review(self, review_text: str, allowed_aspects: tuple[str, ...]) -> str | dict[str, Any]:
        self.received_review = review_text
        self.received_aspects = tuple(allowed_aspects)
        return deepcopy(self.response)

    def repair_evidence(
        self,
        review_text: str,
        invalid_aspects: list[dict[str, Any]],
    ) -> str | dict[str, Any]:
        self.repair_calls.append((review_text, deepcopy(invalid_aspects)))
        if self.repair_error:
            raise self.repair_error
        return deepcopy(self.repair_response)


def _response(
    summary: str,
    aspects: list[dict[str, str]],
    positive_points: list[str] | None = None,
    negative_points: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary": summary,
        "aspects": aspects,
    }
    if positive_points is not None:
        payload["positive_points"] = positive_points
    if negative_points is not None:
        payload["negative_points"] = negative_points
    return payload


class ReviewAnalyzerTests(unittest.TestCase):
    def _analyze(self, review: str, response: dict[str, Any] | str) -> dict[str, Any]:
        return analyze_review(review, llm_client=_FakeLLMClient(response))

    def test_single_positive_aspect(self) -> None:
        review = "The picture quality is excellent."
        result = self._analyze(
            review,
            _response(
                "Excellent picture quality.",
                [{
                    "raw_aspect": "picture quality",
                    "aspect": "Picture Quality",
                    "sentiment": "positive",
                    "evidence": "The picture quality is excellent",
                }],
                positive_points=["Excellent picture quality"],
            ),
        )
        self.assertEqual("positive", result["aspects"][0]["sentiment"])
        self.assertEqual(
            ["The picture quality is excellent"], result["positive_points"]
        )

    def test_single_negative_aspect(self) -> None:
        review = "The remote feels cheap."
        result = self._analyze(
            review,
            _response(
                "The remote feels cheap.",
                [{
                    "raw_aspect": "remote",
                    "aspect": "Remote Control",
                    "sentiment": "negative",
                    "evidence": "The remote feels cheap",
                }],
                negative_points=["Remote feels cheap"],
            ),
        )
        self.assertEqual("negative", result["aspects"][0]["sentiment"])
        self.assertEqual(["The remote feels cheap"], result["negative_points"])

    def test_multiple_aspects_are_preserved(self) -> None:
        review = "The picture is clear, the sound is fine, and the remote is responsive."
        aspects = [
            {"raw_aspect": "picture", "aspect": "Picture Quality", "sentiment": "positive", "evidence": "The picture is clear"},
            {"raw_aspect": "sound", "aspect": "Sound Quality", "sentiment": "neutral", "evidence": "the sound is fine"},
            {"raw_aspect": "remote", "aspect": "Remote Control", "sentiment": "positive", "evidence": "the remote is responsive"},
        ]
        result = self._analyze(review, _response("Several product attributes are discussed.", aspects))
        self.assertEqual(3, len(result["aspects"]))

    def test_positive_and_negative_aspects_can_coexist(self) -> None:
        review = "The screen looks great but the speakers sound terrible."
        result = self._analyze(
            review,
            _response(
                "Great screen, but terrible sound.",
                [
                    {"raw_aspect": "screen", "aspect": "Picture Quality", "sentiment": "positive", "evidence": "The screen looks great"},
                    {"raw_aspect": "speakers", "aspect": "Sound Quality", "sentiment": "negative", "evidence": "the speakers sound terrible"},
                ],
            ),
        )
        self.assertEqual(["positive", "negative"], [item["sentiment"] for item in result["aspects"]])

    def test_delivery_and_product_aspects_can_coexist(self) -> None:
        review = "Delivery was slow, but the TV itself works perfectly."
        result = self._analyze(
            review,
            _response(
                "Slow delivery, but the TV works perfectly.",
                [
                    {"raw_aspect": "Delivery", "aspect": "Delivery", "sentiment": "negative", "evidence": "Delivery was slow"},
                    {"raw_aspect": "TV itself", "aspect": "Reliability", "sentiment": "positive", "evidence": "the TV itself works perfectly"},
                ],
            ),
        )
        self.assertEqual(["Delivery", "Reliability"], [item["aspect"] for item in result["aspects"]])

    def test_neutral_sentiment_is_allowed(self) -> None:
        review = "The remote is fine."
        result = self._analyze(
            review,
            _response(
                "The remote is adequate.",
                [{"raw_aspect": "remote", "aspect": "Remote Control", "sentiment": "neutral", "evidence": "The remote is fine"}],
            ),
        )
        self.assertEqual("neutral", result["aspects"][0]["sentiment"])

    def test_negative_word_is_interpreted_in_context(self) -> None:
        review = "The sound is not good."
        result = self._analyze(
            review,
            _response(
                "The sound quality is poor.",
                [{"raw_aspect": "sound", "aspect": "Sound Quality", "sentiment": "negative", "evidence": "The sound is not good"}],
            ),
        )
        self.assertEqual("negative", result["aspects"][0]["sentiment"])

    def test_isnt_slow_anymore_is_not_forced_negative(self) -> None:
        review = "The menu isn't slow anymore."
        result = self._analyze(
            review,
            _response(
                "The menu performance has improved.",
                [{"raw_aspect": "menu", "aspect": "System Performance", "sentiment": "positive", "evidence": "The menu isn't slow anymore"}],
            ),
        )
        self.assertEqual("positive", result["aspects"][0]["sentiment"])

    def test_good_but_not_amazing_remains_positive(self) -> None:
        review = "The picture is good, but not amazing."
        result = self._analyze(
            review,
            _response(
                "The picture is good, though not exceptional.",
                [{"raw_aspect": "picture", "aspect": "Picture Quality", "sentiment": "positive", "evidence": "The picture is good, but not amazing"}],
            ),
        )
        self.assertEqual("positive", result["aspects"][0]["sentiment"])

    def test_short_generic_review_does_not_invent_aspect(self) -> None:
        result = self._analyze(
            "Great!",
            _response("The customer expresses a positive overall opinion.", []),
        )
        self.assertEqual([], result["aspects"])

    def test_invalid_sentiment_is_rejected(self) -> None:
        response = _response(
            "The picture is excellent.",
            [{"raw_aspect": "picture", "aspect": "Picture Quality", "sentiment": "very_positive", "evidence": "The picture is excellent"}],
        )
        with self.assertRaisesRegex(ReviewAnalysisError, "sentiment must be one of"):
            self._analyze("The picture is excellent.", response)

    def test_taxonomy_outside_value_falls_back_to_other(self) -> None:
        review = "The battery lasts two hours."
        result = self._analyze(
            review,
            _response(
                "Battery life is mentioned.",
                [{"raw_aspect": "battery", "aspect": "Battery Life", "sentiment": "negative", "evidence": "The battery lasts two hours"}],
            ),
        )
        self.assertEqual("Other", result["aspects"][0]["aspect"])

    def test_known_outside_label_is_normalized(self) -> None:
        review = "The screen looks great."
        result = self._analyze(
            review,
            _response(
                "The screen looks great.",
                [{"raw_aspect": "screen", "aspect": "Display", "sentiment": "positive", "evidence": "The screen looks great"}],
            ),
        )
        self.assertEqual("Picture Quality", result["aspects"][0]["aspect"])

    def test_fabricated_evidence_is_rejected(self) -> None:
        response = _response(
            "Battery life is poor.",
            [{"raw_aspect": "battery", "aspect": "Other", "sentiment": "negative", "evidence": "Battery life is extremely poor"}],
        )
        with self.assertRaisesRegex(ReviewAnalysisError, "evidence was not found"):
            validate_analysis_result(response, "Battery lasts only 2 hours.")

    def test_malformed_json_has_controlled_error(self) -> None:
        with self.assertRaisesRegex(ReviewAnalysisError, "not valid JSON"):
            self._analyze("Great!", "{not-json")

    def test_original_review_and_fake_response_are_not_modified(self) -> None:
        review = "  The picture is GREAT!!!  "
        response = _response("The picture is great.", [])
        original_response = deepcopy(response)
        fake_client = _FakeLLMClient(response)

        analyze_review(review, llm_client=fake_client)

        self.assertEqual("  The picture is GREAT!!!  ", review)
        self.assertEqual(review, fake_client.received_review)
        self.assertEqual(original_response, response)

    def test_evidence_must_preserve_source_casing(self) -> None:
        review = "The Picture is Excellent."
        response = _response(
            "Excellent picture.",
            [{"raw_aspect": "Picture", "aspect": "Picture Quality", "sentiment": "positive", "evidence": "the picture is excellent"}],
        )
        with self.assertRaisesRegex(ReviewAnalysisError, "evidence was not found"):
            validate_analysis_result(response, review)

    def test_same_aspect_can_have_two_sentiment_mentions(self) -> None:
        review = "The sound is rich at low volume, but it distorts when loud."
        result = self._analyze(
            review,
            _response(
                "Sound is rich at low volume but distorts when loud.",
                [
                    {"raw_aspect": "sound", "aspect": "Sound Quality", "sentiment": "positive", "evidence": "The sound is rich at low volume"},
                    {"raw_aspect": "sound", "aspect": "Sound Quality", "sentiment": "negative", "evidence": "it distorts when loud"},
                ],
            ),
        )
        self.assertEqual(["positive", "negative"], [item["sentiment"] for item in result["aspects"]])

    def test_partial_evidence_failure_is_repaired_once_and_keeps_valid_aspects(self) -> None:
        review = "The TV looks beautiful and feels very solid."
        client = _FakeLLMClient(
            _response(
                "The TV looks beautiful and feels solid.",
                [
                    {
                        "raw_aspect": "looks",
                        "aspect": "Design",
                        "sentiment": "positive",
                        "evidence": "looks beautiful",
                    },
                    {
                        "raw_aspect": "solid",
                        "aspect": "Build Quality",
                        "sentiment": "positive",
                        "evidence": "feels extremely solid",
                    },
                ],
            ),
            repair_response={
                "repairs": [{"aspect_index": 1, "evidence": "feels very solid"}]
            },
        )

        result = analyze_review(review, llm_client=client)

        self.assertEqual(["Design", "Build Quality"], [a["aspect"] for a in result["aspects"]])
        self.assertEqual(
            ["looks beautiful", "feels very solid"], result["positive_points"]
        )
        self.assertEqual(1, len(client.repair_calls))
        self.assertEqual([1], [item["aspect_index"] for item in client.repair_calls[0][1]])

    def test_failed_evidence_repair_drops_only_invalid_aspect_with_warning(self) -> None:
        review = "The TV looks beautiful and feels very solid."
        client = _FakeLLMClient(
            _response(
                "The TV looks beautiful and feels solid.",
                [
                    {
                        "raw_aspect": "looks",
                        "aspect": "Design",
                        "sentiment": "positive",
                        "evidence": "looks beautiful",
                    },
                    {
                        "raw_aspect": "solid",
                        "aspect": "Build Quality",
                        "sentiment": "positive",
                        "evidence": "feels extremely solid",
                    },
                ],
            ),
            repair_response={
                "repairs": [{"aspect_index": 1, "evidence": "feels sturdy"}]
            },
        )

        with self.assertLogs("review_analyzer", level="WARNING") as captured:
            result = analyze_review(review, llm_client=client)

        self.assertEqual(["Design"], [a["aspect"] for a in result["aspects"]])
        self.assertEqual(["looks beautiful"], result["positive_points"])
        self.assertIn("Dropping aspect 1", "\n".join(captured.output))
        self.assertEqual(1, len(client.repair_calls))

    def test_repair_provider_error_preserves_valid_aspects(self) -> None:
        review = "The screen is great but the remote feels cheap."
        client = _FakeLLMClient(
            _response(
                "Great screen, cheap remote.",
                [
                    {
                        "raw_aspect": "screen",
                        "aspect": "Picture Quality",
                        "sentiment": "positive",
                        "evidence": "The screen is great",
                    },
                    {
                        "raw_aspect": "remote",
                        "aspect": "Remote Control",
                        "sentiment": "negative",
                        "evidence": "remote is cheap",
                    },
                ],
            ),
            repair_error=LLMClientError("repair unavailable"),
        )

        with self.assertLogs("review_analyzer", level="WARNING"):
            result = analyze_review(review, llm_client=client)

        self.assertEqual(["Picture Quality"], [a["aspect"] for a in result["aspects"]])
        self.assertEqual(["The screen is great"], result["positive_points"])
        self.assertEqual([], result["negative_points"])

    def test_all_invalid_evidence_can_be_dropped_without_losing_summary(self) -> None:
        review = "The remote feels cheap."
        client = _FakeLLMClient(
            _response(
                "The remote feels cheap.",
                [{
                    "raw_aspect": "remote",
                    "aspect": "Remote Control",
                    "sentiment": "negative",
                    "evidence": "The remote is cheaply made",
                }],
            )
        )

        with self.assertLogs("review_analyzer", level="WARNING"):
            result = analyze_review(review, llm_client=client)

        self.assertEqual("The remote feels cheap.", result["summary"])
        self.assertEqual([], result["aspects"])
        self.assertEqual([], result["positive_points"])
        self.assertEqual([], result["negative_points"])
        self.assertEqual(1, len(client.repair_calls))

    def test_points_are_derived_from_validated_evidence_not_model_lists(self) -> None:
        review = "The picture is clear, the remote is bad, and the sound is average."
        response = _response(
            "The review covers picture, remote, and sound.",
            [
                {"raw_aspect": "picture", "aspect": "Picture Quality", "sentiment": "positive", "evidence": "The picture is clear"},
                {"raw_aspect": "remote", "aspect": "Remote Control", "sentiment": "negative", "evidence": "the remote is bad"},
                {"raw_aspect": "sound", "aspect": "Sound Quality", "sentiment": "neutral", "evidence": "the sound is average"},
            ],
            positive_points=["fabricated positive point"],
            negative_points=["fabricated negative point"],
        )

        result = self._analyze(review, response)

        self.assertEqual(["The picture is clear"], result["positive_points"])
        self.assertEqual(["the remote is bad"], result["negative_points"])

    def test_valid_evidence_does_not_trigger_repair(self) -> None:
        client = _FakeLLMClient(_response("Positive.", []))

        analyze_review("Great!", llm_client=client)

        self.assertEqual([], client.repair_calls)

    def test_json_string_response_is_supported(self) -> None:
        payload = _response("Positive overall.", [], positive_points=["Positive overall"])
        result = self._analyze("Great!", json.dumps(payload))
        self.assertEqual("Positive overall.", result["summary"])


if __name__ == "__main__":
    unittest.main()
