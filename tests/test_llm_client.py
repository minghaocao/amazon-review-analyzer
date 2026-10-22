"""Tests for the isolated OpenAI client adapter without network calls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspect_taxonomy import STANDARD_ASPECTS  # noqa: E402
from llm_client import (  # noqa: E402
    LLMConfigurationError,
    LLMResponseError,
    OpenAIReviewClient,
)


class _FakeResponsesAPI:
    def __init__(self, output_text: str | None) -> None:
        self.output_text = output_text
        self.last_request: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.last_request = kwargs
        return SimpleNamespace(output_text=self.output_text)


class _FakeSDKClient:
    def __init__(self, output_text: str | None) -> None:
        self.responses = _FakeResponsesAPI(output_text)


class OpenAIReviewClientTests(unittest.TestCase):
    def test_missing_api_key_has_clear_error(self) -> None:
        with self.assertRaisesRegex(LLMConfigurationError, "API key is not configured"):
            OpenAIReviewClient(api_key="")

    def test_responses_api_uses_strict_json_schema(self) -> None:
        output = '{"summary":"Positive overall.","aspects":[]}'
        fake_sdk = _FakeSDKClient(output)
        client = OpenAIReviewClient(
            api_key="test-key",
            model="test-model",
            sdk_client=fake_sdk,
        )

        self.assertEqual(output, client.analyze_review("Great!", STANDARD_ASPECTS))
        request = fake_sdk.responses.last_request
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual("test-model", request["model"])
        self.assertFalse(request["store"])
        response_format = request["text"]["format"]  # type: ignore[index]
        self.assertEqual("json_schema", response_format["type"])
        self.assertTrue(response_format["strict"])
        self.assertNotIn("positive_points", response_format["schema"]["properties"])
        self.assertNotIn("negative_points", response_format["schema"]["properties"])

    def test_evidence_repair_uses_independent_strict_schema(self) -> None:
        output = '{"repairs":[{"aspect_index":0,"evidence":"looks beautiful"}]}'
        fake_sdk = _FakeSDKClient(output)
        client = OpenAIReviewClient(
            api_key="test-key",
            model="test-model",
            sdk_client=fake_sdk,
        )

        self.assertEqual(
            output,
            client.repair_evidence(
                "The TV looks beautiful.",
                [{
                    "aspect_index": 0,
                    "raw_aspect": "looks",
                    "aspect": "Design",
                    "sentiment": "positive",
                    "invalid_evidence": "looks very beautiful",
                }],
            ),
        )
        request = fake_sdk.responses.last_request
        self.assertIsNotNone(request)
        assert request is not None
        response_format = request["text"]["format"]  # type: ignore[index]
        self.assertEqual("evidence_repair", response_format["name"])
        self.assertEqual({"repairs"}, set(response_format["schema"]["properties"]))
        self.assertIn("Do not re-analyze", request["instructions"])

    def test_empty_model_output_has_clear_error(self) -> None:
        client = OpenAIReviewClient(
            api_key="test-key",
            sdk_client=_FakeSDKClient(None),
        )
        with self.assertRaisesRegex(LLMResponseError, "no structured output"):
            client.analyze_review("Great!", STANDARD_ASPECTS)

    def test_openai_output_respects_caller_taxonomy_subset(self) -> None:
        output = (
            '{"summary":"Positive picture.","aspects":['
            '{"raw_aspect":"picture","aspect":"Picture Quality",'
            '"sentiment":"positive","evidence":"picture is good"}]}'
        )
        client = OpenAIReviewClient(
            api_key="test-key",
            sdk_client=_FakeSDKClient(output),
        )

        with self.assertRaisesRegex(LLMResponseError, "outside the allowed taxonomy"):
            client.analyze_review("The picture is good.", ["Sound Quality"])


if __name__ == "__main__":
    unittest.main()
