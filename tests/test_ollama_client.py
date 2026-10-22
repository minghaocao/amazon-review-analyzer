"""Offline tests for the Ollama provider adapter."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspect_taxonomy import STANDARD_ASPECTS  # noqa: E402
from llm_client import (  # noqa: E402
    LLMClientError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMResponseError,
    OllamaReviewClient,
)


class _FakeResponseError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.error = message
        self.status_code = status_code


class _FakeOllamaSDK:
    def __init__(self, output_text: str | None = None) -> None:
        self.output_text = output_text
        self.list_error: Exception | None = None
        self.show_error: Exception | None = None
        self.chat_error: Exception | None = None
        self.list_calls = 0
        self.show_calls: list[str] = []
        self.chat_calls: list[dict[str, Any]] = []

    def list(self) -> SimpleNamespace:
        self.list_calls += 1
        if self.list_error:
            raise self.list_error
        return SimpleNamespace(models=[SimpleNamespace(model="qwen3:8b")])

    def show(self, model: str) -> SimpleNamespace:
        self.show_calls.append(model)
        if self.show_error:
            raise self.show_error
        return SimpleNamespace(model=model)

    def chat(self, **kwargs: Any) -> SimpleNamespace:
        self.chat_calls.append(kwargs)
        if self.chat_error:
            raise self.chat_error
        return SimpleNamespace(
            message=SimpleNamespace(content=self.output_text),
        )


def _valid_output() -> str:
    return json.dumps(
        {
            "summary": "Excellent picture quality.",
            "aspects": [
                {
                    "raw_aspect": "picture quality",
                    "aspect": "Picture Quality",
                    "sentiment": "positive",
                    "evidence": "The picture quality is excellent",
                }
            ],
        }
    )


class OllamaReviewClientTests(unittest.TestCase):
    def test_connection_check_confirms_service_and_model_without_chat(self) -> None:
        sdk = _FakeOllamaSDK(_valid_output())
        client = OllamaReviewClient(
            model="qwen3:8b",
            host="http://test-host:11434",
            sdk_client=sdk,
        )

        status = client.check_connection()

        self.assertEqual(
            {"host": "http://test-host:11434", "model": "qwen3:8b"},
            status,
        )
        self.assertEqual(1, sdk.list_calls)
        self.assertEqual(["qwen3:8b"], sdk.show_calls)
        self.assertEqual([], sdk.chat_calls)

    def test_connection_failure_has_clear_message(self) -> None:
        sdk = _FakeOllamaSDK()
        sdk.list_error = ConnectionError("connection refused")
        client = OllamaReviewClient(sdk_client=sdk)

        with self.assertRaisesRegex(LLMConnectionError, "Ollama is not running"):
            client.check_connection()

    def test_unavailable_local_endpoint_has_clear_message(self) -> None:
        sdk = _FakeOllamaSDK()
        sdk.list_error = _FakeResponseError("service unavailable", 503)
        client = OllamaReviewClient(sdk_client=sdk)

        with self.assertRaisesRegex(LLMConnectionError, "Ollama is not running"):
            client.check_connection()

    def test_missing_model_includes_pull_command(self) -> None:
        sdk = _FakeOllamaSDK()
        sdk.show_error = _FakeResponseError("model not found", 404)
        client = OllamaReviewClient(model="qwen3:8b", sdk_client=sdk)

        with self.assertRaisesRegex(
            LLMModelNotFoundError,
            r"ollama pull qwen3:8b",
        ):
            client.check_connection()

    def test_chat_uses_schema_zero_temperature_and_no_thinking(self) -> None:
        sdk = _FakeOllamaSDK(_valid_output())
        client = OllamaReviewClient(
            model="test-model",
            keep_alive="7m",
            num_ctx=2048,
            sdk_client=sdk,
        )

        result = client.analyze_review(
            "The picture quality is excellent.",
            STANDARD_ASPECTS,
        )

        self.assertEqual("positive", result["aspects"][0]["sentiment"])
        self.assertEqual(1, len(sdk.chat_calls))
        request = sdk.chat_calls[0]
        self.assertEqual("test-model", request["model"])
        self.assertEqual({"temperature": 0, "num_ctx": 2048}, request["options"])
        self.assertEqual("7m", request["keep_alive"])
        self.assertFalse(request["stream"])
        self.assertFalse(request["think"])
        self.assertNotIn("positive_points", request["format"]["properties"])
        self.assertNotIn("negative_points", request["format"]["properties"])
        aspect_schema = request["format"]["$defs"]["ReviewAspectOutput"]
        self.assertEqual(
            list(STANDARD_ASPECTS),
            aspect_schema["properties"]["aspect"]["enum"],
        )
        self.assertEqual("system", request["messages"][0]["role"])
        prompt = request["messages"][0]["content"]
        prompt_flat = " ".join(prompt.split())
        self.assertIn("evidence", prompt)
        self.assertIn("opinion target before choosing a taxonomy label", prompt_flat)
        self.assertIn("not merely on the presence of a related keyword", prompt_flat)
        self.assertIn("volume or navigation button is Remote Control", prompt_flat)
        self.assertIn("home-screen lag or freezing", prompt_flat)
        self.assertIn("comparison device is context", prompt_flat)
        self.assertIn("semantically mixed", prompt_flat)
        self.assertIn("same standard Aspect", prompt_flat)
        self.assertIn("do not quote only a bare adjective", prompt_flat)
        self.assertIn("good, but not amazing", prompt)
        self.assertIn("Do not emit such a qualifier", prompt)
        self.assertIn("exactly one Picture Quality mention", prompt)
        self.assertIn(
            "The picture quality is excellent.",
            request["messages"][1]["content"],
        )

    def test_system_prompt_does_not_embed_complete_gold_reviews(self) -> None:
        sdk = _FakeOllamaSDK(_valid_output())
        client = OllamaReviewClient(sdk_client=sdk)
        client.analyze_review(
            "The picture quality is excellent.",
            STANDARD_ASPECTS,
        )
        prompt = sdk.chat_calls[0]["messages"][0]["content"]
        cases = json.loads(
            (PROJECT_ROOT / "evaluation" / "gold_cases.json").read_text(
                encoding="utf-8"
            )
        )

        for case in cases:
            with self.subTest(case_id=case["case_id"]):
                self.assertNotIn(case["review"], prompt)

    def test_model_is_checked_only_once_per_client(self) -> None:
        sdk = _FakeOllamaSDK(_valid_output())
        client = OllamaReviewClient(
            model="test-model",
            keep_alive="7m",
            num_ctx=2048,
            sdk_client=sdk,
        )

        client.analyze_review("The picture quality is excellent.", STANDARD_ASPECTS)
        client.analyze_review("The picture quality is excellent.", STANDARD_ASPECTS)

        self.assertEqual(1, sdk.list_calls)
        self.assertEqual(1, len(sdk.show_calls))
        self.assertEqual(2, len(sdk.chat_calls))

    def test_malformed_json_is_rejected_by_pydantic(self) -> None:
        client = OllamaReviewClient(sdk_client=_FakeOllamaSDK("not-json"))

        with self.assertRaisesRegex(LLMResponseError, "does not match"):
            client.analyze_review("Great!", STANDARD_ASPECTS)

    def test_invalid_sentiment_is_rejected_by_pydantic(self) -> None:
        payload = json.loads(_valid_output())
        payload["aspects"][0]["sentiment"] = "mixed"
        client = OllamaReviewClient(sdk_client=_FakeOllamaSDK(json.dumps(payload)))

        with self.assertRaisesRegex(LLMResponseError, "does not match"):
            client.analyze_review(
                "The picture quality is excellent.",
                STANDARD_ASPECTS,
            )

    def test_missing_required_field_is_rejected_by_pydantic(self) -> None:
        payload = json.loads(_valid_output())
        del payload["summary"]
        client = OllamaReviewClient(sdk_client=_FakeOllamaSDK(json.dumps(payload)))

        with self.assertRaisesRegex(LLMResponseError, "does not match"):
            client.analyze_review(
                "The picture quality is excellent.",
                STANDARD_ASPECTS,
            )

    def test_evidence_repair_uses_small_schema_and_restricted_prompt(self) -> None:
        sdk = _FakeOllamaSDK(
            json.dumps(
                {"repairs": [{"aspect_index": 1, "evidence": "feels very solid"}]}
            )
        )
        client = OllamaReviewClient(
            model="test-model",
            keep_alive="7m",
            num_ctx=2048,
            sdk_client=sdk,
        )

        result = client.repair_evidence(
            "The TV looks beautiful and feels very solid.",
            [
                {
                    "aspect_index": 1,
                    "raw_aspect": "solid",
                    "aspect": "Build Quality",
                    "sentiment": "positive",
                    "invalid_evidence": "feels extremely solid",
                }
            ],
        )

        self.assertEqual("feels very solid", result["repairs"][0]["evidence"])
        self.assertEqual(1, len(sdk.chat_calls))
        request = sdk.chat_calls[0]
        self.assertEqual({"temperature": 0, "num_ctx": 2048}, request["options"])
        self.assertEqual("7m", request["keep_alive"])
        self.assertEqual({"repairs"}, set(request["format"]["properties"]))
        self.assertIn("Do not re-analyze", request["messages"][0]["content"])
        self.assertNotIn("Allowed aspect categories", request["messages"][0]["content"])
        self.assertIn("feels extremely solid", request["messages"][1]["content"])

    def test_unknown_taxonomy_is_rejected_by_pydantic(self) -> None:
        payload = json.loads(_valid_output())
        payload["aspects"][0]["aspect"] = "Battery Life"
        client = OllamaReviewClient(sdk_client=_FakeOllamaSDK(json.dumps(payload)))

        with self.assertRaisesRegex(LLMResponseError, "does not match"):
            client.analyze_review(
                "The picture quality is excellent.",
                STANDARD_ASPECTS,
            )

    def test_aspect_outside_caller_taxonomy_subset_is_rejected(self) -> None:
        client = OllamaReviewClient(sdk_client=_FakeOllamaSDK(_valid_output()))

        with self.assertRaisesRegex(LLMResponseError, "outside the allowed taxonomy"):
            client.analyze_review(
                "The picture quality is excellent.",
                ["Sound Quality"],
            )

    def test_empty_output_has_clear_error(self) -> None:
        client = OllamaReviewClient(sdk_client=_FakeOllamaSDK(None))

        with self.assertRaisesRegex(LLMResponseError, "no structured output"):
            client.analyze_review("Great!", STANDARD_ASPECTS)

    def test_request_failure_is_not_retried(self) -> None:
        sdk = _FakeOllamaSDK(_valid_output())
        sdk.chat_error = RuntimeError("local inference failed")
        client = OllamaReviewClient(sdk_client=sdk)

        with self.assertRaisesRegex(LLMClientError, "Ollama request failed"):
            client.analyze_review("Great!", STANDARD_ASPECTS)
        self.assertEqual(1, len(sdk.chat_calls))

    def test_connection_reset_is_classified_as_transient_connection_error(self) -> None:
        sdk = _FakeOllamaSDK(_valid_output())
        sdk.chat_error = httpx.ReadError(
            "connection reset",
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )
        client = OllamaReviewClient(sdk_client=sdk)

        with self.assertRaisesRegex(LLMConnectionError, "connection was interrupted"):
            client.analyze_review("Great!", STANDARD_ASPECTS)

    def test_timeout_is_classified_as_transient_connection_error(self) -> None:
        sdk = _FakeOllamaSDK(_valid_output())
        sdk.chat_error = httpx.ReadTimeout(
            "timed out",
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )
        client = OllamaReviewClient(sdk_client=sdk)

        with self.assertRaisesRegex(LLMConnectionError, "timed out"):
            client.analyze_review("Great!", STANDARD_ASPECTS)


if __name__ == "__main__":
    unittest.main()
