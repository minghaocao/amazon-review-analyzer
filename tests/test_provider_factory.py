"""Offline tests for selecting an LLM provider."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llm_client  # noqa: E402
from llm_client import LLMConfigurationError, create_review_client  # noqa: E402


class ProviderFactoryTests(unittest.TestCase):
    def test_default_provider_is_ollama_and_does_not_check_openai_key(self) -> None:
        sentinel = object()
        with (
            patch.object(llm_client, "LLM_PROVIDER", "ollama"),
            patch.object(llm_client, "OllamaReviewClient", return_value=sentinel) as factory,
            patch.object(llm_client, "OpenAIReviewClient") as openai_factory,
        ):
            result = create_review_client()

        self.assertIs(sentinel, result)
        factory.assert_called_once_with()
        openai_factory.assert_not_called()

    def test_provider_name_is_case_and_whitespace_insensitive(self) -> None:
        sentinel = object()
        with patch.object(
            llm_client,
            "OllamaReviewClient",
            return_value=sentinel,
        ):
            self.assertIs(sentinel, create_review_client("  OLLAMA  "))

    def test_openai_is_created_only_when_explicitly_selected(self) -> None:
        sentinel = object()
        with patch.object(
            llm_client,
            "OpenAIReviewClient",
            return_value=sentinel,
        ) as factory:
            result = create_review_client("openai")

        self.assertIs(sentinel, result)
        factory.assert_called_once_with()

    def test_unknown_provider_has_clear_error(self) -> None:
        with self.assertRaisesRegex(LLMConfigurationError, "Unsupported LLM_PROVIDER"):
            create_review_client("unknown-provider")


if __name__ == "__main__":
    unittest.main()
