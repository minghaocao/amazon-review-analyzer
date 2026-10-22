"""Offline release-documentation contracts for Amazon Review Analyzer v1.0."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"


class ReadmeV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = README_PATH.read_text(encoding="utf-8")

    def test_required_v1_user_sections_are_present(self) -> None:
        required = (
            "# Amazon Review Analyzer v1.0",
            "## What it does",
            "## Features",
            "## Requirements",
            "## Quick Start",
            "## One-click launcher",
            "## English / 中文",
            "## Input Methods",
            "## Amazon Quick Scan limitations",
            "## Saved Analyses",
            "## Competitor Comparison",
            "## Report Export",
            "## Privacy",
            "## Known Limitations",
            "## Troubleshooting",
        )
        for heading in required:
            with self.subTest(heading=heading):
                self.assertIn(heading, self.readme)

    def test_accuracy_claim_is_scoped_to_frozen_blind_v2(self) -> None:
        self.assertIn(
            "Aspect + Sentiment F1 = 85.66% on the frozen 150-case Blind V2 evaluation.",
            self.readme,
        )
        self.assertNotIn("97% accurate", self.readme)
        self.assertIn("不应表述为通用准确率", self.readme)

    def test_amazon_access_is_not_overpromised(self) -> None:
        self.assertIn(
            "Amazon may require login or limit automated access.",
            self.readme,
        )
        self.assertIn("不承诺稳定获取商品的全部评论", self.readme)
        self.assertIn("不绕过 CAPTCHA", self.readme)

    def test_local_privacy_boundaries_are_documented(self) -> None:
        for statement in (
            "`qwen3:8b` 通过本机 Ollama 推理",
            "Saved Analyses、comparisons 和 reports 均存储在本机项目目录",
            "Amazon Browser Import 会访问 Amazon 网站",
        ):
            with self.subTest(statement=statement):
                self.assertIn(statement, self.readme)

    def test_launcher_and_stable_runtime_are_documented(self) -> None:
        self.assertIn("~/Desktop/Amazon Review Analyzer.command", self.readme)
        self.assertIn("Stable mode / 1 worker", self.readme)
        self.assertIn("Terminal 中按 `Ctrl+C`", self.readme)


if __name__ == "__main__":
    unittest.main()
