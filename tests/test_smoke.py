"""Step 1 的最小项目结构自检。"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import check_project_setup  # noqa: E402


class ProjectSetupTests(unittest.TestCase):
    """确认基础目录和入口模块可以正常加载。"""

    def test_required_directories_exist(self) -> None:
        is_ready, missing = check_project_setup()
        self.assertTrue(is_ready, f"Missing directories: {missing}")

    def test_required_module_files_exist(self) -> None:
        required_modules = {
            "main.py",
            "config.py",
            "input_loader.py",
            "cleaner.py",
            "aspect_taxonomy.py",
            "review_analyzer.py",
            "llm_client.py",
            "batch_analyzer.py",
            "aggregator.py",
            "keyword_analyzer.py",
            "evaluator.py",
            "visualization.py",
            "scraper.py",
        }
        missing = sorted(name for name in required_modules if not (PROJECT_ROOT / name).is_file())
        self.assertEqual([], missing)

    def test_scaffold_modules_can_be_imported(self) -> None:
        module_names = (
            "aggregator",
            "aspect_taxonomy",
            "batch_analyzer",
            "cleaner",
            "config",
            "evaluator",
            "input_loader",
            "keyword_analyzer",
            "llm_client",
            "review_analyzer",
            "scraper",
            "visualization",
        )
        for module_name in module_names:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))


if __name__ == "__main__":
    unittest.main()
