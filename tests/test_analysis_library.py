"""Offline tests for Step 13.1 local saved-analysis persistence."""

from __future__ import annotations

import ast
import csv
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import analysis_library as library


class AnalysisLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.library_root = self.root / "library"
        self._write_source(self.source)
        self.metadata = {
            "analysis_run_id": "RUN_A",
            "analysis_candidate": "generalization_v2_candidate",
            "taxonomy_version": "generalization_v2_candidate",
            "model_provider": "ollama",
            "model_name": "qwen3:8b",
            "product_id": "P-A",
            "asin": "B012345678",
            "source_url": "https://www.amazon.com/dp/B012345678",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_source(
        root: Path,
        *,
        title: str = "Product",
        statuses: tuple[str, ...] = ("success", "success", "failed", "skipped"),
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "analyzed_reviews.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("review_id", "rating", "analysis_status", "summary"),
            )
            writer.writeheader()
            for index, status in enumerate(statuses, start=1):
                writer.writerow(
                    {
                        "review_id": f"R{index:03d}",
                        "rating": 5 if index == 1 else "",
                        "analysis_status": status,
                        "summary": f"{title} {index}",
                    }
                )
        (root / "aspect_summary.csv").write_text(
            "aspect,aspect_review_count,positive_rate,negative_rate\n"
            "Picture Quality,2,100,0\n",
            encoding="utf-8",
        )
        payloads = {
            "aspect_evidence.json": {"Picture Quality": {"positive": []}},
            "business_summary.json": {
                "data_quality": {
                    "total_input_rows": len(statuses),
                    "successful_reviews": statuses.count("success"),
                    "failed_reviews": statuses.count("failed"),
                    "skipped_reviews": statuses.count("skipped"),
                    "analysis_coverage": (
                        statuses.count("success") / len(statuses) if statuses else 0
                    ),
                },
                "sample_level": {"key": "early_signals"},
            },
            "sentiment_keywords.json": {"positive": [], "negative": []},
            "aspect_keywords.json": {"Picture Quality": []},
        }
        for filename, payload in payloads.items():
            (root / filename).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        (root / "keyword_summary.csv").write_text(
            "term,review_count,review_share\npicture,2,50\n", encoding="utf-8"
        )

    def _save(
        self,
        *,
        source: Path | None = None,
        product_name: str = "Roku Select Series 55\"",
        metadata: dict | None = None,
        token: str = "a7f3",
    ) -> library.SavedAnalysis:
        return library.save_analysis(
            source or self.source,
            product_name,
            "amazon",
            metadata or self.metadata,
            library_root=self.library_root,
            now=datetime(2026, 10, 15, 18, 30, 45, tzinfo=UTC),
            token_factory=lambda: token,
        )

    def test_save_analysis_creates_required_snapshot_and_manifest(self) -> None:
        saved = self._save()

        self.assertRegex(saved.analysis_id, r"^AN_20261015_183045_[a-f0-9]{4}$")
        self.assertTrue(saved.root.name.startswith("roku-select-series-55_an_"))
        self.assertEqual(library.ANALYSIS_LIBRARY_SCHEMA_VERSION, saved.manifest["schema_version"])
        self.assertEqual("Roku Select Series 55\"", saved.product_name)
        self.assertEqual(4, saved.manifest["imported_reviews"])
        self.assertEqual(2, saved.manifest["successful_reviews"])
        self.assertEqual(1, saved.manifest["failed_reviews"])
        self.assertEqual(1, saved.manifest["skipped_reviews"])
        self.assertEqual(0.5, saved.manifest["coverage"])
        self.assertTrue(saved.manifest["has_rating"])
        for filename in library.REQUIRED_ANALYSIS_FILES:
            self.assertEqual(
                (self.source / filename).read_bytes(),
                saved.artifact(filename).read_bytes(),
            )
            self.assertIn(filename, saved.manifest["content_hashes"])

    def test_duplicate_analysis_run_is_blocked_with_existing_id(self) -> None:
        first = self._save()
        with self.assertRaises(library.DuplicateAnalysisError) as caught:
            self._save(product_name="Renamed Product", token="beef")
        self.assertEqual(first.analysis_id, caught.exception.existing_analysis_id)
        self.assertEqual(1, len(library.list_saved_analyses(library_root=self.library_root)))

    def test_list_and_load_preserve_stable_product_name(self) -> None:
        saved = self._save()
        # Later source metadata and files cannot rename or mutate the snapshot.
        (self.source / "analysis_run_metadata.json").write_text(
            '{"product_name":"Changed"}', encoding="utf-8"
        )

        listed = library.list_saved_analyses(library_root=self.library_root)
        loaded = library.load_analysis(saved.analysis_id, library_root=self.library_root)

        self.assertEqual([saved.analysis_id], [item.analysis_id for item in listed])
        self.assertEqual("Roku Select Series 55\"", loaded.product_name)
        self.assertEqual(saved.manifest, library.get_manifest(saved.root, library_root=self.library_root))

    def test_unique_analysis_ids_do_not_depend_on_product_name(self) -> None:
        first = self._save(token="a7f3")
        source_b = self.root / "source-b"
        self._write_source(source_b, title="B")
        metadata_b = dict(self.metadata, analysis_run_id="RUN_B")
        second = self._save(source=source_b, metadata=metadata_b, token="b9e1")

        self.assertNotEqual(first.analysis_id, second.analysis_id)
        self.assertNotEqual(first.root, second.root)

    def test_saved_a_is_unchanged_after_source_mutation_and_saving_b(self) -> None:
        saved_a = self._save()
        original_bytes = {
            name: saved_a.artifact(name).read_bytes()
            for name in library.REQUIRED_ANALYSIS_FILES
        }
        (self.source / "analyzed_reviews.csv").write_text(
            "review_id,rating,analysis_status\nNEW,1,success\n", encoding="utf-8"
        )
        source_b = self.root / "source-b"
        self._write_source(source_b, title="B")
        self._save(
            source=source_b,
            metadata=dict(self.metadata, analysis_run_id="RUN_B"),
            token="b9e1",
        )

        reloaded_a = library.load_analysis(saved_a.analysis_id, library_root=self.library_root)
        self.assertEqual(
            original_bytes,
            {
                name: reloaded_a.artifact(name).read_bytes()
                for name in library.REQUIRED_ANALYSIS_FILES
            },
        )

    def test_content_hash_detects_snapshot_tampering(self) -> None:
        saved = self._save()
        saved.artifact("business_summary.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(library.SnapshotIntegrityError):
            library.load_analysis(saved.analysis_id, library_root=self.library_root)

    def test_missing_required_output_fails_before_creating_snapshot(self) -> None:
        (self.source / "aspect_keywords.json").unlink()
        with self.assertRaises(library.MissingAnalysisOutputError):
            self._save()
        self.assertFalse(self.library_root.exists())

    def test_arbitrary_source_requires_explicit_version_metadata(self) -> None:
        incomplete = {
            key: value
            for key, value in self.metadata.items()
            if key not in {"analysis_candidate", "taxonomy_version"}
        }
        with self.assertRaises(library.ManifestValidationError):
            self._save(metadata=incomplete)

        saved = self._save()
        self.assertEqual(
            "generalization_v2_candidate", saved.manifest["analysis_candidate"]
        )

    def test_arbitrary_source_cannot_claim_trusted_current_outputs(self) -> None:
        with self.assertRaises(library.LibraryPathError):
            library.save_analysis(
                self.source,
                "Product",
                "upload",
                {},
                library_root=self.library_root,
                trusted_current_outputs=True,
            )

    def test_path_traversal_and_unknown_artifact_are_rejected(self) -> None:
        saved = self._save()
        with self.assertRaises(library.LibraryPathError):
            library.load_analysis("../source", library_root=self.library_root)
        with self.assertRaises(library.LibraryPathError):
            saved.artifact("../analysis_manifest.json")

    def test_list_skips_corrupt_entries_without_hiding_valid_saved_analysis(self) -> None:
        saved = self._save()
        corrupt = self.library_root / "corrupt-entry"
        corrupt.mkdir()
        (corrupt / "analysis_manifest.json").write_text("{}", encoding="utf-8")

        self.assertEqual(
            [saved.analysis_id],
            [item.analysis_id for item in library.list_saved_analyses(library_root=self.library_root)],
        )

    def test_manifest_validation_rejects_schema_change(self) -> None:
        saved = self._save()
        manifest_path = saved.root / "analysis_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "future-schema"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaises(library.ManifestValidationError):
            library.load_analysis(saved.root, library_root=self.library_root)

    def test_module_has_no_pipeline_or_network_imports(self) -> None:
        tree = ast.parse(Path(library.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(
            imported.isdisjoint(
                {
                    "llm_client",
                    "review_analyzer",
                    "batch_analyzer",
                    "aggregator",
                    "keyword_analyzer",
                    "scraper",
                    "amazon_analysis",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
