"""Offline tests for the Step 8.10 diagnostic study."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import socket
import subprocess
import urllib.request

import generalization_error_analysis as study
import main as main_module
from evaluator import evaluate_case


def _mention(aspect: str, sentiment: str, evidence: str) -> dict[str, str]:
    return {"aspect": aspect, "sentiment": sentiment, "evidence": evidence}


def _actual(aspect: str, sentiment: str, evidence: str) -> dict[str, str]:
    return {
        "raw_aspect": aspect,
        "aspect": aspect,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _evaluated_case(
    case_id: str,
    review: str,
    gold: list[dict[str, str]],
    predicted: list[dict[str, str]],
    *,
    tags: list[str] | None = None,
) -> dict[str, object]:
    result = evaluate_case(
        {"case_id": case_id, "review": review, "expected_aspects": gold},
        {
            "summary": "offline",
            "aspects": predicted,
            "positive_points": [],
            "negative_points": [],
        },
    )
    result["base_exact_match"] = bool(result["exact_match"])
    # Synthetic fixtures use evaluator exactness; all evidence is verbatim.
    result["tags"] = tags or []
    result["notes"] = "Offline fixture."
    return result


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _full_summary(**overrides: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "aspect_precision": 0.0,
        "aspect_recall": 0.0,
        "aspect_f1": 0.0,
        "aspect_only_precision": 0.0,
        "aspect_only_recall": 0.0,
        "aspect_only_f1": 0.0,
        "sentiment_accuracy": 0.0,
        "evidence_validity_rate": 1.0,
        "evidence_match_rate": 0.0,
        "exact_case_accuracy": 0.0,
        "exact_matches": 0,
        "aspect_true_positives": 0,
        "aspect_false_positives": 0,
        "aspect_false_negatives": 0,
    }
    summary.update(overrides)
    return summary


class FreezeManifestTests(unittest.TestCase):
    def _files(self, root: Path) -> dict[str, Path]:
        paths = {
            "dataset": root / "evaluation" / "final_blind_cases.json",
            "manifest": root / "evaluation" / "final_blind_manifest.json",
            "predictions": root / "outputs" / "final_blind_predictions.json",
            "report": root / "outputs" / "final_blind_evaluation.json",
            "errors": root / "outputs" / "final_blind_errors.csv",
            "demo": root / "outputs" / "demo.json",
        }
        _write_json(paths["dataset"], [{"case_id": "F001"}])
        _write_json(paths["manifest"], {"sha256": study.file_sha256(paths["dataset"])})
        _write_json(
            paths["predictions"],
            {
                "provider": "ollama",
                "model": "qwen3:8b",
                "dataset_sha256": study.file_sha256(paths["dataset"]),
                "predictions": [],
            },
        )
        _write_json(
            paths["report"],
            {
                "provider": "ollama",
                "model": "qwen3:8b",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "case_count": 1,
                "summary": _full_summary(aspect_f1=0.5),
            },
        )
        paths["errors"].parent.mkdir(parents=True, exist_ok=True)
        paths["errors"].write_text("case_id,error_type\n", encoding="utf-8")
        _write_json(paths["demo"], {"frozen": True})
        return paths

    def test_freeze_manifest_records_and_verifies_all_five_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._files(root)
            destination = root / "evaluation" / "final_blind_freeze_manifest.json"
            frozen = study.create_final_blind_freeze_manifest(
                destination,
                dataset_path=paths["dataset"],
                manifest_path=paths["manifest"],
                predictions_path=paths["predictions"],
                report_path=paths["report"],
                errors_path=paths["errors"],
                project_root=root,
                demo_paths=(paths["demo"],),
            )
            verified = study.verify_final_blind_freeze(destination, project_root=root)
            self.assertEqual(frozen, verified)
            self.assertEqual("ollama", verified["provider"])
            self.assertEqual("qwen3:8b", verified["model"])
            self.assertEqual(1, len(study.verify_demo_freeze(verified, project_root=root)))
            self.assertEqual(set(study.FINAL_METRIC_KEYS), set(verified["metrics"]))

    def test_hash_guard_fails_closed_after_frozen_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._files(root)
            destination = root / "evaluation" / "final_blind_freeze_manifest.json"
            study.create_final_blind_freeze_manifest(
                destination,
                dataset_path=paths["dataset"],
                manifest_path=paths["manifest"],
                predictions_path=paths["predictions"],
                report_path=paths["report"],
                errors_path=paths["errors"],
                project_root=root,
                demo_paths=(paths["demo"],),
            )
            paths["errors"].write_text("case_id,error_type\nF001,CHANGED\n")
            with self.assertRaisesRegex(study.GeneralizationStudyError, "changed"):
                study.verify_final_blind_freeze(destination, project_root=root)

    def test_manifest_rejects_missing_metric_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._files(root)
            report = json.loads(paths["report"].read_text())
            del report["summary"]["aspect_only_precision"]
            _write_json(paths["report"], report)
            with self.assertRaisesRegex(study.GeneralizationStudyError, "missing required metrics"):
                study.create_final_blind_freeze_manifest(
                    root / "evaluation" / "freeze.json",
                    dataset_path=paths["dataset"],
                    manifest_path=paths["manifest"],
                    predictions_path=paths["predictions"],
                    report_path=paths["report"],
                    errors_path=paths["errors"],
                    project_root=root,
                    demo_paths=(paths["demo"],),
                )

    def test_frozen_path_binding_rejects_substituted_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._files(root)
            destination = root / "evaluation" / "final_blind_freeze_manifest.json"
            frozen = study.create_final_blind_freeze_manifest(
                destination,
                dataset_path=paths["dataset"],
                manifest_path=paths["manifest"],
                predictions_path=paths["predictions"],
                report_path=paths["report"],
                errors_path=paths["errors"],
                project_root=root,
                demo_paths=(paths["demo"],),
            )
            replacement = root / "replacement.json"
            _write_json(replacement, [{"case_id": "F999"}])
            with self.assertRaisesRegex(study.GeneralizationStudyError, "not the frozen"):
                study._assert_frozen_input_paths(
                    frozen,
                    project_root=root,
                    dataset_path=replacement,
                    pre_evaluation_manifest_path=paths["manifest"],
                    predictions_path=paths["predictions"],
                    final_report_path=paths["report"],
                    errors_path=paths["errors"],
                )

    def test_demo_hash_guard_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._files(root)
            destination = root / "evaluation" / "final_blind_freeze_manifest.json"
            frozen = study.create_final_blind_freeze_manifest(
                destination,
                dataset_path=paths["dataset"],
                manifest_path=paths["manifest"],
                predictions_path=paths["predictions"],
                report_path=paths["report"],
                errors_path=paths["errors"],
                project_root=root,
                demo_paths=(paths["demo"],),
            )
            paths["demo"].write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(study.GeneralizationStudyError, "Demo file changed"):
                study.verify_demo_freeze(frozen, project_root=root)


class ErrorAnalysisTests(unittest.TestCase):
    def test_directional_confusions_use_evaluator_wrong_aspect_pairs(self) -> None:
        cases = [
            _evaluated_case(
                "F001",
                "The box looked polished.",
                [_mention("Packaging", "positive", "box looked polished")],
                [_actual("Design", "positive", "box looked polished")],
            ),
            _evaluated_case(
                "F002",
                "The box looked refined.",
                [_mention("Packaging", "positive", "box looked refined")],
                [_actual("Design", "positive", "box looked refined")],
            ),
        ]
        directional, pairs = study.build_directional_confusions(cases)
        self.assertEqual(2, directional[0]["count"])
        self.assertEqual("Packaging", directional[0]["gold_aspect"])
        self.assertEqual("Design", directional[0]["predicted_aspect"])
        self.assertEqual(2, pairs[0]["count"])

    def test_per_aspect_tp_fp_fn_use_multiset_counts(self) -> None:
        case = _evaluated_case(
            "F001",
            "The image is crisp but later looks dim.",
            [
                _mention("Picture Quality", "positive", "image is crisp"),
                _mention("Picture Quality", "negative", "later looks dim"),
            ],
            [_actual("Picture Quality", "positive", "image is crisp")],
        )
        rows, unknown = study.calculate_aspect_metrics([case])
        picture = next(row for row in rows if row["aspect"] == "Picture Quality")
        self.assertEqual([], unknown)
        self.assertEqual((2, 1, 0, 1), (
            picture["support"], picture["tp"], picture["fp"], picture["fn"]
        ))
        self.assertEqual(1.0, picture["precision"])
        self.assertEqual(0.5, picture["recall"])
        self.assertAlmostEqual(2 / 3, picture["f1"])
        self.assertEqual(1, picture["aspect_sentiment_tp"])
        self.assertEqual(0, picture["aspect_sentiment_fp"])
        self.assertEqual(1, picture["aspect_sentiment_fn"])
        self.assertEqual(15, len(rows))
        sound = next(row for row in rows if row["aspect"] == "Sound Quality")
        self.assertEqual((0, 0, 0, 0.0), (
            sound["support"], sound["tp"], sound["fp"], sound["f1"]
        ))

    def test_per_aspect_false_positive_is_assigned_to_predicted_label(self) -> None:
        case = _evaluated_case(
            "F001",
            "The box looks polished.",
            [_mention("Packaging", "positive", "box looks polished")],
            [_actual("Design", "positive", "box looks polished")],
        )
        rows, _ = study.calculate_aspect_metrics([case])
        by_aspect = {row["aspect"]: row for row in rows}
        self.assertEqual(1, by_aspect["Packaging"]["fn"])
        self.assertEqual(1, by_aspect["Design"]["fp"])
        stable, difficult = study.rank_aspect_stability(rows)
        self.assertTrue(all(row["support"] > 0 for row in stable + difficult))

    def test_over_split_is_high_confidence_and_multilabel(self) -> None:
        case = _evaluated_case(
            "F001",
            "The agent listened, emailed a form, and followed up.",
            [
                _mention(
                    "Customer Service",
                    "positive",
                    "agent listened, emailed a form, and followed up",
                )
            ],
            [
                _actual("Customer Service", "positive", "agent listened"),
                _actual("Customer Service", "positive", "emailed a form"),
                _actual("Customer Service", "positive", "followed up"),
            ],
        )
        classified = study.classify_non_exact_case(case)
        self.assertEqual(["Customer Service"], classified["over_split_aspects"])
        self.assertIn("MENTION_OVER_SPLITTING", classified["all_error_categories"])
        self.assertIn("OVER_EXTRACTION", classified["all_error_categories"])
        self.assertEqual("high", classified["classification_confidence"])

    def test_over_merge_detects_two_gold_spans_inside_one_prediction(self) -> None:
        case = _evaluated_case(
            "F001",
            "Pans stay clean and subtitles remain crisp.",
            [
                _mention("Picture Quality", "positive", "Pans stay clean"),
                _mention("Picture Quality", "positive", "subtitles remain crisp"),
            ],
            [
                _actual(
                    "Picture Quality",
                    "positive",
                    "Pans stay clean and subtitles remain crisp",
                )
            ],
        )
        classified = study.classify_non_exact_case(case)
        self.assertEqual(["Picture Quality"], classified["over_merged_aspects"])
        self.assertIn("MENTION_OVER_MERGING", classified["all_error_categories"])

    def test_undercount_without_span_containment_is_not_over_merge(self) -> None:
        case = _evaluated_case(
            "F001",
            "The screen is crisp, subtitles are detailed, and brightness varies.",
            [
                _mention("Picture Quality", "positive", "screen is crisp"),
                _mention("Picture Quality", "positive", "subtitles are detailed"),
            ],
            [_actual("Picture Quality", "neutral", "brightness varies")],
        )
        classified = study.classify_non_exact_case(case)
        self.assertEqual([], classified["over_merged_aspects"])
        self.assertNotIn("MENTION_OVER_MERGING", classified["all_error_categories"])

    def test_independent_same_aspect_extra_is_not_called_over_split(self) -> None:
        case = _evaluated_case(
            "F001",
            "The picture is crisp, and the unrelated frame is shiny.",
            [_mention("Picture Quality", "positive", "picture is crisp")],
            [
                _actual("Picture Quality", "positive", "picture is crisp"),
                _actual("Picture Quality", "positive", "frame is shiny"),
            ],
        )
        classified = study.classify_non_exact_case(case)
        self.assertEqual([], classified["over_split_aspects"])
        self.assertIn("OVER_EXTRACTION", classified["all_error_categories"])

    def test_cross_aspect_and_equal_count_cases_are_not_split_or_merge(self) -> None:
        cross = _evaluated_case(
            "F001",
            "The picture is crisp and the frame is solid.",
            [
                _mention("Picture Quality", "positive", "picture is crisp"),
                _mention("Build Quality", "positive", "frame is solid"),
            ],
            [
                _actual("Picture Quality", "positive", "picture is crisp"),
                _actual("Design", "positive", "frame is solid"),
            ],
        )
        equal_count = _evaluated_case(
            "F002",
            "The picture starts crisp but later looks dim.",
            [
                _mention("Picture Quality", "positive", "starts crisp"),
                _mention("Picture Quality", "negative", "later looks dim"),
            ],
            [
                _actual("Picture Quality", "positive", "starts crisp"),
                _actual("Picture Quality", "positive", "later looks dim"),
            ],
        )
        for case in (cross, equal_count):
            classified = study.classify_non_exact_case(case)
            self.assertEqual([], classified["over_split_aspects"])
            self.assertEqual([], classified["over_merged_aspects"])

    def test_gold_ambiguity_requires_explicit_annotation_signal(self) -> None:
        ambiguous = _evaluated_case(
            "F001",
            "The surface feels unusual.",
            [_mention("Other", "neutral", "surface feels unusual")],
            [_actual("Design", "neutral", "surface feels unusual")],
            tags=["ambiguous"],
        )
        clear = dict(ambiguous)
        clear["tags"] = ["boundary"]
        self.assertTrue(
            study.classify_non_exact_case(ambiguous)["possible_gold_ambiguity"]
        )
        self.assertFalse(
            study.classify_non_exact_case(clear)["possible_gold_ambiguity"]
        )

    def test_same_aspect_repeated_analysis_separates_component_and_whole_case(self) -> None:
        case = _evaluated_case(
            "F001",
            "Pans are clean and subtitles crisp, but delivery was late.",
            [
                _mention("Picture Quality", "positive", "Pans are clean"),
                _mention("Picture Quality", "positive", "subtitles crisp"),
                _mention("Delivery", "negative", "delivery was late"),
            ],
            [
                _actual("Picture Quality", "positive", "Pans are clean"),
                _actual("Picture Quality", "positive", "subtitles crisp"),
                _actual("Delivery", "positive", "delivery was late"),
            ],
        )
        rows, summary = study.analyze_same_aspect_multi_mentions([case])
        self.assertEqual(1, summary["total_cases"])
        self.assertTrue(rows[0]["repeated_component_fully_correct"])
        self.assertFalse(rows[0]["whole_case_strict_exact"])

    def test_sentiment_error_classification_is_one_primary_per_error(self) -> None:
        case = _evaluated_case(
            "F001",
            "The picture is not bad.",
            [_mention("Picture Quality", "positive", "not bad")],
            [_actual("Picture Quality", "negative", "not bad")],
        )
        rows, summary = study.classify_sentiment_errors([case])
        self.assertEqual(1, len(rows))
        self.assertEqual("negation", rows[0]["primary_category"])
        self.assertEqual(1, sum(summary["primary_category_counts"].values()))

    def test_weak_evidence_cross_opinion_alignment_uses_contract_category(self) -> None:
        case = {
            "case_id": "F001",
            "expected_aspects": [
                _mention("System Performance", "positive", "results appear instantly"),
                _mention("Ease of Use", "negative", "sequence is hard to remember"),
            ],
            "errors": [
                {
                    "error_type": "WEAK_EVIDENCE_MATCH",
                    "expected_aspect": "System Performance",
                    "expected_evidence": "results appear instantly",
                    "actual_evidence": "sequence is hard to remember",
                    "similarity": 0,
                }
            ],
        }
        analysis = study.analyze_weak_evidence([case])
        self.assertEqual("none_of_requested_categories", analysis["records"][0]["classification"])
        self.assertEqual("cross_opinion_alignment", analysis["records"][0]["alignment_subtype"])

    def test_unrelated_review_cues_do_not_become_causal_categories(self) -> None:
        case = _evaluated_case(
            "F001",
            "The connection is awkward, but the picture is not bad after a week.",
            [_mention("Connectivity", "negative", "connection is awkward")],
            [_actual("Setup / Installation", "negative", "connection is awkward")],
        )
        classified = study.classify_non_exact_case(case)
        self.assertIn("ASPECT_BOUNDARY_CONFUSION", classified["all_error_categories"])
        self.assertNotIn("NEGATION_SCOPE_ERROR", classified["all_error_categories"])
        self.assertNotIn("CONTRAST_ERROR", classified["all_error_categories"])
        self.assertNotIn("TEMPORAL_CHANGE_ERROR", classified["all_error_categories"])
        self.assertIn("negation", classified["observed_linguistic_cues"])

    def test_gold_empty_hallucination_is_not_other_underuse(self) -> None:
        case = _evaluated_case(
            "F001",
            "The box contains a cable.",
            [],
            [_actual("Connectivity", "neutral", "contains a cable")],
        )
        classified = study.classify_non_exact_case(case)
        self.assertIn("ATTRIBUTE_VS_EVALUATION_ERROR", classified["all_error_categories"])
        self.assertNotIn("OTHER_UNDERUSE", classified["all_error_categories"])

    def test_strict_exact_breakdown_distinguishes_aspect_and_granularity(self) -> None:
        case = _evaluated_case(
            "F001",
            "The box looks polished.",
            [_mention("Packaging", "positive", "box looks polished")],
            [_actual("Design", "positive", "box looks polished")],
        )
        classifications = [study.classify_non_exact_case(case)]
        breakdown = study.strict_exact_failure_breakdown([case], classifications)
        self.assertEqual(1, breakdown["aspect_classification_failure_cases"])
        self.assertEqual(0, breakdown["mention_granularity_failure_cases"])

    def test_strict_breakdown_uses_exclusive_primary_attribution(self) -> None:
        split = _evaluated_case(
            "F001",
            "The screen stays crisp and detailed.",
            [_mention("Picture Quality", "positive", "screen stays crisp and detailed")],
            [
                _actual("Picture Quality", "positive", "screen stays crisp"),
                _actual("Picture Quality", "positive", "detailed"),
            ],
        )
        classifications = [study.classify_non_exact_case(split)]
        breakdown = study.strict_exact_failure_breakdown([split], classifications)
        self.assertEqual(0, breakdown["aspect_classification_failure_cases"])
        self.assertEqual(1, breakdown["mention_granularity_failure_cases"])
        self.assertTrue(breakdown["exclusive_primary_attribution"])

    def test_sentiment_alignment_disagreement_falls_back_to_manual_other(self) -> None:
        case = {
            "case_id": "F001",
            "review": "The picture is crisp but sound is rough.",
            "expected_aspects": [
                _mention("Picture Quality", "positive", "picture is crisp"),
                _mention("Sound Quality", "negative", "sound is rough"),
            ],
            "errors": [
                {
                    "error_type": "WRONG_SENTIMENT",
                    "expected_aspect": "Picture Quality",
                    "expected_sentiment": "positive",
                    "actual_sentiment": "negative",
                    "expected_evidence": "picture is crisp",
                    "actual_evidence": "sound is rough",
                }
            ],
        }
        rows, _ = study.classify_sentiment_errors([case])
        self.assertEqual("other", rows[0]["primary_category"])
        self.assertTrue(rows[0]["alignment_disagreement"])
        self.assertTrue(rows[0]["needs_manual_review"])

    def test_multi_aspect_sentiment_only_is_not_structural_failure(self) -> None:
        case = _evaluated_case(
            "F001",
            "The screen is crisp and the sound is rough.",
            [
                _mention("Picture Quality", "positive", "screen is crisp"),
                _mention("Sound Quality", "negative", "sound is rough"),
            ],
            [
                _actual("Picture Quality", "negative", "screen is crisp"),
                _actual("Sound Quality", "positive", "sound is rough"),
            ],
        )
        classified = study.classify_non_exact_case(case)
        self.assertNotIn("MULTI_ASPECT_SENTENCE_FAILURE", classified["all_error_categories"])
        self.assertEqual("UNRESOLVED", classified["primary_error_category"])
        summary = study._second_level_summary_rows([classified])
        unresolved = next(row for row in summary if row["error_category"] == "UNRESOLVED")
        self.assertEqual(1, unresolved["case_count"])

    def test_cross_opinion_alignment_is_multi_aspect_failure(self) -> None:
        case = _evaluated_case(
            "F001",
            "Search is instant, yet navigation is hard.",
            [
                _mention("System Performance", "positive", "Search is instant"),
                _mention("Ease of Use", "negative", "navigation is hard"),
            ],
            [
                _actual("System Performance", "negative", "navigation is hard"),
                _actual("Ease of Use", "positive", "Search is instant"),
            ],
        )
        classified = study.classify_non_exact_case(case)
        self.assertIn("MULTI_ASPECT_SENTENCE_FAILURE", classified["all_error_categories"])
        self.assertTrue(classified["needs_manual_review"])


class DifficultyAndSpecificationTests(unittest.TestCase):
    def test_dataset_difficulty_statistics_cover_required_structures(self) -> None:
        cases = [
            {
                "case_id": "G001",
                "review": "The screen is bright, but the sound is not good after a week.",
                "expected_aspects": [
                    _mention("Picture Quality", "positive", "screen is bright"),
                    _mention("Sound Quality", "negative", "sound is not good"),
                ],
            },
            {"case_id": "G002", "review": "Arrived Tuesday.", "expected_aspects": []},
        ]
        stats = study.dataset_difficulty_statistics(cases)
        self.assertEqual(2, stats["review_count"])
        self.assertEqual(1.0, stats["mentions_per_review"])
        self.assertEqual(1, stats["mention_cardinality"]["0"]["count"])
        self.assertEqual(1, stats["language_structures"]["negation"]["count"])
        self.assertEqual(1, stats["language_structures"]["contrast"]["count"])

    def test_dev_v2_spec_is_only_a_spec_and_mentions_sum_to_target(self) -> None:
        spec = study.build_generalization_dev_v2_spec([], [])
        self.assertTrue(spec["specification_only"])
        self.assertEqual(210, sum(spec["target_aspect_distribution"].values()))
        self.assertEqual(110, sum(spec["case_cardinality_distribution"].values()))
        self.assertIn("mixed_positive_negative", spec["overlapping_language_and_structure_targets"])
        self.assertIn("same_aspect_mixed_polarity", spec["overlapping_language_and_structure_targets"])
        serialized = json.dumps(spec)
        self.assertNotIn("F001", serialized)


class SerializationAndCliTests(unittest.TestCase):
    def test_all_required_csv_and_json_outputs_serialize(self) -> None:
        report = {
            "case_analysis": [{
                "case_id": "F001",
                "review_text": "The box looks polished.",
                "gold_mentions": [_mention("Packaging", "positive", "box looks polished")],
                "predicted_mentions": [_actual("Design", "positive", "box looks polished")],
                "original_error_types": ["WRONG_ASPECT"],
                "primary_error_category": "ASPECT_BOUNDARY_CONFUSION",
                "secondary_error_categories": [],
                "affected_gold_aspects": ["Packaging"],
                "affected_predicted_aspects": ["Design"],
                "possible_gold_ambiguity": False,
                "classification_confidence": "high",
                "needs_manual_review": False,
                "observed_linguistic_cues": [],
                "over_split_aspects": [],
                "over_merged_aspects": [],
                "notes": "fixture",
            }],
            "second_level_error_distribution": [{
                "error_category": "ASPECT_BOUNDARY_CONFUSION",
                "case_count": 1,
                "percentage_of_non_exact": 1.0,
                "case_ids": ["F001"],
            }],
            "aspect_confusion_directions": [{
                "gold_aspect": "Packaging",
                "predicted_aspect": "Design",
                "count": 1,
                "case_ids": ["F001"],
                "percentage_of_wrong_aspect": 1.0,
            }],
            "aspect_metrics": [{
                "aspect": "Packaging", "support": 1, "tp": 0, "fp": 0,
                "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "aspect_sentiment_tp": 0, "aspect_sentiment_fp": 0,
                "aspect_sentiment_fn": 1, "aspect_sentiment_precision": 0.0,
                "aspect_sentiment_recall": 0.0, "aspect_sentiment_f1": 0.0,
                "metric_scope": "aspect_only_multiset",
            }],
            "same_aspect_multi_mention_rows": [{
                "case_id": "F002", "repeated_aspects": ["Connectivity"],
                "gold_repeated_mention_count": 2, "predicted_repeated_mention_count": 1,
                "whole_case_strict_exact": False, "repeated_component_fully_correct": False,
                "one_mention_missed": True, "all_repeated_mentions_missed": False,
                "over_merged": True, "over_split": False, "wrong_sentiment": False,
                "wrong_aspect": False, "duplicate_generated": False,
            }],
            "sentiment_error_rows": [{
                "case_id": "F003", "aspect": "Picture Quality",
                "expected_sentiment": "neutral", "predicted_sentiment": "positive",
                "expected_evidence": "looks ordinary", "predicted_evidence": "looks ordinary",
                "primary_category": "neutral_boundary", "secondary_categories": [],
                "classification_confidence": "high", "needs_manual_review": False,
                "alignment_disagreement": False, "diagnostic_note": "fixture",
            }],
            "dataset_difficulty_comparison": {"ok": True},
            "generalization_dev_v2_spec": {"specification_only": True},
        }
        with tempfile.TemporaryDirectory() as temporary:
            paths = study.save_generalization_error_outputs(
                report, output_dir=Path(temporary)
            )
            self.assertEqual(9, len(paths))
            self.assertEqual({
                "generalization_error_report",
                "generalization_case_analysis",
                "generalization_error_summary",
                "aspect_confusion_matrix",
                "aspect_metrics",
                "same_aspect_multi_mention_analysis",
                "sentiment_error_analysis",
                "dataset_difficulty_comparison",
                "generalization_dev_v2_spec",
            }, set(paths))
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            for key in (
                "generalization_error_report",
                "dataset_difficulty_comparison",
                "generalization_dev_v2_spec",
            ):
                self.assertIsInstance(json.loads(Path(paths[key]).read_text()), dict)
            case_rows = list(csv.DictReader(StringIO(
                Path(paths["generalization_case_analysis"]).read_text(encoding="utf-8")
            )))
            self.assertEqual("F001", case_rows[0]["case_id"])
            self.assertEqual(["Packaging"], json.loads(case_rows[0]["affected_gold_aspects"]))
            confusion_rows = list(csv.DictReader(StringIO(
                Path(paths["aspect_confusion_matrix"]).read_text(encoding="utf-8")
            )))
            self.assertEqual(["F001"], json.loads(confusion_rows[0]["case_ids"]))
            sentiment_rows = list(csv.DictReader(StringIO(
                Path(paths["sentiment_error_analysis"]).read_text(encoding="utf-8")
            )))
            self.assertEqual([], json.loads(sentiment_rows[0]["secondary_categories"]))
            for key in (
                "generalization_error_summary",
                "aspect_metrics",
                "same_aspect_multi_mention_analysis",
            ):
                parsed = list(csv.DictReader(StringIO(Path(paths[key]).read_text(encoding="utf-8"))))
                self.assertTrue(parsed)

    def test_cli_dispatches_offline_study_without_other_entrypoints(self) -> None:
        fake_report = {
            "final_blind_metrics": {},
            "strict_exact_failure_analysis": {},
            "second_level_error_distribution": [],
            "aspect_confusion_directions": [],
            "most_difficult_5_aspects": [],
            "same_aspect_multi_mention_analysis": {},
            "non_exact_case_count": 0,
            "outputs": {},
        }
        forbidden = AssertionError("forbidden online/model entrypoint was called")
        with (
            patch.object(main_module, "run_generalization_error_study", return_value=fake_report) as run,
            patch.object(main_module, "create_review_client", side_effect=forbidden) as llm,
            patch.object(main_module, "run_amazon_access_diagnostic", side_effect=forbidden) as amazon,
            patch.object(main_module, "run_final_blind_evaluation", side_effect=forbidden) as final,
            patch.object(main_module, "run_generalization_evaluation", side_effect=forbidden) as dev,
            redirect_stdout(StringIO()) as output,
        ):
            code = main_module.cli(["--analyze-generalization-errors"])
        self.assertEqual(0, code)
        run.assert_called_once_with()
        llm.assert_not_called()
        amazon.assert_not_called()
        final.assert_not_called()
        dev.assert_not_called()
        self.assertIn("No model inference was performed", output.getvalue())


class OfflineRunnerIntegrationTests(unittest.TestCase):
    def test_runner_reads_frozen_inputs_writes_outputs_and_preserves_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review = "The box looks polished."
            gold_case = {
                "case_id": "F001",
                "review": review,
                "expected_aspects": [
                    _mention("Packaging", "positive", "box looks polished")
                ],
                "notes": "Boundary fixture.",
                "tags": ["boundary"],
            }
            final_case = _evaluated_case(
                "F001",
                review,
                gold_case["expected_aspects"],
                [_actual("Design", "positive", "box looks polished")],
            )
            dev_gold = {
                "case_id": "G001",
                "review": "The image is crisp.",
                "expected_aspects": [
                    _mention("Picture Quality", "positive", "image is crisp")
                ],
                "notes": "Dev fixture.",
                "tags": ["single"],
            }
            dev_case = _evaluated_case(
                "G001",
                dev_gold["review"],
                dev_gold["expected_aspects"],
                [_actual("Picture Quality", "positive", "image is crisp")],
            )

            dataset = root / "evaluation" / "final_blind_cases.json"
            pre_manifest = root / "evaluation" / "final_blind_manifest.json"
            predictions = root / "outputs" / "final_blind_predictions.json"
            final_report = root / "outputs" / "final_blind_evaluation.json"
            errors = root / "outputs" / "final_blind_errors.csv"
            dev_dataset = root / "evaluation" / "generalization_dev_cases.json"
            dev_manifest = root / "evaluation" / "generalization_dev_manifest.json"
            dev_report = root / "evaluation" / "evaluation_generalization_dev_qwen3_8b.json"
            demo = root / "outputs" / "demo.json"
            _write_json(dataset, [gold_case])
            dataset_hash = study.file_sha256(dataset)
            _write_json(pre_manifest, {"sha256": dataset_hash})
            _write_json(
                predictions,
                {
                    "provider": "ollama",
                    "model": "qwen3:8b",
                    "dataset_sha256": dataset_hash,
                    "predictions": [{"case_id": "F001"}],
                },
            )
            _write_json(
                final_report,
                {
                    "provider": "ollama",
                    "model": "qwen3:8b",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "case_count": 1,
                    "dataset_sha256": dataset_hash,
                    "summary": _full_summary(),
                    "error_counts": {"WRONG_ASPECT": 1},
                    "cases": [final_case],
                },
            )
            errors.parent.mkdir(parents=True, exist_ok=True)
            with errors.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("case_id", "error_type"))
                writer.writeheader()
                writer.writerow({"case_id": "F001", "error_type": "WRONG_ASPECT"})
            _write_json(dev_dataset, [dev_gold])
            dev_hash = study.file_sha256(dev_dataset)
            _write_json(
                dev_manifest,
                {"dataset": dev_dataset.name, "case_count": 1, "sha256": dev_hash},
            )
            _write_json(
                dev_report,
                {
                    "provider": "ollama",
                    "model": "qwen3:8b",
                    "stage": "optimized",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "case_count": 1,
                    "dataset_sha256": dev_hash,
                    "summary": _full_summary(
                        aspect_precision=1.0,
                        aspect_recall=1.0,
                        aspect_f1=1.0,
                        aspect_only_precision=1.0,
                        aspect_only_recall=1.0,
                        aspect_only_f1=1.0,
                        sentiment_accuracy=1.0,
                        evidence_match_rate=1.0,
                        exact_case_accuracy=1.0,
                        exact_matches=1,
                        aspect_true_positives=1,
                    ),
                    "cases": [dev_case],
                },
            )
            _write_json(demo, {"frozen": True})
            freeze = root / "evaluation" / "final_blind_freeze_manifest.json"
            study.create_final_blind_freeze_manifest(
                freeze,
                dataset_path=dataset,
                manifest_path=pre_manifest,
                predictions_path=predictions,
                report_path=final_report,
                errors_path=errors,
                project_root=root,
                demo_paths=(demo,),
            )
            frozen_paths = (dataset, pre_manifest, predictions, final_report, errors)
            before = {path: study.file_sha256(path) for path in frozen_paths}
            forbidden = AssertionError("offline runner attempted network or process access")
            with (
                patch.object(socket, "create_connection", side_effect=forbidden) as socket_call,
                patch.object(urllib.request, "urlopen", side_effect=forbidden) as urlopen,
                patch.object(subprocess, "Popen", side_effect=forbidden) as popen,
                patch.object(subprocess, "run", side_effect=forbidden) as process_run,
            ):
                report = study.run_generalization_error_study(
                    project_root=root,
                    freeze_manifest_path=freeze,
                    dataset_path=dataset,
                    pre_evaluation_manifest_path=pre_manifest,
                    predictions_path=predictions,
                    final_report_path=final_report,
                    errors_path=errors,
                    development_dataset_path=dev_dataset,
                    development_manifest_path=dev_manifest,
                    development_report_path=dev_report,
                    output_dir=root / "study",
                )
            socket_call.assert_not_called()
            urlopen.assert_not_called()
            popen.assert_not_called()
            process_run.assert_not_called()
            after = {path: study.file_sha256(path) for path in frozen_paths}
        self.assertEqual(before, after)
        self.assertTrue(report["freeze_verification"]["hashes_unchanged"])
        self.assertTrue(report["freeze_verification"]["demo_hashes_unchanged"])
        self.assertTrue(report["freeze_verification"]["development_hashes_unchanged"])
        self.assertFalse(report["input_cross_checks"]["model_inference_performed"])
        self.assertIn("multiple_error_record_cases", report["strict_exact_failure_analysis"])


class RealFrozenArtifactContractTests(unittest.TestCase):
    def _require_real_artifacts(self) -> None:
        required = (
            study.FINAL_FREEZE_MANIFEST_PATH,
            study.FINAL_DATASET_PATH,
            study.FINAL_MANIFEST_PATH,
            study.FINAL_PREDICTIONS_PATH,
            study.FINAL_REPORT_PATH,
            study.FINAL_ERRORS_PATH,
            study.DEVELOPMENT_DATASET_PATH,
            study.DEVELOPMENT_MANIFEST_PATH,
            study.DEVELOPMENT_REPORT_PATH,
        )
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest("Formal frozen Final Blind artifacts are unavailable.")

    def test_real_freeze_manifest_verifies_without_touching_artifacts(self) -> None:
        self._require_real_artifacts()
        paths = (
            study.FINAL_DATASET_PATH,
            study.FINAL_MANIFEST_PATH,
            study.FINAL_PREDICTIONS_PATH,
            study.FINAL_REPORT_PATH,
            study.FINAL_ERRORS_PATH,
        )
        before_hashes = {path: study.file_sha256(path) for path in paths}
        before_mtimes = {path: path.stat().st_mtime_ns for path in paths}
        verified = study.verify_final_blind_freeze()
        after_hashes = {path: study.file_sha256(path) for path in paths}
        after_mtimes = {path: path.stat().st_mtime_ns for path in paths}
        self.assertEqual("step8.10", verified["freeze_version"])
        self.assertEqual(before_hashes, after_hashes)
        self.assertEqual(before_mtimes, after_mtimes)

    def test_real_offline_metrics_reproduce_frozen_aspect_and_repeat_counts(self) -> None:
        self._require_real_artifacts()
        report = json.loads(study.FINAL_REPORT_PATH.read_text(encoding="utf-8"))
        cases = report["cases"]
        rows, unknown = study.calculate_aspect_metrics(cases)
        self.assertEqual([], unknown)
        self.assertEqual(144, sum(row["tp"] for row in rows))
        self.assertEqual(43, sum(row["fp"] for row in rows))
        self.assertEqual(34, sum(row["fn"] for row in rows))
        self.assertEqual(134, sum(row["aspect_sentiment_tp"] for row in rows))
        self.assertEqual(53, sum(row["aspect_sentiment_fp"] for row in rows))
        self.assertEqual(44, sum(row["aspect_sentiment_fn"] for row in rows))
        summary = report["summary"]
        self.assertAlmostEqual(0.7165775401069518, summary["aspect_precision"])
        self.assertAlmostEqual(0.7528089887640449, summary["aspect_recall"])
        self.assertAlmostEqual(0.7342465753424657, summary["aspect_f1"])
        self.assertAlmostEqual(0.7700534759358288, summary["aspect_only_precision"])
        self.assertAlmostEqual(0.8089887640449438, summary["aspect_only_recall"])
        self.assertAlmostEqual(0.7890410958904108, summary["aspect_only_f1"])
        self.assertAlmostEqual(1.0, summary["evidence_validity_rate"])
        self.assertAlmostEqual(0.9861111111111112, summary["evidence_match_rate"])
        self.assertAlmostEqual(0.9305555555555556, summary["sentiment_accuracy"])
        self.assertEqual((134, 53, 44), (
            summary["aspect_true_positives"],
            summary["aspect_false_positives"],
            summary["aspect_false_negatives"],
        ))
        self.assertEqual(54, summary["exact_matches"])
        self.assertAlmostEqual(0.54, summary["exact_case_accuracy"])
        repeated_rows, repeated = study.analyze_same_aspect_multi_mentions(cases)
        sentiment_rows, _ = study.classify_sentiment_errors(cases)
        self.assertEqual(22, len(repeated_rows))
        self.assertEqual(8, repeated["cases_with_repeated_component_failure"])
        self.assertEqual(10, len(sentiment_rows))

    def test_real_development_manifest_report_and_case_ids_align(self) -> None:
        self._require_real_artifacts()
        dataset = json.loads(study.DEVELOPMENT_DATASET_PATH.read_text())
        manifest = json.loads(study.DEVELOPMENT_MANIFEST_PATH.read_text())
        report = json.loads(study.DEVELOPMENT_REPORT_PATH.read_text())
        dataset_hash = study.file_sha256(study.DEVELOPMENT_DATASET_PATH)
        self.assertEqual(dataset_hash, manifest["sha256"])
        self.assertEqual(dataset_hash, report["dataset_sha256"])
        self.assertEqual(80, manifest["case_count"])
        self.assertEqual(80, report["case_count"])
        self.assertEqual(
            [case["case_id"] for case in dataset],
            [case["case_id"] for case in report["cases"]],
        )
        self.assertEqual(("ollama", "qwen3:8b", "optimized"), (
            report["provider"], report["model"], report["stage"]
        ))


if __name__ == "__main__":
    unittest.main()
