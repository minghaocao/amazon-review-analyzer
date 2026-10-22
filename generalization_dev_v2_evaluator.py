"""Frozen-baseline evaluator for Step 8.11 Generalization Dev V2.

This module provides dataset validation, leakage checks, an immutable manifest,
and a one-time qwen3:8b baseline runner.  It deliberately reuses the established
``evaluator.evaluate_case`` and ``evaluator.calculate_metrics`` definitions.
Importing it performs no model or network work.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aspect_taxonomy import STANDARD_ASPECTS
from config import LLM_PROVIDER, OLLAMA_MODEL, PROJECT_ROOT
from evaluator import calculate_metrics, evaluate_case, load_gold_cases
from final_blind_evaluator import strict_exact_case
from holdout_evaluator import ERROR_CSV_FIELDS, file_sha256, select_top_errors
from llm_client import (
    LLMClientError,
    LLMConnectionError,
    LLMResponseError,
    ReviewAnalysisClient,
    create_review_client,
)
from review_analyzer import ReviewAnalysisError, analyze_review


V2_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "generalization_dev_v2"
V2_CASES_PATH = V2_OUTPUT_DIR / "generalization_dev_v2_cases.json"
V2_MANIFEST_PATH = V2_OUTPUT_DIR / "generalization_dev_v2_manifest.json"
V2_BASELINE_PATH = V2_OUTPUT_DIR / "baseline_generalization_dev_v2_qwen3_8b.json"
V2_ERRORS_PATH = V2_OUTPUT_DIR / "baseline_generalization_dev_v2_errors.csv"
V2_PREDICTIONS_PATH = (
    V2_OUTPUT_DIR / "baseline_generalization_dev_v2_predictions.json"
)
V2_SPEC_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "generalization_error_study"
    / "generalization_dev_v2_spec.json"
)

EXPECTED_V2_CASE_COUNT = 110
EXPECTED_V2_ID_PREFIX = "V2D"
SIMILARITY_THRESHOLD = 0.75
TRANSIENT_RETRY_DELAY_SECONDS = 1.0
PRIORITY_ASPECTS = (
    "Price / Value",
    "Other",
    "Design",
    "Packaging",
    "Setup / Installation",
    "Build Quality",
)
ERROR_TYPES = (
    "WRONG_ASPECT",
    "EXTRA_ASPECT",
    "MISS_ASPECT",
    "WRONG_SENTIMENT",
    "WEAK_EVIDENCE_MATCH",
    "INVALID_EVIDENCE",
    "MODEL_ERROR",
    "SCHEMA_ERROR",
)
STEP_NOTICE = "Step 8.11 created and evaluated Generalization Dev V2 only."
OPTIMIZATION_NOTICE = "No model optimization was performed."

CORE_MODEL_FILES = (
    PROJECT_ROOT / "llm_client.py",
    PROJECT_ROOT / "aspect_taxonomy.py",
    PROJECT_ROOT / "review_analyzer.py",
    PROJECT_ROOT / "evaluator.py",
)
COMPARISON_DATASETS = (
    PROJECT_ROOT / "evaluation" / "gold_cases.json",
    PROJECT_ROOT / "evaluation" / "holdout_cases.json",
    PROJECT_ROOT / "evaluation" / "generalization_dev_cases.json",
    PROJECT_ROOT / "evaluation" / "final_blind_cases.json",
)
DEMO_REVIEW_FILES = (PROJECT_ROOT / "data" / "raw_reviews.csv",)

# Canonical tags expected in the V2 annotations.  Values are paths in the
# frozen Step 8.10 specification, and are treated as minimum overlapping
# coverage targets rather than mutually exclusive buckets.
SPEC_TAG_TARGETS: Mapping[str, tuple[str, ...]] = {
    "taxonomy_boundary": ("boundary_case_distribution", "taxonomy_boundary_cases"),
    "product_vs_experience": (
        "overlapping_language_and_structure_targets",
        "experience_vs_product",
    ),
    "same_aspect_multi_mention": (
        "overlapping_language_and_structure_targets",
        "same_aspect_multi_mention",
    ),
    "multi_aspect_sentence": (
        "overlapping_language_and_structure_targets",
        "multi_aspect_sentences",
    ),
    "negation": ("overlapping_language_and_structure_targets", "negation"),
    "contrast": ("overlapping_language_and_structure_targets", "contrast"),
    "comparison": ("overlapping_language_and_structure_targets", "comparison"),
    "temporal_change": (
        "overlapping_language_and_structure_targets",
        "temporal_change",
    ),
    "weak_sentiment": (
        "overlapping_language_and_structure_targets",
        "weak_sentiment",
    ),
    "long_review": ("overlapping_language_and_structure_targets", "long_reviews"),
    "no_aspect": ("overlapping_language_and_structure_targets", "no_aspect"),
}


class GeneralizationDevV2IntegrityError(ValueError):
    """The V2 dataset, manifest, frozen inputs, or outputs are not trustworthy."""


def _expected_ids(count: int) -> list[str]:
    return [f"{EXPECTED_V2_ID_PREFIX}{index:03d}" for index in range(1, count + 1)]


def normalize_review_text(value: str) -> str:
    """Normalize solely for duplicate/leakage checks, never for evaluation."""

    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _stable_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _mapping_fingerprint(values: Mapping[str, Any]) -> str:
    data = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def snapshot_file_hashes(paths: Sequence[str | Path]) -> dict[str, str]:
    """Return stable labels and SHA-256 values for an explicit protected set."""

    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise GeneralizationDevV2IntegrityError(
                f"Protected file was not found: {path}"
            )
        hashes[_stable_label(path)] = file_sha256(path)
    return dict(sorted(hashes.items()))


def verify_file_hashes(
    expected: Mapping[str, str], paths: Sequence[str | Path]
) -> dict[str, str]:
    """Fail closed if any protected file changed, disappeared, or was added."""

    current = snapshot_file_hashes(paths)
    if dict(expected) != current:
        raise GeneralizationDevV2IntegrityError(
            "Protected file hashes changed during Generalization Dev V2 baseline."
        )
    return current


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeneralizationDevV2IntegrityError(f"{label} was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GeneralizationDevV2IntegrityError(
            f"{label} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise GeneralizationDevV2IntegrityError(f"{label} must be a JSON object.")
    return dict(payload)


def _load_raw_cases(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeneralizationDevV2IntegrityError(
            f"Generalization Dev V2 dataset was not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GeneralizationDevV2IntegrityError(
            f"Generalization Dev V2 dataset is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise GeneralizationDevV2IntegrityError(
            "Generalization Dev V2 dataset must be a JSON list."
        )
    return payload


def validate_generalization_dev_v2_cases(
    path: str | Path = V2_CASES_PATH,
    *,
    expected_count: int = EXPECTED_V2_CASE_COUNT,
) -> list[dict[str, Any]]:
    """Validate schema, taxonomy, IDs, uniqueness, and verbatim Gold Evidence."""

    resolved = Path(path)
    raw_cases = _load_raw_cases(resolved)
    if len(raw_cases) != expected_count:
        raise GeneralizationDevV2IntegrityError(
            f"Generalization Dev V2 must contain exactly {expected_count} cases; "
            f"found {len(raw_cases)}."
        )
    try:
        validated = load_gold_cases(resolved)
    except (OSError, ValueError) as exc:
        raise GeneralizationDevV2IntegrityError(str(exc)) from exc
    if [case["case_id"] for case in validated] != _expected_ids(expected_count):
        raise GeneralizationDevV2IntegrityError(
            f"V2 case IDs must be consecutive and ordered from V2D001 through "
            f"V2D{expected_count:03d}."
        )
    normalized = [normalize_review_text(case["review"]) for case in validated]
    if any(not value for value in normalized):
        raise GeneralizationDevV2IntegrityError("V2 reviews must not be empty.")
    if len(normalized) != len(set(normalized)):
        raise GeneralizationDevV2IntegrityError(
            "V2 reviews must be unique after normalization."
        )

    enriched: list[dict[str, Any]] = []
    for raw, case in zip(raw_cases, validated, strict=True):
        if not isinstance(raw, Mapping):
            raise GeneralizationDevV2IntegrityError(
                f"{case['case_id']} must be a JSON object."
            )
        tags = raw.get("tags", [])
        notes = raw.get("notes", "")
        if not isinstance(tags, list) or not tags or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            raise GeneralizationDevV2IntegrityError(
                f"{case['case_id']}.tags must contain non-empty strings."
            )
        if not isinstance(notes, str) or not notes.strip():
            raise GeneralizationDevV2IntegrityError(
                f"{case['case_id']} must include annotation notes."
            )
        for mention in case["expected_aspects"]:
            aspect = mention["aspect"]
            if aspect not in STANDARD_ASPECTS:
                raise GeneralizationDevV2IntegrityError(
                    f"{case['case_id']} uses an unknown Aspect: {aspect!r}."
                )
            if mention["sentiment"] not in {"positive", "negative", "neutral"}:
                raise GeneralizationDevV2IntegrityError(
                    f"{case['case_id']} uses an unsupported sentiment."
                )
            evidence = mention["evidence"]
            values = evidence if isinstance(evidence, list) else [evidence]
            for value in values:
                if not isinstance(value, str) or not value or value not in case["review"]:
                    raise GeneralizationDevV2IntegrityError(
                        f"{case['case_id']} Evidence is not a case-sensitive, "
                        f"contiguous review substring: {value!r}"
                    )
        enriched.append(
            {
                **case,
                "tags": [str(tag).strip() for tag in tags],
                "notes": notes.strip(),
            }
        )
    return enriched


def dataset_distribution(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe Gold labels and overlapping V2 annotation tags."""

    aspects: Counter[str] = Counter()
    sentiments: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    cardinality: Counter[str] = Counter()
    for case in cases:
        mentions = case.get("expected_aspects", [])
        mention_list = mentions if isinstance(mentions, list) else []
        count = len(mention_list)
        bucket = (
            "no_aspect"
            if count == 0
            else "one_aspect_mention"
            if count == 1
            else "two_aspect_mentions"
            if count == 2
            else "three_to_five_aspect_mentions"
            if count <= 5
            else "six_plus_aspect_mentions"
        )
        cardinality[bucket] += 1
        for mention in mention_list:
            if isinstance(mention, Mapping):
                aspects[str(mention.get("aspect", ""))] += 1
                sentiments[str(mention.get("sentiment", ""))] += 1
        raw_tags = case.get("tags", [])
        if isinstance(raw_tags, list):
            tags.update(str(tag) for tag in raw_tags)
    return {
        "case_count": len(cases),
        "gold_mention_count": sum(aspects.values()),
        "aspect_distribution": dict(sorted(aspects.items())),
        "sentiment_distribution": dict(sorted(sentiments.items())),
        "hard_case_tag_distribution": dict(sorted(tags.items())),
        "case_cardinality_distribution": dict(sorted(cardinality.items())),
    }


def _nested_int(spec: Mapping[str, Any], path: Sequence[str]) -> int:
    value: Any = spec
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise GeneralizationDevV2IntegrityError(
                f"V2 specification is missing: {'.'.join(path)}"
            )
        value = value[key]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise GeneralizationDevV2IntegrityError(
            f"V2 specification value is not an integer: {'.'.join(path)}"
        ) from exc


def validate_distribution_against_spec(
    cases: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Enforce the frozen Step 8.10 case, mention, Aspect, and tag targets."""

    distribution = dataset_distribution(cases)
    if distribution["case_count"] != _nested_int(spec, ("target_cases",)):
        raise GeneralizationDevV2IntegrityError("V2 case count does not match the spec.")
    if distribution["gold_mention_count"] != _nested_int(spec, ("target_mentions",)):
        raise GeneralizationDevV2IntegrityError(
            "V2 Gold mention count does not match the spec."
        )
    target_aspects = spec.get("target_aspect_distribution")
    if not isinstance(target_aspects, Mapping):
        raise GeneralizationDevV2IntegrityError(
            "V2 specification has no target Aspect distribution."
        )
    normalized_targets = {str(key): int(value) for key, value in target_aspects.items()}
    unknown_targets = sorted(set(normalized_targets) - set(STANDARD_ASPECTS))
    if unknown_targets:
        raise GeneralizationDevV2IntegrityError(
            f"V2 specification contains unknown Aspects: {unknown_targets}"
        )
    if distribution["aspect_distribution"] != dict(sorted(normalized_targets.items())):
        raise GeneralizationDevV2IntegrityError(
            "V2 Aspect distribution does not match the frozen specification."
        )
    target_cardinality = spec.get("case_cardinality_distribution")
    if not isinstance(target_cardinality, Mapping):
        raise GeneralizationDevV2IntegrityError(
            "V2 specification has no case-cardinality distribution."
        )
    normalized_cardinality = {
        str(key): int(value) for key, value in target_cardinality.items()
    }
    if distribution["case_cardinality_distribution"] != dict(
        sorted(normalized_cardinality.items())
    ):
        raise GeneralizationDevV2IntegrityError(
            "V2 case-cardinality distribution does not match the specification."
        )
    tag_counts = distribution["hard_case_tag_distribution"]
    tag_targets: dict[str, int] = {}
    for tag, path in SPEC_TAG_TARGETS.items():
        target = _nested_int(spec, path)
        tag_targets[tag] = target
        if int(tag_counts.get(tag, 0)) < target:
            raise GeneralizationDevV2IntegrityError(
                f"V2 tag {tag!r} requires at least {target} cases; "
                f"found {int(tag_counts.get(tag, 0))}."
            )
    # The production specification names all taxonomy labels.  Keeping this
    # conditional makes the validator reusable with small offline fixtures
    # without weakening the real 110-case check.
    missing_taxonomy = sorted(
        aspect
        for aspect in STANDARD_ASPECTS
        if aspect in normalized_targets
        and int(distribution["aspect_distribution"].get(aspect, 0)) <= 0
    )
    if set(normalized_targets) == set(STANDARD_ASPECTS) and missing_taxonomy:
        raise GeneralizationDevV2IntegrityError(
            f"V2 dataset does not cover every taxonomy Aspect: {missing_taxonomy}"
        )
    missing_priority = [
        aspect
        for aspect in PRIORITY_ASPECTS
        if aspect in normalized_targets
        and int(distribution["aspect_distribution"].get(aspect, 0)) <= 0
    ]
    if missing_priority:
        raise GeneralizationDevV2IntegrityError(
            f"V2 dataset lacks Priority Aspect support: {missing_priority}"
        )
    return {**distribution, "minimum_tag_targets": tag_targets}


def _source_records(
    dataset_paths: Sequence[str | Path],
    demo_paths: Sequence[str | Path],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    records: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    for raw_path in dataset_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise GeneralizationDevV2IntegrityError(
                f"Leakage comparison dataset was not found: {path}"
            )
        hashes[_stable_label(path)] = file_sha256(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GeneralizationDevV2IntegrityError(
                f"Leakage comparison dataset is invalid JSON: {path}: {exc}"
            ) from exc
        if not isinstance(payload, list):
            raise GeneralizationDevV2IntegrityError(
                f"Leakage comparison dataset must be a list: {path}"
            )
        for position, item in enumerate(payload, start=1):
            if not isinstance(item, Mapping):
                continue
            review = item.get("review", item.get("review_text", ""))
            if isinstance(review, str) and review.strip():
                records.append(
                    {
                        "source": path.name,
                        "source_id": str(item.get("case_id", position)),
                        "review": review.strip(),
                    }
                )
    for raw_path in demo_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise GeneralizationDevV2IntegrityError(
                f"Demo leakage comparison file was not found: {path}"
            )
        hashes[_stable_label(path)] = file_sha256(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for position, row in enumerate(csv.DictReader(handle), start=1):
                review = row.get("content") or row.get("original_content") or ""
                if review.strip():
                    records.append(
                        {
                            "source": path.name,
                            "source_id": str(row.get("review_id") or position),
                            "review": review.strip(),
                        }
                    )
    return records, dict(sorted(hashes.items()))


def check_generalization_dev_v2_leakage(
    dataset_path: str | Path = V2_CASES_PATH,
    *,
    expected_count: int = EXPECTED_V2_CASE_COUNT,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Check exact, normalized, and pairwise similarity against all old data."""

    cases = validate_generalization_dev_v2_cases(
        dataset_path, expected_count=expected_count
    )
    sources, source_hashes = _source_records(comparison_dataset_paths, demo_paths)
    exact: list[dict[str, str]] = []
    normalized: list[dict[str, str]] = []
    maximum_similarity = 0.0
    closest: dict[str, Any] | None = None
    for case in cases:
        review = case["review"].strip()
        normalized_review = normalize_review_text(review)
        for source in sources:
            old_review = source["review"].strip()
            old_normalized = normalize_review_text(old_review)
            identity = {
                "v2_case_id": case["case_id"],
                "source": source["source"],
                "source_id": source["source_id"],
            }
            if review == old_review:
                exact.append(identity)
            if normalized_review == old_normalized:
                normalized.append(identity)
            similarity = difflib.SequenceMatcher(
                None, normalized_review, old_normalized
            ).ratio()
            if similarity > maximum_similarity:
                maximum_similarity = similarity
                closest = {**identity, "similarity": similarity}
    passed = (
        not exact
        and not normalized
        and maximum_similarity < float(similarity_threshold)
    )
    return {
        "exact_duplicate_count": len(exact),
        "normalized_duplicate_count": len(normalized),
        "exact_duplicates": exact,
        "normalized_duplicates": normalized,
        "maximum_similarity": maximum_similarity,
        "similarity_threshold": float(similarity_threshold),
        "closest_source": closest,
        "source_review_count": len(sources),
        "source_file_sha256": source_hashes,
        "source_corpus_fingerprint": _mapping_fingerprint(source_hashes),
        "passed": passed,
    }


def create_generalization_dev_v2_manifest(
    dataset_path: str | Path = V2_CASES_PATH,
    manifest_path: str | Path = V2_MANIFEST_PATH,
    *,
    spec_path: str | Path = V2_SPEC_PATH,
    expected_count: int = EXPECTED_V2_CASE_COUNT,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
    core_model_paths: Sequence[str | Path] = CORE_MODEL_FILES,
) -> dict[str, Any]:
    """Freeze V2, its source corpus, and model logic before first inference."""

    dataset = Path(dataset_path)
    cases = validate_generalization_dev_v2_cases(dataset, expected_count=expected_count)
    spec_file = Path(spec_path)
    spec = _load_json_mapping(spec_file, label="Generalization Dev V2 specification")
    distribution = validate_distribution_against_spec(cases, spec)
    leakage = check_generalization_dev_v2_leakage(
        dataset,
        expected_count=expected_count,
        comparison_dataset_paths=comparison_dataset_paths,
        demo_paths=demo_paths,
    )
    if not leakage["passed"]:
        raise GeneralizationDevV2IntegrityError(
            "Generalization Dev V2 leakage check failed; overlapping cases must "
            "be replaced before creating the manifest."
        )
    destination = Path(manifest_path)
    if destination.exists():
        raise GeneralizationDevV2IntegrityError(
            "Generalization Dev V2 manifest already exists and is immutable."
        )
    core_hashes = snapshot_file_hashes(core_model_paths)
    manifest = {
        "dataset_file": dataset.name,
        "dataset": dataset.name,
        "sha256": file_sha256(dataset),
        "case_count": distribution["case_count"],
        "gold_mention_count": distribution["gold_mention_count"],
        "sentiment_distribution": distribution["sentiment_distribution"],
        "aspect_distribution": distribution["aspect_distribution"],
        "hard_case_tag_distribution": distribution["hard_case_tag_distribution"],
        "case_cardinality_distribution": distribution[
            "case_cardinality_distribution"
        ],
        "created_before_evaluation": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification_file": spec_file.name,
        "specification_sha256": file_sha256(spec_file),
        "core_model_file_sha256": core_hashes,
        "core_model_fingerprint": _mapping_fingerprint(core_hashes),
        "leakage_check": leakage,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_generalization_dev_v2_manifest(
    dataset_path: str | Path = V2_CASES_PATH,
    manifest_path: str | Path = V2_MANIFEST_PATH,
    *,
    spec_path: str | Path = V2_SPEC_PATH,
    expected_count: int = EXPECTED_V2_CASE_COUNT,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
    core_model_paths: Sequence[str | Path] = CORE_MODEL_FILES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify every frozen input before a client may be constructed."""

    dataset = Path(dataset_path)
    manifest = _load_json_mapping(Path(manifest_path), label="V2 manifest")
    if manifest.get("dataset_file", manifest.get("dataset")) != dataset.name:
        raise GeneralizationDevV2IntegrityError("V2 manifest dataset name changed.")
    if manifest.get("case_count") != expected_count:
        raise GeneralizationDevV2IntegrityError("V2 manifest case count changed.")
    if manifest.get("created_before_evaluation") is not True:
        raise GeneralizationDevV2IntegrityError(
            "V2 manifest must confirm creation before evaluation."
        )
    if not isinstance(manifest.get("created_at"), str):
        raise GeneralizationDevV2IntegrityError("V2 manifest has no timestamp.")
    if manifest.get("sha256") != file_sha256(dataset):
        raise GeneralizationDevV2IntegrityError(
            "Generalization Dev V2 dataset changed after it was frozen."
        )
    spec_file = Path(spec_path)
    if manifest.get("specification_sha256") != file_sha256(spec_file):
        raise GeneralizationDevV2IntegrityError("Step 8.10 V2 specification changed.")
    cases = validate_generalization_dev_v2_cases(dataset, expected_count=expected_count)
    distribution = validate_distribution_against_spec(
        cases,
        _load_json_mapping(spec_file, label="Generalization Dev V2 specification"),
    )
    for key in (
        "gold_mention_count",
        "sentiment_distribution",
        "aspect_distribution",
        "hard_case_tag_distribution",
        "case_cardinality_distribution",
    ):
        if manifest.get(key) != distribution[key]:
            raise GeneralizationDevV2IntegrityError(
                f"V2 manifest {key} no longer matches the dataset."
            )
    verify_file_hashes(manifest.get("core_model_file_sha256", {}), core_model_paths)
    leakage = manifest.get("leakage_check")
    if not isinstance(leakage, Mapping) or leakage.get("passed") is not True:
        raise GeneralizationDevV2IntegrityError(
            "V2 manifest does not contain a passing leakage check."
        )
    _, source_hashes = _source_records(comparison_dataset_paths, demo_paths)
    if leakage.get("source_corpus_fingerprint") != _mapping_fingerprint(source_hashes):
        raise GeneralizationDevV2IntegrityError(
            "A leakage comparison dataset changed after V2 was frozen."
        )
    return manifest, cases


def _rates(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def calculate_priority_aspect_metrics(
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aspect+sentiment multiset metrics for the six Step 8.11 priorities."""

    totals = {
        aspect: {"support": 0, "tp": 0, "fp": 0, "fn": 0}
        for aspect in PRIORITY_ASPECTS
    }
    for case in case_results:
        expected_raw = case.get("expected_aspects", [])
        actual_raw = case.get("evaluation_aspects", case.get("actual_aspects", []))
        expected = expected_raw if isinstance(expected_raw, list) else []
        actual = actual_raw if isinstance(actual_raw, list) else []
        expected_pairs = Counter(
            (str(item.get("aspect", "")), str(item.get("sentiment", "")))
            for item in expected
            if isinstance(item, Mapping)
        )
        actual_pairs = Counter(
            (str(item.get("aspect", "")), str(item.get("sentiment", "")))
            for item in actual
            if isinstance(item, Mapping)
        )
        for aspect in PRIORITY_ASPECTS:
            gold_count = sum(
                count for (label, _), count in expected_pairs.items() if label == aspect
            )
            predicted_count = sum(
                count for (label, _), count in actual_pairs.items() if label == aspect
            )
            tp = sum(
                min(count, actual_pairs[(label, sentiment)])
                for (label, sentiment), count in expected_pairs.items()
                if label == aspect
            )
            totals[aspect]["support"] += gold_count
            totals[aspect]["tp"] += tp
            totals[aspect]["fp"] += predicted_count - tp
            totals[aspect]["fn"] += gold_count - tp
    return {
        aspect: {**values, **_rates(values["tp"], values["fp"], values["fn"])}
        for aspect, values in totals.items()
    }


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def run_generalization_dev_v2_baseline(
    *,
    dataset_path: str | Path = V2_CASES_PATH,
    manifest_path: str | Path = V2_MANIFEST_PATH,
    spec_path: str | Path = V2_SPEC_PATH,
    expected_count: int = EXPECTED_V2_CASE_COUNT,
    provider: str = LLM_PROVIDER,
    model: str = OLLAMA_MODEL,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
    core_model_paths: Sequence[str | Path] = CORE_MODEL_FILES,
    client: ReviewAnalysisClient | None = None,
    client_factory: Callable[[], ReviewAnalysisClient] | None = None,
    analyzer: Callable[..., Mapping[str, Any]] = analyze_review,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    retry_delay_seconds: float = TRANSIENT_RETRY_DELAY_SECONDS,
    progress: Callable[[int, int, str, str, float], None] | None = None,
) -> dict[str, Any]:
    """Run one untouched qwen3:8b baseline with bounded connection retry."""

    manifest, cases = verify_generalization_dev_v2_manifest(
        dataset_path,
        manifest_path,
        spec_path=spec_path,
        expected_count=expected_count,
        comparison_dataset_paths=comparison_dataset_paths,
        demo_paths=demo_paths,
        core_model_paths=core_model_paths,
    )
    if provider != "ollama" or model != "qwen3:8b":
        raise GeneralizationDevV2IntegrityError(
            "Generalization Dev V2 baseline is locked to Ollama qwen3:8b."
        )
    if client is not None and client_factory is not None:
        raise ValueError("Provide client or client_factory, not both.")
    resolved_client = client or (
        client_factory() if client_factory is not None else create_review_client()
    )
    core_before = snapshot_file_hashes(core_model_paths)
    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    retry_count = 0
    for position, case in enumerate(cases, start=1):
        started = clock()
        actual: Mapping[str, Any] | None = None
        model_error: str | None = None
        schema_error: str | None = None
        attempts = 0
        while attempts < 2:
            attempts += 1
            try:
                actual = analyzer(case["review"], llm_client=resolved_client)
                break
            except LLMConnectionError as exc:
                if attempts == 1:
                    retry_count += 1
                    sleeper(max(0.0, retry_delay_seconds))
                    continue
                model_error = str(exc)
                break
            except (LLMResponseError, ReviewAnalysisError) as exc:
                schema_error = str(exc)
                break
            except LLMClientError as exc:
                model_error = str(exc)
                break
            except Exception as exc:  # keep one bad case from aborting the baseline
                model_error = f"{type(exc).__name__}: {exc}"
                break
        elapsed = max(0.0, float(clock() - started))
        result = evaluate_case(
            case,
            actual,
            analysis_time_seconds=elapsed,
            model_error=model_error,
            schema_error=schema_error,
        )
        result["base_exact_match"] = bool(result["exact_match"])
        result["exact_match"] = strict_exact_case(result)
        result["notes"] = case["notes"]
        result["tags"] = case["tags"]
        result["attempts"] = attempts
        result["retried"] = attempts > 1
        observed_types = {
            str(error.get("error_type"))
            for error in result.get("errors", [])
            if isinstance(error, Mapping)
        }
        fatal_type = (
            "SCHEMA_ERROR"
            if "SCHEMA_ERROR" in observed_types
            else "MODEL_ERROR"
            if "MODEL_ERROR" in observed_types
            else ""
        )
        status = "failed" if fatal_type else "success"
        predictions.append(
            {
                "case_id": case["case_id"],
                "review": case["review"],
                "status": status,
                "attempts": attempts,
                "retried": attempts > 1,
                "analysis_time_seconds": elapsed,
                "analysis": dict(actual) if isinstance(actual, Mapping) else None,
                "error_type": fatal_type,
                "error_message": schema_error or model_error or "",
            }
        )
        results.append(result)
        if progress is None:
            print(
                f"[{position}/{len(cases)}] {case['case_id']} {status} | "
                f"{elapsed:.2f}s"
            )
        else:
            progress(position, len(cases), case["case_id"], status, elapsed)

    core_after = verify_file_hashes(core_before, core_model_paths)
    # Also re-check the dataset, old corpora, specification, and manifest after
    # the final analyzer call.  No result is returned if a frozen input changed.
    verify_generalization_dev_v2_manifest(
        dataset_path,
        manifest_path,
        spec_path=spec_path,
        expected_count=expected_count,
        comparison_dataset_paths=comparison_dataset_paths,
        demo_paths=demo_paths,
        core_model_paths=core_model_paths,
    )
    summary = calculate_metrics(results, model=model)
    runtimes = [float(item["analysis_time_seconds"]) for item in predictions]
    observed_errors = Counter(
        str(error.get("error_type"))
        for result in results
        for error in result.get("errors", [])
        if isinstance(error, Mapping) and error.get("error_type")
    )
    error_counts = {
        error_type: int(observed_errors.get(error_type, 0))
        for error_type in ERROR_TYPES
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    summary["timestamp"] = timestamp
    return {
        "evaluation_type": "generalization_dev_v2_baseline",
        "evaluation_policy": f"{STEP_NOTICE} {OPTIMIZATION_NOTICE}",
        "stage": "baseline",
        "provider": provider,
        "model": model,
        "dataset": Path(dataset_path).name,
        "dataset_sha256": manifest["sha256"],
        "manifest_created_at": manifest["created_at"],
        "timestamp": timestamp,
        "case_count": len(cases),
        "dataset_distribution": dataset_distribution(cases),
        "leakage_check": manifest["leakage_check"],
        "core_model_file_sha256_before": core_before,
        "core_model_file_sha256_after": core_after,
        "core_model_hashes_unchanged": core_before == core_after,
        "summary": summary,
        "runtime_statistics": {
            "total_seconds": sum(runtimes),
            "average_seconds_per_review": sum(runtimes) / len(runtimes)
            if runtimes
            else 0.0,
            "median_seconds_per_review": statistics.median(runtimes)
            if runtimes
            else 0.0,
            "p90_seconds_per_review": _percentile_nearest_rank(runtimes, 0.90),
            "p90_method": "nearest_rank",
            "cold_start_first_case_seconds": runtimes[0] if runtimes else 0.0,
            "retry_count": retry_count,
            "successful_cases": int(summary.get("successful_cases", 0)),
            "failed_cases": int(summary.get("failed_cases", 0)),
        },
        "confusion_matrix": summary["confusion_matrix"],
        "error_counts": error_counts,
        "priority_aspect_metrics": calculate_priority_aspect_metrics(results),
        "top_errors": select_top_errors(results, limit=10),
        "cases": results,
        "predictions": predictions,
    }


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _error_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = report.get("cases", [])
    if not isinstance(cases, list):
        return rows
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        expected = case.get("expected_aspects", [])
        actual = case.get("actual_aspects", [])
        expected_list = expected if isinstance(expected, list) else []
        actual_list = actual if isinstance(actual, list) else []
        errors = case.get("errors", [])
        if not isinstance(errors, list):
            continue
        for error in errors:
            if not isinstance(error, Mapping):
                continue
            rows.append(
                {
                    "case_id": case.get("case_id", ""),
                    "review": case.get("review", ""),
                    "error_type": error.get("error_type", ""),
                    "expected_aspects": _json_cell(
                        [item.get("aspect", "") for item in expected_list]
                    ),
                    "actual_aspects": _json_cell(
                        [item.get("aspect", "") for item in actual_list]
                    ),
                    "expected_sentiments": _json_cell(
                        [item.get("sentiment", "") for item in expected_list]
                    ),
                    "actual_sentiments": _json_cell(
                        [item.get("sentiment", "") for item in actual_list]
                    ),
                    "expected_evidence": _json_cell(
                        [item.get("evidence", "") for item in expected_list]
                    ),
                    "actual_evidence": _json_cell(
                        [item.get("evidence", "") for item in actual_list]
                    ),
                    "notes": case.get("notes", ""),
                    "message": error.get("message", ""),
                }
            )
    return rows


def baseline_outputs_exist(
    paths: Sequence[str | Path] = (
        V2_BASELINE_PATH,
        V2_ERRORS_PATH,
        V2_PREDICTIONS_PATH,
    ),
) -> bool:
    return any(Path(path).exists() for path in paths)


def save_generalization_dev_v2_outputs(
    report: Mapping[str, Any],
    *,
    baseline_path: str | Path = V2_BASELINE_PATH,
    errors_path: str | Path = V2_ERRORS_PATH,
    predictions_path: str | Path = V2_PREDICTIONS_PATH,
) -> dict[str, str]:
    """Persist one immutable baseline; no optimized-stage output is supported."""

    destinations = (Path(baseline_path), Path(errors_path), Path(predictions_path))
    if any(path.exists() for path in destinations):
        raise GeneralizationDevV2IntegrityError(
            "Generalization Dev V2 baseline outputs already exist and will not "
            "be overwritten."
        )
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "baseline": str(destinations[0]),
        "errors": str(destinations[1]),
        "predictions": str(destinations[2]),
    }
    payload = dict(report)
    payload["outputs"] = output_paths
    destinations[0].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with destinations[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_error_rows(payload))
    predictions_payload = {
        "evaluation_policy": payload.get("evaluation_policy", ""),
        "stage": "baseline",
        "provider": payload.get("provider", ""),
        "model": payload.get("model", ""),
        "dataset_sha256": payload.get("dataset_sha256", ""),
        "timestamp": payload.get("timestamp", ""),
        "case_count": payload.get("case_count", 0),
        "predictions": payload.get("predictions", []),
    }
    destinations[2].write_text(
        json.dumps(predictions_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output_paths


def format_generalization_dev_v2_report(report: Mapping[str, Any]) -> str:
    """Human-readable CLI summary including all Priority Aspect baselines."""

    summary = report.get("summary", {})
    runtime = report.get("runtime_statistics", {})
    priority = report.get("priority_aspect_metrics", {})
    lines = [
        "Generalization Dev V2 — untouched baseline",
        STEP_NOTICE,
        OPTIMIZATION_NOTICE,
        f"Cases: {int(summary.get('total_cases', 0))}",
        f"Successful: {int(summary.get('successful_cases', 0))}",
        f"Failed: {int(summary.get('failed_cases', 0))}",
        f"Aspect Precision: {float(summary.get('aspect_precision', 0.0)):.2%}",
        f"Aspect Recall: {float(summary.get('aspect_recall', 0.0)):.2%}",
        f"Aspect F1: {float(summary.get('aspect_f1', 0.0)):.2%}",
        f"Aspect-only Precision: {float(summary.get('aspect_only_precision', 0.0)):.2%}",
        f"Aspect-only Recall: {float(summary.get('aspect_only_recall', 0.0)):.2%}",
        f"Aspect-only F1: {float(summary.get('aspect_only_f1', 0.0)):.2%}",
        f"Sentiment Accuracy: {float(summary.get('sentiment_accuracy', 0.0)):.2%}",
        f"Evidence Validity: {float(summary.get('evidence_validity_rate', 0.0)):.2%}",
        f"Evidence Match: {float(summary.get('evidence_match_rate', 0.0)):.2%}",
        (
            f"Strict Exact: {int(summary.get('exact_matches', 0))}/"
            f"{int(summary.get('total_cases', 0))} "
            f"({float(summary.get('exact_case_accuracy', 0.0)):.2%})"
        ),
        (
            "TP / FP / FN: "
            f"{int(summary.get('aspect_true_positives', 0))} / "
            f"{int(summary.get('aspect_false_positives', 0))} / "
            f"{int(summary.get('aspect_false_negatives', 0))}"
        ),
        f"Error counts: {json.dumps(report.get('error_counts', {}), sort_keys=True)}",
        "Priority Aspects (Aspect + Sentiment):",
    ]
    for aspect in PRIORITY_ASPECTS:
        metrics = priority.get(aspect, {}) if isinstance(priority, Mapping) else {}
        lines.append(
            f"  {aspect}: support={int(metrics.get('support', 0))}, "
            f"P={float(metrics.get('precision', 0.0)):.2%}, "
            f"R={float(metrics.get('recall', 0.0)):.2%}, "
            f"F1={float(metrics.get('f1', 0.0)):.2%}"
        )
    lines.extend(
        [
            f"Total runtime: {float(runtime.get('total_seconds', 0.0)):.2f}s",
            f"Average runtime: {float(runtime.get('average_seconds_per_review', 0.0)):.2f}s/review",
            f"Median runtime: {float(runtime.get('median_seconds_per_review', 0.0)):.2f}s",
            f"P90 runtime: {float(runtime.get('p90_seconds_per_review', 0.0)):.2f}s",
            f"Cold-start first case: {float(runtime.get('cold_start_first_case_seconds', 0.0)):.2f}s",
            f"Retries: {int(runtime.get('retry_count', 0))}",
            (
                "Core model hashes unchanged: "
                f"{bool(report.get('core_model_hashes_unchanged', False))}"
            ),
        ]
    )
    return "\n".join(lines)


__all__ = [
    "CORE_MODEL_FILES",
    "COMPARISON_DATASETS",
    "DEMO_REVIEW_FILES",
    "ERROR_TYPES",
    "EXPECTED_V2_CASE_COUNT",
    "GeneralizationDevV2IntegrityError",
    "PRIORITY_ASPECTS",
    "SIMILARITY_THRESHOLD",
    "V2_BASELINE_PATH",
    "V2_CASES_PATH",
    "V2_ERRORS_PATH",
    "V2_MANIFEST_PATH",
    "V2_OUTPUT_DIR",
    "V2_PREDICTIONS_PATH",
    "V2_SPEC_PATH",
    "baseline_outputs_exist",
    "calculate_priority_aspect_metrics",
    "check_generalization_dev_v2_leakage",
    "create_generalization_dev_v2_manifest",
    "dataset_distribution",
    "format_generalization_dev_v2_report",
    "normalize_review_text",
    "run_generalization_dev_v2_baseline",
    "save_generalization_dev_v2_outputs",
    "snapshot_file_hashes",
    "validate_distribution_against_spec",
    "validate_generalization_dev_v2_cases",
    "verify_file_hashes",
    "verify_generalization_dev_v2_manifest",
]
