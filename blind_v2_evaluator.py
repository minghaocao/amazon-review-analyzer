"""One-time, SHA-locked evaluator for Step 8.13 Blind Test V2.

This module is intentionally evaluation-only.  It freezes a new 150-case
dataset, every historical comparison source, selected non-model project assets,
and the four production analysis files before the first provider call.  It
reuses the established evaluator metrics without changing model behaviour.
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
from config import (
    DEMO_OUTPUT_PATHS,
    LLM_PROVIDER,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    PROJECT_ROOT,
)
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


BLIND_V2_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "blind_v2"
BLIND_V2_CASES_PATH = BLIND_V2_OUTPUT_DIR / "blind_v2_cases.json"
BLIND_V2_MANIFEST_PATH = BLIND_V2_OUTPUT_DIR / "blind_v2_manifest.json"
BLIND_V2_PREDICTIONS_PATH = BLIND_V2_OUTPUT_DIR / "blind_v2_predictions.json"
BLIND_V2_REPORT_PATH = BLIND_V2_OUTPUT_DIR / "blind_v2_evaluation.json"
BLIND_V2_ERRORS_PATH = BLIND_V2_OUTPUT_DIR / "blind_v2_errors.csv"
BLIND_V2_FREEZE_PATH = BLIND_V2_OUTPUT_DIR / "blind_v2_freeze_manifest.json"
BLIND_V2_STARTED_PATH = BLIND_V2_OUTPUT_DIR / "blind_v2_evaluation_started.json"

EXPECTED_CASE_COUNT = 150
CASE_ID_PREFIX = "B2-"
SIMILARITY_THRESHOLD = 0.75
TRANSIENT_RETRY_DELAY_SECONDS = 1.0
TAXONOMY_VERSION = "generalization_v2_candidate"
EVALUATION_NOTICE = (
    "Blind V2 is evaluation-only. No model optimization was performed after "
    "seeing Blind V2 results."
)
RUN_ONCE_NOTICE = (
    "Blind V2 was run once against the frozen generalization_v2_candidate."
)

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
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "generalization_dev_v2"
    / "generalization_dev_v2_cases.json",
)
DEMO_REVIEW_FILES = (PROJECT_ROOT / "data" / "raw_reviews.csv",)

COMPARISON_REPORTS = {
    "step8.9_final_blind_v1": (
        PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / "final_blind"
        / "final_blind_evaluation.json"
    ),
    "step8.11_dev_v2_baseline": (
        PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / "generalization_dev_v2"
        / "baseline_generalization_dev_v2_qwen3_8b.json"
    ),
    "step8.12.1_dev_v2_candidate": (
        PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / "generalization_dev_v2"
        / "repair"
        / "repair_candidate_evaluation.json"
    ),
}

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

GAP_METRICS = (
    "aspect_precision",
    "aspect_recall",
    "aspect_f1",
    "aspect_only_f1",
    "sentiment_accuracy",
    "evidence_match_rate",
    "exact_case_accuracy",
)


class BlindV2IntegrityError(ValueError):
    """Blind V2 cannot proceed without a complete immutable audit trail."""


def _expected_ids(count: int) -> list[str]:
    return [f"{CASE_ID_PREFIX}{index:03d}" for index in range(1, count + 1)]


def normalize_review_text(value: str) -> str:
    """Normalize only for leakage comparison, never for model evaluation."""

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
    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise BlindV2IntegrityError(f"Protected file is unavailable: {path}")
        hashes[_stable_label(path)] = file_sha256(path)
    return dict(sorted(hashes.items()))


def verify_file_hashes(
    expected: Mapping[str, str], paths: Sequence[str | Path]
) -> dict[str, str]:
    current = snapshot_file_hashes(paths)
    if dict(expected) != current:
        raise BlindV2IntegrityError("A protected file changed during Blind V2.")
    return current


def protected_non_model_paths() -> tuple[Path, ...]:
    """Resolve existing Demo, UI/import, and historical evaluation assets."""

    paths: set[Path] = {Path(path) for path in DEMO_OUTPUT_PATHS}
    paths.add(PROJECT_ROOT / "data" / "raw_reviews.csv")
    paths.update(
        PROJECT_ROOT / name
        for name in (
            "app.py",
            "visualization.py",
            "interactive_analysis.py",
            "input_loader.py",
            "amazon_analysis.py",
            "amazon_url.py",
            "amazon_access.py",
            "browser_amazon_import.py",
            "scraper.py",
            "aggregator.py",
            "keyword_analyzer.py",
            "batch_analyzer.py",
            "cleaner.py",
        )
    )
    for folder in (
        PROJECT_ROOT / "evaluation",
        PROJECT_ROOT / "outputs" / "evaluation" / "holdout",
        PROJECT_ROOT / "outputs" / "evaluation" / "final_blind",
        PROJECT_ROOT / "outputs" / "evaluation" / "generalization_error_study",
        PROJECT_ROOT / "outputs" / "evaluation" / "generalization_dev_v2",
        PROJECT_ROOT / "outputs" / "amazon" / "current",
    ):
        if folder.is_dir():
            paths.update(
                path
                for path in folder.rglob("*")
                if path.is_file() and BLIND_V2_OUTPUT_DIR not in path.parents
            )
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise BlindV2IntegrityError(
            "A protected historical/project file is unavailable: "
            + ", ".join(missing)
        )
    return tuple(sorted(paths, key=str))


def _raw_cases(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BlindV2IntegrityError(f"Blind V2 dataset was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BlindV2IntegrityError(f"Blind V2 dataset is invalid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise BlindV2IntegrityError("Blind V2 dataset must be a JSON list.")
    return payload


def validate_blind_v2_cases(
    path: str | Path = BLIND_V2_CASES_PATH,
    *,
    expected_count: int = EXPECTED_CASE_COUNT,
) -> list[dict[str, Any]]:
    """Validate schema, taxonomy, IDs, metadata, and exact Gold evidence."""

    resolved = Path(path)
    raw = _raw_cases(resolved)
    if len(raw) != expected_count:
        raise BlindV2IntegrityError(
            f"Blind V2 must contain exactly {expected_count} cases; found {len(raw)}."
        )
    try:
        cases = load_gold_cases(resolved)
    except (OSError, ValueError) as exc:
        raise BlindV2IntegrityError(str(exc)) from exc
    if [case["case_id"] for case in cases] != _expected_ids(expected_count):
        raise BlindV2IntegrityError(
            f"Blind V2 IDs must be ordered {CASE_ID_PREFIX}001 through "
            f"{CASE_ID_PREFIX}{expected_count:03d}."
        )
    normalized = [normalize_review_text(case["review"]) for case in cases]
    if any(not item for item in normalized):
        raise BlindV2IntegrityError("Blind V2 reviews must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise BlindV2IntegrityError("Blind V2 reviews are not unique after normalization.")

    enriched: list[dict[str, Any]] = []
    for raw_case, case in zip(raw, cases, strict=True):
        if not isinstance(raw_case, Mapping):
            raise BlindV2IntegrityError(f"{case['case_id']} must be an object.")
        tags = raw_case.get("tags", [])
        notes = raw_case.get("notes", "")
        if not isinstance(tags, list) or not tags or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            raise BlindV2IntegrityError(
                f"{case['case_id']}.tags must contain non-empty strings."
            )
        if not isinstance(notes, str) or not notes.strip():
            raise BlindV2IntegrityError(f"{case['case_id']} must include notes.")
        mentions = case["expected_aspects"]
        if len(mentions) > 5:
            raise BlindV2IntegrityError(
                f"{case['case_id']} exceeds the five-mention Blind V2 limit."
            )
        for mention in mentions:
            if mention["aspect"] not in STANDARD_ASPECTS:
                raise BlindV2IntegrityError(
                    f"{case['case_id']} has unknown Aspect {mention['aspect']!r}."
                )
            if mention["sentiment"] not in {"positive", "negative", "neutral"}:
                raise BlindV2IntegrityError(
                    f"{case['case_id']} has unsupported sentiment."
                )
            evidence = mention["evidence"]
            values = evidence if isinstance(evidence, list) else [evidence]
            for value in values:
                if not isinstance(value, str) or not value or value not in case["review"]:
                    raise BlindV2IntegrityError(
                        f"{case['case_id']} Evidence is not an exact continuous "
                        f"substring: {value!r}"
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
    aspects: Counter[str] = Counter()
    sentiments: Counter[str] = Counter()
    cardinality: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    word_lengths: list[int] = []
    character_lengths: list[int] = []
    for case in cases:
        review = str(case.get("review", ""))
        word_lengths.append(len(review.split()))
        character_lengths.append(len(review))
        mentions = case.get("expected_aspects", [])
        mention_list = mentions if isinstance(mentions, list) else []
        count = len(mention_list)
        cardinality[
            "no_aspect"
            if count == 0
            else "one_mention"
            if count == 1
            else "two_mentions"
            if count == 2
            else "three_to_five_mentions"
            if count <= 5
            else "six_plus_mentions"
        ] += 1
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
        "case_cardinality_distribution": dict(sorted(cardinality.items())),
        "hard_case_distribution": dict(sorted(tags.items())),
        "average_review_words": statistics.mean(word_lengths) if word_lengths else 0.0,
        "median_review_words": statistics.median(word_lengths) if word_lengths else 0.0,
        "average_review_characters": (
            statistics.mean(character_lengths) if character_lengths else 0.0
        ),
        "median_review_characters": (
            statistics.median(character_lengths) if character_lengths else 0.0
        ),
    }


def validate_blind_v2_distribution(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Enforce the declared 15/40/60/35 design and requested sentiment range."""

    distribution = dataset_distribution(cases)
    expected_cardinality = {
        "no_aspect": 15,
        "one_mention": 40,
        "two_mentions": 60,
        "three_to_five_mentions": 35,
    }
    if distribution["case_cardinality_distribution"] != expected_cardinality:
        raise BlindV2IntegrityError(
            "Blind V2 cardinality must be exactly 15/40/60/35."
        )
    mentions = int(distribution["gold_mention_count"])
    if not mentions:
        raise BlindV2IntegrityError("Blind V2 must contain Gold mentions.")
    sentiments = distribution["sentiment_distribution"]
    ratios = {
        label: int(sentiments.get(label, 0)) / mentions
        for label in ("positive", "negative", "neutral")
    }
    if not (0.40 <= ratios["positive"] <= 0.45):
        raise BlindV2IntegrityError("Positive sentiment must be 40–45% of mentions.")
    if not (0.40 <= ratios["negative"] <= 0.45):
        raise BlindV2IntegrityError("Negative sentiment must be 40–45% of mentions.")
    if not (0.10 <= ratios["neutral"] <= 0.15):
        raise BlindV2IntegrityError("Neutral sentiment must be 10–15% of mentions.")
    aspect_counts = distribution["aspect_distribution"]
    missing = [aspect for aspect in STANDARD_ASPECTS if int(aspect_counts.get(aspect, 0)) < 8]
    if missing:
        raise BlindV2IntegrityError(
            "Blind V2 has insufficient taxonomy support for: " + ", ".join(missing)
        )
    distribution["sentiment_ratios"] = ratios
    return distribution


def _source_records(
    dataset_paths: Sequence[str | Path],
    demo_paths: Sequence[str | Path],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    records: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    for raw_path in dataset_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise BlindV2IntegrityError(f"Leakage source is unavailable: {path}")
        hashes[_stable_label(path)] = file_sha256(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BlindV2IntegrityError(f"Leakage source is invalid JSON: {path}") from exc
        if not isinstance(payload, list):
            raise BlindV2IntegrityError(f"Leakage source must be a list: {path}")
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
            raise BlindV2IntegrityError(f"Demo leakage source is unavailable: {path}")
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


def check_blind_v2_leakage(
    dataset_path: str | Path = BLIND_V2_CASES_PATH,
    *,
    expected_count: int = EXPECTED_CASE_COUNT,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    cases = validate_blind_v2_cases(dataset_path, expected_count=expected_count)
    sources, source_hashes = _source_records(comparison_dataset_paths, demo_paths)
    exact: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    maximum_similarity = 0.0
    closest: dict[str, Any] | None = None
    threshold_violations: list[dict[str, Any]] = []
    for case in cases:
        text = case["review"].strip()
        normalized_text = normalize_review_text(text)
        for source in sources:
            old_text = source["review"].strip()
            old_normalized = normalize_review_text(old_text)
            identity = {
                "blind_v2_case_id": case["case_id"],
                "source": source["source"],
                "source_id": source["source_id"],
            }
            if text == old_text:
                exact.append(identity)
            if normalized_text == old_normalized:
                normalized.append(identity)
            similarity = difflib.SequenceMatcher(
                None, normalized_text, old_normalized
            ).ratio()
            if similarity > maximum_similarity:
                maximum_similarity = similarity
                closest = {**identity, "similarity": similarity}
            if similarity >= similarity_threshold:
                threshold_violations.append({**identity, "similarity": similarity})
    return {
        "exact_duplicate_count": len(exact),
        "normalized_duplicate_count": len(normalized),
        "similarity_violation_count": len(threshold_violations),
        "exact_duplicates": exact,
        "normalized_duplicates": normalized,
        "similarity_violations": threshold_violations,
        "maximum_similarity": maximum_similarity,
        "similarity_threshold": float(similarity_threshold),
        "closest_source": closest,
        "source_review_count": len(sources),
        "source_file_sha256": source_hashes,
        "source_corpus_fingerprint": _mapping_fingerprint(source_hashes),
        "passed": not exact and not normalized and not threshold_violations,
    }


def current_runtime_configuration() -> dict[str, Any]:
    return {
        "provider": LLM_PROVIDER,
        "model": OLLAMA_MODEL,
        "ollama_num_ctx": OLLAMA_NUM_CTX,
        "ollama_keep_alive": OLLAMA_KEEP_ALIVE,
        "taxonomy_version": TAXONOMY_VERSION,
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BlindV2IntegrityError(f"{label} was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BlindV2IntegrityError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise BlindV2IntegrityError(f"{label} must be a JSON object.")
    return dict(payload)


def create_blind_v2_manifest(
    dataset_path: str | Path = BLIND_V2_CASES_PATH,
    manifest_path: str | Path = BLIND_V2_MANIFEST_PATH,
    *,
    expected_count: int = EXPECTED_CASE_COUNT,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
    core_model_paths: Sequence[str | Path] = CORE_MODEL_FILES,
    protected_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Freeze the complete dataset and environment before any model call."""

    dataset = Path(dataset_path)
    destination = Path(manifest_path)
    if destination.exists():
        raise BlindV2IntegrityError("Blind V2 manifest already exists and is immutable.")
    cases = validate_blind_v2_cases(dataset, expected_count=expected_count)
    distribution = validate_blind_v2_distribution(cases)
    leakage = check_blind_v2_leakage(
        dataset,
        expected_count=expected_count,
        comparison_dataset_paths=comparison_dataset_paths,
        demo_paths=demo_paths,
    )
    if not leakage["passed"]:
        raise BlindV2IntegrityError(
            "Blind V2 leakage check failed; replace overlapping cases before evaluation."
        )
    protected = tuple(protected_paths or protected_non_model_paths())
    core_hashes = snapshot_file_hashes(core_model_paths)
    protected_hashes = snapshot_file_hashes(protected)
    manifest = {
        "dataset": dataset.name,
        "dataset_filename": dataset.name,
        "sha256": file_sha256(dataset),
        "case_count": distribution["case_count"],
        "gold_mention_count": distribution["gold_mention_count"],
        "aspect_distribution": distribution["aspect_distribution"],
        "sentiment_distribution": distribution["sentiment_distribution"],
        "sentiment_ratios": distribution["sentiment_ratios"],
        "case_cardinality_distribution": distribution[
            "case_cardinality_distribution"
        ],
        "hard_case_distribution": distribution["hard_case_distribution"],
        "average_review_words": distribution["average_review_words"],
        "median_review_words": distribution["median_review_words"],
        "average_review_characters": distribution["average_review_characters"],
        "median_review_characters": distribution["median_review_characters"],
        "created_before_evaluation": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy_version": TAXONOMY_VERSION,
        "runtime_configuration": current_runtime_configuration(),
        "configuration_fingerprint": _mapping_fingerprint(
            current_runtime_configuration()
        ),
        "core_model_file_sha256": core_hashes,
        "core_model_fingerprint": _mapping_fingerprint(core_hashes),
        "protected_asset_sha256": protected_hashes,
        "protected_asset_fingerprint": _mapping_fingerprint(protected_hashes),
        "leakage_check": leakage,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_blind_v2_manifest(
    dataset_path: str | Path = BLIND_V2_CASES_PATH,
    manifest_path: str | Path = BLIND_V2_MANIFEST_PATH,
    *,
    expected_count: int = EXPECTED_CASE_COUNT,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
    core_model_paths: Sequence[str | Path] = CORE_MODEL_FILES,
    protected_paths: Sequence[str | Path] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = Path(dataset_path)
    manifest = _load_json_object(Path(manifest_path), "Blind V2 manifest")
    if manifest.get("dataset") != dataset.name:
        raise BlindV2IntegrityError("Blind V2 manifest dataset name changed.")
    if manifest.get("case_count") != expected_count:
        raise BlindV2IntegrityError("Blind V2 manifest case count changed.")
    if manifest.get("created_before_evaluation") is not True:
        raise BlindV2IntegrityError("Manifest must predate model evaluation.")
    if manifest.get("sha256") != file_sha256(dataset):
        raise BlindV2IntegrityError("Blind V2 dataset changed after freezing.")
    if manifest.get("taxonomy_version") != TAXONOMY_VERSION:
        raise BlindV2IntegrityError("Blind V2 taxonomy fingerprint changed.")
    if manifest.get("configuration_fingerprint") != _mapping_fingerprint(
        current_runtime_configuration()
    ):
        raise BlindV2IntegrityError("Blind V2 runtime configuration changed.")
    cases = validate_blind_v2_cases(dataset, expected_count=expected_count)
    distribution = validate_blind_v2_distribution(cases)
    for key in (
        "gold_mention_count",
        "aspect_distribution",
        "sentiment_distribution",
        "case_cardinality_distribution",
        "hard_case_distribution",
        "average_review_words",
        "median_review_words",
    ):
        if manifest.get(key) != distribution[key]:
            raise BlindV2IntegrityError(f"Manifest {key} does not match the dataset.")
    verify_file_hashes(manifest.get("core_model_file_sha256", {}), core_model_paths)
    protected = tuple(protected_paths or protected_non_model_paths())
    verify_file_hashes(manifest.get("protected_asset_sha256", {}), protected)
    leakage = manifest.get("leakage_check")
    if not isinstance(leakage, Mapping) or leakage.get("passed") is not True:
        raise BlindV2IntegrityError("Manifest has no passing leakage check.")
    _, source_hashes = _source_records(comparison_dataset_paths, demo_paths)
    if leakage.get("source_corpus_fingerprint") != _mapping_fingerprint(source_hashes):
        raise BlindV2IntegrityError("A leakage comparison source changed after freeze.")
    return manifest, cases


def _rates(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0,
    }


def calculate_per_aspect_metrics(
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    totals = {
        aspect: {"support": 0, "tp": 0, "fp": 0, "fn": 0}
        for aspect in STANDARD_ASPECTS
    }
    for case in case_results:
        expected_raw = case.get("expected_aspects", [])
        actual_raw = case.get("evaluation_aspects", case.get("actual_aspects", []))
        expected = expected_raw if isinstance(expected_raw, list) else []
        actual = actual_raw if isinstance(actual_raw, list) else []
        gold = Counter(
            (str(item.get("aspect", "")), str(item.get("sentiment", "")))
            for item in expected
            if isinstance(item, Mapping)
        )
        predicted = Counter(
            (str(item.get("aspect", "")), str(item.get("sentiment", "")))
            for item in actual
            if isinstance(item, Mapping)
        )
        for aspect in STANDARD_ASPECTS:
            support = sum(count for (label, _), count in gold.items() if label == aspect)
            predicted_count = sum(
                count for (label, _), count in predicted.items() if label == aspect
            )
            tp = sum(
                min(count, predicted[(label, sentiment)])
                for (label, sentiment), count in gold.items()
                if label == aspect
            )
            totals[aspect]["support"] += support
            totals[aspect]["tp"] += tp
            totals[aspect]["fp"] += predicted_count - tp
            totals[aspect]["fn"] += support - tp
    return {
        aspect: {**counts, **_rates(counts["tp"], counts["fp"], counts["fn"])}
        for aspect, counts in totals.items()
    }


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def _load_comparison_summaries(
    paths: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for label, raw_path in paths.items():
        payload = _load_json_object(Path(raw_path), f"{label} comparison report")
        summary = payload.get("summary")
        if not isinstance(summary, Mapping):
            raise BlindV2IntegrityError(f"{label} report has no summary.")
        summaries[label] = dict(summary)
    return summaries


def _generalization_gap(
    development: Mapping[str, Any], blind: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "repair_dev_v2": float(development.get(metric, 0.0)),
            "blind_v2": float(blind.get(metric, 0.0)),
            "blind_minus_repair": (
                float(blind.get(metric, 0.0))
                - float(development.get(metric, 0.0))
            ),
        }
        for metric in GAP_METRICS
    }


def interpret_generalization(aspect_f1: float) -> dict[str, Any]:
    """Apply the reporting-only thresholds fixed in the Step 8.13 brief."""

    value = float(aspect_f1)
    if value >= 0.80:
        band = "strong_generalization_evidence"
        message = "Step 8.12.1 improvements show strong generalization evidence."
        supports = True
    elif value >= 0.75:
        band = "some_generalization_with_boundary_risk"
        message = (
            "The candidate shows some generalization improvement, while Aspect "
            "boundaries still need long-term improvement."
        )
        supports = True
    elif value >= 0.70:
        band = "development_specific_improvement_risk"
        message = (
            "The development gain appears materially development-specific."
        )
        supports = False
    else:
        band = "no_reliable_generalization"
        message = "The V2 improvement does not show reliable generalization."
        supports = False
    return {
        "aspect_sentiment_f1": value,
        "threshold_band": band,
        "supports_repair_improvements_generalized": supports,
        "interpretation": message,
        "reporting_only": True,
    }


def evaluation_outputs_exist(
    paths: Sequence[str | Path] = (
        BLIND_V2_STARTED_PATH,
        BLIND_V2_PREDICTIONS_PATH,
        BLIND_V2_REPORT_PATH,
        BLIND_V2_ERRORS_PATH,
        BLIND_V2_FREEZE_PATH,
    ),
) -> bool:
    return any(Path(path).exists() for path in paths)


def _write_started_sentinel(
    path: str | Path,
    *,
    dataset_sha256: str,
    core_hashes: Mapping[str, str],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_type": "blind_v2",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_sha256,
        "core_model_file_sha256": dict(core_hashes),
        "provider": "ollama",
        "model": "qwen3:8b",
        "policy": RUN_ONCE_NOTICE,
    }
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise BlindV2IntegrityError(
            "Blind V2 was already started; rerun is forbidden."
        ) from exc
    return destination


def run_blind_v2_evaluation(
    *,
    dataset_path: str | Path = BLIND_V2_CASES_PATH,
    manifest_path: str | Path = BLIND_V2_MANIFEST_PATH,
    started_path: str | Path = BLIND_V2_STARTED_PATH,
    expected_count: int = EXPECTED_CASE_COUNT,
    provider: str = LLM_PROVIDER,
    model: str = OLLAMA_MODEL,
    comparison_dataset_paths: Sequence[str | Path] = COMPARISON_DATASETS,
    demo_paths: Sequence[str | Path] = DEMO_REVIEW_FILES,
    core_model_paths: Sequence[str | Path] = CORE_MODEL_FILES,
    protected_paths: Sequence[str | Path] | None = None,
    comparison_report_paths: Mapping[str, str | Path] = COMPARISON_REPORTS,
    client: ReviewAnalysisClient | None = None,
    client_factory: Callable[[], ReviewAnalysisClient] | None = None,
    analyzer: Callable[..., Mapping[str, Any]] = analyze_review,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    retry_delay_seconds: float = TRANSIENT_RETRY_DELAY_SECONDS,
    progress: Callable[[int, int, str, str, float], None] | None = None,
) -> dict[str, Any]:
    """Perform the one permitted serial qwen3:8b Blind V2 evaluation."""

    protected = tuple(protected_paths or protected_non_model_paths())
    manifest, cases = verify_blind_v2_manifest(
        dataset_path,
        manifest_path,
        expected_count=expected_count,
        comparison_dataset_paths=comparison_dataset_paths,
        demo_paths=demo_paths,
        core_model_paths=core_model_paths,
        protected_paths=protected,
    )
    if provider != "ollama" or model != "qwen3:8b":
        raise BlindV2IntegrityError("Blind V2 is locked to Ollama qwen3:8b.")
    if client is not None and client_factory is not None:
        raise ValueError("Provide client or client_factory, not both.")
    _write_started_sentinel(
        started_path,
        dataset_sha256=str(manifest["sha256"]),
        core_hashes=manifest["core_model_file_sha256"],
    )
    resolved_client = client or (
        client_factory() if client_factory is not None else create_review_client()
    )
    core_before = snapshot_file_hashes(core_model_paths)
    protected_before = snapshot_file_hashes(protected)
    comparison_summaries = _load_comparison_summaries(comparison_report_paths)

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
            except Exception as exc:
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
        result["tags"] = case["tags"]
        result["notes"] = case["notes"]
        result["attempts"] = attempts
        result["retried"] = attempts > 1
        observed = {
            str(error.get("error_type"))
            for error in result.get("errors", [])
            if isinstance(error, Mapping)
        }
        fatal = (
            "SCHEMA_ERROR"
            if "SCHEMA_ERROR" in observed
            else "MODEL_ERROR"
            if "MODEL_ERROR" in observed
            else ""
        )
        status = "failed" if fatal else "success"
        predictions.append(
            {
                "case_id": case["case_id"],
                "review": case["review"],
                "status": status,
                "attempts": attempts,
                "retried": attempts > 1,
                "analysis_time_seconds": elapsed,
                "analysis": dict(actual) if isinstance(actual, Mapping) else None,
                "error_type": fatal,
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
    protected_after = verify_file_hashes(protected_before, protected)
    verify_blind_v2_manifest(
        dataset_path,
        manifest_path,
        expected_count=expected_count,
        comparison_dataset_paths=comparison_dataset_paths,
        demo_paths=demo_paths,
        core_model_paths=core_model_paths,
        protected_paths=protected,
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
    repair_summary = comparison_summaries["step8.12.1_dev_v2_candidate"]
    return {
        "evaluation_type": "blind_v2",
        "evaluation_policy": EVALUATION_NOTICE,
        "run_once_notice": RUN_ONCE_NOTICE,
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
        "protected_asset_sha256_before": protected_before,
        "protected_asset_sha256_after": protected_after,
        "protected_assets_unchanged": protected_before == protected_after,
        "summary": summary,
        "runtime_statistics": {
            "total_seconds": sum(runtimes),
            "mean_seconds_per_review": sum(runtimes) / len(runtimes)
            if runtimes
            else 0.0,
            "median_seconds_per_review": statistics.median(runtimes)
            if runtimes
            else 0.0,
            "p90_seconds_per_review": _percentile_nearest_rank(runtimes, 0.90),
            "p90_method": "nearest_rank",
            "cold_start_first_case_seconds": runtimes[0] if runtimes else 0.0,
            "retry_count": retry_count,
            "model_failure_count": error_counts["MODEL_ERROR"],
            "schema_failure_count": error_counts["SCHEMA_ERROR"],
            "step8.12.1_mean_seconds_per_review": 6.24,
        },
        "confusion_matrix": summary["confusion_matrix"],
        "error_counts": error_counts,
        "per_aspect_metrics": calculate_per_aspect_metrics(results),
        "top_errors": select_top_errors(results, limit=10),
        "development_comparison": comparison_summaries,
        "repair_to_blind_v2_gap": _generalization_gap(repair_summary, summary),
        "generalization_interpretation": interpret_generalization(
            float(summary.get("aspect_f1", 0.0))
        ),
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


def save_blind_v2_outputs(
    report: Mapping[str, Any],
    *,
    report_path: str | Path = BLIND_V2_REPORT_PATH,
    errors_path: str | Path = BLIND_V2_ERRORS_PATH,
    predictions_path: str | Path = BLIND_V2_PREDICTIONS_PATH,
    freeze_path: str | Path = BLIND_V2_FREEZE_PATH,
    dataset_path: str | Path = BLIND_V2_CASES_PATH,
    manifest_path: str | Path = BLIND_V2_MANIFEST_PATH,
    started_path: str | Path = BLIND_V2_STARTED_PATH,
) -> dict[str, str]:
    """Persist the one-time report and create its permanent freeze manifest."""

    report_file = Path(report_path)
    errors_file = Path(errors_path)
    predictions_file = Path(predictions_path)
    freeze_file = Path(freeze_path)
    destinations = (report_file, errors_file, predictions_file, freeze_file)
    if any(path.exists() for path in destinations):
        raise BlindV2IntegrityError("Blind V2 outputs already exist and are immutable.")
    if not Path(started_path).is_file():
        raise BlindV2IntegrityError("Blind V2 start sentinel is missing.")
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "report": str(report_file),
        "errors": str(errors_file),
        "predictions": str(predictions_file),
        "freeze_manifest": str(freeze_file),
    }
    payload = dict(report)
    payload["outputs"] = output_paths
    report_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with errors_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_error_rows(payload))
    predictions_payload = {
        "evaluation_policy": EVALUATION_NOTICE,
        "run_once_notice": RUN_ONCE_NOTICE,
        "provider": payload.get("provider", ""),
        "model": payload.get("model", ""),
        "dataset_sha256": payload.get("dataset_sha256", ""),
        "timestamp": payload.get("timestamp", ""),
        "case_count": payload.get("case_count", 0),
        "predictions": payload.get("predictions", []),
    }
    predictions_file.write_text(
        json.dumps(predictions_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    frozen_paths = {
        "dataset": Path(dataset_path),
        "manifest": Path(manifest_path),
        "predictions": predictions_file,
        "evaluation_report": report_file,
        "errors_csv": errors_file,
        "started_sentinel": Path(started_path),
    }
    frozen_hashes = {
        label: file_sha256(path) for label, path in frozen_paths.items()
    }
    freeze_payload = {
        "evaluation_type": "blind_v2",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "case_count": payload.get("case_count", 0),
        "dataset_sha256": payload.get("dataset_sha256", ""),
        "frozen_file_sha256": frozen_hashes,
        "core_model_file_sha256": payload.get(
            "core_model_file_sha256_after", {}
        ),
        "protected_asset_fingerprint": _mapping_fingerprint(
            payload.get("protected_asset_sha256_after", {})
        ),
        "run_once_notice": RUN_ONCE_NOTICE,
        "no_post_blind_optimization": True,
    }
    freeze_file.write_text(
        json.dumps(freeze_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output_paths


def format_blind_v2_report(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    runtime = report.get("runtime_statistics", {})
    per_aspect = report.get("per_aspect_metrics", {})
    leakage = report.get("leakage_check", {})
    lines = [
        "Step 8.13 — Blind Test V2",
        RUN_ONCE_NOTICE,
        EVALUATION_NOTICE,
        f"Cases: {int(summary.get('total_cases', 0))}",
        f"Successful / Failed: {int(summary.get('successful_cases', 0))} / "
        f"{int(summary.get('failed_cases', 0))}",
        f"Aspect+Sentiment P/R/F1: "
        f"{float(summary.get('aspect_precision', 0.0)):.2%} / "
        f"{float(summary.get('aspect_recall', 0.0)):.2%} / "
        f"{float(summary.get('aspect_f1', 0.0)):.2%}",
        f"Aspect-only P/R/F1: "
        f"{float(summary.get('aspect_only_precision', 0.0)):.2%} / "
        f"{float(summary.get('aspect_only_recall', 0.0)):.2%} / "
        f"{float(summary.get('aspect_only_f1', 0.0)):.2%}",
        f"Sentiment Accuracy: {float(summary.get('sentiment_accuracy', 0.0)):.2%}",
        f"Evidence Validity: {float(summary.get('evidence_validity_rate', 0.0)):.2%}",
        f"Evidence Match: {float(summary.get('evidence_match_rate', 0.0)):.2%}",
        f"Strict Exact: {int(summary.get('exact_matches', 0))}/"
        f"{int(summary.get('total_cases', 0))} "
        f"({float(summary.get('exact_case_accuracy', 0.0)):.2%})",
        "TP / FP / FN: "
        f"{int(summary.get('aspect_true_positives', 0))} / "
        f"{int(summary.get('aspect_false_positives', 0))} / "
        f"{int(summary.get('aspect_false_negatives', 0))}",
        f"Leakage max similarity: {float(leakage.get('maximum_similarity', 0.0)):.4f}",
        f"Errors: {json.dumps(report.get('error_counts', {}), sort_keys=True)}",
        "Per-Aspect (Aspect + Sentiment):",
    ]
    for aspect in STANDARD_ASPECTS:
        metrics = per_aspect.get(aspect, {}) if isinstance(per_aspect, Mapping) else {}
        lines.append(
            f"  {aspect}: support={int(metrics.get('support', 0))}, "
            f"P={float(metrics.get('precision', 0.0)):.2%}, "
            f"R={float(metrics.get('recall', 0.0)):.2%}, "
            f"F1={float(metrics.get('f1', 0.0)):.2%}"
        )
    lines.extend(
        [
            f"Total runtime: {float(runtime.get('total_seconds', 0.0)):.2f}s",
            f"Mean runtime: {float(runtime.get('mean_seconds_per_review', 0.0)):.2f}s/review",
            f"Median runtime: {float(runtime.get('median_seconds_per_review', 0.0)):.2f}s",
            f"P90 runtime: {float(runtime.get('p90_seconds_per_review', 0.0)):.2f}s",
            f"Cold-start: {float(runtime.get('cold_start_first_case_seconds', 0.0)):.2f}s",
            f"Retries: {int(runtime.get('retry_count', 0))}",
            f"Model failures: {int(runtime.get('model_failure_count', 0))}",
            f"Production hashes unchanged: {bool(report.get('core_model_hashes_unchanged'))}",
            f"Protected assets unchanged: {bool(report.get('protected_assets_unchanged'))}",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "BLIND_V2_CASES_PATH",
    "BLIND_V2_ERRORS_PATH",
    "BLIND_V2_FREEZE_PATH",
    "BLIND_V2_MANIFEST_PATH",
    "BLIND_V2_OUTPUT_DIR",
    "BLIND_V2_PREDICTIONS_PATH",
    "BLIND_V2_REPORT_PATH",
    "BLIND_V2_STARTED_PATH",
    "BlindV2IntegrityError",
    "COMPARISON_DATASETS",
    "COMPARISON_REPORTS",
    "CORE_MODEL_FILES",
    "DEMO_REVIEW_FILES",
    "ERROR_TYPES",
    "EXPECTED_CASE_COUNT",
    "calculate_per_aspect_metrics",
    "check_blind_v2_leakage",
    "create_blind_v2_manifest",
    "dataset_distribution",
    "evaluation_outputs_exist",
    "format_blind_v2_report",
    "interpret_generalization",
    "normalize_review_text",
    "protected_non_model_paths",
    "run_blind_v2_evaluation",
    "save_blind_v2_outputs",
    "snapshot_file_hashes",
    "validate_blind_v2_cases",
    "validate_blind_v2_distribution",
    "verify_blind_v2_manifest",
    "verify_file_hashes",
]
