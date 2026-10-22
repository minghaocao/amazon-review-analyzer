"""Step 8.12.1 regression isolation and one-shot repair evaluation.

This module is intentionally limited to the frozen Generalization Dev V2 set.
It never invokes Final Blind, Holdout, Amazon, OpenAI, or dashboard code.
Importing it performs no model or file-writing work.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from aspect_taxonomy import STANDARD_ASPECTS, taxonomy_definition_text
from config import (
    DEMO_OUTPUT_PATHS,
    LLM_PROVIDER,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    PROJECT_ROOT,
)
from generalization_dev_v2_evaluator import (
    CORE_MODEL_FILES,
    ERROR_TYPES,
    PRIORITY_ASPECTS,
    TRANSIENT_RETRY_DELAY_SECONDS,
    GeneralizationDevV2IntegrityError,
    _error_rows,
    _percentile_nearest_rank,
    snapshot_file_hashes,
    verify_file_hashes,
)
from generalization_dev_v2_optimizer import (
    FROZEN_STEP_811_PATHS,
    FROZEN_STEP_811_SHA256,
    STABLE_ASPECTS,
    calculate_aspect_metrics,
    calculate_report_deltas,
    run_generalization_dev_v2_optimization_round,
    stable_aspect_regressions,
    verify_frozen_step_811_inputs,
)
from holdout_evaluator import ERROR_CSV_FIELDS, file_sha256
from llm_client import ReviewAnalysisClient, _analysis_instructions, _response_schema
from review_analyzer import analyze_review


REPAIR_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "evaluation" / "generalization_dev_v2" / "repair"
)
REGRESSION_ANALYSIS_PATH = REPAIR_OUTPUT_DIR / "round1_regression_analysis.json"
REGRESSION_CASES_PATH = REPAIR_OUTPUT_DIR / "round1_regression_cases.csv"
REPAIR_REPORT_PATH = REPAIR_OUTPUT_DIR / "repair_candidate_evaluation.json"
REPAIR_ERRORS_PATH = REPAIR_OUTPUT_DIR / "repair_candidate_errors.csv"
REPAIR_PREDICTIONS_PATH = REPAIR_OUTPUT_DIR / "repair_candidate_predictions.json"
REPAIR_RUNTIME_PATH = REPAIR_OUTPUT_DIR / "repair_runtime_analysis.json"
REPAIR_STARTED_PATH = REPAIR_OUTPUT_DIR / "repair_evaluation_started.json"

V2_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "generalization_dev_v2"
ROUND1_PATHS: Mapping[str, Path] = {
    "diagnostic": V2_OUTPUT_DIR / "optimization_baseline_diagnostic.json",
    "report": V2_OUTPUT_DIR / "optimization_round1_generalization_dev_v2.json",
    "errors": V2_OUTPUT_DIR / "optimization_round1_errors.csv",
    "predictions": V2_OUTPUT_DIR / "optimization_round1_predictions.json",
    "best": V2_OUTPUT_DIR / "best_generalization_dev_v2_qwen3_8b.json",
}
FROZEN_ROUND1_SHA256: Mapping[str, str] = {
    "diagnostic": "6e3aa35c8eed5656ae92c0fa1b6f2fe8657f4ebc94b2f920a630b67aad4b886c",
    "report": "69b1085aa7793667ff241ae3247029a9fe669c50912bd1d291edad9fc4b2000b",
    "errors": "10923db16eff272aec2e80d1b700553a9cd4d4da1970376e4d9c63f4db146e62",
    "predictions": "97c39ece447e3ea5da26c0132dd266747b15eefbf07efc66f09252cce7b194ff",
    "best": "f734079d86dfbfcbd52eb3992226632c5d16b00cf62a37bef2cb2d92f27e28d5",
}

# Exact measurements from the SHA-verified Step 8.11 and Step 8.12 source
# snapshots. Tokens are deliberately labelled estimates (characters / 4), not
# tokenizer output. The JSON schema is included because Ollama receives it in
# the system message.
BASELINE_PROMPT_PROFILE: Mapping[str, Any] = {
    "source": "SHA-verified Step 8.11 source snapshot",
    "instruction_chars": 13_892,
    "instruction_words": 2_053,
    "instruction_approx_tokens_chars_div_4": 3_473,
    "taxonomy_chars": 2_726,
    "schema_chars": 1_783,
    "ollama_system_chars_with_schema": 15_689,
    "ollama_system_approx_tokens_chars_div_4": 3_922,
}
ROUND1_PROMPT_PROFILE: Mapping[str, Any] = {
    "source": "SHA-verified Step 8.12 Round 1 source snapshot",
    "instruction_chars": 18_316,
    "instruction_words": 2_632,
    "instruction_approx_tokens_chars_div_4": 4_579,
    "taxonomy_chars": 4_357,
    "schema_chars": 1_783,
    "ollama_system_chars_with_schema": 20_113,
    "ollama_system_approx_tokens_chars_div_4": 5_028,
}

EXPECTED_PROVIDER_CONFIGURATION: Mapping[str, Any] = {
    "provider": "ollama",
    "model": "qwen3:8b",
    "num_ctx": 4096,
    "temperature": 0,
    "keep_alive": "30m",
    "timeout_seconds": 120.0,
    "structured_format": "Pydantic JSON Schema",
    "think": False,
    "num_predict": None,
}

REGRESSION_CASE_FIELDS = (
    "case_id",
    "transition_types",
    "focus",
    "baseline_exact",
    "round1_exact",
    "baseline_errors",
    "round1_errors",
    "expected_aspects",
    "baseline_aspects",
    "round1_aspects",
    "root_cause",
    "review",
)


def protected_non_model_paths() -> tuple[Path, ...]:
    """Return immutable evaluation, Demo, and UI/import files for this run."""

    paths: set[Path] = {Path(path) for path in DEMO_OUTPUT_PATHS}
    for folder in (
        PROJECT_ROOT / "evaluation",
        PROJECT_ROOT / "outputs" / "evaluation" / "final_blind",
        PROJECT_ROOT / "outputs" / "evaluation" / "holdout",
        PROJECT_ROOT / "outputs" / "evaluation" / "generalization_error_study",
    ):
        if folder.is_dir():
            paths.update(path for path in folder.glob("*") if path.is_file())
    paths.update(
        PROJECT_ROOT / name
        for name in (
            "evaluator.py",
            "app.py",
            "browser_amazon_import.py",
            "amazon_url.py",
            "scraper.py",
            "amazon_analysis.py",
            "input_loader.py",
            "visualization.py",
        )
    )
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise GeneralizationDevV2IntegrityError(
            "A protected Step 8.12.1 file is unavailable: " + ", ".join(missing)
        )
    return tuple(sorted(paths, key=lambda path: str(path)))


def current_provider_configuration() -> dict[str, Any]:
    """Describe the effective local provider settings used by the repair."""

    return {
        "provider": LLM_PROVIDER,
        "model": OLLAMA_MODEL,
        "num_ctx": OLLAMA_NUM_CTX,
        "temperature": 0,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "timeout_seconds": 120.0,
        "structured_format": "Pydantic JSON Schema",
        "think": False,
        "num_predict": None,
    }


def verify_provider_configuration() -> dict[str, Any]:
    """Fail closed when the repair would not match the frozen benchmark setup."""

    current = current_provider_configuration()
    if current != dict(EXPECTED_PROVIDER_CONFIGURATION):
        raise GeneralizationDevV2IntegrityError(
            "Step 8.12.1 provider configuration changed from the frozen benchmark."
        )
    return current


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeneralizationDevV2IntegrityError(
            f"Frozen {label} was not found: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GeneralizationDevV2IntegrityError(
            f"Frozen {label} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise GeneralizationDevV2IntegrityError(
            f"Frozen {label} must be a JSON object."
        )
    return dict(payload)


def verify_frozen_round1_files(
    *,
    paths: Mapping[str, str | Path] = ROUND1_PATHS,
    expected_sha256: Mapping[str, str] = FROZEN_ROUND1_SHA256,
) -> dict[str, str]:
    """Fail closed if any Step 8.12 Round 1 artifact changed."""

    if set(paths) != set(ROUND1_PATHS) or set(expected_sha256) != set(ROUND1_PATHS):
        raise GeneralizationDevV2IntegrityError(
            "The frozen Step 8.12 Round 1 file set is incomplete."
        )
    observed: dict[str, str] = {}
    for label in sorted(paths):
        path = Path(paths[label])
        if not path.is_file():
            raise GeneralizationDevV2IntegrityError(
                f"Frozen Step 8.12 Round 1 {label} was not found: {path}"
            )
        observed[label] = file_sha256(path)
        if observed[label] != expected_sha256[label]:
            raise GeneralizationDevV2IntegrityError(
                f"Frozen Step 8.12 Round 1 {label} SHA-256 changed."
            )
    return observed


def current_prompt_profile() -> dict[str, Any]:
    """Measure the current production prompt without making a model call."""

    instructions = _analysis_instructions(STANDARD_ASPECTS)
    schema = json.dumps(_response_schema(STANDARD_ASPECTS), ensure_ascii=False)
    separator = "\nJSON Schema:\n"
    system_chars = len(instructions) + len(separator) + len(schema)
    return {
        "source": "current production source",
        "instruction_chars": len(instructions),
        "instruction_words": len(instructions.split()),
        "instruction_approx_tokens_chars_div_4": round(len(instructions) / 4),
        "taxonomy_chars": len(taxonomy_definition_text()),
        "schema_chars": len(schema),
        "ollama_system_chars_with_schema": system_chars,
        "ollama_system_approx_tokens_chars_div_4": round(system_chars / 4),
    }


def _normalized_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def _text_similarity(left: Any, right: Any) -> float:
    first = _normalized_text(left)
    second = _normalized_text(right)
    if not first or not second:
        return 0.0
    if first in second or second in first:
        return 1.0
    return SequenceMatcher(None, first, second).ratio()


def _extras(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        for error in case.get("errors", []):
            if isinstance(error, Mapping) and error.get("error_type") == "EXTRA_ASPECT":
                rows.append({"case_id": case.get("case_id", ""), **dict(error)})
    return rows


def _pair_extra_mentions(
    baseline_extras: Sequence[Mapping[str, Any]],
    round1_extras: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair the same false claim even if Round 1 changed its taxonomy label."""

    candidates: list[tuple[float, int, int]] = []
    for left_index, left in enumerate(baseline_extras):
        for right_index, right in enumerate(round1_extras):
            if left.get("case_id") != right.get("case_id"):
                continue
            similarity = _text_similarity(
                left.get("actual_evidence", ""), right.get("actual_evidence", "")
            )
            if similarity >= threshold:
                candidates.append((similarity, left_index, right_index))
    used_left: set[int] = set()
    used_right: set[int] = set()
    persistent: list[dict[str, Any]] = []
    for similarity, left_index, right_index in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        persistent.append(
            {
                "similarity": similarity,
                "baseline": dict(baseline_extras[left_index]),
                "round1": dict(round1_extras[right_index]),
            }
        )
    fixed = [
        dict(item) for index, item in enumerate(baseline_extras) if index not in used_left
    ]
    new = [
        dict(item) for index, item in enumerate(round1_extras) if index not in used_right
    ]
    return persistent, fixed, new


_FACT_PATTERN = re.compile(
    r"\b(?:there (?:is|are)|has|have|contains?|lists?|shows?|says?|records?|"
    r"ordered|invoice|receipt|inventory|serial|batch|tracking showed|"
    r"assigned|assembled|does not begin until|left the parcel)\b",
    flags=re.IGNORECASE,
)
_COMPARISON_PATTERN = re.compile(
    r"\b(?:old|previous|former|rival|competitor|alternative)\b",
    flags=re.IGNORECASE,
)


def _round1_extra_root_causes(
    extras: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Attach transparent, possibly overlapping root-cause tags."""

    case_aspect_counts = Counter(
        (str(item.get("case_id", "")), str(item.get("actual_aspect", "")))
        for item in extras
    )
    tagged: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for item in extras:
        aspect = str(item.get("actual_aspect", ""))
        sentiment = str(item.get("actual_sentiment", ""))
        evidence = str(item.get("actual_evidence", ""))
        tags: list[str] = []
        if sentiment == "neutral" or _FACT_PATTERN.search(evidence):
            tags.append("attribute_vs_evaluation")
        if case_aspect_counts[(str(item.get("case_id", "")), aspect)] > 1:
            tags.append("repeated_opinion_over_split")
        if _COMPARISON_PATTERN.search(evidence):
            tags.append("comparison_product_leakage")
        if aspect == "Design":
            tags.append("design_overreach")
        elif aspect == "Packaging":
            tags.append("packaging_overreach")
        elif aspect == "Setup / Installation":
            tags.append("setup_overreach")
        elif aspect == "Ease of Use":
            tags.append("ease_overreach")
        elif aspect == "Other":
            tags.append("other_misuse")
        elif aspect == "Delivery":
            tags.append("delivery_overreach")
        if not tags:
            tags.append("other")
        counts.update(tags)
        tagged.append({**dict(item), "root_causes": tags})
    return tagged, counts


def _errors_for_aspect(case: Mapping[str, Any], aspect: str) -> list[dict[str, Any]]:
    return [
        dict(error)
        for error in case.get("errors", [])
        if isinstance(error, Mapping)
        and (
            error.get("expected_aspect") == aspect
            or error.get("actual_aspect") == aspect
        )
    ]


def _aspect_regression(
    baseline: Mapping[str, Any], round1: Mapping[str, Any], aspect: str
) -> dict[str, Any]:
    baseline_cases = {
        str(case.get("case_id")): case
        for case in baseline.get("cases", [])
        if isinstance(case, Mapping)
    }
    round1_cases = {
        str(case.get("case_id")): case
        for case in round1.get("cases", [])
        if isinstance(case, Mapping)
    }
    details: list[dict[str, Any]] = []
    for case_id, before in baseline_cases.items():
        after = round1_cases[case_id]
        before_errors = _errors_for_aspect(before, aspect)
        after_errors = _errors_for_aspect(after, aspect)
        before_pairs = Counter(
            (item.get("aspect"), item.get("sentiment"))
            for item in before.get("evaluation_aspects", [])
            if isinstance(item, Mapping) and item.get("aspect") == aspect
        )
        after_pairs = Counter(
            (item.get("aspect"), item.get("sentiment"))
            for item in after.get("evaluation_aspects", [])
            if isinstance(item, Mapping) and item.get("aspect") == aspect
        )
        if before_errors != after_errors or before_pairs != after_pairs:
            details.append(
                {
                    "case_id": case_id,
                    "review": before.get("review", ""),
                    "expected": [
                        item
                        for item in before.get("expected_aspects", [])
                        if isinstance(item, Mapping) and item.get("aspect") == aspect
                    ],
                    "baseline_actual": [
                        item
                        for item in before.get("evaluation_aspects", [])
                        if isinstance(item, Mapping) and item.get("aspect") == aspect
                    ],
                    "round1_actual": [
                        item
                        for item in after.get("evaluation_aspects", [])
                        if isinstance(item, Mapping) and item.get("aspect") == aspect
                    ],
                    "baseline_errors": before_errors,
                    "round1_errors": after_errors,
                }
            )
    return {"aspect": aspect, "changed_cases": details}


def build_round1_regression_analysis(
    baseline: Mapping[str, Any], round1: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the required pre-repair diff without modifying model logic."""

    baseline_cases = {
        str(case.get("case_id")): case
        for case in baseline.get("cases", [])
        if isinstance(case, Mapping)
    }
    round1_cases = {
        str(case.get("case_id")): case
        for case in round1.get("cases", [])
        if isinstance(case, Mapping)
    }
    if set(baseline_cases) != set(round1_cases):
        raise GeneralizationDevV2IntegrityError(
            "Baseline and Round 1 case identities do not match."
        )

    baseline_extras = _extras(baseline)
    round1_extras = _extras(round1)
    persistent, fixed, new = _pair_extra_mentions(baseline_extras, round1_extras)
    tagged_extras, root_counts = _round1_extra_root_causes(round1_extras)

    transitions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    model_failures: list[str] = []
    for case_id in sorted(baseline_cases):
        before = baseline_cases[case_id]
        after = round1_cases[case_id]
        tags: list[str] = []
        before_exact = bool(before.get("exact_match"))
        after_exact = bool(after.get("exact_match"))
        if before_exact and not after_exact:
            tags.append("baseline_correct_to_round1_incorrect")
        if not before_exact and after_exact:
            tags.append("baseline_incorrect_to_round1_correct")
        before_error_types = Counter(
            str(item.get("error_type"))
            for item in before.get("errors", [])
            if isinstance(item, Mapping)
        )
        after_error_types = Counter(
            str(item.get("error_type"))
            for item in after.get("errors", [])
            if isinstance(item, Mapping)
        )
        for error_type in ERROR_TYPES:
            delta = after_error_types[error_type] - before_error_types[error_type]
            if delta > 0:
                tags.append(f"new_{error_type.lower()}")
        if after_error_types["MODEL_ERROR"]:
            model_failures.append(case_id)
        focus = sorted(
            {
                str(item.get("expected_aspect") or item.get("actual_aspect") or "")
                for item in [*before.get("errors", []), *after.get("errors", [])]
                if isinstance(item, Mapping)
                and (item.get("expected_aspect") or item.get("actual_aspect"))
            }
        )
        if tags:
            transitions.update(tags)
            root_cause = []
            if case_id in model_failures:
                root_cause.append("transient_provider_failure_after_one_retry")
            if any(item.get("error_type") == "EXTRA_ASPECT" for item in after.get("errors", [])):
                root_cause.append("false_mention_creation_or_over_split")
            if any(item.get("error_type") == "WRONG_ASPECT" for item in after.get("errors", [])):
                root_cause.append("taxonomy_boundary_or_relabel")
            if any(item.get("error_type") == "MISS_ASPECT" for item in after.get("errors", [])):
                root_cause.append("missed_or_failed_model_output")
            rows.append(
                {
                    "case_id": case_id,
                    "transition_types": ";".join(tags),
                    "focus": ";".join(focus),
                    "baseline_exact": before_exact,
                    "round1_exact": after_exact,
                    "baseline_errors": json.dumps(before.get("errors", []), ensure_ascii=False),
                    "round1_errors": json.dumps(after.get("errors", []), ensure_ascii=False),
                    "expected_aspects": json.dumps(before.get("expected_aspects", []), ensure_ascii=False),
                    "baseline_aspects": json.dumps(before.get("evaluation_aspects", []), ensure_ascii=False),
                    "round1_aspects": json.dumps(after.get("evaluation_aspects", []), ensure_ascii=False),
                    "root_cause": ";".join(root_cause),
                    "review": before.get("review", ""),
                }
            )

    analysis = {
        "evaluation_type": "step_8_12_1_round1_regression_analysis",
        "policy": "Offline comparison of frozen Step 8.11 and Step 8.12 Round 1 only.",
        "dataset_sha256": round1.get("dataset_sha256", ""),
        "baseline_summary": baseline.get("summary", {}),
        "round1_summary": round1.get("summary", {}),
        "baseline_error_counts": baseline.get("error_counts", {}),
        "round1_error_counts": round1.get("error_counts", {}),
        "case_transitions": dict(sorted(transitions.items())),
        "model_failure_case_ids": model_failures,
        "delivery_regression": {
            "metrics": {
                "baseline": calculate_aspect_metrics(baseline.get("cases", []), ("Delivery",))["Delivery"],
                "round1": calculate_aspect_metrics(round1.get("cases", []), ("Delivery",))["Delivery"],
            },
            "diff": _aspect_regression(baseline, round1, "Delivery"),
            "root_cause": (
                "Delivery TP and FN were unchanged; FP rose from 6 to 13. "
                "Round 1 created evaluative Delivery mentions from order/receipt, "
                "routing, inventory, and placement facts. The suggested labels "
                "survived normalization because there was no generic Delivery "
                "opinion-support guard. This was false mention creation and "
                "over-splitting, not Packaging/Customer Service/Reliability relabeling."
            ),
        },
        "setup_regression": {
            "metrics": {
                "baseline": calculate_aspect_metrics(baseline.get("cases", []), ("Setup / Installation",))["Setup / Installation"],
                "round1": calculate_aspect_metrics(round1.get("cases", []), ("Setup / Installation",))["Setup / Installation"],
            },
            "diff": _aspect_regression(baseline, round1, "Setup / Installation"),
            "root_cause": (
                "One correct Setup mention was lost to a final provider failure; "
                "one explicit 'not hard' setup judgment changed from positive to "
                "neutral; and one positive setup span gained a duplicate neutral "
                "mention. A first-time account-setup clarity claim remained "
                "misclassified as Ease of Use because no target-local initial-setup "
                "relabel corrected the model label."
            ),
        },
        "extra_aspect_diff": {
            "baseline_count": len(baseline_extras),
            "round1_count": len(round1_extras),
            "persistent_count": len(persistent),
            "fixed_count": len(fixed),
            "new_count": len(new),
            "persistent": persistent,
            "fixed": fixed,
            "new": new,
            "round1_tagged": tagged_extras,
            "root_cause_counts_nonexclusive": dict(root_counts.most_common()),
            "top_three_root_causes": [
                {"root_cause": label, "count": count}
                for label, count in root_counts.most_common(3)
            ],
        },
        "prompt_profiles": {
            "baseline": dict(BASELINE_PROMPT_PROFILE),
            "round1": dict(ROUND1_PROMPT_PROFILE),
            "round1_system_char_increase_percent": (
                (ROUND1_PROMPT_PROFILE["ollama_system_chars_with_schema"]
                 / BASELINE_PROMPT_PROFILE["ollama_system_chars_with_schema"] - 1)
                * 100
            ),
        },
        "regression_case_count": len(rows),
    }
    return analysis, rows


def save_round1_regression_analysis(
    analysis: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_path: str | Path = REGRESSION_ANALYSIS_PATH,
    cases_path: str | Path = REGRESSION_CASES_PATH,
) -> dict[str, str]:
    """Save the pre-repair analysis once; never overwrite it."""

    destinations = {"analysis": Path(analysis_path), "cases": Path(cases_path)}
    if any(path.exists() for path in destinations.values()):
        raise GeneralizationDevV2IntegrityError(
            "Round 1 regression analysis outputs already exist and will not be overwritten."
        )
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    destinations["analysis"].write_text(
        json.dumps(dict(analysis), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with destinations["cases"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGRESSION_CASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {key: str(path) for key, path in destinations.items()}


def generate_and_save_round1_regression_analysis() -> dict[str, str]:
    """Verify frozen inputs, build the diff, and save it without model work."""

    baseline, _cases, _hashes = verify_frozen_step_811_inputs()
    verify_frozen_round1_files()
    round1 = _load_json_object(ROUND1_PATHS["report"], label="Round 1 report")
    analysis, rows = build_round1_regression_analysis(baseline, round1)
    return save_round1_regression_analysis(analysis, rows)


class InstrumentedReviewClient:
    """Measure provider calls without changing their behavior or retry policy."""

    def __init__(self, delegate: ReviewAnalysisClient, clock: Callable[[], float] = time.perf_counter) -> None:
        self.delegate = delegate
        self.clock = clock
        self.by_review: dict[str, dict[str, Any]] = {}

    def _row(self, review_text: str) -> dict[str, Any]:
        return self.by_review.setdefault(
            review_text,
            {
                "analysis_call_count": 0,
                "repair_call_count": 0,
                "llm_analysis_seconds": 0.0,
                "llm_repair_seconds": 0.0,
                "generated_output_chars": 0,
                "generated_mentions": 0,
            },
        )

    def analyze_review(self, review_text: str, allowed_aspects: Sequence[str]):
        row = self._row(review_text)
        row["analysis_call_count"] += 1
        started = self.clock()
        try:
            result = self.delegate.analyze_review(review_text, allowed_aspects)
        finally:
            row["llm_analysis_seconds"] += max(0.0, self.clock() - started)
        if isinstance(result, str):
            row["generated_output_chars"] += len(result)
            try:
                payload = json.loads(result)
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = dict(result)
            row["generated_output_chars"] += len(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
        aspects = payload.get("aspects", []) if isinstance(payload, Mapping) else []
        row["generated_mentions"] += len(aspects) if isinstance(aspects, list) else 0
        return result

    def repair_evidence(self, review_text: str, invalid_aspects: Sequence[Mapping[str, Any]]):
        row = self._row(review_text)
        row["repair_call_count"] += 1
        started = self.clock()
        try:
            return self.delegate.repair_evidence(review_text, invalid_aspects)
        finally:
            row["llm_repair_seconds"] += max(0.0, self.clock() - started)


def _saved_prediction_proxy(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path, label=path.name)
    predictions = payload.get("predictions", [])
    successful = [
        item
        for item in predictions
        if isinstance(item, Mapping)
        and item.get("status") == "success"
        and isinstance(item.get("analysis"), Mapping)
    ]
    # Compare the same provider-owned fields measured by InstrumentedReviewClient;
    # positive_points/negative_points are Python-derived and are intentionally
    # excluded from all three versions.
    projected_outputs = [
        {
            "summary": item["analysis"].get("summary", ""),
            "aspects": item["analysis"].get("aspects", []),
        }
        for item in successful
    ]
    output_chars = [
        len(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
        for item in projected_outputs
    ]
    mentions = [len(item.get("aspects", [])) for item in projected_outputs]
    retried = [
        float(item.get("analysis_time_seconds", 0.0))
        for item in predictions
        if isinstance(item, Mapping) and item.get("retried")
    ]
    not_retried = [
        float(item.get("analysis_time_seconds", 0.0))
        for item in predictions
        if isinstance(item, Mapping) and not item.get("retried")
    ]
    return {
        "successful_outputs": len(successful),
        "average_structured_output_chars_proxy": statistics.mean(output_chars) if output_chars else 0.0,
        "average_structured_output_approx_tokens_chars_div_4": statistics.mean(output_chars) / 4 if output_chars else 0.0,
        "average_mentions": statistics.mean(mentions) if mentions else 0.0,
        "total_mentions": sum(mentions),
        "retry_case_count": len(retried),
        "retry_case_average_latency": statistics.mean(retried) if retried else 0.0,
        "non_retry_case_average_latency": statistics.mean(not_retried) if not_retried else 0.0,
    }


def _repair_runtime_analysis(
    report: Mapping[str, Any], instrumented: InstrumentedReviewClient
) -> dict[str, Any]:
    predictions = report.get("predictions", [])
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            continue
        review = str(prediction.get("review", ""))
        timing = instrumented.by_review.get(review, {})
        total = float(prediction.get("analysis_time_seconds", 0.0))
        llm = float(timing.get("llm_analysis_seconds", 0.0)) + float(
            timing.get("llm_repair_seconds", 0.0)
        )
        retry_wait = (
            TRANSIENT_RETRY_DELAY_SECONDS if prediction.get("retried") else 0.0
        )
        rows.append(
            {
                "case_id": prediction.get("case_id", ""),
                "status": prediction.get("status", ""),
                "retried": bool(prediction.get("retried")),
                "total_seconds": total,
                "llm_latency_seconds": llm,
                "retry_wait_seconds": retry_wait,
                "postprocess_latency_seconds": max(0.0, total - llm - retry_wait),
                **timing,
            }
        )
    def mean(key: str) -> float:
        values = [float(row.get(key, 0.0)) for row in rows]
        return statistics.mean(values) if values else 0.0
    retries = [row for row in rows if row["retried"]]
    no_retries = [row for row in rows if not row["retried"]]
    generated = [row for row in rows if row.get("generated_output_chars", 0)]
    return {
        "prompt_profiles": {
            "baseline": dict(BASELINE_PROMPT_PROFILE),
            "round1": dict(ROUND1_PROMPT_PROFILE),
            "repair": current_prompt_profile(),
        },
        "provider_configuration": {
            **current_provider_configuration(),
            "expected": dict(EXPECTED_PROVIDER_CONFIGURATION),
            "changed_from_baseline": (
                current_provider_configuration()
                != dict(EXPECTED_PROVIDER_CONFIGURATION)
            ),
        },
        "saved_output_proxies": {
            "baseline": _saved_prediction_proxy(Path(FROZEN_STEP_811_PATHS["baseline_predictions"])),
            "round1": _saved_prediction_proxy(Path(ROUND1_PATHS["predictions"])),
        },
        "repair": {
            "case_count": len(rows),
            "total_seconds": sum(float(row["total_seconds"]) for row in rows),
            "average_seconds_per_review": mean("total_seconds"),
            "average_llm_latency_seconds": mean("llm_latency_seconds"),
            "average_postprocess_latency_seconds": mean("postprocess_latency_seconds"),
            "retry_case_count": len(retries),
            "retry_case_average_latency": statistics.mean(
                float(row["total_seconds"]) for row in retries
            ) if retries else 0.0,
            "non_retry_case_average_latency": statistics.mean(
                float(row["total_seconds"]) for row in no_retries
            ) if no_retries else 0.0,
            "average_generated_output_chars": statistics.mean(
                float(row["generated_output_chars"]) for row in generated
            ) if generated else 0.0,
            "average_generated_output_approx_tokens_chars_div_4": statistics.mean(
                float(row["generated_output_chars"]) for row in generated
            ) / 4 if generated else 0.0,
            "average_generated_mentions": statistics.mean(
                float(row["generated_mentions"]) for row in generated
            ) if generated else 0.0,
        },
        "latency_attribution": (
            "Provider call timing is measured directly. Post-processing is the "
            "bounded remainder after subtracting provider time and the fixed "
            "single retry wait from the evaluator's per-case timer."
        ),
        "cases": rows,
    }


def repair_outputs_exist(output_dir: str | Path = REPAIR_OUTPUT_DIR) -> bool:
    root = Path(output_dir)
    return any(
        (root / name).exists()
        for name in (
            REPAIR_STARTED_PATH.name,
            REPAIR_REPORT_PATH.name,
            REPAIR_ERRORS_PATH.name,
            REPAIR_PREDICTIONS_PATH.name,
            REPAIR_RUNTIME_PATH.name,
        )
    )


def reserve_repair_evaluation(started_path: str | Path) -> Path:
    """Atomically and permanently reserve the one allowed real repair run."""

    destination = Path(started_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_type": "step_8_12_1_repair_run_reservation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": FROZEN_STEP_811_SHA256["dataset"],
        "provider_configuration": current_provider_configuration(),
        "policy": (
            "This marker is never removed. Its presence forbids a second repair "
            "evaluation even if the first process was interrupted."
        ),
    }
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
    except FileExistsError as exc:
        raise GeneralizationDevV2IntegrityError(
            "A Step 8.12.1 repair evaluation was already started."
        ) from exc
    return destination


def run_repair_candidate_evaluation(
    *,
    client: ReviewAnalysisClient,
    analyzer: Callable[..., Mapping[str, Any]] = analyze_review,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    progress: Callable[[int, int, str, str, float], None] | None = None,
    output_dir: str | Path = REPAIR_OUTPUT_DIR,
    regression_analysis_path: str | Path = REGRESSION_ANALYSIS_PATH,
    regression_cases_path: str | Path = REGRESSION_CASES_PATH,
    started_path: str | Path | None = None,
    protected_paths: Sequence[str | Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the single permitted repair candidate evaluation."""

    provider_configuration = verify_provider_configuration()
    root = Path(output_dir)
    reservation = Path(started_path) if started_path is not None else (
        root / REPAIR_STARTED_PATH.name
    )
    if repair_outputs_exist(root) or reservation.exists():
        raise GeneralizationDevV2IntegrityError(
            "A repair evaluation was already started; a second evaluation is forbidden."
        )
    if not Path(regression_analysis_path).is_file() or not Path(
        regression_cases_path
    ).is_file():
        raise GeneralizationDevV2IntegrityError(
            "The frozen Round 1 regression analysis must exist before repair evaluation."
        )
    baseline, _cases, frozen_before = verify_frozen_step_811_inputs()
    round1_hashes_before = verify_frozen_round1_files()
    round1 = _load_json_object(ROUND1_PATHS["report"], label="Round 1 report")
    guarded_paths = (
        tuple(protected_paths)
        if protected_paths is not None
        else protected_non_model_paths()
    )
    protected_before = snapshot_file_hashes(guarded_paths)
    reserve_repair_evaluation(reservation)
    instrumented = InstrumentedReviewClient(client, clock=clock)
    report = run_generalization_dev_v2_optimization_round(
        1,
        provider=LLM_PROVIDER,
        model=OLLAMA_MODEL,
        client=instrumented,
        analyzer=analyzer,
        clock=clock,
        sleeper=sleeper,
        progress=progress,
        prior_round_reports=(),
    )
    report["evaluation_type"] = "generalization_dev_v2_repair_candidate"
    report["evaluation_policy"] = (
        "Step 8.12.1 ran exactly one repair evaluation against Generalization Dev V2. "
        "Final Blind and Holdout were not rerun."
    )
    report["stage"] = "repair_candidate"
    report.pop("round_number", None)
    report.pop("stop_evaluation", None)
    report["comparison_to_round1"] = calculate_report_deltas(report, round1)
    report["baseline_deltas"] = calculate_report_deltas(report, baseline)
    runtime = _repair_runtime_analysis(report, instrumented)
    report["prompt_profiles"] = runtime["prompt_profiles"]
    report["repair_acceptance"] = repair_acceptance(report, round1)
    report["candidate_selection"] = select_production_candidate(
        baseline, round1, report
    )
    report["provider_configuration"] = provider_configuration
    report["protected_file_sha256_before"] = protected_before
    report["frozen_round1_sha256"] = round1_hashes_before
    report["frozen_step_811_sha256"] = frozen_before
    report["repair_run_reservation"] = str(reservation)
    report["protected_file_sha256_after"] = verify_file_hashes(
        protected_before, guarded_paths
    )
    if verify_frozen_round1_files() != round1_hashes_before:
        raise GeneralizationDevV2IntegrityError(
            "A frozen Round 1 artifact changed during repair evaluation."
        )
    return report, runtime


def repair_acceptance(
    report: Mapping[str, Any], round1: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the explicit Step 8.12.1 acceptance checks transparently."""

    summary = report.get("summary", {})
    errors = report.get("error_counts", {})
    priority = report.get("priority_aspect_metrics", {})
    round1_priority = round1.get("priority_aspect_metrics", {})
    stable = report.get("stable_aspect_metrics", {})
    runtime = report.get("runtime_statistics", {})
    round1_runtime = round1.get("runtime_statistics", {})
    delivery = stable.get("Delivery", {})
    setup = stable.get("Setup / Installation", {})
    design_f1 = float(priority.get("Design", {}).get("f1", 0.0))
    packaging_f1 = float(priority.get("Packaging", {}).get("f1", 0.0))
    price_f1 = float(priority.get("Price / Value", {}).get("f1", 0.0))
    round1_design_f1 = float(round1_priority.get("Design", {}).get("f1", 0.0))
    round1_packaging_f1 = float(
        round1_priority.get("Packaging", {}).get("f1", 0.0)
    )
    round1_price_f1 = float(
        round1_priority.get("Price / Value", {}).get("f1", 0.0)
    )
    failed_cases = int(runtime.get("failed_cases", 0))
    round1_failed_cases = int(round1_runtime.get("failed_cases", 0))
    average_runtime = float(runtime.get("average_seconds_per_review", 0.0))
    checks = {
        "overall_f1_at_least_76_5_percent": float(summary.get("aspect_f1", 0.0)) >= 0.765,
        "delivery_improved_over_round1": float(delivery.get("f1", 0.0)) > 0.5625,
        "setup_improved_over_round1": float(setup.get("f1", 0.0)) > 0.80,
        "extra_aspect_below_27": int(errors.get("EXTRA_ASPECT", 0)) < 27,
        "wrong_aspect_at_most_23": int(errors.get("WRONG_ASPECT", 0)) <= 23,
        "miss_aspect_at_most_12": int(errors.get("MISS_ASPECT", 0)) <= 12,
        "evidence_validity_100_percent": float(summary.get("evidence_validity_rate", 0.0)) == 1.0,
        "price_value_f1_at_least_72_percent": price_f1 >= 0.72,
        "price_value_drop_at_most_5pp": price_f1 + 1e-12 >= round1_price_f1 - 0.05,
        "design_drop_at_most_5pp": design_f1 + 1e-12 >= round1_design_f1 - 0.05,
        "packaging_drop_at_most_5pp": (
            packaging_f1 + 1e-12 >= round1_packaging_f1 - 0.05
        ),
        "average_runtime_below_15_seconds": average_runtime < 15.0,
        "model_failures_not_worse_than_round1": failed_cases <= round1_failed_cases,
        "zero_model_failures": failed_cases == 0,
    }
    return {
        "checks": checks,
        "passed_count": sum(checks.values()),
        "check_count": len(checks),
        "all_required_quality_checks_passed": all(
            checks[key]
            for key in (
                "overall_f1_at_least_76_5_percent",
                "delivery_improved_over_round1",
                "setup_improved_over_round1",
                "extra_aspect_below_27",
                "wrong_aspect_at_most_23",
                "miss_aspect_at_most_12",
                "evidence_validity_100_percent",
                "price_value_f1_at_least_72_percent",
                "price_value_drop_at_most_5pp",
                "design_drop_at_most_5pp",
                "packaging_drop_at_most_5pp",
                "average_runtime_below_15_seconds",
                "model_failures_not_worse_than_round1",
            )
        ),
        "round1_aspect_f1": round1.get("summary", {}).get("aspect_f1", 0.0),
    }


def select_production_candidate(
    baseline: Mapping[str, Any],
    round1: Mapping[str, Any],
    repair: Mapping[str, Any],
) -> dict[str, Any]:
    """Prefer balance and stable-Aspect safety before a small F1 difference."""

    baseline_stable = calculate_aspect_metrics(baseline.get("cases", []), STABLE_ASPECTS)
    entries: list[dict[str, Any]] = []
    for name, report in (
        ("step_8_11_baseline", baseline),
        ("step_8_12_round1", round1),
        ("step_8_12_1_repair", repair),
    ):
        metrics = calculate_aspect_metrics(report.get("cases", []), STABLE_ASPECTS)
        warnings = stable_aspect_regressions(metrics, baseline_stable)
        entries.append(
            {
                "candidate": name,
                "aspect_f1": float(report.get("summary", {}).get("aspect_f1", 0.0)),
                "evidence_validity": float(report.get("summary", {}).get("evidence_validity_rate", 0.0)),
                "average_runtime": float(report.get("summary", {}).get("average_analysis_time", 0.0)),
                "failed_cases": int(report.get("summary", {}).get("failed_cases", 0)),
                "error_counts": deepcopy(report.get("error_counts", {})),
                "stable_regressions": warnings,
            }
        )
    repair_entry = entries[-1]
    repair_balanced = (
        not repair_entry["stable_regressions"]
        and repair.get("repair_acceptance", {}).get(
            "all_required_quality_checks_passed", False
        )
    )
    if repair_balanced:
        selected = "step_8_12_1_repair"
        status = "generalization_v2_candidate"
        reason = "Repair met the quality gates without a major stable-Aspect regression."
    else:
        selected = "step_8_12_round1"
        status = "candidate_blocked_by_regression"
        reason = (
            "Repair did not clear every required balance gate. Round 1 remains the "
            "best optimization measurement, but production candidacy stays blocked."
        )
    return {
        "selection_priority": [
            "no_major_stable_aspect_regression",
            "overall_f1",
            "extra_wrong_miss_balance",
            "evidence",
            "runtime",
            "model_reliability",
        ],
        "candidates": entries,
        "selected": selected,
        "status": status,
        "reason": reason,
    }


def save_repair_candidate_outputs(
    report: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    output_dir: str | Path = REPAIR_OUTPUT_DIR,
) -> dict[str, str]:
    """Persist the sole repair run without replacing any previous artifact."""

    root = Path(output_dir)
    destinations = {
        "report": root / REPAIR_REPORT_PATH.name,
        "errors": root / REPAIR_ERRORS_PATH.name,
        "predictions": root / REPAIR_PREDICTIONS_PATH.name,
        "runtime": root / REPAIR_RUNTIME_PATH.name,
    }
    if any(path.exists() for path in destinations.values()):
        raise GeneralizationDevV2IntegrityError(
            "Repair candidate output already exists and will not be overwritten."
        )
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["outputs"] = {key: str(path) for key, path in destinations.items()}
    destinations["report"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with destinations["errors"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_error_rows(payload))
    prediction_payload = {
        "evaluation_policy": payload.get("evaluation_policy", ""),
        "stage": payload.get("stage", ""),
        "provider": payload.get("provider", ""),
        "model": payload.get("model", ""),
        "dataset_sha256": payload.get("dataset_sha256", ""),
        "timestamp": payload.get("timestamp", ""),
        "case_count": payload.get("case_count", 0),
        "predictions": payload.get("predictions", []),
    }
    destinations["predictions"].write_text(
        json.dumps(prediction_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destinations["runtime"].write_text(
        json.dumps(dict(runtime), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {key: str(path) for key, path in destinations.items()}


def format_repair_candidate_report(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    runtime = report.get("runtime_statistics", {})
    errors = report.get("error_counts", {})
    selection = report.get("candidate_selection", {})
    return "\n".join(
        [
            "Generalization Dev V2 — Step 8.12.1 Repair Candidate",
            f"Aspect P/R/F1: {float(summary.get('aspect_precision', 0.0)):.2%} / "
            f"{float(summary.get('aspect_recall', 0.0)):.2%} / "
            f"{float(summary.get('aspect_f1', 0.0)):.2%}",
            f"Aspect-only F1: {float(summary.get('aspect_only_f1', 0.0)):.2%}",
            f"Evidence Validity: {float(summary.get('evidence_validity_rate', 0.0)):.2%}",
            f"Strict Exact: {int(summary.get('exact_matches', 0))}/"
            f"{int(summary.get('total_cases', 0))} "
            f"({float(summary.get('exact_case_accuracy', 0.0)):.2%})",
            "TP / FP / FN: "
            f"{int(summary.get('aspect_true_positives', 0))} / "
            f"{int(summary.get('aspect_false_positives', 0))} / "
            f"{int(summary.get('aspect_false_negatives', 0))}",
            "WRONG / EXTRA / MISS: "
            f"{int(errors.get('WRONG_ASPECT', 0))} / "
            f"{int(errors.get('EXTRA_ASPECT', 0))} / "
            f"{int(errors.get('MISS_ASPECT', 0))}",
            f"Runtime: {float(runtime.get('average_seconds_per_review', 0.0)):.2f}s/review",
            f"Candidate status: {selection.get('status', 'unknown')}",
            "Only one repair evaluation was performed.",
            "Final Blind was not rerun.",
        ]
    )


__all__ = [
    "BASELINE_PROMPT_PROFILE",
    "EXPECTED_PROVIDER_CONFIGURATION",
    "FROZEN_ROUND1_SHA256",
    "REGRESSION_ANALYSIS_PATH",
    "REGRESSION_CASES_PATH",
    "REPAIR_ERRORS_PATH",
    "REPAIR_OUTPUT_DIR",
    "REPAIR_PREDICTIONS_PATH",
    "REPAIR_REPORT_PATH",
    "REPAIR_RUNTIME_PATH",
    "REPAIR_STARTED_PATH",
    "ROUND1_PROMPT_PROFILE",
    "ROUND1_PATHS",
    "InstrumentedReviewClient",
    "build_round1_regression_analysis",
    "current_prompt_profile",
    "current_provider_configuration",
    "format_repair_candidate_report",
    "generate_and_save_round1_regression_analysis",
    "repair_acceptance",
    "repair_outputs_exist",
    "protected_non_model_paths",
    "reserve_repair_evaluation",
    "run_repair_candidate_evaluation",
    "save_repair_candidate_outputs",
    "save_round1_regression_analysis",
    "select_production_candidate",
    "verify_frozen_round1_files",
    "verify_provider_configuration",
]
