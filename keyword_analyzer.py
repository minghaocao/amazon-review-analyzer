"""Step 7: deterministic keyword and phrase statistics.

The module never calls an LLM and never mutates the analyzed-review input. The
overall corpus comes from successful reviews' ``cleaned_content``. Sentiment
and Aspect corpora come only from the already validated ``aspects.evidence``
records, so no new sentiment or taxonomy decisions are introduced here.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable

import pandas as pd

from aspect_taxonomy import ASPECT_ALIASES, STANDARD_ASPECTS
from config import (
    ANALYZED_REVIEWS_PATH,
    ASPECT_KEYWORDS_PATH,
    KEYWORD_MIN_REVIEW_COUNT,
    KEYWORD_SUMMARY_PATH,
    SENTIMENT_KEYWORDS_PATH,
)


TERM_TYPES = ("unigram", "bigram", "trigram")
KEYWORD_SUMMARY_COLUMNS = (
    "term",
    "term_type",
    "count",
    "review_count",
    "review_share",
    "review_refs",
)
REQUIRED_ANALYZED_COLUMNS = (
    "source_row_id",
    "review_id",
    "analysis_status",
    "cleaned_content",
    "aspects",
)

# Local stopwords keep Step 7 dependency-free. Negations are removed explicitly
# below because they materially change meaning.
BASE_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "although",
    "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "of",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}
NEGATIONS = {"not", "no", "never"}
BASE_STOPWORDS.difference_update(NEGATIONS)

# Low-information ecommerce/review words are centralized for easy tuning. The
# final terms remove metadata-like prose present in this project's deliberately
# invalid-rating fixture rows; no source text is changed.
DOMAIN_STOPWORDS = {
    "product", "item", "amazon", "buy", "buys", "buying", "bought",
    "purchase", "purchased", "thing", "things", "stuff", "really", "also",
    "tv", "television", "review", "row", "intentionally", "use", "uses",
    "cover", "covers",
    "rating", "ratings", "valid", "star", "stars", "maximum", "minimum",
}

# These retain value inside phrases/Evidence (``not good``, ``picture is
# excellent``) but are too generic for an overall unigram business ranking.
GENERIC_SENTIMENT_UNIGRAMS = {
    "amazing", "average", "awesome", "bad", "decent", "excellent", "fine",
    "good", "great", "nice", "perfect", "poor", "satisfactory", "terrible",
    "wonderful",
}
LOW_INFORMATION_UNIGRAMS = {
    "arrive", "arrived", "day", "days", "feel", "feels", "hour", "hours",
    "look", "looks", "minute", "minutes", "room", "second", "seconds",
}
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve",
}
GENERIC_PHRASE_CONTENT = GENERIC_SENTIMENT_UNIGRAMS | {
    "like", "liked", "love", "loved",
}
TIME_AND_MEASURE_UNITS = {
    "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days",
    "week", "weeks", "month", "months", "year", "years", "inch", "inches",
    "gb", "tb", "hz", "watts", "watt", "percent",
}
BLOCKED_PHRASES = {"very good", "this product", "really like", "room well"}
INCOMPLETE_PHRASE_ENDINGS = {
    "arrive", "arrived", "cover", "covers", "deep", "every", "feel", "feels",
    "few", "fits", "four", "less", "look", "looks", "major", "makes", "medium",
    "normal", "one", "poorly", "several", "set", "slightly", "sometimes", "ten",
    "three", "took", "two", "uses",
}

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?")
_SEGMENT_BOUNDARY_RE = re.compile(r"[.!?;,:\n\r\u2026]+")
_HYPHEN_RE = re.compile(r"(?<=[a-z0-9])[\-\u2010-\u2015](?=[a-z0-9])")


class KeywordAnalysisError(ValueError):
    """The analyzed-review input cannot be processed safely."""


@dataclass(frozen=True)
class KeywordAnalysisResult:
    """All Step 7 derived outputs before persistence."""

    keyword_summary: pd.DataFrame
    sentiment_keywords: dict[str, list[dict[str, Any]]]
    aspect_keywords: dict[str, list[dict[str, Any]]]
    successful_reviews: int
    issues: list[dict[str, Any]]


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _python_scalar(value: object) -> object:
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    return value


def _normalize_source_id(value: object) -> int | str:
    value = _python_scalar(value)
    if value is None or not str(value).strip():
        raise KeywordAnalysisError("source_row_id contains missing values")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        number = int(text)
    except ValueError:
        return text
    return number if str(number) == text else text


def _normalize_review_id(value: object) -> str | None:
    value = _python_scalar(value)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalize_text(text: object) -> str:
    """Normalize text for statistics without changing the stored source."""

    if text is None or _is_missing(text):
        return ""
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = re.sub(r"\bwi[\s\-\u2010-\u2015]*fi\b", "wifi", value)
    # Preserve explicit negation when expanding common contractions.
    value = re.sub(r"\bwon't\b", "will not", value)
    value = re.sub(r"\bcan't\b", "can not", value)
    value = re.sub(r"n['’]t\b", " not", value)
    value = re.sub(r"\b(it|that|there|he|she|what)['’]s\b", r"\1 is", value)
    value = re.sub(r"\b([a-z0-9]+)['’]s\b", r"\1", value)
    value = _HYPHEN_RE.sub(" ", value)
    return value


def _tokenize_segments(text: object) -> list[list[str]]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    segments: list[list[str]] = []
    for segment in _SEGMENT_BOUNDARY_RE.split(normalized):
        tokens = _WORD_RE.findall(segment)
        if tokens:
            segments.append(tokens)
    return segments


def clean_for_keywords(text: object) -> str:
    """Return lowercase tokenized text used only by Step 7 statistics.

    Stopwords stay in this intermediate text so meaningful phrases such as
    ``easy to use`` remain intact. They are filtered when terms are selected.
    ``None`` and pandas/NumPy missing values return an empty string.
    """

    return " ".join(
        token
        for segment in _tokenize_segments(text)
        for token in segment
    )


def _normalized_phrase(value: str) -> str:
    return clean_for_keywords(value)


KNOWN_ATTRIBUTE_PHRASES = {
    phrase
    for aliases in ASPECT_ALIASES.values()
    for alias in aliases
    if (phrase := _normalized_phrase(alias))
}


def _is_numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _is_overall_unigram(token: str) -> bool:
    if _is_numeric(token) or len(token) < 2:
        return False
    if token in BASE_STOPWORDS or token in DOMAIN_STOPWORDS:
        return False
    return (
        token not in GENERIC_SENTIMENT_UNIGRAMS
        and token not in LOW_INFORMATION_UNIGRAMS
        and token not in NUMBER_WORDS
    )


def _is_evidence_unigram(token: str) -> bool:
    if _is_numeric(token) or len(token) < 2:
        return False
    return (
        token not in BASE_STOPWORDS
        and token not in DOMAIN_STOPWORDS
        and token not in LOW_INFORMATION_UNIGRAMS
        and token not in NUMBER_WORDS
    )


def _is_informative_phrase(tokens: tuple[str, ...]) -> bool:
    phrase = " ".join(tokens)
    if phrase in BLOCKED_PHRASES or all(_is_numeric(token) for token in tokens):
        return False

    # Known taxonomy phrases and two intentionally stopword-bearing business
    # expressions are safe before applying boundary rules.
    if phrase in KNOWN_ATTRIBUTE_PHRASES or phrase in {
        "easy to use",
        "hard to use",
        "difficult to use",
        "value for money",
    }:
        return True

    stopwords = BASE_STOPWORDS | DOMAIN_STOPWORDS
    if tokens[0] in stopwords or tokens[-1] in stopwords:
        return False
    if any(token in {"and", "but", "although"} for token in tokens[1:-1]):
        return False
    if tokens[-1] in INCOMPLETE_PHRASE_ENDINGS or _is_numeric(tokens[-1]):
        return False
    content = [
        token
        for token in tokens
        if token not in stopwords and not _is_numeric(token)
    ]
    has_contextual_number = (
        any(_is_numeric(token) for token in tokens)
        and any(token in TIME_AND_MEASURE_UNITS for token in tokens)
    )
    if has_contextual_number and content:
        return True
    if len(content) < 2:
        return False
    if set(content).issubset(GENERIC_PHRASE_CONTENT):
        return False
    return True


def _extract_term_counts(
    text: object,
    *,
    evidence: bool = False,
) -> Counter[tuple[str, str]]:
    """Extract filtered terms without crossing punctuation boundaries."""

    segments = _tokenize_segments(text)
    counts: Counter[tuple[str, str]] = Counter()
    unigram_filter = _is_evidence_unigram if evidence else _is_overall_unigram
    for tokens in segments:
        counts.update(
            ("unigram", token)
            for token in tokens
            if unigram_filter(token)
        )
        for size, term_type in ((2, "bigram"), (3, "trigram")):
            for position in range(len(tokens) - size + 1):
                gram = tuple(tokens[position : position + size])
                if _is_informative_phrase(gram):
                    counts[(term_type, " ".join(gram))] += 1
    return counts


class _TermAccumulator:
    """Track occurrences and distinct source rows for one corpus."""

    def __init__(self) -> None:
        self.counts: Counter[tuple[str, str]] = Counter()
        self.sources: dict[tuple[str, str], set[Hashable]] = defaultdict(set)
        self.references: dict[Hashable, dict[str, Any]] = {}

    def add(
        self,
        term_counts: Counter[tuple[str, str]],
        *,
        source_key: Hashable,
        source_row_id: int | str,
        review_id: str | None,
    ) -> None:
        self.references[source_key] = {
            "source_row_id": source_row_id,
            "review_id": review_id,
        }
        for key, count in term_counts.items():
            self.counts[key] += count
            self.sources[key].add(source_key)

    def records(
        self,
        *,
        min_review_count: int,
        successful_reviews: int,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for (term_type, term), count in self.counts.items():
            source_keys = self.sources[(term_type, term)]
            review_count = len(source_keys)
            if review_count < min_review_count:
                continue
            refs = [self.references[key] for key in source_keys]
            refs.sort(
                key=lambda ref: (
                    0 if isinstance(ref["source_row_id"], int) else 1,
                    ref["source_row_id"]
                    if isinstance(ref["source_row_id"], int)
                    else str(ref["source_row_id"]),
                    str(ref["review_id"]),
                )
            )
            records.append(
                {
                    "term": term,
                    "term_type": term_type,
                    "count": int(count),
                    "review_count": review_count,
                    "review_share": round(
                        review_count / successful_reviews * 100,
                        4,
                    ) if successful_reviews else 0.0,
                    "review_refs": refs,
                }
            )
        type_order = {term_type: index for index, term_type in enumerate(TERM_TYPES)}
        records.sort(
            key=lambda record: (
                -record["review_count"],
                -record["count"],
                type_order[record["term_type"]],
                record["term"],
            )
        )
        return records


def _validate_analyzed_reviews(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = [column for column in REQUIRED_ANALYZED_COLUMNS if column not in df.columns]
    if missing:
        raise KeywordAnalysisError(
            f"Missing analyzed review columns: {', '.join(missing)}"
        )
    normalized_ids = df["source_row_id"].map(_normalize_source_id)
    if normalized_ids.astype(str).duplicated().any():
        raise KeywordAnalysisError("source_row_id must be unique")


def load_keyword_reviews(
    file_path: str | Path = ANALYZED_REVIEWS_PATH,
) -> pd.DataFrame:
    """Load the formal analyzed output without consulting batch checkpoints."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Analyzed reviews file not found: {path}")
    try:
        frame = pd.read_csv(
            path,
            dtype="string",
            keep_default_na=False,
            encoding="utf-8",
        )
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise KeywordAnalysisError(
            f"Analyzed reviews CSV could not be read: {exc}"
        ) from exc
    _validate_analyzed_reviews(frame)
    return frame


def _issue(row: pd.Series, error: str) -> dict[str, Any]:
    return {
        "source_row_id": _normalize_source_id(row["source_row_id"]),
        "review_id": _normalize_review_id(row["review_id"]),
        "column": "aspects",
        "error": error,
    }


def _parse_aspects(
    row: pd.Series,
    issues: list[dict[str, Any]],
) -> list[dict[str, str]]:
    value = row["aspects"]
    if isinstance(value, list):
        decoded = value
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            issues.append(_issue(row, f"invalid JSON: {exc.msg}"))
            return []
    else:
        issues.append(_issue(row, "missing JSON array"))
        return []
    if not isinstance(decoded, list):
        issues.append(_issue(row, "JSON value must be an array"))
        return []

    valid: list[dict[str, str]] = []
    for position, entry in enumerate(decoded):
        if not isinstance(entry, dict):
            issues.append(_issue(row, f"item {position} must be an object"))
            continue
        aspect = entry.get("aspect")
        sentiment = entry.get("sentiment")
        evidence = entry.get("evidence")
        normalized_sentiment = (
            sentiment.strip().casefold() if isinstance(sentiment, str) else ""
        )
        if not isinstance(aspect, str) or aspect not in STANDARD_ASPECTS:
            issues.append(_issue(row, f"item {position} has unknown aspect"))
            continue
        if normalized_sentiment not in {"positive", "negative", "neutral"}:
            issues.append(_issue(row, f"item {position} has unsupported sentiment"))
            continue
        if not isinstance(evidence, str) or not evidence.strip():
            issues.append(_issue(row, f"item {position} has no evidence"))
            continue
        valid.append({
            "aspect": aspect,
            "sentiment": normalized_sentiment,
            "evidence": evidence,
        })
    return valid


def _to_json_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "term": record["term"],
        "term_type": record["term_type"],
        "count": record["count"],
        "review_count": record["review_count"],
        "review_share": record["review_share"],
        "review_refs": record["review_refs"],
    }


def analyze_keywords(
    df: pd.DataFrame,
    *,
    min_review_count: int = KEYWORD_MIN_REVIEW_COUNT,
) -> KeywordAnalysisResult:
    """Build overall, sentiment, and Aspect keyword statistics.

    ``count`` measures occurrences. ``review_count`` measures unique source
    rows, not business ``review_id`` values, because those may be duplicated.
    """

    if isinstance(min_review_count, bool) or not isinstance(min_review_count, int):
        raise TypeError("min_review_count must be an integer")
    if min_review_count <= 0:
        raise ValueError("min_review_count must be a positive integer")
    _validate_analyzed_reviews(df)

    successful = df.loc[
        df["analysis_status"].astype(str).str.strip().eq("success")
    ]
    successful_reviews = len(successful)
    overall = _TermAccumulator()
    sentiment = {
        "positive": _TermAccumulator(),
        "negative": _TermAccumulator(),
    }
    aspects: dict[str, _TermAccumulator] = {}
    issues: list[dict[str, Any]] = []

    for _, row in successful.iterrows():
        source_row_id = _normalize_source_id(row["source_row_id"])
        source_key: Hashable = (type(source_row_id).__name__, source_row_id)
        review_id = _normalize_review_id(row["review_id"])
        overall.add(
            _extract_term_counts(row["cleaned_content"]),
            source_key=source_key,
            source_row_id=source_row_id,
            review_id=review_id,
        )

        seen_sentiment_evidence: set[tuple[str, str]] = set()
        seen_aspect_evidence: set[tuple[str, str, str]] = set()
        for entry in _parse_aspects(row, issues):
            aspect = entry["aspect"]
            sentiment_name = entry["sentiment"]
            evidence = entry["evidence"]
            evidence_key = clean_for_keywords(evidence)
            if not evidence_key:
                continue
            term_counts = _extract_term_counts(evidence, evidence=True)

            aspect_key = (aspect, sentiment_name, evidence_key)
            if aspect_key not in seen_aspect_evidence:
                seen_aspect_evidence.add(aspect_key)
                aspects.setdefault(aspect, _TermAccumulator()).add(
                    term_counts,
                    source_key=source_key,
                    source_row_id=source_row_id,
                    review_id=review_id,
                )

            if sentiment_name in sentiment:
                sentiment_key = (sentiment_name, evidence_key)
                if sentiment_key not in seen_sentiment_evidence:
                    seen_sentiment_evidence.add(sentiment_key)
                    sentiment[sentiment_name].add(
                        term_counts,
                        source_key=source_key,
                        source_row_id=source_row_id,
                        review_id=review_id,
                    )

    overall_records = overall.records(
        min_review_count=min_review_count,
        successful_reviews=successful_reviews,
    )
    keyword_rows = [
        {
            **{key: record[key] for key in KEYWORD_SUMMARY_COLUMNS[:-1]},
            "review_refs": json.dumps(
                record["review_refs"], ensure_ascii=False, separators=(",", ":")
            ),
        }
        for record in overall_records
    ]
    keyword_summary = pd.DataFrame(keyword_rows, columns=KEYWORD_SUMMARY_COLUMNS)

    sentiment_keywords = {
        name: [
            _to_json_record(record)
            for record in accumulator.records(
                min_review_count=min_review_count,
                successful_reviews=successful_reviews,
            )
        ]
        for name, accumulator in sentiment.items()
    }
    aspect_keywords = {
        aspect: [
            _to_json_record(record)
            for record in accumulator.records(
                min_review_count=min_review_count,
                successful_reviews=successful_reviews,
            )
        ]
        for aspect, accumulator in sorted(aspects.items())
    }

    return KeywordAnalysisResult(
        keyword_summary=keyword_summary,
        sentiment_keywords=sentiment_keywords,
        aspect_keywords=aspect_keywords,
        successful_reviews=successful_reviews,
        issues=issues,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_keyword_outputs(
    result: KeywordAnalysisResult,
    *,
    keyword_summary_path: str | Path = KEYWORD_SUMMARY_PATH,
    sentiment_keywords_path: str | Path = SENTIMENT_KEYWORDS_PATH,
    aspect_keywords_path: str | Path = ASPECT_KEYWORDS_PATH,
) -> dict[str, Path]:
    """Atomically save the three Step 7 outputs."""

    if result.successful_reviews <= 0:
        raise KeywordAnalysisError(
            "No successfully analyzed reviews are available for keyword analysis"
        )
    paths = {
        "keyword_summary": Path(keyword_summary_path),
        "sentiment_keywords": Path(sentiment_keywords_path),
        "aspect_keywords": Path(aspect_keywords_path),
    }
    _atomic_write_csv(paths["keyword_summary"], result.keyword_summary)
    _atomic_write_text(
        paths["sentiment_keywords"],
        json.dumps(result.sentiment_keywords, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        paths["aspect_keywords"],
        json.dumps(result.aspect_keywords, ensure_ascii=False, indent=2) + "\n",
    )
    return paths
