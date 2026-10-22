"""Offline synthetic tests for Step 7 keyword and phrase statistics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from keyword_analyzer import (  # noqa: E402
    KEYWORD_SUMMARY_COLUMNS,
    KeywordAnalysisError,
    analyze_keywords,
    clean_for_keywords,
    load_keyword_reviews,
    save_keyword_outputs,
)


def _aspect(aspect: str, sentiment: str, evidence: str) -> dict[str, str]:
    return {
        "raw_aspect": aspect,
        "aspect": aspect,
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _row(
    source_row_id: int,
    review_id: str,
    content: str,
    *,
    status: str = "success",
    aspects: object | None = None,
) -> dict[str, object]:
    aspect_value = [] if aspects is None else aspects
    return {
        "source_row_id": source_row_id,
        "review_id": review_id,
        "analysis_status": status,
        "cleaned_content": content,
        "aspects": (
            json.dumps(aspect_value)
            if not isinstance(aspect_value, str)
            else aspect_value
        ),
    }


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(
                0,
                "R001",
                "Picture quality picture quality is excellent. Wi-Fi is fast.",
                aspects=[
                    _aspect(
                        "Picture Quality", "positive", "Picture quality is excellent"
                    ),
                    _aspect("Connectivity", "positive", "Wi-Fi is fast"),
                ],
            ),
            _row(
                1,
                "R002",
                "Amazon product picture quality is clear. WiFi is never slow.",
                aspects=[
                    _aspect("Picture Quality", "positive", "picture quality is clear"),
                    _aspect("Connectivity", "positive", "WiFi is never slow"),
                ],
            ),
            _row(
                2,
                "R003",
                "The remote-control feels cheap. Not good.",
                aspects=[
                    _aspect("Remote Control", "negative", "remote-control feels cheap"),
                    _aspect("Other", "negative", "Not good"),
                ],
            ),
            _row(
                3,
                "R004",
                "No sound. Sound quality is terrible and never clear.",
                aspects=[
                    _aspect("Sound Quality", "negative", "No sound"),
                    _aspect("Sound Quality", "negative", "Sound quality is terrible"),
                    _aspect("Sound Quality", "negative", "never clear"),
                ],
            ),
            _row(
                4,
                "R005",
                "Sound quality is average. Easy to use.",
                aspects=[
                    _aspect("Sound Quality", "neutral", "Sound quality is average"),
                    _aspect("Ease of Use", "positive", "Easy to use"),
                ],
            ),
            _row(
                5,
                "R006",
                "Battery lasts 2 hours. Setup takes 10 minutes.",
                aspects=[
                    _aspect("Reliability", "negative", "Battery lasts 2 hours"),
                    _aspect("Setup / Installation", "positive", "Setup takes 10 minutes"),
                ],
            ),
            _row(
                6,
                "R007",
                "picture quality wifi phantom phantom",
                status="failed",
                aspects=[_aspect("Picture Quality", "positive", "picture quality")],
            ),
            _row(
                7,
                "R008",
                "picture quality wifi phantom phantom",
                status="skipped",
                aspects=[_aspect("Picture Quality", "positive", "picture quality")],
            ),
        ]
    )


def _summary_record(
    result: object,
    term_type: str,
    term: str,
) -> dict[str, object]:
    frame = result.keyword_summary  # type: ignore[attr-defined]
    rows = frame.loc[
        frame["term_type"].eq(term_type) & frame["term"].eq(term)
    ]
    if rows.empty:
        raise AssertionError(f"Missing {term_type}: {term}")
    return rows.iloc[0].to_dict()


def _json_record(
    records: list[dict[str, object]],
    term_type: str,
    term: str,
) -> dict[str, object]:
    for record in records:
        if record["term_type"] == term_type and record["term"] == term:
            return record
    raise AssertionError(f"Missing {term_type}: {term}")


class KeywordCleaningTests(unittest.TestCase):
    def test_clean_for_keywords_lowercases_and_removes_punctuation(self) -> None:
        self.assertEqual("picture quality", clean_for_keywords(" Picture, QUALITY! "))

    def test_none_and_nan_are_safe(self) -> None:
        self.assertEqual("", clean_for_keywords(None))
        self.assertEqual("", clean_for_keywords(float("nan")))

    def test_wifi_spellings_are_normalized(self) -> None:
        self.assertEqual("wifi wifi wifi", clean_for_keywords("Wi-Fi WiFi wi fi"))

    def test_hyphenated_words_become_token_sequence(self) -> None:
        self.assertEqual("remote control", clean_for_keywords("remote-control"))

    def test_negative_contraction_is_preserved(self) -> None:
        self.assertEqual("sound is not good", clean_for_keywords("Sound isn't good"))


class OverallKeywordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _fixture()
        self.result = analyze_keywords(self.frame, min_review_count=1)

    def test_successful_review_count(self) -> None:
        self.assertEqual(6, self.result.successful_reviews)

    def test_unigram_occurrence_count(self) -> None:
        self.assertEqual(5, _summary_record(self.result, "unigram", "quality")["count"])

    def test_unigram_review_count(self) -> None:
        self.assertEqual(
            4,
            _summary_record(self.result, "unigram", "quality")["review_count"],
        )

    def test_same_review_repetition_does_not_inflate_review_count(self) -> None:
        picture = _summary_record(self.result, "unigram", "picture")
        self.assertEqual(3, picture["count"])
        self.assertEqual(2, picture["review_count"])

    def test_review_share_uses_all_successful_reviews(self) -> None:
        self.assertAlmostEqual(
            66.6667,
            float(_summary_record(self.result, "unigram", "quality")["review_share"]),
            places=4,
        )

    def test_bigram_is_generated_from_original_sequence(self) -> None:
        phrase = _summary_record(self.result, "bigram", "picture quality")
        self.assertEqual((3, 2), (phrase["count"], phrase["review_count"]))

    def test_trigram_preserves_internal_stopword(self) -> None:
        phrase = _summary_record(self.result, "trigram", "easy to use")
        self.assertEqual((1, 1), (phrase["count"], phrase["review_count"]))

    def test_ngrams_do_not_cross_sentence_boundaries(self) -> None:
        terms = set(self.result.keyword_summary["term"])
        self.assertNotIn("excellent wifi", terms)
        self.assertNotIn("fast amazon", terms)
        self.assertNotIn("good no", terms)

    def test_common_stopwords_are_removed_from_unigrams(self) -> None:
        unigrams = set(
            self.result.keyword_summary.loc[
                self.result.keyword_summary["term_type"].eq("unigram"), "term"
            ]
        )
        self.assertTrue({"the", "is", "to"}.isdisjoint(unigrams))

    def test_not_is_retained(self) -> None:
        _summary_record(self.result, "unigram", "not")
        _summary_record(self.result, "bigram", "not good")

    def test_no_is_retained(self) -> None:
        _summary_record(self.result, "unigram", "no")
        _summary_record(self.result, "bigram", "no sound")

    def test_never_is_retained(self) -> None:
        record = _summary_record(self.result, "unigram", "never")
        self.assertEqual((2, 2), (record["count"], record["review_count"]))

    def test_domain_stopwords_do_not_rank(self) -> None:
        unigrams = set(
            self.result.keyword_summary.loc[
                self.result.keyword_summary["term_type"].eq("unigram"), "term"
            ]
        )
        self.assertTrue({"amazon", "product", "item", "buy"}.isdisjoint(unigrams))

    def test_low_value_phrases_are_filtered(self) -> None:
        frame = pd.DataFrame(
            [_row(0, "R1", "Very good. This product. Really like it.")]
        )
        terms = set(analyze_keywords(frame, min_review_count=1).keyword_summary["term"])
        self.assertTrue({"very good", "this product", "really like"}.isdisjoint(terms))

    def test_wifi_normalization_combines_reviews(self) -> None:
        wifi = _summary_record(self.result, "unigram", "wifi")
        self.assertEqual((2, 2), (wifi["count"], wifi["review_count"]))

    def test_remote_control_normalization_supports_bigram(self) -> None:
        _summary_record(self.result, "bigram", "remote control")

    def test_contextual_numbers_are_kept_in_phrases(self) -> None:
        _summary_record(self.result, "bigram", "2 hours")
        _summary_record(self.result, "trigram", "lasts 2 hours")
        _summary_record(self.result, "bigram", "10 minutes")

    def test_standalone_numbers_are_not_unigrams(self) -> None:
        frame = pd.DataFrame([_row(0, "R1", "Model 2026 code 12345")])
        result = analyze_keywords(frame, min_review_count=1)
        self.assertTrue({"2026", "12345"}.isdisjoint(set(result.keyword_summary["term"])))

    def test_failed_review_is_excluded(self) -> None:
        self.assertNotIn("phantom", set(self.result.keyword_summary["term"]))

    def test_skipped_review_is_excluded(self) -> None:
        picture = _summary_record(self.result, "unigram", "picture")
        self.assertEqual(3, picture["count"])

    def test_keyword_records_include_review_references(self) -> None:
        record = _summary_record(self.result, "bigram", "picture quality")
        refs = json.loads(str(record["review_refs"]))
        self.assertEqual([0, 1], [ref["source_row_id"] for ref in refs])
        self.assertEqual(["R001", "R002"], [ref["review_id"] for ref in refs])

    def test_duplicate_review_id_different_source_rows_count_separately(self) -> None:
        frame = pd.DataFrame(
            [_row(0, "R009", "clear screen"), _row(1, "R009", "clear screen")]
        )
        result = analyze_keywords(frame, min_review_count=1)
        self.assertEqual(2, _summary_record(result, "bigram", "clear screen")["review_count"])

    def test_minimum_review_threshold_applies(self) -> None:
        result = analyze_keywords(self.frame, min_review_count=2)
        self.assertIn("picture quality", set(result.keyword_summary["term"]))
        self.assertNotIn("remote control", set(result.keyword_summary["term"]))

    def test_sorting_prefers_review_count_then_occurrences(self) -> None:
        frame = pd.DataFrame(
            [
                _row(0, "R1", "alpha charlie charlie"),
                _row(1, "R2", "alpha charlie"),
                _row(2, "R3", "bravo bravo bravo bravo bravo"),
            ]
        )
        result = analyze_keywords(frame, min_review_count=1)
        unigrams = result.keyword_summary.loc[
            result.keyword_summary["term_type"].eq("unigram"), "term"
        ].tolist()
        self.assertEqual(["charlie", "alpha", "bravo"], unigrams[:3])

    def test_input_dataframe_is_not_modified(self) -> None:
        original = self.frame.copy(deep=True)
        analyze_keywords(self.frame, min_review_count=1)
        assert_frame_equal(original, self.frame)


class EvidenceKeywordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = analyze_keywords(_fixture(), min_review_count=1)

    def test_positive_evidence_populates_positive_corpus(self) -> None:
        record = _json_record(
            self.result.sentiment_keywords["positive"],
            "bigram",
            "picture quality",
        )
        self.assertEqual((2, 2), (record["count"], record["review_count"]))

    def test_negative_evidence_populates_negative_corpus(self) -> None:
        record = _json_record(
            self.result.sentiment_keywords["negative"], "unigram", "sound"
        )
        self.assertEqual((2, 1), (record["count"], record["review_count"]))

    def test_neutral_evidence_is_not_in_sentiment_corpora(self) -> None:
        positive = {(record["term_type"], record["term"]) for record in self.result.sentiment_keywords["positive"]}
        negative = {(record["term_type"], record["term"]) for record in self.result.sentiment_keywords["negative"]}
        self.assertNotIn(("bigram", "sound quality"), positive)
        self.assertEqual(1, _json_record(self.result.sentiment_keywords["negative"], "bigram", "sound quality")["count"])

    def test_aspect_keywords_follow_existing_aspect(self) -> None:
        picture = _json_record(
            self.result.aspect_keywords["Picture Quality"],
            "bigram",
            "picture quality",
        )
        self.assertEqual((2, 2), (picture["count"], picture["review_count"]))

    def test_neutral_evidence_remains_in_aspect_corpus(self) -> None:
        sound = _json_record(
            self.result.aspect_keywords["Sound Quality"], "unigram", "sound"
        )
        self.assertEqual((3, 2), (sound["count"], sound["review_count"]))

    def test_duplicate_identical_evidence_is_not_double_counted(self) -> None:
        entry = _aspect("Picture Quality", "positive", "picture quality is clear")
        frame = pd.DataFrame([_row(0, "R1", "picture quality is clear", aspects=[entry, entry])])
        result = analyze_keywords(frame, min_review_count=1)
        positive = _json_record(result.sentiment_keywords["positive"], "unigram", "picture")
        aspect = _json_record(result.aspect_keywords["Picture Quality"], "unigram", "picture")
        self.assertEqual(1, positive["count"])
        self.assertEqual(1, aspect["count"])

    def test_malformed_aspect_json_does_not_remove_overall_terms(self) -> None:
        frame = pd.DataFrame([_row(0, "R1", "clear picture", aspects="{bad")])
        result = analyze_keywords(frame, min_review_count=1)
        self.assertEqual(1, len(result.issues))
        _summary_record(result, "unigram", "picture")


class KeywordOutputAndValidationTests(unittest.TestCase):
    def test_json_outputs_are_serializable(self) -> None:
        result = analyze_keywords(_fixture(), min_review_count=1)
        json.dumps(result.sentiment_keywords)
        json.dumps(result.aspect_keywords)

    def test_csv_and_json_outputs_are_saved_and_readable(self) -> None:
        result = analyze_keywords(_fixture(), min_review_count=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = save_keyword_outputs(
                result,
                keyword_summary_path=root / "keywords.csv",
                sentiment_keywords_path=root / "sentiment.json",
                aspect_keywords_path=root / "aspects.json",
            )
            saved = pd.read_csv(paths["keyword_summary"])
            self.assertEqual(list(KEYWORD_SUMMARY_COLUMNS), saved.columns.tolist())
            self.assertIn("picture quality", set(saved["term"]))
            self.assertIn("positive", json.loads(paths["sentiment_keywords"].read_text()))
            self.assertIn("Picture Quality", json.loads(paths["aspect_keywords"].read_text()))

    def test_source_analyzed_csv_bytes_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "analyzed.csv"
            _fixture().to_csv(source, index=False)
            before = source.read_bytes()
            loaded = load_keyword_reviews(source)
            result = analyze_keywords(loaded, min_review_count=1)
            save_keyword_outputs(
                result,
                keyword_summary_path=root / "keywords.csv",
                sentiment_keywords_path=root / "sentiment.json",
                aspect_keywords_path=root / "aspects.json",
            )
            self.assertEqual(before, source.read_bytes())

    def test_missing_required_column_is_rejected(self) -> None:
        with self.assertRaises(KeywordAnalysisError):
            analyze_keywords(_fixture().drop(columns="aspects"), min_review_count=1)

    def test_duplicate_source_row_id_is_rejected(self) -> None:
        frame = _fixture().iloc[:2].copy()
        frame.loc[:, "source_row_id"] = [0, 0]
        with self.assertRaises(KeywordAnalysisError):
            analyze_keywords(frame, min_review_count=1)

    def test_nonpositive_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze_keywords(_fixture(), min_review_count=0)

    def test_zero_success_is_safe_but_not_persisted(self) -> None:
        frame = pd.DataFrame([_row(0, "R1", "clear picture", status="failed")])
        result = analyze_keywords(frame, min_review_count=1)
        self.assertEqual(0, result.successful_reviews)
        self.assertTrue(result.keyword_summary.empty)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeywordAnalysisError):
                save_keyword_outputs(
                    result,
                    keyword_summary_path=Path(directory) / "keywords.csv",
                    sentiment_keywords_path=Path(directory) / "sentiment.json",
                    aspect_keywords_path=Path(directory) / "aspects.json",
                )


if __name__ == "__main__":
    unittest.main()
