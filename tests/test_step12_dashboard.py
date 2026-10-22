"""Synthetic Streamlit acceptance guards for the Step 12 five-tab dashboard."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregator import aggregate_reviews, save_aggregation_outputs  # noqa: E402


def _synthetic_reviews(*, include_rating: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for index in range(100):
        aspects: list[dict] = []
        if index < 60:
            sentiment = "negative" if index % 10 == 0 else "positive"
            aspects.append(
                {
                    "raw_aspect": "picture",
                    "aspect": "Picture Quality",
                    "sentiment": sentiment,
                    "evidence": f"picture signal {sentiment}",
                }
            )
        if index < 40:
            aspects.append(
                {
                    "raw_aspect": "remote",
                    "aspect": "Remote Control",
                    "sentiment": "negative",
                    "evidence": "remote response negative",
                }
            )
        if 60 <= index < 80:
            sentiment = "positive" if index % 3 else "negative"
            aspects.append(
                {
                    "raw_aspect": "delivery",
                    "aspect": "Delivery",
                    "sentiment": sentiment,
                    "evidence": f"delivery service {sentiment}",
                }
            )
        unique = [
            {
                "aspect": entry["aspect"],
                "sentiment": entry["sentiment"],
                "mention_count": 1,
            }
            for entry in aspects
        ]
        content = ". ".join(entry["evidence"] for entry in aspects) or "No aspect."
        row = {
            "source_row_id": index,
            "review_id": f"SYN{index:03d}",
            "title": f"Synthetic review {index}",
            "original_content": content,
            "cleaned_content": content,
            "verified_purchase": index % 2 == 0,
            "analysis_status": "success",
            "summary": "Synthetic deterministic fixture.",
            "aspects": json.dumps(aspects),
            "unique_aspects": json.dumps(unique),
            "positive_points": "[]",
            "negative_points": "[]",
            "error_message": "",
        }
        if include_rating:
            row["rating"] = (index % 5) + 1
        rows.append(row)
    return pd.DataFrame(rows)


def _write_outputs(root: Path, *, include_rating: bool) -> None:
    reviews = _synthetic_reviews(include_rating=include_rating)
    reviews.to_csv(root / "analyzed_reviews.csv", index=False)
    result = aggregate_reviews(reviews, min_aspect_reviews=3)
    save_aggregation_outputs(
        result,
        aspect_summary_path=root / "aspect_summary.csv",
        business_summary_path=root / "business_summary.json",
        evidence_path=root / "aspect_evidence.json",
    )
    pd.DataFrame(
        [
            {
                "term": "picture signal",
                "term_type": "bigram",
                "count": 60,
                "review_count": 60,
                "review_share": 60.0,
            }
        ]
    ).to_csv(root / "keyword_summary.csv", index=False)
    (root / "sentiment_keywords.json").write_text(
        json.dumps(
            {
                "positive": [
                    {
                        "term": "picture signal",
                        "term_type": "bigram",
                        "count": 54,
                        "review_count": 54,
                    }
                ],
                "negative": [
                    {
                        "term": "remote response",
                        "term_type": "bigram",
                        "count": 40,
                        "review_count": 40,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "aspect_keywords.json").write_text(
        json.dumps(
            {
                "Picture Quality": [
                    {
                        "term": "picture signal",
                        "term_type": "bigram",
                        "count": 60,
                        "review_count": 60,
                        "review_refs": [
                            {"source_row_id": index, "review_id": f"SYN{index:03d}"}
                            for index in range(60)
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _run_dashboard(root: Path) -> AppTest:
    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(PROJECT_ROOT)!r})
import app
app._render_saved_dashboard(
    Path({str(root)!r}),
    dataset_label='Synthetic 100 Review Dataset',
    key_prefix='synthetic',
    compact_view=True,
)
"""
    rendered = AppTest.from_string(script, default_timeout=20)
    rendered.run()
    return rendered


class Step12SyntheticDashboardTests(unittest.TestCase):
    def test_one_hundred_review_fixture_renders_all_five_tabs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_outputs(root, include_rating=True)
            app = _run_dashboard(root)
            self.assertFalse(app.exception)
            labels = {
                "Overview",
                "Product Insights",
                "Customer Language",
                "Aspect Deep Dive",
                "Review Explorer",
            }
            self.assertEqual(labels, {tab.label for tab in app.tabs if tab.label in labels})
            subheaders = {item.value for item in app.subheader}
            self.assertIn("Aspect by Star Rating", subheaders)
            self.assertTrue(
                any(
                    "Hidden complaints in positive reviews" in item.value
                    for item in app.markdown
                )
            )

    def test_rating_sections_hide_when_rating_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_outputs(root, include_rating=False)
            app = _run_dashboard(root)
            self.assertFalse(app.exception)
            self.assertNotIn(
                "Aspect by Star Rating", {item.value for item in app.subheader}
            )


if __name__ == "__main__":
    unittest.main()
