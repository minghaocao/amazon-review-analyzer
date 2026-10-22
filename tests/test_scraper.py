"""Offline tests for the compliance-first Amazon review scraper."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scraper  # noqa: E402


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "amazon"
ASIN = "B0TEST1234"
MARKETPLACE = "amazon.com"
PRODUCT_URL = f"https://www.{MARKETPLACE}/dp/{ASIN}"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _parse(name: str, *, page_number: int = 1) -> scraper.ParseResult:
    return scraper.parse_review_page(
        _fixture(name),
        asin=ASIN,
        marketplace=MARKETPLACE,
        source_url=f"https://www.{MARKETPLACE}/product-reviews/{ASIN}/",
        page_number=page_number,
    )


class ParserTests(unittest.TestCase):
    def test_normal_fixture_returns_multiple_reviews(self) -> None:
        result = _parse("normal_reviews.html")
        self.assertEqual(scraper.ScrapeStatus.SUCCESS, result.status)
        self.assertEqual(3, len(result.reviews))
        self.assertIn("pageNumber=2", result.next_page_url or "")

    def test_required_and_optional_fields_are_extracted(self) -> None:
        review = _parse("normal_reviews.html").reviews[0]
        self.assertEqual("RNORMAL001", review["review_id"])
        self.assertEqual(5.0, review["rating"])
        self.assertEqual("Excellent picture quality", review["title"])
        self.assertEqual(
            "The picture is excellent & colors are natural.",
            review["content"],
        )
        self.assertIn("January 5, 2026", review["date"])
        self.assertIs(review["verified_purchase"], True)
        self.assertEqual("Alex", review["reviewer_name"])
        self.assertEqual("12 people found this helpful", review["helpful_votes"])
        self.assertEqual("Size: 55-inch", review["product_variant"])
        self.assertEqual(ASIN, review["asin"])
        self.assertEqual(MARKETPLACE, review["marketplace"])

    def test_id_attribute_prefix_is_removed(self) -> None:
        review = _parse("normal_reviews.html").reviews[0]
        self.assertEqual("RNORMAL001", review["review_id"])

    def test_no_verified_badge_is_false(self) -> None:
        review = _parse("normal_reviews.html").reviews[1]
        self.assertIs(review["verified_purchase"], False)

    def test_missing_rating_and_title_are_none_without_losing_review(self) -> None:
        review = _parse("normal_reviews.html").reviews[2]
        self.assertIsNone(review["rating"])
        self.assertIsNone(review["title"])
        self.assertEqual("Works reliably every day.", review["content"])

    def test_generated_review_id_is_deterministic(self) -> None:
        first = _parse("normal_reviews.html", page_number=1).reviews[2]["review_id"]
        second = _parse("normal_reviews.html", page_number=9).reviews[2]["review_id"]
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("GENERATED_"))

    def test_generated_id_normalizes_content_case_and_spacing(self) -> None:
        first = scraper.generate_review_id(ASIN, " Works  RELIABLY every day. ")
        second = scraper.generate_review_id(ASIN, "works reliably every day.")
        self.assertEqual(first, second)

    def test_localized_rating_formats(self) -> None:
        cases = {
            "5.0 out of 5 stars": 5.0,
            "4,0 von 5 Sternen": 4.0,
            "3,5 sur 5 étoiles": 3.5,
            "2,0 su 5 stelle": 2.0,
            "1,0 de 5 estrellas": 1.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(expected, scraper.parse_rating(text))

    def test_unparseable_or_out_of_range_rating_is_none(self) -> None:
        for value in (None, "", "not rated", "0 out of 5 stars", "6 out of 5"):
            with self.subTest(value=value):
                self.assertIsNone(scraper.parse_rating(value))

    def test_challenge_page_is_blocked(self) -> None:
        result = _parse("challenge.html")
        self.assertEqual(scraper.ScrapeStatus.BLOCKED, result.status)
        self.assertEqual("amazon_challenge", result.reason)
        self.assertEqual([], result.reviews)

    def test_explicit_no_review_page_is_distinct(self) -> None:
        result = _parse("no_reviews.html")
        self.assertEqual(scraper.ScrapeStatus.NO_REVIEWS, result.status)
        self.assertEqual("no_public_reviews", result.reason)

    def test_unknown_structure_is_not_misreported_as_no_reviews(self) -> None:
        result = _parse("structure_unknown.html")
        self.assertEqual(scraper.ScrapeStatus.STRUCTURE_NOT_RECOGNIZED, result.status)
        self.assertEqual("review_container_not_found", result.reason)

    def test_empty_html_is_a_structure_error(self) -> None:
        result = scraper.parse_review_page(
            "",
            asin=ASIN,
            marketplace=MARKETPLACE,
            source_url=PRODUCT_URL,
        )
        self.assertEqual(scraper.ScrapeStatus.STRUCTURE_NOT_RECOGNIZED, result.status)

    def test_non_text_response_has_parse_error_status(self) -> None:
        result = scraper.parse_review_page(  # type: ignore[arg-type]
            b"not text",
            asin=ASIN,
            marketplace=MARKETPLACE,
            source_url=PRODUCT_URL,
        )
        self.assertEqual(scraper.ScrapeStatus.PARSE_ERROR, result.status)


class DeduplicationTests(unittest.TestCase):
    def test_reliable_review_id_has_priority(self) -> None:
        reviews = [
            {"review_id": "R1", "content": "First body"},
            {"review_id": "R1", "content": "Changed body"},
        ]
        result = scraper.deduplicate_reviews(reviews)
        self.assertEqual(1, len(result))
        self.assertEqual("First body", result[0]["content"])

    def test_generated_ids_are_deduplicated_by_normalized_content(self) -> None:
        reviews = [
            {"review_id": "GENERATED_A", "content": "Works reliably every day."},
            {"review_id": "GENERATED_B", "content": " works  RELIABLY every day. "},
        ]
        self.assertEqual(1, len(scraper.deduplicate_reviews(reviews)))

    def test_distinct_reliable_ids_are_retained_even_when_content_matches(self) -> None:
        reviews = [
            {"review_id": "R1", "content": "Same"},
            {"review_id": "R2", "content": "Same"},
        ]
        self.assertEqual(2, len(scraper.deduplicate_reviews(reviews)))

    def test_generated_copy_after_reliable_id_is_removed(self) -> None:
        reviews = [
            {"review_id": "R1", "content": "Same review body"},
            {"review_id": "GENERATED_A", "content": " same  REVIEW body "},
        ]
        result = scraper.deduplicate_reviews(reviews)
        self.assertEqual(["R1"], [item["review_id"] for item in result])

    def test_reliable_id_replaces_generated_copy_seen_first(self) -> None:
        reviews = [
            {"review_id": "GENERATED_A", "content": "Same review body"},
            {"review_id": "R1", "content": " same  REVIEW body "},
        ]
        result = scraper.deduplicate_reviews(reviews)
        self.assertEqual(["R1"], [item["review_id"] for item in result])

    def test_deduplication_does_not_mutate_input(self) -> None:
        reviews = [{"review_id": "R1", "content": "A"}]
        original = [dict(reviews[0])]
        scraper.deduplicate_reviews(reviews)
        self.assertEqual(original, reviews)


class FetchTests(unittest.TestCase):
    class _Response:
        status = 200
        url = "https://www.amazon.com/final"
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self) -> "FetchTests._Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            del size
            return "<html>café</html>".encode("utf-8")

    def test_fetch_success_decodes_text_with_bounded_read(self) -> None:
        opener = Mock(return_value=self._Response())
        result = scraper.fetch_review_page(PRODUCT_URL, timeout=3, opener=opener)
        self.assertEqual(scraper.ScrapeStatus.SUCCESS, result.status)
        self.assertEqual("<html>café</html>", result.html)
        self.assertEqual(200, result.http_status)
        self.assertEqual(3, opener.call_args.kwargs["timeout"])

    def test_fetch_timeout_is_structured(self) -> None:
        def timeout_opener(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise socket.timeout("timed out")

        result = scraper.fetch_review_page(PRODUCT_URL, opener=timeout_opener)
        self.assertEqual(scraper.ScrapeStatus.TIMEOUT, result.status)
        self.assertEqual("connection_timed_out", result.reason)

    def test_parser_tests_never_call_urlopen(self) -> None:
        with patch("scraper.urlopen") as urlopen:
            _parse("normal_reviews.html")
            _parse("challenge.html")
            _parse("no_reviews.html")
        urlopen.assert_not_called()


@dataclass(frozen=True)
class _Product:
    marketplace: str = MARKETPLACE
    asin: str = ASIN
    canonical_product_url: str = PRODUCT_URL


class PaginationTests(unittest.TestCase):
    def test_serial_pagination_delays_once_and_deduplicates(self) -> None:
        calls: list[str] = []
        delays: list[float] = []
        progress: list[tuple[int, int, str]] = []

        def fetch(url: str) -> str:
            calls.append(url)
            if "pageNumber=2" in url:
                return _fixture("duplicate_reviews.html")
            return _fixture("normal_reviews.html")

        result = scraper.scrape_reviews(
            _Product(),
            max_reviews=10,
            fetcher=fetch,
            request_delay_seconds=1.25,
            sleeper=delays.append,
            progress_callback=lambda page, count, status: progress.append(
                (page, count, status)
            ),
        )

        self.assertEqual(scraper.ScrapeStatus.SUCCESS, result.status)
        self.assertEqual(4, result.reviews_collected)
        self.assertEqual(2, result.pages_fetched)
        self.assertEqual(2, len(calls))
        self.assertEqual([1.25], delays)
        self.assertEqual((1, 3, "success"), progress[0])
        self.assertEqual((2, 4, "success"), progress[1])

    def test_review_limit_stops_before_second_page(self) -> None:
        fetch = Mock(return_value=_fixture("normal_reviews.html"))
        sleeper = Mock()
        result = scraper.scrape_reviews(
            _Product(),
            max_reviews=2,
            fetcher=fetch,
            sleeper=sleeper,
        )
        self.assertEqual(2, result.reviews_collected)
        self.assertEqual(1, result.pages_fetched)
        fetch.assert_called_once()
        sleeper.assert_not_called()

    def test_challenge_stops_immediately_without_delay_or_retry(self) -> None:
        fetch = Mock(return_value=_fixture("challenge.html"))
        sleeper = Mock()
        result = scraper.scrape_reviews(
            _Product(), max_reviews=5, fetcher=fetch, sleeper=sleeper
        )
        self.assertEqual(scraper.ScrapeStatus.BLOCKED, result.status)
        self.assertEqual("amazon_challenge", result.reason)
        fetch.assert_called_once()
        sleeper.assert_not_called()

    def test_http_error_with_challenge_body_is_blocked(self) -> None:
        fetch = Mock(
            return_value=scraper.FetchResult(
                status=scraper.ScrapeStatus.HTTP_ERROR,
                url=PRODUCT_URL,
                html=_fixture("challenge.html"),
                reason="http_503",
                http_status=503,
            )
        )
        result = scraper.scrape_reviews(_Product(), max_reviews=5, fetcher=fetch)
        self.assertEqual(scraper.ScrapeStatus.BLOCKED, result.status)

    def test_timeout_stops_with_structured_status(self) -> None:
        fetch = Mock(
            return_value=scraper.FetchResult(
                status=scraper.ScrapeStatus.TIMEOUT,
                url=PRODUCT_URL,
                reason="request_timeout",
            )
        )
        result = scraper.scrape_reviews(_Product(), max_reviews=5, fetcher=fetch)
        self.assertEqual(scraper.ScrapeStatus.TIMEOUT, result.status)
        self.assertEqual("request_timeout", result.reason)
        self.assertEqual([], result.reviews)

    def test_no_reviews_and_structure_error_remain_distinct(self) -> None:
        cases = (
            ("no_reviews.html", scraper.ScrapeStatus.NO_REVIEWS),
            ("structure_unknown.html", scraper.ScrapeStatus.STRUCTURE_NOT_RECOGNIZED),
        )
        for fixture_name, expected in cases:
            with self.subTest(fixture=fixture_name):
                result = scraper.scrape_reviews(
                    _Product(), max_reviews=5, fetcher=lambda url: _fixture(fixture_name)
                )
                self.assertEqual(expected, result.status)

    def test_result_metadata_has_only_safe_summary_fields(self) -> None:
        result = scraper.scrape_reviews(
            _Product(), max_reviews=2, fetcher=lambda url: _fixture("normal_reviews.html")
        )
        self.assertEqual(
            {
                "asin",
                "marketplace",
                "source_url",
                "requested_limit",
                "reviews_collected",
                "scrape_status",
                "reason",
                "pages_fetched",
            },
            set(result.to_metadata()),
        )
        self.assertEqual(2, result.to_metadata()["reviews_collected"])

    def test_progress_callback_failure_does_not_break_import(self) -> None:
        def broken_progress(page: int, count: int, status: str) -> None:
            del page, count, status
            raise RuntimeError("display disappeared")

        result = scraper.scrape_reviews(
            _Product(),
            max_reviews=2,
            fetcher=lambda url: _fixture("normal_reviews.html"),
            progress_callback=broken_progress,
        )
        self.assertEqual(scraper.ScrapeStatus.SUCCESS, result.status)

    def test_invalid_limits_are_rejected_before_fetching(self) -> None:
        fetch = Mock()
        for value in (0, -1, 101, True):
            with self.subTest(max_reviews=value):
                with self.assertRaises(ValueError):
                    scraper.scrape_reviews(_Product(), max_reviews=value, fetcher=fetch)
        fetch.assert_not_called()

    def test_mapping_product_input_is_supported(self) -> None:
        product = {
            "marketplace": "amazon.co.uk",
            "asin": "b0test1234",
            "canonical_product_url": "https://www.amazon.co.uk/dp/B0TEST1234",
        }
        result = scraper.scrape_reviews(
            product,
            max_reviews=1,
            fetcher=lambda url: _fixture("normal_reviews.html"),
        )
        self.assertEqual("amazon.co.uk", result.marketplace)
        self.assertEqual(ASIN, result.asin)
        self.assertEqual(1, result.reviews_collected)

    def test_unsupported_marketplace_is_rejected_before_fetch(self) -> None:
        fetch = Mock()
        with self.assertRaisesRegex(ValueError, "marketplace is not supported"):
            scraper.scrape_reviews(
                {
                    "marketplace": "example.com",
                    "asin": ASIN,
                    "canonical_product_url": f"https://example.com/dp/{ASIN}",
                },
                max_reviews=1,
                fetcher=fetch,
            )
        fetch.assert_not_called()

    def test_invalid_asin_is_rejected_before_fetch(self) -> None:
        fetch = Mock()
        with self.assertRaisesRegex(ValueError, "10-character"):
            scraper.scrape_reviews(
                {
                    "marketplace": "amazon.com",
                    "asin": "TOO-SHORT",
                    "canonical_product_url": "https://www.amazon.com/dp/TOO-SHORT",
                },
                max_reviews=1,
                fetcher=fetch,
            )
        fetch.assert_not_called()

    def test_external_pagination_link_is_never_fetched(self) -> None:
        html = _fixture("normal_reviews.html").replace(
            "/product-reviews/B0TEST1234/?pageNumber=2",
            "https://example.com/steal",
        )
        fetch = Mock(return_value=html)

        result = scraper.scrape_reviews(
            _Product(),
            max_reviews=10,
            fetcher=fetch,
            sleeper=Mock(),
        )

        self.assertEqual(scraper.ScrapeStatus.PARSE_ERROR, result.status)
        self.assertEqual("unsafe_pagination_url", result.reason)
        fetch.assert_called_once()

    def test_cross_asin_pagination_link_is_never_fetched(self) -> None:
        html = _fixture("normal_reviews.html").replace(
            "/product-reviews/B0TEST1234/?pageNumber=2",
            "/product-reviews/B0OTHER999/?pageNumber=2",
        )
        fetch = Mock(return_value=html)

        result = scraper.scrape_reviews(
            _Product(),
            max_reviews=10,
            fetcher=fetch,
            sleeper=Mock(),
        )

        self.assertEqual(scraper.ScrapeStatus.PARSE_ERROR, result.status)
        self.assertEqual("unsafe_pagination_url", result.reason)
        fetch.assert_called_once()

    def test_all_network_access_is_dependency_injectable(self) -> None:
        with patch("scraper.urlopen") as urlopen:
            result = scraper.scrape_reviews(
                _Product(),
                max_reviews=2,
                fetcher=lambda url: _fixture("normal_reviews.html"),
            )
        urlopen.assert_not_called()
        self.assertEqual(scraper.ScrapeStatus.SUCCESS, result.status)


if __name__ == "__main__":
    unittest.main()
