"""Offline tests for the visible-browser Amazon review importer.

No test in this module starts Playwright or makes a network request.  The
small fake browser graph exercises the public session API while preserving the
same browser/context/page objects across commands.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import re
import sys
import unittest
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amazon_url import AmazonProductURL, parse_amazon_url  # noqa: E402
from browser_amazon_import import (  # noqa: E402
    BROWSER_RUNTIME_INSTALL_MESSAGE,
    MAX_AMAZON_QUICK_SCAN_REVIEWS,
    AmazonBrowserImportResult,
    AmazonBrowserSession,
    BrowserImportStatus,
    ReviewAvailabilityStatus,
    ReviewCollectionStopReason,
    classify_browser_page,
    detect_login_required,
    quick_scan_availability,
)
from scraper import detect_amazon_challenge  # noqa: E402


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "amazon"
ASIN = "B0TEST1234"
MARKETPLACE = "amazon.com"
PRODUCT_URL = f"https://www.{MARKETPLACE}/dp/{ASIN}"
REVIEWS_BASE_URL = f"https://www.{MARKETPLACE}/product-reviews/{ASIN}/"
REVIEWS_URL = f"{REVIEWS_BASE_URL}?reviewerType=all_reviews&pageNumber=1"
PRODUCT_HTML = """
<!doctype html><html><head><title>Example TV</title></head>
<body><span id="productTitle">Example TV</span></body></html>
"""


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _product() -> AmazonProductURL:
    return AmazonProductURL(MARKETPLACE, ASIN, PRODUCT_URL)


def _review_page(
    reviews: list[tuple[str, str]],
    *,
    next_page: int | None = None,
) -> str:
    blocks = "".join(
        f"""
        <div data-hook="review" data-review-id="{review_id}">
          <i data-hook="review-star-rating">4.0 out of 5 stars</i>
          <a data-hook="review-title"><span>Review {review_id}</span></a>
          <span data-hook="review-date">Reviewed on January 1, 2026</span>
          <span data-hook="review-body">{content}</span>
        </div>
        """
        for review_id, content in reviews
    )
    pagination = ""
    if next_page is not None:
        pagination = (
            '<ul class="a-pagination"><li class="a-last">'
            f'<a href="/product-reviews/{ASIN}/?pageNumber={next_page}">Next</a>'
            "</li></ul>"
        )
    return f"<!doctype html><html><body>{blocks}{pagination}</body></html>"


class FakeTimeoutError(RuntimeError):
    pass


class _FakePage:
    def __init__(
        self,
        *,
        routes: dict[str, str] | None = None,
        goto_actions: list[BaseException | None] | None = None,
    ) -> None:
        self.routes = routes or {}
        self.goto_actions = deque(goto_actions or [])
        self.goto_calls: list[tuple[str, str, int]] = []
        self.url = "about:blank"
        self.html = ""
        self.default_timeout: int | None = None
        self.close_calls = 0

    def set_default_navigation_timeout(self, timeout: int) -> None:
        self.default_timeout = timeout

    def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.goto_calls.append((url, wait_until, timeout))
        if self.goto_actions:
            action = self.goto_actions.popleft()
            if action is not None:
                raise action
        self.url = url
        self.html = self.routes.get(url, self.routes.get("*", self.html))

    def content(self) -> str:
        return self.html

    def title(self) -> str:
        match = re.search(r"<title>(.*?)</title>", self.html, re.I | re.S)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""

    def close(self) -> None:
        self.close_calls += 1


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.new_page_calls = 0
        self.close_calls = 0

    def new_page(self) -> _FakePage:
        self.new_page_calls += 1
        return self.page

    def close(self) -> None:
        self.close_calls += 1


class _FakeBrowser:
    def __init__(self, context: _FakeContext) -> None:
        self.context = context
        self.new_context_calls = 0
        self.close_calls = 0

    def new_context(self) -> _FakeContext:
        self.new_context_calls += 1
        return self.context

    def close(self) -> None:
        self.close_calls += 1


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser) -> None:
        self.browser = browser
        self.launch_calls: list[dict[str, Any]] = []

    def launch(self, **kwargs: Any) -> _FakeBrowser:
        self.launch_calls.append(kwargs)
        return self.browser


class _FakePlaywright:
    def __init__(self, chromium: _FakeChromium) -> None:
        self.chromium = chromium
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


class _FakeManager:
    def __init__(self, playwright: _FakePlaywright) -> None:
        self.playwright = playwright
        self.start_calls = 0

    def start(self) -> _FakePlaywright:
        self.start_calls += 1
        return self.playwright


class _BrowserGraph:
    def __init__(
        self,
        *,
        routes: dict[str, str] | None = None,
        goto_actions: list[BaseException | None] | None = None,
    ) -> None:
        self.page = _FakePage(routes=routes, goto_actions=goto_actions)
        self.context = _FakeContext(self.page)
        self.browser = _FakeBrowser(self.context)
        self.chromium = _FakeChromium(self.browser)
        self.playwright = _FakePlaywright(self.chromium)
        self.manager = _FakeManager(self.playwright)
        self.factory_calls = 0

    def factory(self) -> _FakeManager:
        self.factory_calls += 1
        return self.manager


class URLAndClassificationTests(unittest.TestCase):
    def test_requested_real_product_url_is_normalized(self) -> None:
        parsed = parse_amazon_url(
            "https://www.amazon.com/Roku-Smart-2026-Television-Streaming/"
            "dp/B0FXZ21F3X/"
        )
        self.assertEqual("amazon.com", parsed.marketplace)
        self.assertEqual("B0FXZ21F3X", parsed.asin)
        self.assertEqual(
            "https://www.amazon.com/dp/B0FXZ21F3X",
            parsed.canonical_product_url,
        )

    def test_normal_product_page_is_classified(self) -> None:
        self.assertEqual(
            BrowserImportStatus.PRODUCT_PAGE_LOADED,
            classify_browser_page(PRODUCT_HTML, current_url=PRODUCT_URL),
        )

    def test_challenge_page_is_classified(self) -> None:
        self.assertEqual(
            BrowserImportStatus.CHALLENGE_REQUIRED,
            classify_browser_page(_fixture("challenge.html"), current_url=PRODUCT_URL),
        )

    def test_login_page_is_classified(self) -> None:
        html = '<html><form name="signIn"><input id="ap_email"></form></html>'
        self.assertEqual(
            BrowserImportStatus.LOGIN_REQUIRED,
            classify_browser_page(
                html,
                current_url="https://www.amazon.com/ap/signin",
            ),
        )

    def test_ordinary_header_sign_in_link_is_not_login_required(self) -> None:
        html = (
            '<html><a href="/ap/signin">Sign in</a>'
            '<span id="productTitle">Television</span></html>'
        )
        self.assertFalse(detect_login_required(html, PRODUCT_URL))
        self.assertEqual(
            BrowserImportStatus.PRODUCT_PAGE_LOADED,
            classify_browser_page(html, current_url=PRODUCT_URL),
        )

    def test_dormant_header_sign_in_form_does_not_hide_product_page(self) -> None:
        html = (
            '<html><span id="productTitle">Television</span>'
            '<form name="signIn"><input id="ap_email"></form></html>'
        )
        self.assertFalse(detect_login_required(html, PRODUCT_URL))
        self.assertEqual(
            BrowserImportStatus.PRODUCT_PAGE_LOADED,
            classify_browser_page(html, current_url=PRODUCT_URL),
        )

    def test_normal_product_dom_ignores_generic_policy_challenge_phrase(self) -> None:
        html = (
            '<html><span id="productTitle">Television</span>'
            '<script>"automated access to Amazon data"</script></html>'
        )
        self.assertFalse(detect_amazon_challenge(html))
        self.assertEqual(
            BrowserImportStatus.PRODUCT_PAGE_LOADED,
            classify_browser_page(html, current_url=PRODUCT_URL),
        )

    def test_reviews_unavailable_has_distinct_classification(self) -> None:
        html = "<html><body>Reviews are not available for this item</body></html>"
        self.assertEqual(
            BrowserImportStatus.REVIEWS_UNAVAILABLE,
            classify_browser_page(html, current_url=REVIEWS_URL, expect_reviews=True),
        )

    def test_unknown_product_structure_is_classified(self) -> None:
        self.assertEqual(
            BrowserImportStatus.PAGE_STRUCTURE_CHANGED,
            classify_browser_page(
                _fixture("structure_unknown.html"),
                current_url=PRODUCT_URL,
            ),
        )


class BrowserSessionTests(unittest.TestCase):
    def _session(self, graph: _BrowserGraph) -> AmazonBrowserSession:
        session = AmazonBrowserSession(_product(), playwright_factory=graph.factory)
        self.addCleanup(session.close)
        return session

    def test_browser_launch_is_headed_and_uses_one_ephemeral_context(self) -> None:
        graph = _BrowserGraph(routes={PRODUCT_URL: PRODUCT_HTML})
        result = self._session(graph).open_product()

        self.assertEqual(BrowserImportStatus.PRODUCT_PAGE_LOADED, result.status)
        self.assertEqual([{"headless": False}], graph.chromium.launch_calls)
        self.assertEqual(1, graph.browser.new_context_calls)
        self.assertEqual(1, graph.context.new_page_calls)

    def test_missing_browser_runtime_is_friendly(self) -> None:
        def missing_runtime() -> Any:
            raise RuntimeError(
                "Executable doesn't exist. Please run playwright install chromium"
            )

        session = AmazonBrowserSession(_product(), playwright_factory=missing_runtime)
        self.addCleanup(session.close)
        result = session.open_product()

        self.assertEqual(BrowserImportStatus.BROWSER_UNAVAILABLE, result.status)
        self.assertEqual(BROWSER_RUNTIME_INSTALL_MESSAGE, result.error_message)

    def test_navigation_timeout_retries_at_most_once(self) -> None:
        graph = _BrowserGraph(
            routes={PRODUCT_URL: PRODUCT_HTML},
            goto_actions=[FakeTimeoutError("timed out"), FakeTimeoutError("timed out")],
        )
        result = self._session(graph).open_product()

        self.assertEqual(BrowserImportStatus.NAVIGATION_TIMEOUT, result.status)
        self.assertEqual(2, len(graph.page.goto_calls))

    def test_network_error_retries_once_then_stops(self) -> None:
        graph = _BrowserGraph(
            routes={PRODUCT_URL: PRODUCT_HTML},
            goto_actions=[OSError("connection reset"), OSError("connection reset")],
        )
        result = self._session(graph).open_product()

        self.assertEqual(BrowserImportStatus.NETWORK_ERROR, result.status)
        self.assertEqual(2, len(graph.page.goto_calls))

    def test_challenge_continue_reuses_same_browser_context_and_page(self) -> None:
        graph = _BrowserGraph(routes={PRODUCT_URL: _fixture("challenge.html")})
        session = self._session(graph)
        first = session.open_product()
        self.assertEqual(BrowserImportStatus.CHALLENGE_REQUIRED, first.status)

        # Simulate the user completing Amazon's challenge in the visible page.
        graph.page.html = PRODUCT_HTML
        graph.page.url = PRODUCT_URL
        continued = session.continue_after_verification()

        self.assertEqual(BrowserImportStatus.PRODUCT_PAGE_LOADED, continued.status)
        self.assertTrue(continued.challenge_detected)
        self.assertEqual(1, graph.factory_calls)
        self.assertEqual(1, len(graph.chromium.launch_calls))
        self.assertEqual(1, graph.browser.new_context_calls)
        self.assertEqual(1, graph.context.new_page_calls)

    def test_no_reviews_and_changed_review_structure_are_distinct(self) -> None:
        cases = (
            ("no_reviews.html", BrowserImportStatus.NO_REVIEWS_FOUND),
            ("structure_unknown.html", BrowserImportStatus.PAGE_STRUCTURE_CHANGED),
        )
        for fixture_name, expected in cases:
            with self.subTest(fixture=fixture_name):
                graph = _BrowserGraph(
                    routes={
                        PRODUCT_URL: PRODUCT_HTML,
                        REVIEWS_URL: _fixture(fixture_name),
                    }
                )
                session = AmazonBrowserSession(
                    _product(), playwright_factory=graph.factory
                )
                try:
                    self.assertEqual(
                        BrowserImportStatus.PRODUCT_PAGE_LOADED,
                        session.open_product().status,
                    )
                    self.assertEqual(expected, session.load_reviews(limit=5).status)
                finally:
                    session.close()

    def test_confirmed_zero_reviews_has_no_reviews_availability(self) -> None:
        graph = _BrowserGraph(
            routes={
                PRODUCT_URL: PRODUCT_HTML,
                REVIEWS_URL: _fixture("no_reviews.html"),
            }
        )
        session = self._session(graph)
        session.open_product()

        result = session.load_reviews(limit=MAX_AMAZON_QUICK_SCAN_REVIEWS)

        self.assertEqual(BrowserImportStatus.NO_REVIEWS_FOUND, result.status)
        self.assertEqual(0, result.unique_review_count)
        self.assertEqual(ReviewAvailabilityStatus.NO_REVIEWS, result.availability_status)
        self.assertEqual(
            ReviewCollectionStopReason.PRODUCT_EXHAUSTED,
            result.stop_reason,
        )

    def test_first_review_page_navigation_failure_has_explicit_stop_reason(self) -> None:
        graph = _BrowserGraph(
            routes={PRODUCT_URL: PRODUCT_HTML},
            goto_actions=[
                None,
                FakeTimeoutError("timed out"),
                FakeTimeoutError("timed out"),
            ],
        )
        session = self._session(graph)
        self.assertEqual(
            BrowserImportStatus.PRODUCT_PAGE_LOADED,
            session.open_product().status,
        )

        result = session.load_reviews(limit=MAX_AMAZON_QUICK_SCAN_REVIEWS)

        self.assertEqual(BrowserImportStatus.NAVIGATION_TIMEOUT, result.status)
        self.assertEqual(0, result.unique_review_count)
        self.assertEqual(
            ReviewCollectionStopReason.NAVIGATION_FAILED,
            result.stop_reason,
        )

    def test_local_fixture_yields_raw_compatible_six_fields(self) -> None:
        fixture_without_next = re.sub(
            r'<ul class="a-pagination">.*?</ul>',
            "",
            _fixture("normal_reviews.html"),
            flags=re.I | re.S,
        )
        graph = _BrowserGraph(
            routes={PRODUCT_URL: PRODUCT_HTML, REVIEWS_URL: fixture_without_next}
        )
        session = self._session(graph)
        session.open_product()
        result = session.load_reviews(limit=5)

        self.assertEqual(BrowserImportStatus.SUCCESS, result.status)
        self.assertEqual(3, len(result.reviews))
        required = {
            "review_id",
            "rating",
            "title",
            "content",
            "date",
            "verified_purchase",
        }
        self.assertTrue(all(required <= set(review) for review in result.reviews))
        self.assertTrue(all(review["content"] for review in result.reviews))
        self.assertEqual(3, result.retrieved_review_count)
        self.assertEqual(3, result.unique_review_count)
        self.assertIsNone(result.target_review_count)
        self.assertEqual(ReviewAvailabilityStatus.PARTIAL, result.availability_status)
        self.assertEqual(ReviewCollectionStopReason.NO_NEXT_PAGE, result.stop_reason)

    def test_raw_review_containers_are_deduplicated_before_import(self) -> None:
        reviews = [(f"R{index}", f"Body {index}") for index in range(1, 9)]
        reviews.extend([("R1", "Duplicate one"), ("R2", "Duplicate two")])
        page = _review_page(reviews)
        graph = _BrowserGraph(routes={PRODUCT_URL: PRODUCT_HTML, REVIEWS_URL: page})
        session = self._session(graph)
        session.open_product()

        result = session.load_reviews(limit=MAX_AMAZON_QUICK_SCAN_REVIEWS)

        self.assertEqual(BrowserImportStatus.SUCCESS, result.status)
        self.assertEqual(10, result.retrieved_review_count)
        self.assertEqual(8, result.unique_review_count)
        self.assertEqual(8, len(result.reviews))
        self.assertEqual(ReviewCollectionStopReason.NO_NEXT_PAGE, result.stop_reason)

    def test_no_new_unique_reviews_stops_bounded_scan(self) -> None:
        first = _review_page([("R1", "One"), ("R2", "Two")], next_page=2)
        second = _review_page(
            [("R1", "One duplicate"), ("R2", "Two duplicate")],
            next_page=3,
        )
        graph = _BrowserGraph(
            routes={
                PRODUCT_URL: PRODUCT_HTML,
                REVIEWS_URL: first,
                f"{REVIEWS_BASE_URL}?pageNumber=2": second,
            }
        )
        session = self._session(graph)
        session.open_product()

        result = session.load_reviews(limit=MAX_AMAZON_QUICK_SCAN_REVIEWS)

        self.assertEqual(BrowserImportStatus.SUCCESS, result.status)
        self.assertEqual(4, result.retrieved_review_count)
        self.assertEqual(2, result.unique_review_count)
        self.assertEqual(ReviewCollectionStopReason.NO_NEW_REVIEWS, result.stop_reason)
        self.assertEqual(ReviewAvailabilityStatus.PARTIAL, result.availability_status)

    def test_access_limited_after_reviews_preserves_analyzable_rows(self) -> None:
        first = _review_page([("R1", "One"), ("R2", "Two")], next_page=2)
        limited = "<html><body>Reviews are not available for this item</body></html>"
        graph = _BrowserGraph(
            routes={
                PRODUCT_URL: PRODUCT_HTML,
                REVIEWS_URL: first,
                f"{REVIEWS_BASE_URL}?pageNumber=2": limited,
            }
        )
        session = self._session(graph)
        session.open_product()

        result = session.load_reviews(limit=MAX_AMAZON_QUICK_SCAN_REVIEWS)

        self.assertEqual(BrowserImportStatus.SUCCESS, result.status)
        self.assertEqual(["R1", "R2"], [row["review_id"] for row in result.reviews])
        self.assertEqual(ReviewAvailabilityStatus.ACCESS_LIMITED, result.availability_status)
        self.assertEqual(
            ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED,
            result.stop_reason,
        )

    def test_parser_block_on_first_page_is_nonfatal_access_limitation(self) -> None:
        weak_block_page = (
            "<html><body>Automated access to Amazon data is limited.</body></html>"
        )
        graph = _BrowserGraph(
            routes={PRODUCT_URL: PRODUCT_HTML, REVIEWS_URL: weak_block_page}
        )
        session = self._session(graph)
        session.open_product()

        result = session.load_reviews(limit=MAX_AMAZON_QUICK_SCAN_REVIEWS)

        self.assertEqual(BrowserImportStatus.REVIEWS_UNAVAILABLE, result.status)
        self.assertEqual(0, result.unique_review_count)
        self.assertEqual(
            ReviewAvailabilityStatus.ACCESS_LIMITED,
            result.availability_status,
        )
        self.assertEqual(
            ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED,
            result.stop_reason,
        )

    def test_parser_block_after_reviews_preserves_available_rows(self) -> None:
        first = _review_page([("R1", "One"), ("R2", "Two")], next_page=2)
        weak_block_page = (
            "<html><body>Automated access to Amazon data is limited.</body></html>"
        )
        graph = _BrowserGraph(
            routes={
                PRODUCT_URL: PRODUCT_HTML,
                REVIEWS_URL: first,
                f"{REVIEWS_BASE_URL}?pageNumber=2": weak_block_page,
            }
        )
        session = self._session(graph)
        session.open_product()

        result = session.load_reviews(limit=MAX_AMAZON_QUICK_SCAN_REVIEWS)

        self.assertEqual(BrowserImportStatus.SUCCESS, result.status)
        self.assertEqual(["R1", "R2"], [row["review_id"] for row in result.reviews])
        self.assertEqual(
            ReviewAvailabilityStatus.ACCESS_LIMITED,
            result.availability_status,
        )
        self.assertEqual(
            ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED,
            result.stop_reason,
        )

    def test_challenge_after_reviews_preserves_rows_without_bypass(self) -> None:
        first = _review_page([("R1", "One"), ("R2", "Two")], next_page=2)
        graph = _BrowserGraph(
            routes={
                PRODUCT_URL: PRODUCT_HTML,
                REVIEWS_URL: first,
                f"{REVIEWS_BASE_URL}?pageNumber=2": _fixture("challenge.html"),
            }
        )
        session = self._session(graph)
        session.open_product()

        result = session.load_reviews(limit=MAX_AMAZON_QUICK_SCAN_REVIEWS)

        self.assertEqual(BrowserImportStatus.SUCCESS, result.status)
        self.assertEqual(["R1", "R2"], [row["review_id"] for row in result.reviews])
        self.assertTrue(result.challenge_detected)
        self.assertEqual(ReviewAvailabilityStatus.ACCESS_LIMITED, result.availability_status)
        self.assertEqual(
            ReviewCollectionStopReason.CHALLENGE_REQUIRED,
            result.stop_reason,
        )

    def test_portal_customer_review_pagination_is_allowed(self) -> None:
        first = _review_page([("R1", "One")], next_page=None).replace(
            "</body>",
            (
                '<ul class="a-pagination"><li class="a-last">'
                f'<a href="/portal/customer-reviews/{ASIN}/?pageNumber=2">Next</a>'
                "</li></ul></body>"
            ),
        )
        portal_url = (
            f"https://www.{MARKETPLACE}/portal/customer-reviews/{ASIN}/"
            "?pageNumber=2"
        )
        second = _review_page([("R2", "Two")])
        graph = _BrowserGraph(
            routes={
                PRODUCT_URL: PRODUCT_HTML,
                REVIEWS_URL: first,
                portal_url: second,
            }
        )
        session = self._session(graph)
        session.open_product()

        result = session.load_reviews(limit=MAX_AMAZON_QUICK_SCAN_REVIEWS)

        self.assertEqual(BrowserImportStatus.SUCCESS, result.status)
        self.assertEqual(["R1", "R2"], [row["review_id"] for row in result.reviews])

    def test_review_limit_stops_before_following_page(self) -> None:
        first = _review_page(
            [(f"R{index}", f"Review body {index}") for index in range(5)],
            next_page=2,
        )
        graph = _BrowserGraph(routes={PRODUCT_URL: PRODUCT_HTML, REVIEWS_URL: first})
        session = self._session(graph)
        session.open_product()
        result = session.load_reviews(limit=5)

        self.assertEqual(BrowserImportStatus.SUCCESS, result.status)
        self.assertEqual(5, len(result.reviews))
        review_calls = [call for call in graph.page.goto_calls if "product-reviews" in call[0]]
        self.assertEqual(1, len(review_calls))
        self.assertEqual(1, result.pages_visited)
        self.assertEqual(
            ReviewCollectionStopReason.SAFETY_LIMIT_REACHED,
            result.stop_reason,
        )
        self.assertEqual(
            ReviewAvailabilityStatus.PARTIAL,
            result.availability_status,
        )

    def test_pagination_is_capped_at_three_pages_and_deduplicates(self) -> None:
        page_1 = _review_page([("R1", "One"), ("R2", "Two")], next_page=2)
        page_2 = _review_page([("R2", "Duplicate"), ("R3", "Three")], next_page=3)
        page_3 = _review_page([("R4", "Four")], next_page=4)
        graph = _BrowserGraph(
            routes={
                PRODUCT_URL: PRODUCT_HTML,
                REVIEWS_URL: page_1,
                f"{REVIEWS_BASE_URL}?pageNumber=2": page_2,
                f"{REVIEWS_BASE_URL}?pageNumber=3": page_3,
            }
        )
        session = self._session(graph)
        session.open_product()
        result = session.load_reviews(limit=20, max_pages=3)

        self.assertEqual(BrowserImportStatus.SUCCESS, result.status)
        self.assertEqual(3, result.pages_visited)
        self.assertEqual(["R1", "R2", "R3", "R4"], [r["review_id"] for r in result.reviews])
        review_calls = [call for call in graph.page.goto_calls if "product-reviews" in call[0]]
        self.assertEqual(3, len(review_calls))
        self.assertEqual(
            ReviewCollectionStopReason.SAFETY_LIMIT_REACHED,
            result.stop_reason,
        )
        self.assertEqual(ReviewAvailabilityStatus.PARTIAL, result.availability_status)

    def test_close_is_idempotent(self) -> None:
        graph = _BrowserGraph(routes={PRODUCT_URL: PRODUCT_HTML})
        session = AmazonBrowserSession(_product(), playwright_factory=graph.factory)
        session.open_product()

        first = session.close()
        second = session.close()

        self.assertEqual(BrowserImportStatus.CANCELLED, first.status)
        self.assertEqual(BrowserImportStatus.CANCELLED, second.status)
        self.assertEqual(1, graph.page.close_calls)
        self.assertEqual(1, graph.context.close_calls)
        self.assertEqual(1, graph.browser.close_calls)
        self.assertEqual(1, graph.playwright.stop_calls)


class ResultMetadataTests(unittest.TestCase):
    def test_access_limited_is_nonfatal_when_unique_reviews_exist(self) -> None:
        status, reason = quick_scan_availability(
            8,
            ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED,
        )
        self.assertEqual(ReviewAvailabilityStatus.ACCESS_LIMITED, status)
        self.assertIn("limited set", reason or "")

    def test_metadata_is_allowlisted_and_excludes_sensitive_values(self) -> None:
        result = AmazonBrowserImportResult(
            status=BrowserImportStatus.SUCCESS,
            marketplace=MARKETPLACE,
            asin=ASIN,
            product_url=PRODUCT_URL,
            reviews_url=REVIEWS_URL,
            reviews=[{"content": "Good"}],
            challenge_detected=True,
            login_required=False,
            pages_visited=2,
            current_url="https://www.amazon.com/private?token=secret",
            page_title="Private account page",
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:01:00+00:00",
            retrieved_review_count=10,
            unique_review_count=1,
            target_review_count=None,
            availability_status=ReviewAvailabilityStatus.ACCESS_LIMITED,
            availability_reason="Limited current session.",
            stop_reason=ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED,
        )

        metadata = result.to_metadata()

        self.assertEqual(
            {
                "marketplace",
                "asin",
                "product_url",
                "reviews_url",
                "review_count",
                "retrieved_review_count",
                "unique_review_count",
                "target_review_count",
                "availability_status",
                "availability_reason",
                "stop_reason",
                "pages_visited",
                "challenge_encountered",
                "login_required",
                "status",
                "started_at",
                "completed_at",
            },
            set(metadata),
        )
        self.assertEqual(10, metadata["retrieved_review_count"])
        self.assertEqual(1, metadata["unique_review_count"])
        self.assertIsNone(metadata["target_review_count"])
        serialized = repr(metadata).casefold()
        for forbidden in ("cookie", "password", "token=secret", "private account"):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
