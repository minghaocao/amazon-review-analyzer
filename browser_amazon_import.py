"""Visible-browser Amazon review import with explicit manual verification.

This module is deliberately conservative.  It opens an ordinary headed
Chromium window, never submits sign-in or challenge forms, and never stores
browser cookies or authentication state.  A dedicated worker thread owns all
Playwright objects so a Streamlit rerun can safely keep using the same browser
context through the :class:`AmazonBrowserSession` object held in session state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from queue import Empty, Queue
import re
import threading
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

from amazon_url import AmazonProductURL
from scraper import (
    ScrapeStatus,
    build_review_page_url,
    deduplicate_reviews,
    parse_review_page,
)


# Kept as a compatibility surface for callers from earlier releases.  Quick
# Scan does not expose these values as user-selectable targets.
ALLOWED_REVIEW_LIMITS = (5, 10, 20, 50, 100)
MAX_AMAZON_QUICK_SCAN_REVIEWS = 100


def _bounded_int_setting(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


DEFAULT_BROWSER_REVIEW_LIMIT = _bounded_int_setting(
    "AMAZON_BROWSER_REVIEW_LIMIT",
    MAX_AMAZON_QUICK_SCAN_REVIEWS,
    minimum=1,
    maximum=MAX_AMAZON_QUICK_SCAN_REVIEWS,
)
MAX_BROWSER_REVIEW_PAGES = _bounded_int_setting(
    "MAX_BROWSER_REVIEW_PAGES",
    3,
    minimum=1,
    maximum=3,
)
DEFAULT_NAVIGATION_TIMEOUT_MS = _bounded_int_setting(
    "AMAZON_BROWSER_NAVIGATION_TIMEOUT_MS",
    25_000,
    minimum=20_000,
    maximum=30_000,
)
BROWSER_RUNTIME_INSTALL_MESSAGE = (
    "Browser runtime is not installed. Run: playwright install chromium"
)


class BrowserImportStatus:
    """Stable statuses shared by the browser importer, metadata, and UI."""

    SUCCESS = "success"
    PRODUCT_PAGE_LOADED = "product_page_loaded"
    CHALLENGE_REQUIRED = "challenge_required"
    LOGIN_REQUIRED = "login_required"
    REVIEWS_UNAVAILABLE = "reviews_unavailable"
    NO_REVIEWS_FOUND = "no_reviews_found"
    PAGE_STRUCTURE_CHANGED = "page_structure_changed"
    NETWORK_ERROR = "network_error"
    NAVIGATION_TIMEOUT = "navigation_timeout"
    BROWSER_UNAVAILABLE = "browser_unavailable"
    PARSE_ERROR = "parse_error"
    CANCELLED = "cancelled"


class ReviewAvailabilityStatus:
    """Non-fatal availability meaning for one bounded Quick Scan."""

    ACCESSIBLE = "accessible"
    ACCESS_LIMITED = "access_limited"
    NO_REVIEWS = "no_reviews"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ReviewCollectionStopReason:
    """Why a bounded Quick Scan stopped collecting unique reviews."""

    NO_NEXT_PAGE = "no_next_page"
    NO_NEW_REVIEWS = "no_new_reviews"
    AMAZON_ACCESS_LIMITED = "amazon_access_limited"
    PRODUCT_EXHAUSTED = "product_exhausted"
    NAVIGATION_FAILED = "navigation_failed"
    SAFETY_LIMIT_REACHED = "safety_limit_reached"
    CHALLENGE_REQUIRED = "challenge_required"
    LOGIN_REQUIRED = "login_required"


def quick_scan_availability(
    unique_review_count: int,
    stop_reason: str | None,
) -> tuple[str, str | None]:
    """Return conservative availability semantics without guessing totals."""

    if unique_review_count < 0:
        raise ValueError("unique_review_count cannot be negative")
    if unique_review_count == 0:
        if stop_reason == ReviewCollectionStopReason.PRODUCT_EXHAUSTED:
            return (
                ReviewAvailabilityStatus.NO_REVIEWS,
                "Amazon reported no currently available public reviews.",
            )
        if stop_reason in {
            ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED,
            ReviewCollectionStopReason.CHALLENGE_REQUIRED,
            ReviewCollectionStopReason.LOGIN_REQUIRED,
        }:
            return (
                ReviewAvailabilityStatus.ACCESS_LIMITED,
                "Amazon did not expose reviews to the current browser session.",
            )
        return ReviewAvailabilityStatus.UNKNOWN, None

    if stop_reason in {
        ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED,
        ReviewCollectionStopReason.CHALLENGE_REQUIRED,
        ReviewCollectionStopReason.LOGIN_REQUIRED,
    }:
        return (
            ReviewAvailabilityStatus.ACCESS_LIMITED,
            "Amazon exposed a limited set of reviews to the current browser session.",
        )
    if stop_reason in {
        ReviewCollectionStopReason.NO_NEXT_PAGE,
        ReviewCollectionStopReason.NO_NEW_REVIEWS,
        ReviewCollectionStopReason.NAVIGATION_FAILED,
        ReviewCollectionStopReason.SAFETY_LIMIT_REACHED,
    }:
        return (
            ReviewAvailabilityStatus.PARTIAL,
            "Quick Scan retained the unique reviews currently accessible to this session.",
        )
    return (
        ReviewAvailabilityStatus.ACCESSIBLE,
        "Quick Scan reached the end of the currently accessible review stream.",
    )


@dataclass(frozen=True)
class AmazonBrowserImportResult:
    """Sanitized result returned to Streamlit and persistence code."""

    status: str
    marketplace: str
    asin: str
    product_url: str
    reviews_url: str
    reviews: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None
    challenge_detected: bool = False
    login_required: bool = False
    pages_visited: int = 0
    current_url: str | None = None
    page_title: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    retrieved_review_count: int | None = None
    unique_review_count: int | None = None
    target_review_count: int | None = None
    availability_status: str = ReviewAvailabilityStatus.UNKNOWN
    availability_reason: str | None = None
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        """Fill compatibility counts while keeping the result immutable."""

        unique_count = len(self.reviews)
        if self.unique_review_count is None:
            object.__setattr__(self, "unique_review_count", unique_count)
        if self.retrieved_review_count is None:
            object.__setattr__(self, "retrieved_review_count", unique_count)

    @property
    def review_count(self) -> int:
        """Backward-compatible alias for the deduplicated review count."""

        return int(self.unique_review_count or 0)

    def to_metadata(self) -> dict[str, Any]:
        """Return an allowlisted, non-sensitive browser import record."""

        return {
            "marketplace": self.marketplace,
            "asin": self.asin,
            "product_url": self.product_url,
            "reviews_url": self.reviews_url,
            "review_count": self.review_count,
            "retrieved_review_count": int(self.retrieved_review_count or 0),
            "unique_review_count": self.review_count,
            "target_review_count": self.target_review_count,
            "availability_status": self.availability_status,
            "availability_reason": self.availability_reason,
            "stop_reason": self.stop_reason,
            "pages_visited": self.pages_visited,
            "challenge_encountered": self.challenge_detected,
            "login_required": self.login_required,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class _Command:
    name: str
    arguments: tuple[Any, ...]
    reply: Queue[Any]
    events: Queue[tuple[int, int, str]]


BrowserProgressCallback = Callable[[int, int, str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_playwright_factory() -> Any:
    """Import Playwright lazily so Demo mode never requires its runtime."""

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - covered through injection
        raise RuntimeError(BROWSER_RUNTIME_INSTALL_MESSAGE) from exc
    return sync_playwright()


def _is_runtime_missing(error: BaseException) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "executable doesn't exist",
            "executable does not exist",
            "playwright install chromium",
            "browser runtime is not installed",
        )
    )


def _is_timeout(error: BaseException) -> bool:
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    return "timeout" in name or "timed out" in message or "timeout" in message


def _is_browser_closed(error: BaseException) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "target page, context or browser has been closed",
            "browser has been closed",
            "page has been closed",
        )
    )


def _safe_error_message(error: BaseException, *, fallback: str) -> str:
    message = re.sub(r"\s+", " ", str(error)).strip()
    if not message:
        return fallback
    # Do not expose an arbitrarily large browser diagnostic or page fragment.
    return message[:300]


def detect_login_required(html: object, current_url: object = "") -> bool:
    """Detect an Amazon sign-in page without interacting with it."""

    text = html.casefold() if isinstance(html, str) else ""
    url = str(current_url or "").casefold()
    url_markers = ("/ap/signin", "/gp/sign-in", "/signin")
    html_markers = (
        'name="signin"',
        "name='signin'",
        'id="ap_email"',
        "id='ap_email'",
        "sign in to your amazon account",
        "authentication required",
    )
    if any(marker in url for marker in url_markers):
        return True
    # Amazon can embed a dormant sign-in form in the header or account flyout
    # of a fully loaded product/review page.  A recognizable normal page takes
    # precedence unless the browser was actually redirected to a sign-in URL.
    normal_page = detect_product_page(html) or any(
        marker in text
        for marker in (
            'data-hook="review"',
            "data-hook='review'",
            "customer_review-",
        )
    )
    return not normal_page and any(marker in text for marker in html_markers)


def detect_product_page(html: object) -> bool:
    """Recognize a normal product page using several public DOM fallbacks."""

    if not isinstance(html, str):
        return False
    lowered = html.casefold()
    markers = (
        'id="producttitle"',
        "id='producttitle'",
        'data-feature-name="title"',
        "data-feature-name='title'",
        'name="asin"',
        "name='asin'",
        'id="dp"',
        "id='dp'",
    )
    return any(marker in lowered for marker in markers)


def detect_reviews_unavailable(html: object) -> bool:
    """Recognize explicit messages that reviews cannot be displayed."""

    if not isinstance(html, str):
        return False
    lowered = re.sub(r"\s+", " ", html.casefold())
    return any(
        marker in lowered
        for marker in (
            "reviews are not available for this item",
            "customer reviews are not available",
            "this product is no longer available",
            "looking for something? we're sorry. the web address you entered",
        )
    )


def detect_browser_challenge(html: object, current_url: object = "") -> bool:
    """Detect a visible challenge using strong browser-page signals only.

    Full Amazon product DOMs can contain generic phrases such as "access
    denied" inside scripts, policies, or user reviews.  Those phrases are
    useful for the small HTTP response parser but are not sufficient to label
    an otherwise recognizable product page as a browser challenge.
    """

    text = html.casefold() if isinstance(html, str) else ""
    url = str(current_url or "").casefold()
    if "validatecaptcha" in url:
        return True
    strong_markers = (
        "validatecaptcha",
        'id="captchacharacters"',
        "id='captchacharacters'",
        'name="captchacharacters"',
        "name='captchacharacters'",
        "enter the characters you see below",
        "type the characters you see in this image",
        "sorry, we just need to make sure you're not a robot",
        "sorry, we just need to make sure you’re not a robot",
        "<title>robot check</title>",
    )
    return any(marker in text for marker in strong_markers)


def classify_browser_page(
    html: object,
    *,
    current_url: object = "",
    expect_reviews: bool = False,
) -> str:
    """Classify a fetched page without attempting any access-control bypass."""

    if detect_login_required(html, current_url):
        return BrowserImportStatus.LOGIN_REQUIRED
    if detect_browser_challenge(html, current_url):
        return BrowserImportStatus.CHALLENGE_REQUIRED
    if expect_reviews and detect_reviews_unavailable(html):
        return BrowserImportStatus.REVIEWS_UNAVAILABLE
    if expect_reviews:
        return BrowserImportStatus.SUCCESS
    if detect_product_page(html):
        return BrowserImportStatus.PRODUCT_PAGE_LOADED
    return BrowserImportStatus.PAGE_STRUCTURE_CHANGED


def _safe_review_page_url(
    base_url: str,
    href: str,
    marketplace: str,
    asin: str,
) -> str | None:
    candidate = urljoin(base_url, href)
    try:
        parsed = urlsplit(candidate)
    except (TypeError, ValueError):
        return None
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if parsed.scheme.casefold() != "https":
        return None
    if hostname != marketplace and not hostname.endswith(f".{marketplace}"):
        return None
    expected_paths = (
        re.compile(
            rf"(?:^|/)product-reviews/{re.escape(asin)}(?:/|$)",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?:^|/)portal/customer-reviews/{re.escape(asin)}(?:/|$)",
            re.IGNORECASE,
        ),
    )
    return (
        candidate
        if any(pattern.search(parsed.path) for pattern in expected_paths)
        else None
    )


class AmazonBrowserSession:
    """Own one ephemeral Playwright browser on a dedicated worker thread."""

    def __init__(
        self,
        product: AmazonProductURL,
        *,
        playwright_factory: Callable[[], Any] | None = None,
        navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
        network_retries: int = 1,
    ) -> None:
        if navigation_timeout_ms <= 0:
            raise ValueError("navigation_timeout_ms must be positive")
        if network_retries not in (0, 1):
            raise ValueError("network_retries must be 0 or 1")
        self.product = product
        self.navigation_timeout_ms = navigation_timeout_ms
        self.network_retries = network_retries
        self.started_at = _utc_now()
        self._playwright_factory = playwright_factory or _default_playwright_factory
        self._commands: Queue[_Command] = Queue()
        self._thread = threading.Thread(
            target=self._worker,
            name=f"amazon-browser-{product.asin}",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    @property
    def is_closed(self) -> bool:
        return self._closed or not self._thread.is_alive()

    def _submit(
        self,
        name: str,
        *arguments: Any,
        progress_callback: BrowserProgressCallback | None = None,
    ) -> AmazonBrowserImportResult:
        if self.is_closed and name != "close":
            return self._base_result(
                BrowserImportStatus.CANCELLED,
                error_message="The browser import session has been closed.",
            )
        reply: Queue[Any] = Queue(maxsize=1)
        events: Queue[tuple[int, int, str]] = Queue()
        self._commands.put(_Command(name, arguments, reply, events))
        while True:
            try:
                outcome = reply.get(timeout=0.05)
                break
            except Empty:
                self._drain_events(events, progress_callback)
                if not self._thread.is_alive():
                    return self._base_result(
                        BrowserImportStatus.BROWSER_UNAVAILABLE,
                        error_message="The browser session ended unexpectedly.",
                    )
        self._drain_events(events, progress_callback)
        if isinstance(outcome, BaseException):
            return self._result_for_exception(outcome)
        return outcome

    @staticmethod
    def _drain_events(
        events: Queue[tuple[int, int, str]],
        callback: BrowserProgressCallback | None,
    ) -> None:
        while True:
            try:
                event = events.get_nowait()
            except Empty:
                return
            if callback is not None:
                try:
                    callback(*event)
                except Exception:
                    pass

    def open_product(self) -> AmazonBrowserImportResult:
        return self._submit("open_product")

    def continue_after_verification(self) -> AmazonBrowserImportResult:
        return self._submit("continue_after_verification")

    def load_reviews(
        self,
        *,
        limit: int = DEFAULT_BROWSER_REVIEW_LIMIT,
        max_pages: int = MAX_BROWSER_REVIEW_PAGES,
        progress_callback: BrowserProgressCallback | None = None,
    ) -> AmazonBrowserImportResult:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_AMAZON_QUICK_SCAN_REVIEWS
        ):
            raise ValueError(
                "limit must be a positive internal safety cap no greater than "
                f"{MAX_AMAZON_QUICK_SCAN_REVIEWS}"
            )
        if not 1 <= max_pages <= MAX_BROWSER_REVIEW_PAGES:
            raise ValueError(
                f"max_pages must be between 1 and {MAX_BROWSER_REVIEW_PAGES}"
            )
        return self._submit(
            "load_reviews",
            limit,
            max_pages,
            progress_callback=progress_callback,
        )

    def close(self) -> AmazonBrowserImportResult:
        if self._closed:
            return self._base_result(BrowserImportStatus.CANCELLED)
        result = self._submit("close")
        self._closed = True
        self._thread.join(timeout=5)
        return result

    def _worker(self) -> None:
        playwright_manager = None
        playwright = None
        browser = None
        context = None
        page = None
        challenge_encountered = False
        login_encountered = False
        pages_visited = 0
        pending_review_verification = False

        def snapshot() -> tuple[str, str, str]:
            if page is None:
                return "", "", ""
            html = page.content()
            current_url = str(getattr(page, "url", "") or "")
            try:
                title = str(page.title() or "")
            except Exception:
                title = ""
            return html, current_url, title

        def result(
            status: str,
            *,
            reviews: list[dict[str, Any]] | None = None,
            error_message: str | None = None,
            current_url: str | None = None,
            page_title: str | None = None,
            retrieved_review_count: int | None = None,
            stop_reason: str | None = None,
            availability_status: str | None = None,
            availability_reason: str | None = None,
        ) -> AmazonBrowserImportResult:
            safe_reviews = reviews or []
            unique_count = len(safe_reviews)
            if availability_status is None and stop_reason is not None:
                availability_status, inferred_reason = quick_scan_availability(
                    unique_count,
                    stop_reason,
                )
                availability_reason = availability_reason or inferred_reason
            return AmazonBrowserImportResult(
                status=status,
                marketplace=self.product.marketplace,
                asin=self.product.asin,
                product_url=self.product.canonical_product_url,
                reviews_url=build_review_page_url(
                    self.product.marketplace, self.product.asin, 1
                ),
                reviews=safe_reviews,
                error_message=error_message,
                challenge_detected=challenge_encountered,
                login_required=login_encountered,
                pages_visited=pages_visited,
                current_url=current_url,
                page_title=page_title,
                started_at=self.started_at,
                completed_at=_utc_now(),
                retrieved_review_count=retrieved_review_count,
                unique_review_count=unique_count,
                target_review_count=None,
                availability_status=(
                    availability_status or ReviewAvailabilityStatus.UNKNOWN
                ),
                availability_reason=availability_reason,
                stop_reason=stop_reason,
            )

        def navigate(url: str) -> BaseException | None:
            assert page is not None
            last_error: BaseException | None = None
            for attempt in range(self.network_retries + 1):
                try:
                    page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=self.navigation_timeout_ms,
                    )
                    return None
                except BaseException as exc:  # Playwright has runtime subclasses
                    last_error = exc
                    if _is_runtime_missing(exc):
                        return exc
                    if attempt >= self.network_retries:
                        return exc
            return last_error

        def inspect_current(expect_reviews: bool = False) -> AmazonBrowserImportResult:
            nonlocal challenge_encountered, login_encountered
            try:
                html, current_url, title = snapshot()
            except BaseException as exc:
                return self._result_for_exception(exc)
            state = classify_browser_page(
                html,
                current_url=current_url,
                expect_reviews=expect_reviews,
            )
            challenge_encountered = challenge_encountered or (
                state == BrowserImportStatus.CHALLENGE_REQUIRED
            )
            login_encountered = login_encountered or (
                state == BrowserImportStatus.LOGIN_REQUIRED
            )
            return result(state, current_url=current_url, page_title=title)

        def cleanup() -> None:
            for resource in (page, context, browser):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        pass
            if playwright is not None:
                try:
                    playwright.stop()
                except Exception:
                    pass
            elif playwright_manager is not None and hasattr(
                playwright_manager, "__exit__"
            ):
                try:
                    playwright_manager.__exit__(None, None, None)
                except Exception:
                    pass

        while True:
            command = self._commands.get()
            try:
                if command.name == "open_product":
                    if page is None:
                        playwright_manager = self._playwright_factory()
                        if hasattr(playwright_manager, "start"):
                            playwright = playwright_manager.start()
                        elif hasattr(playwright_manager, "__enter__"):
                            playwright = playwright_manager.__enter__()
                        else:
                            playwright = playwright_manager
                        browser = playwright.chromium.launch(headless=False)
                        context = browser.new_context()
                        page = context.new_page()
                        if hasattr(page, "set_default_navigation_timeout"):
                            page.set_default_navigation_timeout(
                                self.navigation_timeout_ms
                            )
                    navigation_error = navigate(
                        self.product.canonical_product_url
                    )
                    if navigation_error is not None:
                        command.reply.put(self._result_for_exception(navigation_error))
                    else:
                        command.reply.put(inspect_current())
                elif command.name == "continue_after_verification":
                    if page is None:
                        command.reply.put(
                            result(
                                BrowserImportStatus.BROWSER_UNAVAILABLE,
                                error_message="No browser page is open.",
                            )
                        )
                    else:
                        checked = inspect_current(
                            expect_reviews=pending_review_verification
                        )
                        if (
                            pending_review_verification
                            and checked.status == BrowserImportStatus.SUCCESS
                        ):
                            # Verification happened on a review page.  Re-enable
                            # the explicit Load Reviews action; it will parse the
                            # page through the normal bounded import path.
                            pending_review_verification = False
                            html, current_url, title = snapshot()
                            del html
                            checked = result(
                                BrowserImportStatus.PRODUCT_PAGE_LOADED,
                                current_url=current_url,
                                page_title=title,
                            )
                        command.reply.put(checked)
                elif command.name == "load_reviews":
                    if page is None:
                        command.reply.put(
                            result(
                                BrowserImportStatus.BROWSER_UNAVAILABLE,
                                error_message="No browser page is open.",
                            )
                        )
                        continue
                    limit, max_pages = command.arguments
                    current_url = build_review_page_url(
                        self.product.marketplace, self.product.asin, 1
                    )
                    collected: list[dict[str, Any]] = []
                    visited_urls: set[str] = set()
                    final_result: AmazonBrowserImportResult | None = None
                    retrieved_review_count = 0
                    stop_reason: str | None = None

                    for page_number in range(1, max_pages + 1):
                        if current_url in visited_urls:
                            if collected:
                                stop_reason = ReviewCollectionStopReason.NO_NEW_REVIEWS
                            else:
                                final_result = result(
                                    BrowserImportStatus.PARSE_ERROR,
                                    error_message=(
                                        "Amazon review pagination repeated a page."
                                    ),
                                    stop_reason=(
                                        ReviewCollectionStopReason.NAVIGATION_FAILED
                                    ),
                                )
                            break
                        visited_urls.add(current_url)
                        command.events.put(
                            (page_number, len(collected), "opening_review_page")
                        )
                        navigation_error = navigate(current_url)
                        if navigation_error is not None:
                            if collected:
                                stop_reason = (
                                    ReviewCollectionStopReason.NAVIGATION_FAILED
                                )
                            else:
                                final_result = self._result_for_exception(
                                    navigation_error,
                                    pages_visited=pages_visited,
                                    stop_reason=(
                                        ReviewCollectionStopReason.NAVIGATION_FAILED
                                    ),
                                )
                            break
                        pages_visited += 1
                        html, actual_url, title = snapshot()
                        state = classify_browser_page(
                            html,
                            current_url=actual_url,
                            expect_reviews=True,
                        )
                        if state == BrowserImportStatus.CHALLENGE_REQUIRED:
                            challenge_encountered = True
                            if collected:
                                stop_reason = (
                                    ReviewCollectionStopReason.CHALLENGE_REQUIRED
                                )
                            else:
                                pending_review_verification = True
                                final_result = result(
                                    state,
                                    current_url=actual_url,
                                    page_title=title,
                                    retrieved_review_count=retrieved_review_count,
                                    stop_reason=(
                                        ReviewCollectionStopReason.CHALLENGE_REQUIRED
                                    ),
                                )
                            break
                        if state == BrowserImportStatus.LOGIN_REQUIRED:
                            login_encountered = True
                            if collected:
                                stop_reason = ReviewCollectionStopReason.LOGIN_REQUIRED
                            else:
                                pending_review_verification = True
                                final_result = result(
                                    state,
                                    current_url=actual_url,
                                    page_title=title,
                                    retrieved_review_count=retrieved_review_count,
                                    stop_reason=(
                                        ReviewCollectionStopReason.LOGIN_REQUIRED
                                    ),
                                )
                            break
                        if state == BrowserImportStatus.REVIEWS_UNAVAILABLE:
                            if collected:
                                stop_reason = (
                                    ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED
                                )
                            else:
                                final_result = result(
                                    state,
                                    current_url=actual_url,
                                    page_title=title,
                                    retrieved_review_count=(
                                        retrieved_review_count
                                    ),
                                    stop_reason=(
                                        ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED
                                    ),
                                )
                            break
                        parsed = parse_review_page(
                            html,
                            asin=self.product.asin,
                            marketplace=self.product.marketplace,
                            source_url=actual_url or current_url,
                            page_number=page_number,
                        )
                        if parsed.status == ScrapeStatus.BLOCKED:
                            if collected:
                                stop_reason = (
                                    ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED
                                )
                            else:
                                final_result = result(
                                    BrowserImportStatus.REVIEWS_UNAVAILABLE,
                                    error_message=(
                                        parsed.reason
                                        or "Amazon limited review-page access."
                                    ),
                                    current_url=actual_url,
                                    page_title=title,
                                    retrieved_review_count=(
                                        retrieved_review_count
                                    ),
                                    stop_reason=(
                                        ReviewCollectionStopReason.AMAZON_ACCESS_LIMITED
                                    ),
                                )
                            break
                        if parsed.status == ScrapeStatus.NO_REVIEWS:
                            if collected:
                                stop_reason = (
                                    ReviewCollectionStopReason.PRODUCT_EXHAUSTED
                                )
                            else:
                                final_result = result(
                                    BrowserImportStatus.NO_REVIEWS_FOUND,
                                    current_url=actual_url,
                                    page_title=title,
                                    retrieved_review_count=(
                                        retrieved_review_count
                                    ),
                                    stop_reason=(
                                        ReviewCollectionStopReason.PRODUCT_EXHAUSTED
                                    ),
                                )
                            break
                        if parsed.status == ScrapeStatus.STRUCTURE_NOT_RECOGNIZED:
                            if collected:
                                stop_reason = (
                                    ReviewCollectionStopReason.NAVIGATION_FAILED
                                )
                            else:
                                final_result = result(
                                    BrowserImportStatus.PAGE_STRUCTURE_CHANGED,
                                    error_message=(
                                        "Amazon review containers were not recognized."
                                    ),
                                    current_url=actual_url,
                                    page_title=title,
                                    stop_reason=(
                                        ReviewCollectionStopReason.NAVIGATION_FAILED
                                    ),
                                )
                            break
                        if parsed.status != ScrapeStatus.SUCCESS:
                            if collected:
                                stop_reason = (
                                    ReviewCollectionStopReason.NAVIGATION_FAILED
                                )
                            else:
                                final_result = result(
                                    BrowserImportStatus.PARSE_ERROR,
                                    error_message=(
                                        parsed.reason
                                        or "Review page could not be parsed."
                                    ),
                                    current_url=actual_url,
                                    page_title=title,
                                    stop_reason=(
                                        ReviewCollectionStopReason.NAVIGATION_FAILED
                                    ),
                                )
                            break
                        valid_reviews = [
                            review
                            for review in parsed.reviews
                            if str(review.get("content") or "").strip()
                        ]
                        retrieved_review_count += len(valid_reviews)
                        previous_unique_count = len(collected)
                        collected.extend(valid_reviews)
                        collected = deduplicate_reviews(collected)
                        command.events.put(
                            (page_number, len(collected), "reviews_found")
                        )
                        if len(collected) >= limit:
                            stop_reason = (
                                ReviewCollectionStopReason.SAFETY_LIMIT_REACHED
                            )
                            break
                        if len(collected) == previous_unique_count:
                            stop_reason = ReviewCollectionStopReason.NO_NEW_REVIEWS
                            break
                        if not parsed.next_page_url:
                            stop_reason = ReviewCollectionStopReason.NO_NEXT_PAGE
                            break
                        next_url = _safe_review_page_url(
                            actual_url or current_url,
                            parsed.next_page_url,
                            self.product.marketplace,
                            self.product.asin,
                        )
                        if next_url is None:
                            if collected:
                                stop_reason = (
                                    ReviewCollectionStopReason.NAVIGATION_FAILED
                                )
                            else:
                                final_result = result(
                                    BrowserImportStatus.PARSE_ERROR,
                                    error_message=(
                                        "Amazon returned an unsafe review pagination URL."
                                    ),
                                    current_url=actual_url,
                                    page_title=title,
                                    stop_reason=(
                                        ReviewCollectionStopReason.NAVIGATION_FAILED
                                    ),
                                )
                            break
                        current_url = next_url
                    else:
                        stop_reason = ReviewCollectionStopReason.SAFETY_LIMIT_REACHED

                    if final_result is None:
                        collected = deduplicate_reviews(collected)[:limit]
                        if collected:
                            final_result = result(
                                BrowserImportStatus.SUCCESS,
                                reviews=collected,
                                current_url=str(getattr(page, "url", "") or ""),
                                page_title=str(page.title() or ""),
                                retrieved_review_count=retrieved_review_count,
                                stop_reason=(
                                    stop_reason
                                    or ReviewCollectionStopReason.NO_NEXT_PAGE
                                ),
                            )
                        else:
                            final_result = result(
                                BrowserImportStatus.NO_REVIEWS_FOUND,
                                current_url=str(getattr(page, "url", "") or ""),
                                page_title=str(page.title() or ""),
                                retrieved_review_count=retrieved_review_count,
                                stop_reason=(
                                    stop_reason
                                    or ReviewCollectionStopReason.PRODUCT_EXHAUSTED
                                ),
                            )
                    command.reply.put(final_result)
                elif command.name == "close":
                    cleanup()
                    command.reply.put(result(BrowserImportStatus.CANCELLED))
                    return
                else:
                    command.reply.put(RuntimeError("Unknown browser command."))
            except BaseException as exc:
                if command.name == "open_product" and page is None:
                    message = (
                        BROWSER_RUNTIME_INSTALL_MESSAGE
                        if _is_runtime_missing(exc)
                        or isinstance(exc, (ImportError, ModuleNotFoundError))
                        else "The visible Chromium browser could not be started."
                    )
                    command.reply.put(
                        result(
                            BrowserImportStatus.BROWSER_UNAVAILABLE,
                            error_message=message,
                        )
                    )
                else:
                    command.reply.put(self._result_for_exception(exc))

    def _base_result(
        self,
        status: str,
        *,
        error_message: str | None = None,
        pages_visited: int = 0,
        stop_reason: str | None = None,
    ) -> AmazonBrowserImportResult:
        availability_status = ReviewAvailabilityStatus.UNKNOWN
        availability_reason: str | None = None
        if stop_reason is not None:
            availability_status, availability_reason = quick_scan_availability(
                0,
                stop_reason,
            )
        return AmazonBrowserImportResult(
            status=status,
            marketplace=self.product.marketplace,
            asin=self.product.asin,
            product_url=self.product.canonical_product_url,
            reviews_url=build_review_page_url(
                self.product.marketplace, self.product.asin, 1
            ),
            error_message=error_message,
            pages_visited=pages_visited,
            started_at=self.started_at,
            completed_at=_utc_now(),
            availability_status=availability_status,
            availability_reason=availability_reason,
            stop_reason=stop_reason,
        )

    def _result_for_exception(
        self,
        error: BaseException,
        *,
        pages_visited: int = 0,
        stop_reason: str | None = None,
    ) -> AmazonBrowserImportResult:
        if _is_browser_closed(error):
            status = BrowserImportStatus.CANCELLED
            message = "The visible Amazon browser was closed."
        elif _is_runtime_missing(error) or isinstance(error, (ImportError, ModuleNotFoundError)):
            status = BrowserImportStatus.BROWSER_UNAVAILABLE
            message = BROWSER_RUNTIME_INSTALL_MESSAGE
        elif _is_timeout(error):
            status = BrowserImportStatus.NAVIGATION_TIMEOUT
            message = "Amazon page navigation timed out."
        else:
            status = BrowserImportStatus.NETWORK_ERROR
            message = _safe_error_message(
                error,
                fallback="The browser could not reach Amazon.",
            )
        return self._base_result(
            status,
            error_message=message,
            pages_visited=pages_visited,
            stop_reason=stop_reason,
        )


def open_amazon_product(
    product: AmazonProductURL,
    *,
    playwright_factory: Callable[[], Any] | None = None,
    navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
) -> tuple[AmazonBrowserSession, AmazonBrowserImportResult]:
    """Create one ephemeral browser session and open the canonical product."""

    session = AmazonBrowserSession(
        product,
        playwright_factory=playwright_factory,
        navigation_timeout_ms=navigation_timeout_ms,
    )
    return session, session.open_product()


__all__ = [
    "ALLOWED_REVIEW_LIMITS",
    "AmazonBrowserImportResult",
    "AmazonBrowserSession",
    "BROWSER_RUNTIME_INSTALL_MESSAGE",
    "BrowserImportStatus",
    "DEFAULT_BROWSER_REVIEW_LIMIT",
    "DEFAULT_NAVIGATION_TIMEOUT_MS",
    "MAX_AMAZON_QUICK_SCAN_REVIEWS",
    "MAX_BROWSER_REVIEW_PAGES",
    "ReviewAvailabilityStatus",
    "ReviewCollectionStopReason",
    "classify_browser_page",
    "detect_login_required",
    "detect_browser_challenge",
    "detect_product_page",
    "detect_reviews_unavailable",
    "open_amazon_product",
    "quick_scan_availability",
]
