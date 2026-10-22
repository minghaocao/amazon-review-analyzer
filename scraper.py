"""Low-frequency, compliance-first Amazon public review importer.

The network and HTML parsing layers are deliberately separate. Unit tests can
therefore exercise every parser branch using local fixtures without contacting
Amazon. This module never signs in, submits challenge forms, manages cookies,
or attempts to bypass an Amazon access-control page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
from functools import lru_cache
from hashlib import sha256
from html.parser import HTMLParser
from http.client import responses as HTTP_STATUS_MESSAGES
import os
from pathlib import Path
import re
import socket
import ssl
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

import certifi

from amazon_url import SUPPORTED_MARKETPLACES


DEFAULT_REQUEST_TIMEOUT_SECONDS = 15.0
DEFAULT_PAGE_DELAY_SECONDS = 1.0
DEFAULT_MAX_PAGES = 20
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
TRUSTED_CA_BUNDLE_UNAVAILABLE = "trusted_ca_bundle_unavailable"
TRUSTED_CA_BUNDLE_UNAVAILABLE_MESSAGE = (
    "Trusted CA certificate bundle is unavailable."
)


class ScrapeStatus:
    """Stable string values used by the scraper, metadata, and UI."""

    SUCCESS = "success"
    BLOCKED = "blocked"
    NO_REVIEWS = "no_reviews"
    STRUCTURE_NOT_RECOGNIZED = "page_structure_not_recognized"
    DNS_ERROR = "dns_error"
    CONNECTION_TIMEOUT = "connection_timeout"
    # Backward-compatible name used by existing callers and tests.
    TIMEOUT = CONNECTION_TIMEOUT
    CONNECTION_REFUSED = "connection_refused"
    TLS_ERROR = "tls_error"
    PROXY_ERROR = "proxy_error"
    REDIRECT_ERROR = "redirect_error"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    RESPONSE_TOO_LARGE = "response_too_large"
    PARSE_ERROR = "parse_error"


class TrustedCABundleError(RuntimeError):
    """The project could not create a securely verified HTTPS context."""


@lru_cache(maxsize=1)
def get_verified_ssl_context() -> ssl.SSLContext:
    """Return the shared HTTPS context backed by certifi's trusted CA bundle.

    Context creation is intentionally lazy so a missing or unreadable bundle
    becomes a structured scraper failure instead of preventing the project
    from importing.  The function never falls back to an unverified context.
    """

    try:
        bundle = Path(certifi.where())
    except (OSError, TypeError, ValueError) as exc:
        raise TrustedCABundleError(
            TRUSTED_CA_BUNDLE_UNAVAILABLE_MESSAGE
        ) from exc

    if not bundle.is_file() or not os.access(bundle, os.R_OK):
        raise TrustedCABundleError(TRUSTED_CA_BUNDLE_UNAVAILABLE_MESSAGE)

    try:
        context = ssl.create_default_context(cafile=str(bundle))
    except (OSError, TypeError, ValueError, ssl.SSLError) as exc:
        raise TrustedCABundleError(
            TRUSTED_CA_BUNDLE_UNAVAILABLE_MESSAGE
        ) from exc

    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        raise TrustedCABundleError(
            "Secure TLS certificate verification could not be enabled."
        )
    return context


@dataclass(frozen=True)
class FetchResult:
    """Result of one HTTP GET; failures are data rather than UI tracebacks."""

    status: str
    url: str
    html: str = ""
    reason: str | None = None
    http_status: int | None = None
    requested_url: str | None = None
    content_type: str | None = None
    response_size: int = 0
    exception_type: str | None = None
    error_message: str | None = None

    @property
    def final_url(self) -> str:
        """Return the final URL while preserving the historical ``url`` field."""

        return self.url

    @property
    def redirected(self) -> bool:
        return bool(self.requested_url and self.requested_url != self.url)


@dataclass(frozen=True)
class ParseResult:
    """Structured result from one Amazon review HTML page."""

    status: str
    reviews: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    next_page_url: str | None = None


@dataclass(frozen=True)
class ScrapeResult:
    """Result of a bounded, serial review-page import."""

    status: str
    reviews: list[dict[str, Any]]
    pages_fetched: int
    requested_limit: int
    asin: str
    marketplace: str
    source_url: str
    reason: str | None = None

    @property
    def reviews_collected(self) -> int:
        return len(self.reviews)

    def to_metadata(self) -> dict[str, Any]:
        """Return only non-sensitive fields suitable for scrape_metadata.json."""

        return {
            "asin": self.asin,
            "marketplace": self.marketplace,
            "source_url": self.source_url,
            "requested_limit": self.requested_limit,
            "reviews_collected": self.reviews_collected,
            "scrape_status": self.status,
            "reason": self.reason,
            "pages_fetched": self.pages_fetched,
        }


ScrapeProgressCallback = Callable[[int, int, str], None]
PageFetcher = Callable[[str], FetchResult | str]


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str]
    parent: "_Node | None" = None
    children: list["_Node | str"] = field(default_factory=list)


class _DocumentParser(HTMLParser):
    """Tiny DOM builder for the small set of selectors used by review pages."""

    _VOID_ELEMENTS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document", {})
        self._stack = [self.root]

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        node = _Node(
            tag.casefold(),
            {name.casefold(): value or "" for name, value in attrs},
            self._stack[-1],
        )
        self._stack[-1].children.append(node)
        if tag.casefold() not in self._VOID_ELEMENTS:
            self._stack.append(node)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() not in self._VOID_ELEMENTS:
            self._stack.pop()

    def handle_endtag(self, tag: str) -> None:
        wanted = tag.casefold()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == wanted:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self._stack[-1].children.append(data)


def _walk(node: _Node) -> Iterable[_Node]:
    for child in node.children:
        if isinstance(child, _Node):
            yield child
            yield from _walk(child)


def _has_class(node: _Node | None, class_name: str) -> bool:
    return bool(node and class_name in node.attrs.get("class", "").split())


def _find_all(
    node: _Node,
    predicate: Callable[[_Node], bool],
) -> list[_Node]:
    return [candidate for candidate in _walk(node) if predicate(candidate)]


def _find_first(
    node: _Node,
    predicate: Callable[[_Node], bool],
) -> _Node | None:
    return next((candidate for candidate in _walk(node) if predicate(candidate)), None)


def _text(
    node: _Node | None,
    *,
    exclude: Callable[[_Node], bool] | None = None,
) -> str:
    if node is None:
        return ""
    pieces: list[str] = []

    def collect(current: _Node) -> None:
        for child in current.children:
            if isinstance(child, str):
                pieces.append(child)
            elif exclude is None or not exclude(child):
                collect(child)

    collect(node)
    return re.sub(r"\s+", " ", " ".join(pieces)).strip()


def _hook(name: str) -> Callable[[_Node], bool]:
    return lambda node: node.attrs.get("data-hook") == name


def parse_rating(value: object) -> float | None:
    """Parse localized Amazon star text such as ``5,0 von 5 Sternen``."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    localized_pattern = (
        r"(?<!\d)(\d+(?:[.,]\d+)?)\s*"
        r"(?:out\s+of|von|sur|su|de)\s+5\b"
    )
    match = re.search(localized_pattern, text, re.IGNORECASE)
    if match is None:
        match = re.search(
            r"(?<!\d)(\d+(?:[.,]\d+)?)\s*"
            r"(?:(?:stars?|sternen|étoiles|stelle|estrellas)\b|$)",
            text,
            re.IGNORECASE,
        )
    if match is None:
        return None
    try:
        rating = float(match.group(1).replace(",", "."))
    except ValueError:
        return None
    return rating if 1 <= rating <= 5 else None


def _normalized_content(content: object) -> str:
    if content is None:
        return ""
    return re.sub(r"\s+", " ", str(content)).strip().casefold()


def _content_hash(content: object) -> str:
    return sha256(_normalized_content(content).encode("utf-8")).hexdigest()


def generate_review_id(
    asin: str,
    content: object,
    *,
    page_number: int = 1,
    position: int = 1,
    title: object = None,
    date: object = None,
) -> str:
    """Generate a deterministic fallback ID without using a random UUID."""

    normalized = _normalized_content(content)
    if normalized:
        material = f"{asin.upper()}|{normalized}"
    else:
        material = "|".join(
            (
                asin.upper(), str(page_number), str(position),
                _normalized_content(title), _normalized_content(date),
            )
        )
    digest = sha256(material.encode("utf-8")).hexdigest()[:20].upper()
    return f"GENERATED_{digest}"


def deduplicate_reviews(reviews: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep first occurrence, preferring Amazon IDs then normalized content hash."""

    deduplicated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    reliable_content: set[str] = set()
    generated_content_indexes: dict[str, int] = {}

    for original in reviews:
        review = dict(original)
        raw_id = str(review.get("review_id") or "").strip()
        reliable_id = bool(raw_id and not raw_id.startswith("GENERATED_"))
        normalized_content = _normalized_content(review.get("content"))
        content_key = _content_hash(review.get("content")) if normalized_content else ""
        if reliable_id:
            key = raw_id.casefold()
            if key in seen_ids:
                continue
            seen_ids.add(key)
            # If a fallback-ID copy appeared first, replace it with the record
            # carrying Amazon's stable review ID. Distinct reliable IDs remain
            # distinct even when their review bodies happen to match.
            if content_key and content_key in generated_content_indexes:
                index = generated_content_indexes.pop(content_key)
                deduplicated[index] = review
                reliable_content.add(content_key)
                continue
            if content_key:
                reliable_content.add(content_key)
        else:
            if content_key:
                if (
                    content_key in reliable_content
                    or content_key in generated_content_indexes
                ):
                    continue
                generated_content_indexes[content_key] = len(deduplicated)
            elif raw_id:
                key = raw_id.casefold()
                if key in seen_ids:
                    continue
                seen_ids.add(key)
        deduplicated.append(review)

    return deduplicated


_STRONG_CHALLENGE_MARKERS = (
    "robot check",
    "enter the characters you see below",
    "type the characters you see in this image",
    "validatecaptcha",
    "sorry, we just need to make sure you're not a robot",
    "sorry, we just need to make sure you’re not a robot",
)

_WEAK_CHALLENGE_MARKERS = (
    "sign in required",
    "sign in to continue",
    "access denied",
    "automated access to amazon data",
)

_NORMAL_AMAZON_PAGE_MARKERS = (
    'id="producttitle"',
    "id='producttitle'",
    'id="dp"',
    "id='dp'",
    'data-hook="review"',
    "data-hook='review'",
    "customer_review-",
)

_NO_REVIEW_MARKERS = (
    "no customer reviews",
    "no global reviews",
    "there are no customer reviews yet",
    "has no customer reviews",
)

_TITLE_RATING_PREFIX = re.compile(
    r"^\s*[0-5](?:[.,]\d+)?\s+(?:out\s+of|von|sur|su|de)\s+5"
    r"(?:\s+(?:stars?|sternen|étoiles|stelle|estrellas))?\s*",
    re.IGNORECASE,
)


def detect_amazon_challenge(html: object) -> bool:
    """Return true for known Amazon access-control pages; never solve them."""

    if not isinstance(html, str):
        return False
    lowered = html.casefold()
    if any(marker in lowered for marker in _STRONG_CHALLENGE_MARKERS):
        return True
    # Generic policy/script phrases can also appear inside a valid, fully
    # rendered product or review page.  Treat them as blocking only when the
    # response lacks recognizable normal-page DOM.
    normal_page = any(marker in lowered for marker in _NORMAL_AMAZON_PAGE_MARKERS)
    return not normal_page and any(
        marker in lowered for marker in _WEAK_CHALLENGE_MARKERS
    )


def _parse_document(html: str) -> _Node:
    parser = _DocumentParser()
    parser.feed(html)
    parser.close()
    return parser.root


def _stable_amazon_review_id(container: _Node) -> str | None:
    review_id = container.attrs.get("data-review-id", "").strip()
    if review_id:
        return review_id
    element_id = container.attrs.get("id", "").strip()
    prefix = "customer_review-"
    if element_id.startswith(prefix):
        return element_id[len(prefix) :]
    return element_id or None


def _extract_title(container: _Node) -> str | None:
    title_node = _find_first(container, _hook("review-title"))
    title = _text(title_node, exclude=lambda node: _has_class(node, "a-icon-alt"))
    title = _TITLE_RATING_PREFIX.sub("", title).strip()
    return title or None


def _next_page_href(root: _Node) -> str | None:
    next_link = _find_first(
        root,
        lambda node: node.tag == "a" and _has_class(node.parent, "a-last"),
    )
    if next_link is None:
        next_link = _find_first(
            root,
            lambda node: node.tag == "a"
            and node.attrs.get("data-hook") in {
                "pagination-next-link", "see-all-reviews-link-foot",
            },
        )
    href = next_link.attrs.get("href", "").strip() if next_link else ""
    return href or None


def parse_review_page(
    html: str,
    *,
    asin: str,
    marketplace: str,
    source_url: str,
    page_number: int = 1,
) -> ParseResult:
    """Parse one already-fetched review page without making a network request."""

    if not isinstance(html, str):
        return ParseResult(
            status=ScrapeStatus.PARSE_ERROR,
            reason="response_was_not_text",
        )

    if detect_amazon_challenge(html):
        return ParseResult(
            status=ScrapeStatus.BLOCKED,
            reason="amazon_challenge",
        )

    try:
        root = _parse_document(html)
    except Exception:
        return ParseResult(
            status=ScrapeStatus.PARSE_ERROR,
            reason="html_parse_error",
        )

    review_nodes = _find_all(root, _hook("review"))
    if not review_nodes:
        # Browser-rendered pages occasionally omit the primary data-hook but
        # retain Amazon's stable review identifiers.  Keep a narrow fallback
        # rather than tying the parser to one product or one review text.
        review_nodes = _find_all(
            root,
            lambda node: (
                bool(node.attrs.get("data-review-id", "").strip())
                or node.attrs.get("id", "").startswith("customer_review-")
            ),
        )
    if not review_nodes:
        explicit_no_reviews = _find_first(root, _hook("no-reviews-message")) is not None
        document_text = _text(root).casefold()
        if explicit_no_reviews or any(
            marker in document_text for marker in _NO_REVIEW_MARKERS
        ):
            return ParseResult(
                status=ScrapeStatus.NO_REVIEWS,
                reason="no_public_reviews",
            )
        return ParseResult(
            status=ScrapeStatus.STRUCTURE_NOT_RECOGNIZED,
            reason="review_container_not_found",
        )

    reviews: list[dict[str, Any]] = []
    for position, container in enumerate(review_nodes, start=1):
        rating_node = _find_first(container, _hook("review-star-rating"))
        if rating_node is None:
            rating_node = _find_first(container, _hook("cmps-review-star-rating"))
        content_node = _find_first(container, _hook("review-body"))
        date_node = _find_first(container, _hook("review-date"))
        verified_node = _find_first(container, _hook("avp-badge"))
        reviewer_node = _find_first(
            container,
            lambda node: _has_class(node, "a-profile-name"),
        )
        helpful_node = _find_first(container, _hook("helpful-vote-statement"))
        variant_node = _find_first(container, _hook("format-strip"))

        title = _extract_title(container)
        content = _text(content_node) or None
        date = _text(date_node) or None
        stable_id = _stable_amazon_review_id(container)
        review_id = stable_id or generate_review_id(
            asin,
            content,
            page_number=page_number,
            position=position,
            title=title,
            date=date,
        )
        reviews.append(
            {
                "review_id": review_id,
                "rating": parse_rating(_text(rating_node)),
                "title": title,
                "content": content,
                "date": date,
                "verified_purchase": verified_node is not None,
                "reviewer_name": _text(reviewer_node) or None,
                "helpful_votes": _text(helpful_node) or None,
                "product_variant": _text(variant_node) or None,
                "source_url": source_url,
                "asin": asin,
                "marketplace": marketplace,
            }
        )

    return ParseResult(
        status=ScrapeStatus.SUCCESS,
        reviews=reviews,
        next_page_url=_next_page_href(root),
    )


def _decode_response(payload: bytes, content_type: str | None) -> str:
    charset_match = re.search(r"charset=([^;\s]+)", content_type or "", re.I)
    charset = charset_match.group(1).strip('"\'') if charset_match else "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _exception_root(error: BaseException) -> BaseException | object:
    if isinstance(error, URLError):
        return error.reason
    return error


def _transport_error_details(error: BaseException) -> tuple[str, str, str]:
    """Classify expected network failures without exposing endpoint secrets."""

    root = _exception_root(error)
    exception_type = type(root).__name__
    message = str(root).casefold()
    error_number = getattr(root, "errno", None)

    if isinstance(root, socket.gaierror) or error_number in {
        getattr(socket, "EAI_AGAIN", object()),
        getattr(socket, "EAI_NONAME", object()),
    } or any(
        marker in message
        for marker in (
            "name or service not known",
            "nodename nor servname provided",
            "temporary failure in name resolution",
        )
    ):
        return (
            ScrapeStatus.DNS_ERROR,
            "dns_resolution_failed",
            exception_type,
        )
    if (
        isinstance(root, (TimeoutError, socket.timeout))
        or error_number == errno.ETIMEDOUT
        or "timed out" in message
    ):
        return (
            ScrapeStatus.CONNECTION_TIMEOUT,
            "connection_timed_out",
            exception_type,
        )
    if (
        isinstance(root, ConnectionRefusedError)
        or error_number == errno.ECONNREFUSED
        or "connection refused" in message
    ):
        return (
            ScrapeStatus.CONNECTION_REFUSED,
            "connection_refused",
            exception_type,
        )
    if isinstance(
        root,
        (ssl.SSLCertVerificationError, ssl.CertificateError, ssl.SSLError),
    ) or any(
        marker in message
        for marker in (
            "certificate verify failed",
            "tlsv1 alert",
            "ssl:",
        )
    ):
        return (
            ScrapeStatus.TLS_ERROR,
            "tls_handshake_or_certificate_failed",
            exception_type,
        )
    if "proxy" in message or "tunnel connection failed" in message:
        return (
            ScrapeStatus.PROXY_ERROR,
            "proxy_connection_failed",
            exception_type,
        )
    if "redirect" in message:
        return (
            ScrapeStatus.REDIRECT_ERROR,
            "redirect_failed",
            exception_type,
        )
    return (
        ScrapeStatus.CONNECTION_ERROR,
        "connection_failed",
        exception_type,
    )


_SAFE_ERROR_MESSAGES = {
    "dns_resolution_failed": "Hostname could not be resolved.",
    "connection_timed_out": "Connection timed out.",
    "connection_refused": "Connection was refused.",
    "tls_handshake_or_certificate_failed": (
        "TLS handshake or certificate verification failed."
    ),
    "proxy_connection_failed": "Proxy connection failed.",
    "redirect_failed": "HTTP redirect could not be completed.",
    "connection_failed": "Connection failed.",
}


def _transport_failure(url: str, error: BaseException) -> FetchResult:
    status, reason, exception_type = _transport_error_details(error)
    return FetchResult(
        status=status,
        url=url,
        requested_url=url,
        reason=reason,
        exception_type=exception_type,
        error_message=_SAFE_ERROR_MESSAGES[reason],
    )


def fetch_review_page(
    url: str,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    opener: Callable[..., Any] | None = None,
) -> FetchResult:
    """Fetch one public page with a bounded timeout and no authentication state."""

    request = Request(
        url,
        headers={
            "User-Agent": "amazon-review-analyzer/1.0 (public review importer)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.8",
        },
        method="GET",
    )
    try:
        ssl_context = get_verified_ssl_context()
    except TrustedCABundleError as exc:
        return FetchResult(
            status=ScrapeStatus.TLS_ERROR,
            url=url,
            requested_url=url,
            reason=TRUSTED_CA_BUNDLE_UNAVAILABLE,
            exception_type=type(exc).__name__,
            error_message=str(exc),
        )

    open_url = opener or urlopen
    try:
        with open_url(
            request,
            timeout=timeout,
            context=ssl_context,
        ) as response:
            headers = getattr(response, "headers", None)
            content_type = headers.get("Content-Type") if headers else None
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                return FetchResult(
                    status=ScrapeStatus.RESPONSE_TOO_LARGE,
                    url=getattr(response, "url", url),
                    requested_url=url,
                    reason="response_exceeded_size_limit",
                    http_status=getattr(response, "status", None),
                    content_type=content_type,
                    response_size=len(payload),
                    error_message="Response exceeded the configured size limit.",
                )
            return FetchResult(
                status=ScrapeStatus.SUCCESS,
                url=getattr(response, "url", url),
                requested_url=url,
                html=_decode_response(payload, content_type),
                http_status=getattr(response, "status", 200),
                content_type=content_type,
                response_size=len(payload),
            )
    except HTTPError as exc:
        try:
            payload = exc.read(MAX_RESPONSE_BYTES + 1)
        except Exception:
            payload = b""
        content_type = exc.headers.get("Content-Type") if exc.headers else None
        response_url = exc.geturl() or url
        lowered_reason = str(exc.reason).casefold()
        if exc.code == 407 or "proxy" in lowered_reason:
            return FetchResult(
                status=ScrapeStatus.PROXY_ERROR,
                url=response_url,
                requested_url=url,
                html=_decode_response(payload[:MAX_RESPONSE_BYTES], content_type),
                reason="proxy_connection_failed",
                http_status=exc.code,
                content_type=content_type,
                response_size=min(len(payload), MAX_RESPONSE_BYTES),
                exception_type=type(exc).__name__,
                error_message="Proxy connection failed.",
            )
        if "redirect" in lowered_reason:
            return FetchResult(
                status=ScrapeStatus.REDIRECT_ERROR,
                url=response_url,
                requested_url=url,
                html=_decode_response(payload[:MAX_RESPONSE_BYTES], content_type),
                reason="redirect_failed",
                http_status=exc.code,
                content_type=content_type,
                response_size=min(len(payload), MAX_RESPONSE_BYTES),
                exception_type=type(exc).__name__,
                error_message="HTTP redirect could not be completed.",
            )
        phrase = HTTP_STATUS_MESSAGES.get(exc.code, "HTTP error")
        return FetchResult(
            status=ScrapeStatus.HTTP_ERROR,
            url=response_url,
            requested_url=url,
            html=_decode_response(payload[:MAX_RESPONSE_BYTES], content_type),
            reason=f"http_{exc.code}",
            http_status=exc.code,
            content_type=content_type,
            response_size=min(len(payload), MAX_RESPONSE_BYTES),
            exception_type=type(exc).__name__,
            error_message=f"HTTP {exc.code} {phrase}.",
        )
    except (URLError, OSError) as exc:
        return _transport_failure(url, exc)


def build_review_page_url(marketplace: str, asin: str, page_number: int = 1) -> str:
    """Build a public review-page URL for a supported parsed marketplace."""

    return (
        f"https://www.{marketplace}/product-reviews/{asin.upper()}/"
        f"?reviewerType=all_reviews&pageNumber={page_number}"
    )


def _coerce_product_fields(
    product: object | None,
    asin: str | None,
    marketplace: str | None,
    source_url: str | None,
) -> tuple[str, str, str]:
    if product is not None:
        if isinstance(product, Mapping):
            asin = asin or product.get("asin")
            marketplace = marketplace or product.get("marketplace")
            source_url = source_url or product.get("canonical_product_url")
        else:
            asin = asin or getattr(product, "asin", None)
            marketplace = marketplace or getattr(product, "marketplace", None)
            source_url = source_url or getattr(product, "canonical_product_url", None)
    if not asin or not marketplace:
        raise ValueError("asin and marketplace are required")
    normalized_asin = str(asin).upper()
    if re.fullmatch(r"[A-Z0-9]{10}", normalized_asin) is None:
        raise ValueError("asin must be a 10-character alphanumeric identifier")
    normalized_marketplace = str(marketplace).casefold().rstrip(".")
    if normalized_marketplace not in SUPPORTED_MARKETPLACES:
        raise ValueError("marketplace is not supported")
    canonical_url = source_url or (
        f"https://www.{normalized_marketplace}/dp/{normalized_asin}"
    )
    return normalized_asin, normalized_marketplace, str(canonical_url)


def _safe_pagination_url(
    base_url: str,
    href: str,
    marketplace: str,
    asin: str,
) -> str | None:
    """Keep pagination on HTTPS pages within the parsed Amazon marketplace."""

    candidate = urljoin(base_url, href)
    try:
        parsed = urlsplit(candidate)
        hostname = (parsed.hostname or "").casefold().rstrip(".")
    except (TypeError, ValueError):
        return None
    if parsed.scheme.casefold() != "https":
        return None
    if hostname != marketplace and not hostname.endswith(f".{marketplace}"):
        return None
    expected_path = re.compile(
        rf"(?:^|/)product-reviews/{re.escape(asin)}(?:/|$)",
        re.IGNORECASE,
    )
    if expected_path.search(parsed.path) is None:
        return None
    return candidate


def _emit_progress(
    callback: ScrapeProgressCallback | None,
    page_number: int,
    reviews_collected: int,
    status: str,
) -> None:
    if callback is None:
        return
    try:
        callback(page_number, reviews_collected, status)
    except Exception:
        # A display callback must never turn a successful import into a failure.
        return


def scrape_reviews(
    product: object | None = None,
    *,
    asin: str | None = None,
    marketplace: str | None = None,
    source_url: str | None = None,
    max_reviews: int = 50,
    fetcher: PageFetcher | None = None,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    request_delay_seconds: float = DEFAULT_PAGE_DELAY_SECONDS,
    max_pages: int = DEFAULT_MAX_PAGES,
    sleeper: Callable[[float], None] = time.sleep,
    progress_callback: ScrapeProgressCallback | None = None,
) -> ScrapeResult:
    """Fetch Amazon pages serially until the limit or last page is reached.

    ``product`` may be an ``AmazonProductURL`` returned by ``amazon_url.py`` or
    a mapping with the same three fields. Supplying a custom ``fetcher`` keeps
    tests and offline callers completely separate from the network.
    """

    if not isinstance(max_reviews, int) or isinstance(max_reviews, bool) or max_reviews <= 0:
        raise ValueError("max_reviews must be a positive integer")
    if max_reviews > 100:
        raise ValueError("max_reviews cannot exceed 100")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")

    resolved_asin, resolved_marketplace, resolved_source_url = _coerce_product_fields(
        product, asin, marketplace, source_url,
    )
    current_url = build_review_page_url(resolved_marketplace, resolved_asin, 1)
    page_number = 1
    pages_fetched = 0
    collected: list[dict[str, Any]] = []
    visited_urls: set[str] = set()

    def finish(status: str, reason: str | None = None) -> ScrapeResult:
        return ScrapeResult(
            status=status,
            reason=reason,
            reviews=deduplicate_reviews(collected)[:max_reviews],
            pages_fetched=pages_fetched,
            requested_limit=max_reviews,
            asin=resolved_asin,
            marketplace=resolved_marketplace,
            source_url=resolved_source_url,
        )

    while page_number <= max_pages and len(deduplicate_reviews(collected)) < max_reviews:
        if current_url in visited_urls:
            return finish(ScrapeStatus.PARSE_ERROR, "pagination_loop_detected")
        visited_urls.add(current_url)

        if fetcher is None:
            fetched = fetch_review_page(current_url, timeout=request_timeout)
        else:
            fetched = fetcher(current_url)
            if isinstance(fetched, str):
                fetched = FetchResult(
                    status=ScrapeStatus.SUCCESS,
                    url=current_url,
                    html=fetched,
                    http_status=200,
                )
        if not isinstance(fetched, FetchResult):
            return finish(ScrapeStatus.PARSE_ERROR, "fetcher_returned_invalid_result")
        pages_fetched += 1

        # Error responses sometimes contain Amazon's challenge page. Parse it
        # solely to detect that state; never submit or work around the challenge.
        if fetched.html:
            parsed = parse_review_page(
                fetched.html,
                asin=resolved_asin,
                marketplace=resolved_marketplace,
                source_url=fetched.url,
                page_number=page_number,
            )
            if parsed.status == ScrapeStatus.BLOCKED:
                _emit_progress(progress_callback, page_number, len(collected), parsed.status)
                return finish(parsed.status, parsed.reason)
        else:
            parsed = None

        if fetched.status != ScrapeStatus.SUCCESS:
            _emit_progress(progress_callback, page_number, len(collected), fetched.status)
            return finish(fetched.status, fetched.reason)

        if parsed is None:
            parsed = parse_review_page(
                fetched.html,
                asin=resolved_asin,
                marketplace=resolved_marketplace,
                source_url=fetched.url,
                page_number=page_number,
            )
        if parsed.status == ScrapeStatus.NO_REVIEWS and collected:
            _emit_progress(progress_callback, page_number, len(collected), ScrapeStatus.SUCCESS)
            return finish(ScrapeStatus.SUCCESS)
        if parsed.status != ScrapeStatus.SUCCESS:
            _emit_progress(progress_callback, page_number, len(collected), parsed.status)
            return finish(parsed.status, parsed.reason)

        collected.extend(parsed.reviews)
        collected = deduplicate_reviews(collected)
        _emit_progress(
            progress_callback,
            page_number,
            min(len(collected), max_reviews),
            ScrapeStatus.SUCCESS,
        )
        if len(collected) >= max_reviews or not parsed.next_page_url:
            return finish(ScrapeStatus.SUCCESS)

        next_url = _safe_pagination_url(
            fetched.url or current_url,
            parsed.next_page_url,
            resolved_marketplace,
            resolved_asin,
        )
        if next_url is None:
            return finish(ScrapeStatus.PARSE_ERROR, "unsafe_pagination_url")
        sleeper(request_delay_seconds)
        current_url = next_url
        page_number += 1

    return finish(ScrapeStatus.SUCCESS)


# Explicit alias used by callers that prefer the longer product-specific name.
scrape_amazon_reviews = scrape_reviews
