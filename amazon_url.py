"""Validation and normalization for supported Amazon product URLs.

This module deliberately contains no network or scraping code.  It converts a
user-provided URL into the marketplace and ASIN needed by the scraper while
rejecting look-alike hosts and malformed product identifiers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from urllib.parse import unquote, urlsplit


SUPPORTED_MARKETPLACES: tuple[str, ...] = (
    "amazon.com",
    "amazon.co.uk",
    "amazon.de",
    "amazon.fr",
    "amazon.it",
    "amazon.es",
)

_ASIN_TOKEN_PATTERN = r"(?P<asin>[A-Z0-9]{10})"

# Keep recognized Amazon routes explicit and ordered.  The parser searches the
# decoded path only, so query-string tracking identifiers can never become an
# ASIN candidate.
_ASIN_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"(?:^|/)dp/{_ASIN_TOKEN_PATTERN}(?=$|/)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?:^|/)gp/product/{_ASIN_TOKEN_PATTERN}(?=$|/)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?:^|/)gp/aw/d/{_ASIN_TOKEN_PATTERN}(?=$|/)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?:^|/)product-reviews/{_ASIN_TOKEN_PATTERN}(?=$|/)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?:^|/)portal/customer-reviews/{_ASIN_TOKEN_PATTERN}(?=$|/)",
        flags=re.IGNORECASE,
    ),
)

_UNSUPPORTED_URL_MESSAGE = (
    "This does not appear to be a supported Amazon product URL."
)
_INVALID_ASIN_MESSAGE = (
    "This Amazon URL does not contain a valid 10-character ASIN in a "
    "supported product path."
)


class AmazonURLValidationError(ValueError):
    """Raised when a user-provided URL cannot be used for Amazon import."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class AmazonProductURL:
    """Normalized identity extracted from one supported Amazon URL."""

    marketplace: str
    asin: str
    canonical_product_url: str

    @property
    def canonical_url(self) -> str:
        """Return a concise alias useful to callers and templates."""

        return self.canonical_product_url

    def to_dict(self) -> dict[str, str]:
        """Return JSON-friendly URL metadata."""

        return asdict(self)


def _marketplace_for_hostname(hostname: str) -> str | None:
    """Return the marketplace for an exact domain or its genuine subdomain."""

    normalized_hostname = hostname.lower().rstrip(".")
    for marketplace in SUPPORTED_MARKETPLACES:
        if normalized_hostname == marketplace or normalized_hostname.endswith(
            f".{marketplace}"
        ):
            return marketplace
    return None


def _asin_from_path(path: str) -> str | None:
    """Return an ASIN from one recognized Amazon path, never from its query."""

    for pattern in _ASIN_PATH_PATTERNS:
        match = pattern.search(path)
        if match is not None:
            return match.group("asin").upper()
    return None


def parse_amazon_url(url: str) -> AmazonProductURL:
    """Validate and normalize an Amazon product or product-review URL.

    Supported paths contain ``/dp/{ASIN}``, ``/gp/product/{ASIN}``,
    ``/gp/aw/d/{ASIN}``, ``/product-reviews/{ASIN}``, or
    ``/portal/customer-reviews/{ASIN}``. Query parameters and fragments are
    ignored; an ASIN supplied only in those locations is not accepted.

    Raises:
        AmazonURLValidationError: If the URL is empty, malformed, hosted on an
            unsupported domain, or lacks a valid ASIN in a supported path.
    """

    if not isinstance(url, str) or not url.strip():
        raise AmazonURLValidationError(
            "Please enter an Amazon product URL.", code="missing_url"
        )

    candidate = url.strip()
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
    except (TypeError, ValueError) as exc:
        raise AmazonURLValidationError(
            _UNSUPPORTED_URL_MESSAGE, code="invalid_url"
        ) from exc

    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise AmazonURLValidationError(
            _UNSUPPORTED_URL_MESSAGE, code="invalid_url"
        )

    marketplace = _marketplace_for_hostname(hostname)
    if marketplace is None:
        raise AmazonURLValidationError(
            _UNSUPPORTED_URL_MESSAGE, code="unsupported_marketplace"
        )

    decoded_path = unquote(parsed.path)
    asin = _asin_from_path(decoded_path)
    if asin is None:
        raise AmazonURLValidationError(_INVALID_ASIN_MESSAGE, code="invalid_asin")

    canonical_product_url = f"https://www.{marketplace}/dp/{asin}"
    return AmazonProductURL(
        marketplace=marketplace,
        asin=asin,
        canonical_product_url=canonical_product_url,
    )


# The longer name reads naturally in scraper code while preserving one parser.
parse_amazon_product_url = parse_amazon_url


__all__ = [
    "AmazonProductURL",
    "AmazonURLValidationError",
    "SUPPORTED_MARKETPLACES",
    "parse_amazon_product_url",
    "parse_amazon_url",
]
