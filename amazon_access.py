"""Read-only, layered network diagnostics for public Amazon access.

This module never persists reviews or analysis outputs. It reuses the scraper's
standard-library HTTP client so diagnostics reflect the same redirect, proxy,
timeout, and TLS behavior used by a real import.
"""

from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass
from typing import Callable, Mapping

from amazon_url import AmazonProductURL, parse_amazon_url
from scraper import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    FetchResult,
    ScrapeStatus,
    build_review_page_url,
    detect_amazon_challenge,
    fetch_review_page,
    parse_review_page,
)


DEFAULT_AMAZON_DIAGNOSTIC_URL = "https://www.amazon.com/dp/B075BCD1LP"
EXTERNAL_SANITY_URL = "https://example.com/"
PROXY_ENVIRONMENT_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
)


class AccessStatus:
    SUCCESS = "success"
    NOT_RUN = "not_run"
    GENERAL_NETWORK_FAILURE = "general_network_failure"
    AMAZON_CHALLENGE = "amazon_challenge"
    DNS_ERROR = ScrapeStatus.DNS_ERROR
    CONNECTION_TIMEOUT = ScrapeStatus.CONNECTION_TIMEOUT
    CONNECTION_REFUSED = ScrapeStatus.CONNECTION_REFUSED
    TLS_ERROR = ScrapeStatus.TLS_ERROR
    PROXY_ERROR = ScrapeStatus.PROXY_ERROR
    HTTP_ERROR = ScrapeStatus.HTTP_ERROR
    REDIRECT_ERROR = ScrapeStatus.REDIRECT_ERROR
    CONNECTION_ERROR = ScrapeStatus.CONNECTION_ERROR
    PAGE_STRUCTURE_NOT_RECOGNIZED = ScrapeStatus.STRUCTURE_NOT_RECOGNIZED
    NO_REVIEWS = ScrapeStatus.NO_REVIEWS


@dataclass(frozen=True)
class AccessStep:
    """One safe diagnostic observation with no response body or credentials."""

    name: str
    url: str
    attempted: bool
    status: str
    http_status: int | None = None
    final_url: str | None = None
    content_type: str | None = None
    response_size: int = 0
    redirected: bool = False
    challenge_detected: bool = False
    review_container_detected: bool = False
    resolved_addresses: int | None = None
    exception_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AmazonAccessReport:
    """Complete read-only report consumed by the CLI and offline tests."""

    product: AmazonProductURL
    proxy_environment: dict[str, bool]
    http_library: str
    timeout_seconds: float
    redirects_enabled: bool
    tls_verification_enabled: bool
    steps: tuple[AccessStep, ...]
    overall_status: str

    def get_step(self, name: str) -> AccessStep:
        return next(step for step in self.steps if step.name == name)

    def to_dict(self) -> dict[str, object]:
        return {
            "product": self.product.to_dict(),
            "proxy_environment": dict(self.proxy_environment),
            "http_library": self.http_library,
            "timeout_seconds": self.timeout_seconds,
            "redirects_enabled": self.redirects_enabled,
            "tls_verification_enabled": self.tls_verification_enabled,
            "steps": [step.to_dict() for step in self.steps],
            "overall_status": self.overall_status,
        }


DiagnosticFetcher = Callable[[str], FetchResult]
DiagnosticResolver = Callable[..., object]


def check_proxy_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, bool]:
    """Report only whether proxy variables exist, never their values."""

    source = os.environ if environ is None else environ
    present = {name: False for name in PROXY_ENVIRONMENT_NAMES}
    for key, value in source.items():
        normalized = str(key).upper()
        if normalized in present and bool(value):
            present[normalized] = True
    return present


def _dns_step(
    hostname: str,
    resolver: DiagnosticResolver,
) -> AccessStep:
    try:
        addresses = resolver(hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return AccessStep(
            name="dns_resolution",
            url=hostname,
            attempted=True,
            status=ScrapeStatus.DNS_ERROR,
            exception_type=type(exc).__name__,
            error_message="Hostname could not be resolved.",
        )
    except OSError as exc:
        return AccessStep(
            name="dns_resolution",
            url=hostname,
            attempted=True,
            status=ScrapeStatus.DNS_ERROR,
            exception_type=type(exc).__name__,
            error_message="DNS resolution failed.",
        )

    try:
        count = len(addresses)  # type: ignore[arg-type]
    except TypeError:
        count = 1
    return AccessStep(
        name="dns_resolution",
        url=hostname,
        attempted=True,
        status=AccessStatus.SUCCESS,
        resolved_addresses=count,
    )


def _fetch_step(name: str, url: str, fetcher: DiagnosticFetcher) -> tuple[AccessStep, FetchResult]:
    try:
        result = fetcher(url)
    except Exception as exc:
        # Injected/custom clients should still produce a safe diagnostic rather
        # than a traceback. The production urllib client already classifies its
        # expected transport exceptions into FetchResult.
        fallback = FetchResult(
            status=ScrapeStatus.CONNECTION_ERROR,
            url=url,
            requested_url=url,
            reason="diagnostic_fetch_failed",
            exception_type=type(exc).__name__,
            error_message="Connection diagnostic failed.",
        )
        return _step_from_fetch(name, url, fallback), fallback
    if not isinstance(result, FetchResult):
        fallback = FetchResult(
            status=ScrapeStatus.CONNECTION_ERROR,
            url=url,
            requested_url=url,
            reason="invalid_fetch_result",
            exception_type=type(result).__name__,
            error_message="Connection diagnostic returned an invalid result.",
        )
        return _step_from_fetch(name, url, fallback), fallback
    return _step_from_fetch(name, url, result), result


def _step_from_fetch(
    name: str,
    requested_url: str,
    result: FetchResult,
) -> AccessStep:
    challenge = name != "external_https" and detect_amazon_challenge(result.html)
    status = AccessStatus.AMAZON_CHALLENGE if challenge else result.status
    return AccessStep(
        name=name,
        url=requested_url,
        attempted=True,
        status=status,
        http_status=result.http_status,
        final_url=result.final_url,
        content_type=result.content_type,
        response_size=result.response_size,
        redirected=result.redirected,
        challenge_detected=challenge,
        exception_type=result.exception_type,
        error_message=(
            "Amazon returned an access challenge."
            if challenge
            else result.error_message
        ),
    )


def _not_run(name: str, url: str) -> AccessStep:
    return AccessStep(
        name=name,
        url=url,
        attempted=False,
        status=AccessStatus.NOT_RUN,
        error_message="Not run because a prerequisite access check failed.",
    )


def _review_step(
    product: AmazonProductURL,
    url: str,
    fetcher: DiagnosticFetcher,
) -> AccessStep:
    step, fetched = _fetch_step("product_reviews_url", url, fetcher)
    if step.status != AccessStatus.SUCCESS:
        return step

    parsed = parse_review_page(
        fetched.html,
        asin=product.asin,
        marketplace=product.marketplace,
        source_url=fetched.final_url,
    )
    if parsed.status == ScrapeStatus.BLOCKED:
        return AccessStep(
            **{
                **step.to_dict(),
                "status": AccessStatus.AMAZON_CHALLENGE,
                "challenge_detected": True,
                "error_message": "Amazon returned an access challenge.",
            }
        )
    if parsed.status == ScrapeStatus.SUCCESS:
        return AccessStep(
            **{
                **step.to_dict(),
                "review_container_detected": True,
            }
        )
    if parsed.status == ScrapeStatus.NO_REVIEWS:
        return AccessStep(
            **{
                **step.to_dict(),
                "status": AccessStatus.NO_REVIEWS,
                "error_message": "The review page reports no public reviews.",
            }
        )
    if parsed.status == ScrapeStatus.STRUCTURE_NOT_RECOGNIZED:
        return AccessStep(
            **{
                **step.to_dict(),
                "status": AccessStatus.PAGE_STRUCTURE_NOT_RECOGNIZED,
                "error_message": "Amazon review containers were not recognized.",
            }
        )
    return AccessStep(
        **{
            **step.to_dict(),
            "status": parsed.status,
            "error_message": "The Amazon review page could not be parsed.",
        }
    )


def run_amazon_access_diagnostic(
    product_url: str = DEFAULT_AMAZON_DIAGNOSTIC_URL,
    *,
    fetcher: DiagnosticFetcher | None = None,
    resolver: DiagnosticResolver | None = None,
    environ: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> AmazonAccessReport:
    """Run staged, read-only checks without writing Amazon or Demo outputs."""

    product = parse_amazon_url(product_url)
    resolve = resolver or socket.getaddrinfo
    if fetcher is None:
        request = lambda url: fetch_review_page(url, timeout=timeout)
    else:
        request = fetcher

    home_url = f"https://www.{product.marketplace}/"
    review_url = build_review_page_url(product.marketplace, product.asin, 1)

    # Prove the Python HTTPS trust configuration against an ordinary public
    # endpoint before making any Amazon request. A failure stops the Amazon
    # stages, preventing misleading parser or marketplace diagnoses.
    external_step, _ = _fetch_step(
        "external_https",
        EXTERNAL_SANITY_URL,
        request,
    )

    hostname = f"www.{product.marketplace}"
    if external_step.status == AccessStatus.SUCCESS:
        dns = _dns_step(hostname, resolve)
        amazon_home, _ = _fetch_step("amazon_home", home_url, request)
    else:
        dns = _not_run("dns_resolution", hostname)
        amazon_home = _not_run("amazon_home", home_url)

    if amazon_home.status == AccessStatus.SUCCESS:
        product_step, _ = _fetch_step(
            "product_url",
            product.canonical_product_url,
            request,
        )
    else:
        product_step = _not_run("product_url", product.canonical_product_url)

    if product_step.status == AccessStatus.SUCCESS:
        review_step = _review_step(product, review_url, request)
    else:
        review_step = _not_run("product_reviews_url", review_url)

    if external_step.status != AccessStatus.SUCCESS:
        overall = AccessStatus.GENERAL_NETWORK_FAILURE
    elif amazon_home.status != AccessStatus.SUCCESS:
        # At this point the external HTTPS prerequisite succeeded, so an
        # Amazon failure is specific to the marketplace connection.
        overall = amazon_home.status
    elif product_step.status != AccessStatus.SUCCESS:
        overall = product_step.status
    else:
        overall = review_step.status

    return AmazonAccessReport(
        product=product,
        proxy_environment=check_proxy_environment(environ),
        http_library="urllib.request",
        timeout_seconds=float(timeout),
        redirects_enabled=True,
        tls_verification_enabled=True,
        steps=(external_step, dns, amazon_home, product_step, review_step),
        overall_status=overall,
    )


__all__ = [
    "AccessStatus",
    "AccessStep",
    "AmazonAccessReport",
    "DEFAULT_AMAZON_DIAGNOSTIC_URL",
    "EXTERNAL_SANITY_URL",
    "PROXY_ENVIRONMENT_NAMES",
    "check_proxy_environment",
    "run_amazon_access_diagnostic",
]
