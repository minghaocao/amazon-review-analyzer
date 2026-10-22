"""Offline tests for Amazon access diagnostics and transport classification.

Every request path in this module receives an injected callable.  The tests
must never depend on DNS, an outbound socket, or a live Amazon response.
"""

from __future__ import annotations

import io
import socket
import ssl
import sys
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scraper  # noqa: E402
import amazon_access  # noqa: E402


ASIN = "B012345678"
MARKETPLACE = "amazon.com"
HOME_URL = f"https://www.{MARKETPLACE}/"
PRODUCT_URL = f"https://www.{MARKETPLACE}/dp/{ASIN}"
REVIEW_URL = (
    f"https://www.{MARKETPLACE}/product-reviews/{ASIN}/"
    "?reviewerType=all_reviews&pageNumber=1"
)
EXTERNAL_URL = "https://example.com/"
HTML_HEADERS = {"Content-Type": "text/html; charset=utf-8"}
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "amazon"
NORMAL_REVIEW_HTML = (FIXTURES / "normal_reviews.html").read_text(encoding="utf-8")
CHALLENGE_HTML = (FIXTURES / "challenge.html").read_text(encoding="utf-8")


class _Response:
    def __init__(
        self,
        body: bytes | str,
        *,
        url: str,
        status: int = 200,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self.url = url
        self.status = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self._body[:size]


def _http_error(
    url: str,
    code: int,
    *,
    reason: str,
    body: bytes | str = b"",
    headers: dict[str, str] | None = None,
) -> HTTPError:
    payload = body.encode("utf-8") if isinstance(body, str) else body
    return HTTPError(
        url,
        code,
        reason,
        headers or HTML_HEADERS,
        io.BytesIO(payload),
    )


def _fetch_result(
    url: str,
    *,
    status: str = scraper.ScrapeStatus.SUCCESS,
    html: str = "<html><body>ok</body></html>",
    http_status: int | None = 200,
    final_url: str | None = None,
    content_type: str = "text/html; charset=utf-8",
    exception_type: str | None = None,
    error_message: str | None = None,
) -> scraper.FetchResult:
    return scraper.FetchResult(
        status=status,
        url=final_url or url,
        requested_url=url,
        html=html,
        http_status=http_status,
        content_type=content_type,
        response_size=len(html.encode("utf-8")),
        exception_type=exception_type,
        error_message=error_message,
    )


class _ScriptedFetcher:
    def __init__(self, results: dict[str, scraper.FetchResult]) -> None:
        self.results = results
        self.calls: list[str] = []

    def __call__(self, url: str) -> scraper.FetchResult:
        self.calls.append(url)
        if url not in self.results:
            raise AssertionError(f"unexpected diagnostic request: {url}")
        return self.results[url]


def _successful_resolver(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
    del args, kwargs
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.10", 443))]


def _run_offline(
    fetcher: _ScriptedFetcher,
    *,
    resolver: object = _successful_resolver,
    environ: dict[str, str] | None = None,
) -> amazon_access.AmazonAccessReport:
    """Run the public diagnostic API while making network fallback fatal."""

    with (
        patch.object(
            scraper,
            "urlopen",
            side_effect=AssertionError("live urlopen must not be called"),
        ),
        patch.object(
            amazon_access.socket,
            "getaddrinfo",
            side_effect=AssertionError("live DNS must not be called"),
        ),
    ):
        return amazon_access.run_amazon_access_diagnostic(
            PRODUCT_URL,
            fetcher=fetcher,
            resolver=resolver,  # type: ignore[arg-type]
            environ={} if environ is None else environ,
        )


class HttpTransportClassificationTests(unittest.TestCase):
    """Exercise the real urllib adapter without ever opening a real socket."""

    def _fetch_error(self, error: BaseException) -> scraper.FetchResult:
        def opener(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise error

        with patch.object(
            scraper,
            "urlopen",
            side_effect=AssertionError("live urlopen must not be called"),
        ):
            return scraper.fetch_review_page(PRODUCT_URL, opener=opener)

    def test_dns_error_is_classified_and_keeps_safe_exception_metadata(self) -> None:
        result = self._fetch_error(
            URLError(socket.gaierror(socket.EAI_NONAME, "host not found"))
        )

        self.assertEqual(scraper.ScrapeStatus.DNS_ERROR, result.status)
        self.assertEqual("gaierror", result.exception_type)
        self.assertTrue(result.error_message)
        self.assertNotIn("host not found", result.error_message or "")

    def test_timeout_is_distinct_from_other_connection_errors(self) -> None:
        result = self._fetch_error(URLError(socket.timeout("timed out")))

        self.assertEqual(scraper.ScrapeStatus.CONNECTION_TIMEOUT, result.status)
        self.assertTrue(result.exception_type)
        self.assertIn("timed out", (result.error_message or "").casefold())

    def test_connection_refused_is_distinct(self) -> None:
        result = self._fetch_error(URLError(ConnectionRefusedError("refused")))

        self.assertEqual(scraper.ScrapeStatus.CONNECTION_REFUSED, result.status)
        self.assertEqual("ConnectionRefusedError", result.exception_type)

    def test_tls_certificate_error_is_classified_without_disabling_verification(self) -> None:
        result = self._fetch_error(
            URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))
        )

        self.assertEqual(scraper.ScrapeStatus.TLS_ERROR, result.status)
        self.assertEqual("SSLCertVerificationError", result.exception_type)
        self.assertIn("certificate", (result.error_message or "").casefold())

    def test_proxy_authentication_error_is_classified(self) -> None:
        result = self._fetch_error(
            _http_error(PRODUCT_URL, 407, reason="Proxy Authentication Required")
        )

        self.assertEqual(scraper.ScrapeStatus.PROXY_ERROR, result.status)
        self.assertEqual(407, result.http_status)
        self.assertEqual("HTTPError", result.exception_type)

    def test_http_403_is_an_http_error_with_response_metadata(self) -> None:
        body = "<html><title>Forbidden</title></html>"
        result = self._fetch_error(
            _http_error(PRODUCT_URL, 403, reason="Forbidden", body=body)
        )

        self.assertEqual(scraper.ScrapeStatus.HTTP_ERROR, result.status)
        self.assertEqual(403, result.http_status)
        self.assertEqual(PRODUCT_URL, result.final_url)
        self.assertEqual(HTML_HEADERS["Content-Type"], result.content_type)
        self.assertEqual(len(body.encode("utf-8")), result.response_size)

    def test_http_503_is_an_http_error_with_response_metadata(self) -> None:
        result = self._fetch_error(
            _http_error(
                PRODUCT_URL,
                503,
                reason="Service Unavailable",
                body="temporarily unavailable",
                headers={"Content-Type": "text/plain"},
            )
        )

        self.assertEqual(scraper.ScrapeStatus.HTTP_ERROR, result.status)
        self.assertEqual(503, result.http_status)
        self.assertEqual("text/plain", result.content_type)

    def test_redirect_failure_is_not_collapsed_into_http_error(self) -> None:
        result = self._fetch_error(
            _http_error(
                PRODUCT_URL,
                302,
                reason="redirect loop detected",
                headers={"Location": PRODUCT_URL},
            )
        )

        self.assertEqual(scraper.ScrapeStatus.REDIRECT_ERROR, result.status)
        self.assertEqual(302, result.http_status)

    def test_success_records_final_url_content_type_size_and_timeout(self) -> None:
        body = "<html><body>ok</body></html>"
        final_url = f"{PRODUCT_URL}/ref=canonical"
        opener = Mock(return_value=_Response(body, url=final_url))

        with patch.object(
            scraper,
            "urlopen",
            side_effect=AssertionError("live urlopen must not be called"),
        ):
            result = scraper.fetch_review_page(
                PRODUCT_URL,
                timeout=4.5,
                opener=opener,
            )

        self.assertEqual(scraper.ScrapeStatus.SUCCESS, result.status)
        self.assertEqual(PRODUCT_URL, result.requested_url)
        self.assertEqual(final_url, result.final_url)
        self.assertTrue(result.redirected)
        self.assertEqual(200, result.http_status)
        self.assertEqual("text/html; charset=utf-8", result.content_type)
        self.assertEqual(len(body.encode("utf-8")), result.response_size)
        self.assertEqual(4.5, opener.call_args.kwargs["timeout"])

        request = opener.call_args.args[0]
        self.assertIn("amazon-review-analyzer", request.get_header("User-agent"))
        self.assertEqual({"timeout", "context"}, set(opener.call_args.kwargs))
        ssl_context = opener.call_args.kwargs["context"]
        self.assertIs(ssl_context.check_hostname, True)
        self.assertEqual(ssl.CERT_REQUIRED, ssl_context.verify_mode)


class AmazonAccessRunnerTests(unittest.TestCase):
    def _happy_results(self) -> dict[str, scraper.FetchResult]:
        return {
            HOME_URL: _fetch_result(HOME_URL),
            PRODUCT_URL: _fetch_result(PRODUCT_URL),
            REVIEW_URL: _fetch_result(REVIEW_URL, html=NORMAL_REVIEW_HTML),
            EXTERNAL_URL: _fetch_result(EXTERNAL_URL),
        }

    def test_access_status_exposes_every_required_diagnostic_category(self) -> None:
        expected = {
            "DNS_ERROR": "dns_error",
            "CONNECTION_TIMEOUT": "connection_timeout",
            "CONNECTION_REFUSED": "connection_refused",
            "TLS_ERROR": "tls_error",
            "PROXY_ERROR": "proxy_error",
            "HTTP_ERROR": "http_error",
            "REDIRECT_ERROR": "redirect_error",
            "AMAZON_CHALLENGE": "amazon_challenge",
            "PAGE_STRUCTURE_NOT_RECOGNIZED": "page_structure_not_recognized",
            "SUCCESS": "success",
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(value, getattr(amazon_access.AccessStatus, name))

    def test_proxy_environment_reports_presence_only(self) -> None:
        secret = "do-not-leak-this-password"
        result = amazon_access.check_proxy_environment(
            {
                "http_proxy": f"http://user:{secret}@proxy.invalid:8080",
                "ALL_PROXY": "socks5://proxy.invalid:1080",
                "UNRELATED": secret,
            }
        )

        self.assertEqual(
            {
                "HTTP_PROXY": True,
                "HTTPS_PROXY": False,
                "ALL_PROXY": True,
                "NO_PROXY": False,
            },
            result,
        )
        self.assertNotIn(secret, repr(result))

    def test_happy_path_runs_all_stages_and_recognizes_review_container(self) -> None:
        fetcher = _ScriptedFetcher(self._happy_results())
        report = _run_offline(fetcher)

        self.assertEqual(
            [EXTERNAL_URL, HOME_URL, PRODUCT_URL, REVIEW_URL],
            fetcher.calls,
        )
        self.assertEqual(amazon_access.AccessStatus.SUCCESS, report.overall_status)
        self.assertEqual("urllib.request", report.http_library)
        self.assertGreater(report.timeout_seconds, 0)
        self.assertTrue(report.redirects_enabled)
        self.assertTrue(report.tls_verification_enabled)
        self.assertEqual(
            amazon_access.AccessStatus.SUCCESS,
            report.get_step("dns_resolution").status,
        )
        self.assertEqual(1, report.get_step("dns_resolution").resolved_addresses)
        reviews = report.get_step("product_reviews_url")
        self.assertTrue(reviews.attempted)
        self.assertTrue(reviews.review_container_detected)
        self.assertFalse(reviews.challenge_detected)

    def test_serialized_report_omits_response_bodies_and_proxy_credentials(self) -> None:
        secret = "proxy-password-must-not-appear"
        report = _run_offline(
            _ScriptedFetcher(self._happy_results()),
            environ={"HTTPS_PROXY": f"http://user:{secret}@proxy.invalid"},
        )

        rendered = repr(report.to_dict())
        self.assertNotIn(secret, rendered)
        self.assertNotIn("RNORMAL001", rendered)
        self.assertNotIn("Excellent picture quality", rendered)

    def test_dns_and_home_dns_failure_identify_dns_layer_but_external_still_runs(self) -> None:
        def dns_failure(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise socket.gaierror(socket.EAI_NONAME, "unresolvable")

        fetcher = _ScriptedFetcher(
            {
                HOME_URL: _fetch_result(
                    HOME_URL,
                    status=scraper.ScrapeStatus.DNS_ERROR,
                    html="",
                    http_status=None,
                    exception_type="gaierror",
                    error_message="Hostname could not be resolved.",
                ),
                EXTERNAL_URL: _fetch_result(EXTERNAL_URL),
            }
        )
        report = _run_offline(fetcher, resolver=dns_failure)

        self.assertEqual([EXTERNAL_URL, HOME_URL], fetcher.calls)
        self.assertEqual("dns_error", report.get_step("dns_resolution").status)
        self.assertEqual("dns_error", report.get_step("amazon_home").status)
        self.assertEqual("dns_error", report.overall_status)
        self.assertFalse(report.get_step("product_url").attempted)
        self.assertFalse(report.get_step("product_reviews_url").attempted)
        self.assertEqual("success", report.get_step("external_https").status)

    def test_home_failure_stops_amazon_stages_but_runs_external_sanity_check(self) -> None:
        fetcher = _ScriptedFetcher(
            {
                HOME_URL: _fetch_result(
                    HOME_URL,
                    status=scraper.ScrapeStatus.CONNECTION_TIMEOUT,
                    html="",
                    http_status=None,
                    exception_type="TimeoutError",
                    error_message="Connection timed out.",
                ),
                EXTERNAL_URL: _fetch_result(EXTERNAL_URL),
            }
        )
        report = _run_offline(fetcher)

        self.assertEqual([EXTERNAL_URL, HOME_URL], fetcher.calls)
        self.assertEqual("connection_timeout", report.overall_status)
        self.assertFalse(report.get_step("product_url").attempted)
        self.assertFalse(report.get_step("product_reviews_url").attempted)

    def test_product_http_error_stops_before_reviews(self) -> None:
        results = self._happy_results()
        results[PRODUCT_URL] = _fetch_result(
            PRODUCT_URL,
            status=scraper.ScrapeStatus.HTTP_ERROR,
            html="<html>Forbidden</html>",
            http_status=403,
            exception_type="HTTPError",
            error_message="HTTP 403 Forbidden.",
        )
        del results[REVIEW_URL]
        fetcher = _ScriptedFetcher(results)
        report = _run_offline(fetcher)

        self.assertEqual([EXTERNAL_URL, HOME_URL, PRODUCT_URL], fetcher.calls)
        self.assertEqual("http_error", report.overall_status)
        self.assertEqual(403, report.get_step("product_url").http_status)
        self.assertFalse(report.get_step("product_reviews_url").attempted)

    def test_review_challenge_is_detected_even_on_http_503(self) -> None:
        results = self._happy_results()
        results[REVIEW_URL] = _fetch_result(
            REVIEW_URL,
            status=scraper.ScrapeStatus.HTTP_ERROR,
            html=CHALLENGE_HTML,
            http_status=503,
            exception_type="HTTPError",
            error_message="HTTP 503 Service Unavailable.",
        )
        fetcher = _ScriptedFetcher(results)
        report = _run_offline(fetcher)

        reviews = report.get_step("product_reviews_url")
        self.assertEqual("amazon_challenge", reviews.status)
        self.assertEqual("amazon_challenge", report.overall_status)
        self.assertEqual(503, reviews.http_status)
        self.assertTrue(reviews.challenge_detected)
        self.assertFalse(reviews.review_container_detected)

    def test_unrecognized_review_html_has_its_own_status(self) -> None:
        results = self._happy_results()
        results[REVIEW_URL] = _fetch_result(
            REVIEW_URL,
            html="<html><body>ordinary product content</body></html>",
        )
        report = _run_offline(_ScriptedFetcher(results))

        reviews = report.get_step("product_reviews_url")
        self.assertEqual("page_structure_not_recognized", reviews.status)
        self.assertFalse(reviews.review_container_detected)
        self.assertFalse(reviews.challenge_detected)

    def test_successful_redirect_records_final_url_without_becoming_an_error(self) -> None:
        results = self._happy_results()
        final_url = f"{PRODUCT_URL}/ref=redirected"
        results[PRODUCT_URL] = _fetch_result(
            PRODUCT_URL,
            final_url=final_url,
        )
        report = _run_offline(_ScriptedFetcher(results))

        product = report.get_step("product_url")
        self.assertEqual("success", product.status)
        self.assertTrue(product.redirected)
        self.assertEqual(final_url, product.final_url)

    def test_external_https_failure_stops_before_dns_and_all_amazon_requests(
        self,
    ) -> None:
        resolver = Mock(side_effect=AssertionError("DNS must not run"))
        fetcher = _ScriptedFetcher(
            {
                EXTERNAL_URL: _fetch_result(
                    EXTERNAL_URL,
                    status=scraper.ScrapeStatus.TLS_ERROR,
                    html="",
                    http_status=None,
                    error_message="TLS connection failed.",
                ),
            }
        )
        report = _run_offline(fetcher, resolver=resolver)

        self.assertEqual([EXTERNAL_URL], fetcher.calls)
        resolver.assert_not_called()
        self.assertEqual("general_network_failure", report.overall_status)
        self.assertEqual("tls_error", report.get_step("external_https").status)
        for name in (
            "dns_resolution",
            "amazon_home",
            "product_url",
            "product_reviews_url",
        ):
            with self.subTest(step=name):
                step = report.get_step(name)
                self.assertFalse(step.attempted)
                self.assertEqual(amazon_access.AccessStatus.NOT_RUN, step.status)


if __name__ == "__main__":
    unittest.main()
