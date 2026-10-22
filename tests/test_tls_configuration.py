"""Offline tests for the project's verified HTTPS configuration."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import ssl
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import amazon_access  # noqa: E402
import scraper  # noqa: E402


ASIN = "B012345678"
MARKETPLACE = "amazon.com"
PRODUCT_URL = f"https://www.{MARKETPLACE}/dp/{ASIN}"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "amazon"
NORMAL_REVIEW_HTML = (
    FIXTURES / "normal_reviews.html"
).read_text(encoding="utf-8").replace("B0TEST1234", ASIN)


def _secure_context() -> Mock:
    context = Mock(spec=ssl.SSLContext)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


class _Response:
    status = 200
    headers = {"Content-Type": "text/html; charset=utf-8"}

    def __init__(self, body: str, url: str) -> None:
        self._body = body.encode("utf-8")
        self.url = url

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self._body[:size]


@dataclass(frozen=True)
class _Product:
    marketplace: str = MARKETPLACE
    asin: str = ASIN
    canonical_product_url: str = PRODUCT_URL


class VerifiedTlsConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        scraper.get_verified_ssl_context.cache_clear()

    def tearDown(self) -> None:
        scraper.get_verified_ssl_context.cache_clear()

    def test_certifi_bundle_path_builds_one_shared_secure_context(self) -> None:
        bundle_path = "/virtual/certifi/cacert.pem"
        context = _secure_context()

        with (
            patch.object(scraper.certifi, "where", return_value=bundle_path) as where,
            patch.object(scraper.Path, "is_file", return_value=True),
            patch.object(scraper.os, "access", return_value=True),
            patch.object(
                scraper.ssl,
                "create_default_context",
                return_value=context,
            ) as create_default_context,
        ):
            first = scraper.get_verified_ssl_context()
            second = scraper.get_verified_ssl_context()

        self.assertIs(context, first)
        self.assertIs(first, second)
        where.assert_called_once_with()
        create_default_context.assert_called_once_with(cafile=bundle_path)
        self.assertIs(first.check_hostname, True)
        self.assertEqual(ssl.CERT_REQUIRED, first.verify_mode)

    def test_insecure_context_settings_are_rejected(self) -> None:
        cases = (
            (False, ssl.CERT_REQUIRED),
            (True, ssl.CERT_NONE),
        )
        for check_hostname, verify_mode in cases:
            with self.subTest(
                check_hostname=check_hostname,
                verify_mode=verify_mode,
            ):
                scraper.get_verified_ssl_context.cache_clear()
                context = Mock(spec=ssl.SSLContext)
                context.check_hostname = check_hostname
                context.verify_mode = verify_mode
                with (
                    patch.object(
                        scraper.certifi,
                        "where",
                        return_value="/virtual/certifi/cacert.pem",
                    ),
                    patch.object(scraper.Path, "is_file", return_value=True),
                    patch.object(scraper.os, "access", return_value=True),
                    patch.object(
                        scraper.ssl,
                        "create_default_context",
                        return_value=context,
                    ),
                ):
                    with self.assertRaisesRegex(
                        scraper.TrustedCABundleError,
                        "Secure TLS certificate verification could not be enabled",
                    ):
                        scraper.get_verified_ssl_context()

    def test_missing_ca_bundle_fails_closed_before_context_creation(self) -> None:
        with (
            patch.object(
                scraper.certifi,
                "where",
                return_value="/missing/certifi/cacert.pem",
            ),
            patch.object(scraper.Path, "is_file", return_value=False),
            patch.object(scraper.os, "access", return_value=False),
            patch.object(scraper.ssl, "create_default_context") as create_context,
        ):
            with self.assertRaisesRegex(
                scraper.TrustedCABundleError,
                "^Trusted CA certificate bundle is unavailable\\.$",
            ):
                scraper.get_verified_ssl_context()

        create_context.assert_not_called()

    def test_fetch_returns_friendly_tls_failure_when_ca_bundle_is_unavailable(
        self,
    ) -> None:
        opener = Mock(side_effect=AssertionError("network opener must not run"))
        error = scraper.TrustedCABundleError(
            scraper.TRUSTED_CA_BUNDLE_UNAVAILABLE_MESSAGE
        )

        with patch.object(
            scraper,
            "get_verified_ssl_context",
            side_effect=error,
        ):
            result = scraper.fetch_review_page(PRODUCT_URL, opener=opener)

        opener.assert_not_called()
        self.assertEqual(scraper.ScrapeStatus.TLS_ERROR, result.status)
        self.assertEqual(
            scraper.TRUSTED_CA_BUNDLE_UNAVAILABLE,
            result.reason,
        )
        self.assertEqual(
            "Trusted CA certificate bundle is unavailable.",
            result.error_message,
        )
        self.assertEqual("TrustedCABundleError", result.exception_type)

    def test_unverified_ssl_context_is_never_used(self) -> None:
        context = _secure_context()
        with (
            patch.object(
                scraper.certifi,
                "where",
                return_value="/virtual/certifi/cacert.pem",
            ),
            patch.object(scraper.Path, "is_file", return_value=True),
            patch.object(scraper.os, "access", return_value=True),
            patch.object(
                scraper.ssl,
                "create_default_context",
                return_value=context,
            ),
            patch.object(
                scraper.ssl,
                "_create_unverified_context",
                side_effect=AssertionError("unverified TLS context is forbidden"),
            ) as unverified_context,
        ):
            returned = scraper.get_verified_ssl_context()

        self.assertIs(context, returned)
        unverified_context.assert_not_called()

    def test_scraper_and_diagnostic_use_the_same_tls_context(self) -> None:
        context = _secure_context()
        observed_contexts: list[object] = []

        def open_url(request: object, **kwargs: object) -> _Response:
            observed_contexts.append(kwargs["context"])
            url = request.full_url  # type: ignore[attr-defined]
            if "/product-reviews/" in url:
                body = NORMAL_REVIEW_HTML
            else:
                body = "<html><body>ok</body></html>"
            return _Response(body, url)

        def resolver(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
            del args, kwargs
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("192.0.2.10", 443),
                )
            ]

        with (
            patch.object(
                scraper,
                "get_verified_ssl_context",
                return_value=context,
            ),
            patch.object(scraper, "urlopen", side_effect=open_url),
        ):
            imported = scraper.scrape_reviews(
                _Product(),
                max_reviews=1,
                sleeper=Mock(),
            )
            diagnostic = amazon_access.run_amazon_access_diagnostic(
                PRODUCT_URL,
                resolver=resolver,
                environ={},
            )

        self.assertIs(amazon_access.fetch_review_page, scraper.fetch_review_page)
        self.assertEqual(scraper.ScrapeStatus.SUCCESS, imported.status)
        self.assertEqual(amazon_access.AccessStatus.SUCCESS, diagnostic.overall_status)
        self.assertGreaterEqual(len(observed_contexts), 5)
        self.assertTrue(all(item is context for item in observed_contexts))


if __name__ == "__main__":
    unittest.main()
