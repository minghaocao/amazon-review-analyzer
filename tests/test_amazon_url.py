"""Offline tests for Amazon marketplace URL parsing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amazon_url import (  # noqa: E402
    AmazonProductURL,
    AmazonURLValidationError,
    SUPPORTED_MARKETPLACES,
    parse_amazon_product_url,
    parse_amazon_url,
)


class AmazonURLTests(unittest.TestCase):
    def test_desktop_canonical_url_is_supported(self) -> None:
        result = parse_amazon_url("https://www.amazon.com/dp/B0DXBYDK56")
        self.assertEqual("amazon.com", result.marketplace)
        self.assertEqual("B0DXBYDK56", result.asin)
        self.assertEqual(
            "https://www.amazon.com/dp/B0DXBYDK56",
            result.canonical_product_url,
        )

    def test_desktop_tracking_path_is_ignored_after_asin(self) -> None:
        result = parse_amazon_url(
            "https://www.amazon.com/dp/B0DXBYDK56/ref=sr_1_1"
            "?qid=1790091747&sr=8-1&th=1&psc=1"
        )
        self.assertEqual("B0DXBYDK56", result.asin)
        self.assertEqual(
            "https://www.amazon.com/dp/B0DXBYDK56",
            result.canonical_product_url,
        )

    def test_gp_product_path_is_supported(self) -> None:
        result = parse_amazon_url(
            "https://www.amazon.com/gp/product/B0DXBYDK56"
        )
        self.assertEqual("B0DXBYDK56", result.asin)
        self.assertEqual(
            "https://www.amazon.com/dp/B0DXBYDK56",
            result.canonical_product_url,
        )

    def test_mobile_aw_path_is_supported(self) -> None:
        result = parse_amazon_url("https://www.amazon.com/gp/aw/d/B0DXBYDK56")
        self.assertEqual("B0DXBYDK56", result.asin)
        self.assertEqual(
            "https://www.amazon.com/dp/B0DXBYDK56",
            result.canonical_product_url,
        )

    def test_mobile_aw_large_tracking_query_does_not_affect_asin(self) -> None:
        result = parse_amazon_url(
            "https://www.amazon.com/gp/aw/d/B0DXBYDK56"
            "?pd_rd_plhdr=t&hsa_cr_id=1234567890&pd_rd_w=example"
            "&pd_rd_wg=tracking&pf_rd_p=ABCDEFGHIJ&pf_rd_r=0123456789"
            "&ref_=example&qid=1790091747&sr=8-1&th=1&psc=1"
            "&encoding=UTF8"
        )
        self.assertEqual("amazon.com", result.marketplace)
        self.assertEqual("B0DXBYDK56", result.asin)
        self.assertEqual(
            "https://www.amazon.com/dp/B0DXBYDK56",
            result.canonical_product_url,
        )

    def test_product_reviews_real_asin_is_supported(self) -> None:
        result = parse_amazon_url(
            "https://www.amazon.com/product-reviews/B0DXBYDK56"
        )
        self.assertEqual("B0DXBYDK56", result.asin)
        self.assertEqual(
            "https://www.amazon.com/dp/B0DXBYDK56",
            result.canonical_product_url,
        )

    def test_portal_customer_reviews_url_is_supported(self) -> None:
        result = parse_amazon_url(
            "https://www.amazon.com/portal/customer-reviews/B0DXBYDK56/"
            "ref=cm_cr_dp_d_show_all_top?_encoding=UTF8&ie=UTF8"
            "&reviewerType=all_reviews"
        )
        self.assertEqual("amazon.com", result.marketplace)
        self.assertEqual("B0DXBYDK56", result.asin)
        self.assertEqual(
            "https://www.amazon.com/dp/B0DXBYDK56",
            result.canonical_product_url,
        )

    def test_portal_customer_reviews_preserves_marketplace(self) -> None:
        result = parse_amazon_url(
            "https://www.amazon.co.uk/portal/customer-reviews/B0DXBYDK56"
        )
        self.assertEqual("amazon.co.uk", result.marketplace)
        self.assertEqual("B0DXBYDK56", result.asin)

    def test_mobile_aw_preserves_uk_marketplace(self) -> None:
        result = parse_amazon_url(
            "https://www.amazon.co.uk/gp/aw/d/B0DXBYDK56?ref_=mobile"
        )
        self.assertEqual("amazon.co.uk", result.marketplace)
        self.assertEqual("B0DXBYDK56", result.asin)
        self.assertEqual(
            "https://www.amazon.co.uk/dp/B0DXBYDK56",
            result.canonical_product_url,
        )

    def test_mobile_aw_without_asin_is_rejected(self) -> None:
        with self.assertRaises(AmazonURLValidationError) as context:
            parse_amazon_url("https://www.amazon.com/gp/aw/d/")
        self.assertEqual("invalid_asin", context.exception.code)

    def test_query_tracking_id_is_never_used_as_asin(self) -> None:
        with self.assertRaises(AmazonURLValidationError) as context:
            parse_amazon_url(
                "https://www.amazon.com/gp/aw/d/"
                "?pd_rd_w=B0DXBYDK56&pf_rd_p=ABCDEFGHIJ"
            )
        self.assertEqual("invalid_asin", context.exception.code)

    def test_unrecognized_path_is_not_guessed_from_random_segment(self) -> None:
        with self.assertRaises(AmazonURLValidationError) as context:
            parse_amazon_url(
                "https://www.amazon.com/some-tracking-path/B0DXBYDK56"
            )
        self.assertEqual("invalid_asin", context.exception.code)

    def test_mobile_routes_still_require_exact_ten_character_asin(self) -> None:
        for url in (
            "https://www.amazon.com/gp/aw/d/B0DXBYDK5",
            "https://www.amazon.com/gp/aw/d/B0DXBYDK567",
            "https://www.amazon.com/gp/product/B0DXBY-K56",
        ):
            with self.subTest(url=url):
                with self.assertRaises(AmazonURLValidationError) as context:
                    parse_amazon_url(url)
                self.assertEqual("invalid_asin", context.exception.code)

    def test_existing_roku_slug_url_remains_supported(self) -> None:
        result = parse_amazon_url(
            "https://www.amazon.com/Roku-Smart-2026-Television-Streaming/"
            "dp/B0FXZ21F3X/ref=sr_1_1?th=1&psc=1"
        )
        self.assertEqual("amazon.com", result.marketplace)
        self.assertEqual("B0FXZ21F3X", result.asin)
        self.assertEqual(
            "https://www.amazon.com/dp/B0FXZ21F3X",
            result.canonical_product_url,
        )

    def test_all_supported_marketplaces_are_normalized(self) -> None:
        asin = "B0ABC12XYZ"
        for marketplace in SUPPORTED_MARKETPLACES:
            with self.subTest(marketplace=marketplace):
                result = parse_amazon_url(
                    f"https://www.{marketplace}/Example-Product/dp/{asin}"
                    "?ref_=example"
                )
                self.assertEqual(marketplace, result.marketplace)
                self.assertEqual(asin, result.asin)
                self.assertEqual(
                    f"https://www.{marketplace}/dp/{asin}",
                    result.canonical_product_url,
                )

    def test_product_reviews_path_is_supported(self) -> None:
        result = parse_amazon_url(
            "https://www.amazon.co.uk/product-reviews/B012345678/"
            "?reviewerType=all_reviews"
        )
        self.assertEqual("amazon.co.uk", result.marketplace)
        self.assertEqual("B012345678", result.asin)
        self.assertEqual(
            "https://www.amazon.co.uk/dp/B012345678",
            result.canonical_product_url,
        )

    def test_marketplace_root_and_real_subdomains_are_supported(self) -> None:
        for url in (
            "https://amazon.de/dp/B012345678",
            "https://smile.amazon.com/dp/B012345678",
            "https://m.amazon.fr/product-reviews/B012345678",
        ):
            with self.subTest(url=url):
                self.assertEqual("B012345678", parse_amazon_url(url).asin)

    def test_route_and_asin_are_normalized_case_insensitively(self) -> None:
        result = parse_amazon_url("https://WWW.AMAZON.IT/DP/b0abc12xyz")
        self.assertEqual("amazon.it", result.marketplace)
        self.assertEqual("B0ABC12XYZ", result.asin)
        self.assertEqual(
            "https://www.amazon.it/dp/B0ABC12XYZ", result.canonical_product_url
        )

    def test_trailing_dot_hostname_is_handled_safely(self) -> None:
        result = parse_amazon_url("https://www.amazon.es./dp/B012345678")
        self.assertEqual("amazon.es", result.marketplace)

    def test_fragment_and_query_do_not_change_canonical_url(self) -> None:
        result = parse_amazon_url(
            " https://www.amazon.com/Name/dp/B012345678/ref=something"
            "?tag=example#customerReviews "
        )
        self.assertEqual(
            "https://www.amazon.com/dp/B012345678", result.canonical_product_url
        )

    def test_result_has_json_friendly_dictionary(self) -> None:
        result = parse_amazon_product_url(
            "https://www.amazon.fr/dp/B012345678"
        )
        self.assertIsInstance(result, AmazonProductURL)
        self.assertEqual(result.canonical_product_url, result.canonical_url)
        self.assertEqual(
            {
                "marketplace": "amazon.fr",
                "asin": "B012345678",
                "canonical_product_url": "https://www.amazon.fr/dp/B012345678",
            },
            result.to_dict(),
        )

    def test_non_amazon_and_unsupported_marketplaces_are_rejected(self) -> None:
        for url in (
            "https://example.com/dp/B012345678",
            "https://example.com/dp/B0DXBYDK56",
            "https://www.amazon.ca/dp/B012345678",
            "https://notamazon.com/dp/B012345678",
        ):
            with self.subTest(url=url):
                with self.assertRaisesRegex(
                    AmazonURLValidationError,
                    "This does not appear to be a supported Amazon product URL",
                ):
                    parse_amazon_url(url)

    def test_subdomain_and_userinfo_deception_are_rejected(self) -> None:
        for url in (
            "https://amazon.com.evil.example/dp/B012345678",
            "https://www.amazon.co.uk.evil.example/dp/B012345678",
            "https://amazon.com@evil.example/dp/B012345678",
        ):
            with self.subTest(url=url):
                with self.assertRaises(AmazonURLValidationError) as context:
                    parse_amazon_url(url)
                self.assertEqual("unsupported_marketplace", context.exception.code)

    def test_empty_and_non_string_values_have_friendly_error(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    AmazonURLValidationError, "Please enter an Amazon product URL"
                ) as context:
                    parse_amazon_url(value)  # type: ignore[arg-type]
                self.assertEqual("missing_url", context.exception.code)

    def test_missing_or_invalid_scheme_is_rejected(self) -> None:
        for url in (
            "www.amazon.com/dp/B012345678",
            "ftp://www.amazon.com/dp/B012345678",
            "https:///dp/B012345678",
        ):
            with self.subTest(url=url):
                with self.assertRaises(AmazonURLValidationError) as context:
                    parse_amazon_url(url)
                self.assertEqual("invalid_url", context.exception.code)

    def test_malformed_host_is_rejected_without_leaking_parser_error(self) -> None:
        with self.assertRaisesRegex(
            AmazonURLValidationError,
            "This does not appear to be a supported Amazon product URL",
        ) as context:
            parse_amazon_url("https://[invalid-host/dp/B012345678")
        self.assertEqual("invalid_url", context.exception.code)

    def test_invalid_asin_paths_are_rejected(self) -> None:
        urls = (
            "https://www.amazon.com/dp/B01234567",  # Too short.
            "https://www.amazon.com/dp/B0123456789",  # Too long.
            "https://www.amazon.com/dp/B01234-678",  # Invalid character.
            "https://www.amazon.com/product/B012345678",  # Unsupported route.
            "https://www.amazon.com/?asin=B012345678",  # ASIN only in query.
            "https://www.amazon.com/portal/customer-reviews/",  # Missing ASIN.
        )
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaisesRegex(
                    AmazonURLValidationError, "valid 10-character ASIN"
                ) as context:
                    parse_amazon_url(url)
                self.assertEqual("invalid_asin", context.exception.code)


if __name__ == "__main__":
    unittest.main()
