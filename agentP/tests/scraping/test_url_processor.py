import unittest
from unittest.mock import MagicMock, patch, mock_open

from scraping.url_processor import UrlProcessor


class TestUrlProcessor(unittest.TestCase):

    def setUp(self):
        self.mock_scraper = MagicMock()
        self.processor = UrlProcessor(self.mock_scraper)

    # ------------------------------------------------------------------
    # process_urls_from_file()
    # ------------------------------------------------------------------

    @patch("scraping.url_processor.os.path.exists", return_value=False)
    def test_should_return_empty_list_when_file_does_not_exist(self, _):
        result = self.processor.process_urls_from_file("/nonexistent/file.txt")
        self.assertEqual(result, [])

    @patch(
        "builtins.open",
        mock_open(read_data="http://example.com\nhttp://example.org\n"),
    )
    @patch("scraping.url_processor.os.path.exists", return_value=True)
    def test_should_scrape_each_url_when_processing_file(self, _):
        self.mock_scraper.scrape.return_value = ""
        self.processor.process_urls_from_file("/some/urls.txt")
        self.assertEqual(self.mock_scraper.scrape.call_count, 2)

    @patch(
        "builtins.open",
        mock_open(read_data="http://example.com\n"),
    )
    @patch("scraping.url_processor.os.path.exists", return_value=True)
    def test_should_skip_url_when_scrape_returns_empty_string(self, _):
        self.mock_scraper.scrape.return_value = ""
        result = self.processor.process_urls_from_file("/some/urls.txt")
        self.assertEqual(result, [])

    @patch(
        "builtins.open",
        mock_open(read_data="http://example.com\n"),
    )
    @patch("scraping.url_processor.os.path.exists", return_value=True)
    def test_should_return_listings_when_content_contains_eur_price(self, _):
        self.mock_scraper.scrape.return_value = "Nice apartment\n\n€1200 per month\n"
        result = self.processor.process_urls_from_file("/some/urls.txt")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rent"], 1200)

    # ------------------------------------------------------------------
    # extract_listings()
    # ------------------------------------------------------------------

    def test_should_return_listing_when_block_contains_eur_sign(self):
        content = "Nice flat\n\n€850 monthly\n"
        result = self.processor.extract_listings(content, "http://test.com")
        self.assertEqual(len(result), 1)

    def test_should_skip_block_when_eur_keyword_present_but_no_price_sign(self):
        # "eur" keyword passes the heuristic filter, but _extract_rent requires
        # the € sign — so the block is still dropped at extraction time.
        content = "Studio apartment\n\n800 eur monthly\n"
        result = self.processor.extract_listings(content, "http://test.com")
        self.assertEqual(len(result), 0)

    def test_should_skip_block_when_no_eur_indicator_present(self):
        content = "Nice place, no price listed here.\n"
        result = self.processor.extract_listings(content, "http://test.com")
        self.assertEqual(result, [])

    def test_should_skip_block_when_rent_pattern_not_matched(self):
        # Has EUR sign but no valid €XXX pattern (amount is too short)
        content = "Cheap place\n\n€99 per month"
        result = self.processor.extract_listings(content, "http://test.com")
        self.assertEqual(result, [])

    def test_should_include_source_url_in_listing_when_extracting(self):
        content = "Flat\n\n€1000 monthly"
        result = self.processor.extract_listings(content, "http://source.com")
        self.assertEqual(result[0]["url"], "http://source.com")

    # ------------------------------------------------------------------
    # _extract_rent()
    # ------------------------------------------------------------------

    def test_should_return_rent_as_int_when_eur_sign_and_digits_present(self):
        result = self.processor._extract_rent("Nice flat €1200 per month")
        self.assertEqual(result, 1200)

    def test_should_handle_space_between_eur_sign_and_digits(self):
        result = self.processor._extract_rent("€ 950 per month")
        self.assertEqual(result, 950)

    def test_should_return_none_when_no_eur_sign_present(self):
        result = self.processor._extract_rent("1200 per month")
        self.assertIsNone(result)

    def test_should_return_none_when_digits_too_short(self):
        result = self.processor._extract_rent("€99 monthly")
        self.assertIsNone(result)

    def test_should_return_first_five_digits_when_amount_has_six_digits(self):
        # The regex \d{3,5} is greedy and matches up to 5 digits, so it
        # captures the first 5 digits of a 6-digit number.
        result = self.processor._extract_rent("€123456 monthly")
        self.assertEqual(result, 12345)

    # ------------------------------------------------------------------
    # _extract_property_type()
    # ------------------------------------------------------------------

    def test_should_return_house_when_house_keyword_present(self):
        result = self.processor._extract_property_type("A nice house in Brussels")
        self.assertEqual(result, "House")

    def test_should_return_house_when_maison_keyword_present(self):
        result = self.processor._extract_property_type("Maison à louer")
        self.assertEqual(result, "House")

    def test_should_return_apartment_when_apartment_keyword_present(self):
        result = self.processor._extract_property_type("Apartment available now")
        self.assertEqual(result, "Apartment")

    def test_should_return_apartment_when_flat_keyword_present(self):
        result = self.processor._extract_property_type("Flat to rent in Antwerp")
        self.assertEqual(result, "Apartment")

    def test_should_return_apartment_by_default_when_no_type_keyword(self):
        result = self.processor._extract_property_type("Nice place €1000")
        self.assertEqual(result, "Apartment")

    # ------------------------------------------------------------------
    # _extract_city()
    # ------------------------------------------------------------------

    def test_should_return_capitalised_city_when_known_city_found_in_text(self):
        result = self.processor._extract_city("Great flat in brussels city centre")
        self.assertEqual(result, "Brussels")

    def test_should_return_unknown_when_no_city_found_in_text(self):
        result = self.processor._extract_city("Nice place somewhere")
        self.assertEqual(result, "Unknown")

    def test_should_detect_antwerp_when_present_in_text(self):
        result = self.processor._extract_city("2-bed apartment in Antwerp")
        self.assertEqual(result, "Antwerp")

    def test_should_detect_ghent_when_present_in_text(self):
        result = self.processor._extract_city("Studio in Ghent city")
        self.assertEqual(result, "Ghent")

    # ------------------------------------------------------------------
    # _extract_description()
    # ------------------------------------------------------------------

    def test_should_truncate_description_to_160_chars_when_text_is_longer(self):
        long_text = "A" * 200
        result = self.processor._extract_description(long_text)
        self.assertEqual(len(result), 160)

    def test_should_not_truncate_when_text_is_shorter_than_160(self):
        short_text = "Short description"
        result = self.processor._extract_description(short_text)
        self.assertEqual(result, "Short description")

    def test_should_collapse_whitespace_when_cleaning_description(self):
        text = "Line   with   extra    spaces"
        result = self.processor._extract_description(text)
        self.assertNotIn("   ", result)
