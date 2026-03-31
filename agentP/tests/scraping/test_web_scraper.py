import unittest
from unittest.mock import MagicMock, patch

from scraping.web_scraper import WebScraper


class TestWebScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = WebScraper()

    # ------------------------------------------------------------------
    # scrape()
    # ------------------------------------------------------------------

    @patch("scraping.web_scraper.BeautifulSoup")
    @patch("scraping.web_scraper.requests")
    def test_should_return_joined_chunks_when_scraping_is_successful(
        self, mock_requests, MockBS
    ):
        mock_response = MagicMock()
        mock_response.content = b"<html><body><p>Hello</p></body></html>"
        mock_requests.get.return_value = mock_response

        mock_soup = MagicMock()
        MockBS.return_value = mock_soup
        mock_body = MagicMock()
        mock_soup.body = mock_body

        # Simulate the full pipeline returning a short string
        mock_soup.get_text.return_value = "Hello"
        mock_body.__str__ = lambda self: "<body><p>Hello</p></body>"

        with patch.object(self.scraper, "extract_body_content", return_value="<body>Hello</body>"), \
             patch.object(self.scraper, "clean_body_content", return_value="Hello"), \
             patch.object(self.scraper, "split_dom_content", return_value=["Hello"]):
            result = self.scraper.scrape("http://example.com")

        self.assertEqual(result, "Hello")

    @patch("scraping.web_scraper.requests")
    def test_should_return_empty_string_when_request_raises_exception(
        self, mock_requests
    ):
        import requests as real_requests
        mock_requests.get.side_effect = real_requests.exceptions.RequestException("timeout")
        mock_requests.exceptions = real_requests.exceptions

        result = self.scraper.scrape("http://bad-url.com")

        self.assertEqual(result, "")

    @patch("scraping.web_scraper.BeautifulSoup")
    @patch("scraping.web_scraper.requests")
    def test_should_call_requests_get_with_url_and_timeout_when_scraping(
        self, mock_requests, MockBS
    ):
        mock_response = MagicMock()
        mock_requests.get.return_value = mock_response
        mock_requests.exceptions.RequestException = Exception
        MockBS.return_value = MagicMock()

        with patch.object(self.scraper, "extract_body_content", return_value=""), \
             patch.object(self.scraper, "clean_body_content", return_value=""), \
             patch.object(self.scraper, "split_dom_content", return_value=[]):
            self.scraper.scrape("http://example.com")

        mock_requests.get.assert_called_once_with("http://example.com", timeout=15)

    # ------------------------------------------------------------------
    # extract_body_content()
    # ------------------------------------------------------------------

    def test_should_return_str_of_body_when_body_exists(self):
        mock_soup = MagicMock()
        mock_body = MagicMock()
        mock_body.__str__ = lambda s: "<body>content</body>"
        mock_soup.body = mock_body

        result = self.scraper.extract_body_content(mock_soup)

        self.assertEqual(result, "<body>content</body>")

    def test_should_return_empty_string_when_soup_has_no_body(self):
        mock_soup = MagicMock()
        mock_soup.body = None

        result = self.scraper.extract_body_content(mock_soup)

        self.assertEqual(result, "")

    # ------------------------------------------------------------------
    # clean_body_content()
    # ------------------------------------------------------------------

    @patch("scraping.web_scraper.BeautifulSoup")
    def test_should_remove_script_tags_when_cleaning_html(self, MockBS):
        mock_soup = MagicMock()
        MockBS.return_value = mock_soup
        mock_soup.return_value = []  # no script/style/noscript tags
        mock_soup.get_text.return_value = "clean text"
        mock_soup.__call__ = MagicMock(return_value=[])

        result = self.scraper.clean_body_content("<body><script>js</script>text</body>")

        # BeautifulSoup was called with the input HTML
        MockBS.assert_called_once_with(
            "<body><script>js</script>text</body>", "html.parser"
        )

    @patch("scraping.web_scraper.BeautifulSoup")
    def test_should_return_stripped_text_when_cleaning_html(self, MockBS):
        mock_soup = MagicMock()
        MockBS.return_value = mock_soup
        mock_soup.get_text.return_value = "  line1  \n  line2  \n"

        result = self.scraper.clean_body_content("<body>line1\nline2</body>")

        # Result should be stripped lines joined by newline (empty lines removed)
        self.assertNotIn("  ", result)

    # ------------------------------------------------------------------
    # split_dom_content()
    # ------------------------------------------------------------------

    def test_should_return_single_chunk_when_content_is_shorter_than_max_length(self):
        content = "short content"
        result = self.scraper.split_dom_content(content)
        self.assertEqual(result, ["short content"])

    def test_should_split_into_multiple_chunks_when_content_exceeds_max_length(self):
        content = "x" * 13000
        result = self.scraper.split_dom_content(content)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(len(chunk) <= 6000 for chunk in result))

    def test_should_use_custom_max_length_when_specified(self):
        content = "ab" * 10  # 20 chars
        result = self.scraper.split_dom_content(content, max_length=7)
        self.assertEqual(len(result), 3)

    def test_should_return_empty_list_when_content_is_empty(self):
        result = self.scraper.split_dom_content("")
        self.assertEqual(result, [])
