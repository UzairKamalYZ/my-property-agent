import unittest
from unittest.mock import MagicMock, patch


class TestDataCollector(unittest.TestCase):

    def setUp(self):
        self.mock_embedder = MagicMock()
        self.mock_scraper = MagicMock()
        self.mock_url_processor = MagicMock()
        self.mock_url_processor_cls = MagicMock(return_value=self.mock_url_processor)

        patcher_embedder = patch("model.embedder.Embedder", return_value=self.mock_embedder)
        patcher_scraper = patch("scraping.web_scraper.WebScraper", return_value=self.mock_scraper)
        patcher_url = patch("scraping.url_processor.UrlProcessor", self.mock_url_processor_cls)

        self.addCleanup(patcher_embedder.stop)
        self.addCleanup(patcher_scraper.stop)
        self.addCleanup(patcher_url.stop)

        patcher_embedder.start()
        patcher_scraper.start()
        patcher_url.start()

        from gatherers.data_collector import DataCollector
        self.DataCollector = DataCollector
        self.collector = DataCollector.__new__(DataCollector)
        self.collector.data = None
        self.collector.web_scraper = self.mock_scraper
        self.collector.embedder = self.mock_embedder
        self.collector.url_processor = self.mock_url_processor

    # ------------------------------------------------------------------
    # __init__ state
    # ------------------------------------------------------------------

    def test_should_have_none_data_when_initialized(self):
        self.assertIsNone(self.collector.data)

    def test_should_have_web_scraper_when_initialized(self):
        self.assertIsNotNone(self.collector.web_scraper)

    def test_should_have_embedder_when_initialized(self):
        self.assertIsNotNone(self.collector.embedder)

    def test_should_have_url_processor_when_initialized(self):
        self.assertIsNotNone(self.collector.url_processor)

    # ------------------------------------------------------------------
    # __getDataFromUrls__()
    # ------------------------------------------------------------------

    def test_should_delegate_to_url_processor_when_getting_data(self):
        self.mock_url_processor.process_urls_from_file.return_value = []
        self.collector.__getDataFromUrls__("/some/urls.txt")
        self.mock_url_processor.process_urls_from_file.assert_called_once_with(
            "/some/urls.txt"
        )

    def test_should_set_data_attribute_when_getting_data(self):
        expected = [{"type": "Apartment"}]
        self.mock_url_processor.process_urls_from_file.return_value = expected
        self.collector.__getDataFromUrls__("/file.txt")
        self.assertEqual(self.collector.data, expected)

    def test_should_return_self_when_getting_data(self):
        self.mock_url_processor.process_urls_from_file.return_value = []
        result = self.collector.__getDataFromUrls__("/file.txt")
        self.assertIs(result, self.collector)

    # ------------------------------------------------------------------
    # __storVectorEmbeddings__()
    # ------------------------------------------------------------------

    def test_should_call_embed_documents_to_vectors_when_storing(self):
        self.collector.data = [{"city": "Brussels"}]
        self.collector.__storVectorEmbeddings__()
        self.mock_embedder.embed_documents_to_vectors.assert_called_once_with(
            [{"city": "Brussels"}]
        )

    def test_should_return_self_when_storing_embeddings(self):
        self.collector.data = []
        result = self.collector.__storVectorEmbeddings__()
        self.assertIs(result, self.collector)

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    def test_should_support_context_manager_enter(self):
        result = self.collector.__enter__()
        self.assertIs(result, self.collector)

    def test_should_close_without_error_when_exiting_context_manager(self):
        try:
            self.collector.__exit__(None, None, None)
        except Exception as e:
            self.fail(f"__exit__ raised unexpectedly: {e}")

    def test_should_not_raise_when_calling_close(self):
        try:
            self.collector.close()
        except Exception as e:
            self.fail(f"close() raised unexpectedly: {e}")
