import unittest
from unittest.mock import MagicMock, patch

from model.rag_context_manager import RagContextManager


class TestRagContextManager(unittest.TestCase):

    def setUp(self):
        self.mock_embedder = MagicMock()
        self.manager = RagContextManager(self.mock_embedder)

    def test_should_store_embedder_when_initialized(self):
        self.assertIs(self.manager.embedder, self.mock_embedder)

    def test_should_call_embedder_search_with_prompt_when_getting_context(self):
        self.mock_embedder.search.return_value = []
        self.manager.get_context("2 bed apartment in Warsaw")
        self.mock_embedder.search.assert_called_once_with("2 bed apartment in Warsaw", k=5)

    def test_should_use_default_k_of_5_when_k_not_specified(self):
        self.mock_embedder.search.return_value = []
        self.manager.get_context("any prompt")
        _, kwargs = self.mock_embedder.search.call_args
        self.assertEqual(kwargs["k"], 5)

    def test_should_use_custom_k_when_k_specified(self):
        self.mock_embedder.search.return_value = []
        self.manager.get_context("any prompt", k=10)
        self.mock_embedder.search.assert_called_once_with("any prompt", k=10)

    def test_should_return_no_listings_message_when_search_returns_empty(self):
        self.mock_embedder.search.return_value = []
        result = self.manager.get_context("anything")
        self.assertEqual(result, "No listings found.")

    def test_should_return_formatted_context_when_listings_found(self):
        self.mock_embedder.search.return_value = [
            {"city": "Warsaw", "price": 400000}
        ]
        result = self.manager.get_context("apartment")
        self.assertIn("Listing 1", result)
        self.assertIn("Warsaw", result)

    def test_should_pass_search_results_to_context_builder_when_getting_context(self):
        listings = [{"city": "Krakow"}, {"city": "Gdansk"}]
        self.mock_embedder.search.return_value = listings
        result = self.manager.get_context("flat")
        self.assertIn("Listing 1", result)
        self.assertIn("Listing 2", result)
