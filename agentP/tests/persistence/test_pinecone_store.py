import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from persistence.pinecone_store import PineconeStore


class TestPineconeStore(unittest.TestCase):

    def _make_store(self, index_exists=False):
        """Helper: build PineconeStore with mocked Pinecone client."""
        mock_client = MagicMock()
        mock_index_list = MagicMock()
        mock_index_list.names.return_value = ["property-agent"] if index_exists else []
        mock_client.list_indexes.return_value = mock_index_list
        mock_client.Index.return_value = MagicMock()

        with patch("persistence.pinecone_store.Pinecone", return_value=mock_client):
            store = PineconeStore(
                index_name="property-agent",
                dim=384,
                api_key="test-key",
                env="us-east-1",
            )
        return store, mock_client

    def test_should_create_pinecone_client_with_api_key_when_initialized(self):
        mock_client = MagicMock()
        mock_client.list_indexes.return_value.names.return_value = ["property-agent"]

        with patch("persistence.pinecone_store.Pinecone", return_value=mock_client) as MockPinecone:
            PineconeStore("property-agent", 384, "my-api-key", "us-east-1")

        MockPinecone.assert_called_once_with(
            api_key="my-api-key", environment="us-east-1"
        )

    def test_should_create_index_when_index_does_not_exist(self):
        mock_client = MagicMock()
        mock_client.list_indexes.return_value.names.return_value = []
        mock_client.Index.return_value = MagicMock()

        with patch("persistence.pinecone_store.Pinecone", return_value=mock_client):
            PineconeStore("new-index", 384, "key", "env")

        mock_client.create_index.assert_called_once()

    def test_should_skip_index_creation_when_index_already_exists(self):
        mock_client = MagicMock()
        mock_client.list_indexes.return_value.names.return_value = ["property-agent"]
        mock_client.Index.return_value = MagicMock()

        with patch("persistence.pinecone_store.Pinecone", return_value=mock_client):
            PineconeStore("property-agent", 384, "key", "env")

        mock_client.create_index.assert_not_called()

    def test_should_assign_index_attribute_when_initialized(self):
        store, _ = self._make_store()
        self.assertIsNotNone(store.index)

    def test_should_call_upsert_when_adding_vectors_with_metadata(self):
        store, _ = self._make_store()
        vectors = np.ones((2, 384), dtype=np.float32)
        metadata = [{"city": "Warsaw"}, {"city": "Krakow"}]
        store.add(vectors, metadata)
        store.index.upsert.assert_called_once()

    def test_should_call_upsert_directly_when_add_vectors_called(self):
        store, _ = self._make_store()
        vectors = [("id1", [0.1, 0.2], {"city": "Warsaw"})]
        store.add_vectors(vectors)
        store.index.upsert.assert_called_once_with(vectors)

    def test_should_return_metadata_list_when_searching(self):
        store, _ = self._make_store()
        store.index.query.return_value = {
            "matches": [
                {"metadata": {"city": "Warsaw"}},
                {"metadata": {"city": "Krakow"}},
            ]
        }
        query = np.ones((1, 384), dtype=np.float32)
        results = store.search(query, k=2)
        self.assertEqual(results, [{"city": "Warsaw"}, {"city": "Krakow"}])

    def test_should_pass_k_as_top_k_when_searching(self):
        store, _ = self._make_store()
        store.index.query.return_value = {"matches": []}
        query = np.ones((1, 384), dtype=np.float32)
        store.search(query, k=7)
        call_kwargs = store.index.query.call_args[1]
        self.assertEqual(call_kwargs["top_k"], 7)

    def test_should_return_empty_list_when_no_matches_found(self):
        store, _ = self._make_store()
        store.index.query.return_value = {"matches": []}
        query = np.ones((1, 384), dtype=np.float32)
        results = store.search(query, k=5)
        self.assertEqual(results, [])

    def test_should_include_metadata_in_upsert_items_when_adding(self):
        store, _ = self._make_store()
        vectors = np.array([[0.1, 0.2]], dtype=np.float32)
        metadata = [{"city": "Warsaw"}]
        store.add(vectors, metadata)
        call_args = store.index.upsert.call_args[0][0]
        # Each item is (uuid, vector_list, metadata_dict)
        self.assertEqual(len(call_args), 1)
        _id, _vector, _meta = call_args[0]
        self.assertEqual(_meta, {"city": "Warsaw"})
