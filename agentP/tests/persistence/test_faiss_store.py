import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from persistence.faiss_store import FAISSStore


class TestFAISSStore(unittest.TestCase):

    def setUp(self):
        # faiss is stubbed in conftest — grab the mock index
        import faiss
        self.mock_index = MagicMock()
        faiss.IndexFlatL2 = MagicMock(return_value=self.mock_index)
        self.store = FAISSStore(dim=384)

    def test_should_call_index_flat_l2_with_dim_when_created(self):
        import faiss
        faiss.IndexFlatL2.assert_called_with(384)

    def test_should_have_empty_metadata_list_when_created(self):
        self.assertEqual(self.store.metadata, [])

    def test_should_call_index_add_when_adding_vectors(self):
        vectors = np.ones((2, 384), dtype=np.float32)
        metadata = [{"id": "a"}, {"id": "b"}]
        self.store.add(vectors, metadata)
        self.mock_index.add.assert_called_once_with(vectors)

    def test_should_extend_metadata_when_adding_vectors(self):
        vectors = np.ones((2, 384), dtype=np.float32)
        metadata = [{"id": "a"}, {"id": "b"}]
        self.store.add(vectors, metadata)
        self.assertEqual(self.store.metadata, metadata)

    def test_should_accumulate_metadata_across_multiple_adds(self):
        vectors = np.ones((1, 384), dtype=np.float32)
        self.store.add(vectors, [{"id": "a"}])
        self.store.add(vectors, [{"id": "b"}])
        self.assertEqual(len(self.store.metadata), 2)

    def test_should_return_matching_metadata_when_searching(self):
        self.store.metadata = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        self.mock_index.search.return_value = (
            np.array([[0.1, 0.2]]),
            np.array([[0, 2]]),
        )
        query = np.ones((1, 384), dtype=np.float32)
        results = self.store.search(query, k=2)
        self.assertEqual(results, [{"id": "a"}, {"id": "c"}])

    def test_should_filter_negative_one_indices_when_searching(self):
        self.store.metadata = [{"id": "a"}, {"id": "b"}]
        self.mock_index.search.return_value = (
            np.array([[0.1, 0.9]]),
            np.array([[0, -1]]),
        )
        query = np.ones((1, 384), dtype=np.float32)
        results = self.store.search(query, k=2)
        self.assertEqual(results, [{"id": "a"}])

    def test_should_return_empty_list_when_all_indices_are_negative_one(self):
        self.store.metadata = [{"id": "a"}]
        self.mock_index.search.return_value = (
            np.array([[99.0]]),
            np.array([[-1]]),
        )
        query = np.ones((1, 384), dtype=np.float32)
        results = self.store.search(query, k=1)
        self.assertEqual(results, [])

    def test_should_pass_query_and_k_to_index_search_when_searching(self):
        self.store.metadata = []
        self.mock_index.search.return_value = (np.array([[]]), np.array([[]]))
        query = np.ones((1, 384), dtype=np.float32)
        self.store.search(query, k=3)
        self.mock_index.search.assert_called_once_with(query, 3)
