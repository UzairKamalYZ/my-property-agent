import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from agentP.src.model.embedder import Embedder
from agentP.src.persistence.vector_store import VectorStore


class TestEmbedder(unittest.TestCase):

    @patch('agentP.src.model.embedder.SentenceTransformer')
    def test_init(self, mock_sentence_transformer):
        embedder = Embedder()
        mock_sentence_transformer.assert_called_once()
        self.assertIsNone(embedder.store)

    def test_build_metadata(self):
        doc = {
            "text": "A modern two-bedroom apartment.",
            "price": 2200,
            "bedrooms": 2,
            "available": True,
            "notes": None,
            "features": ["garden", "parking"],
            "rating": 4.5
        }
        metadata = Embedder.build_metadata(doc)
        expected_metadata = {
            "text": "A modern two-bedroom apartment.",
            "price": 2200,
            "bedrooms": 2,
            "available": True,
            "features": "['garden', 'parking']",
            "rating": 4.5
        }
        self.assertEqual(metadata, expected_metadata)

    @patch('agentP.src.model.embedder.SentenceTransformer')
    def test_embed_documents_to_vectors(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        embedder = Embedder()

        documents = {
            "doc1": {"text": "This is a test document."},
            "doc2": {"text": "This is another test document."}
        }
        
        mock_model.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        
        vectors = embedder.embed_documents_to_vectors(documents)
        
        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0]['id'], 'doc1')
        self.assertEqual(vectors[1]['id'], 'doc2')
        self.assertTrue(np.array_equal(vectors[0]['values'], np.array([0.1, 0.2], dtype='float32')))
        self.assertTrue(np.array_equal(vectors[1]['values'], np.array([0.3, 0.4], dtype='float32')))

    @patch('agentP.src.model.embedder.create_vector_store')
    @patch('agentP.src.model.embedder.SentenceTransformer')
    def test_get_store(self, mock_sentence_transformer, mock_create_vector_store):
        mock_store = MagicMock(spec=VectorStore)
        mock_create_vector_store.return_value = mock_store
        embedder = Embedder()
        
        store = embedder.get_store()
        
        self.assertIsNotNone(store)
        self.assertEqual(store, mock_store)
        mock_create_vector_store.assert_called_once()
        
        # Test that the store is not created again
        store2 = embedder.get_store()
        self.assertEqual(store2, mock_store)
        mock_create_vector_store.assert_called_once()

    @patch('agentP.src.model.embedder.SentenceTransformer')
    def test_search(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        embedder = Embedder()

        mock_store = MagicMock(spec=VectorStore)
        mock_store.search.return_value = "search results"
        embedder.store = mock_store
        
        mock_model.encode.return_value = [np.array([0.1, 0.2])]
        
        results = embedder.search("test query")
        
        self.assertEqual(results, "search results")
        mock_store.search.assert_called_once()

    @patch('agentP.src.model.embedder.SentenceTransformer')
    def test_embed_texts(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        embedder = Embedder()
        
        texts = ["text1", "text2"]
        mock_model.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        
        vectors = embedder.embed_texts(texts)
        
        self.assertTrue(np.array_equal(vectors, np.array([[0.1, 0.2], [0.3, 0.4]], dtype='float32')))

    @patch('agentP.src.model.embedder.SentenceTransformer')
    def test_embed_query(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        embedder = Embedder()

        query = "test query"
        mock_model.encode.return_value = [np.array([0.1, 0.2])]
        
        vector = embedder.embed_query(query)
        
        self.assertTrue(np.array_equal(vector, np.array([[0.1, 0.2]], dtype='float32')))

if __name__ == '__main__':
    unittest.main()