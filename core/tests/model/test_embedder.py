import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.src.model.embedder import Embedder
from core.src.persistence.vector_store import VectorStore


class TestEmbedder(unittest.TestCase):

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------

    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_initialise_sentence_transformer_when_created(self, mock_sentence_transformer):
        """Embedder.__init__ loads the SentenceTransformer model exactly once."""
        Embedder()
        mock_sentence_transformer.assert_called_once()

    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_have_no_store_when_first_created(self, _mock_st):
        """store is None until get_store() is called for the first time."""
        embedder = Embedder()
        self.assertIsNone(embedder.store)

    # ------------------------------------------------------------------
    # build_metadata()
    # ------------------------------------------------------------------

    def test_should_include_primitive_values_when_building_metadata(self):
        """build_metadata keeps str, int, float, and bool values as-is."""
        doc = {
            "text": "A modern two-bedroom apartment.",
            "price": 2200,
            "bedrooms": 2,
            "available": True,
            "rating": 4.5,
        }
        metadata = Embedder.build_metadata(doc)
        self.assertEqual(metadata, {
            "text": "A modern two-bedroom apartment.",
            "price": 2200,
            "bedrooms": 2,
            "available": True,
            "rating": 4.5,
        })

    def test_should_exclude_none_values_when_building_metadata(self):
        """build_metadata silently drops keys whose value is None."""
        doc = {"text": "Nice flat", "notes": None}
        metadata = Embedder.build_metadata(doc)
        self.assertNotIn("notes", metadata)
        self.assertEqual(metadata["text"], "Nice flat")

    def test_should_exclude_nan_string_when_building_metadata(self):
        """build_metadata drops keys whose stringified value is 'nan'."""
        doc = {"text": "Nice flat", "floor": float("nan")}
        metadata = Embedder.build_metadata(doc)
        self.assertNotIn("floor", metadata)

    def test_should_exclude_empty_string_when_building_metadata(self):
        """build_metadata drops keys whose value is an empty string."""
        doc = {"text": "Nice flat", "description": ""}
        metadata = Embedder.build_metadata(doc)
        self.assertNotIn("description", metadata)

    def test_should_exclude_none_string_when_building_metadata(self):
        """build_metadata drops keys whose stringified value is 'none'."""
        doc = {"text": "Nice flat", "owner": "none"}
        metadata = Embedder.build_metadata(doc)
        self.assertNotIn("owner", metadata)

    def test_should_convert_non_primitive_values_to_string_when_building_metadata(self):
        """build_metadata converts lists and other complex types to their str() form."""
        doc = {"text": "flat", "features": ["garden", "parking"]}
        metadata = Embedder.build_metadata(doc)
        self.assertEqual(metadata["features"], "['garden', 'parking']")

    # ------------------------------------------------------------------
    # embed_documents_to_vectors()
    # ------------------------------------------------------------------

    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_return_one_vector_per_document_when_embedding_documents(self, mock_sentence_transformer):
        """embed_documents_to_vectors returns a list with the same length as the input dict."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        embedder = Embedder()

        documents = {
            "doc1": {"text": "This is a test document."},
            "doc2": {"text": "This is another test document."},
        }
        mock_model.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

        vectors = embedder.embed_documents_to_vectors(documents)

        self.assertEqual(len(vectors), 2)

    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_use_document_key_as_id_when_embedding_documents(self, mock_sentence_transformer):
        """embed_documents_to_vectors sets the document dict key as the vector id."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        mock_model.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        embedder = Embedder()

        vectors = embedder.embed_documents_to_vectors({
            "doc1": {"text": "first"},
            "doc2": {"text": "second"},
        })

        self.assertEqual(vectors[0]["id"], "doc1")
        self.assertEqual(vectors[1]["id"], "doc2")

    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_cast_embeddings_to_float32_when_embedding_documents(self, mock_sentence_transformer):
        """embed_documents_to_vectors stores float32 numpy arrays in the 'values' field."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        mock_model.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        embedder = Embedder()

        vectors = embedder.embed_documents_to_vectors({
            "doc1": {"text": "a"},
            "doc2": {"text": "b"},
        })

        self.assertTrue(np.array_equal(vectors[0]["values"], np.array([0.1, 0.2], dtype="float32")))
        self.assertTrue(np.array_equal(vectors[1]["values"], np.array([0.3, 0.4], dtype="float32")))

    # ------------------------------------------------------------------
    # get_store()
    # ------------------------------------------------------------------

    @patch('core.src.model.embedder.create_vector_store')
    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_create_and_return_store_when_get_store_called_first_time(
        self, _mock_st, mock_create_vector_store
    ):
        """get_store() creates the vector store on the first call and returns it."""
        mock_store = MagicMock(spec=VectorStore)
        mock_create_vector_store.return_value = mock_store
        embedder = Embedder()

        store = embedder.get_store()

        self.assertEqual(store, mock_store)
        mock_create_vector_store.assert_called_once()

    @patch('core.src.model.embedder.create_vector_store')
    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_not_recreate_store_when_get_store_called_multiple_times(
        self, _mock_st, mock_create_vector_store
    ):
        """get_store() is lazy — the store factory is called only once, no matter how many calls."""
        mock_store = MagicMock(spec=VectorStore)
        mock_create_vector_store.return_value = mock_store
        embedder = Embedder()

        store1 = embedder.get_store()
        store2 = embedder.get_store()

        self.assertEqual(store1, store2)
        mock_create_vector_store.assert_called_once()

    # ------------------------------------------------------------------
    # save_vectors_in_store()
    # ------------------------------------------------------------------

    @patch('core.src.model.embedder.create_vector_store')
    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_call_add_vectors_on_store_when_saving_vectors(
        self, _mock_st, mock_create_vector_store
    ):
        """save_vectors_in_store() delegates to store.add_vectors() with the given payload."""
        # No spec= here: add_vectors is defined on PineconeStore, not the abstract base.
        mock_store = MagicMock()
        mock_create_vector_store.return_value = mock_store
        embedder = Embedder()

        vectors = [{"id": "v1", "values": np.array([0.1, 0.2]), "metadata": {}}]
        embedder.save_vectors_in_store(vectors)

        mock_store.add_vectors.assert_called_once_with(vectors)

    # ------------------------------------------------------------------
    # search()
    # ------------------------------------------------------------------

    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_return_store_search_results_when_searching(self, mock_sentence_transformer):
        """search() embeds the query and returns whatever the store returns."""
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

    # ------------------------------------------------------------------
    # embed_texts()
    # ------------------------------------------------------------------

    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_return_float32_array_when_embedding_texts(self, mock_sentence_transformer):
        """embed_texts() returns a float32 numpy array with one row per input text."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        embedder = Embedder()

        mock_model.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

        vectors = embedder.embed_texts(["text1", "text2"])

        self.assertTrue(
            np.array_equal(vectors, np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32"))
        )

    # ------------------------------------------------------------------
    # embed_query()
    # ------------------------------------------------------------------

    @patch('core.src.model.embedder.SentenceTransformer')
    def test_should_return_float32_array_when_embedding_query(self, mock_sentence_transformer):
        """embed_query() wraps the single query in a list and returns a float32 array."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        embedder = Embedder()

        mock_model.encode.return_value = [np.array([0.1, 0.2])]

        vector = embedder.embed_query("test query")

        self.assertTrue(
            np.array_equal(vector, np.array([[0.1, 0.2]], dtype="float32"))
        )


if __name__ == "__main__":
    unittest.main()
