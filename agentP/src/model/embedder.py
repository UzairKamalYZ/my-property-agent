import logging
from typing import Any, Dict, List, Union, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

from agentP.src.config.config import Config
from agentP.src.persistence.factory import create_vector_store
from agentP.src.persistence.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Embedder:

    def __init__(self) -> None:
        logger.info("Loading SentenceTransformer model=%s",
                    Config.SENTENCE_TRANSFORMER_MODEL)
        self.model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
        self.store: Union[VectorStore, None] = None
        logger.debug("Embedder ready")

    @staticmethod
    def build_metadata(doc: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        metadata: Dict[str, Union[str, int, float, bool]] = {}
        for key, value in doc.items():
            if value is None:
                continue

            value_str = str(value).lower()
            if value_str in {"nan", "", "none"}:
                continue

            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = str(value)

        return metadata

    def embed_documents_to_vectors(self, documents: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert documents into vectors."""
        logger.info("Embedding %d document(s)", len(documents))
        ids, texts, metadatas = self._preprocess_documents(documents)
        embeddings = self.embed_texts(texts)
        vectors = self._create_vectors(ids, embeddings, metadatas)
        logger.debug("Produced %d vector(s)", len(vectors))
        return vectors

    def _preprocess_documents(self, documents: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Union[str, int, float, bool]]]]:
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Union[str, int, float, bool]]] = []

        for doc_id, doc in documents.items():
            ids.append(doc_id)
            texts.append(doc["text"])
            metadatas.append(self.build_metadata(doc))

        return ids, texts, metadatas

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into a numpy array of vectors."""
        logger.debug("Encoding %d text(s)", len(texts))
        vectors = self.model.encode(texts)
        return np.array(vectors, dtype="float32")

    @staticmethod
    def _create_vectors(ids: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Union[str, int, float, bool]]]) -> List[Dict[str, Any]]:
        vectors: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            vectors.append({
                "id": ids[i],
                "values": embeddings[i],
                "metadata": metadatas[i]
            })
        return vectors

    def save_vectors_in_store(self, vectors: List[Dict[str, Any]]) -> None:
        logger.info("Saving %d vector(s) to store", len(vectors))
        self.get_store().add_vectors(vectors)
        logger.debug("Vectors saved successfully")

    def get_store(self, dim: int = 384) -> VectorStore:
        if not self.store:
            logger.info("Initializing vector store type=%s dim=%d",
                        Config.STORE_TYPE, dim)
            self.store = create_vector_store(
                store_type=Config.STORE_TYPE,
                dim=dim,
                config={}
            )
            logger.debug("Vector store initialized")
        return self.store

    def search(self, query: str, k: int = 5) -> Any:
        logger.debug("Searching vector store: query_len=%d k=%d", len(query), k)
        store = self.get_store()
        if store is None:
            logger.error("Vector index not initialized — call embed_documents_to_vectors first")
            raise ValueError("Vector index not initialized. Build embeddings first.")

        query_vec = self.embed_query(query)
        results = store.search(query_vec, k)
        logger.info("Search returned %d result(s) for query_len=%d",
                    len(results) if results else 0, len(query))
        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query into a numpy array."""
        logger.debug("Embedding query: query_len=%d", len(query))
        return np.array(self.model.encode([query]), dtype="float32")
