from typing import Any, Dict, List, Union, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

from config.config import Config
from persistence.factory import create_vector_store
from persistence.vector_store import VectorStore


class Embedder:

    def __init__(self) -> None:
        self.model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
        self.store: Union[VectorStore, None] = None

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
        ids, texts, metadatas = self._preprocess_documents(documents)
        embeddings = self.embed_texts(texts)
        return self._create_vectors(ids, embeddings, metadatas)

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
        self.get_store().add_vectors(vectors)

    def get_store(self, dim: int = 384) -> VectorStore:
        if not self.store:
            self.store = create_vector_store(
                store_type=Config.STORE_TYPE,
                dim=dim,
                config={}
            )
        return self.store

    def search(self, query: str, k: int = 5) -> Any:
        store = self.get_store()
        if store is None:
            raise ValueError("Vector index not initialized. Build embeddings first.")

        query_vec = self.embed_query(query)
        return store.search(query_vec, k)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query into a numpy array."""
        return np.array(self.model.encode([query]), dtype="float32")