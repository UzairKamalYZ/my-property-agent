from typing import Any

from sentence_transformers import SentenceTransformer
import numpy as np

from agentP.src.config import Config
from agentP.src.scraping.utils import listings_to_documents
from agentP.src.persistence.factory import create_vector_store


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
        self.store = None
        self.documents = []

    def build_metadata(self, doc: dict) -> dict:
        metadata = {}

        for key, value in doc.items():
            if value is None:
                continue

            value_str = str(value).lower()
            if value_str in {"nan", "", "none"}:
                continue

            # Pinecone metadata must be primitive types
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = str(value)

        return metadata

    def embed_documents_to_vectors(self, documents: dict) -> list:
        """Convert listings into vectors."""
        # self.documents = listings_to_documents(listings)
        # if not self.documents:
        #     raise ValueError("No documents to embed")
        #
        # metadata = [doc.metadata for doc in self.documents]
        texts = []
        ids = []
        metadatas = []

        for doc_id, doc in documents.items():
            ids.append(doc_id)
            texts.append(doc["text"])
            metadatas.append(self.build_metadata(doc))

        embeddings = np.array([self.model.encode(doc) for doc in texts], dtype='float32')

        vectors = []
        for i in range(len(ids)):
            vectors.append({
                "id": ids[i],
                "values": embeddings[i],
                "metadata": metadatas[i]
            })


        return vectors

    def save_vectors_in_store(self, vectors: list[Any]):
        self.__getStore__().add_vectors(vectors)

    def __getStore__(self,dim=384):
        self.store = create_vector_store(
            store_type=Config.STORE_TYPE,
            dim=dim,
            config={}
        )
        return self.store

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts)
        return np.array(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        return np.array([self.model.encode(query)], dtype="float32")