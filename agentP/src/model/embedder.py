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


    def embed(self, listings: dict):
        """Convert listings into vectors."""
        self.documents = listings_to_documents(listings)
        if not self.documents:
            raise ValueError("No documents to embed")

        metadata = [doc.metadata for doc in self.documents]

        # Create vectors
        vectors = np.array([self.model.encode(doc.page_content) for doc in self.documents], dtype='float32')

        self.__getStore__().add(vectors, metadata)

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
