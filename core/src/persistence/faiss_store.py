# vector_store/faiss_store.py
import faiss
import numpy as np
from .vector_store import VectorStore

class FAISSStore(VectorStore):

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, vectors: np.ndarray, metadata: list[dict]):
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int):
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")

        distances, indices = self.index.search(query_vector, k)
        return [
            self.metadata[i]
            for i in indices[0]
            if i != -1
        ]

