# vector_store/faiss_store.py
import logging
import faiss
import numpy as np
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class FAISSStore(VectorStore):

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        logger.debug("FAISSStore initialized dim=%d", dim)

    def add(self, vectors: np.ndarray, metadata: list[dict]):
        self.index.add(vectors)
        self.metadata.extend(metadata)
        logger.debug("FAISSStore: added %d vector(s), total=%d",
                     len(metadata), len(self.metadata))

    def search(self, query_vector: np.ndarray, k: int):
        if self.index is None:
            logger.error("FAISS index not initialized")
            raise RuntimeError("FAISS index not initialized")

        distances, indices = self.index.search(query_vector, k)
        results = [
            self.metadata[i]
            for i in indices[0]
            if i != -1
        ]
        logger.debug("FAISSStore: search k=%d returned %d result(s)",
                     k, len(results))
        return results
