# vector_store/base.py
from abc import ABC, abstractmethod
import numpy as np

class VectorStore(ABC):

    @abstractmethod
    def add(self, vectors: np.ndarray, metadatas: list[dict]):
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int):
        pass
