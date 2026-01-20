from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from .utils import listings_to_documents

class Embedder:
    """Builds and queries a FAISS vector index using SentenceTransformer embeddings."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def embed(self, listings: list[dict]):
        """Convert listings into vectors and build FAISS index."""
        self.documents = listings_to_documents(listings)
        if not self.documents:
            raise ValueError("No documents to embed")

        # Create vectors
        vectors = np.array([self.model.encode(doc.page_content) for doc in self.documents], dtype='float32')

        # Build FAISS index
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

        return self.index

    def rank_listings(self, query: str, k: int = 5):
        """Return top-k listings by semantic similarity."""
        if self.index is None:
            raise ValueError("Index not built yet. Call embed() first.")

        query_vec = np.array([self.model.encode(query)], dtype='float32')
        distances, indices = self.index.search(query_vec, k)
        top_docs = [self.documents[i].metadata for i in indices[0]]
        return top_docs
