# vector_store/pinecone_store.py
import pinecone
import uuid
from .vector_store import VectorStore

class PineconeStore(VectorStore):

    def __init__(self, index_name: str, dim: int, api_key: str, env: str):
        pinecone.init(api_key=api_key, environment=env)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dim)
        self.index = pinecone.Index(index_name)

    def add(self, vectors, metadata):
        items = []
        for vector, meta in zip(vectors, metadata):
            items.append((str(uuid.uuid4()), vector.tolist(), meta))
        self.index.upsert(items)

    def search(self, query_vector, k):
        results = self.index.query(
            vector=query_vector[0].tolist(),
            top_k=k,
            include_metadata=True
        )
        return [m["metadata"] for m in results["matches"]]
