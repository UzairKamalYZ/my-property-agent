# vector_store/pinecone_store.py
import logging
import uuid
from pinecone import Pinecone, ServerlessSpec
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class PineconeStore(VectorStore):

    def __init__(self, index_name: str, dim: int, api_key: str, env: str):
        logger.info("Connecting to Pinecone index=%s env=%s", index_name, env)
        pinecone = Pinecone(api_key=api_key, environment=env)
        existing = pinecone.list_indexes().names()
        if index_name not in existing:
            logger.info("Index '%s' not found — creating (dim=%d, metric=cosine)",
                        index_name, dim)
            pinecone.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        else:
            logger.debug("Index '%s' already exists — skipping creation", index_name)
        self.index = pinecone.Index(index_name)
        logger.debug("PineconeStore ready index=%s", index_name)

    def add(self, vectors, metadata):
        items = []
        for vector, meta in zip(vectors, metadata):
            items.append((str(uuid.uuid4()), vector.tolist(), meta))
        self.index.upsert(items)
        logger.debug("Pinecone: upserted %d vector(s)", len(items))

    def add_vectors(self, vectors):
        self.index.upsert(vectors)
        logger.debug("Pinecone: add_vectors upserted %d item(s)", len(vectors))

    def search(self, query_vector, k):
        logger.debug("Pinecone: querying top_k=%d", k)
        results = self.index.query(
            vector=query_vector[0].tolist(),
            top_k=k,
            include_metadata=True
        )
        matches = [m["metadata"] for m in results["matches"]]
        logger.debug("Pinecone: query returned %d match(es)", len(matches))
        return matches
