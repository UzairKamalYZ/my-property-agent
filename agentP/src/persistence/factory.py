# vector_store/factory.py
from .faiss_store import FAISSStore
from .pinecone_store import PineconeStore
from agentP.src.config.config import Config


def create_vector_store(store_type: str, dim: int, config: dict):
    if store_type == "local":
        return FAISSStore(dim)
    elif store_type == "pinecone":
        return PineconeStore(
            index_name=Config.PINECONE_INDEX_NAME,
            dim=dim,
            api_key=Config.PINECONE_API_KEY,
            env=Config.PINECONE_ENVIRONMENT
        )
    else:
        raise ValueError(f"Unknown vector store profile: {store_type}")
