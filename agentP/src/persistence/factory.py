# vector_store/factory.py
import logging
from .faiss_store import FAISSStore
from .pinecone_store import PineconeStore
from agentP.src.config.config import Config

logger = logging.getLogger(__name__)


def create_vector_store(store_type: str, dim: int, config: dict):
    logger.info("Creating vector store type=%s dim=%d", store_type, dim)
    if store_type == "local":
        store = FAISSStore(dim)
        logger.debug("FAISSStore created dim=%d", dim)
        return store
    elif store_type == "pinecone":
        store = PineconeStore(
            index_name=Config.PINECONE_INDEX_NAME,
            dim=dim,
            api_key=Config.PINECONE_API_KEY,
            env=Config.PINECONE_ENVIRONMENT
        )
        logger.debug("PineconeStore created index=%s dim=%d",
                     Config.PINECONE_INDEX_NAME, dim)
        return store
    else:
        logger.error("Unknown vector store type: %s", store_type)
        raise ValueError(f"Unknown vector store profile: {store_type}")
