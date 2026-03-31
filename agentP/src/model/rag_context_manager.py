import logging
from agentP.src.model.embedder import Embedder
from agentP.src.model.context_builder import build_context_from_listings

logger = logging.getLogger(__name__)


class RagContextManager:
    """Manages the retrieval and building of context for RAG prompts."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def get_context(self, user_prompt: str, k: int = 5) -> str:
        """
        Retrieves relevant listings based on a user prompt and builds a
        formatted context string.
        """
        logger.debug("Retrieving context: query=%r k=%d", user_prompt, k)
        listings = self.embedder.search(user_prompt, k=k)
        count = len(listings) if listings else 0
        logger.info("Context retrieved: %d listing(s) for query_len=%d",
                    count, len(user_prompt))
        return build_context_from_listings(listings)
