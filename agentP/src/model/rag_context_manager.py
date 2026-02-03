from agentP.src.model.embedder import Embedder
from agentP.src.model.context_builder import build_context_from_listings


class RagContextManager:
    """Manages the retrieval and building of context for RAG prompts."""

    def __init__(self):
        self.embedder = Embedder()

    def _get_context(self, user_prompt: str, k: int = 5) -> str:
        """
        Retrieves relevant listings based on a user prompt and builds a
        formatted context string.
        """
        listings = self.embedder.search(user_prompt, k=k)
        return build_context_from_listings(listings)

    def prepare_rag_prompt(self, user_prompt: str) -> str:
        """
        Prepares the full RAG prompt by retrieving context and merging it
        with the user's question.
        """
        context = self._get_context(user_prompt)
        return f"""
                        User question:
                        {user_prompt}
                        
                        Available listings:
                        {context}
                        
                        Answer strictly using the listings above.
                        If no listings match, say "No matching listings found".
                    """
