from pathlib import Path

from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.model.embedder import Embedder
from core.src.model.rag_context_manager import RagContextManager
from core.src.utils import load_prompt

_PROMPT_FILE = Path(__file__).parent / "prompts" / "System_Prompt.txt"


class WhatsAppAgent(BaseAgent):
    """
    Agent optimised for WhatsApp interactions.

    Produces short, mobile-friendly replies suited to WhatsApp's message format:
    - No markdown tables or heavy formatting.
    - Bullet points kept brief.
    - Answers property questions with RAG context when enabled.
    - Falls back to NullRagContextManager when RAG is disabled.
    """

    def get_system_prompt(self) -> str:
        return load_prompt(_PROMPT_FILE)

    def get_rag_context_manager(self):
        if self._rag_enabled:
            return RagContextManager(Embedder())
        return NullRagContextManager()

    def get_mcp_tools(self) -> list:
        return []
