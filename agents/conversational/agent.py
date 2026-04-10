from pathlib import Path

from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.utils import load_prompt

_PROMPT_FILE = Path(__file__).parent / "prompts" / "System_Prompt.txt"


class ConversationalAgent(BaseAgent):
    """
    Lightweight agent for greetings and general small talk.

    - No RAG: no vector search.
    - No MCP tools: no finance/currency tools bound to the LLM.
    """

    def get_system_prompt(self) -> str:
        return load_prompt(_PROMPT_FILE)

    def get_rag_context_manager(self):
        return NullRagContextManager()

    def get_mcp_tools(self) -> list:
        return []
