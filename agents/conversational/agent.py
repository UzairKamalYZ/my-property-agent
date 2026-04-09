from pathlib import Path

from core.src.base_agent import BaseAgent
from core.src.model.llm_model_graph import LlmModelGraph

_PROMPT_FILE = Path(__file__).parent / "prompts" / "System_Prompt.txt"


class ConversationalAgent(BaseAgent):
    """
    Lightweight agent for greetings and general small talk.

    - No RAG: rag_enabled=False (default from BaseAgent) — no vector search.
    - No MCP tools: returns [] so no finance/currency tools are bound to the LLM.
    """

    def get_system_prompt(self) -> str:
        return LlmModelGraph._load_file(str(_PROMPT_FILE))

    def get_mcp_tools(self) -> list:
        return []
