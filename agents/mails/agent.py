from pathlib import Path

from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.utils import load_prompt

_PROMPT_FILE = Path(__file__).parent / "prompts" / "System_Prompt.txt"


class MailAgent(BaseAgent):
    """
    Email-topic agent for the orchestrator.

    Answers conversational questions about emails (e.g. "have I received
    anything from X?", "draft a reply to the invoice email").  No RAG or
    MCP tools — the LLM works solely from the context provided in the prompt
    or the conversation history.
    """

    def get_system_prompt(self) -> str:
        return load_prompt(_PROMPT_FILE)

    def get_rag_context_manager(self):
        return NullRagContextManager()

    def get_mcp_tools(self) -> list:
        return []
