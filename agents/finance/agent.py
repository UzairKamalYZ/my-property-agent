from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.model.embedder import Embedder
from core.src.model.mcp_tools import _mcp
from core.src.model.rag_context_manager import RagContextManager
from core.src.utils import load_prompt

from .config import FinanceConfig


class FinanceAgent(BaseAgent):
    """Stock analysis and investment advice agent — uses the stocks MCP server."""

    def get_system_prompt(self) -> str:
        return load_prompt(FinanceConfig.FINANCE_PROMPT_FILE)

    def get_rag_context_manager(self):
        if self._rag_enabled:
            return RagContextManager(Embedder())
        return NullRagContextManager()

    def get_mcp_tools(self) -> list:
        return _mcp.langchain_tools(server_name="stocks")
