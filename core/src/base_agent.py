import logging
import uuid

from .config.config import Config
from .model.embedder import Embedder
from .model.llm_factory import create_llm
from .model.llm_model_graph import LlmModelGraph
from .model.rag_context_manager import RagContextManager

logger = logging.getLogger(__name__)


class _NullRagContextManager:
    """No-op RAG context manager — returns empty string, skips all vector search."""

    def get_context(self, user_prompt: str, k: int = 5) -> str:
        return ""


class BaseAgent:
    """
    Shared base for all domain agents.

    Owns LLM creation, LangGraph pipeline wiring, session identity, and the
    ask / stream / close interface.  Subclasses customise behaviour by
    overriding hook methods:

        get_system_prompt()       — return the fully-loaded prompt string
        get_rag_context_manager() — controlled by rag_enabled; override to customise
        get_mcp_tools()           — return [] to disable, None for all, or a filtered list

    RAG is disabled by default.  Pass rag_enabled=True (or set "rag": true in
    agents.json) to activate vector-store retrieval for an agent.

    Note: hooks are called inside __init__ via Python's MRO.  Do not read
    subclass instance attributes inside a hook unless they were set before
    calling super().__init__().
    """

    def __init__(self, session_id: str = None, rag_enabled: bool = False):
        self.session_id = session_id or str(uuid.uuid4())
        self._rag_enabled = rag_enabled
        system_prompt = self.get_system_prompt()
        rag_context_manager = self.get_rag_context_manager()
        mcp_tools = self.get_mcp_tools()
        llm = create_llm(Config.LLM_PROVIDER, Config.LLM_MODEL_NAME)
        self.model = LlmModelGraph(
            llm,
            system_prompt=system_prompt,
            rag_context_manager=rag_context_manager,
            tools=mcp_tools,
        )
        logger.debug("%s initialised (session=%s)", self.__class__.__name__, self.session_id)

    # ------------------------------------------------------------------
    # Hook points — override in subclasses
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        return LlmModelGraph._load_file(Config.PROMPT_FILE)

    def get_rag_context_manager(self):
        # RAG is opt-in: only enabled when rag_enabled=True is passed at construction.
        # Controlled via the "rag" flag in orchestrator/agents.json.
        if self._rag_enabled:
            return RagContextManager(Embedder())
        return _NullRagContextManager()

    def get_mcp_tools(self) -> list | None:
        """
        Return the MCP tools to bind to this agent's LLM.

        Default: None — LlmModelGraph will load the full tool list from mcp.json.
        Override and return [] to disable all tool use for an agent (e.g. conversational).
        Override and return a filtered list to restrict which tools an agent may call.
        """
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask(self, prompt: str, stream: bool = False, session_id: str = None):
        sid = session_id or self.session_id
        if stream:
            return self.model.ask_stream(prompt, session_id=sid)
        return self.model.ask(prompt, session_id=sid)

    def close(self) -> None:
        self.model.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
