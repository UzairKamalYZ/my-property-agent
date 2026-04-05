import logging
import uuid

from .config.config import Config
from .model.embedder import Embedder
from .model.llm_factory import create_llm
from .model.llm_model_graph import LlmModelGraph
from .model.rag_context_manager import RagContextManager

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Shared base for all domain agents.

    Owns LLM creation, LangGraph pipeline wiring, session identity, and the
    ask / stream / close interface.  Subclasses customise behaviour by
    overriding the two hook methods:

        get_system_prompt()       — return the fully-loaded prompt string
        get_rag_context_manager() — return the RagContextManager to inject

    Both hooks have working defaults so a subclass that is happy with the
    property-search defaults need not override anything.

    Note: hooks are called inside __init__ via Python's MRO.  Do not read
    subclass instance attributes inside a hook unless they were set before
    calling super().__init__().
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        system_prompt = self.get_system_prompt()
        rag_context_manager = self.get_rag_context_manager()
        llm = create_llm(Config.LLM_PROVIDER, Config.LLM_MODEL_NAME)
        self.model = LlmModelGraph(llm, system_prompt=system_prompt, rag_context_manager=rag_context_manager)
        logger.debug("%s initialised (session=%s)", self.__class__.__name__, self.session_id)

    # ------------------------------------------------------------------
    # Hook points — override in subclasses
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        return LlmModelGraph._load_file(Config.PROMPT_FILE)

    def get_rag_context_manager(self) -> RagContextManager:
        return RagContextManager(Embedder())

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
