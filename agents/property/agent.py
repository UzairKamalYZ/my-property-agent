import json
import logging

from rich.console import Console

from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.config.config import Config
from core.src.model.embedder import Embedder
from core.src.model.rag_context_manager import RagContextManager
from core.src.utils import load_prompt
from .scraping.web_scraper import WebScraper

logger = logging.getLogger(__name__)


class PropertySearchAgent(BaseAgent):
    """Property-search agent."""

    def get_system_prompt(self) -> str:
        return load_prompt(Config.PROMPT_FILE)

    def get_rag_context_manager(self):
        if self._rag_enabled:
            return RagContextManager(Embedder())
        return NullRagContextManager()

    def get_mcp_tools(self) -> list | None:
        return None

    def __init__(self, session_id: str = None, rag_enabled: bool = False):
        super().__init__(session_id, rag_enabled=rag_enabled)
        # self.web_scraper = WebScraper()
        # with open(Config.INTERACTION_FILE, "r") as f:
        #     self.interaction_texts = json.load(f)
