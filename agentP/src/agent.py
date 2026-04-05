import json
import logging

from rich.console import Console

from core.src.base_agent import BaseAgent
from core.src.config.config import Config
from .scraping.web_scraper import WebScraper

logger = logging.getLogger(__name__)


class LocalAgent(BaseAgent):
    """Property-search agent.  Uses the default system prompt and RAG pipeline."""

    def __init__(self, session_id: str = None):
        super().__init__(session_id)
        self.web_scraper = WebScraper()
        with open(Config.INTERACTION_FILE, "r") as f:
            self.interaction_texts = json.load(f)


# ------------------- CLI -------------------

if __name__ == "__main__":
    console = Console()
    agent = LocalAgent()

    console.print(agent.interaction_texts["welcome_message"])
    console.print(agent.interaction_texts["welcome_prompt"])

    with agent:
        try:
            while True:
                q = input(agent.interaction_texts["input_prompt"])

                if q.lower() in ["exit", "quit"]:
                    break

                with console.status(agent.interaction_texts["thinking_message"]):
                    response = agent.ask(q)

                console.print(response)

        except KeyboardInterrupt:
            console.print(agent.interaction_texts["goodbye_message"])
        finally:
            console.print(agent.interaction_texts["session_ended_message"])
