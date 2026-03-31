import logging
import uuid
import json
from rich.console import Console

from agentP.src.config.config import Config
from agentP.src.model.llm_factory import create_llm
from agentP.src.model.llm_model import LlmModel
from agentP.src.scraping.web_scraper import WebScraper

logger = logging.getLogger(__name__)


class LocalAgent:
    """Agent that uses a local language model and keeps conversation memory."""

    def __init__(self):
        logger.info("LocalAgent initializing (provider=%s, model=%s)",
                    Config.LLM_PROVIDER, Config.LLM_MODEL_NAME)
        llm = create_llm(
            provider=Config.LLM_PROVIDER,
            model_name=Config.LLM_MODEL_NAME
        )
        self.model = LlmModel(llm)
        self.web_scraper = WebScraper()
        self.session_id = str(uuid.uuid4())
        with open(Config.INTERACTION_FILE, "r") as f:
            self.interaction_texts = json.load(f)
        logger.info("LocalAgent ready (session_id=%s)", self.session_id)

    def ask(self, prompt: str, stream=False):
        """
        Asks the model a question using the full RAG and reformulation pipeline.
        """
        logger.info("ask: session_id=%s stream=%s prompt_len=%d",
                    self.session_id, stream, len(prompt))
        logger.debug("ask: prompt=%r", prompt)
        with open(Config.PROMPT_FILE, "r") as f:
            system_prompt = f.read()
        response = self.model.ask(system_prompt, prompt, self.session_id, stream=stream)
        if not stream:
            logger.info("ask: response ready session_id=%s", self.session_id)
        else:
            logger.debug("ask: streaming response for session_id=%s", self.session_id)
        return response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the agent and release resources."""
        logger.debug("LocalAgent closing session_id=%s", self.session_id)
        self.model.close()


if __name__ == "__main__":
    from agentP.src.config.logging_config import setup_logging
    setup_logging()

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
                    response = ""
                    for chunk in agent.ask(q, stream=True):
                        response += chunk
                    console.print(response)

        except KeyboardInterrupt:
            console.print(agent.interaction_texts["goodbye_message"])
        finally:
            console.print(agent.interaction_texts["session_ended_message"])
