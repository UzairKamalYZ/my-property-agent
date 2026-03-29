import uuid
import json
from rich.console import Console

from config.config import Config
from model.llm_factory import create_llm
from model.llm_model import LlmModel
from scraping.web_scraper import WebScraper


class LocalAgent:
    """Agent that uses a LangGraph-based local language model with memory."""

    def __init__(self):
        print(">>>>>> LocalAgent.__init__ called <<<<<<")
        llm = create_llm(
            provider=Config.LLM_PROVIDER,
            model_name=Config.LLM_MODEL_NAME
        )
        self.model = LlmModel(llm)
        self.web_scraper = WebScraper()
        self.session_id = str(uuid.uuid4())
        with open(Config.INTERACTION_FILE, "r") as f:
            self.interaction_texts = json.load(f)

    def ask(self, prompt: str, stream=False):
        """
        Asks the model a question using the full RAG and reformulation pipeline.
        """
        with open(Config.PROMPT_FILE, "r") as f:
            system_prompt = f.read()
        return self.model.ask(system_prompt, prompt, self.session_id, stream=stream)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Cleanup resources."""
        self.model.close()


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