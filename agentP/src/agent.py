import uuid
import json
from rich.console import Console
from rich.spinner import Spinner

from agentP.src.config.config import Config
from agentP.src.model.llm_factory import create_llm
from agentP.src.model.llm_model import LlmModel
from agentP.src.scraping.web_scraper import WebScraper

class LocalAgent:
    """Agent that uses a local language model and keeps conversation memory."""

    def __init__(self):
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
        return self.model.ask_with_reformulation(system_prompt, prompt, self.session_id, stream=stream)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the agent and release resources."""
        self.model.close()

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
                    response = ""
                    for chunk in agent.ask(q, stream=True):
                        response += chunk
                    console.print(response)

        except KeyboardInterrupt:
            console.print(agent.interaction_texts["goodbye_message"])
        finally:
            console.print(agent.interaction_texts["session_ended_message"])