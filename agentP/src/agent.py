import json
from rich.console import Console

from config.config import Config
from model.llm_factory import create_llm
from model.llm_model_graph import LlmModelGraph
from scraping.web_scraper import WebScraper


class LocalAgent:
    """Agent that uses a LangGraph-based local language model with memory."""

    def __init__(self):
        print(">>>>>> LocalAgent.__init__ called <<<<<<")
        llm = create_llm(
            provider=Config.LLM_PROVIDER,
            model_name=Config.LLM_MODEL_NAME
        )
        self.model = LlmModelGraph(llm)
        self.web_scraper = WebScraper()
        with open(Config.INTERACTION_FILE, "r") as f:
            self.interaction_texts = json.load(f)

    def ask(self, prompt: str, stream=False):
        """
        Asks the model a question using the full RAG and reformulation pipeline.
        """
        if stream:
            return self.model.ask_stream(prompt)
        return self.model.ask(prompt)

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