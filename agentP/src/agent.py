import uuid

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

    def ask(self, prompt: str, stream=False):
        """
        Asks the model a question using the full RAG and reformulation pipeline.
        """
        return self.model.ask_with_reformulation(prompt, self.session_id, stream=stream)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the agent and release resources."""
        self.model.close()


if __name__ == "__main__":
    print("🤖 Local Agent is ready!")
    with LocalAgent() as agent:
        try:
            while True:
                q = input("You: ")
                if q.lower() in ["exit", "quit"]:
                    break

                print("AI:", end="", flush=True)
                for chunk in agent.ask(q, stream=True):
                    print(chunk, end="", flush=True)
                print()
        except KeyboardInterrupt:
            print("\n🤖 Agent session ended.")