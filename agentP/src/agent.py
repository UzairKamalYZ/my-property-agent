import uuid

from model.llm_model import LlmModel
from scraping.web_scraper import WebScraper

class LocalAgent:
    """Agent that uses a local language model and keeps conversation memory."""

    def __init__(self):
        self.model = LlmModel()
        self.web_scraper = WebScraper()
        self.session_id = str(uuid.uuid4())

    def ask(self, prompt: str, stream=False):
        """Send prompt to model and remember conversation."""
        return self.model.chat_with_context(prompt, self.session_id, stream=stream)

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
