from src.model.llm_model import LlmModel
from src.config import Config
from src.memory_manager import MemoryManager
from src.scraping.web_scraper import WebScraper
from src.scraping.url_processor import UrlProcessor
import os,json
import re

class LocalAgent:
    """Agent that uses a local language model and keeps conversation memory."""

    def __init__(self, memory_file=Config.MEMORY_FILE):
        self.model = LlmModel()
        self.memory_manager = MemoryManager(memory_file)
        self.web_scraper = WebScraper()
        self.memory = self.memory_manager.load_memory()

    def ask(self, prompt: str, stream=False):
        """Send prompt to model and remember conversation."""
        self.memory.append({"role": "user", "content": prompt})
        if stream:
            return self._ask_stream(prompt)
        else:
            reply = self.model.chat(self.memory)
            self.memory.append({"role": "assistant", "content": reply})
            self.memory_manager.save_memory(self.memory)
            return reply

    def _ask_stream(self, prompt):
        """Handle streaming responses for the ask method."""
        full_reply = ""
        for chunk in self.model.chat(self.memory, stream=True):
            full_reply += chunk
            yield chunk
        self.memory.append({"role": "assistant", "content": full_reply})
        self.memory_manager.save_memory(self.memory)

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
        url_processor = UrlProcessor(agent.web_scraper, agent.memory)
        url_processor.process_urls_from_file(Config.URLS_FILE)
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