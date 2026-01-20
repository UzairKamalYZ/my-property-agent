from model.llm_model import LlmModel
from config import Config
from scraping.web_scraper import WebScraper
from scraping.url_processor import UrlProcessor
import uuid
from scraping.embedder import Embedder
class LocalAgent:
    """Agent that uses a local language model and keeps conversation memory."""

    def __init__(self): 
        self.model = LlmModel()
        self.web_scraper = WebScraper()
        self.embedder = Embedder()
        self.session_id = str(uuid.uuid4())

    def ask(self, prompt: str, listings: list[dict], stream=False):
        """Send prompt to model and remember conversation."""
        return self.model.chat(prompt, self.session_id, listings, stream=stream)

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

        url_processor = UrlProcessor(agent.web_scraper)
        listings = url_processor.process_urls_from_file(Config.URLS_FILE)
        print(f"Total listings scraped: {len(listings)}")
        vectorstore = agent.embedder.embed(listings)
        top_listings = agent.embedder.rank_listings( "Apartments or House in Belgium under 1500 EUR", k=5)
        print("Top listings based on embedding search:")
        for i, listing in enumerate(top_listings, 1):
            print(f"{i}. {listing}")
            
        try:
            while True:
                q = input("You: ")
                if q.lower() in ["exit", "quit"]:
                    break
                print("AI:", end="", flush=True)
                for chunk in agent.ask(q, listings, stream=True):
                    print(chunk, end="", flush=True)
                print()
        except KeyboardInterrupt:
            print("\n🤖 Agent session ended.")