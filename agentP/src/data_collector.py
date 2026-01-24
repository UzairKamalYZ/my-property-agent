import uuid

from model.embedder import Embedder
from scraping.url_processor import UrlProcessor
from scraping.web_scraper import WebScraper
import config

class DataCollector:

    def __init__(self):
        self.web_scraper = WebScraper()
        self.embedder = Embedder()
        self.session_id = str(uuid.uuid4())
        self.url_processor = UrlProcessor(self.web_scraper)

    def getDataFromUrls(self, file_path: str) -> list[dict]:
        return self.url_processor.process_urls_from_file(file_path)

    def getVectorEmbbedings(self,listings: dict) :
        embed = self.embedder.embed(listings)
        return embed
    def getRankedListings(self) :
        rank_listings = self.embedder.rank_listings(
            "Apartments or House in Belgium under 1500 EUR", k=5)
        return rank_listings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the agent and release resources."""
        ## will close db here


if __name__ == "__main__":
    print("🤖 Local Agent is ready!")
    with DataCollector() as collector:
        cf = config.Config()

        data_from_urls = collector.getDataFromUrls(cf.URLS_FILE)
        embeddings = collector.getVectorEmbbedings(data_from_urls)

        top_listings = collector.getRankedListings()
        for i, listing in enumerate(top_listings, 1):
            print(f"{i}. {listing}")

