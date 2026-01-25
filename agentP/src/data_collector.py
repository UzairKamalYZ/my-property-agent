import uuid

from model.embedder import Embedder
from scraping.url_processor import UrlProcessor
from scraping.web_scraper import WebScraper
import config


class DataCollector:

    def __init__(self):
        self.data = None
        self.web_scraper = WebScraper()
        self.embedder = Embedder()
        self.session_id = str(uuid.uuid4())
        self.url_processor = UrlProcessor(self.web_scraper)

    def __getDataFromUrls__(self, file_path: str):
        self.data = self.url_processor.process_urls_from_file(file_path)
        return self

    def __getVectorEmbeddings__(self):
        self.embedder.embed(self.data)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the agent and release resources."""
        ## will close db here


if __name__ == "__main__":
    print("🤖 Lets Collect data and build index for it.!")
    with (DataCollector() as collector):
        cf = config.Config()
        embeddings = collector.__getDataFromUrls__(cf.URLS_FILE).__getVectorEmbeddings__()

        top_listings = collector.__getRankedListings__()
        for i, listing in enumerate(top_listings, 1):
            print(f"{i}. {listing}")
