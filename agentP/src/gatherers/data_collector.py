import uuid

from core.src.model.embedder import Embedder
from core.src.config.config import Config
from ..scraping.url_processor import UrlProcessor
from ..scraping.web_scraper import WebScraper


class DataCollector:

    def __init__(self):
        self.data = None
        self.web_scraper = WebScraper()
        self.embedder = Embedder()
        self.session_id = str(uuid.uuid4())
        self.url_processor = UrlProcessor(self.web_scraper)

    def get_data_from_urls(self, file_path: str):
        self.data = self.url_processor.process_urls_from_file(file_path)
        return self

    def store_vector_embeddings(self):
        self.embedder.embed_documents_to_vectors(self.data)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the collector and release resources."""
        pass


if __name__ == "__main__":
    print("🤖 Lets Collect data and build index for it.!")
    with DataCollector() as collector:
        cf = Config()
        collector.get_data_from_urls(cf.URLS_FILE).store_vector_embeddings()
