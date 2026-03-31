import logging
import uuid

from model.embedder import Embedder
from scraping.url_processor import UrlProcessor
from scraping.web_scraper import WebScraper
import config

logger = logging.getLogger(__name__)


class DataCollector:

    def __init__(self):
        self.data = None
        self.web_scraper = WebScraper()
        self.embedder = Embedder()
        self.session_id = str(uuid.uuid4())
        self.url_processor = UrlProcessor(self.web_scraper)
        logger.debug("DataCollector initialized session_id=%s", self.session_id)

    def __getDataFromUrls__(self, file_path: str):
        logger.info("DataCollector: fetching listings from %s", file_path)
        self.data = self.url_processor.process_urls_from_file(file_path)
        logger.info("DataCollector: retrieved %d listing(s)", len(self.data))
        return self

    def __storVectorEmbeddings__(self):
        logger.info("DataCollector: embedding and storing %d document(s)",
                    len(self.data) if self.data else 0)
        self.embedder.embed_documents_to_vectors(self.data)
        logger.debug("DataCollector: embeddings stored")
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the collector and release resources."""
        logger.debug("DataCollector closed")


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging()

    logger.info("DataCollector starting — building index from URLs")
    with DataCollector() as collector:
        cf = config.Config()
        collector.__getDataFromUrls__(cf.URLS_FILE).__storVectorEmbeddings__()
    logger.info("DataCollector finished")
