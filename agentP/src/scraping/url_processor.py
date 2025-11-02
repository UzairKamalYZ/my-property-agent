import os
from .web_scraper import WebScraper
from langchain_core.messages import SystemMessage

class UrlProcessor:
    """Handles processing of URLs from a file."""

    def __init__(self, web_scraper: WebScraper, memory):
        self.web_scraper = web_scraper
        self.memory = memory

    def process_urls_from_file(self, file_path: str):
        """Reads URLs from a file and scrapes them."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                urls = f.readlines()
            for url in urls:
                url = url.strip()
                if url:
                    print(f"Scraping {url}...")
                    content = self.web_scraper.scrape(url)
                    if content:
                        message = SystemMessage(content=f"Scraped content from {url}:\n{content}")
                        self.memory.add_message(message)
                        print(f"Successfully scraped and processed {url}")
                    else:
                        print(f"Failed to scrape {url}")