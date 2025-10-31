import os
import json
from .web_scraper import WebScraper

class UrlProcessor:
    """Handles processing of URLs from a file."""

    def __init__(self, web_scraper: WebScraper, memory: list):
        self.web_scraper = web_scraper
        self.memory = memory

    def process_urls_from_file(self, file_path: str, cache_file: str = 'scraped_content.json'):
        """Reads URLs from a file and scrapes them, using a cache."""
        if os.path.exists(cache_file):
            print(f"Loading scraped content from {cache_file}...")
            with open(cache_file, 'r') as f:
                scraped_data = json.load(f)
            self.memory.extend(scraped_data)
            print("Successfully loaded scraped content.")
            return

        print("No cache found. Starting scraping process...")
        scraped_data = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                urls = f.readlines()
            for url in urls:
                url = url.strip()
                if url:
                    print(f"Scraping {url}...")
                    content = self.web_scraper.scrape(url)
                    if content:
                        # Storing scraped content in memory for context
                        data = {"role": "system", "content": f"Scraped content from {url}:\n{content}"}
                        scraped_data.append(data)
                        self.memory.append(data)
                        print(f"Successfully scraped and processed {url}")
                    else:
                        print(f"Failed to scrape {url}")
        
        with open(cache_file, 'w') as f:
            json.dump(scraped_data, f, indent=4)
        print(f"Scraped content cached to {cache_file}")
