import requests
from bs4 import BeautifulSoup

class WebScraper:
    """Scrapes text content from a given URL."""

    def scrape(self, url: str) -> str:
        """Fetch and parse the text content of a URL."""
        try:
            response = requests.get(url)
            print(f"scraping----> {url}")
            response.raise_for_status()  # Raise an exception for bad status codes
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {e}")
            return ""
