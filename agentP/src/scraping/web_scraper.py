import requests
from bs4 import BeautifulSoup


class WebScraper:
    """Scrapes text content from a given URL."""

    def scrape(self, url: str) -> str:
        """Fetch and parse the text content of a URL."""
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            body_html = self.extract_body_content(soup)
            cleaned_content = self.clean_body_content(body_html)
            split_content = self.split_dom_content(cleaned_content)

            return "\n".join(split_content)


        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return ""

    def extract_body_content(self, soup: BeautifulSoup) -> str:
        body = soup.body
        return str(body) if body else ""

    def clean_body_content(self, body_content: str) -> str:
        soup = BeautifulSoup(body_content, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        cleaned = soup.get_text(separator="\n")
        cleaned = "\n".join(
            line.strip() for line in cleaned.splitlines() if line.strip()
        )

        return cleaned

    def split_dom_content(self, dom_content: str, max_length: int = 6000) -> list[str]:
        return [
            dom_content[i : i + max_length]
            for i in range(0, len(dom_content), max_length)
        ]
