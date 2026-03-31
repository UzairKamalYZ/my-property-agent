import logging
import os
from .web_scraper import WebScraper
import re

logger = logging.getLogger(__name__)


class UrlProcessor:
    """Scrapes URLs and extracts property listings."""

    def __init__(self, web_scraper: WebScraper):
        self.web_scraper = web_scraper

    def process_urls_from_file(self, file_path: str) -> list[dict]:
        listings = []

        if not os.path.exists(file_path):
            logger.warning("URLs file not found: %s", file_path)
            return listings

        with open(file_path, "r") as f:
            urls = [u.strip() for u in f.readlines() if u.strip()]

        logger.info("Processing %d URL(s) from %s", len(urls), file_path)
        for url in urls:
            logger.info("Scraping: %s", url)
            content = self.web_scraper.scrape(url)

            if not content:
                logger.warning("No content scraped from %s", url)
                continue

            extracted = self.extract_listings(content, url)
            listings.extend(extracted)
            logger.info("Extracted %d listing(s) from %s", len(extracted), url)

        logger.info("URL processing complete: %d total listing(s)", len(listings))
        return listings

    def extract_listings(self, content: str, source_url: str) -> list[dict]:
        listings = []
        blocks = re.split(r"\n\s*\n", content)

        for block in blocks:
            if not re.search(r"€|\beur\b", block.lower()):
                continue

            rent = self._extract_rent(block)
            if rent is None:
                continue

            listing = {
                "type": self._extract_property_type(block),
                "city": self._extract_city(block),
                "rent": rent,
                "description": self._extract_description(block),
                "url": source_url,
            }

            listings.append(listing)

        logger.debug("extract_listings: found %d listing(s) in content of %d block(s)",
                     len(listings), len(blocks))
        return listings

    def _extract_rent(self, text: str) -> int | None:
        match = re.search(r"€\s?(\d{3,5})", text)
        if match:
            return int(match.group(1))
        return None

    def _extract_property_type(self, text: str) -> str:
        text = text.lower()
        if "house" in text or "maison" in text:
            return "House"
        if "apartment" in text or "appartement" in text or "flat" in text:
            return "Apartment"
        return "Apartment"

    def _extract_city(self, text: str) -> str:
        # Extend this list as needed
        cities = [
            "brussels", "antwerp", "ghent", "bruges",
            "leuven", "liège", "namur"
        ]

        text_lower = text.lower()
        for city in cities:
            if city in text_lower:
                return city.capitalize()

        return "Unknown"

    def _extract_description(self, text: str) -> str:
        # Keep it short and clean
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned[:160]
