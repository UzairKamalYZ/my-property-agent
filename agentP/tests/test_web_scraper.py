import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
import requests
from src.scraping.web_scraper import WebScraper

@pytest.fixture
def web_scraper():
    return WebScraper()

def test_web_scraper_scrape_success(web_scraper):
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'<html><head><title>Test</title></head><body><p>Hello</p></body></html>'
        mock_get.return_value = mock_response

        content = web_scraper.scrape('http://example.com')
        assert content == 'Test\nHello'

def test_web_scraper_scrape_failure(web_scraper):
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException('Test error')
        content = web_scraper.scrape('http://example.com')
        assert content == ''

