import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
from src.scraping.url_processor import UrlProcessor

@pytest.fixture
def url_processor():
    web_scraper = MagicMock()
    memory = MagicMock()
    return UrlProcessor(web_scraper, memory)

def test_process_urls_from_file_no_file(url_processor):
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False
        url_processor.process_urls_from_file('non_existent_file.txt')
        url_processor.web_scraper.scrape.assert_not_called()

def test_process_urls_from_file_empty_file(url_processor):
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.readlines.return_value = []
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            url_processor.process_urls_from_file('empty_file.txt')
            url_processor.web_scraper.scrape.assert_not_called()

def test_process_urls_from_file_empty_lines(url_processor):
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.readlines.return_value = ['\n', '\n']
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            url_processor.process_urls_from_file('empty_lines.txt')
            url_processor.web_scraper.scrape.assert_not_called()

def test_process_urls_from_file_scrape_fails(url_processor):
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.readlines.return_value = ['http://example.com']
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            url_processor.web_scraper.scrape.return_value = ''
            url_processor.process_urls_from_file('scrape_fails.txt')
            url_processor.memory.add_message.assert_not_called()

def test_process_urls_from_file_success(url_processor):
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.readlines.return_value = ['http://example.com']
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            url_processor.web_scraper.scrape.return_value = 'Test content'
            url_processor.process_urls_from_file('success.txt')
            url_processor.memory.add_message.assert_called_once()
