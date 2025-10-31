import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import json
from unittest.mock import patch, MagicMock
from src.agent import LocalAgent

@pytest.fixture(autouse=True)
def clear_memory_file():
    if os.path.exists("memory.json"):
        os.remove("memory.json")
    if os.path.exists("scraped_content.json"):
        os.remove("scraped_content.json")

def test_agent_with_memory():
    agent = LocalAgent()
    with patch('src.model.llm_model.LlmModel.chat') as mock_chat:
        mock_chat.return_value = "Hello there!"
        response1 = agent.ask("Hello!")
        assert response1 == "Hello there!"

        mock_chat.return_value = "I said hello."
        response2 = agent.ask("What did I just say?")
        assert response2 == "I said hello."
        assert len(agent.memory) == 5 # system, user, assistant, user, assistant

def test_agent_context_manager():
    with patch('src.model.llm_model.LlmModel.close') as mock_close:
        with LocalAgent() as agent:
            pass
        mock_close.assert_called_once()

def test_agent_ask_stream():
    agent = LocalAgent()
    with patch('src.model.llm_model.LlmModel.chat') as mock_chat:
        mock_chat.return_value = iter(["Hello", " there!"])
        response_generator = agent.ask("Hello", stream=True)
        response = "".join(response_generator)
        assert response == "Hello there!"
        assert agent.memory[-1]["content"] == "Hello there!"

def test_agent_ask_with_url():
    agent = LocalAgent()
    with patch('src.scraping.web_scraper.WebScraper.scrape') as mock_scrape:
        with patch('src.model.llm_model.LlmModel.chat') as mock_chat:
            mock_scrape.return_value = "Scraped content"
            mock_chat.return_value = "This is about the scraped content."

            response = agent.ask("http://example.com")

            mock_scrape.assert_not_called()
            assert "system" in [m["role"] for m in agent.memory]
            assert response == "This is about the scraped content."