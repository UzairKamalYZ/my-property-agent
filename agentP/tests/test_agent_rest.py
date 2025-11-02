import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from src.agentRest import app, lifespan
from unittest.mock import patch, MagicMock
from contextlib import asynccontextmanager

# Prevent the lifespan from running during most tests
@asynccontextmanager
async def mock_lifespan(app):
    yield

app.router.lifespan_context = mock_lifespan

@pytest.fixture
def client():
    with patch('src.agent.LocalAgent') as mock_agent:
        mock_agent_instance = mock_agent.return_value
        mock_agent_instance.ask.return_value = "Test response"
        app.state.agent = mock_agent_instance
        
        with TestClient(app) as test_client:
            yield test_client

def test_ask_no_stream(client):
    response = client.get("/ask?prompt=Test%20prompt")
    assert response.status_code == 200
    assert response.json() == {"response": "Test response"}

def test_ask_stream(client):
    # Mock the streaming response on the mocked agent instance
    client.app.state.agent.ask.return_value = iter(["Test ", "response"])

    response = client.get("/ask?prompt=Test%20prompt&stream=True")
    assert response.status_code == 200
    assert response.text == "data: Test \n\ndata: response\n\n"

def test_lifespan():
    with patch('src.agentRest.LocalAgent') as mock_agent:
        with patch('src.scraping.url_processor.UrlProcessor.process_urls_from_file') as mock_process_urls:
            app.router.lifespan_context = lifespan
            with TestClient(app) as test_client:
                pass
            mock_agent.assert_called_once()
            mock_process_urls.assert_called_once()
            mock_agent.return_value.close.assert_called_once()
