import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
from src.agent import LocalAgent

@pytest.fixture
def agent():
    return LocalAgent()

def test_agent_ask_no_stream(agent):
    with patch.object(agent.model, 'chat') as mock_chat:
        mock_chat.return_value = "Hello there!"
        response = agent.ask("Hello!")
        assert response == "Hello there!"
        mock_chat.assert_called_once_with("Hello!", agent.session_id, stream=False)

def test_agent_ask_stream(agent):
    with patch.object(agent.model, 'chat') as mock_chat:
        mock_chat.return_value = iter(["Hello", " there!"])
        response_generator = agent.ask("Hello", stream=True)
        response = "".join(response_generator)
        assert response == "Hello there!"
        mock_chat.assert_called_once_with("Hello", agent.session_id, stream=True)

def test_agent_context_manager():
    with patch('src.model.llm_model.LlmModel.close') as mock_close:
        with LocalAgent() as agent:
            pass
        mock_close.assert_called_once()
