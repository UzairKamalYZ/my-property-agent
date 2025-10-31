import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
from src.model.llm_model import LlmModel

@pytest.fixture
def llm_model():
    return LlmModel()

def test_llm_model_chat_no_stream(llm_model):
    with patch('ollama.chat') as mock_chat:
        mock_chat.return_value = {"message": {"content": "Hello there!"}}
        response = llm_model.chat([{"role": "user", "content": "Hello"}])
        assert response == "Hello there!"
        mock_chat.assert_called_once_with(
            model=llm_model.model,
            messages=[{"role": "user", "content": "Hello"}],
            stream=False
        )

def test_llm_model_chat_stream(llm_model):
    with patch('ollama.chat') as mock_chat:
        mock_response = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " there!"}}
        ]
        mock_chat.return_value = iter(mock_response)
        response_generator = llm_model.chat([{"role": "user", "content": "Hello"}], stream=True)
        response = "".join(response_generator)
        assert response == "Hello there!"
        mock_chat.assert_called_once_with(
            model=llm_model.model,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

def test_llm_model_chat_error(llm_model):
    with patch('ollama.chat') as mock_chat:
        mock_chat.side_effect = Exception("Test error")
        response = llm_model.chat([{"role": "user", "content": "Hello"}])
        assert response == ""
