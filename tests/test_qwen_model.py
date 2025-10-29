import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
from src.model.qwen_model import QwenModel

@pytest.fixture
def qwen_model():
    return QwenModel()

def test_qwen_model_chat_no_stream(qwen_model):
    with patch('ollama.chat') as mock_chat:
        mock_chat.return_value = {"message": {"content": "Hello there!"}}
        response = qwen_model.chat([{"role": "user", "content": "Hello"}])
        assert response == "Hello there!"
        mock_chat.assert_called_once_with(
            model=qwen_model.model,
            messages=[{"role": "user", "content": "Hello"}],
            stream=False
        )

def test_qwen_model_chat_stream(qwen_model):
    with patch('ollama.chat') as mock_chat:
        mock_response = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " there!"}}
        ]
        mock_chat.return_value = iter(mock_response)
        response_generator = qwen_model.chat([{"role": "user", "content": "Hello"}], stream=True)
        response = "".join(response_generator)
        assert response == "Hello there!"
        mock_chat.assert_called_once_with(
            model=qwen_model.model,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

def test_qwen_model_chat_error(qwen_model):
    with patch('ollama.chat') as mock_chat:
        mock_chat.side_effect = Exception("Test error")
        response = qwen_model.chat([{"role": "user", "content": "Hello"}])
        assert response == ""
