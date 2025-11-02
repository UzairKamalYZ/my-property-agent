import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
from src.model.llm_model import LlmModel

@pytest.fixture
def llm_model():
    return LlmModel()

def test_llm_model_chat_no_stream(llm_model):
    with patch('langchain_core.runnables.history.RunnableWithMessageHistory.invoke') as mock_invoke:
        mock_invoke.return_value = "Hello there!"
        response = llm_model.chat("Hello", "test_session")
        assert response == "Hello there!"
        mock_invoke.assert_called_once()

def test_llm_model_chat_stream(llm_model):
    with patch('langchain_core.runnables.history.RunnableWithMessageHistory.stream') as mock_stream:
        mock_stream.return_value = iter(["Hello", " there!"])
        response_generator = llm_model.chat("Hello", "test_session", stream=True)
        response = "".join(response_generator)
        assert response == "Hello there!"
        mock_stream.assert_called_once()

def test_llm_model_chat_error(llm_model):
    with patch('langchain_core.runnables.history.RunnableWithMessageHistory.invoke') as mock_invoke:
        mock_invoke.side_effect = Exception("Test error")
        with pytest.raises(Exception):
            llm_model.chat("Hello", "test_session")