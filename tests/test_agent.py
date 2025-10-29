import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import json
from unittest.mock import patch, MagicMock
from src.agent import LocalQwenAgent

@pytest.fixture
def temp_memory_file(tmp_path):
    return tmp_path / "memory.json"

@pytest.fixture
def agent(temp_memory_file):
    return LocalQwenAgent(memory_file=temp_memory_file)

def test_agent_with_memory(agent):
    with patch('src.model.qwen_model.QwenModel.chat') as mock_chat:
        mock_chat.return_value = "Hello there!"
        response1 = agent.ask("Hello!")
        assert response1 == "Hello there!"

        mock_chat.return_value = "I said hello."
        response2 = agent.ask("What did I just say?")
        assert response2 == "I said hello."
        assert len(agent.memory) == 5 # system, user, assistant, user, assistant

def test_agent_memory_persistence(temp_memory_file):
    with patch('src.model.qwen_model.QwenModel.chat') as mock_chat:
        mock_chat.return_value = "Response"
        agent1 = LocalQwenAgent(memory_file=temp_memory_file)
        agent1.ask("Hello")
        del agent1

        agent2 = LocalQwenAgent(memory_file=temp_memory_file)
        assert len(agent2.memory) == 3 # system, user, assistant

def test_agent_context_manager():
    with patch('src.model.qwen_model.QwenModel.close') as mock_close:
        with LocalQwenAgent() as agent:
            pass
        mock_close.assert_called_once()

def test_agent_ask_stream(agent):
    with patch('src.model.qwen_model.QwenModel.chat') as mock_chat:
        mock_chat.return_value = iter(["Hello", " there!"])
        response_generator = agent.ask("Hello", stream=True)
        response = "".join(response_generator)
        assert response == "Hello there!"
        assert agent.memory[-1]["content"] == "Hello there!"