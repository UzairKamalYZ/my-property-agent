"""
Tests for agent.LocalAgent on the claude/add-claude-documentation-XhFqO branch.

agent.py on this branch uses absolute imports:
  from agentP.src.config.config import Config
  from agentP.src.model.llm_factory import create_llm
  from agentP.src.model.llm_model import LlmModel
  from agentP.src.scraping.web_scraper import WebScraper

and ask() reads PROMPT_FILE then delegates to model.ask(system_prompt, prompt,
session_id, stream=stream).
"""
import json
import unittest
from unittest.mock import MagicMock, patch, mock_open

from agentP.src.agent import LocalAgent


_INTERACTION = {
    "welcome_message": "Welcome!",
    "welcome_prompt": "How can I help?",
    "input_prompt": "> ",
    "thinking_message": "Thinking...",
    "goodbye_message": "Bye!",
    "session_ended_message": "Session ended.",
}

_SYSTEM_PROMPT = "You are a helpful property agent."


def _make_agent():
    """Create a LocalAgent with all heavy deps patched."""
    mock_model = MagicMock()
    with patch("agentP.src.agent.create_llm", return_value=MagicMock()), \
         patch("agentP.src.agent.LlmModel", return_value=mock_model), \
         patch("agentP.src.agent.WebScraper"), \
         patch("agentP.src.agent.Config"), \
         patch(
             "builtins.open",
             mock_open(read_data=json.dumps(_INTERACTION)),
         ):
        agent = LocalAgent()
        agent.model = mock_model
        return agent, mock_model


class TestLocalAgentInit(unittest.TestCase):

    @patch("agentP.src.agent.WebScraper")
    @patch("agentP.src.agent.LlmModel")
    @patch("agentP.src.agent.create_llm")
    @patch("agentP.src.agent.Config")
    def test_should_create_llm_model_when_initialized(
        self, MockConfig, mock_create_llm, MockLlmModel, _
    ):
        mock_create_llm.return_value = MagicMock()
        with patch("builtins.open", mock_open(read_data=json.dumps(_INTERACTION))):
            LocalAgent()
        MockLlmModel.assert_called_once()

    @patch("agentP.src.agent.WebScraper")
    @patch("agentP.src.agent.LlmModel")
    @patch("agentP.src.agent.create_llm")
    @patch("agentP.src.agent.Config")
    def test_should_call_create_llm_with_config_values_when_initialized(
        self, MockConfig, mock_create_llm, MockLlmModel, _
    ):
        MockConfig.LLM_PROVIDER = "ollama"
        MockConfig.LLM_MODEL_NAME = "llama3.2"
        mock_create_llm.return_value = MagicMock()
        with patch("builtins.open", mock_open(read_data=json.dumps(_INTERACTION))):
            LocalAgent()
        mock_create_llm.assert_called_once_with(
            provider=MockConfig.LLM_PROVIDER,
            model_name=MockConfig.LLM_MODEL_NAME,
        )

    @patch("agentP.src.agent.WebScraper")
    @patch("agentP.src.agent.LlmModel")
    @patch("agentP.src.agent.create_llm")
    @patch("agentP.src.agent.Config")
    def test_should_load_interaction_texts_from_file_when_initialized(
        self, MockConfig, mock_create_llm, MockLlmModel, _
    ):
        mock_create_llm.return_value = MagicMock()
        interaction = dict(_INTERACTION, welcome_message="Howdy!")
        with patch("builtins.open", mock_open(read_data=json.dumps(interaction))):
            agent = LocalAgent()
        self.assertEqual(agent.interaction_texts["welcome_message"], "Howdy!")


class TestLocalAgentAsk(unittest.TestCase):

    def setUp(self):
        self.agent, self.mock_model = _make_agent()

    @patch("builtins.open", mock_open(read_data=_SYSTEM_PROMPT))
    def test_should_call_model_ask_when_stream_is_false(self):
        self.mock_model.ask.return_value = "Some answer"
        result = self.agent.ask("find me a flat", stream=False)
        self.mock_model.ask.assert_called_once()
        self.assertEqual(result, "Some answer")

    @patch("builtins.open", mock_open(read_data=_SYSTEM_PROMPT))
    def test_should_call_model_ask_with_session_id_when_asking(self):
        self.agent.ask("find me a flat", stream=False)
        call_args = self.mock_model.ask.call_args
        # args: (system_prompt, prompt, session_id, stream=False)
        self.assertIsNotNone(call_args[0][2])  # session_id is 3rd positional arg

    @patch("builtins.open", mock_open(read_data=_SYSTEM_PROMPT))
    def test_should_pass_stream_flag_to_model_ask_when_calling(self):
        self.agent.ask("prompt", stream=True)
        call_kwargs = self.mock_model.ask.call_args[1]
        self.assertEqual(call_kwargs.get("stream"), True)

    @patch("builtins.open", mock_open(read_data=_SYSTEM_PROMPT))
    def test_should_return_model_response_when_asking(self):
        self.mock_model.ask.return_value = "Warsaw listing found"
        result = self.agent.ask("Warsaw flat", stream=False)
        self.assertEqual(result, "Warsaw listing found")


class TestLocalAgentClose(unittest.TestCase):

    def setUp(self):
        self.agent, self.mock_model = _make_agent()

    def test_should_call_model_close_when_close_is_called(self):
        self.agent.close()
        self.mock_model.close.assert_called_once()


class TestLocalAgentContextManager(unittest.TestCase):

    def setUp(self):
        self.agent, self.mock_model = _make_agent()

    def test_should_return_self_when_entering_context_manager(self):
        result = self.agent.__enter__()
        self.assertIs(result, self.agent)

    def test_should_call_close_when_exiting_context_manager(self):
        self.agent.__exit__(None, None, None)
        self.mock_model.close.assert_called_once()
