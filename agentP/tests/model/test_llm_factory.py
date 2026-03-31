import unittest
from unittest.mock import patch, MagicMock

from model.llm_factory import create_llm


class TestCreateLlm(unittest.TestCase):

    @patch("model.llm_factory.Config")
    @patch("model.llm_factory.Ollama")
    def test_should_return_ollama_instance_when_provider_is_ollama(
        self, MockOllama, MockConfig
    ):
        MockConfig.LLM_SEED = 42
        MockConfig.LLM_TEMPERATURE = 0.0
        mock_instance = MagicMock()
        MockOllama.return_value = mock_instance

        result = create_llm("ollama", "llama3.2")

        self.assertIs(result, mock_instance)

    @patch("model.llm_factory.Config")
    @patch("model.llm_factory.Ollama")
    def test_should_pass_model_name_to_ollama_when_creating(
        self, MockOllama, MockConfig
    ):
        MockConfig.LLM_SEED = 42
        MockConfig.LLM_TEMPERATURE = 0.0

        create_llm("ollama", "qwen3:8b")

        MockOllama.assert_called_once()
        call_kwargs = MockOllama.call_args[1]
        self.assertEqual(call_kwargs["model"], "qwen3:8b")

    @patch("model.llm_factory.Config")
    @patch("model.llm_factory.Ollama")
    def test_should_pass_config_seed_to_ollama_when_creating(
        self, MockOllama, MockConfig
    ):
        MockConfig.LLM_SEED = 365
        MockConfig.LLM_TEMPERATURE = 0.0

        create_llm("ollama", "any-model")

        call_kwargs = MockOllama.call_args[1]
        self.assertEqual(call_kwargs["seed"], 365)

    @patch("model.llm_factory.Config")
    @patch("model.llm_factory.Ollama")
    def test_should_pass_config_temperature_to_ollama_when_creating(
        self, MockOllama, MockConfig
    ):
        MockConfig.LLM_SEED = 42
        MockConfig.LLM_TEMPERATURE = 0.7

        create_llm("ollama", "any-model")

        call_kwargs = MockOllama.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)

    def test_should_raise_value_error_when_provider_is_unknown(self):
        with self.assertRaises(ValueError) as ctx:
            create_llm("openai", "gpt-4")
        self.assertIn("openai", str(ctx.exception))

    def test_should_raise_value_error_when_provider_is_empty_string(self):
        with self.assertRaises(ValueError):
            create_llm("", "any-model")

    def test_should_include_provider_name_in_error_message_when_unsupported(self):
        with self.assertRaises(ValueError) as ctx:
            create_llm("gemini", "gemini-pro")
        self.assertIn("gemini", str(ctx.exception))
