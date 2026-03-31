import unittest
from unittest.mock import patch, MagicMock

from agentP.src.model.llm_model import LlmModel


_FAKE_SYSTEM_PROMPT = "You are a helpful property agent."
_FAKE_REFORMULATION_TEMPLATE = "Improve this query: {user_prompt}"


class TestLlmModel(unittest.TestCase):

    @patch("agentP.src.model.llm_model.SessionManager")
    @patch("agentP.src.model.llm_model.RagContextManager")
    @patch("agentP.src.model.llm_model.Embedder")
    @patch.object(LlmModel, "getPrompt")
    def setUp(self, mock_get_prompt, MockEmbedder, MockRagCtx, MockSession):
        """Build LlmModel with filesystem and external deps mocked."""
        mock_get_prompt.side_effect = [
            _FAKE_SYSTEM_PROMPT,
            _FAKE_REFORMULATION_TEMPLATE,
        ]
        self.mock_llm = MagicMock()
        self.model = LlmModel(self.mock_llm)
        # Replace built chains with controllable mocks
        self.model.full_chain = MagicMock(name="FullChainMock")
        self.model.direct_chain_with_history = MagicMock(name="DirectChainMock")

    # ------------------------------------------------------------------
    # ask() — non-streaming
    # ------------------------------------------------------------------

    def test_should_invoke_full_chain_when_ask_called_without_stream(self):
        self.model.full_chain.invoke.return_value = "answer"
        self.model.ask(_FAKE_SYSTEM_PROMPT, "find flat", "session-1", stream=False)
        self.model.full_chain.invoke.assert_called_once()

    def test_should_return_chain_result_when_asking_without_stream(self):
        self.model.full_chain.invoke.return_value = "three listings found"
        result = self.model.ask(_FAKE_SYSTEM_PROMPT, "flats", "session-1", stream=False)
        self.assertEqual(result, "three listings found")

    def test_should_pass_user_prompt_to_full_chain_when_asking(self):
        self.model.full_chain.invoke.return_value = "ok"
        self.model.ask(_FAKE_SYSTEM_PROMPT, "2-bed in Brussels", "session-1", stream=False)
        call_args = self.model.full_chain.invoke.call_args[0][0]
        self.assertEqual(call_args["user_prompt"], "2-bed in Brussels")

    def test_should_pass_session_id_in_config_when_asking(self):
        self.model.full_chain.invoke.return_value = "ok"
        self.model.ask(_FAKE_SYSTEM_PROMPT, "any", "my-session-42", stream=False)
        call_kwargs = self.model.full_chain.invoke.call_args[0][1]
        self.assertEqual(call_kwargs["configurable"]["session_id"], "my-session-42")

    # ------------------------------------------------------------------
    # ask() — streaming
    # ------------------------------------------------------------------

    def test_should_call_full_chain_stream_when_ask_called_with_stream_true(self):
        self.model.full_chain.stream.return_value = iter(["chunk1", "chunk2"])
        list(self.model.ask(_FAKE_SYSTEM_PROMPT, "flat", "session-1", stream=True))
        self.model.full_chain.stream.assert_called_once()

    def test_should_not_call_invoke_when_stream_is_true(self):
        self.model.full_chain.stream.return_value = iter([])
        list(self.model.ask(_FAKE_SYSTEM_PROMPT, "flat", "session-1", stream=True))
        self.model.full_chain.invoke.assert_not_called()

    def test_should_return_iterable_when_streaming(self):
        self.model.full_chain.stream.return_value = iter(["a", "b"])
        result = self.model.ask(_FAKE_SYSTEM_PROMPT, "flat", "session-1", stream=True)
        self.assertEqual(list(result), ["a", "b"])

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    def test_should_not_raise_when_close_is_called(self):
        try:
            self.model.close()
        except Exception as e:
            self.fail(f"close() raised unexpectedly: {e}")
