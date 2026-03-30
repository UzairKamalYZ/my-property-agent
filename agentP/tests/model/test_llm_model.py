import unittest
from unittest.mock import patch, MagicMock

from agentP.src.model.llm_model import LlmModel


class TestLlmModel(unittest.TestCase):

    @patch("agentP.src.model.llm_model.LlmModel.getPrompt")
    @patch("agentP.src.model.llm_model.SessionManager")
    @patch("agentP.src.model.llm_model.RagContextManager")
    @patch("agentP.src.model.llm_model.Embedder")
    def setUp(self, MockEmbedder, MockRagContextManager, MockSessionManager, mock_get_prompt):
        # getPrompt is called twice during __init__:
        #   1. LlmModel.getPrompt(Config.PROMPT_FILE)           → system prompt
        #   2. _reformulated_prompt_template() → getPrompt(...)  → reformulation template
        # The second value must contain {user_prompt} so ChatPromptTemplate can render it.
        mock_get_prompt.side_effect = [
            "You are a helpful property agent.",
            "Rewrite this real-estate query: {user_prompt}",
        ]

        self.mock_llm = MagicMock()
        self.model = LlmModel(self.mock_llm)

        # Replace built chains with plain mocks so tests only verify routing,
        # not chain internals.
        self.model.direct_chain_with_history = MagicMock(name="DirectChainMock")
        self.model.full_chain = MagicMock(name="FullChainMock")

    # ------------------------------------------------------------------
    # ask_direct()
    # ------------------------------------------------------------------

    def test_should_invoke_direct_chain_with_correct_args_when_ask_direct_is_called(self):
        """ask_direct() passes {input: prompt} and the session config to the direct chain."""
        user_prompt = "What apartments are available?"
        session_id = "session-abc"
        self.model.direct_chain_with_history.invoke.return_value = "some answer"

        self.model.ask_direct(user_prompt, session_id)

        self.model.direct_chain_with_history.invoke.assert_called_once_with(
            {"input": user_prompt},
            config={"configurable": {"session_id": session_id}},
        )

    def test_should_return_direct_chain_response_when_ask_direct_is_called(self):
        """ask_direct() returns whatever the direct chain produces."""
        self.model.direct_chain_with_history.invoke.return_value = "Direct answer."

        result = self.model.ask_direct("Any question", "sess-1")

        self.assertEqual(result, "Direct answer.")

    def test_should_not_invoke_full_chain_when_ask_direct_is_called(self):
        """ask_direct() never touches the RAG / reformulation pipeline."""
        self.model.direct_chain_with_history.invoke.return_value = ""

        self.model.ask_direct("Any question", "sess-1")

        self.model.full_chain.invoke.assert_not_called()
        self.model.full_chain.stream.assert_not_called()

    # ------------------------------------------------------------------
    # ask_with_reformulation() — blocking
    # ------------------------------------------------------------------

    def test_should_invoke_full_chain_with_correct_args_when_ask_with_reformulation_is_called(self):
        """ask_with_reformulation() passes the raw prompt and session config to full_chain."""
        user_prompt = "2-bed flat in Warsaw"
        session_id = "sess-2"
        self.model.full_chain.invoke.return_value = "RAG answer."

        self.model.ask_with_reformulation(user_prompt, session_id)

        self.model.full_chain.invoke.assert_called_once_with(
            user_prompt,
            config={"configurable": {"session_id": session_id}},
        )

    def test_should_return_full_chain_response_when_ask_with_reformulation_is_called(self):
        """ask_with_reformulation() returns whatever the full chain produces."""
        self.model.full_chain.invoke.return_value = "RAG answer."

        result = self.model.ask_with_reformulation("2-bed flat in Warsaw", "sess-2")

        self.assertEqual(result, "RAG answer.")

    def test_should_not_invoke_direct_chain_when_ask_with_reformulation_is_called(self):
        """ask_with_reformulation() never calls the direct (no-RAG) chain."""
        self.model.full_chain.invoke.return_value = ""

        self.model.ask_with_reformulation("2-bed flat in Warsaw", "sess-2")

        self.model.direct_chain_with_history.invoke.assert_not_called()

    # ------------------------------------------------------------------
    # ask_with_reformulation() — streaming
    # ------------------------------------------------------------------

    def test_should_use_stream_method_on_full_chain_when_stream_flag_is_true(self):
        """ask_with_reformulation(stream=True) calls full_chain.stream, not .invoke."""
        chunks = ["chunk1", "chunk2"]
        self.model.full_chain.stream.return_value = iter(chunks)

        gen = self.model.ask_with_reformulation("query", "sess-3", stream=True)
        result = list(gen)

        self.model.full_chain.stream.assert_called_once_with(
            "query",
            config={"configurable": {"session_id": "sess-3"}},
        )
        self.model.full_chain.invoke.assert_not_called()
        self.assertEqual(result, chunks)

    def test_should_return_iterable_not_string_when_stream_flag_is_true(self):
        """ask_with_reformulation(stream=True) returns an iterable, not a plain string."""
        self.model.full_chain.stream.return_value = iter(["a"])

        result = self.model.ask_with_reformulation("query", "sess-4", stream=True)

        self.assertTrue(hasattr(result, "__iter__"))
        self.assertNotIsInstance(result, str)

    # ------------------------------------------------------------------
    # ask() — public facade
    # ------------------------------------------------------------------

    def test_should_delegate_to_full_chain_when_ask_is_called(self):
        """ask() is a facade that routes to the full RAG pipeline."""
        self.model.full_chain.invoke.return_value = "facade answer"

        result = self.model.ask("system prompt text", "user query", "sess-5")

        self.model.full_chain.invoke.assert_called_once()
        self.assertEqual(result, "facade answer")

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    def test_should_not_raise_when_close_is_called(self):
        """close() is a no-op and must not raise any exception."""
        try:
            self.model.close()
        except Exception as exc:
            self.fail(f"close() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
