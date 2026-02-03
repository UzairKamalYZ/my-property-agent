import unittest
from unittest.mock import patch, MagicMock

from agentP.src.model.llm_model import LlmModel


class TestLlmModel(unittest.TestCase):

    @patch('agentP.src.model.llm_model.RunnableWithMessageHistory')
    @patch('agentP.src.model.llm_model.SessionManager')
    @patch('agentP.src.model.llm_model.RagContextManager')
    @patch('agentP.src.model.llm_model.Ollama')
    def setUp(self, MockOllama, MockRagContextManager, MockSessionManager, MockRunnable):
        """Set up a fresh LlmModel instance for each test."""
        self.mock_ollama = MockOllama()
        self.mock_rag_context_manager = MockRagContextManager()
        self.mock_session_manager = MockSessionManager()
        self.mock_runnable = MockRunnable.return_value  # This is the instance of the runnable

        # Instantiate the class, which will use the mocked dependencies
        self.llm_model = LlmModel(model_name="test-model")


    def test_ask_method(self):
        """
        Tests the 'ask' method for direct LLM queries.
        """
        # --- Arrange ---
        user_prompt = "Hello, who are you?"
        session_id = "session_123"
        expected_response = "I am a test bot."
        self.mock_runnable.invoke.return_value = expected_response

        # --- Act ---
        response = self.llm_model.ask(user_prompt, session_id)

        # --- Assert ---
        # Verify that the runnable's invoke method was called correctly
        self.mock_runnable.invoke.assert_called_once_with(
            {"input": user_prompt},
            config={'configurable': {'session_id': session_id}}
        )
        self.assertEqual(response, expected_response)
        
        # Ensure RAG context manager was NOT used. We access this via the instance.
        self.llm_model.rag_context_manager.prepare_rag_prompt.assert_not_called()


    def test_chat_with_context_method(self):
        """
        Tests the 'chat_with_context' method for RAG queries.
        """
        # --- Arrange ---
        user_prompt = "Find me an apartment in Warsaw."
        session_id = "session_456"
        rag_prompt = "User question: ... Available listings: ..."
        expected_response = "Here is an apartment in Warsaw..."

        # Configure mocks
        self.llm_model.rag_context_manager.prepare_rag_prompt.return_value = rag_prompt
        self.mock_runnable.invoke.return_value = expected_response

        # --- Act ---
        response = self.llm_model.chat_with_context(user_prompt, session_id)

        # --- Assert ---
        # Verify that the RAG context manager was called
        self.llm_model.rag_context_manager.prepare_rag_prompt.assert_called_once_with(user_prompt)

        # Verify that the runnable's invoke method was called with the RAG prompt
        self.mock_runnable.invoke.assert_called_once_with(
            {"input": rag_prompt},
            config={'configurable': {'session_id': session_id}}
        )
        self.assertEqual(response, expected_response)


    def test_ask_method_streaming(self):
        """
        Tests the 'ask' method in streaming mode.
        """
        # --- Arrange ---
        user_prompt = "Stream a response."
        session_id = "session_stream_1"
        expected_stream_chunks = ["chunk1", "chunk2"]
        self.mock_runnable.stream.return_value = expected_stream_chunks

        # --- Act ---
        stream_response = self.llm_model.ask(user_prompt, session_id, stream=True)

        # --- Assert ---
        self.mock_runnable.stream.assert_called_once_with(
            {"input": user_prompt},
            config={'configurable': {'session_id': session_id}}
        )
        self.assertEqual(stream_response, expected_stream_chunks)


if __name__ == '__main__':
    unittest.main()
