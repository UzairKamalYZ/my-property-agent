import unittest
from unittest.mock import patch, MagicMock

from agentP.src.model.llm_model import LlmModel


class TestLlmModel(unittest.TestCase):

    @patch('agentP.src.model.llm_model.SessionManager')
    @patch('agentP.src.model.llm_model.RagContextManager')
    @patch('agentP.src.model.llm_model.Embedder')
    def setUp(self, MockEmbedderClass, MockRagContextManagerClass, MockSessionManagerClass):
        """Set up a fresh LlmModel instance for each test."""
        
        # --- Create a mock for the LLM, which is now injected ---
        self.mock_llm = MagicMock()
        
        # --- Instantiate the LlmModel ---
        # The LlmModel.__init__ now takes a pre-made LLM instance.
        self.llm_model = LlmModel(self.mock_llm)

        # --- Override the constructed chains with MagicMocks ---
        # This allows us to test the public methods ('ask_direct',
        # 'ask_with_reformulation') by checking if they call the correct
        # internal chain.
        self.llm_model.direct_chain_with_history = MagicMock(name="DirectChainMock")
        self.llm_model.full_chain = MagicMock(name="FullChainMock")


    def test_ask_direct(self):
        """
        Tests that the ask_direct method invokes the direct_chain_with_history.
        """
        # --- Arrange ---
        user_prompt = "A direct question."
        session_id = "session_direct_1"
        expected_response = "A direct answer."
        self.llm_model.direct_chain_with_history.invoke.return_value = expected_response
        
        # --- Act ---
        response = self.llm_model.ask_direct(user_prompt, session_id)

        # --- Assert ---
        # Verify that the correct chain was invoked with the correct parameters
        self.llm_model.direct_chain_with_history.invoke.assert_called_once_with(
            {"input": user_prompt},
            config={'configurable': {'session_id': session_id}}
        )
        self.llm_model.full_chain.invoke.assert_not_called() # Ensure other chains weren't called
        self.assertEqual(response, expected_response)


    def test_ask_with_reformulation(self):
        """
        Tests that the ask_with_reformulation method invokes the full_chain.
        """
        # --- Arrange ---
        user_prompt = "A question to be reformulated."
        session_id = "session_reformulate_1"
        expected_response = "An answer from the full RAG pipeline."
        self.llm_model.full_chain.invoke.return_value = expected_response
        
        # --- Act ---
        response = self.llm_model.ask_with_reformulation(user_prompt, session_id)

        # --- Assert ---
        # Verify that the correct chain was invoked with the correct parameters
        self.llm_model.full_chain.invoke.assert_called_once_with(
            user_prompt,
            config={'configurable': {'session_id': session_id}}
        )
        self.llm_model.direct_chain_with_history.invoke.assert_not_called() # Ensure other chains weren't called
        self.assertEqual(response, expected_response)


    def test_ask_with_reformulation_streaming(self):
        """
        Tests the streaming version of ask_with_reformulation.
        """
        # --- Arrange ---
        user_prompt = "A question to be streamed."
        session_id = "session_stream_1"
        expected_chunks = ["chunk1", "chunk2"]
        self.llm_model.full_chain.stream.return_value = iter(expected_chunks)
        
        # --- Act ---
        response_generator = self.llm_model.ask_with_reformulation(user_prompt, session_id, stream=True)
        response_list = list(response_generator)

        # --- Assert ---
        # Verify that the correct chain's stream method was called
        self.llm_model.full_chain.stream.assert_called_once_with(
            user_prompt,
            config={'configurable': {'session_id': session_id}}
        )
        self.llm_model.full_chain.invoke.assert_not_called()
        self.assertEqual(response_list, expected_chunks)


if __name__ == '__main__':
    unittest.main()
