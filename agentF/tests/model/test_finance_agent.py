import sys
import importlib.machinery
import unittest
from unittest.mock import patch, MagicMock

if "faiss" in sys.modules and getattr(sys.modules["faiss"], "__spec__", None) is None:
    sys.modules["faiss"].__spec__ = importlib.machinery.ModuleSpec("faiss", None)

if "langchain_openai" not in sys.modules:
    sys.modules["langchain_openai"] = MagicMock()

from agentF.src.agent import FinanceAgent

_FAKE_FINANCE_PROMPT = "You are a financial assistant."


class TestFinanceAgent(unittest.TestCase):

    def setUp(self):
        # create_llm / LlmModelGraph construction live in BaseAgent — patch there
        self._llm_factory_patcher = patch("core.src.base_agent.create_llm")
        self._graph_patcher = patch("core.src.base_agent.LlmModelGraph")
        self._embedder_patcher = patch("core.src.base_agent.Embedder")
        self._rag_patcher = patch("core.src.base_agent.RagContextManager")
        # _load_file is called on the real LlmModelGraph class inside
        # FinanceAgent.get_system_prompt() — patch it on the class itself
        self._load_file_patcher = patch(
            "core.src.model.llm_model_graph.LlmModelGraph._load_file",
            return_value=_FAKE_FINANCE_PROMPT,
        )

        self.mock_create_llm = self._llm_factory_patcher.start()
        self.mock_graph_cls = self._graph_patcher.start()
        self._embedder_patcher.start()
        self._rag_patcher.start()
        self._load_file_patcher.start()

        self.mock_llm = MagicMock()
        self.mock_create_llm.return_value = self.mock_llm
        self.mock_model = MagicMock()
        self.mock_graph_cls.return_value = self.mock_model

        self.agent = FinanceAgent(session_id="test-session")

    def tearDown(self):
        self._llm_factory_patcher.stop()
        self._graph_patcher.stop()
        self._embedder_patcher.stop()
        self._rag_patcher.stop()
        self._load_file_patcher.stop()

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_should_create_llm_with_config_values(self):
        self.mock_create_llm.assert_called_once()

    def test_should_pass_finance_system_prompt_to_graph(self):
        _, kwargs = self.mock_graph_cls.call_args
        self.assertIn("system_prompt", kwargs)
        self.assertEqual(kwargs["system_prompt"], _FAKE_FINANCE_PROMPT)

    def test_should_assign_session_id(self):
        self.assertEqual(self.agent.session_id, "test-session")

    def test_should_generate_session_id_when_none_provided(self):
        agent = FinanceAgent()
        self.assertIsNotNone(agent.session_id)

    # ------------------------------------------------------------------
    # ask()
    # ------------------------------------------------------------------

    def test_should_delegate_ask_to_model(self):
        self.mock_model.ask.return_value = "The USD/EUR rate is 1.08."
        result = self.agent.ask("What is USD to EUR?")
        self.mock_model.ask.assert_called_once()
        self.assertEqual(result, "The USD/EUR rate is 1.08.")

    def test_should_use_instance_session_id_when_none_passed(self):
        self.mock_model.ask.return_value = "ok"
        self.agent.ask("question")
        _, kwargs = self.mock_model.ask.call_args
        self.assertEqual(kwargs.get("session_id"), "test-session")

    def test_should_use_explicit_session_id_over_instance_id(self):
        self.mock_model.ask.return_value = "ok"
        self.agent.ask("question", session_id="override")
        _, kwargs = self.mock_model.ask.call_args
        self.assertEqual(kwargs.get("session_id"), "override")

    def test_should_return_stream_when_stream_true(self):
        self.mock_model.ask_stream.return_value = iter(["token1", "token2"])
        result = self.agent.ask("convert 100 PLN", stream=True)
        self.mock_model.ask_stream.assert_called_once()
        self.assertEqual(list(result), ["token1", "token2"])

    # ------------------------------------------------------------------
    # close() / context manager
    # ------------------------------------------------------------------

    def test_should_delegate_close_to_model(self):
        self.agent.close()
        self.mock_model.close.assert_called_once()

    def test_should_close_on_context_manager_exit(self):
        with self.agent:
            pass
        self.mock_model.close.assert_called_once()

    def test_should_not_raise_when_close_called(self):
        try:
            self.agent.close()
        except Exception as exc:
            self.fail(f"close() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
