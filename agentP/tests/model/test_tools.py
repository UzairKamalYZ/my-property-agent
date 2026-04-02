import unittest
from unittest.mock import MagicMock, patch

from agentP.src.model.tools import make_property_search_tool, _TOOL_DESCRIPTION


class TestMakePropertySearchTool(unittest.TestCase):

    def setUp(self):
        self.mock_rag = MagicMock()
        self.mock_rag.get_context.return_value = "2-bed flat in Warsaw, €800/month"

    # ------------------------------------------------------------------
    # Tool identity
    # ------------------------------------------------------------------

    def test_should_return_tool_named_property_search(self):
        """make_property_search_tool() returns a tool whose name is 'property_search'."""
        tool = make_property_search_tool(self.mock_rag)
        self.assertEqual(tool.name, "property_search")

    def test_should_produce_independent_tools_when_called_twice(self):
        """Calling make_property_search_tool() twice returns two distinct tool objects."""
        tool_a = make_property_search_tool(self.mock_rag)
        tool_b = make_property_search_tool(self.mock_rag)
        self.assertIsNot(tool_a, tool_b)
        self.assertEqual(tool_a.name, tool_b.name)

    # ------------------------------------------------------------------
    # Happy-path invocation
    # ------------------------------------------------------------------

    def test_should_delegate_to_rag_context_manager_when_tool_is_invoked(self):
        """Invoking the tool calls rag_context_manager.get_context with the query."""
        tool = make_property_search_tool(self.mock_rag)

        tool.invoke({"query": "2 bedroom Warsaw"})

        self.mock_rag.get_context.assert_called_once()
        call_query = self.mock_rag.get_context.call_args[0][0]
        self.assertEqual(call_query, "2 bedroom Warsaw")

    def test_should_return_context_string_when_rag_succeeds(self):
        """Tool returns the string produced by rag_context_manager.get_context."""
        tool = make_property_search_tool(self.mock_rag)

        result = tool.invoke({"query": "cheap studio Krakow"})

        self.assertEqual(result, "2-bed flat in Warsaw, €800/month")

    def test_should_pass_rag_k_to_get_context(self):
        """Tool passes Config.RAG_K as the k argument to get_context."""
        with patch("agentP.src.model.tools.Config") as mock_config:
            mock_config.RAG_K = 7
            tool = make_property_search_tool(self.mock_rag)
            tool.invoke({"query": "flat Warsaw"})

        _, kwargs = self.mock_rag.get_context.call_args
        self.assertEqual(kwargs.get("k"), 7)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_should_return_unavailable_message_when_rag_raises(self):
        """Tool returns a user-friendly message instead of propagating the exception."""
        self.mock_rag.get_context.side_effect = RuntimeError("vector store offline")
        tool = make_property_search_tool(self.mock_rag)

        result = tool.invoke({"query": "flat Warsaw"})

        self.assertIn("unavailable", result.lower())
        self.assertNotIn("RuntimeError", result)
        self.assertNotIn("Traceback", result)

    def test_should_not_raise_when_rag_raises(self):
        """Tool never raises — exceptions from the vector store are caught internally."""
        self.mock_rag.get_context.side_effect = Exception("boom")
        tool = make_property_search_tool(self.mock_rag)

        try:
            tool.invoke({"query": "flat Warsaw"})
        except Exception as exc:
            self.fail(f"Tool raised unexpectedly: {exc}")

    # ------------------------------------------------------------------
    # Description vocabulary (routing quality)
    # ------------------------------------------------------------------

    def test_should_mention_core_property_vocabulary_in_description(self):
        """Tool description contains key terms the LLM uses to decide when to call it."""
        vocab = ["apartment", "flat", "house", "studio", "rent", "buy", "listing", "price"]
        missing = [word for word in vocab if word not in _TOOL_DESCRIPTION.lower()]
        self.assertEqual(
            missing,
            [],
            msg=f"Missing words in tool description: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
