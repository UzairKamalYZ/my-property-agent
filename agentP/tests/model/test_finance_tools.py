import json
import unittest
from unittest.mock import MagicMock, patch, call

from agentP.src.model.finance_tools import (
    FinanceMCPProcess,
    _CurrencyConvertInput,
    make_currency_convert_tool,
    close_finance_mcp,
)


class TestCurrencyConvertInput(unittest.TestCase):
    """Schema validation for the Pydantic args model."""

    def test_should_require_from_currency(self):
        """from_currency is a required field."""
        fields = _CurrencyConvertInput.model_fields
        self.assertIn("from_currency", fields)
        self.assertTrue(fields["from_currency"].is_required())

    def test_should_require_to_currency(self):
        """to_currency is a required field."""
        fields = _CurrencyConvertInput.model_fields
        self.assertIn("to_currency", fields)
        self.assertTrue(fields["to_currency"].is_required())

    def test_should_default_amount_to_one(self):
        """amount defaults to 1.0 when omitted."""
        model = _CurrencyConvertInput(from_currency="PLN", to_currency="USD")
        self.assertEqual(model.amount, 1.0)

    def test_should_accept_explicit_amount(self):
        """amount can be overridden."""
        model = _CurrencyConvertInput(from_currency="PLN", to_currency="USD", amount=800.0)
        self.assertEqual(model.amount, 800.0)


class TestMakeCurrencyConvertTool(unittest.TestCase):
    """Tests for the StructuredTool factory."""

    def test_should_return_tool_named_currency_convert(self):
        """make_currency_convert_tool() returns a tool named 'currency_convert'."""
        tool = make_currency_convert_tool()
        self.assertEqual(tool.name, "currency_convert")

    def test_should_use_explicit_args_schema(self):
        """Tool uses _CurrencyConvertInput as its args_schema."""
        tool = make_currency_convert_tool()
        self.assertIs(tool.args_schema, _CurrencyConvertInput)

    def test_should_produce_independent_tools_when_called_twice(self):
        """Calling the factory twice returns two distinct tool objects."""
        t1 = make_currency_convert_tool()
        t2 = make_currency_convert_tool()
        self.assertIsNot(t1, t2)
        self.assertEqual(t1.name, t2.name)

    def test_should_mention_non_usd_in_description(self):
        """Tool description tells the LLM when to call it (non-USD prices)."""
        tool = make_currency_convert_tool()
        self.assertIn("non-USD", tool.description)
        self.assertIn("PLN", tool.description)

    @patch("agentP.src.model.finance_tools._mcp")
    def test_should_delegate_to_mcp_with_correct_arguments(self, mock_mcp):
        """Invoking the tool calls _mcp.call_tool with 'from'/'to' keys."""
        mock_mcp.call_tool.return_value = "800.00 PLN = 200.00 USD"
        tool = make_currency_convert_tool()

        result = tool.invoke({"from_currency": "PLN", "to_currency": "USD", "amount": 800.0})

        mock_mcp.call_tool.assert_called_once_with(
            "currency_convert",
            {"from": "PLN", "to": "USD", "amount": 800.0},
        )
        self.assertEqual(result, "800.00 PLN = 200.00 USD")

    @patch("agentP.src.model.finance_tools._mcp")
    def test_should_return_mcp_result_string_on_success(self, mock_mcp):
        """Tool returns the string produced by the MCP server."""
        mock_mcp.call_tool.return_value = "1 EUR = 1.08 USD (rate date: 2024-01-15)"
        tool = make_currency_convert_tool()

        result = tool.invoke({"from_currency": "EUR", "to_currency": "USD", "amount": 1.0})

        self.assertEqual(result, "1 EUR = 1.08 USD (rate date: 2024-01-15)")


class TestFinanceMCPProcess(unittest.TestCase):
    """Unit tests for the subprocess manager (no real npx calls)."""

    def setUp(self):
        self.mcp = FinanceMCPProcess()

    def _make_mock_process(self, responses: list[str]):
        """Return a mock Popen whose stdout.readline() yields the given strings."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process is alive
        mock_proc.stdout.readline.side_effect = responses
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.flush = MagicMock()
        return mock_proc

    # ------------------------------------------------------------------
    # call_tool — happy path
    # ------------------------------------------------------------------

    @patch("agentP.src.model.finance_tools.subprocess.Popen")
    def test_should_start_subprocess_on_first_call(self, mock_popen):
        """call_tool starts the npx process if it is not running."""
        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}})
        tool_resp = json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "200 USD"}]}})
        mock_proc = self._make_mock_process([init_resp, tool_resp])
        mock_popen.return_value = mock_proc

        result = self.mcp.call_tool("currency_convert", {"from": "PLN", "to": "USD", "amount": 800})

        mock_popen.assert_called_once()
        self.assertEqual(result, "200 USD")

    @patch("agentP.src.model.finance_tools.subprocess.Popen")
    def test_should_reuse_process_on_second_call(self, mock_popen):
        """call_tool does not restart the subprocess on subsequent calls."""
        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})
        tool_resp = json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "ok"}]}})
        tool_resp2 = json.dumps({"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "ok2"}]}})
        mock_proc = self._make_mock_process([init_resp, tool_resp, tool_resp2])
        mock_popen.return_value = mock_proc

        self.mcp.call_tool("currency_convert", {"from": "PLN", "to": "USD", "amount": 1})
        self.mcp.call_tool("currency_convert", {"from": "EUR", "to": "USD", "amount": 1})

        # Popen called only once — same process reused
        self.assertEqual(mock_popen.call_count, 1)

    @patch("agentP.src.model.finance_tools.subprocess.Popen")
    def test_should_extract_text_content_from_mcp_response(self, mock_popen):
        """call_tool joins all 'text' content items from the MCP response."""
        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})
        tool_resp = json.dumps({
            "jsonrpc": "2.0", "id": 2,
            "result": {"content": [
                {"type": "text", "text": "Rate: 1 PLN = 0.25 USD"},
                {"type": "text", "text": "Date: 2024-01-15"},
            ]},
        })
        mock_proc = self._make_mock_process([init_resp, tool_resp])
        mock_popen.return_value = mock_proc

        result = self.mcp.call_tool("currency_convert", {"from": "PLN", "to": "USD", "amount": 1})

        self.assertIn("Rate: 1 PLN = 0.25 USD", result)
        self.assertIn("Date: 2024-01-15", result)

    # ------------------------------------------------------------------
    # call_tool — error handling
    # ------------------------------------------------------------------

    @patch("agentP.src.model.finance_tools.subprocess.Popen")
    def test_should_return_error_string_when_subprocess_raises(self, mock_popen):
        """call_tool never raises — returns a user-friendly error string on failure."""
        mock_popen.side_effect = FileNotFoundError("npx not found")

        result = self.mcp.call_tool("currency_convert", {"from": "PLN", "to": "USD", "amount": 1})

        self.assertIn("unavailable", result.lower())
        self.assertNotIn("FileNotFoundError", result)

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    def test_should_terminate_process_when_close_is_called(self):
        """close() calls terminate() on a running process."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        self.mcp._process = mock_proc

        self.mcp.close()

        mock_proc.terminate.assert_called_once()
        self.assertIsNone(self.mcp._process)

    def test_should_not_raise_when_close_called_with_no_process(self):
        """close() is safe to call before any tool call has been made."""
        try:
            self.mcp.close()
        except Exception as exc:
            self.fail(f"close() raised unexpectedly: {exc}")

    def test_should_not_raise_when_close_called_twice(self):
        """close() is idempotent."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        self.mcp._process = mock_proc

        self.mcp.close()
        try:
            self.mcp.close()
        except Exception as exc:
            self.fail(f"Second close() raised unexpectedly: {exc}")


class TestCloseFinanceMcp(unittest.TestCase):
    """Tests for the module-level close helper."""

    @patch("agentP.src.model.finance_tools._mcp")
    def test_should_delegate_to_singleton_close(self, mock_mcp):
        """close_finance_mcp() calls close() on the module-level _mcp singleton."""
        close_finance_mcp()
        mock_mcp.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
