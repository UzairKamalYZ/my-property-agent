import json
import logging
import subprocess
import threading
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# MCP initialize handshake — sent once when the subprocess starts.
_INIT_PAYLOAD = (
    json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "property-agent", "version": "1.0"},
        },
    })
    + "\n"
)

_TOOL_DESCRIPTION = (
    "Convert a monetary amount from one currency to another using live exchange "
    "rates (European Central Bank data via Frankfurter API). "
    "Use this tool when property prices are displayed in a non-USD currency "
    "(e.g. PLN, EUR, GBP) to show the equivalent amount in the user's preferred "
    "currency. Returns the converted amount, exchange rate, and rate date."
)


class _CurrencyConvertInput(BaseModel):
    """Input schema for the currency_convert tool."""

    from_currency: str = Field(
        description="Source 3-letter ISO currency code (e.g. 'PLN', 'EUR', 'GBP')"
    )
    to_currency: str = Field(
        description="Target 3-letter ISO currency code (e.g. 'USD', 'EUR', 'GBP')"
    )
    amount: float = Field(
        default=1.0,
        description="Amount to convert (default 1)",
    )


class FinanceMCPProcess:
    """
    Manages a persistent stdio subprocess running the finance MCP server.

    The process is started lazily on first use and kept alive for the
    lifetime of the application so that every tool call does not pay the
    npx startup cost.  A threading lock serialises concurrent calls.

    Usage
    -----
    Use the module-level singleton ``_mcp`` via ``call_tool()``.
    Call ``close()`` during application shutdown to terminate the process.
    """

    _MCP_COMMAND = ["npx", "--yes", "@easysolutions906/mcp-finance"]

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._req_id = 1

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _ensure_running(self) -> None:
        """Start the MCP subprocess and complete the initialization handshake."""
        if self._process is not None and self._process.poll() is None:
            return  # already alive

        logger.info("finance-mcp: starting subprocess")
        self._process = subprocess.Popen(
            self._MCP_COMMAND,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Send the MCP initialize message and discard the response —
        # we only care that the handshake completes without error.
        self._process.stdin.write(_INIT_PAYLOAD)
        self._process.stdin.flush()
        self._process.stdout.readline()
        logger.info("finance-mcp: ready")

    def _rpc(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC 2.0 request and return the parsed response."""
        self._req_id += 1
        payload = (
            json.dumps({
                "jsonrpc": "2.0",
                "id": self._req_id,
                "method": method,
                "params": params,
            })
            + "\n"
        )
        self._process.stdin.write(payload)
        self._process.stdin.flush()
        return json.loads(self._process.stdout.readline())

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Call a tool on the MCP server by name and return its text output.

        Starts the subprocess lazily on first call.  Thread-safe.
        Never raises — exceptions are caught and an error string is
        returned so the LLM sees a clean message instead of a traceback.

        Parameters
        ----------
        name      : MCP tool name, e.g. ``"currency_convert"``
        arguments : Dict of tool arguments as defined by the MCP schema
        """
        with self._lock:
            try:
                self._ensure_running()
                response = self._rpc("tools/call", {"name": name, "arguments": arguments})
                content = response.get("result", {}).get("content", [])
                return "\n".join(
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                )
            except Exception:
                logger.exception(
                    "finance-mcp: tool call failed name=%s args=%s", name, arguments
                )
                return "Currency conversion is currently unavailable."

    def close(self) -> None:
        """Terminate the MCP subprocess if it is running."""
        with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()
                logger.info("finance-mcp: subprocess terminated")
            self._process = None


# ---------------------------------------------------------------------------
# Module-level singleton — one subprocess shared by all tool instances.
# ---------------------------------------------------------------------------
_mcp = FinanceMCPProcess()


def make_currency_convert_tool() -> StructuredTool:
    """
    Return a LangChain StructuredTool that delegates to the finance MCP server.

    The underlying MCP subprocess is shared (singleton ``_mcp``) so only one
    ``npx`` process is ever started regardless of how many times this factory
    is called.

    The tool is automatically available to the LLM whenever it is bound via
    ``llm.bind_tools([..., make_currency_convert_tool()])``.  The LLM will
    call it when it identifies non-USD prices in property search results.
    """

    def _convert(from_currency: str, to_currency: str, amount: float = 1.0) -> str:
        logger.debug(
            "currency_convert: %.2f %s → %s", amount, from_currency, to_currency
        )
        # MCP arguments use "from" and "to" — reserved in Python, so we rename
        # them in the Pydantic schema and map back here.
        result = _mcp.call_tool(
            "currency_convert",
            {"from": from_currency, "to": to_currency, "amount": amount},
        )
        logger.info(
            "currency_convert: result_len=%d from=%s to=%s",
            len(result), from_currency, to_currency,
        )
        return result

    return StructuredTool.from_function(
        func=_convert,
        name="currency_convert",
        description=_TOOL_DESCRIPTION,
        args_schema=_CurrencyConvertInput,
    )


def close_finance_mcp() -> None:
    """Shut down the shared MCP subprocess.  Call during application teardown."""
    _mcp.close()
