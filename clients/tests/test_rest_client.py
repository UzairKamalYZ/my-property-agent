import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from clients.rest.main import RestClient, app, sse_formatter


class TestRestClient(unittest.TestCase):

    # ------------------------------------------------------------------
    # Constructor defaults
    # ------------------------------------------------------------------

    def test_should_use_default_host_when_created_without_args(self):
        """RestClient defaults to host 0.0.0.0."""
        self.assertEqual(RestClient().host, "0.0.0.0")

    def test_should_use_default_port_when_created_without_args(self):
        """RestClient defaults to port 8000."""
        self.assertEqual(RestClient().port, 8000)

    def test_should_use_provided_host_when_custom_host_is_given(self):
        """RestClient stores the custom host passed to __init__."""
        self.assertEqual(RestClient(host="127.0.0.1").host, "127.0.0.1")

    def test_should_use_provided_port_when_custom_port_is_given(self):
        """RestClient stores the custom port passed to __init__."""
        self.assertEqual(RestClient(port=9999).port, 9999)

    # ------------------------------------------------------------------
    # start()
    # ------------------------------------------------------------------

    @patch("clients.rest.main.uvicorn.run")
    def test_should_call_uvicorn_run_once_when_start_is_called(self, mock_run):
        """start() delegates to uvicorn.run exactly once."""
        RestClient().start()
        mock_run.assert_called_once()

    @patch("clients.rest.main.uvicorn.run")
    def test_should_pass_fastapi_app_to_uvicorn_when_start_is_called(self, mock_run):
        """start() passes the module-level FastAPI app as the first argument."""
        RestClient().start()
        self.assertIs(mock_run.call_args[0][0], app)

    @patch("clients.rest.main.uvicorn.run")
    def test_should_pass_host_and_port_to_uvicorn_when_start_is_called(self, mock_run):
        """start() forwards the configured host and port to uvicorn.run."""
        RestClient(host="127.0.0.1", port=9999).start()
        mock_run.assert_called_once_with(app, host="127.0.0.1", port=9999)

    # ------------------------------------------------------------------
    # BaseClient conformance
    # ------------------------------------------------------------------

    def test_should_implement_base_client_interface(self):
        """RestClient is a concrete implementation of BaseClient."""
        from clients.base import BaseClient
        self.assertIsInstance(RestClient(), BaseClient)

    @patch("clients.rest.main.uvicorn.run")
    def test_should_call_stop_via_context_manager_when_block_exits(self, _mock_run):
        """Context-manager exit triggers stop() without raising."""
        client = RestClient()
        with client:
            pass  # stop() inherited no-op — must not raise


class TestSseFormatter(unittest.TestCase):
    """Unit tests for the SSE helper used by the /ask streaming endpoint."""

    def _collect(self, iterable):
        """Run the async generator to completion and return all yielded values."""
        async def _gather():
            return [chunk async for chunk in sse_formatter(iterable)]

        return asyncio.run(_gather())

    def test_should_wrap_each_chunk_in_sse_data_event_when_formatting(self):
        """sse_formatter wraps every string as 'data: <value>\\n\\n'."""
        result = self._collect(["hello", "world"])
        self.assertEqual(result, ["data: hello\n\n", "data: world\n\n"])

    def test_should_return_empty_list_when_stream_is_empty(self):
        """sse_formatter yields nothing for an empty iterable."""
        self.assertEqual(self._collect([]), [])

    def test_should_preserve_chunk_content_verbatim_when_formatting(self):
        """sse_formatter does not modify the chunk text."""
        result = self._collect(["chunk with spaces  "])
        self.assertIn("chunk with spaces  ", result[0])


if __name__ == "__main__":
    unittest.main()
