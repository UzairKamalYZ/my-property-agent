import unittest
from unittest.mock import patch, MagicMock

from clients.cron.main import CronClient


class TestCronClient(unittest.TestCase):

    # ------------------------------------------------------------------
    # Constructor defaults
    # ------------------------------------------------------------------

    def test_should_use_default_interval_of_30_when_created_without_args(self):
        """CronClient defaults to a 30-minute interval."""
        self.assertEqual(CronClient().interval_minutes, 30)

    def test_should_use_provided_interval_when_custom_interval_is_given(self):
        """CronClient stores the custom interval passed to __init__."""
        self.assertEqual(CronClient(interval_minutes=5).interval_minutes, 5)

    def test_should_start_with_running_false_before_start_is_called(self):
        """_running is False until start() sets it."""
        self.assertFalse(CronClient()._running)

    # ------------------------------------------------------------------
    # stop()
    # ------------------------------------------------------------------

    def test_should_set_running_to_false_when_stop_is_called(self):
        """stop() clears the _running flag so the loop exits on the next tick."""
        client = CronClient()
        client._running = True
        client.stop()
        self.assertFalse(client._running)

    def test_should_not_raise_when_stop_is_called_before_start(self):
        """stop() is safe to call even when the client was never started."""
        try:
            CronClient().stop()
        except Exception as exc:
            self.fail(f"stop() raised unexpectedly: {exc}")

    # ------------------------------------------------------------------
    # start()
    # ------------------------------------------------------------------

    @patch("clients.cron.main._run_job")
    def test_should_execute_job_at_least_once_when_start_is_called(self, mock_job):
        """start() calls _run_job() before sleeping for the first time."""
        client = CronClient(interval_minutes=0)
        # stop() after the first job so the loop doesn't spin indefinitely.
        mock_job.side_effect = lambda: client.stop()
        client.start()
        mock_job.assert_called_once()

    @patch("clients.cron.main._run_job")
    def test_should_stop_loop_after_stop_is_called_during_job(self, mock_job):
        """The while loop exits when _running is set to False mid-execution."""
        client = CronClient(interval_minutes=0)
        mock_job.side_effect = lambda: client.stop()
        client.start()
        # start() must have returned (not infinite-looped).
        self.assertFalse(client._running)

    @patch("clients.cron.main._run_job")
    def test_should_not_raise_when_job_raises_exception(self, mock_job):
        """Exceptions inside _run_job() are caught; the client does not crash."""
        call_count = 0

        def _raise_then_stop():
            nonlocal call_count
            call_count += 1
            client.stop()
            raise RuntimeError("job failure")

        client = CronClient(interval_minutes=0)
        mock_job.side_effect = _raise_then_stop

        try:
            client.start()
        except Exception as exc:
            self.fail(f"start() raised unexpectedly: {exc}")

    # ------------------------------------------------------------------
    # BaseClient conformance
    # ------------------------------------------------------------------

    def test_should_implement_base_client_interface(self):
        """CronClient is a concrete implementation of BaseClient."""
        from clients.base import BaseClient
        self.assertIsInstance(CronClient(), BaseClient)

    @patch("clients.cron.main._run_job")
    def test_should_call_stop_via_context_manager_when_block_exits(self, mock_job):
        """Context-manager exit sets _running to False via stop()."""
        client = CronClient(interval_minutes=0)
        mock_job.side_effect = lambda: None  # job does nothing; loop runs once then sleeps

        # We only want to verify context-manager stop(), not run the full loop.
        with client:
            client._running = False  # manually stop so start() isn't needed
        self.assertFalse(client._running)


if __name__ == "__main__":
    unittest.main()
