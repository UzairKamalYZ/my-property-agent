import unittest
from unittest.mock import MagicMock, patch

from clients.telegram.main import TelegramClient


class TestTelegramClient(unittest.TestCase):

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @patch("clients.telegram.main.LocalAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_create_bot_instance_when_client_is_created(self, MockBot, _MockAgent):
        """TelegramClient instantiates _Bot during __init__."""
        TelegramClient()
        MockBot.assert_called_once()

    @patch("clients.telegram.main.LocalAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_store_bot_as_attribute_when_client_is_created(self, MockBot, _MockAgent):
        """TelegramClient stores the _Bot instance as self._bot."""
        mock_bot = MagicMock()
        MockBot.return_value = mock_bot
        client = TelegramClient()
        self.assertIs(client._bot, mock_bot)

    # ------------------------------------------------------------------
    # start()
    # ------------------------------------------------------------------

    @patch("clients.telegram.main.LocalAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_call_bot_run_when_start_is_called(self, MockBot, _MockAgent):
        """start() delegates to _bot.run() which starts long-polling."""
        mock_bot = MagicMock()
        MockBot.return_value = mock_bot
        client = TelegramClient()
        client.start()
        mock_bot.run.assert_called_once()

    @patch("clients.telegram.main.LocalAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_not_call_bot_run_before_start_is_called(self, MockBot, _MockAgent):
        """_bot.run() is never called during construction — only on start()."""
        mock_bot = MagicMock()
        MockBot.return_value = mock_bot
        TelegramClient()  # construction only
        mock_bot.run.assert_not_called()

    # ------------------------------------------------------------------
    # BaseClient conformance
    # ------------------------------------------------------------------

    @patch("clients.telegram.main.LocalAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_implement_base_client_interface(self, _MockBot, _MockAgent):
        """TelegramClient is a concrete implementation of BaseClient."""
        from clients.base import BaseClient
        self.assertIsInstance(TelegramClient(), BaseClient)

    @patch("clients.telegram.main.LocalAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_not_raise_when_stop_is_called(self, _MockBot, _MockAgent):
        """stop() is a safe no-op on TelegramClient."""
        client = TelegramClient()
        try:
            client.stop()
        except Exception as exc:
            self.fail(f"stop() raised unexpectedly: {exc}")

    @patch("clients.telegram.main.LocalAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_call_stop_via_context_manager_when_block_exits(self, MockBot, _MockAgent):
        """Context-manager exit triggers stop() without raising."""
        client = TelegramClient()
        with client:
            pass


class TestBot(unittest.TestCase):
    """Tests for the internal _Bot helper class."""

    @patch("clients.telegram.main.LocalAgent")
    def test_should_create_local_agent_when_bot_is_created(self, MockAgent):
        """_Bot instantiates LocalAgent during construction."""
        from clients.telegram.main import _Bot
        _Bot()
        MockAgent.assert_called_once()

    @patch("clients.telegram.main.LocalAgent")
    def test_should_build_telegram_app_with_bot_token_when_created(self, _MockAgent):
        """_Bot calls ApplicationBuilder().token(Config.TELEGRAM_BOT_TOKEN).build()."""
        from clients.telegram.main import _Bot
        from agentP.src.config.config import Config
        with patch("clients.telegram.main.ApplicationBuilder") as MockBuilder:
            _Bot()
            MockBuilder.return_value.token.assert_called_once_with(Config.TELEGRAM_BOT_TOKEN)


if __name__ == "__main__":
    unittest.main()
