import logging
import os
import random

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from agentP.src.agent import LocalAgent
from agentP.src.config.config import Config
from clients.base import BaseClient

WAITING_JOKES = [
    "Why did the scarecrow win an award? Because he was outstanding in his field — much like your future home!",
    "Why do real estate agents make great comedians? Because they always know how to close!",
    "What do you call a haunted property? A buy-one-get-one-scare deal.",
    "Why did the house go to therapy? It had too many issues with its foundation.",
    "What did the ocean say to the beach house? Nothing, it just waved.",
    "Why don't houses ever get lonely? Because they're always in a good neighbourhood.",
    "What's a real estate agent's favourite type of music? House!",
]

LOG_FILE = "logs/telegram_agent.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class _Bot:
    """Internal Telegram bot logic wired to LocalAgent."""

    def __init__(self):
        self.local_agent = LocalAgent()
        self.app = ApplicationBuilder().token(Config.TELEGRAM_BOT_TOKEN).build()
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Hello! I'm your property agent. Ask me anything about properties."
        )

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        logger.info(f"Received message from {update.effective_user.id}: {user_text}")

        joke = random.choice(WAITING_JOKES)
        sent = await update.message.reply_text(
            f"Your personal property agent is thinking...\n\n💡 While you wait: {joke}"
        )

        try:
            accumulated = ""
            last_sent = ""
            chunk_count = 0

            for token in self.local_agent.ask(user_text, stream=True):
                accumulated += token
                chunk_count += 1
                if chunk_count % 20 == 0 and accumulated != last_sent:
                    await sent.edit_text(accumulated)
                    last_sent = accumulated

            if accumulated and accumulated != last_sent:
                await sent.edit_text(accumulated)
            elif not accumulated:
                await sent.edit_text("No response generated.")

        except Exception as e:
            logger.error(f"Error from LocalAgent: {e}")
            await sent.edit_text("Sorry, something went wrong. Please try again.")

    def run(self):
        self.app.run_polling()


class TelegramClient(BaseClient):
    """Runs the Telegram bot via long-polling."""

    def __init__(self):
        self._bot = _Bot()

    def start(self) -> None:
        logger.info("Starting Telegram client...")
        self._bot.run()


if __name__ == "__main__":
    TelegramClient().start()
