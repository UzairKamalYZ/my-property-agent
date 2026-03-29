import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters

from .agent import LocalAgent
from .config.config import Config

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MyTelegramAgent:
    """Telegram bot that forwards user messages to LocalAgent and replies with the response."""

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

        sent = await update.message.reply_text("Your personal property agent is thinking...")

        try:
            accumulated = ""
            last_sent = ""
            chunk_count = 0

            for token in self.local_agent.ask(user_text, stream=True):
                accumulated += token
                chunk_count += 1
                # Edit message every 20 tokens to avoid Telegram rate limits
                if chunk_count % 20 == 0 and accumulated != last_sent:
                    await sent.edit_text(accumulated)
                    last_sent = accumulated

            # Send the final complete response
            if accumulated and accumulated != last_sent:
                await sent.edit_text(accumulated)
            elif not accumulated:
                await sent.edit_text("No response generated.")

        except Exception as e:
            logger.error(f"Error from LocalAgent: {e}")
            await sent.edit_text("Sorry, something went wrong. Please try again.")

    def run(self):
        logger.info("Starting Telegram bot...")
        self.app.run_polling()


if __name__ == "__main__":
    bot = MyTelegramAgent()
    bot.run()
