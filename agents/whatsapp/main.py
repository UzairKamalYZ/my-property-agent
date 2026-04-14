"""
agents/whatsapp/main.py — entry point for the WhatsApp group listener.

Run from the project root:
    python -m agents.whatsapp.main

What it does
------------
1. Connects to your WhatsApp account through the Green API instance
   configured in .env (WHATSAPP_INSTANCE_ID + WHATSAPP_INSTANCE_TOKEN).
2. Loads the target group specified by WHATSAPP_GROUP_ID.
3. Listens to all new messages arriving in that group.
4. For each message, asks the LLM whether Uzair is being asked or mentioned.
5. If yes, automatically replies that Uzair is away from his phone.

Finding your group ID
---------------------
Run this snippet once to list all chats and find the group's chatId:

    from agents.whatsapp.green_api_client import GreenAPIClient
    for c in GreenAPIClient().get_contacts():
        if c.get("type") == "group":
            print(c["id"], c.get("name"))

Copy the printed ID (format: 120363xxxxxxxxx@g.us) into WHATSAPP_GROUP_ID in .env.

Stopping
--------
Press Ctrl+C — the listener catches the KeyboardInterrupt and shuts down cleanly.
"""
import logging
import signal
import sys

from logging_config import setup_logging

setup_logging()

from .listener import WhatsAppListener

logger = logging.getLogger(__name__)


def main() -> None:
    listener = WhatsAppListener()

    # Graceful shutdown on SIGTERM (e.g. systemd / Docker stop)
    def _handle_sigterm(signum, frame):
        logger.info("[whatsapp/main] SIGTERM received — shutting down")
        listener.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        listener.start()
    except KeyboardInterrupt:
        logger.info("[whatsapp/main] keyboard interrupt — shutting down")
        listener.stop()
    except ValueError as exc:
        # Config missing — tell the user clearly instead of dumping a traceback
        logger.error("[whatsapp/main] configuration error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
