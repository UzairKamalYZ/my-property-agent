"""
agents/mails/main.py — entry point for the email monitor.

Run from the project root:
    python -m agents.mails.main
"""
import logging

from logging_config import setup_logging

setup_logging()

from core.src.config.config import Config
from core.src.utils import run_service
from .monitor import MailMonitor

logger = logging.getLogger(__name__)


def _make_whatsapp_sender(group_id: str):
    """Return a delivery callable that sends text to a WhatsApp group."""
    def _send(message: str) -> None:
        from agents.whatsapp.green_api_client import GreenAPIClient
        GreenAPIClient().send_message(group_id, message)
        logger.info("[mails/main] summary delivered to WhatsApp group %s", group_id)
    return _send


def main() -> None:
    send_fn = _make_whatsapp_sender(Config.MAIL_SUMMARY_WHATSAPP_GROUP_ID) \
        if Config.MAIL_SUMMARY_WHATSAPP_GROUP_ID else None
    run_service(MailMonitor(send_summary=send_fn), "mails")


if __name__ == "__main__":
    main()
