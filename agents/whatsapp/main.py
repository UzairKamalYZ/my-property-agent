"""
agents/whatsapp/main.py — entry point for the WhatsApp group listener.

Run from the project root:
    python -m agents.whatsapp.main

Finding your group ID
---------------------
    from agents.whatsapp.green_api_client import GreenAPIClient
    for c in GreenAPIClient().get_contacts():
        if c.get("type") == "group":
            print(c["id"], c.get("name"))

Copy the printed ID (format: 120363xxxxxxxxx@g.us) into WHATSAPP_GROUP_ID in .env.
"""
from logging_config import setup_logging

setup_logging()

from core.src.utils import run_service
from .listener import WhatsAppListener


def main() -> None:
    run_service(WhatsAppListener(), "whatsapp")


if __name__ == "__main__":
    main()
