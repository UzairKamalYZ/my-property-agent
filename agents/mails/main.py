"""
agents/mails/main.py — entry point for the email monitor.

Run from the project root:
    python -m agents.mails.main

What it does
------------
1. Reads MAIL_ACCOUNTS from .env — a JSON array of IMAP account configs.
2. Every MAIL_CHECK_INTERVAL_HOURS (default 2), connects to each inbox via
   IMAP and fetches all emails received since the last check.
3. New emails (not yet seen, tracked by Message-ID in a local SQLite DB) are
   summarised individually by the LLM.
4. Every MAIL_SUMMARY_INTERVAL_HOURS (default 5), if there are accumulated
   summaries, they are delivered in the format:

       1. From sender@domain.com to uzair@box.com
       Summary: ...

       2. From other@domain.com to uzair@other.com
       Summary: ...

   Delivery goes to the WhatsApp group set in MAIL_SUMMARY_WHATSAPP_GROUP_ID.
   If that is not set, the summary is written to the application log instead.

5. Already-sent summaries are never repeated — each Message-ID is recorded
   in MAIL_SEEN_DB (default: mail_seen.db) so repetitions are impossible even
   after a restart.

Stopping
--------
Press Ctrl+C or send SIGTERM.  The current tick finishes cleanly first.
"""
import logging
import signal
import sys

from logging_config import setup_logging

setup_logging()

from .monitor import MailMonitor

logger = logging.getLogger(__name__)


def main() -> None:
    monitor = MailMonitor()

    def _handle_sigterm(signum, frame):
        logger.info("[mails/main] SIGTERM received — shutting down")
        monitor.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        monitor.start()
    except KeyboardInterrupt:
        logger.info("[mails/main] keyboard interrupt — shutting down")
        monitor.stop()
    except ValueError as exc:
        logger.error("[mails/main] configuration error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
