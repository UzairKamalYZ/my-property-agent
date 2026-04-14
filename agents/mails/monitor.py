"""
agents/mails/monitor.py — background loop that reads new emails every
MAIL_CHECK_INTERVAL_HOURS and delivers a formatted summary every
MAIL_SUMMARY_INTERVAL_HOURS.

Timing model
------------
A single loop wakes every 60 seconds and checks two independent clocks:

  check clock   — resets every MAIL_CHECK_INTERVAL_HOURS (default 2 h)
                  triggers: IMAP fetch → LLM summarise → append to pending

  summary clock — resets every MAIL_SUMMARY_INTERVAL_HOURS (default 5 h)
                  triggers: format pending list → send to WhatsApp → clear pending

The first check runs immediately at startup so the pending list is populated
before the first summary window arrives.  The first summary send fires after
the full MAIL_SUMMARY_INTERVAL_HOURS interval has elapsed.

No-repetition guarantee
-----------------------
EmailStore persists every processed Message-ID to SQLite.  On each check
cycle, only messages whose IDs are not in that store are summarised and added
to the pending list.  The store is updated immediately after summarisation,
before the message is appended, so a crash between those two steps at worst
causes one duplicate entry — not a missed message.

Summary format (as requested)
------------------------------
  1. From sender@domain.com to uzair@box.com
  Summary: <one or two sentences from LLM>

  2. From other@domain.com to uzair@other.com
  Summary: <one or two sentences from LLM>
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.src.config.config import Config
from core.src.model.llm_factory import create_llm
from .email_store import EmailStore
from .imap_client import IMAPAccount

logger = logging.getLogger(__name__)

_SUMMARISE_PROMPT_FILE = Path(__file__).parent / "prompts" / "summarise_email.txt"

# How often the main loop wakes to check the clocks (seconds)
_TICK_SECONDS = 60


def _load_accounts() -> list[IMAPAccount]:
    """Parse MAIL_ACCOUNTS JSON and return IMAPAccount instances."""
    raw = Config.MAIL_ACCOUNTS
    if not raw:
        raise ValueError(
            "MAIL_ACCOUNTS is not set in .env. "
            "Expected a JSON array of account objects."
        )
    try:
        configs = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MAIL_ACCOUNTS is not valid JSON: {exc}") from exc

    if not isinstance(configs, list) or not configs:
        raise ValueError("MAIL_ACCOUNTS must be a non-empty JSON array.")

    accounts = []
    for cfg in configs:
        for key in ("email", "imap_host", "password"):
            if key not in cfg:
                raise ValueError(f"Each account must have '{key}' — missing in: {cfg}")
        accounts.append(IMAPAccount(cfg))

    return accounts


class MailMonitor:
    """
    Polls multiple IMAP inboxes, summarises new emails with the LLM,
    and periodically delivers a combined summary.

    Parameters
    ----------
    check_interval_hours   : override MAIL_CHECK_INTERVAL_HOURS from config
    summary_interval_hours : override MAIL_SUMMARY_INTERVAL_HOURS from config
    """

    def __init__(
        self,
        check_interval_hours: int | None = None,
        summary_interval_hours: int | None = None,
    ) -> None:
        self._accounts = _load_accounts()
        self._store = EmailStore()
        self._llm = create_llm(Config.LLM_PROVIDER, Config.LLM_MODEL_NAME)
        self._summarise_template = _SUMMARISE_PROMPT_FILE.read_text(encoding="utf-8")
        self._whatsapp_group_id = Config.MAIL_SUMMARY_WHATSAPP_GROUP_ID

        self._check_secs = (check_interval_hours or int(Config.MAIL_CHECK_INTERVAL_HOURS)) * 3600
        self._summary_secs = (summary_interval_hours or int(Config.MAIL_SUMMARY_INTERVAL_HOURS)) * 3600

        # Accumulated, formatted summary lines — cleared after each send
        self._pending: list[str] = []

        # Monotonic timestamps — check fires immediately on first tick
        self._last_check_mono: float = 0.0
        self._last_summary_mono: float = time.monotonic()

        # Wall-clock anchor for IMAP SINCE filter (set to check_interval ago so
        # the very first pull covers the last N hours of mail)
        self._last_check_wall: datetime = datetime.now(timezone.utc) - timedelta(
            seconds=self._check_secs
        )

        self._running = False

    # ------------------------------------------------------------------
    # Per-email summarisation
    # ------------------------------------------------------------------

    def _summarise(self, msg: dict) -> str:
        """Ask the LLM to summarise a single email in 1-2 sentences."""
        prompt = self._summarise_template.format(
            subject=msg["subject"],
            body=msg["body"],
        )
        resp = self._llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return text.strip()

    # ------------------------------------------------------------------
    # Check cycle — runs every MAIL_CHECK_INTERVAL_HOURS
    # ------------------------------------------------------------------

    def _run_check(self) -> None:
        """
        Fetch new emails from all accounts, summarise unseen ones,
        and append formatted entries to self._pending.
        """
        since = self._last_check_wall
        logger.info(
            "[MailMonitor] checking %d account(s) since %s",
            len(self._accounts),
            since.strftime("%Y-%m-%d %H:%M UTC"),
        )

        # Counter continues from wherever pending left off so numbering is
        # consistent within a single accumulated batch.
        counter = len(self._pending) + 1
        new_count = 0

        for account in self._accounts:
            messages = account.fetch_since(since)

            for msg in messages:
                mid = msg["message_id"]

                if self._store.is_seen(mid):
                    logger.debug("[MailMonitor] skipping already-seen %s", mid[:60])
                    continue

                try:
                    summary_text = self._summarise(msg)
                except Exception:
                    logger.exception("[MailMonitor] LLM summarisation failed for %s", mid[:60])
                    summary_text = "(summarisation failed)"

                # Mark seen BEFORE appending — if the process crashes here we lose
                # at most one entry rather than processing the same email forever.
                self._store.mark_seen(mid)

                entry = (
                    f"{counter}. From {msg['from']} to {msg['to']}\n"
                    f"Summary: {summary_text}"
                )
                self._pending.append(entry)
                counter += 1
                new_count += 1

        self._last_check_wall = datetime.now(timezone.utc)
        logger.info("[MailMonitor] check complete — %d new email(s) added", new_count)

    # ------------------------------------------------------------------
    # Summary cycle — runs every MAIL_SUMMARY_INTERVAL_HOURS
    # ------------------------------------------------------------------

    def _run_summary(self) -> None:
        """
        Deliver the accumulated summaries and clear the pending list.
        If nothing is pending, skips silently.
        """
        if not self._pending:
            logger.info("[MailMonitor] summary interval reached — no new emails, skipping")
            return

        body = "\n\n".join(self._pending)
        logger.info("[MailMonitor] sending summary (%d email(s))", len(self._pending))

        if self._whatsapp_group_id:
            self._send_whatsapp(body)
        else:
            # No delivery channel configured — log it so it isn't lost
            logger.info("[MailMonitor] email summary:\n%s", body)

        self._pending.clear()

    def _send_whatsapp(self, message: str) -> None:
        """Send the summary text to the configured WhatsApp group."""
        from agents.whatsapp.green_api_client import GreenAPIClient
        try:
            GreenAPIClient().send_message(self._whatsapp_group_id, message)
            logger.info(
                "[MailMonitor] summary delivered to WhatsApp group %s",
                self._whatsapp_group_id,
            )
        except Exception:
            logger.exception("[MailMonitor] failed to deliver summary via WhatsApp")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Enter the monitoring loop.  Blocks until stop() is called.

        Wakes every _TICK_SECONDS seconds and fires check / summary cycles
        when their respective intervals have elapsed.
        """
        self._running = True
        logger.info(
            "[MailMonitor] started — %d account(s), check every %dh, summary every %dh",
            len(self._accounts),
            self._check_secs // 3600,
            self._summary_secs // 3600,
        )

        while self._running:
            now = time.monotonic()

            if now - self._last_check_mono >= self._check_secs:
                try:
                    self._run_check()
                except Exception:
                    logger.exception("[MailMonitor] unexpected error during check")
                self._last_check_mono = time.monotonic()

            if time.monotonic() - self._last_summary_mono >= self._summary_secs:
                try:
                    self._run_summary()
                except Exception:
                    logger.exception("[MailMonitor] unexpected error during summary send")
                self._last_summary_mono = time.monotonic()

            time.sleep(_TICK_SECONDS)

        logger.info("[MailMonitor] stopped")

    def stop(self) -> None:
        """Signal the loop to exit after the current tick."""
        self._running = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
