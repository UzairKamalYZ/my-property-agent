"""
agents/mails/email_store.py — SQLite-backed store that tracks which email
Message-IDs have already been summarised.

This prevents the same email from appearing in multiple summaries across
check cycles.  The DB file path is controlled by MAIL_SEEN_DB in .env.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.src.config.config import Config

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS seen_emails (
    message_id TEXT PRIMARY KEY,
    seen_at    TEXT NOT NULL
)
"""


class EmailStore:
    """
    Persist and query the set of already-processed email Message-IDs.

    Thread-safety: each call opens and closes its own connection, which is
    safe for a single-threaded polling loop.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._path = Path(db_path or Config.MAIL_SEEN_DB)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()
        logger.debug("[EmailStore] initialised at %s", self._path)

    def is_seen(self, message_id: str) -> bool:
        """Return True if this Message-ID was already processed."""
        with sqlite3.connect(self._path) as conn:
            row = conn.execute(
                "SELECT 1 FROM seen_emails WHERE message_id = ?", (message_id,)
            ).fetchone()
        return row is not None

    def mark_seen(self, message_id: str) -> None:
        """Record a Message-ID as processed. Idempotent (INSERT OR IGNORE)."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO seen_emails (message_id, seen_at) VALUES (?, ?)",
                (message_id, now),
            )
            conn.commit()
        logger.debug("[EmailStore] marked seen: %s", message_id[:60])
