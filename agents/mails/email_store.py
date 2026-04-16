"""
agents/mails/email_store.py — SQLite-backed store that tracks which email
Message-IDs have already been summarised.

Uses a persistent connection for the lifetime of the process rather than
opening a new connection on every call.
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

    A single connection is kept open for the lifetime of the store.
    check_same_thread=False is safe here because the monitor runs in one thread.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._path = Path(db_path or Config.MAIL_SEEN_DB)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        logger.debug("[EmailStore] initialised at %s", self._path)

    def is_seen(self, message_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM seen_emails WHERE message_id = ?", (message_id,)
        ).fetchone()
        return row is not None

    def mark_seen(self, message_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_emails (message_id, seen_at) VALUES (?, ?)",
            (message_id, now),
        )
        self._conn.commit()
        logger.debug("[EmailStore] marked seen: %s", message_id[:60])

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
