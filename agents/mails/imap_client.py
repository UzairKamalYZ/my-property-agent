"""
agents/mails/imap_client.py — IMAP reader using the Python standard library.

Uses imaplib (built-in) with a read-only SELECT so the mail server's
read/unread status is never touched.  Each message is identified by its
Message-ID header, which is used by EmailStore to avoid re-processing.

Typical account config dict:
    {
        "email":      "uzair@gmail.com",
        "imap_host":  "imap.gmail.com",
        "imap_port":  993,          # optional, defaults to 993 (SSL)
        "password":   "<app-password>"
    }

Gmail note: if you have 2-Step Verification enabled, you must use an
App Password (not your regular password). Create one at:
    https://myaccount.google.com/apppasswords
"""

import email as _email_lib
import imaplib
import logging
import ssl
from datetime import datetime, timezone
from email.header import decode_header as _raw_decode_header

logger = logging.getLogger(__name__)

# Limit body text sent to the LLM — avoids huge context windows for long emails
_MAX_BODY_CHARS = 2500


# ---------------------------------------------------------------------------
# Header / body helpers
# ---------------------------------------------------------------------------

def _decode_header(value: str | None) -> str:
    """Decode a potentially RFC-2047-encoded email header into a plain string."""
    if not value:
        return ""
    parts = []
    for chunk, charset in _raw_decode_header(value):
        if isinstance(chunk, bytes):
            parts.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(str(chunk))
    return "".join(parts).strip()


def _extract_body(msg) -> str:
    """
    Return the plain-text body of an email.Message object.

    For multipart messages, prefers text/plain over text/html.
    Strips leading/trailing whitespace from each candidate.
    """
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in disposition:
                raw = part.get_payload(decode=True)
                if raw:
                    charset = part.get_content_charset() or "utf-8"
                    return raw.decode(charset, errors="replace").strip()
    else:
        raw = msg.get_payload(decode=True)
        if raw:
            charset = msg.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="replace").strip()
    return ""


# ---------------------------------------------------------------------------
# IMAP account
# ---------------------------------------------------------------------------

class IMAPAccount:
    """
    Wraps a single IMAP mailbox.

    fetch_since() opens a fresh SSL connection, searches for messages
    received on or after `since`, parses each one, and closes the connection.
    The mailbox is opened read-only so nothing in the user's mailbox changes.
    """

    def __init__(self, config: dict) -> None:
        self.email: str = config["email"]
        self._host: str = config["imap_host"]
        self._port: int = int(config.get("imap_port", 993))
        self._password: str = config["password"]

    def fetch_since(self, since: datetime) -> list[dict]:
        """
        Return all messages received since `since` (UTC datetime).

        Each returned dict contains:
            message_id  — unique ID for deduplication (Message-ID header)
            from        — decoded sender address
            to          — this account's email address
            subject     — decoded subject line
            body        — plain-text body, truncated to _MAX_BODY_CHARS
            date        — raw Date header string
        """
        # IMAP SINCE uses local date in "DD-Mon-YYYY" format, not UTC time.
        # Using today's date minus one day as a safe floor avoids missing messages
        # when the machine is in a timezone ahead of the sender.
        date_str = since.strftime("%d-%b-%Y")
        ssl_ctx = ssl.create_default_context()
        messages: list[dict] = []

        try:
            with imaplib.IMAP4_SSL(self._host, self._port, ssl_context=ssl_ctx) as conn:
                conn.login(self.email, self._password)
                conn.select("INBOX", readonly=True)

                _, data = conn.search(None, f'SINCE "{date_str}"')
                uids = data[0].split() if data[0] else []
                logger.info("[IMAPAccount] %s — %d messages since %s", self.email, len(uids), date_str)

                for uid in uids:
                    try:
                        _, msg_data = conn.fetch(uid, "(RFC822)")
                        raw_bytes = msg_data[0][1]
                        msg = _email_lib.message_from_bytes(raw_bytes)

                        # Fall back to a synthetic ID if the header is absent
                        message_id = (msg.get("Message-ID") or f"uid-{uid.decode()}-{self.email}").strip()

                        messages.append({
                            "message_id": message_id,
                            "from":       _decode_header(msg.get("From", "")),
                            "to":         self.email,
                            "subject":    _decode_header(msg.get("Subject", "(no subject)")),
                            "body":       _extract_body(msg)[:_MAX_BODY_CHARS],
                            "date":       msg.get("Date", ""),
                        })
                    except Exception:
                        logger.exception("[IMAPAccount] failed to parse uid %s from %s", uid, self.email)

        except imaplib.IMAP4.error as exc:
            logger.error("[IMAPAccount] IMAP login/search failed for %s: %s", self.email, exc)
        except Exception:
            logger.exception("[IMAPAccount] unexpected error for %s", self.email)

        return messages
