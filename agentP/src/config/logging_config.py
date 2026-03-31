"""
Centralised logging configuration for my-property-agent.

Call ``setup_logging()`` once at each process entry point (REST server,
CLI agent, cron job).  Every other module only needs:

    import logging
    logger = logging.getLogger(__name__)

Log level is controlled by the LOG_LEVEL env var (default: INFO).
Logs are written to stdout and to a rotating file under agentP/logs/.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).parents[2] / "logs"
_LOG_FILE = _LOG_DIR / "property_agent.log"

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

_FMT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure the root logger with console + rotating-file handlers.

    Safe to call multiple times — handlers are only added once.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        # Already configured (e.g., called twice in the same process)
        return

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file — 5 MB per file, keep 3 backups
    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "langsmith", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
