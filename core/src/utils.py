"""
core/src/utils.py — shared file-loading and service-lifecycle utilities.
"""
import logging
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_prompt(path: str | Path) -> str:
    """Read a prompt file and return its contents as a string."""
    return Path(path).read_text(encoding="utf-8")


def run_service(service, name: str) -> None:
    """
    Start a service that exposes start() / stop(), wiring SIGTERM and
    KeyboardInterrupt so it shuts down cleanly.

    Accepts any object with:
        service.start()  — blocks until stopped
        service.stop()   — signals the loop to exit
    """
    def _handle_sigterm(signum, frame):
        logger.info("[%s] SIGTERM — shutting down", name)
        service.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("[%s] keyboard interrupt — shutting down", name)
        service.stop()
    except ValueError as exc:
        logger.error("[%s] configuration error: %s", name, exc)
        sys.exit(1)
