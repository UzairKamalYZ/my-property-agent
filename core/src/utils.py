"""
core/src/utils.py — shared file-loading utilities.
"""
from pathlib import Path


def load_prompt(path: str | Path) -> str:
    """Read a prompt file and return its contents as a string."""
    return Path(path).read_text(encoding="utf-8")
