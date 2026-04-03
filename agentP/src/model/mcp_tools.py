"""
MCP tool registry for the property agent.

MCP servers are declared in ``mcp.json`` at the project root.  Add a server
by appending an entry — no code changes required:

.. code-block:: json

    {
      "servers": [
        {"name": "finance", "command": ["npx", "--yes", "@easysolutions906/mcp-finance"]},
        {"name": "maps",    "command": ["npx", "--yes", "@acme/mcp-maps"]}
      ]
    }

The registry is built automatically at import time from that file.
"""

import json
import logging
from pathlib import Path

from .mcp_registry import MCPRegistry

logger = logging.getLogger(__name__)

# mcp.json lives at the project root — same convention as .env
_CONFIG_PATH = Path(__file__).parents[3] / "mcp.json"


def _load_registry(config_path: Path) -> MCPRegistry:
    registry = MCPRegistry()

    if not config_path.exists():
        logger.warning("mcp-tools: config not found at %s — no servers registered", config_path)
        return registry

    try:
        data = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.exception("mcp-tools: failed to parse %s — no servers registered", config_path)
        return registry

    for entry in data.get("servers", []):
        name = entry.get("name")
        command = entry.get("command")
        if name and command:
            logger.info(f"mcp-tools: adding server {name} command: {command}")
            registry.register(name, command)
        else:
            logger.warning("mcp-tools: skipping invalid entry %s", entry)

    return registry


_mcp = _load_registry(_CONFIG_PATH)


def close_mcp() -> None:
    """Shut down all registered MCP subprocesses. Call during application teardown."""
    _mcp.close()
