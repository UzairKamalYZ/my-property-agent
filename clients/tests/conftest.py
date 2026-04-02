"""
Pytest configuration for clients/tests/.

Responsibilities:
1. Stub agentP.src.agent and agentP.src.config.config at the sys.modules
   level.  This stops Python following the deep LLM/vector-store import chain
   when client modules are collected (e.g. `from agentP.src.agent import
   LocalAgent`), while leaving agentP.src itself as a real package so that
   agentP/tests can still import the real model and persistence modules.
2. Stub third-party packages that are absent from the minimal test environment:
   uvicorn, fastapi, and telegram.
"""

import sys
import importlib.machinery
from unittest.mock import MagicMock


def _stub(name: str) -> MagicMock:
    """Register a MagicMock in sys.modules and return it.

    Sets __spec__ to a real ModuleSpec so that importlib.util.find_spec()
    does not raise ValueError (Python 3.13 + transformers compatibility).
    """
    mock = MagicMock()
    mock.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = mock
    return mock


# ---------------------------------------------------------------------------
# agentP.src.agent — clients import LocalAgent from here.
# Stubbing the *module* (not the package) blocks the whole agent import chain
# (llm_factory → langchain_ollama, llm_model_graph → langgraph, etc.)
# without touching the real agentP.src package that agentP/tests rely on.
# ---------------------------------------------------------------------------
_agent_mod = _stub("agentP.src.agent")
_agent_mod.LocalAgent = MagicMock

# agentP.src.config.config — clients read Config attributes (API_KEY, tokens…)
_config_mod = _stub("agentP.src.config.config")
_Config = MagicMock()
_Config.API_KEY = ""
_Config.TELEGRAM_BOT_TOKEN = "test-token"
_Config.CRON_SEARCH_PROMPT = "find properties"
# llm_model_graph.py reads these three at module level via os.environ.setdefault;
# they must be real strings or the call raises TypeError when both test suites run together.
_Config.LANGCHAIN_TRACING_V2 = "false"
_Config.LANGCHAIN_PROJECT = "my-property-agent"
_Config.LANGCHAIN_API_KEY = ""
_Config.AI_PROVIDER_API_KEY = "ollama"
_Config.AI_PROVIDER_BASE_URL = "http://localhost:11434/v1"
_config_mod.Config = _Config

# ---------------------------------------------------------------------------
# Third-party packages not installed in the minimal test environment
# ---------------------------------------------------------------------------

# uvicorn / fastapi
_stub("uvicorn")
_fastapi = _stub("fastapi")
_fastapi.FastAPI       = MagicMock
_fastapi.Depends       = MagicMock()
_fastapi.HTTPException = MagicMock
_fastapi.Security      = MagicMock()
_stub("fastapi.responses").StreamingResponse = MagicMock
_stub("fastapi.security").APIKeyHeader       = MagicMock

# telegram (python-telegram-bot)
_tg            = _stub("telegram")
_tg.Update     = MagicMock()
_tg_ext        = _stub("telegram.ext")
_tg_ext.ApplicationBuilder = MagicMock()
_tg_ext.CommandHandler     = MagicMock()
_tg_ext.MessageHandler     = MagicMock()
_ctx_types                 = MagicMock()
_ctx_types.DEFAULT_TYPE    = MagicMock()
_tg_ext.ContextTypes       = _ctx_types
_tg_ext.filters            = MagicMock()
