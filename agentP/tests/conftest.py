"""
Pytest configuration for agentP/tests/.

Two responsibilities:
1. Add agentP/src to sys.path so that bare module imports used in the source
   (e.g. `from config.config import Config`) resolve correctly when tests are
   run from the project root.

2. Stub out heavy, optional third-party packages (pinecone, faiss,
   sentence_transformers) that are not available in the test environment so
   that test modules can be collected and imported without them installed.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 0. Minimal environment variables required by Config at class-body import time
# ---------------------------------------------------------------------------
os.environ.setdefault("STORE_TYPE", "local")
os.environ.setdefault("LLM_MODEL_NAME", "test-model")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_SEED", "42")
os.environ.setdefault("LLM_TEMPERATURE", "0.0")
os.environ.setdefault("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("MEMORY_FILE", "memory.json")
os.environ.setdefault("URLS_FILE", "../urls.txt")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_ENVIRONMENT", "test-env")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")
os.environ.setdefault("PROMPT_FILE", "agentP/prompts/System_Prompt.txt")
os.environ.setdefault("INTERACTION_FILE", "agentP/prompts/interaction.json")
os.environ.setdefault("REFORMULATION_PROMPT", "agentP/prompts/reformulated_prompt.txt")

# ---------------------------------------------------------------------------
# 1. Path fix
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# ---------------------------------------------------------------------------
# 2. Stub heavy third-party packages before any source module imports them
# ---------------------------------------------------------------------------

def _stub(name: str) -> MagicMock:
    """Register a MagicMock in sys.modules under *name* and return it."""
    mock = MagicMock()
    sys.modules[name] = mock
    return mock


# pinecone — used by persistence/pinecone_store.py
_pinecone = _stub("pinecone")
_pinecone.Pinecone = MagicMock
_pinecone.ServerlessSpec = MagicMock

# faiss — used by persistence/faiss_store.py
_stub("faiss")

# sentence_transformers — used by model/embedder.py
_st = _stub("sentence_transformers")
_st.SentenceTransformer = MagicMock

# langchain_community — used by model/session_manager.py
_lc_comm = _stub("langchain_community")
_lc_hist = _stub("langchain_community.chat_message_histories")
_lc_hist.SQLChatMessageHistory = MagicMock
_lc_comm.chat_message_histories = _lc_hist
