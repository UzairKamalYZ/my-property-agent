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

import sys
from pathlib import Path
from unittest.mock import MagicMock

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
