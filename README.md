# My Property Agent

A conversational AI system for property search. Users describe what they're looking for in natural language and the agent retrieves relevant listings via a **two-node LangGraph RAG pipeline** backed by a FAISS or Pinecone vector store. The system exposes multiple client interfaces — REST API, Streamlit web UI, Telegram bot, and a scheduled cron job — all built on a shared `BaseClient` interface.

---

## Features

- **Shared agent core** — `BaseAgent` in `core/` owns LLM creation, the LangGraph pipeline, tool calling, and session history; domain agents extend it with one prompt override
- **Minimal LangGraph pipeline** — two nodes, no conditional edges: `retrieve → generate`
- **MCP tool integration** — add any MCP server by dropping a line in `mcp.json`; tools are auto-discovered and bound to the LLM at startup
- **Streaming responses** — token-by-token output via SSE (REST) or live message edits (Telegram)
- **Conversation memory** — per-session sliding-window history threaded through every graph invocation
- **Multiple clients** — REST API, Streamlit UI, Telegram bot, scheduled cron job
- **Pluggable vector store** — FAISS (local) or Pinecone (cloud), switched via config
- **LangSmith observability** — every run traced with `@traceable`
- **Web scraping** — ingest listings from URLs or CSV files
- **133 tests** — TDD-style naming (`test_should_<result>_when_<condition>`) across all layers

---

## Architecture

### Package layout

```
my-property-agent/
├── core/           ← shared AI infrastructure (all agents import from here)
├── agentP/         ← property-search agent (property-specific code only)
├── agentF/         ← finance Q&A agent
└── clients/        ← REST, Streamlit, Telegram, Cron
```

### Request flow

```
User (any client)
  │
  ▼
LocalAgent.ask(prompt)          ← agentP — property-search agent
  │  (or FinanceAgent.ask())    ← agentF — finance Q&A agent
  ▼
BaseAgent                       ← core — owns the pipeline
  │
  ▼
LlmModelGraph
  ├── [retrieve]  user_prompt → vector search → context (formatted listings)
  └── [generate]  system prompt + context + history + user_prompt → answer
                   └── tool call loop (MCP tools bound at startup)
  │
  ▼
Response (blocking string or token stream)
```

### BaseAgent (`core/src/base_agent.py`)

All domain agents extend `BaseAgent`, which owns every shared capability:

```python
class BaseAgent:
    def get_system_prompt(self) -> str: ...      # override with your prompt
    def get_rag_context_manager(self) -> ...: ... # override for custom RAG
    def ask(prompt, stream, session_id): ...
    def close(): ...
```

Adding a new agent:

```python
class MyAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return LlmModelGraph._load_file("path/to/my_prompt.txt")
```

That's all. `ask`, `stream`, `close`, context manager, session history, MCP tools — all inherited.

### LangGraph pipeline (`core/src/model/llm_model_graph.py`)

```
START → retrieve → generate → END
```

| Node | What it does |
|---|---|
| `retrieve` | Embeds `user_prompt`, runs cosine similarity search against the vector store, returns formatted listing context |
| `generate` | Sends system prompt + context + session history + original question to the LLM; runs a tool-call loop until the model produces a plain-text answer |

**State**

```python
class State(TypedDict):
    user_prompt: str
    context: str
    answer: str
    session_history: List[AnyMessage]
```

**Why two nodes?** Vector search costs ~1 ms. A dedicated LLM classify or reformulate node costs ~1 s each with no meaningful quality gain — `all-MiniLM-L6-v2` handles natural-language queries directly.

### MCP tool integration

Tools are declared in **`mcp.json`** at the project root — no code changes needed to add a server:

```json
{
  "servers": [
    {"name": "finance", "command": ["npx", "--yes", "@easysolutions906/mcp-finance"]}
  ]
}
```

At startup, `mcp_tools.py` reads this file and registers each server. Subprocesses start lazily on first use. All discovered tools are bound to the LLM via `llm.bind_tools()`. System prompts never name specific tool names — the LLM learns them from the bound schema.

### Client layer (`clients/`)

All clients implement `BaseClient`:

```python
class BaseClient(ABC):
    @abstractmethod
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

| Client | Entry point | Transport |
|---|---|---|
| `RestClient` | `clients/rest/main.py` | FastAPI + SSE (port 8000) |
| `StreamlitClient` | `clients/streamlit/main.py` | Streamlit web UI (port 8501) |
| `TelegramClient` | `clients/telegram/main.py` | Telegram long-polling |
| `CronClient` | `clients/cron/main.py` | Scheduled loop (every 30 min) |

---

## Repository structure

```
my-property-agent/
├── .env                              # Runtime config (see Configuration)
├── mcp.json                          # MCP server registry
├── pyproject.toml                    # Project metadata and dependencies
├── logging_config.py                 # Central logging setup (logs/agent.log)
│
├── core/                             # Shared AI agent infrastructure
│   ├── src/
│   │   ├── base_agent.py             # BaseAgent — extended by all domain agents
│   │   ├── config/config.py          # Centralised settings loaded from .env
│   │   ├── model/
│   │   │   ├── llm_model_graph.py    # LangGraph pipeline (retrieve → generate)
│   │   │   ├── llm_factory.py        # Creates the LLM (Ollama via OpenAI-compat API)
│   │   │   ├── embedder.py           # SentenceTransformer embeddings
│   │   │   ├── rag_context_manager.py# Vector search → formatted context
│   │   │   ├── context_builder.py    # Formats listing dicts as readable text
│   │   │   ├── session_manager.py    # SQLite-backed session history
│   │   │   ├── mcp_registry.py       # Generic MCPProcess + MCPRegistry
│   │   │   └── mcp_tools.py          # Loads mcp.json, exposes shared _mcp registry
│   │   └── persistence/
│   │       ├── vector_store.py       # Abstract base (add / search)
│   │       ├── factory.py            # Selects FAISS or Pinecone at runtime
│   │       ├── faiss_store.py        # Local FAISS implementation
│   │       └── pinecone_store.py     # Cloud Pinecone implementation
│   └── tests/
│       ├── conftest.py
│       ├── test_base_agent.py
│       └── model/
│           ├── test_embedder.py
│           ├── test_llm_model_graph.py
│           └── test_mcp_tools.py
│
├── agentP/                           # Property-search agent
│   ├── src/
│   │   ├── agent.py                  # LocalAgent(BaseAgent) — adds WebScraper + CLI
│   │   ├── housing/                  # CSV → embedding-ready text
│   │   ├── gatherers/                # Data ingestion (web + Polish housing)
│   │   ├── scraping/                 # HTTP + Selenium scrapers
│   │   └── prompts/
│   │       ├── System_Prompt.txt     # Property agent persona
│   │       └── interaction.json      # CLI strings
│   ├── tests/
│   └── urls.txt
│
├── agentF/                           # Finance Q&A agent
│   ├── src/
│   │   ├── agent.py                  # FinanceAgent(BaseAgent) — overrides get_system_prompt()
│   │   ├── config/config.py          # FinanceConfig (FINANCE_PROMPT_FILE, FINANCE_PINECONE_INDEX_NAME)
│   │   └── prompts/
│   │       └── System_Prompt.txt     # Finance agent persona
│   └── tests/
│       └── model/test_finance_agent.py
│
├── clients/
│   ├── base.py                       # BaseClient ABC
│   ├── rest/main.py                  # FastAPI app + RestClient
│   ├── streamlit/main.py             # Streamlit UI + StreamlitClient
│   ├── telegram/main.py              # Telegram bot + TelegramClient
│   ├── cron/main.py                  # Scheduled search + CronClient
│   └── tests/
│
└── logs/
    └── agent.log                     # Unified runtime log (all clients)
```

---

## Configuration

All settings live in `.env` at the project root. Loaded automatically by `core/src/config/config.py` via an absolute path — works regardless of where the process is launched from.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM backend (`ollama` is the only supported provider) |
| `LLM_MODEL_NAME` | `llama3.2` | Model name passed to the LLM endpoint |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output |
| `LLM_SEED` | `365` | Reproducibility seed |
| `AI_PROVIDER_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible LLM endpoint |
| `AI_PROVIDER_API_KEY` | `ollama` | API key for the LLM endpoint |
| `STORE_TYPE` | `pinecone` | `local` (FAISS) or `pinecone` |
| `SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RAG_K` | `5` | Listings returned per vector search |
| `PINECONE_API_KEY` | _(required for cloud)_ | Pinecone API key |
| `PINECONE_INDEX_NAME` | `property-agent` | Pinecone index name |
| `PROMPT_FILE` | `agentP/src/prompts/System_Prompt.txt` | Property agent system prompt path |
| `INTERACTION_FILE` | `agentP/src/prompts/interaction.json` | CLI interaction strings path |
| `SESSION_DB_FILE` | `sessions.db` | SQLite file for chat session history |
| `MAX_HISTORY_TURNS` | `10` | Sliding window depth for session history |
| `TELEGRAM_BOT_TOKEN` | _(required for Telegram)_ | Token from [@BotFather](https://t.me/BotFather) |
| `API_KEY` | _(empty — auth disabled)_ | REST API key; set to enforce `X-API-Key` header |
| `CRON_SEARCH_PROMPT` | _(2-bed Poland <1000)_ | Prompt used by the scheduled job |
| `SBR_WEBDRIVER` | _(empty)_ | Selenium WebDriver URL for `scrape_se.py` |
| `FINANCE_PROMPT_FILE` | `agentF/src/prompts/System_Prompt.txt` | Finance agent system prompt path |
| `FINANCE_PINECONE_INDEX_NAME` | _(falls back to PINECONE_INDEX_NAME)_ | Separate Pinecone index for finance agent |
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | _(empty)_ | LangSmith API key |
| `LANGCHAIN_PROJECT` | `my-property-agent` | LangSmith project name |

---

## Getting started

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai/) installed and running
- Node.js (for MCP servers via `npx`)
- Pinecone account (or set `STORE_TYPE=local` for offline FAISS)

### Installation

```bash
git clone https://github.com/your-username/my-property-agent.git
cd my-property-agent
uv venv && uv sync
```

Pull the LLM model:

```bash
ollama pull llama3.2
```

Edit the environment file at the project root:

```bash
cp .env.template .env   # or edit .env directly
```

---

## Running the clients

All commands are run from the **project root**.

### REST API

```bash
python -m clients.rest.main
# Runs on http://localhost:8000
```

| Method | Path | Params | Response |
|---|---|---|---|
| GET | `/ask` | `prompt` (str), `stream` (bool) | `{"response": str}` or SSE stream |

```bash
# Blocking
curl "http://localhost:8000/ask?prompt=3+bed+Warsaw"

# Streaming (SSE)
curl "http://localhost:8000/ask?prompt=3+bed+Warsaw&stream=true"
```

When `API_KEY` is set in `.env`, every request must include `X-API-Key: <key>`.

### Streamlit UI

Requires the REST API to be running first.

```bash
streamlit run clients/streamlit/main.py
# Opens at http://localhost:8501
```

### Telegram bot

```bash
python -m clients.telegram.main
```

### Cron job

```bash
python -m clients.cron.main
# Fires CRON_SEARCH_PROMPT every 30 minutes
```

### Finance agent (CLI)

```bash
python -m agentF.src.agent
```

---

## Adding an MCP server

Edit `mcp.json` at the project root and restart:

```json
{
  "servers": [
    {"name": "finance", "command": ["npx", "--yes", "@easysolutions906/mcp-finance"]},
    {"name": "maps",    "command": ["npx", "--yes", "@acme/mcp-maps"]}
  ]
}
```

No Python changes required. Tools are auto-discovered via `tools/list` and bound to the LLM at startup.

---

## Adding a new domain agent

1. Create `agentX/src/agent.py`:

```python
from core.src.base_agent import BaseAgent
from core.src.model.llm_model_graph import LlmModelGraph

class MyAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return LlmModelGraph._load_file("agentX/src/prompts/System_Prompt.txt")
```

2. Add a system prompt at `agentX/src/prompts/System_Prompt.txt`.
3. Run with `python -m agentX.src.agent`.

No changes to `core/`, `agentP/`, or any client required.

---

## Data ingestion

### Polish housing CSV data

Place CSV files in `agentP/datasets/pl-housing/`, then:

```bash
python -m agentP.src.gatherers.pl_housing_data_collector
```

### From URLs

Add URLs to `agentP/urls.txt` (one per line), then:

```bash
python -m agentP.src.gatherers.data_collector
```

---

## Testing

```bash
# All tests (133 total)
python -m pytest

# Explicit paths
python -m pytest core/tests/ agentP/tests/ agentF/tests/ clients/tests/ -v

# With coverage
python -m pytest --cov=core/src --cov=agentP/src --cov=agentF/src --cov=clients --cov-report=term-missing
```

All external dependencies (LLM, vector store, network, Telegram, MCP subprocesses) are mocked. No real services are required to run the test suite.

---

## LangSmith observability

Set the following in `.env` to enable tracing:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
LANGCHAIN_PROJECT=my-property-agent
```

Every `ask()` and `ask_stream()` call is decorated with `@traceable` and produces a trace with two child spans — `retrieve` and `generate` — visible at [smith.langchain.com](https://smith.langchain.com).
