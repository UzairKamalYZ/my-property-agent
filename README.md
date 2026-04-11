# My Property Agent

A conversational multi-agent AI system for property search and stock analysis. Users describe what they need in natural language and the system routes their request to the right specialist agent — property search (RAG-powered), stock analysis (live data via MCP tools), or general conversation. All clients — REST API, Streamlit UI, Telegram bot, and scheduled cron job — share a single `OrchestratorAgent` facade.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Package Layout](#package-layout)
- [Multi-Agent Orchestration](#multi-agent-orchestration)
- [Specialist Agents](#specialist-agents)
- [Core Infrastructure](#core-infrastructure)
- [MCP Tool Integration](#mcp-tool-integration)
- [Client Layer](#client-layer)
- [Data Ingestion](#data-ingestion)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Running the Clients](#running-the-clients)
- [Adding an MCP Server](#adding-an-mcp-server)
- [Adding a New Agent](#adding-a-new-agent)
- [Testing](#testing)
- [LangSmith Observability](#langsmith-observability)

---

## Architecture Overview

```
User (REST / Streamlit / Telegram / Cron)
  │
  ▼
OrchestratorAgent           ← single facade — all clients talk to this
  │
  ▼
MultiAgentGraph             ← LangGraph supervisor multi-agent graph
  │
  ├─► supervisor            ← LLM decides which agent to call next
  │       │
  │       ├─► conversational_agent  ← greetings, small talk
  │       │         │
  │       │         └─► supervisor  (loop back)
  │       │
  │       ├─► property_agent        ← RAG: vector search → listings
  │       │         │
  │       │         └─► supervisor  (loop back)
  │       │
  │       └─► finance_agent         ← live stock data via Yahoo Finance MCP
  │                 │
  │                 └─► supervisor  (loop back)
  │
  └─► synthesiser           ← merges all agent outputs into one answer
          │
          ▼
        END
```

Every user message enters at the supervisor. The supervisor LLM reads the user's intent and the agent outputs collected so far, then decides which specialist to invoke next or emits `FINISH`. Once finished, the synthesiser combines all outputs into a clean, deduplicated reply.

---

## Package Layout

```
my-property-agent/
├── .env                              # Runtime configuration (project root)
├── mcp.json                          # MCP server registry
├── agents.json                       # Agent registry (name, class, model, RAG flag)
├── pyproject.toml                    # Project metadata and dependencies
├── logging_config.py                 # Central logging → logs/agent.log
│
├── core/                             # Shared AI infrastructure (all agents import from here)
│   └── src/
│       ├── base_agent.py             # BaseAgent ABC — Template Method pattern
│       ├── utils.py                  # load_prompt() file loader
│       ├── config/
│       │   └── config.py             # Settings loaded from .env (absolute path)
│       ├── model/
│       │   ├── llm_model_graph.py    # LangGraph pipeline (retrieve → generate)
│       │   ├── llm_factory.py        # LLM provider factory (Ollama / OpenAI-compat)
│       │   ├── embedder.py           # SentenceTransformer + vector search
│       │   ├── rag_context_manager.py# Vector search → formatted context string
│       │   ├── context_builder.py    # Formats listing dicts as readable text
│       │   ├── session_manager.py    # SQLite-backed session history
│       │   ├── mcp_registry.py       # MCPProcess, MCPHttpProcess, MCPRegistry
│       │   └── mcp_tools.py          # Loads mcp.json → shared _mcp registry
│       └── persistence/
│           ├── vector_store.py       # Abstract base (add_vectors / search)
│           ├── factory.py            # Selects FAISS or Pinecone at runtime
│           ├── faiss_store.py        # Local CPU-based FAISS store
│           └── pinecone_store.py     # Cloud Pinecone store
│
├── agents/                           # Domain-specific agent implementations
│   ├── conversational/
│   │   ├── agent.py                  # ConversationalAgent(BaseAgent)
│   │   └── prompts/System_Prompt.txt
│   ├── property/
│   │   ├── agent.py                  # PropertySearchAgent(BaseAgent) — RAG
│   │   ├── config.py                 # PropertyConfig (prompt path, Pinecone index)
│   │   ├── housing/                  # CSV → embedding-ready text pipeline
│   │   ├── gatherers/                # URL + Polish housing data ingestion
│   │   ├── scraping/                 # HTTP (BeautifulSoup) + Selenium scrapers
│   │   └── prompts/
│   │       ├── System_Prompt.txt
│   │       └── reformulated_prompt.txt
│   └── finance/
│       ├── agent.py                  # FinanceAgent(BaseAgent) — stocks MCP only
│       ├── config.py                 # FinanceConfig (prompt path)
│       └── prompts/System_Prompt.txt
│
├── orchestrator/                     # Multi-agent supervisor graph
│   ├── agent_interface.py            # OrchestratorAgent — public facade
│   ├── agent_registry.py             # Reads agents.json → live agent instances
│   ├── graph.py                      # MultiAgentGraph — builds + compiles graph
│   ├── state.py                      # GraphState TypedDict, MAX_AGENT_TURNS
│   ├── nodes/
│   │   ├── supervisor.py             # SupervisorNode — routes between agents
│   │   ├── agent.py                  # AgentNode — wraps a BaseAgent instance
│   │   └── synthesiser.py            # SynthesiserNode — merges outputs
│   └── prompts/
│       ├── supervisor.txt            # Routing rules + agent list placeholder
│       └── synthesiser.txt           # Merge/deduplicate instructions
│
├── clients/                          # Delivery interfaces
│   ├── base.py                       # BaseClient ABC (start / stop)
│   ├── rest/main.py                  # FastAPI REST API (port 8000)
│   ├── streamlit/main.py             # Streamlit web UI (port 8501)
│   ├── telegram/main.py              # Telegram long-polling bot
│   └── cron/main.py                  # Scheduled search (every 30 min)
│
└── logs/
    └── agent.log                     # Unified runtime log (all clients write here)
```

---

## Multi-Agent Orchestration

### How the supervisor graph works

The supervisor graph is a LangGraph `StateGraph` compiled once at startup inside `MultiAgentGraph.__init__`. Every user request is a fresh invocation of this graph.

**State** (`orchestrator/state.py`):

```python
class GraphState(TypedDict):
    user_input: str
    session_id: str
    messages: Annotated[list, add_messages]  # append-only
    agent_outputs: dict[str, str]            # name → answer collected so far
    next: str                                # routing token from supervisor
    turns: int                               # number of agent calls this turn
```

**Graph topology** (`orchestrator/graph.py`):

```
START → supervisor → [conditional edge] → agent_X → supervisor → ... → synthesiser → END
```

The conditional edge logic (the *router*) runs after every supervisor call:

| Condition | Routes to |
|---|---|
| `turns >= MAX_AGENT_TURNS` (6) | `synthesiser` (loop guard) |
| supervisor says an agent name AND that agent hasn't run yet | that agent |
| supervisor says an agent that already ran | `synthesiser` (prevents re-call loops) |
| supervisor says `FINISH` or anything else | `synthesiser` |

### Supervisor node (`orchestrator/nodes/supervisor.py`)

The supervisor is an LLM call that reads the current state and emits JSON:

```json
{"next": "property_agent"}   // call this agent next
{"next": "FINISH"}           // enough information, go to synthesiser
```

Routing rules (from `orchestrator/prompts/supervisor.txt`):
1. Greeting / small talk → `conversational_agent` → FINISH immediately
2. Property listings query and property agent hasn't run → `property_agent`
3. Finance / stocks query and finance agent hasn't run → `finance_agent`
4. Agent already produced satisfactory output → FINISH
5. When in doubt → FINISH (never over-call)

The supervisor's output is validated against the known agent names. Malformed responses are fuzzy-matched (e.g. `"property_agent | FINISH"` → `property_agent`) before defaulting to `FINISH`.

### Agent node (`orchestrator/nodes/agent.py`)

Wraps a `BaseAgent` instance. When invoked:
1. Calls `agent.ask(user_input, session_id=session_id)`
2. Stores result in `state["agent_outputs"][agent_name]`
3. Increments `state["turns"]`

### Synthesiser node (`orchestrator/nodes/synthesiser.py`)

Merges all agent outputs into one clean answer.

- **Single-agent passthrough**: if only one agent ran, the synthesiser returns that output directly without an extra LLM call.
- **Multi-agent synthesis**: injects all collected outputs as context and calls the LLM to write a fresh, deduplicated, markdown-formatted reply.

---

## Specialist Agents

All agents extend `BaseAgent` (`core/src/base_agent.py`) using the **Template Method** pattern. `BaseAgent.__init__` calls three hook methods that each subclass overrides:

```python
class BaseAgent(ABC):
    def get_system_prompt(self) -> str: ...      # load your prompt file
    def get_rag_context_manager(self): ...       # RAG manager or NullRagContextManager
    def get_mcp_tools(self) -> list | None: ...  # None=all tools, []=none, or filtered list
```

Agents are registered in **`agents.json`** at the project root. The orchestrator reads this file at startup — no code changes are needed to add, remove, or reconfigure an agent.

---

### Conversational Agent

**Class**: `agents.conversational.agent.ConversationalAgent`

**Role**: Handles greetings, small talk, and off-topic questions. Keeps the conversation friendly without invoking RAG or tools.

**Configuration**:
- RAG: disabled (`NullRagContextManager`)
- MCP tools: none (returns `[]`)
- System prompt: warm, brief responses; redirects property/finance questions to the right agent

**Flow**:
```
User: "Hi there!"
  │
  ▼
supervisor → conversational_agent
  │
  ▼
LlmModelGraph (retrieve skips vector search, generate answers directly)
  │
  ▼
"Hello! Great to have you here. How can I help you find your perfect property today?"
  │
  ▼
supervisor → FINISH → synthesiser (passthrough)
```

---

### Property Search Agent

**Class**: `agents.property.agent.PropertySearchAgent`

**Role**: Finds property listings that match the user's natural-language criteria. Only presents data that exists in the vector store — never invents or estimates.

**Configuration**:
- RAG: enabled (`RagContextManager` + `Embedder`)
- MCP tools: default (all registered MCP tools — e.g. currency conversion if available)
- Vector store: FAISS (local) or Pinecone (cloud), configured via `STORE_TYPE`
- System prompt: strict data scope — if no matching listings in context, says so honestly

**Internal pipeline** (`core/src/model/llm_model_graph.py`):

```
user_prompt
  │
  ▼
[retrieve node]
  Embedder.embed_query(user_prompt)
  → VectorStore.search(query_vector, k=RAG_K)
  → RagContextManager formats results as readable text
  → state["context"] = "Property 1: ...\nProperty 2: ..."
  │
  ▼
[generate node]
  SystemMessage(system_prompt)
  SystemMessage("Relevant listings:\n{context}")    ← only when context is non-empty
  SystemMessage("No matching listings found...")    ← only when context empty AND rag_enabled
  + session_history + HumanMessage(user_prompt)
  │
  LLM invoked → may call MCP tools (loop up to MAX_TOOL_CALLS=5)
  │
  ▼
answer string
```

**Tool call loop** (inside `generate` node):

```
LLM response
  │
  ├─ has tool_calls? ──► invoke each tool → append ToolMessage → re-invoke LLM
  │                                                    ↑
  │                       (repeat up to MAX_TOOL_CALLS=5 times)
  │
  └─ no tool_calls → break → return response.content as answer
```

**Data ingestion**:

Properties are pre-indexed into the vector store offline. Two ingestion methods are provided:

```bash
# From CSV files (Polish housing data)
python -m agents.property.housing.housing_data_collector

# From URLs (web scraping)
python -m agents.property.gatherers.data_collector
```

---

### Finance Agent

**Class**: `agents.finance.agent.FinanceAgent`

**Role**: Stock analysis and investment advice using live market data fetched via MCP tools. Strictly scoped to stocks and equities — will not handle currency conversion or mortgages.

**Configuration**:
- RAG: disabled (`NullRagContextManager`)
- MCP tools: `stocks` server only (`_mcp.langchain_tools(server_name="stocks")`)
- MCP server: `@fre4x/yahoo-finance` — Yahoo Finance via stdio subprocess
- System prompt: always fetch live data first; present price, P/E, EPS, news sentiment, recommendation

**Available tools** (from Yahoo Finance MCP):

| Tool | Description |
|---|---|
| `yfin_get_quotes` | Real-time price quotes for one or more tickers |
| `yfin_get_historical_prices` | OHLCV data for a ticker over a time period |
| `yfin_get_stock_info` | Full profile: price, market cap, fundamentals |
| `yfin_get_news` | Latest news articles for a ticker |
| `yfin_get_financial_statement` | Income statement, balance sheet, or cashflow |
| `yfin_get_holder_info` | Major, institutional, mutual fund, or insider holders |
| `yfin_get_option_dates` | Available option expiration dates |
| `yfin_get_option_chain` | Calls or puts chain for a given expiry |
| `yfin_get_recommendations` | Analyst recommendations and upgrades/downgrades |
| `yfin_get_stock_actions` | Dividend and split history |

**Flow**:
```
User: "What's the outlook for AAPL?"
  │
  ▼
supervisor → finance_agent
  │
  ▼
LlmModelGraph (retrieve skips — NullRagContextManager returns "")
  │
  ▼
generate node: LLM sees system prompt + user question
  LLM calls yfin_get_stock_info(ticker="AAPL")
  → MCP subprocess returns price, P/E, EPS, market cap
  LLM calls yfin_get_news(ticker="AAPL")
  → MCP subprocess returns recent headlines + sentiment
  LLM synthesises: "AAPL is trading at $X. P/E is Y. Recent news is..."
  │
  ▼
supervisor → FINISH → synthesiser (passthrough)
```

---

## Core Infrastructure

### BaseAgent (`core/src/base_agent.py`)

```python
class BaseAgent(ABC):
    def __init__(self, session_id=None, rag_enabled=False,
                 llm_provider=None, llm_model=None):
        # Calls get_system_prompt(), get_rag_context_manager(), get_mcp_tools()
        # Builds LlmModelGraph with the returned values
        ...

    # --- Hooks — override in every subclass ---
    @abstractmethod
    def get_system_prompt(self) -> str: ...

    @abstractmethod
    def get_rag_context_manager(self): ...

    @abstractmethod
    def get_mcp_tools(self) -> list | None: ...

    # --- Public interface — inherited as-is ---
    def ask(self, prompt, stream=False, session_id=None): ...
    def close(self): ...
```

`NullRagContextManager` (also in `base_agent.py`) is the no-op placeholder returned by agents that don't need vector search. It always returns `""` from `get_context()`.

### LangGraph pipeline (`core/src/model/llm_model_graph.py`)

Two-node graph: `retrieve → generate`.

**Node: retrieve**
- Calls `rag_context_manager.get_context(user_prompt)`
- For RAG-enabled agents: embeds the query, searches the vector store, returns formatted listings
- For non-RAG agents (`NullRagContextManager`): returns `""` instantly — no vector search

**Node: generate**
- Builds the message list: system prompt → (context if present) → session history → user message
- Calls `llm.invoke(messages)` — LLM may request tool calls
- Loops up to `MAX_TOOL_CALLS = 5`, executing each tool and appending `ToolMessage` results
- Returns `response.content` as `state["answer"]`

**Session history**: kept per `session_id` in memory (dict of lists). Sliding window: last `MAX_HISTORY_TURNS * 2` messages.

**LangSmith tracing**: every `ask()` call is wrapped in `@traceable(name=f"agent.{agent_name}")`. Tool invocations are wrapped in `@traceable(name=f"tool.{tool_name}")`.

### Embedder (`core/src/model/embedder.py`)

Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional float32 vectors).

| Method | Purpose |
|---|---|
| `embed_query(text)` | Single query → numpy vector |
| `embed_texts(texts)` | Batch of texts → numpy array |
| `embed_documents_to_vectors(docs)` | Dict of documents → list of vector dicts |
| `search(query, k=5)` | End-to-end: embed + vector store search |
| `save_vectors_in_store(vectors)` | Persist vectors to the configured store |

### Vector stores (`core/src/persistence/`)

| Backend | Class | When to use |
|---|---|---|
| FAISS | `FAISSStore` | Local development, no cloud dependencies |
| Pinecone | `PineconeStore` | Production; auto-creates index if absent |

Switch between them with `STORE_TYPE=local` or `STORE_TYPE=pinecone` in `.env`.

---

## MCP Tool Integration

Tools are declared in **`mcp.json`** at the project root. No Python changes are needed to add a server.

```json
{
  "servers": [
    {"name": "finance", "command": ["npx", "--yes", "@easysolutions906/mcp-finance"]},
    {"name": "stocks",  "command": ["npx", "--yes", "@fre4x/yahoo-finance"]}
  ]
}
```

At startup, `mcp_tools.py` reads this file and registers each server in the `MCPRegistry`. Two transport types are supported:

| Type | Config key | Class |
|---|---|---|
| stdio subprocess | `"command": [...]` | `MCPProcess` |
| HTTP JSON-RPC | `"url": "..."` | `MCPHttpProcess` |

**How it works**:
1. `MCPRegistry` stores one `MCPProcess` (or `MCPHttpProcess`) per server name.
2. `langchain_tools(server_name=None)` calls `tools/list` on each server and auto-generates LangChain `StructuredTool` objects from the returned JSON schemas.
3. The LLM is bound to these tools via `llm.bind_tools(tools)`.
4. The `generate` node inspects `response.tool_calls` and routes execution to the right tool.

**Per-agent tool filtering**: agents can restrict which tools they receive:

```python
# FinanceAgent only gets tools from the "stocks" server
def get_mcp_tools(self) -> list:
    return _mcp.langchain_tools(server_name="stocks")

# PropertySearchAgent gets all tools (or pass [] for none)
def get_mcp_tools(self) -> list | None:
    return None   # None = all registered tools
```

**JSON-encoded argument coercion**: some LLMs emit array/object parameters as JSON strings (e.g. `tickers: "[\"AAPL\"]"` instead of `tickers: ["AAPL"]`). `LlmModelGraph._coerce_args()` detects and deserializes these before invoking the tool.

---

## Client Layer

All clients implement `BaseClient` (`clients/base.py`):

```python
class BaseClient(ABC):
    @abstractmethod
    def start(self) -> None: ...   # blocks until stopped
    def stop(self) -> None: ...    # graceful shutdown
    # + __enter__ / __exit__ context manager
```

| Client | Entry point | Transport | Notes |
|---|---|---|---|
| `RestClient` | `clients/rest/main.py` | FastAPI + SSE (port 8000) | Optional `X-API-Key` auth |
| `StreamlitClient` | `clients/streamlit/main.py` | Web UI (port 8501) | Requires REST API running first |
| `TelegramClient` | `clients/telegram/main.py` | Telegram long-polling | Per-chat session IDs |
| `CronClient` | `clients/cron/main.py` | Scheduled loop | Fires `CRON_SEARCH_PROMPT` every 30 min |

All four use `OrchestratorAgent` as their backend. The Telegram client runs blocking `agent.ask()` calls in a thread executor to avoid blocking the async event loop.

### REST API endpoints

| Method | Path | Params | Response |
|---|---|---|---|
| `GET` | `/ask` | `prompt` (str), `stream` (bool, default `false`) | `{"response": str}` or SSE stream |

Streaming format (SSE): `data: {chunk}\n\n` with 10 ms delay between chunks.

---

## Data Ingestion

### CSV (Polish housing data)

```bash
# Place CSV files in agents/property/datasets/pl-housing/
python -m agents.property.housing.housing_data_collector
```

Streams the CSV in 5 000-row chunks, converts each row to a semantic text document, embeds in batches of 100, and persists to the configured vector store.

### From URLs

```bash
# Edit agents/property/urls.txt — one URL per line
python -m agents.property.gatherers.data_collector
```

Fetches each URL with BeautifulSoup (or Selenium for anti-bot sites), extracts listing data, and indexes it into the vector store.

---

## Configuration

All settings live in `.env` at the **project root**. `core/src/config/config.py` loads it via an absolute path derived from `__file__` — works regardless of working directory.

### Core LLM

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM backend (`ollama` is the supported provider) |
| `LLM_MODEL_NAME` | `llama3.2` | Model name passed to the endpoint |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output |
| `LLM_SEED` | `365` | Reproducibility seed |
| `AI_PROVIDER_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `AI_PROVIDER_API_KEY` | `ollama` | API key for the endpoint |

### Vector store

| Variable | Default | Description |
|---|---|---|
| `STORE_TYPE` | `pinecone` | `local` (FAISS) or `pinecone` |
| `SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RAG_K` | `5` | Listings returned per vector search |
| `PINECONE_API_KEY` | _(required for cloud)_ | Pinecone API key |
| `PINECONE_INDEX_NAME` | `property-agent` | Pinecone index name |

### Paths

| Variable | Default | Description |
|---|---|---|
| `PROMPT_FILE` | `agents/property/prompts/System_Prompt.txt` | Property agent system prompt |
| `SUPERVISOR_PROMPT_FILE` | `orchestrator/prompts/supervisor.txt` | Supervisor routing prompt |
| `SYNTHESISER_PROMPT_FILE` | `orchestrator/prompts/synthesiser.txt` | Synthesiser merge prompt |
| `AGENTS_FILE` | `agents.json` | Agent registry |
| `MCP_FILE` | `mcp.json` | MCP server registry |
| `SESSION_DB_FILE` | `sessions.db` | SQLite session history |
| `MAX_HISTORY_TURNS` | `10` | Sliding window depth |

### Clients

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | _(required)_ | Token from [@BotFather](https://t.me/BotFather) |
| `API_KEY` | _(empty — auth disabled)_ | REST API key; set to enforce `X-API-Key` header |
| `CRON_SEARCH_PROMPT` | _(2-bed Poland <1000 PLN)_ | Prompt used by the scheduled job |

### Observability

| Variable | Default | Description |
|---|---|---|
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | _(empty)_ | LangSmith API key |
| `LANGCHAIN_PROJECT` | `my-property-agent` | LangSmith project name |

---

## Getting Started

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai/) installed and running
- Node.js + `npx` (for MCP servers)
- Pinecone account, or set `STORE_TYPE=local` for offline FAISS

### Installation

```bash
git clone https://github.com/your-username/my-property-agent.git
cd my-property-agent
uv venv && uv sync
```

Pull the model:

```bash
ollama pull llama3.2
```

Configure the environment:

```bash
cp .env.template .env
# Edit .env with your Pinecone key, Telegram token, etc.
```

---

## Running the Clients

All commands are run from the **project root** using the `-m` flag so that all packages are importable.

### REST API

```bash
python -m clients.rest.main
# Listening on http://localhost:8000
```

```bash
# Blocking
curl "http://localhost:8000/ask?prompt=3+bed+Warsaw"

# Streaming (SSE)
curl "http://localhost:8000/ask?prompt=3+bed+Warsaw&stream=true"
```

### Streamlit UI

Requires the REST API to be running first.

```bash
python -m clients.streamlit.main
# Opens http://localhost:8501
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

---

## Adding an MCP Server

Edit `mcp.json` and restart. No Python changes required.

**stdio server** (subprocess):
```json
{"name": "my_server", "command": ["npx", "--yes", "@acme/my-mcp-server"]}
```

**HTTP server** (remote JSON-RPC):
```json
{
  "name": "my_api",
  "url": "https://api.example.com/mcp",
  "api_key_env": "MY_API_KEY",
  "api_key_param": "apikey"
}
```

Tools are auto-discovered via `tools/list` and bound to the LLM at startup. System prompts never hardcode tool names — the LLM learns them from the schema.

---

## Adding a New Agent

1. Create the agent class:

```python
# agents/myagent/agent.py
from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.utils import load_prompt

class MyAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return load_prompt("agents/myagent/prompts/System_Prompt.txt")

    def get_rag_context_manager(self):
        return NullRagContextManager()   # or RagContextManager(Embedder())

    def get_mcp_tools(self) -> list | None:
        return []   # [], None (all tools), or filtered list
```

2. Add a system prompt at `agents/myagent/prompts/System_Prompt.txt`.

3. Register in `agents.json`:

```json
{
  "name": "my_agent",
  "class": "agents.myagent.agent.MyAgent",
  "description": "Handles questions about X. Route here when the user asks about X.",
  "enabled": true,
  "rag": false,
  "llm_provider": "ollama",
  "llm_model": "llama3.2"
}
```

The supervisor will automatically route to your agent based on its description. No changes to `core/`, `orchestrator/`, or any client are needed.

---

## Testing

```bash
# All tests
python -m pytest -v

# By layer
python -m pytest core/tests/ -v
python -m pytest clients/tests/ -v

# With coverage
python -m pytest --cov=core/src --cov=agents --cov=orchestrator --cov=clients --cov-report=term-missing
```

All external dependencies (LLM calls, vector store, network, Telegram API, MCP subprocesses) are mocked. No real services are required to run the test suite.

---

## LangSmith Observability

Set in `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
LANGCHAIN_PROJECT=my-property-agent
```

**Trace hierarchy** for a typical property search:

```
orchestrator.request
  └─ agent.PropertySearchAgent        ← @traceable on LlmModelGraph.ask()
       ├─ retrieve                    ← LangGraph node
       └─ generate                    ← LangGraph node
            └─ tool.yfin_get_quotes   ← @traceable on each tool invocation
```

Traces are visible at [smith.langchain.com](https://smith.langchain.com) under the configured project name.
