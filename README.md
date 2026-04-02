# My Property Agent

A conversational AI system for property search. Users describe what they're looking for in natural language and the agent retrieves relevant listings via a **LangGraph RAG pipeline** backed by a FAISS or Pinecone vector store. The system exposes multiple client interfaces — REST API, Streamlit web UI, Telegram bot, and a scheduled cron job — all built on a shared `BaseClient` interface.

---

## Features

- **LangGraph RAG Pipeline** — four-node graph with conditional retrieval: reformulate → classify → [retrieve →] generate
- **Conditional RAG** — an LLM classifier decides per-query whether to hit the vector store; greetings and follow-ups skip retrieval entirely
- **Streaming responses** — token-by-token output via SSE (REST) or live message edits (Telegram)
- **Conversation memory** — per-session history threaded through every graph invocation
- **Multiple clients** — REST API, Streamlit UI, Telegram bot, scheduled cron job
- **Pluggable vector store** — FAISS (local) or Pinecone (cloud), switched via config
- **LangSmith observability** — every run traced with `@traceable`
- **Web scraping** — ingest listings from URLs or CSV files
- **108 tests** — TDD-style naming (`test_should_<result>_when_<condition>`) across all layers

---

## Architecture

### Request flow

```
User (any client)
  │
  ▼
LocalAgent.ask(prompt, stream)
  │
  ▼
LlmModelGraph
  ├── [reformulate]  user_prompt → reformulated_question   (LLM)
  ├── [classify]     reformulated_question → needs_search  (LLM YES/NO)
  ├── [retrieve]     reformulated_question → context        (vector search, only when needs_search=True)
  └── [generate]     question + context + history → answer  (LLM)
  │
  ▼
Response (blocking string or token stream)
```

### LangGraph pipeline (`agentP/src/model/llm_model_graph.py`)

```
START → reformulate → classify ──(needs_search=True)──► retrieve → generate → END
                               └──(needs_search=False)──► generate → END
```

| Node | Input | Output |
|---|---|---|
| `reformulate` | `user_prompt` | `reformulated_question` |
| `classify` | `reformulated_question` | `needs_search` (bool) |
| `retrieve` | `reformulated_question` | `context` (formatted listings) |
| `generate` | `reformulated_question` + `context` + `session_history` | `answer` |

The `classify` node prompts the LLM for a plain YES/NO decision. Only queries about finding, renting, buying, or comparing properties route to `retrieve`. Greetings, follow-ups, and general questions go directly to `generate` — the vector store is never touched unnecessarily.

### Client layer (`clients/`)

All clients implement `BaseClient`:

```python
class BaseClient(ABC):
    @abstractmethod
    def start(self) -> None: ...  # blocks until stopped
    def stop(self) -> None: ...   # graceful shutdown (override if needed)
    # + context manager (__enter__ / __exit__)
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
├── .env                             # Runtime config (see Configuration)
├── pyproject.toml                   # Project metadata and dependencies
├── logging_config.py                # Central logging setup (logs/agent.log)
├── agentP/
│   ├── requirements.txt             # Pinned dependency lockfile
│   ├── urls.txt                     # URLs for web scraping
│   ├── src/
│   │   ├── agent.py                 # LocalAgent — thin facade over LlmModelGraph
│   │   ├── config/
│   │   │   └── config.py            # Centralised settings loaded from .env
│   │   ├── model/
│   │   │   ├── llm_factory.py       # Creates the LLM (Ollama via OpenAI-compatible API)
│   │   │   ├── llm_model_graph.py   # LangGraph RAG pipeline (4-node conditional graph)
│   │   │   ├── tools.py             # make_property_search_tool() factory
│   │   │   ├── embedder.py          # SentenceTransformer embeddings
│   │   │   ├── rag_context_manager.py  # Vector search → formatted context
│   │   │   ├── context_builder.py   # Formats listing dicts as readable text
│   │   │   └── session_manager.py   # SQLite-backed session history
│   │   ├── persistence/
│   │   │   ├── vector_store.py      # Abstract base (add / search)
│   │   │   ├── factory.py           # Selects FAISS or Pinecone at runtime
│   │   │   ├── faiss_store.py       # Local FAISS implementation
│   │   │   └── pinecone_store.py    # Cloud Pinecone implementation
│   │   ├── housing/
│   │   │   ├── housing_data_collector.py   # Streams CSV → embeddings
│   │   │   └── housing_csv_reader.py       # CSV parsing utility
│   │   ├── gatherers/
│   │   │   ├── data_collector.py           # Scrapes listing URLs
│   │   │   └── pl_housing_data_collector.py # Polish housing ingestion
│   │   ├── scraping/
│   │   │   ├── web_scraper.py       # HTTP fetch + HTML cleanup
│   │   │   ├── url_processor.py     # Regex-based listing extraction
│   │   │   ├── scrape_se.py         # Selenium-based scraper
│   │   │   └── utils.py             # Converts listings to LangChain Documents
│   │   └── prompts/
│   │       ├── System_Prompt.txt        # Agent persona and response style
│   │       ├── reformulated_prompt.txt  # Query rewrite template
│   │       └── interaction.json         # CLI interaction strings
│   └── tests/
│       ├── conftest.py              # sys.path fix + heavy-package stubs
│       └── model/
│           ├── test_embedder.py         # 17 tests
│           ├── test_llm_model_graph.py  # 32 tests
│           └── test_tools.py            # 11 tests
│
├── clients/
│   ├── base.py                      # BaseClient ABC
│   ├── rest/
│   │   ├── main.py                  # FastAPI app + RestClient
│   │   └── requirements.txt
│   ├── streamlit/
│   │   ├── main.py                  # Streamlit UI + StreamlitClient
│   │   └── requirements.txt
│   ├── telegram/
│   │   ├── main.py                  # Telegram bot + TelegramClient
│   │   └── requirements.txt
│   ├── cron/
│   │   ├── main.py                  # Scheduled search + CronClient
│   │   └── requirements.txt
│   └── tests/
│       ├── conftest.py              # agentP.src.agent stub + third-party stubs
│       ├── test_base_client.py      # 6 tests
│       ├── test_rest_client.py      # 12 tests
│       ├── test_cron_client.py      # 10 tests
│       ├── test_telegram_client.py  # 9 tests
│       └── test_streamlit_client.py # 11 tests
│
└── logs/
    └── agent.log                    # Unified runtime log (all clients)
```

---

## Configuration

All settings live in `.env` at the project root. Loaded automatically by `agentP/src/config/config.py` via an absolute path — works regardless of where the process is launched from.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM backend (`ollama` is the only supported provider) |
| `LLM_MODEL_NAME` | `llama3.2` | Model name passed to the LLM endpoint |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output |
| `LLM_SEED` | `365` | Reproducibility seed |
| `AI_PROVIDER_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible LLM endpoint (Ollama default) |
| `AI_PROVIDER_API_KEY` | `ollama` | API key for the LLM endpoint |
| `STORE_TYPE` | `pinecone` | `local` (FAISS) or `pinecone` |
| `SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RAG_K` | `5` | Number of listings to retrieve per vector search |
| `PINECONE_API_KEY` | _(required for cloud)_ | Pinecone API key |
| `PINECONE_INDEX_NAME` | `property-agent` | Pinecone index name |
| `PROMPT_FILE` | `agentP/src/prompts/System_Prompt.txt` | Agent system prompt path |
| `REFORMULATION_PROMPT` | `agentP/src/prompts/reformulated_prompt.txt` | Query rewrite template path |
| `INTERACTION_FILE` | `agentP/src/prompts/interaction.json` | CLI interaction strings path |
| `SESSION_DB_FILE` | `sessions.db` | SQLite file for chat session history |
| `TELEGRAM_BOT_TOKEN` | _(required for Telegram)_ | Token from [@BotFather](https://t.me/BotFather) |
| `API_KEY` | _(empty — auth disabled)_ | REST API key; set to enforce `X-API-Key` header |
| `CRON_SEARCH_PROMPT` | _(2-bed Poland <1000)_ | Prompt used by the scheduled job |
| `SBR_WEBDRIVER` | _(empty)_ | Selenium WebDriver URL for `scrape_se.py` |
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | _(empty)_ | LangSmith API key |
| `LANGCHAIN_PROJECT` | `my-property-agent` | LangSmith project name |

---

## Getting started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Pinecone account (or set `STORE_TYPE=local` for offline FAISS)

### Installation

```bash
git clone https://github.com/your-username/my-property-agent.git
cd my-property-agent
uv venv && uv sync
```

Or with pip:

```bash
pip install -r agentP/requirements.txt
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
# Runs on http://localhost:8000
python -m clients.rest.main
# or equivalently:
uvicorn clients.rest.main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

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

Or via the client interface:

```bash
python -c "from clients.streamlit.main import StreamlitClient; StreamlitClient().start()"
```

### Telegram bot

```bash
python -m clients.telegram.main
```

Set `TELEGRAM_BOT_TOKEN` in `.env` before starting. The bot streams the property agent's response back to the user, editing the message every 20 tokens.

### Cron job (scheduled search)

```bash
python -m clients.cron.main
# Runs the configured CRON_SEARCH_PROMPT every 30 minutes
```

### CLI (interactive)

```bash
PYTHONPATH=. python agentP/src/agent.py
```

---

## Data ingestion

### Ingest Polish housing CSV data

Place CSV files in `agentP/datasets/pl-housing/`, then:

```bash
python -m agentP.src.gatherers.pl_housing_data_collector
```

### Ingest from URLs

Add URLs to `agentP/urls.txt` (one per line), then run the data collector:

```bash
python -m agentP.src.gatherers.data_collector
```

---

## Testing

```bash
# All tests (108 total)
python -m pytest agentP/tests/ clients/tests/ -v

# agentP core only (60 tests)
python -m pytest agentP/tests/ -v

# Client layer only (48 tests)
python -m pytest clients/tests/ -v

# With coverage
python -m pytest agentP/tests/ clients/tests/ --cov=agentP/src --cov=clients --cov-report=term-missing
```

All external dependencies (LLM, vector store, network, Telegram, FastAPI) are mocked. No real services are required to run the test suite.

---

## LangSmith observability

Set the following in `.env` to enable tracing:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
LANGCHAIN_PROJECT=my-property-agent
```

Every `ask()` and `ask_stream()` call is decorated with `@traceable` and produces a trace with four child spans — `reformulate`, `classify`, `retrieve` (conditional), and `generate` — visible at [smith.langchain.com](https://smith.langchain.com).

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.
