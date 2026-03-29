# CLAUDE.md — AI Assistant Guide for my-property-agent

## Project Overview

**my-property-agent** is a conversational AI system for property search. Users describe what they're looking for and the agent retrieves relevant listings via a RAG (Retrieval-Augmented Generation) pipeline backed by local (FAISS) or cloud (Pinecone) vector stores. It exposes a REST API (FastAPI) and a web chat UI (Streamlit).

---

## Repository Structure

```
my-property-agent/
├── agentP/                         # Core agent module
│   ├── .env                        # Runtime configuration (see Configuration section)
│   ├── requirements.txt            # All Python dependencies (88 packages)
│   ├── urls.txt                    # URLs for web scraping
│   ├── prompts/                    # Prompt templates
│   │   ├── System_Prompt.txt       # Agent system prompt (personality, behavior)
│   │   ├── interaction.json        # CLI UI messages (welcome, goodbye, etc.)
│   │   └── reformulated_prompt.txt # Query reformulation prompt template
│   ├── src/                        # All source code
│   │   ├── agent.py                # CLI entry point (LocalAgent)
│   │   ├── agentRest.py            # FastAPI REST API (port 8000)
│   │   ├── cron_agent.py           # Scheduled property searches (every 30 min)
│   │   ├── config/config.py        # Pydantic settings loaded from .env
│   │   ├── model/                  # LLM orchestration and RAG pipeline
│   │   │   ├── llm_model.py        # LlmModel: main chain orchestrator
│   │   │   ├── llm_factory.py      # LLM provider factory (Ollama, OpenAI, ...)
│   │   │   ├── embedder.py         # SentenceTransformer embeddings + search
│   │   │   ├── rag_context_manager.py  # Retrieves and formats RAG context
│   │   │   ├── context_builder.py  # Formats property listings as readable text
│   │   │   └── session_manager.py  # Per-session in-memory chat history
│   │   ├── persistence/            # Vector store backends
│   │   │   ├── vector_store.py     # Abstract base class
│   │   │   ├── factory.py          # Selects FAISS or Pinecone at runtime
│   │   │   ├── faiss_store.py      # Local FAISS implementation
│   │   │   └── pinecone_store.py   # Cloud Pinecone implementation
│   │   ├── housing/                # Property data ingestion
│   │   │   ├── housing_data_collector.py  # Streams CSV → embedding-ready text
│   │   │   └── housing_csv_reader.py      # CSV parsing utility
│   │   ├── gatherers/              # Web-based data collection
│   │   │   ├── data_collector.py   # Scrapes listing URLs
│   │   │   └── pl_housing_data_collector.py
│   │   └── scraping/               # HTML scraping utilities
│   │       ├── web_scraper.py      # HTTP fetch + HTML cleanup
│   │       ├── url_processor.py    # Regex-based listing extraction
│   │       ├── scrape_se.py        # Sweden-specific scraper
│   │       └── utils.py
│   └── tests/
│       └── model/
│           ├── test_llm_model.py   # Unit tests for LlmModel
│           └── test_embedder.py    # Unit tests for Embedder
└── streamlit_app/                  # Frontend
    ├── app.py                      # Streamlit chat UI (port 8501)
    └── requirements.txt            # Streamlit-specific deps
```

---

## Key Architecture

### Request Flow

```
User (Streamlit UI)
  → POST /ask?prompt=...&stream=True  (FastAPI, port 8000)
  → LocalAgent.ask()
      1. Reformulation chain:  user query → improved search query
      2. RAG retrieval:        reformed query → top-5 property listings
      3. Context builder:      listings → formatted text block
      4. Full chain:           [system prompt + history + context + query] → LLM
  → Streaming response via SSE → UI
```

### LLM Chain Architecture (`llm_model.py`)

`LlmModel` composes four LangChain chains:

| Chain | Purpose |
|---|---|
| `direct_chain_with_history` | Plain Q&A without retrieval |
| `reformulation_chain` | Rewrites the user query for better search recall |
| `rag_chain_with_history` | RAG-augmented answer using retrieved listings |
| `full_chain` | **Default** — reformulate → retrieve → generate |

The public API is `LlmModel.ask(prompt, session_id, stream=False)`.

### Vector Store Backends (`persistence/`)

- `VectorStore` is the abstract base class with `add()` and `search()` methods.
- The factory (`persistence/factory.py`) reads `STORE_TYPE` from env and returns either `FAISSStore` or `PineconeStore`.
- To add a new backend, implement `VectorStore` and register it in the factory.

### Embeddings

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors)
- Controlled by `SENTENCE_TRANSFORMER_MODEL` in `.env`
- `Embedder.embed_documents_to_vectors(docs: dict)` — indexes documents
- `Embedder.search(query: str, k: int)` — returns top-k matches

### Session Management

- `SessionManager` holds `InMemoryChatMessageHistory` per `session_id` (UUID).
- History is in-process RAM only — it does not persist across restarts.
- LangChain's `RunnableWithMessageHistory` injects history into chains automatically.

---

## Configuration

All configuration lives in `agentP/.env`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `STORE_TYPE` | `pinecone` | `local` (FAISS) or `pinecone` |
| `LLM_MODEL_NAME` | `qwen3:8b` | Ollama model name |
| `LLM_PROVIDER` | `ollama` | LLM provider (ollama / openai) |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output |
| `LLM_SEED` | `365` | Reproducibility seed |
| `SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `MEMORY_FILE` | `memory.json` | Chat history file path |
| `URLS_FILE` | `../urls.txt` | URLs for web scraping |
| `PINECONE_API_KEY` | _(required for cloud)_ | Pinecone API key |
| `PINECONE_ENVIRONMENT` | `gcpstart` | Pinecone environment |
| `PINECONE_INDEX_NAME` | `course-ai` | Pinecone index name |
| `PROMPT_FILE` | `agentP/prompts/System_Prompt.txt` | Agent system prompt path |
| `INTERACTION_FILE` | `agentP/prompts/interaction.json` | CLI messages |
| `REFORMULATION_PROMPT` | `agentP/prompts/reformulated_prompt.txt` | Query reformulation template |

Config is loaded via Pydantic `BaseSettings` in `agentP/src/config/config.py`.

---

## Development Workflows

### Setup

```bash
cd agentP
pip install -r requirements.txt
# Copy and fill in .env variables
# Ensure Ollama is running with the target model pulled:
ollama pull qwen3:8b
```

### Running the Backend API

```bash
cd agentP/src
uvicorn agentRest:app --host 0.0.0.0 --port 8000
```

### Running the CLI Agent

```bash
cd agentP/src
python agent.py
```

### Running the Streamlit Frontend

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py          # Connects to localhost:8000
```

### Running the Scheduled Job

```bash
cd agentP/src
python cron_agent.py          # Runs property searches every 30 minutes
```

### Running Tests

```bash
cd agentP
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

Tests use `unittest.mock` extensively — all LLM and vector store calls are mocked. Do not add tests that make real network or model calls.

---

## Code Conventions

### Python Style

- Python 3.10+ with type hints throughout.
- Class-based design; no top-level procedural logic in modules.
- Private methods prefixed with `_` (e.g., `_build_direct_chain()`).
- Environment/config values are always read through `config.py` — never hardcode paths or keys.

### Adding a New LLM Provider

1. Add a new branch in `agentP/src/model/llm_factory.py`.
2. The factory receives a `Config` object; read the necessary keys from it.
3. Return a LangChain-compatible `BaseLLM` or `BaseChatModel` instance.

### Adding a New Vector Store

1. Create a new file under `agentP/src/persistence/`.
2. Implement the `VectorStore` abstract class (`add()` and `search()`).
3. Register the new type in `agentP/src/persistence/factory.py`.

### Modifying Prompts

- `System_Prompt.txt` — controls the agent's persona, response length, and tone.
- `reformulated_prompt.txt` — controls how user queries are cleaned up before retrieval.
- Keep prompts modular; do not embed business logic inside prompt strings.

### Data Ingestion

- CSV files are streamed row-by-row via `housing_data_collector.stream_csv_files()` to avoid memory pressure on large datasets.
- Each row is converted to an embedding-ready text block by `generateEmbededDocument()`, which highlights: location, price, rooms, ownership type, and amenities.
- Web-scraped content is chunked into 6000-character segments before embedding.

---

## Testing Guidelines

- Tests live in `agentP/tests/`, mirroring the `src/` structure.
- All external dependencies (LLM, vector store, HTTP) **must** be mocked.
- Use `pytest` as the test runner.
- Use `unittest.mock.patch` or `MagicMock` for injecting fakes.
- Test coverage is tracked; aim to cover at least the `model/` and `persistence/` packages.

---

## Important File Locations

| Purpose | Path |
|---|---|
| Agent system prompt | `agentP/prompts/System_Prompt.txt` |
| Query reformulation prompt | `agentP/prompts/reformulated_prompt.txt` |
| CLI interaction messages | `agentP/prompts/interaction.json` |
| Environment config | `agentP/.env` |
| All Python dependencies | `agentP/requirements.txt` |
| Streamlit UI dependencies | `streamlit_app/requirements.txt` |
| URLs for scraping | `agentP/urls.txt` |

---

## Common Gotchas

- **`.env` path sensitivity**: Config file paths (e.g., `PROMPT_FILE`) are relative to where the process is launched — always run from the `agentP/src/` directory or adjust paths accordingly.
- **Pinecone vs FAISS**: Set `STORE_TYPE=local` for offline/local development; `pinecone` requires a valid `PINECONE_API_KEY`.
- **Session memory is ephemeral**: Chat history lives in RAM and is lost on process restart. There is no database-backed persistence layer yet.
- **Ollama must be running**: The backend will fail at startup if Ollama is not reachable. Pull the target model before starting (`ollama pull qwen3:8b`).
- **Streaming**: The `/ask` endpoint uses Server-Sent Events (SSE). Pass `stream=True` as a query param to enable it. The Streamlit app always uses streaming.
