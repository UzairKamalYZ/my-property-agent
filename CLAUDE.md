# CLAUDE.md — AI Assistant Guide for my-property-agent

## Project Overview

**my-property-agent** is a conversational AI system for property search. Users describe what they're looking for and the agent retrieves relevant listings via a RAG (Retrieval-Augmented Generation) pipeline backed by local (FAISS) or cloud (Pinecone) vector stores. It exposes a REST API (FastAPI) and a web chat UI (Streamlit).

---

## Repository Structure

```
my-property-agent/
├── README.md
├── CLAUDE.md
├── .gitignore
├── agentP/
│   ├── .env                        # Runtime configuration (see Configuration section)
│   ├── requirements.txt            # All Python dependencies (88 packages)
│   ├── urls.txt                    # URLs for web scraping
│   ├── img.png                     # Architecture diagram
│   ├── prompts/
│   │   ├── System_Prompt.txt       # Agent system prompt (personality, behavior)
│   │   ├── interaction.json        # CLI UI messages (welcome, goodbye, etc.)
│   │   └── reformulated_prompt.txt # Query reformulation prompt template
│   └── src/
│       ├── agent.py                # CLI entry point (LocalAgent)
│       ├── agentRest.py            # FastAPI REST API (port 8000)
│       ├── cron_agent.py           # Scheduled property searches (every 30 min)
│       ├── config/
│       │   └── config.py           # Pydantic settings loaded from .env
│       ├── model/
│       │   ├── llm_model.py        # LlmModel: main chain orchestrator
│       │   ├── llm_factory.py      # LLM provider factory (Ollama, OpenAI, ...)
│       │   ├── embedder.py         # SentenceTransformer embeddings + search
│       │   ├── rag_context_manager.py  # Retrieves and formats RAG context
│       │   ├── context_builder.py  # Formats property listings as readable text
│       │   └── session_manager.py  # Per-session in-memory chat history
│       ├── persistence/
│       │   ├── vector_store.py     # Abstract base class
│       │   ├── factory.py          # Selects FAISS or Pinecone at runtime
│       │   ├── faiss_store.py      # Local FAISS implementation
│       │   └── pinecone_store.py   # Cloud Pinecone implementation
│       ├── housing/
│       │   ├── housing_data_collector.py  # Streams CSV → embedding-ready text
│       │   └── housing_csv_reader.py      # CSV parsing utility (incomplete)
│       ├── gatherers/
│       │   ├── data_collector.py          # Scrapes listing URLs
│       │   └── pl_housing_data_collector.py  # Polish housing data ingestion script
│       └── scraping/
│           ├── web_scraper.py      # HTTP fetch + HTML cleanup
│           ├── url_processor.py    # Regex-based listing extraction
│           ├── scrape_se.py        # Selenium-based scraper (has known bug)
│           └── utils.py            # Converts listings to LangChain Documents
├── streamlit_app/
│   ├── app.py                      # Streamlit chat UI (port 8501)
│   └── requirements.txt            # Streamlit-specific deps
└── agentP/tests/
    └── model/
        ├── test_llm_model.py       # Unit tests for LlmModel (has known issues)
        └── test_embedder.py        # Unit tests for Embedder
```

---

## Key Architecture

### Request Flow

```
User (Streamlit UI / curl)
  → GET /ask?prompt=...&stream=True   (FastAPI, port 8000)
  → LocalAgent.ask(prompt, stream)
      1. Reformulation chain:  user query → improved search query
      2. RAG retrieval:        reformed query → top-5 property listings
      3. Context builder:      listings → formatted text block
      4. Full chain:           [system prompt + history + context + query] → LLM
  → Streaming response via SSE (or plain JSON) → UI
```

### LLM Chain Architecture (`llm_model.py`)

`LlmModel` composes four LangChain chains built in `_build_chains()`:

| Chain | Purpose |
|---|---|
| `direct_chain_with_history` | Plain Q&A with conversation history, no retrieval |
| `reformulation_chain` | Rewrites user query for better vector search recall |
| `rag_chain_with_history` | RAG-augmented answer using retrieved listings + history |
| `full_chain` | **Default** — reformulation → RAG chain |

**Only public method:** `LlmModel.ask(system_prompt, user_query, session_id, stream=False)`

The full chain composition uses the `|` operator:
```python
full_chain = reformulation_chain | (lambda x: {"question": x}) | rag_chain_with_history
```

### Vector Store Backends (`persistence/`)

- `VectorStore` — abstract base with `add(vectors, metadatas)` and `search(query_vector, k)`.
- `FAISSStore` — local L2-indexed FAISS store; stores metadata in a parallel Python list.
- `PineconeStore` — cloud store; auto-creates index (cosine metric, serverless AWS us-east-1) if absent.
- `create_vector_store(store_type, dim, config)` in `factory.py` selects the backend at runtime.

### Embeddings (`embedder.py`)

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional float32 vectors)
- Controlled by `SENTENCE_TRANSFORMER_MODEL` in `.env`
- Key methods:
  - `embed_documents_to_vectors(docs: dict) -> list[dict]` — encodes and assembles vector objects
  - `embed_texts(texts: list[str]) -> np.ndarray` — batch encode
  - `embed_query(query: str) -> np.ndarray` — single query encode
  - `search(query: str, k: int = 5)` — embed → store.search()
  - `get_store(dim=384) -> VectorStore` — lazy-initializes vector store via factory
  - `save_vectors_in_store(vectors)` — calls store.add_vectors()

### Session Management (`session_manager.py`)

- `SessionManager.get_session_history(session_id: str)` returns an `InMemoryChatMessageHistory`.
- Sessions are keyed by UUID, stored in a plain dict.
- **History is RAM-only** — lost on process restart. No persistent storage exists yet.
- LangChain's `RunnableWithMessageHistory` wraps chains and injects history automatically.

### Context Building (`context_builder.py`)

`build_context_from_listings(listings: list[dict]) -> str` formats each listing as:

```
Listing N
Location: <city or N/A>
Price: <price PLN or N/A>
Rooms: <rooms or N/A>
Surface: <squareMeters m² or N/A>
Floor: <floor / floorCount or N/A>
Type: <type or N/A>
Ownership: <ownership or N/A>
Building material: <buildingMaterial or N/A>
Condition: <condition or N/A>
Amenities: Parking, Balcony, Elevator, Security, Storage  (or None)
Description: <text or N/A>
```

---

## Configuration

All configuration lives in `agentP/.env`. Loaded via `Config` in `agentP/src/config/config.py`.

| Variable | Default | Description |
|---|---|---|
| `STORE_TYPE` | `pinecone` | `local` (FAISS) or `pinecone` |
| `LLM_MODEL_NAME` | `qwen3:8b` | Ollama model name |
| `LLM_PROVIDER` | `ollama` | LLM provider (`ollama` only currently) |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output |
| `LLM_SEED` | `365` | Reproducibility seed |
| `SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `MEMORY_FILE` | `memory.json` | Chat history file path (legacy) |
| `URLS_FILE` | `../urls.txt` | URLs for web scraping (relative to launch dir) |
| `PINECONE_API_KEY` | _(required for cloud)_ | Pinecone API key |
| `PINECONE_ENVIRONMENT` | `gcpstart` | Pinecone environment |
| `PINECONE_INDEX_NAME` | `course-ai` | Pinecone index name |
| `PROMPT_FILE` | `agentP/prompts/System_Prompt.txt` | Agent system prompt path |
| `INTERACTION_FILE` | `agentP/prompts/interaction.json` | CLI interaction messages |
| `REFORMULATION_PROMPT` | `agentP/prompts/reformulated_prompt.txt` | Query reformulation template |

---

## Development Workflows

### Setup

```bash
cd agentP
pip install -r requirements.txt
# Edit .env with real values
# Ensure Ollama is running and the model is pulled:
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

Requires the REST API to be running first.

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py     # http://localhost:8501
```

### Running the Scheduled Job

```bash
cd agentP/src
python cron_agent.py     # Fires every 30 minutes
```

### Ingesting Polish Housing Data

```bash
cd agentP/src/gatherers
python pl_housing_data_collector.py   # Expects CSV files at ../datasets/pl-housing/
```

### Running Tests

```bash
cd agentP
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

All external dependencies (LLM, vector store, HTTP) must be mocked. Do not write tests that make real network or model calls.

---

## REST API Reference

| Method | Path | Params | Response |
|---|---|---|---|
| GET | `/ask` | `prompt` (str), `stream` (bool, default `false`) | `{"response": str}` or SSE stream |

**Streaming format** (SSE): Each chunk is sent as `data: {chunk}\n\n` with a 0.01 s inter-chunk delay.

---

## Code Conventions

### Python Style

- Python 3.10+ with type hints throughout.
- Class-based design; no top-level procedural logic in modules.
- Private methods prefixed with `_` (e.g., `_build_chains()`, `_initialize_components()`).
- Config values always accessed through `Config` in `config.py` — never hardcode paths or keys.
- Context managers (`__enter__`/`__exit__`/`close()`) used for resource-owning classes (`LocalAgent`, `housing_data_collector`).

### Adding a New LLM Provider

1. Add a branch in `agentP/src/model/llm_factory.py` matching the provider name string.
2. Read required keys from the `Config` object passed to `create_llm()`.
3. Return a LangChain-compatible `BaseLLM` or `BaseChatModel`.

### Adding a New Vector Store

1. Create a file under `agentP/src/persistence/`.
2. Implement `VectorStore` abstract class — `add(vectors, metadatas)` and `search(query_vector, k)`.
3. Register the `store_type` string in `agentP/src/persistence/factory.py`.

### Modifying Prompts

- `System_Prompt.txt` — agent persona, response style, follow-up behaviour.
- `reformulated_prompt.txt` — template used to rewrite user queries; must retain the `{user_prompt}` placeholder for LangChain formatting.
- Keep prompts modular; do not embed conditional business logic inside prompt strings.

### Data Ingestion

- CSV files are streamed row-by-row via `housing_data_collector.stream_csv_files()` in configurable chunks (default 5 000 rows) to avoid memory pressure.
- Each row is converted to an embedding-ready text block by `generateEmbededDocument()`.
- `pl_housing_data_collector.py` persists in batches of 100 documents.
- Web-scraped content is chunked into 6 000-character segments before embedding.

---

## Testing Guidelines

- Tests live in `agentP/tests/`, mirroring `src/` structure.
- All external dependencies (LLM, vector store, HTTP) **must** be mocked with `unittest.mock.patch` or `MagicMock`.
- Use `pytest` as the runner; `pytest-cov` for coverage.
- Aim to cover at least the `model/` and `persistence/` packages.

---

## Known Issues & Limitations

### Critical

1. **Test/implementation mismatch** (`tests/model/test_llm_model.py`): Tests call `ask_direct()` and `ask_with_reformulation()` which do not exist. `LlmModel` only exposes `ask()`. Tests will fail without fixes.

2. **Broken import in cron job** (`cron_agent.py:4`): Uses `from agent import LocalAgent` (relative) — fails when run from any directory other than `agentP/src/`. Should be an absolute/package import.

3. **`os.getenv()` called without argument** (`scraping/scrape_se.py:9`): Returns `None`; likely should be `os.getenv("SBR_WEBDRIVER")` or equivalent.

4. **`housing_csv_reader.py` is incomplete**: The file contains only a single character (`d`). `stream_csv_files()` is implemented directly in `housing_data_collector.py`.

### Major

5. **No authentication on REST API**: `/ask` endpoint is unauthenticated and accepts arbitrary prompts.

6. **Session history not persistent**: `InMemoryChatMessageHistory` is lost on server restart. There is no database-backed persistence layer.

7. **Hardcoded cron prompt** (`cron_agent.py:16`): Search query is a hardcoded string; should be configurable.

8. **Belgium-centric city extraction** (`url_processor.py`): `_extract_city()` checks against a hardcoded Belgian city list (Brussels, Antwerp, Bruges) while the main data pipeline targets Polish housing data.

9. **PEP 8 naming** (`housing_data_collector.py`): Class name `housing_data_collector` should be `HousingDataCollector`.

### Minor

10. **Relative path sensitivity** (`.env:URLS_FILE=../urls.txt`): Breaks if the process is not launched from `agentP/src/`.

11. **`PINECONE_INDEX_NAME=course-ai`**: Suggests a course/prototype origin; rename before any production use.

12. **Dunder-named public methods** (`data_collector.py`): `__getDataFromUrls__` and `__storVectorEmbeddings__` (also a typo — "stor" vs "store") use double-underscore naming but are meant to be called externally.

---

## Important File Locations

| Purpose | Path |
|---|---|
| Agent system prompt | `agentP/prompts/System_Prompt.txt` |
| Query reformulation prompt | `agentP/prompts/reformulated_prompt.txt` |
| CLI interaction messages | `agentP/prompts/interaction.json` |
| Runtime environment config | `agentP/.env` |
| Python dependencies (backend) | `agentP/requirements.txt` |
| Python dependencies (frontend) | `streamlit_app/requirements.txt` |
| URLs for web scraping | `agentP/urls.txt` |

---

## Common Gotchas

- **`.env` path sensitivity**: `PROMPT_FILE`, `URLS_FILE`, etc. are relative to where the process is launched. Always run the backend from `agentP/src/`, or use absolute paths.
- **Pinecone vs FAISS**: Set `STORE_TYPE=local` for offline/local development; `pinecone` requires a valid `PINECONE_API_KEY` and network access.
- **Ollama must be running**: The backend calls the Ollama HTTP API at startup. Run `ollama pull qwen3:8b` and ensure the daemon is active before starting.
- **Streaming**: The `/ask` endpoint uses Server-Sent Events (SSE). Pass `stream=true` as a query param to enable it. The Streamlit app connects to the non-streaming endpoint by default.
- **Session memory is ephemeral**: Chat history lives in RAM and is lost on process restart.
- **Test suite**: `test_llm_model.py` currently fails because it references non-existent methods. Fix method names before running the test suite.
