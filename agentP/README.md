# agentP

Conversational property search engine powered by a two-node LangGraph RAG pipeline.

---

## Architecture

### Request flow

```
User
 │
 ▼
LocalAgent.ask(prompt)
 │
 ▼
LlmModelGraph
 ├── [retrieve]   user_prompt → vector search → context (property listings)
 └── [generate]   system prompt + context + history + user_prompt → answer
                   └── tool call loop (MCP tools, e.g. currency_convert)
 │
 ▼
Response (blocking string or token stream)
```

### Graph (`llm_model_graph.py`)

Two nodes, no conditional edges:

| Node | What it does |
|---|---|
| `retrieve` | Embeds `user_prompt`, runs cosine similarity search against the vector store, returns formatted listing context |
| `generate` | Sends system prompt + context + session history + user question to the LLM; runs a tool-call loop until the LLM produces a plain-text answer |

**State**

```python
class State(TypedDict):
    user_prompt: str
    context: str
    answer: str
    session_history: List[AnyMessage]
```

**Why two nodes?**
Vector search is ~1 ms. Adding an LLM node to decide whether to search (classify) or to rewrite the query (reformulate) costs ~1 s each and adds no meaningful quality gain with a sentence-transformer embedding model.

### MCP tool integration

Tools are declared in **`mcp.json`** at the project root — no code changes needed to add a server:

```json
{
  "servers": [
    {"name": "finance", "command": ["npx", "--yes", "@easysolutions906/mcp-finance"]}
  ]
}
```

At startup `mcp_tools.py` reads this file, starts each server lazily (subprocess on first use), and binds all discovered tools to the LLM via `llm.bind_tools()`. The `generate` node runs a tool-call loop that handles both native function-calling responses and Ollama's JSON-text fallback transparently.

**Adding a new MCP server**: add one line to `mcp.json` and restart — no Python changes required.

### Embeddings

Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional float32)

Key operations in `embedder.py`:

| Method | Description |
|---|---|
| `embed_query(query)` | Single query → vector |
| `embed_documents_to_vectors(docs)` | Batch → list of vectors |
| `search(query, k)` | Embed + nearest-neighbour lookup |

### Vector store backends

Selected at runtime via `STORE_TYPE` in `.env`:

| Backend | `STORE_TYPE` | Notes |
|---|---|---|
| `FAISSStore` | `local` | Local L2 index, no network required |
| `PineconeStore` | `pinecone` | Cloud, auto-creates cosine index on AWS us-east-1 |

---

## Key files

```
agentP/
├── src/
│   ├── agent.py                    # LocalAgent — thin façade over LlmModelGraph
│   ├── config/config.py            # Settings loaded from project-root .env
│   ├── model/
│   │   ├── llm_model_graph.py      # LangGraph pipeline (retrieve → generate)
│   │   ├── llm_factory.py          # LLM provider factory
│   │   ├── embedder.py             # Embedding + vector search
│   │   ├── rag_context_manager.py  # Retrieves and formats RAG context
│   │   ├── context_builder.py      # Formats listing dicts as readable text
│   │   ├── session_manager.py      # SQLite-backed session history
│   │   ├── mcp_registry.py         # Generic MCPProcess + MCPRegistry
│   │   └── mcp_tools.py            # Loads mcp.json, exposes shared _mcp registry
│   ├── persistence/
│   │   ├── vector_store.py         # Abstract base (add / search)
│   │   ├── factory.py              # Selects FAISS or Pinecone at runtime
│   │   ├── faiss_store.py
│   │   └── pinecone_store.py
│   ├── housing/                    # CSV → embedding-ready text
│   ├── gatherers/                  # Data ingestion (web + Polish housing)
│   ├── scraping/                   # HTTP + Selenium scrapers
│   └── prompts/
│       ├── System_Prompt.txt       # Agent persona and tool instructions
│       └── interaction.json        # CLI strings
├── urls.txt                        # URLs for web scraping
└── requirements.txt                # Pinned dependencies
```

---

## Configuration (`.env` at project root)

| Variable | Default | Description |
|---|---|---|
| `STORE_TYPE` | `pinecone` | `local` (FAISS) or `pinecone` |
| `LLM_MODEL_NAME` | `llama3.2` | Ollama model name |
| `LLM_PROVIDER` | `ollama` | LLM provider |
| `LLM_TEMPERATURE` | `0.0` | |
| `LLM_SEED` | `365` | Reproducibility seed |
| `SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | |
| `RAG_K` | `5` | Listings returned per search |
| `PINECONE_API_KEY` | _(required for cloud)_ | |
| `PINECONE_INDEX_NAME` | `course-ai` | |
| `PROMPT_FILE` | `agentP/src/prompts/System_Prompt.txt` | |
| `SESSION_DB_FILE` | `sessions.db` | SQLite chat history |
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` for LangSmith traces |
| `LANGCHAIN_API_KEY` | _(empty)_ | |

---

## Data

Property listings sourced from Polish rental/sale CSV datasets (2023–2024):

```
apartments_rent_pl_2024_06.csv
apartments_rent_pl_2024_05.csv
apartments_rent_pl_2024_04.csv
apartments_rent_pl_2024_03.csv
apartments_rent_pl_2024_02.csv
apartments_rent_pl_2024_01.csv
apartments_rent_pl_2023_12.csv
apartments_pl_2024_03.csv
apartments_pl_2024_01.csv
apartments_pl_2023_12.csv
apartments_pl_2023_11.csv
apartments_pl_2023_10.csv
apartments_pl_2023_09.csv
apartments_pl_2023_08.csv
```
