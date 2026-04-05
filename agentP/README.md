# agentP — Property Search Agent

Conversational property search engine. Extends `BaseAgent` from `core/` with a property-specific system prompt and RAG pipeline over Polish housing listings.

---

## What lives here

`agentP` contains only property-specific code. All shared AI infrastructure (LLM, LangGraph pipeline, MCP, vector store, embeddings) lives in `core/`.

```
agentP/
├── src/
│   ├── agent.py          # LocalAgent(BaseAgent) — adds WebScraper + CLI interaction
│   ├── housing/          # CSV → embedding-ready text
│   ├── gatherers/        # Data ingestion (web scraping + Polish housing CSV)
│   ├── scraping/         # HTTP + Selenium scrapers, URL processing
│   └── prompts/
│       ├── System_Prompt.txt   # Property agent persona and output format
│       └── interaction.json    # CLI welcome/goodbye strings
├── tests/                # Property-specific tests
└── urls.txt              # URLs for web scraping
```

---

## LocalAgent

```python
from agentP.src.agent import LocalAgent

with LocalAgent() as agent:
    print(agent.ask("2 bed apartment Warsaw under 3000 PLN"))
```

`LocalAgent` inherits `ask()`, `ask_stream()`, `close()`, and the context manager from `BaseAgent`. It adds:
- `self.web_scraper` — `WebScraper` instance for on-demand scraping
- `self.interaction_texts` — CLI strings loaded from `interaction.json`

It overrides nothing in `BaseAgent` — the default system prompt (`Config.PROMPT_FILE`) and default RAG pipeline (`RagContextManager(Embedder())`) are the correct property-search defaults.

---

## System prompt conventions

`agentP/src/prompts/System_Prompt.txt` contains the agent persona and output format. It does **not** name specific MCP tool names — the LLM discovers those from the bound schema at startup. Only tool use *policy* is stated:

```
Tool use policy:
- Only call a tool if the answer cannot be derived without it.
- Do not call the same tool more than once for the same input.
- Always return a final answer after using tools.
```

---

## Data ingestion

### Polish housing CSV

Place CSV files in `agentP/datasets/pl-housing/`, then run:

```bash
python -m agentP.src.gatherers.pl_housing_data_collector
```

### From URLs

Add URLs to `agentP/urls.txt` (one per line), then run:

```bash
python -m agentP.src.gatherers.data_collector
```

---

## Running the CLI

```bash
python -m agentP.src.agent
```

---

## Configuration

All settings are in `.env` at the project root, loaded by `core/src/config/config.py`.

| Variable | Default | Description |
|---|---|---|
| `PROMPT_FILE` | `agentP/src/prompts/System_Prompt.txt` | Property agent system prompt |
| `INTERACTION_FILE` | `agentP/src/prompts/interaction.json` | CLI strings |
| `STORE_TYPE` | `pinecone` | `local` (FAISS) or `pinecone` |
| `PINECONE_INDEX_NAME` | `property-agent` | Vector index for property listings |
| `RAG_K` | `5` | Listings returned per vector search |
| `CRON_SEARCH_PROMPT` | _(2-bed Poland <1000)_ | Prompt for the cron client |
| `SESSION_DB_FILE` | `sessions.db` | SQLite chat history |
| `URLS_FILE` | `agentP/urls.txt` | URLs for web scraping |
