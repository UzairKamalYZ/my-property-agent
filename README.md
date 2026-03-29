# My Property Agent

My Property Agent is a conversational AI agent designed to answer your questions about properties. It uses a local large language model (LLM) via Ollama to understand and respond to your queries. This project also includes a Streamlit frontend for interactive chat.

## Features

*   **Conversational Interface:** Interact with the agent in a natural, conversational way.
*   **Conversation Memory:** The agent remembers the context of your conversation, allowing for follow-up questions.
*   **Streaming Responses:** Get responses from the agent as they are generated, providing a more interactive experience.
*   **Configurable:** The agent's model can be easily configured through a `.env` file.
*   **Web Scraping:** The agent can scrape content from a list of URLs to provide more context for its answers.
*   **RESTful API:** The agent can be exposed as a RESTful service.
*   **Streamlit Frontend:** An interactive web application built with Streamlit to chat with the agent.
*   **Telegram Bot:** Receive and respond to messages via a Telegram bot (`MyTelegramAgent`).
*   **LangGraph RAG Pipeline:** Core inference is powered by a structured LangGraph graph (`LlmModelGraph`).
*   **LangSmith Observability:** Full trace visibility of every run via LangSmith.

## Architecture

### LangGraph RAG Pipeline (`LlmModelGraph`)

Every user query flows through a three-node LangGraph pipeline before a response is returned.

```
┌─────────────────────────────────────────────────────────────────┐
│                      LlmModelGraph                              │
│                                                                 │
│   User Query                                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────┐     reformulated     ┌──────────┐                 │
│  │ REFORMU- │ ──────question──────▶│ RETRIEVE │                 │
│  │  LATE    │                      │          │                 │
│  │          │                      │ Pinecone │                 │
│  │ llama3.2 │                      │ (RAG)    │                 │
│  └──────────┘                      └──────────┘                 │
│                                         │                       │
│                                      context                    │
│                                         │                       │
│                                         ▼                       │
│                                   ┌──────────┐                  │
│                                   │ GENERATE │                  │
│                                   │          │                  │
│                                   │ system   │                  │
│                                   │ prompt + │                  │
│                                   │ history  │                  │
│                                   │ + context│                  │
│                                   │          │                  │
│                                   │ llama3.2 │                  │
│                                   └──────────┘                  │
│                                         │                       │
│                                      answer                     │
│                                         │                       │
│                                         ▼                       │
│                                    END / stream                 │
└─────────────────────────────────────────────────────────────────┘
```

| Node | Input | What it does | Output |
|---|---|---|---|
| **reformulate** | `user_prompt` | Rewrites the raw user query into a precise search question using `reformulated_prompt.txt` | `reformulated_question` |
| **retrieve** | `reformulated_question` | Queries Pinecone via sentence-transformer embeddings to fetch relevant property listings | `context` |
| **generate** | `reformulated_question` + `context` + `history` | Calls the LLM with the system prompt, full conversation history, and retrieved context | `answer` + updated `history` |

### Telegram Integration

```
Telegram User
     │
     │  message text
     ▼
MyTelegramAgent
     │
     │  ask(prompt, stream=True)
     ▼
LocalAgent
     │
     │  ask_stream(user_query)
     ▼
LlmModelGraph  ──▶  LangGraph pipeline (above)
     │
     │  token chunks (streamed)
     ▼
MyTelegramAgent  ──▶  edit_text() every 20 tokens
     │
     ▼
Telegram User (live-updating message)
```

### Observability (LangSmith)

```
LlmModelGraph.ask() / ask_stream()
         │
         │  @traceable
         ▼
  LangSmith Trace
  ├── reformulate  (LLM call)
  ├── retrieve     (vector search)
  └── generate     (LLM call + streamed tokens)
```

Traces are visible at [smith.langchain.com](https://smith.langchain.com) under the project `my-property-agent`.

---

## Getting Started

### Prerequisites

*   Python 3.10 or later
*   [Ollama](https://ollama.ai/) installed and running

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/my-property-agent.git
    cd my-property-agent
    ```

2.  Create and activate a virtual environment for `agentP` (recommended):
    ```bash
    cd agentP
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd .. # Go back to project root
    ```

3.  Create and activate a virtual environment for `streamlit_app` (recommended):
    ```bash
    cd streamlit_app
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd .. # Go back to project root
    ```

### Configuration

1.  Create a `.env` file in the `agentP` directory (e.g., `agentP/.env`).

2.  Modify the `agentP/.env` file to set your desired configuration. Key variables:

    | Variable | Description |
    |---|---|
    | `LLM_PROVIDER` | LLM backend (e.g. `ollama`) |
    | `LLM_MODEL_NAME` | Model name (e.g. `llama3.2`) |
    | `PINECONE_API_KEY` | Pinecone API key for vector search |
    | `PINECONE_INDEX_NAME` | Name of the Pinecone index |
    | `TELEGRAM_BOT_TOKEN` | Token from [@BotFather](https://t.me/BotFather) |
    | `LANGCHAIN_TRACING_V2` | Set `true` to enable LangSmith tracing |
    | `LANGCHAIN_API_KEY` | API key from [smith.langchain.com](https://smith.langchain.com) |
    | `LANGCHAIN_PROJECT` | LangSmith project name (default: `my-property-agent`) |

3.  The Streamlit app's API URL is configured in `streamlit_app/config.py`.

## Running the Agent

### Running the Telegram Bot

```bash
cd agentP
uv sync
uv run main.py
```

The bot will start polling Telegram. Send any message to your bot — it will stream the response back token by token.

### Running the Agent as a Standalone Script

To start the agent as a standalone script (for testing or direct interaction), run the following command from the project root:

```bash
cd /Users/uzairkamal/work/my-property-agent
# Activate agentP's venv if not already active
# source agentP/venv/bin/activate
PYTHONPATH=. python3 agentP/src/agent.py
```

### Running the RESTful Service (Backend for Streamlit App)

To run the agent as a RESTful service, which the Streamlit frontend will connect to, use the following command from the **project root**:

```bash
cd /Users/uzairkamal/work/my-property-agent
# Activate agentP's venv
# source agentP/venv/bin/activate
uvicorn agentP.src.agentRest:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at `http://localhost:8000`. Keep this terminal window open and the server running.

### Running the Streamlit Frontend

To run the interactive chat application:

1.  **Ensure the `agentP` RESTful Service is running** (as described above).
2.  Open a **new terminal window**.
3.  Navigate to the `streamlit_app` directory:
    ```bash
    cd /Users/uzairkamal/work/my-property-agent/streamlit_app
    ```
4.  Activate the `streamlit_app`'s virtual environment:
    ```bash
    source venv/bin/activate
    ```
5.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    This will open the Streamlit application in your web browser.

For more detailed instructions on the Streamlit application, refer to `streamlit_app/README.md`.

## API Usage (for the RESTful Service)

You can interact with the service using `curl` or any other API client.

**Example `curl` command:**

```bash
curl "http://localhost:8000/ask?prompt=Hello"
```

**Streaming Example:**

```bash
curl "http://localhost:8000/ask?prompt=Hello&stream=True"
```

## Testing

To run the tests for the `agentP` module, navigate to the `agentP` directory and use pytest:

```bash
cd agentP
pytest
```

## Project Structure

```
.
├── README.md                        <- This file
├── agentP/
│   ├── main.py                      <- Entry point — starts the Telegram bot
│   ├── pyproject.toml               <- Python dependencies (uv)
│   ├── .env                         <- Environment variables
│   ├── prompts/
│   │   ├── System_Prompt.txt        <- LLM system prompt
│   │   ├── reformulated_prompt.txt  <- Prompt used by the reformulate node
│   │   └── interaction.json         <- CLI interaction strings
│   └── src/
│       ├── agent.py                 <- LocalAgent — thin wrapper over the graph
│       ├── telegram_agent.py        <- MyTelegramAgent — Telegram bot interface
│       ├── config/
│       │   └── config.py            <- Centralised config from .env
│       └── model/
│           ├── llm_factory.py       <- Creates the LLM (Ollama, etc.)
│           ├── llm_model_graph.py   <- LangGraph RAG pipeline (reformulate → retrieve → generate)
│           ├── embedder.py          <- Sentence-transformer embeddings
│           ├── rag_context_manager.py <- Pinecone vector search
│           └── session_manager.py   <- Conversation history store
├── streamlit_app/                   <- Streamlit frontend
│   ├── app.py
│   ├── config.py
│   └── README.md
└── ... (.gitignore, .idea, etc.)
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
