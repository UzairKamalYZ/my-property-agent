# My Property Agent

My Property Agent is a conversational AI agent designed to answer your questions about properties. It uses a local large language model (LLM) through Ollama to understand and respond to your queries.

## Features

*   **Conversational Interface:** Interact with the agent in a natural, conversational way.
*   **Conversation Memory:** The agent remembers the context of your conversation, allowing for follow-up questions.
*   **Streaming Responses:** Get responses from the agent as they are generated, providing a more interactive experience.
*   **Configurable:** The agent's model and memory settings can be easily configured through a `.env` file.
*   **Web Scraping:** The agent can scrape content from a list of URLs to provide more context for its answers.
*   **RESTful API:** The agent can be exposed as a RESTful service.

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

2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  Create a `.env` file in the root of the project by copying the example file:
    ```bash
    cp .env.example .env
    ```

2.  Modify the `.env` file to set your desired configuration:
    *   `QWEN_MODEL_NAME`: The name of the Ollama model to use (e.g., `qwen3:1.7b`).
    *   `MEMORY_FILE`: The name of the file to use for conversation memory (e.g., `memory.json`).
    *   `URLS_FILE`: The name of the file containing URLs to scrape (e.g., `urls.txt`).

## Running the Agent as a Standalone Script

To start the agent as a standalone script, run the following command:

```bash
PYTHONPATH=. python3 src/agent.py
```

## Running the RESTful Service

To run the agent as a RESTful service, use the following command:

```bash
PYTHONPATH=. uvicorn src.main:app --reload
```

The service will be available at `http://127.0.0.1:8000`.

### API Usage

You can interact with the service using `curl` or any other API client.

**Example `curl` command:**

```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello"}' http://127.0.0.1:8000/ask
```

**Streaming Example:**

```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello", "stream": true}' http://127.0.0.1:8000/ask
```

## Testing

To run the tests, use pytest:

```bash
pytest
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── scraped_content.json
├── src
│   ├── agent.py
│   ├── config.py
│   ├── __init__.py
│   ├── main.py
│   ├── memory_manager.py
│   ├── model
│   │   ├── __init__.py
│   │   └── llm_model.py
│   └── scraping
│       ├── __init__.py
│       ├── url_processor.py
│       └── web_scraper.py
├── tests
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_config.py
│   ├── test_llm_model.py
│   ├── test_memory_manager.py
│   └── test_web_scraper.py
└── urls.txt
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.