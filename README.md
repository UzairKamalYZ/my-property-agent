# My Property Agent

My Property Agent is a conversational AI agent designed to answer your questions about properties. It uses a local large language model (LLM) through Ollama to understand and respond to your queries.

## Features

*   **Conversational Interface:** Interact with the agent in a natural, conversational way.
*   **Conversation Memory:** The agent remembers the context of your conversation, allowing for follow-up questions.
*   **Streaming Responses:** Get responses from the agent as they are generated, providing a more interactive experience.
*   **Configurable:** The agent's model and memory settings can be easily configured through a `.env` file.

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

1.  Create a `.env` file in the root of the project:
    ```bash
    cp .env.example .env
    ```

2.  Modify the `.env` file to set your desired configuration:
    *   `QWEN_MODEL_NAME`: The name of the Ollama model to use (e.g., `qwen3:1.7b`).
    *   `MEMORY_FILE`: The name of the file to use for conversation memory (e.g., `memory.json`).

### Running the Agent

To start the agent, run the following command:

```bash
python -m src.agent
```

## Usage

Once the agent is running, you can start a conversation by typing your questions into the console.

To exit the agent, type `exit` or `quit`.

## Testing

To run the tests, use pytest:

```bash
pytest
```

## Project Structure

```
.my-property-agent/
├── src/
│   ├── __init__.py
│   ├── agent.py          # Main agent logic
│   ├── config.py         # Configuration management
│   └── model/
│       └── qwen_model.py # Qwen model wrapper
├── tests/
│   ├── test_agent.py
│   ├── test_config.py
│   └── test_qwen_model.py
├── .env                  # Environment variables
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
