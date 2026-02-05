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

2.  Modify the `agentP/.env` file to set your desired configuration (e.g., LLM model name, API keys). An example is provided in the `agentP` directory.

3.  The Streamlit app's API URL is configured in `streamlit_app/config.py`.

## Running the Agent

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
├── README.md                  <- This main README file
├── agentP/                    <- Contains the core agent logic and REST API
│   ├── .env                   <- Environment variables for agentP
│   ├── prompts/               <- Agent's prompts and interaction texts
│   │   ├── interaction.json
│   │   └── System_Prompt.txt
│   ├── requirements.txt       <- Python dependencies for agentP
│   ├── src/                   <- AgentP source code
│   │   ├── agent.py           <- Core LocalAgent implementation
│   │   ├── agentRest.py       <- FastAPI application for agentP REST API
│   │   └── ... (other agentP source files)
│   └── venv/                  <- Python virtual environment for agentP
├── streamlit_app/             <- Streamlit frontend application
│   ├── app.py                 <- Main Streamlit application file
│   ├── config.py              <- Configuration for Streamlit app (e.g., API URL)
│   ├── README.md              <- README for the Streamlit app
│   ├── requirements.txt       <- Python dependencies for Streamlit app
│   └── venv/                  <- Python virtual environment for Streamlit app
└── ... (other project files like .gitignore, .idea, etc.)
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
