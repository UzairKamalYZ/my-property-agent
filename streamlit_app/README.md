# Streamlit Property Agent Chat Application

This Streamlit application provides a user interface for interacting with the Property Agent backend. It allows users to enter prompts and receive responses from the agent via a REST API.

## Setup

1.  **Navigate to the Streamlit application directory:**
    ```bash
    cd /Users/uzairkamal/work/my-property-agent/streamlit_app
    ```

2.  **Activate your Python virtual environment:**
    ```bash
    source venv/bin/activate
    ```
    (If you don't have one, create it first: `python3 -m venv venv`)

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

This application requires the `agentP` REST API server to be running in the background.

1.  **Start the Agent REST API Server:**

    Open a **new terminal window**.
    Navigate to the **project root directory** (`/Users/uzairkamal/work/my-property-agent`):
    ```bash
    cd /Users/uzairkamal/work/my-property-agent
    ```
    Activate the virtual environment for `agentP` (if you have one, e.g., `agentP/venv`):
    ```bash
    source agentP/venv/bin/activate
    ```
    Run the `uvicorn` command to start the API server with auto-reloading:
    ```bash
    uvicorn agentP.src.agentRest:app --host 0.0.0.0 --port 8000 --reload
    ```
    Keep this terminal window open and the server running.

2.  **Run the Streamlit Application:**

    In your **first terminal window** (where you set up the Streamlit app), ensure your Streamlit virtual environment is active.
    Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    This will open the Streamlit application in your web browser, which will now send requests to the running `agentP` REST API server.