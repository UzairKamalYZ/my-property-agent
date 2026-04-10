import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parents[3]

# Always load from the project root .env, regardless of where the process is launched
load_dotenv(_PROJECT_ROOT / ".env")

class Config:
    """Configuration class for the application."""
    STORE_TYPE = os.getenv("STORE_TYPE")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER")
    LLM_SEED = int(os.getenv("LLM_SEED"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE"))
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL")
    MEMORY_FILE = os.getenv("MEMORY_FILE")
    URLS_FILE = os.getenv("URLS_FILE")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    PROMPT_FILE = os.path.join(os.getenv("PROMPT_FILE"))
    INTERACTION_FILE = os.path.join(os.getenv("INTERACTION_FILE"))
    REFORMULATION_PROMPT = os.path.join(os.getenv("REFORMULATION_PROMPT"))
    # Telegram / LangSmith integrations
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    AI_PROVIDER_BASE_URL = os.getenv("AI_PROVIDER_BASE_URL", "http://localhost:11434/v1")
    AI_PROVIDER_API_KEY = os.getenv("AI_PROVIDER_API_KEY", "ollama")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "my-property-agent")
    # API security — leave empty to disable auth
    API_KEY = os.getenv("API_KEY", "")

    # SQLite file for persistent session history
    SESSION_DB_FILE = os.getenv("SESSION_DB_FILE", "sessions.db")
    # Selenium scraping browser connection string
    SBR_WEBDRIVER = os.getenv("SBR_WEBDRIVER", "")
    # Number of listings to retrieve per RAG search
    RAG_K = int(os.getenv("RAG_K", "5"))
    MAX_HISTORY_TURNS= os.getenv("MAX_HISTORY_TURNS", 10)
    # Orchestrator prompt files
    SUPERVISOR_PROMPT_FILE = os.getenv("SUPERVISOR_PROMPT_FILE", "orchestrator/prompts/supervisor.txt")
    SYNTHESISER_PROMPT_FILE = os.getenv("SYNTHESISER_PROMPT_FILE", "orchestrator/prompts/synthesiser.txt")
    # Registry config files — resolved to absolute paths from the project root
    AGENTS_FILE = str(_PROJECT_ROOT / os.getenv("AGENTS_FILE", "agents.json"))
    MCP_FILE    = str(_PROJECT_ROOT / os.getenv("MCP_FILE",    "mcp.json"))
