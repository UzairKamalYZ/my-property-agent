import os
from dotenv import load_dotenv

load_dotenv()

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
    PROMPT_FILE = os.path.join( os.getenv("PROMPT_FILE"))
    INTERACTION_FILE = os.path.join(os.getenv("INTERACTION_FILE"))
    REFORMULATION_PROMPT = os.path.join(os.getenv("REFORMULATION_PROMPT"))