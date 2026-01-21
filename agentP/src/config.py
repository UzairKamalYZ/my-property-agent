import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the application."""

    LLM_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen3:1.7b")
    MEMORY_FILE = os.getenv("MEMORY_FILE", "memory.json")
    URLS_FILE = os.getenv("URLS_FILE", "../urls.txt")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2khez9_6ERgE39wfzAjTg55WxhidPj7JTZ2tHQxCmxMvnQ8LFubPRWsc4pgzGrhoeMC8pZ")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "course-ai")