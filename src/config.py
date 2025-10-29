import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the application."""

    QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen3:1.7b")
    MEMORY_FILE = os.getenv("MEMORY_FILE", "memory.json")
