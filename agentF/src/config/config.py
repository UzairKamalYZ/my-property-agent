import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")


class FinanceConfig:
    """Configuration for the finance agent."""

    FINANCE_PROMPT_FILE = os.getenv(
        "FINANCE_PROMPT_FILE",
        str(Path(__file__).parents[1] / "prompts" / "System_Prompt.txt"),
    )
    FINANCE_PINECONE_INDEX_NAME = os.getenv(
        "FINANCE_PINECONE_INDEX_NAME",
        os.getenv("PINECONE_INDEX_NAME", "finance-agent"),
    )
