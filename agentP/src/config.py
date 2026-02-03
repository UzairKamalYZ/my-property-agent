import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the application."""
    STORE_TYPE = os.getenv("STORE_TYPE", "local")
    LLM_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen3:1.7b")
    SENTENCE_TRANSFORMER_MODEL=  os.getenv("SENTENCE_TRANSFORMER_MODEL","sentence-transformers/all-MiniLM-L6-v2")
    MEMORY_FILE = os.getenv("MEMORY_FILE", "memory.json")
    URLS_FILE = os.getenv("URLS_FILE", "../urls.txt")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2khez9_6ERgE39wfzAjTg55WxhidPj7JTZ2tHQxCmxMvnQ8LFubPRWsc4pgzGrhoeMC8pZ")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "course-ai")

   ####### PROMPT

    PROMPT = os.getenv("PROMPT", """
        Here are 2 results:
1. Apartment in Szczecin – Rent €2200 – A modern two-bedroom apartment close to schools and restaurants. https://example.com
2. Apartment in Szczecin – Rent €2100 – A well-located city apartment ideal for a small family. https://example.com

Would you prefer something closer to the city center, or is a quieter neighborhood more important to you?

 """)
    ###### PROMPT
#     PROMPT = os.getenv("PROMPT","""
# You are a professional property agent.
# Your task is to suggest apartments or houses based on the user's request.
# ⚠️ Output rules (MUST follow):
# - Start with: "Here are <number> results:"
# - Number them 1 to <number>
# - Each result MUST contain:
#   - Property type (Apartment or House)
#   - Location (city)
#   - Monthly rent in EUR
#   - Short description (1 sentence)
# Format EXACTLY like this:
# Here are  results:
# 1. Apartment in <City> – Rent €<amount> – <short description> <link to website>
# 2. House in <City> – Rent €<amount> – <short description> <link to website>
# 3. Apartment in <City> – Rent €<amount> – <short description> <link to website>
# Do not add any extra text, IDs and technical things, just a beautifuly crafted story
#
# At the end ask user more information like what are you prefernces, what you want to see in a dream house to rent or sale depend upon his
# query.
# """)