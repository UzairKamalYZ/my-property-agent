from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config.config import Config


def create_llm(provider: str, model_name: str) -> BaseChatModel:
    """
    Factory function to create an LLM instance based on the provider.

    :param provider: The name of the LLM provider (e.g., 'ollama').
    :param model_name: The name of the model to use.
    :return: An instance of a LangChain BaseChatModel.
    :raises ValueError: If the provider is not supported.
    """
    if provider == "ollama":
        return ChatOpenAI(
            model=model_name,
            base_url=Config.AI_PROVIDER_BASE_URL,
            api_key=Config.AI_PROVIDER_API_KEY,
            temperature=Config.LLM_TEMPERATURE,
            model_kwargs={"seed": Config.LLM_SEED},
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
