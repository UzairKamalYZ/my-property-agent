from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import OllamaLLM as Ollama

from config.config import Config


def create_llm(provider: str, model_name: str) -> BaseLanguageModel:
    """
    Factory function to create an LLM instance based on the provider.

    :param provider: The name of the LLM provider (e.g., 'ollama').
    :param model_name: The name of the model to use.
    :return: An instance of a LangChain BaseLanguageModel.
    :raises ValueError: If the provider is not supported.
    """
    if provider == "ollama":
        return Ollama(
            model=model_name,
            seed=Config.LLM_SEED,
            temperature=Config.LLM_TEMPERATURE
        )
    # Add other providers here in the future, e.g.:
    # elif provider == "openai":
    #     return OpenAI(...)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
