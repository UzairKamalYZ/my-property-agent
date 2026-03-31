import logging
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import OllamaLLM as Ollama

from agentP.src.config.config import Config

logger = logging.getLogger(__name__)


def create_llm(provider: str, model_name: str) -> BaseLanguageModel:
    """
    Factory function to create an LLM instance based on the provider.

    :param provider: The name of the LLM provider (e.g., 'ollama').
    :param model_name: The name of the model to use.
    :return: An instance of a LangChain BaseLanguageModel.
    :raises ValueError: If the provider is not supported.
    """
    logger.info("Creating LLM provider=%s model=%s", provider, model_name)
    if provider == "ollama":
        llm = Ollama(
            model=model_name,
            seed=Config.LLM_SEED,
            temperature=Config.LLM_TEMPERATURE
        )
        logger.debug("Ollama LLM created model=%s seed=%s temperature=%s",
                     model_name, Config.LLM_SEED, Config.LLM_TEMPERATURE)
        return llm
    else:
        logger.error("Unsupported LLM provider: %s", provider)
        raise ValueError(f"Unsupported LLM provider: {provider}")
