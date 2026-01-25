import json
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from agentP.src.config import Config
import time
from agentP.src.model.embedder import Embedder


class LlmModel:
    """Wrapper for local language model using Langchain and Ollama."""

    def __init__(self, model_name=Config.LLM_MODEL_NAME):
        self.llm = Ollama(model=model_name, seed=365, temperature=0)
        prompt = Config.PROMPT
        self.embedder = Embedder()

        self.chat_template = ChatPromptTemplate.from_messages(
            [
                ("system",
                 prompt 
                 ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.chain = self.chat_template | self.llm
        self.store = {}

    def rank_listings(self, query: str, k: int = 5):
        """Return top-k listings by semantic similarity."""
        __store__ = self.embedder.__getStore__()
        if __store__ is None:
            raise ValueError("Index not built yet. Call embed() first.")

        query_vec = self.embedder.embed_query(query)
        top_docs = __store__.search(query_vec, k)
        return top_docs

    def __getRankedListings__(self, query="Apartments or House in Belgium under 1500 EUR", results=5):
        rank_listings = self.rank_listings(
            "%s" % query, k=results)
        return rank_listings
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def chat(self, user_prompt: str, session_id: str, stream=False):
        """Send a prompt to the language model."""  
        runnable_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        config = {"configurable": {"session_id": session_id}}
        start = time.perf_counter()
        listings = self.__getRankedListings__(query=user_prompt, results=5)
        print("Top listings based on embedding search:")
        for i, listing in enumerate(listings, 1):
            print(f"{i}. {listing}")
        payload = {
                "input": user_prompt,
                "listings": json.dumps(listings, indent=2),
                }

        if stream:
            return runnable_with_history.stream(payload, config=config)
        else:
           return runnable_with_history.invoke(payload, config=config)

        end = time.perf_counter()
        print(f"Time taken: {end - start:.4f} seconds")

    def close(self):
        """Close the model and release resources."""
        pass