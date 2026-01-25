import time
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from agentP.src.config import Config
from agentP.src.model.embedder import Embedder


class LlmModel:

    def __init__(self, model_name=Config.LLM_MODEL_NAME):
        self.llm = Ollama(
            model=model_name,
            seed=365,
            temperature=0
        )

        self.embedder = Embedder()
        self.store = {}

        self.chat_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    Config.PROMPT
                    + "\n\n"
                    + "Listings are provided per question and may change. "
                    + "Do not rely on listings from previous messages."
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        self.chain = self.chat_template | self.llm

    # -------------------------------
    # Vector Retrieval
    # -------------------------------

    def retrieve(self, query: str, k: int = 5):
        store = self.embedder.__getStore__()
        if store is None:
            raise ValueError("Vector index not initialized. Build embeddings first.")

        query_vec = self.embedder.embed_query(query)
        return store.search(query_vec, k)

    # -------------------------------
    # Context Builder
    # -------------------------------

    def _build_context(self, listings):
        if not listings:
            return "No listings found."

        blocks = []
        for i, l in enumerate(listings, 1):
            blocks.append(
                f"""
                Listing {i}:
                Location: {l.get("city", "N/A")}
                Price: {l.get("price", "N/A")} EUR
                Bedrooms: {l.get("bedrooms", "N/A")}
                Surface: {l.get("surface", "N/A")} m²
                Type: {l.get("type", "N/A")}
                Description: {l.get("description", "N/A")}
                """
            )

        return "\n".join(blocks)

    # -------------------------------
    # Session Memory
    # -------------------------------

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    # -------------------------------
    # Chat Entry Point (RAG)
    # -------------------------------

    def chat(self, user_prompt: str, session_id: str, stream: bool = False):

        runnable = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        start = time.perf_counter()

        # 1. Retrieve relevant listings
        listings = self.retrieve(user_prompt, k=5)

        # 2. Build LLM-ready context
        context = self._build_context(listings)

        # 3. Merge context into input
        full_input = f"""
                        User question:
                        {user_prompt}
                        
                        Available listings:
                        {context}
                        
                        Answer strictly using the listings above.
                        If no listings match, say "No matching listings found".
                    """

        payload = {"input": full_input}

        config = {"configurable": {"session_id": session_id}}

        # 4. Invoke LLM
        if stream:
            return runnable.stream(payload, config=config)
        else:
            response = runnable.invoke(payload, config=config)

        end = time.perf_counter()
        print(f"Time taken: {end - start:.4f} seconds")

        return response

    def close(self):
        """Cleanup hook (not needed for Ollama)."""
        pass
