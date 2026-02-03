import time
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from agentP.src.config import Config
from agentP.src.model.rag_context_manager import RagContextManager
from agentP.src.model.session_manager import SessionManager


class LlmModel:

    def __init__(self, model_name=Config.LLM_MODEL_NAME):
        self.session_manager = SessionManager()
        self.rag_context_manager = RagContextManager()

        self.llm = Ollama(
            model=model_name,
            seed=365,
            temperature=0
        )

        chat_template = self._get_chat_template()
        chain = chat_template | self.llm

        self.runnable_with_history = RunnableWithMessageHistory(
            chain,
            self.session_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    @staticmethod
    def _get_chat_template() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
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

    def _invoke_runnable(self, payload: dict, config: dict, stream: bool = False):
        if stream:
            return self.runnable_with_history.stream(payload, config=config)
        else:
            return self.runnable_with_history.invoke(payload, config=config)

    # -------------------------------
    # Chat Entry Point (Direct)
    # -------------------------------
    def ask(self, user_prompt: str, session_id: str, stream: bool = False):
        payload = {"input": user_prompt}
        config = {"configurable": {"session_id": session_id}}
        return self._invoke_runnable(payload, config, stream)

    # -------------------------------
    # Chat Entry Point (RAG)
    # -------------------------------
    def chat_with_context(self, user_prompt: str, session_id: str, stream: bool = False):
        start = time.perf_counter()

        # 1. Prepare RAG prompt
        full_input = self.rag_context_manager.prepare_rag_prompt(user_prompt)

        # 2. Invoke LLM
        payload = {"input": full_input}
        config = {"configurable": {"session_id": session_id}}
        response = self._invoke_runnable(payload, config, stream)

        end = time.perf_counter()
        print(f"Time taken: {end - start:.4f} seconds")

        return response

    def close(self):
        """Cleanup hook (not needed for Ollama)."""
        pass