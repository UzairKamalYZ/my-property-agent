from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from src.config import Config

class LlmModel:
    """Wrapper for local language model using Langchain and Ollama."""

    def __init__(self, model_name=Config.LLM_MODEL_NAME):
        self.llm = Ollama(model=model_name)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.chain = self.prompt | self.llm
        self.store = {}

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def chat(self, prompt: str, session_id: str, stream=False):
        """Send a prompt to the language model."""
        runnable_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        config = {"configurable": {"session_id": session_id}}
        if stream:
            return runnable_with_history.stream({"input": prompt}, config=config)
        else:
            return runnable_with_history.invoke({"input": prompt}, config=config)

    def close(self):
        """Close the model and release resources."""
        pass