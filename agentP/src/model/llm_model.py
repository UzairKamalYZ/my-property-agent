from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM as Ollama

from agentP.src.config.config import Config
from agentP.src.model.embedder import Embedder
from agentP.src.model.rag_context_manager import RagContextManager
from agentP.src.model.session_manager import SessionManager


class LlmModel:

    def __init__(self, model_name=Config.LLM_MODEL_NAME):
        self._initialize_components(model_name)
        self._build_chains()


    def ask_direct(self, user_prompt: str, session_id: str, stream: bool = False):
        """Invokes the direct LLM chain without RAG."""
        config = {"configurable": {"session_id": session_id}}
        if stream:
            return self.direct_chain_with_history.stream({"input": user_prompt}, config=config)
        else:
            return self.direct_chain_with_history.invoke({"input": user_prompt}, config=config)

    def ask_with_reformulation(self, user_prompt: str, session_id: str, stream: bool = False):
        """Invokes the full RAG chain which includes prompt reformulation."""
        config = {"configurable": {"session_id": session_id}}
        if stream:
            return self.full_chain.stream(user_prompt, config=config)
        else:
            return self.full_chain.invoke(user_prompt, config=config)


    # ------------------- Chain Builders -------------------

    def _initialize_components(self, model_name: str):
        self.session_manager = SessionManager()
        self.rag_context_manager = RagContextManager(Embedder())
        self.llm = Ollama(model=model_name, seed=365, temperature=0)

    def _build_chains(self):
        self.direct_chain_with_history = self._build_direct_chain_with_history()
        self.reformulation_chain = self._build_reformulation_chain()
        self.rag_chain_with_history = self._build_rag_chain_with_history()
        self.full_chain = self._build_full_chain()

    def _build_direct_chain_with_history(self) -> RunnableWithMessageHistory:
        direct_prompt = self._get_chat_template()
        direct_chain = direct_prompt | self.llm | StrOutputParser()
        return RunnableWithMessageHistory(
            direct_chain,
            self.session_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _build_reformulation_chain(self) -> RunnableWithMessageHistory:
        reformulate_prompt = ChatPromptTemplate.from_template(
            "Based on the user's request, formulate a concise and effective "
            "prompt to search for real estate listings. The user's request is: '{user_prompt}'"
        )
        return reformulate_prompt | self.llm | StrOutputParser()

    def _build_rag_chain_with_history(self) -> RunnableWithMessageHistory:
        rag_retrieval_chain = RunnablePassthrough.assign(
            context=lambda x: self.rag_context_manager.get_context(x["question"])
        )
        rag_prompt = self._get_rag_prompt_template()
        rag_chain = rag_retrieval_chain | rag_prompt | self.llm | StrOutputParser()
        return RunnableWithMessageHistory(
            rag_chain,
            self.session_manager.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def _build_full_chain(self) -> RunnableWithMessageHistory:
        # This is the final chain the user requested: formulated-prompt | ask again
        # It takes a "user_prompt", reformulates it, and pipes the result to the RAG chain.
        # We add a lambda to reshape the string output from the first chain into a
        # dictionary for the second chain.
        return (
            self.reformulation_chain
            | RunnableLambda(lambda x: {"question": x})
            | self.rag_chain_with_history
        )


    @staticmethod
    def _get_chat_template() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", Config.PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

    @staticmethod
    def _get_rag_prompt_template() -> ChatPromptTemplate:
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        return ChatPromptTemplate.from_template(template)


    def close(self):
        """Cleanup hook (not needed for Ollama)."""
        pass