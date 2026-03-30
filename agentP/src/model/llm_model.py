from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSerializable
from langchain_core.runnables.history import RunnableWithMessageHistory

from config.config import Config
from .embedder import Embedder
from .rag_context_manager import RagContextManager
from .session_manager import SessionManager


class LlmModel:

    def __init__(self, llm: BaseLanguageModel):
        self.system_prompt = LlmModel.getPrompt(Config.PROMPT_FILE)
        self._initialize_components(llm)
        self._build_chains()

    def _initialize_components(self, llm: BaseLanguageModel):
        self.session_manager = SessionManager()
        self.rag_context_manager = RagContextManager(Embedder())
        self.llm = llm

    def _build_chains(self):
        self.direct_chain_with_history = self._build_direct_chain_with_history()
        self.reformulation_chain = self._build_reformulation_chain()
        self.rag_chain_with_history = self._build_rag_chain_with_history()
        self.full_chain = self._build_full_chain()

    # ------------------- Public Methods -------------------
    def ask(self, system_prompt: str, user_query: str, session_id: str, stream=False):
        """Invokes the full RAG chain which includes prompt reformulation."""
        return self.ask_with_reformulation(user_query, session_id, stream=stream)

    def ask_direct(self, user_prompt: str, session_id: str) -> str:
        """Invokes the direct chain (no RAG, no reformulation)."""
        config = {"configurable": {"session_id": session_id}}
        return self.direct_chain_with_history.invoke({"input": user_prompt}, config=config)

    def ask_with_reformulation(self, user_prompt: str, session_id: str, stream=False):
        """Invokes the full RAG chain with query reformulation."""
        config = {"configurable": {"session_id": session_id}}
        if stream:
            return self.full_chain.stream(user_prompt, config=config)
        return self.full_chain.invoke(user_prompt, config=config)

    # ------------------- Private Chain Builders -------------------

    def _build_direct_chain_with_history(self) -> RunnableWithMessageHistory:
        direct_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        direct_chain = direct_prompt | self.llm | StrOutputParser()
        return RunnableWithMessageHistory(
            direct_chain,
            self.session_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _build_reformulation_chain(self) -> RunnableSerializable[dict, str]:
        reformulate_prompt = LlmModel._reformulated_prompt_template()
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
        # Accepts a raw string, wraps it for the reformulation chain, then pipes
        # the reformulated string into the RAG chain.
        return (
            RunnableLambda(lambda x: {"user_prompt": x})
            | self.reformulation_chain
            | RunnableLambda(lambda x: {"question": x})
            | self.rag_chain_with_history
        )



    @staticmethod
    def _get_rag_prompt_template() -> ChatPromptTemplate:
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        return ChatPromptTemplate.from_template(template)

    @staticmethod
    def getPrompt(file) -> str:
        with open(file, "r") as f:
            template = f.read()
        return template

    @staticmethod
    def _reformulated_prompt_template() -> ChatPromptTemplate:
        template = LlmModel.getPrompt(Config.REFORMULATION_PROMPT)
        return ChatPromptTemplate.from_template(template)
    def close(self):
        """Cleanup hook (not needed for Ollama)."""
        pass
