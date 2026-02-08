from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSerializable
from langchain_core.runnables.history import RunnableWithMessageHistory

from agentP.src.config.config import Config
from agentP.src.model.embedder import Embedder
from agentP.src.model.rag_context_manager import RagContextManager
from agentP.src.model.session_manager import SessionManager


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

        config = {"configurable": {"session_id": session_id}}

        input_for_full_chain = {"user_prompt": user_query}

        if stream:
            return self.full_chain.stream(input_for_full_chain, config)
        else:
            return self.full_chain.invoke(input_for_full_chain, config)

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
