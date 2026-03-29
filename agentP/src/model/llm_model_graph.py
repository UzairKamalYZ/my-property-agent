import os
import uuid
from typing import TypedDict, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import traceable

from langgraph.graph import StateGraph, START, END

from src.config.config import Config
from src.model.embedder import Embedder
from src.model.rag_context_manager import RagContextManager

# LangSmith picks these up automatically from the environment
os.environ.setdefault("LANGCHAIN_TRACING_V2", Config.LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_PROJECT", Config.LANGCHAIN_PROJECT)
if Config.LANGCHAIN_API_KEY:
     os.environ.setdefault("LANGCHAIN_API_KEY", Config.LANGCHAIN_API_KEY)


# ------------------- STATE -------------------

class State(TypedDict):
    user_prompt: str
    reformulated_question: str
    context: str
    answer: str
    history: List


# ------------------- MODEL -------------------

class LlmModelGraph:
    """LangGraph-based reimplementation of LlmModel: reformulate → retrieve → generate."""

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.system_prompt = self._load_file(Config.PROMPT_FILE)
        self.reformulation_template = self._load_file(Config.REFORMULATION_PROMPT)
        self.rag_context_manager = RagContextManager(Embedder())
        self.history: List = []
        self.session_id = str(uuid.uuid4())
        self.graph = self._build_graph()

    # ------------------- PUBLIC -------------------

    @traceable(name="LlmModelGraph.ask", run_type="chain", tags=["property-agent"])
    def ask(self, user_query: str, session_id: str = None) -> str:
        state = self._initial_state(user_query)
        result = self.graph.invoke(state, config={"run_name": "property-rag-pipeline"})
        self.history = result["history"]
        return result["answer"]

    @traceable(name="LlmModelGraph.ask_stream", run_type="chain", tags=["property-agent", "streaming"])
    def ask_stream(self, user_query: str, session_id: str = None):
        """Yields token chunks from the generate node."""
        state = self._initial_state(user_query)
        full_answer = ""
        for chunk, metadata in self.graph.stream(
            state,
            stream_mode="messages",
            config={"run_name": "property-rag-pipeline-stream"},
        ):
            if metadata.get("langgraph_node") == "generate" and hasattr(chunk, "content"):
                token = chunk.content
                if token:
                    full_answer += token
                    yield token

        self.history = self.history + [
            HumanMessage(content=user_query),
            AIMessage(content=full_answer),
        ]

    def close(self):
        pass

    # ------------------- GRAPH -------------------

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("reformulate", self._reformulate_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)

        graph.add_edge(START, "reformulate")
        graph.add_edge("reformulate", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    # ------------------- NODES -------------------

    def _reformulate_node(self, state: State) -> dict:
        prompt = ChatPromptTemplate.from_template(self.reformulation_template)
        chain = prompt | self.llm | StrOutputParser()
        reformulated = chain.invoke({"user_prompt": state["user_prompt"]})
        return {"reformulated_question": reformulated}

    def _retrieve_node(self, state: State) -> dict:
        context = self.rag_context_manager.get_context(state["reformulated_question"])
        return {"context": context}

    def _generate_node(self, state: State) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
            ("system", "Relevant property listings:\n{context}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "question": state["reformulated_question"],
            "context": state["context"],
            "history": state["history"],
        })
        updated_history = state["history"] + [
            HumanMessage(content=state["user_prompt"]),
            AIMessage(content=answer),
        ]
        return {"answer": answer, "history": updated_history}

    # ------------------- HELPERS -------------------

    def _initial_state(self, user_query: str) -> State:
        return {
            "user_prompt": user_query,
            "reformulated_question": "",
            "context": "",
            "answer": "",
            "history": list(self.history),
        }

    @staticmethod
    def _load_file(path: str) -> str:
        with open(path, "r") as f:
            return f.read()
