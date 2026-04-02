import logging
import os
import uuid
from typing import Annotated, List, TypedDict

logger = logging.getLogger(__name__)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    AIMessageChunk,
    AnyMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ..config.config import Config
from .embedder import Embedder
from .rag_context_manager import RagContextManager
from .tools import make_property_search_tool

# LangSmith picks these up automatically from the environment
os.environ.setdefault("LANGCHAIN_TRACING_V2", Config.LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_PROJECT", Config.LANGCHAIN_PROJECT)
if Config.LANGCHAIN_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", Config.LANGCHAIN_API_KEY)

# Maximum agent→tools→agent loop iterations before LangGraph raises an error.
_RECURSION_LIMIT = 10


# ------------------- STATE -------------------

class State(TypedDict):
    user_prompt: str
    reformulated_question: str
    messages: Annotated[List[AnyMessage], add_messages]
    session_history: List[AnyMessage]  # per-session conversation history passed into the graph


# ------------------- MODEL -------------------

class LlmModelGraph:
    """LangGraph agent that calls the property-search RAG tool only when needed."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.system_prompt = self._load_file(Config.PROMPT_FILE)
        self.reformulation_template = self._load_file(Config.REFORMULATION_PROMPT)
        self.rag_context_manager = RagContextManager(Embedder())
        # Per-session history store: {session_id: [HumanMessage, AIMessage, ...]}
        self._histories: dict[str, list] = {}
        self._property_search_tool = make_property_search_tool(self.rag_context_manager)
        self._llm_with_tools = self.llm.bind_tools([self._property_search_tool])
        self.graph = self._build_graph()

    # ------------------- PUBLIC -------------------

    @traceable(name="LlmModelGraph.ask", run_type="chain", tags=["property-agent"])
    def ask(self, user_query: str, session_id: str = None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.debug("ask: no session_id provided, generated %s", session_id)

        history = self._get_history(session_id)
        state = self._initial_state(user_query, history)
        result = self.graph.invoke(
            state,
            config={"run_name": "property-rag-pipeline", "recursion_limit": _RECURSION_LIMIT},
        )
        answer = result["messages"][-1].content
        self._histories[session_id] = history + [
            HumanMessage(content=user_query),
            AIMessage(content=answer),
        ]
        return answer

    @traceable(name="LlmModelGraph.ask_stream", run_type="chain", tags=["property-agent", "streaming"])
    def ask_stream(self, user_query: str, session_id: str = None):
        """Yields text token chunks from the agent node, skipping tool-call chunks."""
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.debug("ask_stream: no session_id provided, generated %s", session_id)

        history = self._get_history(session_id)
        state = self._initial_state(user_query, history)
        full_answer = ""
        for chunk, metadata in self.graph.stream(
            state,
            stream_mode="messages",
            config={"run_name": "property-rag-pipeline-stream", "recursion_limit": _RECURSION_LIMIT},
        ):
            if (
                metadata.get("langgraph_node") == "agent"
                and isinstance(chunk, AIMessageChunk)
                and chunk.content
                and not chunk.tool_call_chunks
            ):
                full_answer += chunk.content
                yield chunk.content
            else:
                logger.debug(
                    "ask_stream: skipping chunk node=%s has_content=%s tool_calls=%s",
                    metadata.get("langgraph_node"),
                    bool(chunk.content),
                    bool(getattr(chunk, "tool_call_chunks", [])),
                )

        self._histories[session_id] = history + [
            HumanMessage(content=user_query),
            AIMessage(content=full_answer),
        ]

    def close(self):
        pass

    # ------------------- GRAPH -------------------

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("reformulate", self._reformulate_node)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode([self._property_search_tool]))

        graph.add_edge(START, "reformulate")
        graph.add_edge("reformulate", "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")

        return graph.compile()

    # ------------------- NODES -------------------

    def _reformulate_node(self, state: State) -> dict:
        logger.info("[reformulate] input: %s", state["user_prompt"])
        prompt = ChatPromptTemplate.from_template(self.reformulation_template)
        chain = prompt | self.llm | StrOutputParser()
        reformulated = chain.invoke({"user_prompt": state["user_prompt"]})
        logger.info("[reformulate] output: %s", reformulated)
        return {"reformulated_question": reformulated}

    def _agent_node(self, state: State) -> dict:
        if not state["messages"]:
            # First call: build message list from system prompt + session history + reformulated question
            logger.info("[agent] first call — building messages from history + reformulated question")
            messages = (
                [SystemMessage(content=self.system_prompt)]
                + list(state["session_history"])
                + [HumanMessage(content=state["reformulated_question"])]
            )
        else:
            # Subsequent call after tool execution: pass accumulated messages as-is
            logger.info("[agent] subsequent call — %d messages in context", len(state["messages"]))
            messages = state["messages"]

        response = self._llm_with_tools.invoke(messages)
        logger.info("[agent] has_tool_calls: %s", bool(getattr(response, "tool_calls", [])))
        return {"messages": [response]}

    # ------------------- HELPERS -------------------

    def _get_history(self, session_id: str) -> list:
        """Returns the history list for the given session, creating it if absent."""
        return self._histories.setdefault(session_id, [])

    def _initial_state(self, user_query: str, history: list = None) -> State:
        return {
            "user_prompt": user_query,
            "reformulated_question": "",
            "messages": [],
            "session_history": list(history) if history else [],
        }

    @staticmethod
    def _load_file(path: str) -> str:
        with open(path, "r") as f:
            return f.read()
