import json
import logging
import os
import re
import uuid
from typing import List, TypedDict

logger = logging.getLogger(__name__)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage, ToolMessage
from langsmith import traceable

from langgraph.graph import StateGraph, START, END

from ..config.config import Config
from .embedder import Embedder
from .mcp_tools import close_mcp, _mcp
from .rag_context_manager import RagContextManager

os.environ.setdefault("LANGCHAIN_TRACING_V2", Config.LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_PROJECT", Config.LANGCHAIN_PROJECT)
if Config.LANGCHAIN_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", Config.LANGCHAIN_API_KEY)


# ------------------- STATE -------------------

class State(TypedDict):
    user_prompt: str
    context: str
    answer: str
    session_history: List[AnyMessage]


# ------------------- MODEL -------------------

class LlmModelGraph:

    def __init__(self, llm: BaseChatModel):
        self.system_prompt = self._load_file(Config.PROMPT_FILE)

        self.rag_context_manager = RagContextManager(Embedder())
        self._histories: dict[str, list] = {}

        # Bind all registered MCP tools so the LLM can call them natively.
        self._tools = _mcp.langchain_tools()
        self.llm = llm.bind_tools(self._tools) if self._tools else llm

        self.graph = self._build_graph()

    # ------------------- PUBLIC API -------------------

    @traceable(name="LlmModelGraph.ask", run_type="chain", tags=["property-agent"])
    def ask(self, user_query: str, session_id: str = None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())

        history = self._get_history(session_id)
        state = self._initial_state(user_query, history)

        result = self.graph.invoke(state)
        answer = result["answer"]

        self._histories[session_id] = history + [
            HumanMessage(content=user_query),
            AIMessage(content=answer),
        ]
        return answer

    @traceable(name="LlmModelGraph.ask_stream", run_type="chain", tags=["streaming"])
    def ask_stream(self, user_query: str, session_id: str = None):
        if session_id is None:
            session_id = str(uuid.uuid4())

        history = self._get_history(session_id)
        state = self._initial_state(user_query, history)
        full_answer = ""

        for chunk, metadata in self.graph.stream(
            state,
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") == "generate" and hasattr(chunk, "content"):
                token = chunk.content
                if token:
                    full_answer += token
                    yield token

        self._histories[session_id] = history + [
            HumanMessage(content=user_query),
            AIMessage(content=full_answer),
        ]

    def close(self):
        close_mcp()
        logger.debug("LlmModelGraph closed")

    # ------------------- GRAPH -------------------

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    # ------------------- NODES -------------------

    def _retrieve_node(self, state: State) -> dict:
        context = self.rag_context_manager.get_context(state["user_prompt"])
        return {"context": context}

    def _generate_node(self, state: State) -> dict:
        messages: list = [SystemMessage(content=self.system_prompt)]

        if state["context"]:
            messages.append(SystemMessage(content=f"Relevant property listings:\n{state['context']}"))

        messages.extend(state["session_history"])
        messages.append(HumanMessage(content=state["user_prompt"]))

        # Tool call loop — the LLM may invoke MCP tools (e.g. currency_convert)
        # before producing a final answer.
        # Ollama models that lack native function-calling output tool calls as
        # JSON text in response.content instead of populating response.tool_calls.
        # _parse_json_tool_call() handles that fallback transparently.
        response = self.llm.invoke(messages)

        while True:
            tool_calls = response.tool_calls or []

            if not tool_calls:
                json_call = self._parse_json_tool_call(response.content)
                if json_call:
                    tool_calls = [json_call]

            if not tool_calls:
                break

            messages.append(response)
            for tool_call in tool_calls:
                result = self._invoke_tool(tool_call["name"], tool_call.get("args", tool_call.get("parameters", {})))
                messages.append(ToolMessage(content=result, tool_call_id=tool_call.get("id", "0")))
                logger.info("tool '%s' executed, result_len=%d", tool_call["name"], len(result))
            response = self.llm.invoke(messages)

        return {"answer": response.content if hasattr(response, "content") else str(response)}

    # ------------------- HELPERS -------------------

    @staticmethod
    def _parse_json_tool_call(text: str) -> dict | None:
        """
        Detect and parse a tool call that Ollama emitted as JSON text rather than
        as a native function call.  Returns a normalised dict with 'name', 'args',
        and 'id' keys, or None if the content is not a tool call.
        """
        if not text:
            return None
        try:
            # Strip optional markdown code fences.
            cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL).strip()
            data = json.loads(cleaned)
            if isinstance(data, dict) and "name" in data:
                args = data.get("parameters") or data.get("arguments") or data.get("args") or {}
                return {"id": "0", "name": data["name"], "args": args}
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _invoke_tool(self, name: str, args: dict) -> str:
        for tool in self._tools:
            if tool.name == name:
                return tool.invoke(args)
        return f"Tool '{name}' not found."

    def _get_history(self, session_id: str) -> list:
        return self._histories.setdefault(session_id, [])

    def _initial_state(self, user_query: str, history: list = None) -> State:
        return {
            "user_prompt": user_query,
            "context": "",
            "answer": "",
            "session_history": list(history) if history else [],
        }

    @staticmethod
    def _load_file(path: str) -> str:
        with open(path, "r") as f:
            return f.read()
