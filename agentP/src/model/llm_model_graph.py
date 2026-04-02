import logging
import os
import re
import uuid
from typing import List, TypedDict

logger = logging.getLogger(__name__)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import traceable

from langgraph.graph import StateGraph, START, END

from ..config.config import Config
from .embedder import Embedder
from .finance_tools import close_finance_mcp, _mcp
from .rag_context_manager import RagContextManager

os.environ.setdefault("LANGCHAIN_TRACING_V2", Config.LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_PROJECT", Config.LANGCHAIN_PROJECT)
if Config.LANGCHAIN_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", Config.LANGCHAIN_API_KEY)

_NON_USD_CURRENCY_RE = re.compile(r'\b(PLN|EUR|GBP|CHF|NOK|SEK|DKK|CZK|HUF|RON)\b')


# ------------------- STATE -------------------

class State(TypedDict):
    user_prompt: str
    reformulated_question: str
    needs_search: bool
    needs_currency_conversion: bool
    context: str
    answer: str
    session_history: List[AnyMessage]


# ------------------- MODEL -------------------

class LlmModelGraph:

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.system_prompt = self._load_file(Config.PROMPT_FILE)
        self.reformulation_template = self._load_file(Config.REFORMULATION_PROMPT)

        self.rag_context_manager = RagContextManager(Embedder())

        self._histories: dict[str, list] = {}
        self._currency_cache: dict[str, str] = {}  # 🔥 caching

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
        close_finance_mcp()
        logger.debug("LlmModelGraph closed")

    # ------------------- GRAPH -------------------

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("reformulate", self._reformulate_node)
        graph.add_node("classify", self._classify_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("currency_check", self._currency_check_node)
        graph.add_node("currency_convert", self._currency_convert_node)
        graph.add_node("generate", self._generate_node)

        graph.add_edge(START, "reformulate")
        graph.add_edge("reformulate", "classify")

        graph.add_conditional_edges(
            "classify",
            lambda state: "retrieve" if state["needs_search"] else "generate",
        )

        graph.add_edge("retrieve", "currency_check")

        graph.add_conditional_edges(
            "currency_check",
            lambda state: "currency_convert" if state["needs_currency_conversion"] else "generate",
        )

        graph.add_edge("currency_convert", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    # ------------------- NODES -------------------

    def _reformulate_node(self, state: State) -> dict:
        prompt = ChatPromptTemplate.from_template(self.reformulation_template)
        chain = prompt | self.llm | StrOutputParser()
        reformulated = chain.invoke({"user_prompt": state["user_prompt"]})
        return {"reformulated_question": reformulated}

    def _classify_node(self, state: State) -> dict:
        prompt = ChatPromptTemplate.from_template(
            "You are a query router. Decide whether this query needs property search.\n"
            "Answer YES or NO.\n\nQuery: {query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": state["reformulated_question"]})
        return {"needs_search": result.strip().upper().startswith("YES")}

    def _retrieve_node(self, state: State) -> dict:
        context = self.rag_context_manager.get_context(state["reformulated_question"])
        return {"context": context}

    def _currency_check_node(self, state: State) -> dict:
        prompt = ChatPromptTemplate.from_template(
            """
            Does the user need currency conversion to USD?

            Answer YES if:
            - USD mentioned
            - conversion implied
            - comparison needed

            Otherwise NO.

            Query: {query}
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": state["reformulated_question"]})
        return {"needs_currency_conversion": result.strip().upper().startswith("YES")}

    def _currency_convert_node(self, state: State) -> dict:
        context = state["context"]
        currencies = sorted(set(_NON_USD_CURRENCY_RE.findall(context)))

        if not currencies:
            return {}

        rates = []
        for currency in currencies:
            if currency not in self._currency_cache:
                rate = _mcp.call_tool(
                    "currency_convert",
                    {"from": currency, "to": "USD", "amount": 1}
                )
                self._currency_cache[currency] = rate
            else:
                rate = self._currency_cache[currency]

            if rate:
                rates.append(f"1 {currency} = {rate}")

        if rates:
            context += "\n\nCurrency conversion rates:\n" + "\n".join(rates)

        return {"context": context}

    def _generate_node(self, state: State) -> dict:
        messages = [("system", self.system_prompt)]

        if state["context"]:
            messages.append(("system", f"Relevant property listings:\n{state['context']}"))

        messages.append(MessagesPlaceholder(variable_name="history"))
        messages.append(("human", "{question}"))

        prompt = ChatPromptTemplate.from_messages(messages)

        formatted = prompt.invoke({
            "question": state["reformulated_question"],
            "history": state["session_history"],
        })

        response = self.llm.invoke(formatted)
        answer = response.content if hasattr(response, "content") else str(response)

        return {"answer": answer}

    # ------------------- HELPERS -------------------

    def _get_history(self, session_id: str) -> list:
        return self._histories.setdefault(session_id, [])

    def _initial_state(self, user_query: str, history: list = None) -> State:
        return {
            "user_prompt": user_query,
            "reformulated_question": "",
            "needs_search": False,
            "needs_currency_conversion": False,
            "context": "",
            "answer": "",
            "session_history": list(history) if history else [],
        }

    @staticmethod
    def _load_file(path: str) -> str:
        with open(path, "r") as f:
            return f.read()