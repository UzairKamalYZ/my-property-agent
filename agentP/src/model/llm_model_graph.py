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

# LangSmith picks these up automatically from the environment
os.environ.setdefault("LANGCHAIN_TRACING_V2", Config.LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_PROJECT", Config.LANGCHAIN_PROJECT)
if Config.LANGCHAIN_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", Config.LANGCHAIN_API_KEY)

# Regex to detect common non-USD ISO currency codes in property listing text.
# When any of these are found in retrieved context, the retrieve node appends
# live USD conversion rates so the LLM can present both the original price and
# its USD equivalent without needing to call a tool itself.
_NON_USD_CURRENCY_RE = re.compile(r'\b(PLN|EUR|GBP|CHF|NOK|SEK|DKK|CZK|HUF|RON)\b')


# ------------------- STATE -------------------

class State(TypedDict):
    """
    Shared state object that flows through every node in the LangGraph pipeline.

    Fields
    ------
    user_prompt : str
        The raw, unmodified question typed by the user.

    reformulated_question : str
        A rewritten version of user_prompt produced by the 'reformulate' node,
        better phrased for vector search and LLM consumption.

    needs_search : bool
        Set by the 'classify' node. True → the 'retrieve' node runs and
        injects property listings into the prompt. False → 'generate' runs
        directly without touching the vector store.

    context : str
        Formatted property listings returned by the 'retrieve' node.
        May include appended currency conversion rates for non-USD prices.
        Empty string when needs_search is False.

    answer : str
        The final text answer produced by the 'generate' node.

    session_history : List[AnyMessage]
        Previous turns for this session (HumanMessage + AIMessage pairs).
        Passed in at the start of each turn; the graph never writes to it.
    """

    user_prompt: str
    reformulated_question: str
    needs_search: bool
    context: str
    answer: str
    session_history: List[AnyMessage]


# ------------------- MODEL -------------------

class LlmModelGraph:
    """
    LangGraph pipeline: reformulate → classify → [retrieve →] generate.

    Graph topology
    --------------
    START → reformulate → classify ──(needs_search=True)──► retrieve → generate → END
                                   └──(needs_search=False)──► generate → END

    Key design decisions
    --------------------
    - Conditional RAG: the classify node uses the LLM for a plain YES/NO
      decision on whether the query needs a property database search.
      Greetings, follow-ups, and general questions skip retrieval entirely.

    - No tool-calling API: local LLMs exposed via OpenAI-compatible endpoints
      (e.g. Ollama /v1) do not reliably produce structured tool_calls — they
      often output the call as JSON text or ignore the schema entirely.
      Explicit conditional routing is deterministic and model-agnostic.

    - Currency conversion: the retrieve node scans listings for non-USD
      currencies (PLN, EUR, GBP, …) and appends live USD rates from the
      finance MCP server. The generate node then has both prices available
      without any tool-calling during generation.

    - Per-session history: each session_id has its own conversation list
      stored in self._histories so multiple concurrent users stay isolated.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        # Load prompt files from paths configured in .env
        self.system_prompt = self._load_file(Config.PROMPT_FILE)
        self.reformulation_template = self._load_file(Config.REFORMULATION_PROMPT)

        # RAG layer: SentenceTransformer embeddings + vector store search
        self.rag_context_manager = RagContextManager(Embedder())

        # In-memory session store keyed by session_id: {session_id: [msg, ...]}
        self._histories: dict[str, list] = {}

        self.graph = self._build_graph()

    # ------------------- PUBLIC API -------------------

    @traceable(name="LlmModelGraph.ask", run_type="chain", tags=["property-agent"])
    def ask(self, user_query: str, session_id: str = None) -> str:
        """
        Run the full pipeline and return the final answer as a string (blocking).
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.debug("ask: no session_id provided, generated %s", session_id)

        history = self._get_history(session_id)
        state = self._initial_state(user_query, history)
        result = self.graph.invoke(state, config={"run_name": "property-rag-pipeline"})
        answer = result["answer"]

        self._histories[session_id] = history + [
            HumanMessage(content=user_query),
            AIMessage(content=answer),
        ]
        return answer

    @traceable(name="LlmModelGraph.ask_stream", run_type="chain", tags=["property-agent", "streaming"])
    def ask_stream(self, user_query: str, session_id: str = None):
        """
        Run the full pipeline and yield text tokens as they are generated.

        Only chunks from the 'generate' node with non-empty content are yielded.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.debug("ask_stream: no session_id provided, generated %s", session_id)

        history = self._get_history(session_id)
        state = self._initial_state(user_query, history)
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

        self._histories[session_id] = history + [
            HumanMessage(content=user_query),
            AIMessage(content=full_answer),
        ]

    def close(self):
        """
        Shut down background resources.

        Terminates the finance MCP subprocess (if running) so no orphaned
        ``npx`` processes are left alive after the agent is stopped.
        """
        close_finance_mcp()
        logger.debug("LlmModelGraph closed")

    # ------------------- GRAPH CONSTRUCTION -------------------

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("reformulate", self._reformulate_node)
        graph.add_node("classify", self._classify_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)

        graph.add_edge(START, "reformulate")
        graph.add_edge("reformulate", "classify")
        graph.add_conditional_edges(
            "classify",
            lambda state: "retrieve" if state["needs_search"] else "generate",
        )
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    # ------------------- NODES -------------------

    def _reformulate_node(self, state: State) -> dict:
        """Rewrite the raw user query into a cleaner, search-friendly question."""
        logger.info("[reformulate] input: %s", state["user_prompt"])
        prompt = ChatPromptTemplate.from_template(self.reformulation_template)
        chain = prompt | self.llm | StrOutputParser()
        reformulated = chain.invoke({"user_prompt": state["user_prompt"]})
        logger.info("[reformulate] output: %s", reformulated)
        return {"reformulated_question": reformulated}

    def _classify_node(self, state: State) -> dict:
        """
        Decide whether the query requires a property database search.

        Prompts the LLM for a plain YES/NO answer. Only YES routes to the
        retrieve node; everything else goes directly to generate so the
        vector store is never touched unnecessarily.
        """
        logger.info("[classify] question: %s", state["reformulated_question"])
        prompt = ChatPromptTemplate.from_template(
            "You are a query router. Decide whether the following query requires "
            "searching a property listings database.\n\n"
            "Answer YES only if the user is asking about:\n"
            "- Finding, renting, buying, or viewing properties\n"
            "- Apartments, houses, flats, or rooms\n"
            "- Property prices, locations, or features\n\n"
            "Answer NO for everything else:\n"
            "- Greetings (hi, hello, thanks, etc.)\n"
            "- General questions not about property\n"
            "- Follow-up questions about a previous answer\n\n"
            "Respond with ONLY the word YES or NO.\n\n"
            "Query: {query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": state["reformulated_question"]})
        needs_search = result.strip().upper().startswith("YES")
        logger.info("[classify] needs_search=%s (raw=%r)", needs_search, result.strip())
        return {"needs_search": needs_search}

    def _retrieve_node(self, state: State) -> dict:
        """
        Search the vector store and return formatted property listings as context.

        After retrieval, scans the context for non-USD currency codes (PLN, EUR,
        GBP, …). When found, calls the finance MCP server to append live USD
        conversion rates so the generate node can present both the original price
        and its USD equivalent without any tool-calling during generation.
        """
        logger.info("[retrieve] querying with: %s", state["reformulated_question"])
        context = self.rag_context_manager.get_context(state["reformulated_question"])
        if context:
            context = self._append_currency_rates(context)
        logger.info("[retrieve] context length: %d chars", len(context))
        return {"context": context}

    def _generate_node(self, state: State) -> dict:
        """
        Generate the final answer using the LLM.

        Builds a prompt from the system instructions, session history, and the
        reformulated question. When context is present (needs_search=True path)
        it is added as a second system message *before* the conversation turns
        so that it sits in the model's context before the human question.

        Context is embedded directly into the message string rather than via a
        {context} template variable to avoid KeyError when property listings
        contain literal curly braces (e.g. in addresses or JSON snippets).
        """
        logger.info(
            "[generate] history_turns=%d has_context=%s",
            len(state["session_history"]),
            bool(state["context"]),
        )
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
        logger.debug("[generate] raw response type=%s content_head=%r", type(response).__name__, answer[:200])
        logger.info("[generate] answer length: %d chars", len(answer))
        return {"answer": answer}

    # ------------------- HELPERS -------------------

    def _append_currency_rates(self, context: str) -> str:
        """
        Detect non-USD currency codes in context and append live USD rates.

        Calls the finance MCP server once per detected currency to fetch the
        current exchange rate. Appends a compact reference block at the end of
        the context so the generate node can include both prices without calling
        any tools at generation time.
        """
        currencies = sorted(set(_NON_USD_CURRENCY_RE.findall(context)))
        if not currencies:
            return context
        rates = []
        for currency in currencies:
            rate = _mcp.call_tool("currency_convert", {"from": currency, "to": "USD", "amount": 1})
            if rate:
                rates.append(f"  1 {currency} = {rate}")
            logger.debug("[retrieve] currency rate fetched: %s", currency)
        if rates:
            context += "\n\nCurrency conversion rates (live, 1 unit → USD):\n" + "\n".join(rates)
        return context

    def _get_history(self, session_id: str) -> list:
        """Return the history list for session_id, creating an empty one if new."""
        return self._histories.setdefault(session_id, [])

    def _initial_state(self, user_query: str, history: list = None) -> State:
        return {
            "user_prompt": user_query,
            "reformulated_question": "",
            "needs_search": False,
            "context": "",
            "answer": "",
            "session_history": list(history) if history else [],
        }

    @staticmethod
    def _load_file(path: str) -> str:
        with open(path, "r") as f:
            return f.read()
