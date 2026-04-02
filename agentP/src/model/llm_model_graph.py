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
    """
    Shared state object that flows through every node in the LangGraph pipeline.

    Fields
    ------
    user_prompt : str
        The raw, unmodified question typed by the user.
        Set once at the start of each turn and never changed afterwards.

    reformulated_question : str
        A rewritten version of user_prompt produced by the 'reformulate' node.
        Better phrased for vector search and LLM consumption.
        Starts as an empty string and is filled in by _reformulate_node.

    messages : List[AnyMessage]  (append-only via add_messages reducer)
        The growing list of LangChain messages exchanged between the agent
        and the tool layer during a single turn:
          SystemMessage  →  HumanMessage  →  AIMessage (tool call)
          →  ToolMessage (tool result)  →  AIMessage (final answer)
        LangGraph's add_messages reducer appends new messages automatically
        instead of replacing the list on each node return.

    session_history : List[AnyMessage]
        Previous turns from this user's session (HumanMessage + AIMessage pairs).
        Injected at the start of the turn so the agent can reference earlier
        context. Managed externally by ask() / ask_stream() and stored in
        self._histories; the graph itself never writes to this field.
    """

    user_prompt: str
    reformulated_question: str
    messages: Annotated[List[AnyMessage], add_messages]
    session_history: List[AnyMessage]


# ------------------- MODEL -------------------

class LlmModelGraph:
    """
    A LangGraph-based conversational agent for property search.

    How it works — the graph executes these steps in order for every user turn:

        START
          │
          ▼
        [reformulate]
          Rewrites the user's raw query into a better-phrased search question
          using the LLM and the reformulation prompt template.
          │
          ▼
        [agent]
          Calls the LLM with the property_search tool available.
          On the FIRST call it assembles the full message list:
            system prompt + session history + reformulated question.
          The LLM decides whether to search for listings or answer directly.
          │
          ├── LLM requested a tool call?
          │       YES → [tools] → back to [agent]  (may loop up to _RECURSION_LIMIT times)
          │       NO  → END
          ▼
        [tools]  (ToolNode)
          Executes the property_search tool: embeds the query, searches the
          vector store, and returns formatted listing text as a ToolMessage.
          Control returns to [agent] so the LLM can read the results and reply.

    Key design decisions
    --------------------
    - RAG is OPTIONAL: the LLM only calls the search tool when the query is
      about properties. Simple greetings or follow-up questions skip RAG entirely.
    - Per-session history: each session_id has its own conversation list stored
      in self._histories so multiple concurrent users stay fully isolated.
    - Recursion limit: the agent→tools loop is capped at _RECURSION_LIMIT
      iterations to prevent infinite loops if the LLM keeps calling the tool.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialise the agent and compile the LangGraph pipeline.

        Steps
        -----
        1. Load the system prompt and reformulation template from disk.
        2. Create the RAG layer (Embedder + RagContextManager).
        3. Initialise the per-session history store (empty dict).
        4. Build the property_search tool and bind it to the LLM so the
           LLM knows it can call it.
        5. Compile the StateGraph into an executable pipeline.
        """
        self.llm = llm

        # Step 1 — load prompt files from paths configured in .env
        self.system_prompt = self._load_file(Config.PROMPT_FILE)
        self.reformulation_template = self._load_file(Config.REFORMULATION_PROMPT)

        # Step 2 — RAG layer: SentenceTransformer embeddings + vector store search
        self.rag_context_manager = RagContextManager(Embedder())

        # Step 3 — in-memory store keyed by session_id: {session_id: [msg, msg, ...]}
        self._histories: dict[str, list] = {}

        # Step 4 — create the tool and bind it to the LLM.
        # bind_tools() adds the tool schema to every LLM call so the model knows
        # it can call property_search when a user asks about listings.
        self._property_search_tool = make_property_search_tool(self.rag_context_manager)
        self._llm_with_tools = self.llm.bind_tools([self._property_search_tool])

        # Step 5 — compile the graph (nodes + edges are wired in _build_graph)
        self.graph = self._build_graph()

    # ------------------- PUBLIC API -------------------

    @traceable(name="LlmModelGraph.ask", run_type="chain", tags=["property-agent"])
    def ask(self, user_query: str, session_id: str = None) -> str:
        """
        Run the full pipeline and return the final answer as a string (blocking).

        Flow
        ----
        1. Resolve the session — generate a UUID if no session_id is given.
        2. Retrieve this session's conversation history from self._histories.
        3. Build the initial State and run graph.invoke() (runs all nodes
           synchronously and returns the final state).
        4. Extract the answer from the last message in the returned state.
        5. Append the user query + answer to the session history for next turn.

        Parameters
        ----------
        user_query  : The user's raw question.
        session_id  : Optional identifier to maintain conversation continuity.
                      If None, a new session UUID is generated automatically.

        Returns
        -------
        The agent's final text answer.
        """
        # Step 1 — ensure we always have a session_id to key history against
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.debug("ask: no session_id provided, generated %s", session_id)

        # Step 2 — fetch existing history (empty list for a brand-new session)
        history = self._get_history(session_id)

        # Step 3 — pack into a State dict and run the compiled graph
        state = self._initial_state(user_query, history)
        result = self.graph.invoke(
            state,
            config={"run_name": "property-rag-pipeline", "recursion_limit": _RECURSION_LIMIT},
        )

        # Step 4 — the last message is always the agent's final text reply
        answer = result["messages"][-1].content

        # Step 5 — persist the new turn so the next call can reference it
        self._histories[session_id] = history + [
            HumanMessage(content=user_query),
            AIMessage(content=answer),
        ]
        return answer

    @traceable(name="LlmModelGraph.ask_stream", run_type="chain", tags=["property-agent", "streaming"])
    def ask_stream(self, user_query: str, session_id: str = None):
        """
        Run the full pipeline and yield text tokens as they are generated (streaming).

        Only chunks that come from the 'agent' node and carry plain text content
        are yielded. The following chunk types are silently skipped:
          - Chunks from 'reformulate' or 'tools' nodes (internal pipeline steps)
          - Empty-content chunks (LangChain emits these as padding)
          - Tool-call chunks (the raw JSON the LLM uses to invoke a tool)

        After the stream is exhausted the full answer is assembled from the
        collected tokens and saved to the session history exactly like ask().

        Parameters
        ----------
        user_query  : The user's raw question.
        session_id  : Optional identifier to maintain conversation continuity.

        Yields
        ------
        str — individual text token chunks from the agent's final response.
        """
        # Resolve session — same logic as ask()
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.debug("ask_stream: no session_id provided, generated %s", session_id)

        history = self._get_history(session_id)
        state = self._initial_state(user_query, history)
        full_answer = ""

        # graph.stream() with stream_mode="messages" yields (chunk, metadata) pairs.
        # metadata["langgraph_node"] identifies which node produced the chunk.
        for chunk, metadata in self.graph.stream(
            state,
            stream_mode="messages",
            config={"run_name": "property-rag-pipeline-stream", "recursion_limit": _RECURSION_LIMIT},
        ):
            if (
                metadata.get("langgraph_node") == "agent"   # only the agent's words
                and isinstance(chunk, AIMessageChunk)        # typed text chunk
                and chunk.content                            # non-empty
                and not chunk.tool_call_chunks               # not a tool invocation
            ):
                full_answer += chunk.content
                yield chunk.content
            else:
                # Log skipped chunks so production traces show why tokens were dropped
                logger.debug(
                    "ask_stream: skipping chunk node=%s has_content=%s tool_calls=%s",
                    metadata.get("langgraph_node"),
                    bool(chunk.content),
                    bool(getattr(chunk, "tool_call_chunks", [])),
                )

        # Persist the completed turn to session history
        self._histories[session_id] = history + [
            HumanMessage(content=user_query),
            AIMessage(content=full_answer),
        ]

    def close(self):
        """No-op cleanup hook — present for lifecycle compatibility with LocalAgent."""
        pass

    # ------------------- GRAPH CONSTRUCTION -------------------

    def _build_graph(self):
        """
        Wire up the LangGraph StateGraph and compile it into an executable pipeline.

        Graph topology
        --------------
        START → reformulate → agent ──(has tool call?)──► tools → agent
                                    └──(no tool call)──► END

        Nodes
        -----
        reformulate : _reformulate_node  — rewrites the user query
        agent       : _agent_node        — LLM reasoning + tool decision
        tools       : ToolNode           — executes the property_search tool

        Edges
        -----
        START → reformulate          Always start with query reformulation.
        reformulate → agent          Hand the rewritten query to the agent.
        agent → tools_condition      LangGraph's built-in router: checks whether
                                     the last AIMessage contains tool_calls.
                                     If yes → routes to "tools"; if no → END.
        tools → agent                After the tool runs, return to the agent
                                     so it can read the result and answer.
        """
        graph = StateGraph(State)

        # Register the three nodes
        graph.add_node("reformulate", self._reformulate_node)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode([self._property_search_tool]))

        # Wire the edges
        graph.add_edge(START, "reformulate")
        graph.add_edge("reformulate", "agent")
        graph.add_conditional_edges("agent", tools_condition)  # routes to "tools" or END
        graph.add_edge("tools", "agent")  # loop back after tool execution

        return graph.compile()

    # ------------------- NODES -------------------

    def _reformulate_node(self, state: State) -> dict:
        """
        STEP 1 — Query reformulation.

        Takes the raw user_prompt and rewrites it into a cleaner, more
        search-friendly question using the LLM and the reformulation prompt
        template (loaded from Config.REFORMULATION_PROMPT).

        Why this step?
        The user might ask something vague like "something cheap with 2 rooms".
        The LLM rewrites this into "2 bedroom apartment affordable price" which
        produces much better vector search results downstream.

        Input state keys used  : user_prompt
        Output state keys set  : reformulated_question
        """
        logger.info("[reformulate] input: %s", state["user_prompt"])

        # Build a simple chain: prompt template → LLM → plain string output
        prompt = ChatPromptTemplate.from_template(self.reformulation_template)
        chain = prompt | self.llm | StrOutputParser()
        reformulated = chain.invoke({"user_prompt": state["user_prompt"]})

        logger.info("[reformulate] output: %s", reformulated)
        # Returning a dict updates only the listed keys in the shared State
        return {"reformulated_question": reformulated}

    def _agent_node(self, state: State) -> dict:
        """
        STEP 2 (and STEP 4 if tool was called) — LLM reasoning.

        This node is called in two distinct situations:

        First call  (state["messages"] is empty)
            Assembles the full message list for this turn:
              [SystemMessage]         ← agent persona and instructions
              + [session_history]     ← previous turns (HumanMessage+AIMessage pairs)
              + [HumanMessage]        ← the reformulated question for this turn
            The LLM with tools bound then decides:
              a) Call property_search → returns an AIMessage with tool_calls
              b) Answer directly      → returns an AIMessage with plain text

        Subsequent call  (state["messages"] is non-empty, after tool execution)
            The accumulated messages list already contains the tool result
            (a ToolMessage appended by ToolNode), so it is passed directly
            to the LLM. The LLM reads the search results and writes the
            final answer as plain text.

        Input state keys used  : messages, session_history, reformulated_question
        Output state keys set  : messages  (appended via add_messages reducer)
        """
        if not state["messages"]:
            # First call — build the full context for this turn
            logger.info("[agent] first call — building messages from history + reformulated question")
            messages = (
                [SystemMessage(content=self.system_prompt)]
                + list(state["session_history"])   # prior turns give conversational context
                + [HumanMessage(content=state["reformulated_question"])]
            )
        else:
            # Subsequent call — tool result is already in the messages list
            logger.info("[agent] subsequent call — %d messages in context", len(state["messages"]))
            messages = state["messages"]

        # Invoke the tool-aware LLM; response is either a tool call or the final answer
        response = self._llm_with_tools.invoke(messages)
        logger.info("[agent] has_tool_calls: %s", bool(getattr(response, "tool_calls", [])))

        # Returning {"messages": [response]} triggers the add_messages reducer,
        # which appends the response to state["messages"] rather than replacing it
        return {"messages": [response]}

    # ------------------- HELPERS -------------------

    def _get_history(self, session_id: str) -> list:
        """
        Return the conversation history list for session_id.
        Creates an empty list for new sessions using dict.setdefault so the
        first call for any session_id is safe without an explicit check.
        """
        return self._histories.setdefault(session_id, [])

    def _initial_state(self, user_query: str, history: list = None) -> State:
        """
        Build a fresh State dict for the start of a new turn.

        user_prompt           ← the raw user query (unchanged throughout the turn)
        reformulated_question ← empty string; filled in by _reformulate_node
        messages              ← empty list; filled in by _agent_node / ToolNode
        session_history       ← a copy of the caller's history list so that
                                mutations inside the graph do not affect the
                                source list stored in self._histories
        """
        return {
            "user_prompt": user_query,
            "reformulated_question": "",
            "messages": [],
            "session_history": list(history) if history else [],
        }

    @staticmethod
    def _load_file(path: str) -> str:
        """Read a text file from disk and return its contents as a string."""
        with open(path, "r") as f:
            return f.read()
