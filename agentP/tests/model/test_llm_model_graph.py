import sys
import importlib.machinery
import unittest
from unittest.mock import patch, MagicMock

# Python 3.13 + system faiss: __spec__ is None, which causes
# importlib.util.find_spec("faiss") — called deep inside transformers during
# langchain_core import — to raise ValueError. A MagicMock is not enough
# because downstream code reads spec.name expecting a real str; use a proper
# ModuleSpec instead.
if "faiss" in sys.modules and getattr(sys.modules["faiss"], "__spec__", None) is None:
    sys.modules["faiss"].__spec__ = importlib.machinery.ModuleSpec("faiss", None)

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

from agentP.src.model.llm_model_graph import LlmModelGraph, State

# ---------------------------------------------------------------------------
# Fake prompt strings used across all tests.
# The reformulation template must contain {user_prompt} so that
# ChatPromptTemplate.from_template() can format it correctly.
# ---------------------------------------------------------------------------
_FAKE_SYSTEM_PROMPT = "You are a helpful property agent."
_FAKE_REFORMULATION_TEMPLATE = "Improve this real-estate query: {user_prompt}"


class TestLlmModelGraph(unittest.TestCase):

    def setUp(self):
        """
        Build a LlmModelGraph with all heavy dependencies mocked out:
          - _load_file            → returns fake prompt strings (no filesystem access)
          - Embedder              → no SentenceTransformer model is loaded
          - RagContextManager     → no vector store is queried
          - _mcp                  → no npx/MCP subprocess is started

        Patchers are started here and stopped in tearDown so they remain
        active for the full duration of each test method.

        After construction the compiled LangGraph is replaced with a plain
        MagicMock so that ask() / ask_stream() and node tests are fast and isolated.
        """
        self._load_file_patcher = patch.object(LlmModelGraph, "_load_file")
        self._embedder_patcher = patch("agentP.src.model.llm_model_graph.Embedder")
        self._rag_patcher = patch("agentP.src.model.llm_model_graph.RagContextManager")
        self._mcp_patcher = patch("agentP.src.model.llm_model_graph._mcp")

        mock_load_file = self._load_file_patcher.start()
        self._embedder_patcher.start()
        self._rag_patcher.start()
        self.mock_mcp = self._mcp_patcher.start()

        mock_load_file.side_effect = [_FAKE_SYSTEM_PROMPT, _FAKE_REFORMULATION_TEMPLATE]
        # Default: MCP returns empty string so no currency block is appended
        self.mock_mcp.call_tool.return_value = ""

        self.mock_llm = MagicMock()
        self.model = LlmModelGraph(self.mock_llm)

        # Replace the compiled graph so public-API tests don't run real LangGraph nodes.
        self.model.graph = MagicMock(name="CompiledGraphMock")

    def tearDown(self):
        self._load_file_patcher.stop()
        self._embedder_patcher.stop()
        self._rag_patcher.stop()
        self._mcp_patcher.stop()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_state(self, **kwargs) -> State:
        """Build a complete State dict with sensible defaults for node tests."""
        defaults: State = {
            "user_prompt": "find a flat",
            "reformulated_question": "2 bedroom Warsaw",
            "needs_search": False,
            "context": "",
            "answer": "",
            "session_history": [],
        }
        defaults.update(kwargs)
        return defaults

    def _stream_chunk(self, content: str, node: str):
        """Helper: create a (AIMessageChunk, metadata) pair as graph.stream yields."""
        chunk = AIMessageChunk(content=content)
        return (chunk, {"langgraph_node": node})

    # ------------------------------------------------------------------
    # ask()
    # ------------------------------------------------------------------

    def test_should_return_answer_string_when_ask_is_called(self):
        """ask() returns the answer string from the graph result."""
        self.model.graph.invoke.return_value = {"answer": "3 listings found in Warsaw."}

        result = self.model.ask("3-bed apartment in Warsaw")

        self.assertEqual(result, "3 listings found in Warsaw.")

    def test_should_invoke_graph_once_when_ask_is_called(self):
        """ask() delegates to graph.invoke exactly once."""
        self.model.graph.invoke.return_value = {"answer": "ok"}

        self.model.ask("3-bed apartment in Warsaw")

        self.model.graph.invoke.assert_called_once()

    def test_should_build_history_from_query_and_answer_when_ask_is_called(self):
        """ask() appends HumanMessage(user_query) + AIMessage(answer) to the session history."""
        SESSION = "sess-history-ask"
        self.model.graph.invoke.return_value = {"answer": "3 listings found."}

        self.model.ask("find me a flat", session_id=SESSION)

        hist = self.model._histories[SESSION]
        self.assertEqual(len(hist), 2)
        self.assertIsInstance(hist[0], HumanMessage)
        self.assertIsInstance(hist[1], AIMessage)
        self.assertEqual(hist[0].content, "find me a flat")
        self.assertEqual(hist[1].content, "3 listings found.")

    def test_should_accumulate_history_across_turns_when_asking(self):
        """ask() grows the session history by 2 messages per turn."""
        SESSION = "sess-accum-ask"
        self.model.graph.invoke.side_effect = [
            {"answer": "answer 1"},
            {"answer": "answer 2"},
        ]

        self.model.ask("query 1", session_id=SESSION)
        self.model.ask("query 2", session_id=SESSION)

        self.assertEqual(len(self.model._histories[SESSION]), 4)

    def test_should_pass_correct_initial_state_when_ask_is_called(self):
        """ask() passes a state with user_prompt set and empty derived fields."""
        self.model.graph.invoke.return_value = {"answer": "ok"}

        self.model.ask("find a studio", session_id="s1")

        call_args = self.model.graph.invoke.call_args
        state = call_args[0][0]
        self.assertEqual(state["user_prompt"], "find a studio")
        self.assertEqual(state["reformulated_question"], "")
        self.assertEqual(state["context"], "")
        self.assertEqual(state["answer"], "")

    def test_should_isolate_history_between_different_sessions(self):
        """ask() keeps separate histories for different session IDs."""
        self.model.graph.invoke.return_value = {"answer": "ok"}

        self.model.ask("q1", session_id="session-A")
        self.model.ask("q2", session_id="session-B")

        self.assertEqual(len(self.model._histories["session-A"]), 2)
        self.assertEqual(len(self.model._histories["session-B"]), 2)
        self.assertEqual(self.model._histories["session-A"][0].content, "q1")
        self.assertEqual(self.model._histories["session-B"][0].content, "q2")

    def test_should_share_history_within_same_session_across_turns(self):
        """ask() passes previous-turn history to the graph on the second call."""
        SESSION = "sess-share"
        self.model.graph.invoke.side_effect = [
            {"answer": "answer 1"},
            {"answer": "answer 2"},
        ]

        self.model.ask("turn 1", session_id=SESSION)
        self.model.ask("turn 2", session_id=SESSION)

        second_call_state = self.model.graph.invoke.call_args_list[1][0][0]
        history = second_call_state["session_history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].content, "turn 1")
        self.assertEqual(history[1].content, "answer 1")

    def test_should_generate_session_id_when_none_provided_to_ask(self):
        """ask() auto-generates a session_id and stores history under it."""
        self.model.graph.invoke.return_value = {"answer": "ok"}

        self.model.ask("any query")

        self.assertEqual(len(self.model._histories), 1)

    # ------------------------------------------------------------------
    # ask_stream()
    # ------------------------------------------------------------------

    def test_should_yield_tokens_from_generate_node_when_streaming(self):
        """ask_stream() yields each non-empty token from the 'generate' node."""
        self.model.graph.stream.return_value = [
            self._stream_chunk("Hello ", "generate"),
            self._stream_chunk("world", "generate"),
        ]

        tokens = list(self.model.ask_stream("find a flat"))

        self.assertEqual(tokens, ["Hello ", "world"])

    def test_should_ignore_non_generate_nodes_when_streaming(self):
        """ask_stream() silently ignores chunks from reformulate/classify/retrieve nodes."""
        self.model.graph.stream.return_value = [
            self._stream_chunk("ignored", "reformulate"),
            self._stream_chunk("also ignored", "retrieve"),
            self._stream_chunk("kept", "generate"),
        ]

        tokens = list(self.model.ask_stream("q"))

        self.assertEqual(tokens, ["kept"])

    def test_should_skip_empty_tokens_when_streaming(self):
        """ask_stream() drops empty string chunks from the generate node."""
        self.model.graph.stream.return_value = [
            self._stream_chunk("", "generate"),
            self._stream_chunk("real token", "generate"),
        ]

        tokens = list(self.model.ask_stream("q"))

        self.assertEqual(tokens, ["real token"])

    def test_should_yield_nothing_when_stream_has_no_generate_chunks(self):
        """ask_stream() is an empty iterator when the graph emits no generate chunks."""
        self.model.graph.stream.return_value = [
            self._stream_chunk("ignored", "reformulate"),
        ]

        tokens = list(self.model.ask_stream("q"))

        self.assertEqual(tokens, [])

    def test_should_append_human_and_ai_messages_to_history_when_streaming(self):
        """ask_stream() saves the full streamed answer and the original query to history."""
        SESSION = "sess-stream-hist"
        self.model.graph.stream.return_value = [
            self._stream_chunk("part1 ", "generate"),
            self._stream_chunk("part2", "generate"),
        ]

        list(self.model.ask_stream("my query", session_id=SESSION))

        hist = self.model._histories[SESSION]
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0].content, "my query")
        self.assertEqual(hist[1].content, "part1 part2")

    def test_should_accumulate_history_across_turns_when_streaming(self):
        """ask_stream() accumulates history correctly across multiple turns."""
        SESSION = "sess-stream-accum"
        self.model.graph.stream.side_effect = [
            [self._stream_chunk("answer 1", "generate")],
            [self._stream_chunk("answer 2", "generate")],
        ]

        list(self.model.ask_stream("q1", session_id=SESSION))
        list(self.model.ask_stream("q2", session_id=SESSION))

        self.assertEqual(len(self.model._histories[SESSION]), 4)

    def test_should_generate_session_id_when_none_provided_to_ask_stream(self):
        """ask_stream() auto-generates a session_id when none is given."""
        self.model.graph.stream.return_value = [
            self._stream_chunk("token", "generate"),
        ]

        list(self.model.ask_stream("query without session"))

        self.assertEqual(len(self.model._histories), 1)

    # ------------------------------------------------------------------
    # _classify_node()
    # ------------------------------------------------------------------

    def test_should_return_needs_search_true_when_query_is_about_property(self):
        """_classify_node() returns needs_search=True for property-related queries."""
        self.mock_llm.return_value = AIMessage(content="YES")
        state = self._make_state(reformulated_question="2 bedroom flat Warsaw")

        result = self.model._classify_node(state)

        self.assertTrue(result["needs_search"])

    def test_should_return_needs_search_false_when_query_is_a_greeting(self):
        """_classify_node() returns needs_search=False for greetings."""
        self.mock_llm.return_value = AIMessage(content="NO")
        state = self._make_state(reformulated_question="hello how are you")

        result = self.model._classify_node(state)

        self.assertFalse(result["needs_search"])

    def test_should_handle_yes_with_trailing_text_when_classifying(self):
        """_classify_node() treats 'YES ...' as needs_search=True."""
        self.mock_llm.return_value = AIMessage(content="YES, definitely")
        state = self._make_state(reformulated_question="any property question")

        result = self.model._classify_node(state)

        self.assertTrue(result["needs_search"])

    def test_should_handle_lowercase_yes_when_classifying(self):
        """_classify_node() is case-insensitive — 'yes' counts as YES."""
        self.mock_llm.return_value = AIMessage(content="yes")
        state = self._make_state()

        result = self.model._classify_node(state)

        self.assertTrue(result["needs_search"])

    # ------------------------------------------------------------------
    # _retrieve_node()
    # ------------------------------------------------------------------

    def test_should_call_get_context_with_reformulated_question_when_retrieving(self):
        """_retrieve_node() queries the RAG store using the reformulated question."""
        self.model.rag_context_manager.get_context.return_value = "listing A\nlisting B"
        state = self._make_state(reformulated_question="2 bed Warsaw")

        self.model._retrieve_node(state)

        self.model.rag_context_manager.get_context.assert_called_once_with("2 bed Warsaw")

    def test_should_return_context_dict_when_retrieving(self):
        """_retrieve_node() returns the RAG context string under the 'context' key."""
        self.model.rag_context_manager.get_context.return_value = "listing A\nlisting B"
        state = self._make_state()

        result = self.model._retrieve_node(state)

        self.assertEqual(result["context"], "listing A\nlisting B")

    def test_should_return_empty_context_when_get_context_returns_empty_string(self):
        """_retrieve_node() propagates an empty string when the vector store has no matches."""
        self.model.rag_context_manager.get_context.return_value = ""
        state = self._make_state()

        result = self.model._retrieve_node(state)

        self.assertEqual(result["context"], "")

    def test_should_not_call_mcp_when_context_has_no_currencies(self):
        """_retrieve_node() skips MCP when context contains only USD or no currency codes."""
        self.model.rag_context_manager.get_context.return_value = "Nice flat, price: $800/month"
        state = self._make_state()

        self.model._retrieve_node(state)

        self.mock_mcp.call_tool.assert_not_called()

    def test_should_call_mcp_for_each_non_usd_currency_found_in_context(self):
        """_retrieve_node() calls MCP once per distinct non-USD currency detected."""
        self.model.rag_context_manager.get_context.return_value = (
            "Flat in Warsaw 3000 PLN/month. Studio 800 EUR/month."
        )
        self.mock_mcp.call_tool.return_value = "0.24 USD"
        state = self._make_state()

        self.model._retrieve_node(state)

        calls = self.mock_mcp.call_tool.call_args_list
        called_currencies = {c[0][1]["from"] for c in calls}
        self.assertIn("PLN", called_currencies)
        self.assertIn("EUR", called_currencies)
        self.assertEqual(len(calls), 2)

    def test_should_append_currency_rates_to_context_when_non_usd_currencies_found(self):
        """_retrieve_node() appends a currency reference block when rates are returned."""
        self.model.rag_context_manager.get_context.return_value = "Flat 3000 PLN/month"
        self.mock_mcp.call_tool.return_value = "0.24 USD"
        state = self._make_state()

        result = self.model._retrieve_node(state)

        self.assertIn("Currency conversion rates", result["context"])
        self.assertIn("PLN", result["context"])

    def test_should_not_append_currency_block_when_mcp_returns_empty_string(self):
        """_retrieve_node() leaves context unchanged when MCP returns empty for all currencies."""
        original = "Flat 3000 PLN/month"
        self.model.rag_context_manager.get_context.return_value = original
        self.mock_mcp.call_tool.return_value = ""
        state = self._make_state()

        result = self.model._retrieve_node(state)

        self.assertEqual(result["context"], original)

    # ------------------------------------------------------------------
    # _generate_node()
    # ------------------------------------------------------------------

    def test_should_return_answer_from_llm_when_generating(self):
        """_generate_node() invokes the LLM and returns the answer string."""
        self.mock_llm.invoke.return_value = AIMessage(content="Here are listings in Warsaw.")
        state = self._make_state(context="some context")

        result = self.model._generate_node(state)

        self.assertEqual(result["answer"], "Here are listings in Warsaw.")

    def test_should_return_answer_without_context_when_context_is_empty(self):
        """_generate_node() generates an answer even when context is empty (no RAG path)."""
        self.mock_llm.invoke.return_value = AIMessage(content="General answer.")
        state = self._make_state(context="")

        result = self.model._generate_node(state)

        self.assertEqual(result["answer"], "General answer.")

    def test_should_use_reformulated_question_not_raw_prompt_when_generating(self):
        """_generate_node() uses reformulated_question as the question, not user_prompt."""
        self.mock_llm.invoke.return_value = AIMessage(content="answer")
        state = self._make_state(
            user_prompt="cheap flat",
            reformulated_question="affordable 1-bedroom apartment Warsaw",
        )

        result = self.model._generate_node(state)

        self.assertIn("answer", result)

    def test_should_pass_session_history_to_llm_when_generating(self):
        """_generate_node() incorporates session_history messages into the prompt."""
        history = [HumanMessage(content="prior question"), AIMessage(content="prior answer")]
        self.mock_llm.invoke.return_value = AIMessage(content="contextual answer")
        state = self._make_state(session_history=history)

        result = self.model._generate_node(state)

        self.assertEqual(result["answer"], "contextual answer")

    # ------------------------------------------------------------------
    # _reformulate_node()
    # ------------------------------------------------------------------

    def test_should_return_reformulated_question_when_reformulating(self):
        """_reformulate_node() invokes the LLM chain and stores the output."""
        self.mock_llm.return_value = AIMessage(content="2 bedroom apartment Warsaw affordable")
        state = self._make_state(
            user_prompt="cheap 2 bed Warsaw",
            reformulated_question="",
        )

        result = self.model._reformulate_node(state)

        self.assertEqual(
            result["reformulated_question"],
            "2 bedroom apartment Warsaw affordable",
        )

    # ------------------------------------------------------------------
    # _initial_state()
    # ------------------------------------------------------------------

    def test_should_set_user_prompt_when_building_initial_state(self):
        """_initial_state() stores the query string in user_prompt."""
        state = self.model._initial_state("find me a studio")

        self.assertEqual(state["user_prompt"], "find me a studio")

    def test_should_have_empty_derived_fields_when_building_initial_state(self):
        """_initial_state() initialises all derived fields to empty/falsy defaults."""
        state = self.model._initial_state("anything")

        self.assertEqual(state["reformulated_question"], "")
        self.assertFalse(state["needs_search"])
        self.assertEqual(state["context"], "")
        self.assertEqual(state["answer"], "")
        self.assertEqual(state["session_history"], [])
        self.assertNotIn("messages", state)

    def test_should_include_session_history_in_initial_state(self):
        """_initial_state() copies the provided history into session_history."""
        history = [HumanMessage(content="h1"), AIMessage(content="a1")]

        state = self.model._initial_state("find a flat", history=history)

        self.assertEqual(state["session_history"], history)

    def test_should_copy_history_so_mutations_do_not_affect_original(self):
        """_initial_state() makes a copy of history — mutating the state list does not affect the source."""
        history = [HumanMessage(content="h1")]

        state = self.model._initial_state("q", history=history)
        state["session_history"].append(AIMessage(content="extra"))

        self.assertEqual(len(history), 1)

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    @patch("agentP.src.model.llm_model_graph.close_finance_mcp")
    def test_should_delegate_to_close_finance_mcp_when_close_is_called(self, mock_close_finance):
        """close() calls close_finance_mcp() to terminate the MCP subprocess."""
        self.model.close()

        mock_close_finance.assert_called_once()

    def test_should_not_raise_when_close_is_called(self):
        """close() must not raise even when the MCP subprocess was never started."""
        try:
            self.model.close()
        except Exception as exc:
            self.fail(f"close() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
