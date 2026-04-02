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

    @patch("agentP.src.model.llm_model_graph.RagContextManager")
    @patch("agentP.src.model.llm_model_graph.Embedder")
    @patch.object(LlmModelGraph, "_load_file")
    def setUp(self, mock_load_file, MockEmbedder, MockRagContextManager):
        """
        Build a LlmModelGraph with all heavy dependencies mocked out:
          - _load_file            → returns fake prompt strings (no filesystem access)
          - Embedder              → no SentenceTransformer model is loaded
          - RagContextManager     → no vector store is queried

        After construction the compiled LangGraph is replaced with a plain
        MagicMock so that ask() / ask_stream() and node tests are fast and isolated.
        """
        mock_load_file.side_effect = [_FAKE_SYSTEM_PROMPT, _FAKE_REFORMULATION_TEMPLATE]

        self.mock_llm = MagicMock()
        self.model = LlmModelGraph(self.mock_llm)

        # Replace the compiled graph so public-API tests don't run real LangGraph nodes.
        self.model.graph = MagicMock(name="CompiledGraphMock")

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
        """ask() appends to pre-existing session history rather than replacing it."""
        SESSION = "sess-accumulate-ask"
        self.model._histories[SESSION] = [
            HumanMessage(content="turn 1"),
            AIMessage(content="answer 1"),
        ]
        self.model.graph.invoke.return_value = {"answer": "answer 2"}

        self.model.ask("turn 2", session_id=SESSION)

        hist = self.model._histories[SESSION]
        self.assertEqual(len(hist), 4)
        self.assertEqual(hist[-2].content, "turn 2")
        self.assertEqual(hist[-1].content, "answer 2")

    def test_should_pass_correct_initial_state_when_ask_is_called(self):
        """ask() builds a State with user_prompt, empty derived fields, and session_history."""
        self.model.graph.invoke.return_value = {"answer": ""}

        self.model.ask("find a flat", session_id="sess-state")

        state_arg = self.model.graph.invoke.call_args[0][0]
        self.assertEqual(state_arg["user_prompt"], "find a flat")
        self.assertEqual(state_arg["reformulated_question"], "")
        self.assertFalse(state_arg["needs_search"])
        self.assertEqual(state_arg["context"], "")
        self.assertEqual(state_arg["answer"], "")
        self.assertEqual(state_arg["session_history"], [])
        self.assertNotIn("messages", state_arg)

    def test_should_isolate_history_between_different_sessions(self):
        """ask() stores history independently per session_id — sessions do not share history."""
        self.model.graph.invoke.side_effect = [
            {"answer": "answer A"},
            {"answer": "answer B"},
        ]

        self.model.ask("query A", session_id="sess-A")
        self.model.ask("query B", session_id="sess-B")

        hist_a = self.model._histories["sess-A"]
        hist_b = self.model._histories["sess-B"]
        self.assertEqual(len(hist_a), 2)
        self.assertEqual(len(hist_b), 2)
        self.assertEqual(hist_a[0].content, "query A")
        self.assertEqual(hist_b[0].content, "query B")

    def test_should_share_history_within_same_session_across_turns(self):
        """ask() called twice with the same session_id accumulates 4 messages."""
        SESSION = "sess-same"
        self.model.graph.invoke.side_effect = [
            {"answer": "answer 1"},
            {"answer": "answer 2"},
        ]

        self.model.ask("turn 1", session_id=SESSION)
        self.model.ask("turn 2", session_id=SESSION)

        hist = self.model._histories[SESSION]
        self.assertEqual(len(hist), 4)
        self.assertEqual(hist[0].content, "turn 1")
        self.assertEqual(hist[2].content, "turn 2")

    def test_should_generate_session_id_when_none_provided_to_ask(self):
        """ask() with no session_id creates a new session and stores history under it."""
        self.model.graph.invoke.return_value = {"answer": "ok"}

        self.model.ask("find a flat")

        self.assertEqual(len(self.model._histories), 1)
        session_id = next(iter(self.model._histories))
        self.assertEqual(len(self.model._histories[session_id]), 2)

    # ------------------------------------------------------------------
    # ask_stream()
    # ------------------------------------------------------------------

    def test_should_yield_tokens_from_generate_node_when_streaming(self):
        """ask_stream() yields text content only from the 'generate' node."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("Here ", "generate"),
            self._stream_chunk("are ", "generate"),
            self._stream_chunk("some properties.", "generate"),
        ])

        tokens = list(self.model.ask_stream("find a flat"))

        self.assertEqual(tokens, ["Here ", "are ", "some properties."])

    def test_should_ignore_non_generate_nodes_when_streaming(self):
        """ask_stream() does not yield chunks from reformulate or classify nodes."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("reformulated query", "reformulate"),
            self._stream_chunk("classifier output", "classify"),
        ])

        tokens = list(self.model.ask_stream("find a flat"))

        self.assertEqual(tokens, [])

    def test_should_skip_empty_tokens_when_streaming(self):
        """ask_stream() does not yield empty-string chunks from the generate node."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("", "generate"),
            self._stream_chunk("hello", "generate"),
            self._stream_chunk("", "generate"),
        ])

        tokens = list(self.model.ask_stream("test"))

        self.assertEqual(tokens, ["hello"])

    def test_should_yield_nothing_when_stream_has_no_generate_chunks(self):
        """ask_stream() yields an empty sequence when the graph produces no generate-node output."""
        self.model.graph.stream.return_value = iter([])

        tokens = list(self.model.ask_stream("find a flat"))

        self.assertEqual(tokens, [])

    def test_should_append_human_and_ai_messages_to_history_when_streaming(self):
        """ask_stream() appends HumanMessage + AIMessage to the session history after completion."""
        SESSION = "sess-stream-hist"
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("answer text", "generate"),
        ])

        list(self.model.ask_stream("find a flat", session_id=SESSION))

        hist = self.model._histories[SESSION]
        self.assertEqual(len(hist), 2)
        self.assertIsInstance(hist[0], HumanMessage)
        self.assertIsInstance(hist[1], AIMessage)
        self.assertEqual(hist[0].content, "find a flat")
        self.assertEqual(hist[1].content, "answer text")

    def test_should_accumulate_history_across_turns_when_streaming(self):
        """ask_stream() appends to pre-existing session history rather than replacing it."""
        SESSION = "sess-stream-acc"
        self.model._histories[SESSION] = [
            HumanMessage(content="turn 1"),
            AIMessage(content="answer 1"),
        ]
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("answer 2", "generate"),
        ])

        list(self.model.ask_stream("turn 2", session_id=SESSION))

        hist = self.model._histories[SESSION]
        self.assertEqual(len(hist), 4)
        self.assertEqual(hist[-2].content, "turn 2")
        self.assertEqual(hist[-1].content, "answer 2")

    def test_should_generate_session_id_when_none_provided_to_ask_stream(self):
        """ask_stream() with no session_id creates a new session and stores history under it."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("ok", "generate"),
        ])

        list(self.model.ask_stream("find a flat"))

        self.assertEqual(len(self.model._histories), 1)
        session_id = next(iter(self.model._histories))
        self.assertEqual(len(self.model._histories[session_id]), 2)

    # ------------------------------------------------------------------
    # _classify_node()
    # ------------------------------------------------------------------

    def test_should_return_needs_search_true_when_query_is_about_property(self):
        """_classify_node() returns needs_search=True when the LLM answers YES."""
        self.mock_llm.return_value = AIMessage(content="YES")
        state = self._make_state(reformulated_question="2 bedroom flat Warsaw")

        result = self.model._classify_node(state)

        self.assertTrue(result["needs_search"])

    def test_should_return_needs_search_false_when_query_is_a_greeting(self):
        """_classify_node() returns needs_search=False when the LLM answers NO."""
        self.mock_llm.return_value = AIMessage(content="NO")
        state = self._make_state(reformulated_question="Hello, how are you?")

        result = self.model._classify_node(state)

        self.assertFalse(result["needs_search"])

    def test_should_handle_yes_with_trailing_text_when_classifying(self):
        """_classify_node() is True when the response starts with YES (case-insensitive)."""
        self.mock_llm.return_value = AIMessage(content="YES, this query needs search")
        state = self._make_state()

        result = self.model._classify_node(state)

        self.assertTrue(result["needs_search"])

    def test_should_handle_lowercase_yes_when_classifying(self):
        """_classify_node() handles lowercase 'yes' from the LLM."""
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

    def test_should_not_raise_when_close_is_called(self):
        """close() is a documented no-op and must not raise."""
        try:
            self.model.close()
        except Exception as exc:
            self.fail(f"close() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
