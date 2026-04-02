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

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk

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

        After construction the compiled LangGraph and the tool-bound LLM are
        replaced with plain MagicMocks so that ask() / ask_stream() and node
        tests are fast and isolated.
        """
        mock_load_file.side_effect = [_FAKE_SYSTEM_PROMPT, _FAKE_REFORMULATION_TEMPLATE]

        self.mock_llm = MagicMock()
        self.model = LlmModelGraph(self.mock_llm)

        # Replace tool-bound LLM so node tests control what the agent LLM returns.
        self.mock_llm_with_tools = MagicMock(name="LlmWithToolsMock")
        self.model._llm_with_tools = self.mock_llm_with_tools

        # Replace the compiled graph so public-API tests don't run real LangGraph nodes.
        self.model.graph = MagicMock(name="CompiledGraphMock")

    # ------------------------------------------------------------------
    # ask()
    # ------------------------------------------------------------------

    def test_should_return_answer_string_when_ask_is_called(self):
        """ask() returns the content of the last message from the graph result."""
        self.model.graph.invoke.return_value = {
            "messages": [AIMessage(content="3 listings found in Warsaw.")],
        }

        result = self.model.ask("3-bed apartment in Warsaw")

        self.assertEqual(result, "3 listings found in Warsaw.")

    def test_should_invoke_graph_once_when_ask_is_called(self):
        """ask() delegates to graph.invoke exactly once."""
        self.model.graph.invoke.return_value = {"messages": [AIMessage(content="ok")]}

        self.model.ask("3-bed apartment in Warsaw")

        self.model.graph.invoke.assert_called_once()

    def test_should_build_history_from_query_and_answer_when_ask_is_called(self):
        """ask() appends HumanMessage(user_query) + AIMessage(answer) to the session history."""
        SESSION = "sess-history-ask"
        self.model.graph.invoke.return_value = {
            "messages": [AIMessage(content="3 listings found.")]
        }

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
        self.model.graph.invoke.return_value = {
            "messages": [AIMessage(content="answer 2")]
        }

        self.model.ask("turn 2", session_id=SESSION)

        hist = self.model._histories[SESSION]
        self.assertEqual(len(hist), 4)
        self.assertEqual(hist[-2].content, "turn 2")
        self.assertEqual(hist[-1].content, "answer 2")

    def test_should_pass_correct_initial_state_when_ask_is_called(self):
        """ask() builds a State with user_prompt, empty reformulated_question, messages=[], and session_history."""
        self.model.graph.invoke.return_value = {"messages": [AIMessage(content="")]}

        self.model.ask("find a flat", session_id="sess-state")

        state_arg = self.model.graph.invoke.call_args[0][0]
        self.assertEqual(state_arg["user_prompt"], "find a flat")
        self.assertEqual(state_arg["reformulated_question"], "")
        self.assertEqual(state_arg["messages"], [])
        self.assertEqual(state_arg["session_history"], [])
        self.assertNotIn("context", state_arg)
        self.assertNotIn("answer", state_arg)
        self.assertNotIn("history", state_arg)

    def test_should_isolate_history_between_different_sessions(self):
        """ask() stores history independently per session_id — sessions do not share history."""
        self.model.graph.invoke.side_effect = [
            {"messages": [AIMessage(content="answer A")]},
            {"messages": [AIMessage(content="answer B")]},
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
            {"messages": [AIMessage(content="answer 1")]},
            {"messages": [AIMessage(content="answer 2")]},
        ]

        self.model.ask("turn 1", session_id=SESSION)
        self.model.ask("turn 2", session_id=SESSION)

        hist = self.model._histories[SESSION]
        self.assertEqual(len(hist), 4)
        self.assertEqual(hist[0].content, "turn 1")
        self.assertEqual(hist[2].content, "turn 2")

    def test_should_generate_session_id_when_none_provided_to_ask(self):
        """ask() with no session_id creates a new session and stores history under it."""
        self.model.graph.invoke.return_value = {"messages": [AIMessage(content="ok")]}

        self.model.ask("find a flat")

        self.assertEqual(len(self.model._histories), 1)
        session_id = next(iter(self.model._histories))
        self.assertEqual(len(self.model._histories[session_id]), 2)

    def test_should_include_recursion_limit_in_graph_invoke_config(self):
        """ask() passes recursion_limit=10 in the config to graph.invoke."""
        self.model.graph.invoke.return_value = {"messages": [AIMessage(content="ok")]}

        self.model.ask("find a flat", session_id="sess-rl")

        config = self.model.graph.invoke.call_args[1]["config"]
        self.assertEqual(config["recursion_limit"], 10)

    # ------------------------------------------------------------------
    # ask_stream()
    # ------------------------------------------------------------------

    def _stream_chunk(self, content: str, node: str, tool_call_chunks=None):
        """Helper: create a (AIMessageChunk, metadata) pair as graph.stream yields."""
        chunk = AIMessageChunk(content=content, tool_call_chunks=tool_call_chunks or [])
        return (chunk, {"langgraph_node": node})

    def test_should_yield_tokens_from_agent_node_when_streaming(self):
        """ask_stream() yields text content only from the 'agent' node."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("Here ", "agent"),
            self._stream_chunk("are ", "agent"),
            self._stream_chunk("some properties.", "agent"),
        ])

        tokens = list(self.model.ask_stream("find a flat"))

        self.assertEqual(tokens, ["Here ", "are ", "some properties."])

    def test_should_ignore_non_agent_nodes_when_streaming(self):
        """ask_stream() does not yield chunks from reformulate or tools nodes."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("reformulated query", "reformulate"),
            self._stream_chunk("tool result", "tools"),
        ])

        tokens = list(self.model.ask_stream("find a flat"))

        self.assertEqual(tokens, [])

    def test_should_skip_empty_tokens_when_streaming(self):
        """ask_stream() does not yield empty-string chunks from the agent node."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("", "agent"),
            self._stream_chunk("hello", "agent"),
            self._stream_chunk("", "agent"),
        ])

        tokens = list(self.model.ask_stream("test"))

        self.assertEqual(tokens, ["hello"])

    def test_should_skip_tool_call_chunks_when_streaming(self):
        """ask_stream() does not yield chunks that carry tool_call_chunks (tool invocations)."""
        tool_chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "property_search", "args": "", "id": "1", "index": 0}],
        )
        text_chunk = AIMessageChunk(content="Here are the results.")
        self.model.graph.stream.return_value = iter([
            (tool_chunk, {"langgraph_node": "agent"}),
            (text_chunk, {"langgraph_node": "agent"}),
        ])

        tokens = list(self.model.ask_stream("find flats in Warsaw"))

        self.assertEqual(tokens, ["Here are the results."])

    def test_should_yield_nothing_when_stream_has_no_agent_chunks(self):
        """ask_stream() yields an empty sequence when the graph produces no agent-node output."""
        self.model.graph.stream.return_value = iter([])

        tokens = list(self.model.ask_stream("find a flat"))

        self.assertEqual(tokens, [])

    def test_should_append_human_and_ai_messages_to_history_when_streaming(self):
        """ask_stream() appends HumanMessage + AIMessage to the session history after completion."""
        SESSION = "sess-stream-hist"
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("answer text", "agent"),
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
            self._stream_chunk("answer 2", "agent"),
        ])

        list(self.model.ask_stream("turn 2", session_id=SESSION))

        hist = self.model._histories[SESSION]
        self.assertEqual(len(hist), 4)
        self.assertEqual(hist[-2].content, "turn 2")
        self.assertEqual(hist[-1].content, "answer 2")

    def test_should_generate_session_id_when_none_provided_to_ask_stream(self):
        """ask_stream() with no session_id creates a new session and stores history under it."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("ok", "agent"),
        ])

        list(self.model.ask_stream("find a flat"))

        self.assertEqual(len(self.model._histories), 1)
        session_id = next(iter(self.model._histories))
        self.assertEqual(len(self.model._histories[session_id]), 2)

    def test_should_include_recursion_limit_in_graph_stream_config(self):
        """ask_stream() passes recursion_limit=10 in the config to graph.stream."""
        self.model.graph.stream.return_value = iter([])

        list(self.model.ask_stream("find a flat", session_id="sess-rl-stream"))

        config = self.model.graph.stream.call_args[1]["config"]
        self.assertEqual(config["recursion_limit"], 10)

    # ------------------------------------------------------------------
    # _agent_node()
    # ------------------------------------------------------------------

    def _make_agent_state(self, messages=None, reformulated="2 bedroom Warsaw", session_history=None):
        """Helper: build a minimal State for _agent_node tests."""
        return {
            "user_prompt": "find a flat",
            "reformulated_question": reformulated,
            "messages": messages if messages is not None else [],
            "session_history": session_history if session_history is not None else [],
        }

    def test_should_build_initial_messages_with_system_prompt_and_history_when_first_agent_call(self):
        """On first call (empty messages), _agent_node prepends system prompt + session_history + question."""
        prior = [
            HumanMessage(content="prev question"),
            AIMessage(content="prev answer"),
        ]
        self.mock_llm_with_tools.invoke.return_value = AIMessage(content="some answer")
        state = self._make_agent_state(messages=[], reformulated="2 bed Warsaw", session_history=prior)

        self.model._agent_node(state)

        call_args = self.mock_llm_with_tools.invoke.call_args[0][0]
        self.assertIsInstance(call_args[0], SystemMessage)
        self.assertEqual(call_args[0].content, _FAKE_SYSTEM_PROMPT)
        self.assertEqual(call_args[1].content, "prev question")
        self.assertEqual(call_args[2].content, "prev answer")
        self.assertIsInstance(call_args[-1], HumanMessage)
        self.assertEqual(call_args[-1].content, "2 bed Warsaw")

    def test_should_pass_existing_messages_unchanged_when_subsequent_agent_call(self):
        """On subsequent calls (non-empty messages), _agent_node passes messages as-is."""
        existing = [
            SystemMessage(content=_FAKE_SYSTEM_PROMPT),
            HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[{"name": "property_search", "args": {}, "id": "1"}]),
        ]
        self.mock_llm_with_tools.invoke.return_value = AIMessage(content="final answer")
        state = self._make_agent_state(messages=existing)

        self.model._agent_node(state)

        call_args = self.mock_llm_with_tools.invoke.call_args[0][0]
        self.assertEqual(call_args, existing)

    def test_should_return_messages_list_from_agent_node(self):
        """_agent_node() returns {"messages": [<AIMessage response>]}."""
        response = AIMessage(content="Here are listings.")
        self.mock_llm_with_tools.invoke.return_value = response
        state = self._make_agent_state()

        result = self.model._agent_node(state)

        self.assertEqual(result, {"messages": [response]})

    def test_should_not_modify_histories_dict_inside_agent_node(self):
        """_agent_node() must not mutate self._histories — history management belongs in ask/ask_stream."""
        self.model._histories["existing"] = [HumanMessage(content="prior")]
        before = {"existing": list(self.model._histories["existing"])}
        self.mock_llm_with_tools.invoke.return_value = AIMessage(content="response")

        self.model._agent_node(self._make_agent_state())

        self.assertEqual(self.model._histories, before)

    # ------------------------------------------------------------------
    # _reformulate_node()
    # ------------------------------------------------------------------

    def test_should_return_reformulated_question_when_reformulating(self):
        """_reformulate_node() invokes the LLM chain and stores the output."""
        self.mock_llm.return_value = AIMessage(
            content="2 bedroom apartment Warsaw affordable"
        )
        state: State = {
            "user_prompt": "cheap 2 bed Warsaw",
            "reformulated_question": "",
            "messages": [],
            "session_history": [],
        }

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

    def test_should_have_empty_fields_when_building_initial_state(self):
        """_initial_state() initialises reformulated_question='', messages=[], session_history=[]."""
        state = self.model._initial_state("anything")
        self.assertEqual(state["reformulated_question"], "")
        self.assertEqual(state["messages"], [])
        self.assertEqual(state["session_history"], [])
        self.assertNotIn("context", state)
        self.assertNotIn("answer", state)
        self.assertNotIn("history", state)

    def test_should_include_session_history_in_initial_state(self):
        """_initial_state() copies the provided history into session_history."""
        history = [HumanMessage(content="h1"), AIMessage(content="a1")]
        state = self.model._initial_state("find a flat", history=history)
        self.assertEqual(state["session_history"], history)

    def test_should_copy_history_so_mutations_do_not_affect_original(self):
        """_initial_state() makes a copy of history — mutating the state does not affect the source list."""
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
