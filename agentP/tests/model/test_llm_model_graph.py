import unittest
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage

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
          - _load_file  → returns fake prompt strings (no filesystem access)
          - Embedder    → no SentenceTransformer model is loaded
          - RagContextManager → no vector store is queried

        After construction the compiled LangGraph is replaced with a plain
        MagicMock so that ask() / ask_stream() tests are fast and isolated.
        """
        mock_load_file.side_effect = [_FAKE_SYSTEM_PROMPT, _FAKE_REFORMULATION_TEMPLATE]

        self.mock_llm = MagicMock()
        self.model = LlmModelGraph(self.mock_llm)

        # Replace the compiled graph so public-API tests don't run real nodes.
        self.model.graph = MagicMock(name="CompiledGraphMock")

    # ------------------------------------------------------------------
    # ask()
    # ------------------------------------------------------------------

    def test_ask_returns_answer(self):
        """ask() invokes the graph and returns the answer string."""
        self.model.graph.invoke.return_value = {
            "answer": "3 listings found in Warsaw.",
            "history": [],
        }

        result = self.model.ask("3-bed apartment in Warsaw")

        self.assertEqual(result, "3 listings found in Warsaw.")
        self.model.graph.invoke.assert_called_once()

    def test_ask_updates_history_from_graph_result(self):
        """ask() replaces self.history with whatever the graph returns."""
        new_history = [HumanMessage(content="hi"), AIMessage(content="hello")]
        self.model.graph.invoke.return_value = {"answer": "ok", "history": new_history}

        self.model.ask("hi")

        self.assertEqual(self.model.history, new_history)

    def test_ask_passes_correct_initial_state(self):
        """ask() builds the correct State dict and passes it to graph.invoke."""
        prior_history = [HumanMessage(content="previous turn")]
        self.model.history = prior_history
        self.model.graph.invoke.return_value = {"answer": "", "history": []}

        self.model.ask("find a flat")

        state_arg = self.model.graph.invoke.call_args[0][0]
        self.assertEqual(state_arg["user_prompt"], "find a flat")
        self.assertEqual(state_arg["reformulated_question"], "")
        self.assertEqual(state_arg["context"], "")
        self.assertEqual(state_arg["answer"], "")
        self.assertEqual(state_arg["history"], prior_history)

    # ------------------------------------------------------------------
    # ask_stream()
    # ------------------------------------------------------------------

    def _stream_chunk(self, content: str, node: str):
        """Helper: create a (chunk, metadata) pair as graph.stream yields."""
        chunk = MagicMock()
        chunk.content = content
        return (chunk, {"langgraph_node": node})

    def test_ask_stream_yields_tokens_from_generate_node(self):
        """ask_stream() yields token content only from the generate node."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("Here ", "generate"),
            self._stream_chunk("are ", "generate"),
            self._stream_chunk("some properties.", "generate"),
        ])

        tokens = list(self.model.ask_stream("find a flat"))

        self.assertEqual(tokens, ["Here ", "are ", "some properties."])

    def test_ask_stream_ignores_non_generate_nodes(self):
        """ask_stream() does not yield chunks from reformulate or retrieve nodes."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("reformulated query", "reformulate"),
            self._stream_chunk("retrieved context", "retrieve"),
        ])

        tokens = list(self.model.ask_stream("find a flat"))

        self.assertEqual(tokens, [])

    def test_ask_stream_skips_empty_tokens(self):
        """ask_stream() does not yield empty-string chunks from the generate node."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("", "generate"),
            self._stream_chunk("hello", "generate"),
            self._stream_chunk("", "generate"),
        ])

        tokens = list(self.model.ask_stream("test"))

        self.assertEqual(tokens, ["hello"])

    def test_ask_stream_appends_messages_to_history(self):
        """ask_stream() appends HumanMessage + AIMessage to self.history after completion."""
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("answer text", "generate"),
        ])

        list(self.model.ask_stream("find a flat"))

        self.assertEqual(len(self.model.history), 2)
        self.assertIsInstance(self.model.history[0], HumanMessage)
        self.assertIsInstance(self.model.history[1], AIMessage)
        self.assertEqual(self.model.history[0].content, "find a flat")
        self.assertEqual(self.model.history[1].content, "answer text")

    def test_ask_stream_accumulates_history_across_turns(self):
        """ask_stream() appends to pre-existing history rather than replacing it."""
        self.model.history = [
            HumanMessage(content="turn 1"),
            AIMessage(content="answer 1"),
        ]
        self.model.graph.stream.return_value = iter([
            self._stream_chunk("answer 2", "generate"),
        ])

        list(self.model.ask_stream("turn 2"))

        self.assertEqual(len(self.model.history), 4)
        self.assertEqual(self.model.history[-2].content, "turn 2")
        self.assertEqual(self.model.history[-1].content, "answer 2")

    # ------------------------------------------------------------------
    # _retrieve_node()
    # ------------------------------------------------------------------

    def test_retrieve_node_calls_context_manager_with_reformulated_question(self):
        """_retrieve_node() passes reformulated_question to get_context()."""
        self.model.rag_context_manager.get_context.return_value = "Listing 1: ..."
        state: State = {
            "user_prompt": "find a flat",
            "reformulated_question": "2 bedroom apartment Warsaw",
            "context": "",
            "answer": "",
            "history": [],
        }

        self.model._retrieve_node(state)

        self.model.rag_context_manager.get_context.assert_called_once_with(
            "2 bedroom apartment Warsaw"
        )

    def test_retrieve_node_returns_context_dict(self):
        """_retrieve_node() returns {"context": <string from get_context>}."""
        self.model.rag_context_manager.get_context.return_value = "Listing 1: Warsaw flat"
        state: State = {
            "user_prompt": "find a flat",
            "reformulated_question": "2 bedroom Warsaw",
            "context": "",
            "answer": "",
            "history": [],
        }

        result = self.model._retrieve_node(state)

        self.assertEqual(result, {"context": "Listing 1: Warsaw flat"})

    # ------------------------------------------------------------------
    # _reformulate_node()
    # ------------------------------------------------------------------

    def test_reformulate_node_returns_reformulated_question(self):
        """_reformulate_node() invokes the LLM chain and stores the output."""
        # The node builds: ChatPromptTemplate | llm | StrOutputParser.
        # Returning an AIMessage makes StrOutputParser extract its .content.
        self.mock_llm.return_value = AIMessage(
            content="2 bedroom apartment Warsaw affordable"
        )
        state: State = {
            "user_prompt": "cheap 2 bed Warsaw",
            "reformulated_question": "",
            "context": "",
            "answer": "",
            "history": [],
        }

        result = self.model._reformulate_node(state)

        self.assertEqual(
            result["reformulated_question"],
            "2 bedroom apartment Warsaw affordable",
        )

    # ------------------------------------------------------------------
    # _generate_node()
    # ------------------------------------------------------------------

    def test_generate_node_returns_answer(self):
        """_generate_node() invokes the LLM chain and returns the answer string."""
        self.mock_llm.return_value = AIMessage(content="Here are 3 properties in Warsaw.")
        state: State = {
            "user_prompt": "find a flat",
            "reformulated_question": "2 bedroom apartment Warsaw",
            "context": "Listing 1: Warsaw flat",
            "answer": "",
            "history": [],
        }

        result = self.model._generate_node(state)

        self.assertEqual(result["answer"], "Here are 3 properties in Warsaw.")

    def test_generate_node_appends_turn_to_history(self):
        """_generate_node() appends HumanMessage + AIMessage for the current turn."""
        self.mock_llm.return_value = AIMessage(content="Found 2 properties.")
        state: State = {
            "user_prompt": "find a flat",
            "reformulated_question": "2 bedroom apartment Warsaw",
            "context": "Listing 1",
            "answer": "",
            "history": [],
        }

        result = self.model._generate_node(state)
        history = result["history"]

        self.assertEqual(len(history), 2)
        self.assertIsInstance(history[0], HumanMessage)
        self.assertIsInstance(history[1], AIMessage)
        self.assertEqual(history[0].content, "find a flat")
        self.assertEqual(history[1].content, "Found 2 properties.")

    def test_generate_node_preserves_existing_history(self):
        """_generate_node() keeps prior history intact before appending the new turn."""
        existing = [HumanMessage(content="old q"), AIMessage(content="old a")]
        self.mock_llm.return_value = AIMessage(content="new answer")
        state: State = {
            "user_prompt": "new question",
            "reformulated_question": "new reformulated",
            "context": "context",
            "answer": "",
            "history": existing,
        }

        result = self.model._generate_node(state)
        history = result["history"]

        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].content, "old q")
        self.assertEqual(history[1].content, "old a")
        self.assertEqual(history[2].content, "new question")
        self.assertEqual(history[3].content, "new answer")

    # ------------------------------------------------------------------
    # _initial_state()
    # ------------------------------------------------------------------

    def test_initial_state_sets_user_prompt(self):
        """_initial_state() stores the query string in user_prompt."""
        state = self.model._initial_state("find me a studio")
        self.assertEqual(state["user_prompt"], "find me a studio")

    def test_initial_state_has_empty_derived_fields(self):
        """_initial_state() initialises reformulated_question, context, and answer to ''."""
        state = self.model._initial_state("anything")
        self.assertEqual(state["reformulated_question"], "")
        self.assertEqual(state["context"], "")
        self.assertEqual(state["answer"], "")

    def test_initial_state_copies_current_history(self):
        """_initial_state() makes an independent copy — mutations don't bleed back."""
        original = [HumanMessage(content="hi")]
        self.model.history = original

        state = self.model._initial_state("test")

        # Same contents …
        self.assertEqual(state["history"], original)
        # … but a separate list.
        state["history"].append(AIMessage(content="extra"))
        self.assertEqual(self.model.history, original)

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    def test_close_does_not_raise(self):
        """close() is a documented no-op and must not raise."""
        try:
            self.model.close()
        except Exception as exc:
            self.fail(f"close() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
