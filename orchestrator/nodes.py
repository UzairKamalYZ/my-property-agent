"""
orchestrator/nodes.py — LangGraph node functions for the multi-agent pipeline.

WHAT IS A NODE?
---------------
A LangGraph node is a plain Python function that:
  - receives the full GraphState dict
  - does some work (calls an LLM, calls an agent, etc.)
  - returns a *partial* dict of only the keys it changed
LangGraph merges that partial dict back into the shared state automatically.

NODE OVERVIEW
-------------
supervisor_node   — asks the LLM which agent to call next (or FINISH).
                    Writes state["next"] with the routing decision.

agent_node        — calls one specialist agent, stores its response in
                    state["agent_outputs"], increments state["turns"].

synthesiser_node  — reads all agent_outputs and asks the LLM to produce
                    one clean, unified answer for the user.

FACTORY PATTERN
---------------
Each node is created by a "make_*" factory function rather than being a
plain function.  The factory captures (closes over) dependencies such as
the LLM instance and the prompt string so the node itself stays stateless
and takes only `state` as its argument — the signature LangGraph requires.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from .state import GraphState, MAX_AGENT_TURNS

logger = logging.getLogger(__name__)

# Strips markdown code fences (```json ... ```) from LLM output before parsing
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# Extracts the first {...} block when full JSON parsing fails
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Step 1 — Supervisor node
# ---------------------------------------------------------------------------

def make_supervisor_node(llm: BaseChatModel, prompt_template: str):
    """
    Factory — returns the supervisor node function.

    The supervisor is called after every agent turn.  It reads the current
    state (user question + what agents have said so far) and decides:
      - which agent to call next, OR
      - "FINISH" when enough information has been gathered

    The decision is returned as a JSON object: {"next": "<agent_name | FINISH>"}

    Parameters
    ----------
    llm              : the language model used to make routing decisions
    prompt_template  : text of orchestrator/prompts/supervisor.txt — contains
                       {user_input} and {agent_outputs} placeholders
    """

    def supervisor_node(state: GraphState) -> dict[str, Any]:
        # --- 1a. Format agent outputs collected so far into readable text ---
        agent_outputs_text = (
            "\n\n".join(
                f"[{name}]:\n{output}"
                for name, output in state.get("agent_outputs", {}).items()
            )
            or "None yet."
        )

        # --- 1b. Fill the prompt template with live state values ---
        prompt = (
            prompt_template
            .replace("{user_input}", state["user_input"])
            .replace("{agent_outputs}", agent_outputs_text)
        )

        # --- 1c. Ask the LLM to decide the next routing step ---
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Which agent should run next, or should we FINISH?"),
        ]
        raw = llm.invoke(messages)
        content = raw.content if hasattr(raw, "content") else str(raw)

        # --- 1d. Parse {"next": "..."} from the LLM response ---
        next_agent = _parse_next(content)

        logger.info("[supervisor] turns=%d → next=%s", state.get("turns", 0), next_agent)

        # --- 1e. Return only the key that changed ---
        return {"next": next_agent}

    return supervisor_node


def _parse_next(text: str) -> str:
    """
    Extract the routing token from the supervisor's raw LLM output.

    The supervisor is instructed to reply with plain JSON, but LLMs sometimes
    wrap output in markdown fences or add extra prose.  This function handles
    both cases gracefully:
      1. Strip markdown fences if present.
      2. Try a full JSON parse first (fast path).
      3. Fall back to regex extraction of the first {...} block.
      4. If everything fails, default to FINISH (safe fallback).
    """
    # Step 1 — remove ``` fences so json.loads can handle the text
    cleaned = _FENCE_RE.sub(r"\1", text).strip()

    # Step 2 — attempt direct parse (most LLMs comply most of the time)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "next" in data:
            return str(data["next"])
    except json.JSONDecodeError:
        pass

    # Step 3 — extract the first {...} block and retry
    m = _JSON_RE.search(cleaned)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict) and "next" in data:
                return str(data["next"])
        except json.JSONDecodeError:
            pass

    # Step 4 — safe fallback: stop the loop rather than route blindly
    logger.warning("[supervisor] could not parse next from %r — defaulting to FINISH", text[:200])
    return "FINISH"


# ---------------------------------------------------------------------------
# Step 2 — Agent nodes
# ---------------------------------------------------------------------------

def make_agent_node(agent_name: str, graph_instance):
    """
    Factory — returns an agent node function for the given agent.

    Each agent node:
      - pulls the user question from state
      - calls the agent's ask() method (which runs its own LangGraph RAG pipeline)
      - stores the response in state["agent_outputs"] under the agent's name
      - increments state["turns"] so the router can enforce the turn cap

    Parameters
    ----------
    agent_name     : the routing key for this agent (e.g. "property_agent")
    graph_instance : an instantiated BaseAgent (must expose ask(prompt, session_id) -> str)
    """

    def agent_node(state: GraphState) -> dict[str, Any]:
        prompt = state["user_input"]
        session_id = state.get("session_id", "default")

        # --- 2a. Invoke the specialist agent ---
        logger.info("[%s] calling with session=%s", agent_name, session_id)
        response_text = graph_instance.ask(prompt, session_id=session_id)
        logger.info("[%s] done, response_len=%d", agent_name, len(response_text))

        # --- 2b. Merge this agent's output into the shared dict ---
        # Use dict() copy so we don't mutate the existing state reference;
        # LangGraph expects us to return a new object for dict fields.
        updated_outputs = dict(state.get("agent_outputs", {}))
        updated_outputs[agent_name] = response_text

        # --- 2c. Return changed keys only ---
        return {
            "agent_outputs": updated_outputs,
            "messages": [AIMessage(content=response_text, name=agent_name)],
            "turns": state.get("turns", 0) + 1,   # router uses this to enforce MAX_AGENT_TURNS
        }

    # Give the closure a readable name for LangGraph's visualisation and logs
    agent_node.__name__ = agent_name
    return agent_node


# ---------------------------------------------------------------------------
# Step 3 — Synthesiser node
# ---------------------------------------------------------------------------

def make_synthesiser_node(llm: BaseChatModel, system_prompt: str):
    """
    Factory — returns the synthesiser node function.

    The synthesiser is the final node before END.  It receives all agent
    outputs collected during the turn and asks the LLM to combine them into
    one clean, user-facing answer.

    It does NOT call any agents — it only reads agent_outputs and writes
    back a single "synthesiser" key to that same dict.

    Parameters
    ----------
    llm           : the language model used to write the final answer
    system_prompt : text of orchestrator/prompts/synthesiser.txt
    """

    def synthesiser_node(state: GraphState) -> dict[str, Any]:
        agent_outputs = state.get("agent_outputs", {})

        # --- 3a. Format all agent outputs into a labelled block ---
        outputs_block = (
            "\n\n".join(
                f"### {name}\n{text}"
                for name, text in agent_outputs.items()
            )
            or "No agent outputs were collected."
        )

        # --- 3b. Build the message list for the synthesis LLM call ---
        #         System prompt sets the tone/format rules.
        #         Second system message injects the raw agent outputs.
        #         Human message provides the original user question as anchor.
        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=f"AGENT OUTPUTS:\n\n{outputs_block}"),
            HumanMessage(content=state["user_input"]),
        ]

        # --- 3c. Generate the final unified answer ---
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        logger.info("[synthesiser] final answer len=%d", len(answer))

        # --- 3d. Store the answer so agent_interface.py can extract it ---
        updated_outputs = dict(agent_outputs)
        updated_outputs["synthesiser"] = answer

        return {
            "agent_outputs": updated_outputs,
            "messages": [AIMessage(content=answer, name="synthesiser")],
        }

    return synthesiser_node
