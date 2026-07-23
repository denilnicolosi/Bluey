"""
LangSmith tracing helpers: run labelling and human-in-the-loop feedback.

LangGraph already traces the whole graph automatically and groups runs into threads
via `thread_id`. What it cannot know is which *agent* a run belongs to, so nothing in
LangSmith can be sliced per agent (cost, latency, approval rate). These helpers attach
that missing dimension, and turn each operator HITL decision into a feedback score.

Everything here degrades to a no-op when tracing is off or LangSmith is unreachable:
observability must never be able to break a conversation.
"""

import os

from langsmith import Client
from langsmith.run_helpers import get_current_run_tree

# Distinguishes demo / evaluation / development traces inside the same project.
ENVIRONMENT = os.getenv("BLUEY_ENV", "dev")

_client_cache: Client | None = None
_client_failed = False


def tracing_enabled() -> bool:
    """True when LANGSMITH_TRACING is explicitly on."""
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("true", "1", "yes")


def _get_client() -> Client | None:
    """Return a cached LangSmith client, or None if tracing is off or unconfigured."""
    global _client_cache, _client_failed

    if _client_failed or not tracing_enabled():
        return None
    if _client_cache is None:
        try:
            _client_cache = Client()
        except Exception:  # noqa: BLE001 - missing/invalid API key must not break the agent
            _client_failed = True
            return None
    return _client_cache


def run_labels(agent: str, model: str | None) -> dict:
    """Tags and metadata identifying an agent, for use as chat-model constructor kwargs.

    Applied to the model rather than to a single call site so that *every* LLM run is
    labelled, including the SQL subgraph's internal nodes, which are never reached
    through `wrap_agent_as_tool`. Tags drive filtering in the UI; metadata drives
    grouping and the per-agent cost/latency breakdown.
    """
    return {
        "tags": [f"agent:{agent}", f"model:{model}", f"env:{ENVIRONMENT}"],
        "metadata": {"agent": agent, "model": model, "env": ENVIRONMENT},
    }


# Approve/edit/reject is a human judgement on a proposed action, so it maps onto a
# 1/0.5/0 score: an edit means the action was right but the arguments were not.
_DECISION_SCORES = {"approve": 1.0, "edit": 0.5, "reject": 0.0, "respond": 0.0}


def log_hitl_decision(agent: str, tool_name: str, decision_type: str, comment: str = "") -> None:
    """Record an operator HITL decision as LangSmith feedback on the current run.

    Every approval an operator gives is a free human label on a real action. Logging it
    turns the HITL gate into a live quality signal (approval rate per agent over time)
    instead of a one-off UI interaction.
    """
    client = _get_client()
    if client is None:
        return

    run_tree = get_current_run_tree()
    if run_tree is None:
        return

    try:
        client.create_feedback(
            run_id=run_tree.id,
            key="hitl_decision",
            score=_DECISION_SCORES.get(decision_type, 0.0),
            value=decision_type,
            comment=comment or f"{agent}: operator chose '{decision_type}' on {tool_name}",
            source_info={"agent": agent, "tool": tool_name, "env": ENVIRONMENT},
        )
    except Exception:  # noqa: BLE001 - feedback is best-effort telemetry
        pass
