from datetime import datetime

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    dynamic_prompt,
)
from langchain_core.messages import SystemMessage

from tracing import log_hitl_decision


@dynamic_prompt
def dynamic_prompt_middleware(request: ModelRequest) -> SystemMessage | str:
    """Format system prompts with run-time dynamic values like current date/time."""
    if request.system_message and "{current_time}" in request.system_message.content:
        current_time = datetime.now().strftime("%A %d %B %Y, %H:%M")
        timezone = datetime.now().astimezone().tzinfo or "Europe/Rome"
        return request.system_message.content.format(
            current_time=current_time,
            timezone=str(timezone)
        )
    return request.system_message or ""


class HITLFeedbackMiddleware(HumanInTheLoopMiddleware):
    """HumanInTheLoopMiddleware that also records each operator decision as LangSmith feedback.

    Every approve/edit/reject an operator gives is a human label on a real proposed action.
    Recording it makes the approval rate per agent a first-class, queryable metric instead of
    a UI interaction that disappears once the conversation ends.
    """

    def __init__(self, *args, agent: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = agent

    # The base method is a @staticmethod (decision, tool_call, config); this override is an
    # instance method because it needs self._agent to label the feedback. The base always
    # calls it as `self._process_decision(...)`, so self binds correctly at runtime — but
    # that makes it an LSP-incompatible override, hence the pyright suppression.
    def _process_decision(self, decision, tool_call, config):  # pyright: ignore[reportIncompatibleMethodOverride]
        # ponytail: hooks `_process_decision`, a private method, because it is the single
        # point every decision passes through. Guarded so that a LangChain upgrade which
        # changes it degrades to "no feedback logged" instead of breaking approvals.
        # Upgrade path: drop this override if the middleware ever exposes a public hook.
        try:
            log_hitl_decision(
                agent=self._agent,
                tool_name=tool_call.get("name", "unknown"),
                decision_type=decision.get("type", "unknown"),
            )
        except Exception:  # noqa: BLE001 - telemetry must never block an approval
            pass
        return super()._process_decision(decision, tool_call, config)
