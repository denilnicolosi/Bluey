from datetime import datetime

from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import SystemMessage


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
