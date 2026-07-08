"""
Script to generate full simulated conversations for Bluey system evaluation.

This script:
1. Reads a CSV file with test requests
2. Runs full conversations with the multi-agent system
3. Automatically handles HITL interrupts (auto-approve)
4. Saves conversations to a LangSmith dataset
"""

import csv
import uuid
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field

# Project directory (parent of evaluate/)
PROJECT_DIR = Path(__file__).parent.parent.resolve()

# Change working directory to the project root
os.chdir(PROJECT_DIR)

# Add parent directory to path
sys.path.insert(0, str(PROJECT_DIR))

from langsmith import Client
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables
load_dotenv(str(PROJECT_DIR / ".env"), override=True)

# Import the agent system
from agents import Agents

# --- CONFIGURATION ---
CSV_PATH = Path(__file__).parent / "evaluation_requests.csv"
DESTINATION_DATASET_NAME = "bluey_valutazioni"
os.environ["LANGSMITH_PROJECT"] = "Bluey_valutazione_generazione" 
MAX_ITERATIONS = 4  # Anti-loop safety
COMPLETION_CHECK_MODEL = "openai:gpt-5-nano" #"gpt-4o-mini"
MOCK_GOOGLE_APIS = True  # If True, Google APIs are not actually called
REPETITIONS_PER_EXAMPLE = 1  # Number of repetitions for each CSV example

client = Client()

# --- MOCK GOOGLE SERVICES ---
def _create_mock_google_service(api_name: str, api_version: str):
    """
    Mock for _build_google_service that simulates Google APIs.

    Tool calls appear in the LangSmith trace with full parameters,
    but no external operation is actually executed.
    """
    mock = MagicMock()
    
    if api_name == "gmail":
        # Mock for send_email: simulate successful send
        mock.users().messages().send.return_value.execute.return_value = {
            "id": "mock_message_id",
            "labelIds": ["SENT"]
        }
    elif api_name == "calendar":
        # Mock for get_available_time_slots: simulate free calendar
        mock.freebusy().query.return_value.execute.return_value = {
            "calendars": {}  # No busy slots = whole day free
        }
        # Mock for create_calendar_event: simulate created event
        mock.events().insert.return_value.execute.return_value = {
            "id": "mock_event_id",
            "htmlLink": "https://calendar.google.com/event?eid=mock_event_123"
        }
    return mock

# Callback to capture the last run_id
class RunIdCapture(BaseCallbackHandler):
    def __init__(self):
        self.run_id = None
    def on_chain_start(self, serialized, inputs, run_id, **kwargs):
        self.run_id = str(run_id)

# --- UTILITY FUNCTIONS ---

class UserSimulationResult(BaseModel):
    """Simulation result: completion evaluation and optional response."""
    is_completed: bool = Field(
        description="True if the initial request has been fully satisfied."
    )
    user_reply: str = Field(
        description="If is_completed=False, this is the simulated user reply to continue the conversation. If True, it is ignored.",
        default=None
    )
    reason: str = Field(
        description="Short reasoning about the decision and generated response."
    )

# Lazy singleton for simulator LLM (avoid reinitialization on each call)
_SIMULATOR_LLM = None

def _get_simulator():
    """Return simulator LLM, initializing only on first call."""
    global _SIMULATOR_LLM
    if _SIMULATOR_LLM is None:
        _SIMULATOR_LLM = init_chat_model(
            model=COMPLETION_CHECK_MODEL, 
            temperature=0.5
        ).with_structured_output(UserSimulationResult)
    return _SIMULATOR_LLM

def _extract_content(msg) -> str:
    """Extract text content from a message, handling lists and dicts."""
    content = getattr(msg, 'content', '')
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) 
            for b in content
        )
    return str(content) if content else ""

def simulate_user_step(result: dict) -> UserSimulationResult:
    """
    Evaluate whether the task is complete by analyzing invoke() result.
    If not complete, generate a simulated user reply.
    """
    # Extract messages (user/assistant with content only)
    messages = []
    for msg in result.get("messages", []):
        msg_type = getattr(msg, 'type', None)
        content = _extract_content(msg)
        if msg_type == "human" and content:
            messages.append(f"[USER]: {content}")
        elif msg_type == "ai" and content:
            messages.append(f"[ASSISTANT]: {content}")
    
    if len(messages) < 2:
        return UserSimulationResult(is_completed=False, user_reply=None, reason="Too few messages")
    
    conversation_text = "\n".join(messages)
    
    prompt = f"""
    You are the simulated USER managing this conversation. Your counterpart is Bluey, an AI assistant helping back-office operators in their daily work.

    Your task is to analyze the conversation against the initial request:
    1. If the assistant has fully satisfied your request (e.g., provided final information, confirmed action), set is_completed=True even if it suggests follow-up activities.
    2. ONLY if the assistant asks for missing DETAILS needed to satisfy and complete the request (e.g., dates, times, codes), set is_completed=False and generate a `user_reply`.

    INSTRUCTIONS FOR `user_reply`:
    - Reply as the user.
    - Be brief, direct, and consistent with the initial message goal.
    - If data is needed (dates, times, names) that is not in the initial request, INVENT it plausibly and consistently. (if an existing policy number is needed, use one of: [POL-5182-xogh, POL-3078-bwzO, POL-9201-yubw, POL-8571-stpu]; if customer references are needed, use one of: [(Name: Bianca; LastName: Paruta; Email: monicasegni@example.com), (Name: Fedele; LastName: Bassi; Email: rpedrazzini@example.org), (Name: Gino; LastName: Abatantuono; Email: adriano99@example.net)])
    - Do not request database creation or insertion operations.

    CONVERSATION:
    {conversation_text}
    """
    
    try:
        return _get_simulator().invoke(prompt)
    except Exception as e:
        print(f"      [WARN] User simulation failed: {e}")
        return UserSimulationResult(is_completed=True, reason="Error in simulation")


def _extract_tool_name_from_interrupt(interrupt) -> str | None:
    """Extract tool name from an Interrupt object."""
    if hasattr(interrupt, 'value') and isinstance(interrupt.value, dict):
        action_requests = interrupt.value.get('action_requests', [])
        if action_requests:
            return action_requests[0].get('name')
    return None


def get_pending_interrupt_info(result: dict, app=None, config=None) -> tuple[bool, str | None]:
    """
    Check whether there is a pending HITL interrupt.
    First checks direct result, then graph state for subgraph interrupts.
    
    Returns:
        Tuple (has_interrupt, tool_name)
    """
    # 1. Check __interrupt__ field in result
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"]
        if isinstance(interrupt_info, (list, tuple)) and interrupt_info:
            tool_name = _extract_tool_name_from_interrupt(interrupt_info[0])
            return True, tool_name
        return True, None
    
    # 2. Check graph state (for interrupts in subgraphs)
    if app and config:
        try:
            state = app.get_state(config)
            if hasattr(state, 'tasks') and state.tasks:
                for task in state.tasks:
                    if hasattr(task, 'interrupts') and task.interrupts:
                        tool_name = _extract_tool_name_from_interrupt(task.interrupts[0])
                        return True, tool_name
        except Exception:
            pass
    
    return False, None


def invoke_with_auto_approve(app, payload, config, max_hitl_retries: int = 3):
    """Invoke graph with auto-approval of HITL interrupts."""
    result = app.invoke(payload, config)
    
    for _ in range(max_hitl_retries):
        has_interrupt, tool_name = get_pending_interrupt_info(result, app, config)
        if not has_interrupt:
            break
            
        action_info = f"({tool_name})" if tool_name else ""
        print(f"      [HITL] Auto-approve {action_info}")
        result = app.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    
    return result


def _build_conversation_result(status: str, run_id: str, user_input: str, messages: list) -> dict:
    """Build conversation result dictionary."""
    return {
        "status": status,
        "run_id": run_id,
        "inputs": {"messages": user_input},
        "outputs": {"messages": _extract_all_messages({"messages": messages[1:]}) if messages else []}
    }


def run_single_conversation(user_input: str) -> dict:
    """
    Run a single full conversation.
    Returns the full conversation to save in the dataset.
    """
    agents = Agents(checkpointer=MemorySaver())
    app = agents.supervisor
    
    thread_id = str(uuid.uuid4())
    run_capture = RunIdCapture()
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [run_capture]
    }
    
    print(f"   --> Thread {thread_id[:8]}...")
    
    current_input = user_input
    all_messages = []
    
    for turn in range(MAX_ITERATIONS):
        try:
            msg_payload = {"messages": [HumanMessage(content=current_input)]}
            result = invoke_with_auto_approve(app, msg_payload, config)
        except Exception as e:
            print(f"      [ERROR] Invocation failed (turn {turn}): {e}")
            break

        all_messages = result.get("messages", [])
        sim_result = simulate_user_step(result)
        
        if sim_result.is_completed:
            print(f"      [COMPLETED] Conversation finished.")
            return _build_conversation_result("completed", run_capture.run_id, user_input, all_messages)
        
        if sim_result.user_reply:
            print(f"      [USER] Reply: '{sim_result.user_reply}'")
            current_input = sim_result.user_reply
        else:
            print(f"      [WARN] Not completed and no reply generated.")
            break
            
    print(f"      [MAX TURNS] Limit reached.")
    return _build_conversation_result("max_iterations", run_capture.run_id, user_input, all_messages)


def _extract_all_messages(result: dict) -> list:
    """Extract all messages including tool and tool_calls."""
    messages = []
    for msg in result.get("messages", []):
        msg_type = getattr(msg, 'type', None)
        content = _extract_content(msg)
        
        if msg_type == "human" and content:
            messages.append({"role": "user", "content": content})
        elif msg_type == "ai":
            msg_data = {"role": "assistant"}
            if content:
                msg_data["content"] = content
            tool_calls = getattr(msg, 'tool_calls', [])
            if tool_calls:
                msg_data["tool_calls"] = [
                    {"name": tc.get("name", ""), "args": tc.get("args", {})}
                    for tc in tool_calls
                ]
            if content or tool_calls:
                messages.append(msg_data)
        elif msg_type == "tool":
            tool_content = getattr(msg, 'content', '')
            if tool_content:
                messages.append({
                    "role": "tool",
                    "name": getattr(msg, 'name', 'unknown'),
                    "content": str(tool_content)
                })
    return messages



def _run_generation_loop(examples: list):
    """Run conversation generation loop."""
    total = len(examples) * REPETITIONS_PER_EXAMPLE
    print(f"\n[START] Generating {total} conversations ({len(examples)} examples x {REPETITIONS_PER_EXAMPLE} repetitions)...")
    print("-" * 60)
    stats = {"completed": 0, "max_iterations": 0, "error": 0}
    count = 0
    
    for example in examples:
        msg = example["message"]
        category = example.get("category", "general")
        
        preview = msg[:65] + "..." if len(msg) > 65 else msg
        print(f"\n[{category.upper()}] '{preview}' x{REPETITIONS_PER_EXAMPLE}")
        
        for i in range(REPETITIONS_PER_EXAMPLE):
            count += 1
            print(f"  [{count}/{total}] Run {i+1}/{REPETITIONS_PER_EXAMPLE}")
             
            try:

                result = run_single_conversation(msg)
                stats[result.get("status", "error")] += 1
                          
                client.create_example(
                    dataset_name=DESTINATION_DATASET_NAME,
                    inputs=result.get("inputs"),
                    outputs=result.get("outputs"),
                    source_run_id=result.get("run_id")
                )
            except Exception as e:
                print(f"      [ERROR] Conversation error {msg}: {e}")
    
    print("\n" + "=" * 60)
    print(f"COMPLETED: {count} conversations")
    for k, v in stats.items():
        if v > 0:
            print(f"  - {k}: {v}")
    print("=" * 60)


def main():
    print("=" * 60)
    try:
        if not client.has_dataset(dataset_name=DESTINATION_DATASET_NAME):
            client.create_dataset(
                dataset_name=DESTINATION_DATASET_NAME,
                description="Simulated conversations for Bluey evaluation"
            )
            print(f"[DATASET] Dataset '{DESTINATION_DATASET_NAME}' created")
        else:
            print(f"[DATASET] Dataset '{DESTINATION_DATASET_NAME}' already exists")
    except Exception as e:
        print(f"[ERROR] Dataset: {e}")
        return
    
    # CSV
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            examples = list(csv.DictReader(f))
        print(f"[CSV] {len(examples)} requests read from {CSV_PATH.name}")
    except Exception as e:
        print(f"[ERROR] CSV: {e}")
        return
    
    if MOCK_GOOGLE_APIS:
        with patch('tools._build_google_service', side_effect=_create_mock_google_service):
            _run_generation_loop(examples)
    else:
        _run_generation_loop(examples)


if __name__ == "__main__":
    main()