"""
Script to generate full simulated conversations for Bluey system evaluation.
PARALLEL VERSION with ThreadPoolExecutor.

This script:
1. Reads a CSV file with test requests
2. Runs conversations in parallel with a configurable worker pool
3. Automatically handles HITL interrupts (auto-approve)
4. Saves conversations to a LangSmith dataset in a thread-safe way
"""

import csv
import uuid
import sys
import os
import threading
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_ITERATIONS = 5  # Anti-loop safety
COMPLETION_CHECK_MODEL = "openai:gpt-5-nano"
MOCK_GOOGLE_APIS = True  # If True, Google APIs are not actually called
REPETITIONS_PER_EXAMPLE = 1  # Number of repetitions for each CSV example
MAX_WORKERS = 5  # Number of concurrent workers (configurable)

# --- LOGGING THREAD-SAFE ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Silence verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# --- LOCKS FOR SHARED RESOURCES ---
# Thread safety notes:
# - _stats_lock: protects stats dict during result aggregation
# - _client_lock: protects LangSmith client calls (create_example)
# - threading.local(): used for simulator LLM, each thread has its own instance
# - Agents/MemorySaver: each worker creates its own instances, no shared state
# - MagicMock: new instance created for each _create_mock_google_service call
# - uuid.uuid4(): thread-safe by design
# - logging: thread-safe natively in Python
_stats_lock = threading.Lock()
_client_lock = threading.Lock()

# LangSmith client (singleton) - API calls protected by _client_lock
client = Client()

# Thread-safe progress counter
_progress_lock = threading.Lock()
_completed_count = 0
_total_count = 0

def _update_progress(worker_id: int, status: str):
    """Update and print progress in a thread-safe way."""
    global _completed_count
    with _progress_lock:
        _completed_count += 1
        logger.info(f"[Worker-{worker_id}] Completed ({status}) - Progress: {_completed_count}/{_total_count}")


# --- MOCK GOOGLE SERVICES ---
def _create_mock_google_service(api_name: str, api_version: str):
    """
    Mock for _build_google_service that simulates Google APIs.

    Tool calls appear in LangSmith traces with full parameters,
    but no external operation is actually executed.
    """
    mock = MagicMock()
    
    if api_name == "gmail":
        mock.users().messages().send.return_value.execute.return_value = {
            "id": "mock_message_id",
            "labelIds": ["SENT"]
        }
    elif api_name == "calendar":
        mock.freebusy().query.return_value.execute.return_value = {
            "calendars": {}
        }
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


# --- MODELS ---
class UserSimulationResult(BaseModel):
    """Simulation result: completion evaluation and optional response."""
    is_completed: bool = Field(
        description="True if the initial request has been fully satisfied."
    )
    user_reply: str | None = Field(
        description="If is_completed=False, this is the simulated user reply to continue the conversation. If True, it is ignored.",
        default=None
    )
    reason: str = Field(
        description="Short reasoning about the decision and generated response."
    )


# Thread-local storage for simulator LLM (each thread has its own instance)
_thread_local = threading.local()

def _get_simulator():
    """Return simulator LLM using thread-local storage to avoid conflicts."""
    if not hasattr(_thread_local, 'simulator'):
        _thread_local.simulator = init_chat_model(
            model=COMPLETION_CHECK_MODEL, 
            temperature=0.5
        ).with_structured_output(UserSimulationResult)
    return _thread_local.simulator


def _extract_content(msg) -> str:
    """Extract text content from a message, handling lists and dicts."""
    content = getattr(msg, 'content', '')
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) 
            for b in content
        )
    return str(content) if content else ""


def simulate_user_step(result: dict, worker_id: int) -> UserSimulationResult:
    """
    Evaluate whether task is complete by analyzing invoke() result.
    If not complete, generate a simulated user reply.
    """
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
        logger.warning(f"[Worker-{worker_id}] User simulation failed: {e}")
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
    """
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"]
        if isinstance(interrupt_info, (list, tuple)) and interrupt_info:
            tool_name = _extract_tool_name_from_interrupt(interrupt_info[0])
            return True, tool_name
        return True, None
    
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


def invoke_with_auto_approve(app, payload, config, worker_id: int, max_hitl_retries: int = 3):
    """Invoke graph with auto-approval of HITL interrupts."""
    result = app.invoke(payload, config)
    
    for _ in range(max_hitl_retries):
        has_interrupt, tool_name = get_pending_interrupt_info(result, app, config)
        if not has_interrupt:
            break
            
        action_info = f"({tool_name})" if tool_name else ""
        logger.debug(f"[Worker-{worker_id}] HITL Auto-approve {action_info}")
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


def run_single_conversation(user_input: str, worker_id: int) -> dict:
    """
    Run a single full conversation (thread-safe version).

    Thread safety:
    - Agents and MemorySaver are instantiated locally (no shared state)
    - RunIdCapture is thread-local
    - uuid.uuid4() is thread-safe
    - Config uses a unique thread_id to isolate conversations
    """
    # Each thread creates its own instances - no shared state
    agents = Agents(checkpointer=MemorySaver())
    app = agents.supervisor
    
    # Thread-safe UUID to guarantee unique thread_id
    thread_id = str(uuid.uuid4())
    run_capture = RunIdCapture()  # Thread-local instance
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [run_capture]
    }
    
    current_input = user_input
    all_messages = []
    
    for turn in range(MAX_ITERATIONS):
        try:
            msg_payload = {"messages": [HumanMessage(content=current_input)]}
            result = invoke_with_auto_approve(app, msg_payload, config, worker_id)
        except Exception as e:
            logger.error(f"[Worker-{worker_id}] Invocation failed (turn {turn}): {e}")
            break

        all_messages = result.get("messages", [])
        sim_result = simulate_user_step(result, worker_id)
        
        if sim_result.is_completed:
            return _build_conversation_result("completed", run_capture.run_id, user_input, all_messages)
        
        if sim_result.user_reply:
            current_input = sim_result.user_reply
        else:
            break
            
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


def worker_task(work_item: dict) -> dict:
    """
    Task executed by each worker thread.
    
    Args:
        work_item: dictionary with 'message', 'category', 'worker_id', 'task_index'
    
    Returns:
        Dictionary with conversation result
    """
    worker_id = work_item["worker_id"]
    task_index = work_item["task_index"]
    total = work_item["total"]
    message = work_item["message"]
    category = work_item["category"]
    
    preview = message[:70] + "..." if len(message) > 70 else message
    logger.info(f"[Worker-{worker_id}] [{task_index}/{total}] Start: '{preview}'")
    
    try:
        result = run_single_conversation(message, worker_id)
        
        # Thread-safe save on LangSmith
        with _client_lock:
            client.create_example(
                dataset_name=DESTINATION_DATASET_NAME,
                inputs=result.get("inputs"),
                outputs=result.get("outputs"),
                source_run_id=result.get("run_id"),
                split="gpt-5"
            )
        
        status = result.get("status", "error")
        _update_progress(worker_id, status)
        return {"status": status, "task_index": task_index}
        
    except Exception as e:
        logger.error(f"[Worker-{worker_id}] [{task_index}/{total}] ERROR - Request NOT saved:")
        logger.error(f"    Message: '{message}'")
        logger.error(f"    Category: {category}")
        logger.error(f"    Exception: {e}")
        _update_progress(worker_id, "error")
        return {"status": "error", "task_index": task_index, "error": str(e), "message": message}


def _run_parallel_generation(examples: list):
    """Run conversation generation loop in parallel."""
    # Prepare work items expanded by repetitions
    work_items = []
    task_index = 0
    for example in examples:
        msg = example["message"]
        category = example.get("category", "general")
        for _ in range(REPETITIONS_PER_EXAMPLE):
            task_index += 1
            work_items.append({
                "message": msg,
                "category": category,
                "task_index": task_index,
                "total": len(examples) * REPETITIONS_PER_EXAMPLE,
                "worker_id": 0  # Updated by ThreadPoolExecutor
            })
    
    total = len(work_items)
    
    # Initialize global progress counter
    global _total_count, _completed_count
    _total_count = total
    _completed_count = 0
    
    logger.info(f"Starting parallel generation: {total} conversations with {MAX_WORKERS} workers")
    
    # Thread-safe statistics
    stats = {"completed": 0, "max_iterations": 0, "error": 0}
    
    # Assign worker_id in round-robin
    for i, item in enumerate(work_items):
        item["worker_id"] = (i % MAX_WORKERS) + 1
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker_task, item): item for item in work_items}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                status = result.get("status", "error")
                with _stats_lock:
                    if status in stats:
                        stats[status] += 1
                    else:
                        stats["error"] += 1
            except Exception as e:
                logger.error(f"Error in future: {e}")
                with _stats_lock:
                    stats["error"] += 1
    
    # Final report
    logger.info("=" * 60)
    logger.info(f"COMPLETED: {total} conversations")
    for k, v in stats.items():
        if v > 0:
            logger.info(f"  - {k}: {v}")
    logger.info("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info(f"Script parallelo - MAX_WORKERS: {MAX_WORKERS}")
    
    # Dataset
    try:
        if not client.has_dataset(dataset_name=DESTINATION_DATASET_NAME):
            client.create_dataset(
                dataset_name=DESTINATION_DATASET_NAME,
                description="Simulated conversations for Bluey evaluation"
            )
            logger.info(f"Dataset '{DESTINATION_DATASET_NAME}' created")
        else:
            logger.info(f"Dataset '{DESTINATION_DATASET_NAME}' already exists")
    except Exception as e:
        logger.error(f"Dataset error: {e}")
        return
    
    # CSV
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            examples = list(csv.DictReader(f))
        logger.info(f"{len(examples)} requests read from {CSV_PATH.name}")
    except Exception as e:
        logger.error(f"CSV error: {e}")
        return
    
    if MOCK_GOOGLE_APIS:
        with patch('tools._build_google_service', side_effect=_create_mock_google_service):
            _run_parallel_generation(examples)
    else:
        _run_parallel_generation(examples)


if __name__ == "__main__":
    main()
