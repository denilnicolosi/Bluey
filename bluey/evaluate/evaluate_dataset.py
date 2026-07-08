"""
Script to evaluate generated conversations using LLM-as-judge evaluators.

Implements the 5 metrics defined in the thesis (Chapter 7):
1. request_fulfilled - Request completion
2. hallucination - Absence of hallucinations
3. ambiguity_handling - Ambiguity management
4. orchestration - Orchestration quality
5. context_preservation - Context preservation

All metrics use a 1-5 numerical scale (LLM-as-a-judge best practice).

Applied openevals best practices:
- Use of `choices` for custom numeric scale
- Use of `system` for evaluator general instructions
- Use of `feedback_key` to identify each metric
- Wrapper to adapt LangSmith dataset data structure

"""

import sys
import os
from pathlib import Path
from langsmith import evaluate, Client
from langsmith.schemas import Run, Example
from openevals.llm import create_llm_as_judge
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Project directory (parent of evaluate/)
PROJECT_DIR = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

# Load environment variables
load_dotenv(str(PROJECT_DIR / ".env"), override=True)

# --- CONFIGURATION ---
DATASET_TO_EVALUATE = "bluey_valutazioni"
SPLIT_TO_EVALUATE = "gpt-5"  # None = full dataset, or e.g. "low_cost_models", "high_performance_models"
JUDGE_LLM = init_chat_model("openai:gpt-4o-mini", temperature=0)
# JUDGE_LLM = init_chat_model("google_genai:gemini-3-flash-preview", temperature=0)
SCORE_CHOICES = [float(i) for i in range(1, 6)]

client = Client()


# =============================================================================
# COMMON SYSTEM PROMPT FOR ALL EVALUATORS
# =============================================================================

SYSTEM_PROMPT = """You are an expert evaluator for multi-agent AI systems.

Your task is to evaluate conversations between users and an AI system called Bluey,
used in an insurance context to assist back-office operators.

The Bluey system is composed of:
- A Supervisor that coordinates requests
- documentation_agent: search in company documentation
- sql_agent: queries on customer/policy/claim database
- email_agent: compose and send emails
- appointment_agent: calendar appointment management

Evaluation scale: ALWAYS respond with a single integer from 1 to 5."""


# =============================================================================
# 1. REQUEST_FULFILLED - Request completion
# =============================================================================

REQUEST_FULFILLED_PROMPT = """Evaluate whether the user request has been completed correctly.

CRITERIA:
- 1: Request not addressed or completely wrong response
- 2: Attempt with significant errors or missing information
- 3: Partially satisfied, some correct information
- 4: Well satisfied with minor omissions
- 5: Completely and correctly satisfied

ORIGINAL REQUEST:
{inputs}

CONVERSATION:
{outputs}
"""

request_fulfilled_evaluator = create_llm_as_judge(
    prompt=REQUEST_FULFILLED_PROMPT,
    system=SYSTEM_PROMPT,
    feedback_key="request_fulfilled",
    judge=JUDGE_LLM,
    choices=SCORE_CHOICES,
)


# =============================================================================
# 2. GROUNDEDNESS - Factual grounding and absence of hallucinations
# =============================================================================

GROUNDEDNESS_PROMPT = """Evaluate how much system responses are grounded in available facts.

DEFINITION: A hallucination is invented information not supported by conversation
context or that contradicts what has already been discussed.

CONSIDER AS CONTEXT:
- Information provided by the user in the conversation
- Results from SQL queries and document searches
- Information already present in the dialogue

CRITERIA (HIGH score = responses well grounded in facts):
- 1: Invented information or contradictions with context
- 2: Multiple statements unsupported by context
- 3: Some inaccuracies or unverifiable generalizations
- 4: Consistent responses with rare imperfections
- 5: All information derived from context, no invention

ORIGINAL REQUEST:
{inputs}

CONVERSATION:
{outputs}
"""

groundedness_evaluator = create_llm_as_judge(
    prompt=GROUNDEDNESS_PROMPT,
    system=SYSTEM_PROMPT,
    feedback_key="groundedness",
    judge=JUDGE_LLM,
    choices=SCORE_CHOICES,
)


# =============================================================================
# 3. AMBIGUITY_HANDLING - Ambiguous request management
# =============================================================================

AMBIGUITY_HANDLING_PROMPT = """Evaluate how the agent handled potentially ambiguous requests.

A well-designed system should:
- Recognize ambiguous or incomplete requests
- Ask for clarification instead of making wrong assumptions
- If the request is clear, respond appropriately

IMPORTANT NOTE:
- If the original request is CLEAR and COMPLETE, the system should respond directly.
    In this case, assign a high score (4-5) if the response is appropriate.
- If the request is AMBIGUOUS or INCOMPLETE, the system should ask for clarification.
    Penalize arbitrary assumptions.

CRITERIA:
- 1: Arbitrary assumptions without asking for clarification when needed
- 2: Partial recognition with incomplete or irrelevant questions
- 3: Reasonable handling but improvable
- 4: Good handling with relevant questions, or direct response when request is clear
- 5: Perfect handling: appropriate clarification for ambiguous requests, correct response for clear requests

ORIGINAL REQUEST:
{inputs}

CONVERSATION:
{outputs}
"""

ambiguity_handling_evaluator = create_llm_as_judge(
    prompt=AMBIGUITY_HANDLING_PROMPT,
    system=SYSTEM_PROMPT,
    feedback_key="ambiguity_handling",
    judge=JUDGE_LLM,
    choices=SCORE_CHOICES,
)


# =============================================================================
# 4. ORCHESTRATION - Correct selection of specialized agents
# =============================================================================

ORCHESTRATION_PROMPT = """Evaluate the correct selection of specialized agents.

AGENT RESPONSIBILITIES:
- documentation_agent: search procedures, FAQs, company manuals.
    Use for: "how to...", "what is the procedure for...", policy questions
- sql_agent: queries on customer, policy, claim, premium database.
    Use for: customer details, policy status, payment history, expirations
- email_agent: compose and send emails to customers/colleagues.
    Use for: communications, reminders, appointment confirmations
- appointment_agent: calendar management, appointment booking.
    Use for: scheduling meetings, checking availability, reminders

CRITERIA:
- 1: Completely wrong agent or omitted when needed
- 2: Partially appropriate agent, questionable choices
- 3: Reasonable but not optimal selection
- 4: Good selection with minor imperfections
- 5: Perfect selection: all and only necessary agents involved

ORIGINAL REQUEST:
{inputs}

CONVERSATION (includes calls to agents):
{outputs}"""

orchestration_evaluator = create_llm_as_judge(
    prompt=ORCHESTRATION_PROMPT,
    system=SYSTEM_PROMPT,
    feedback_key="orchestration",
    judge=JUDGE_LLM,
    choices=SCORE_CHOICES,
)


# =============================================================================
# 5. CONTEXT_PRESERVATION - Context transfer to sub-agents
# =============================================================================

CONTEXT_PRESERVATION_PROMPT = """Evaluate the ability to preserve conversational context.

NOTE: Sub-agents do NOT have access to full history.
The Supervisor must include in delegated requests all relevant information
(names, policy numbers, dates, email addresses, etc.).

WHAT TO OBSERVE:
- Tool calls sent by the Supervisor to specialized agents
- Verify they contain relevant information mentioned in the conversation
- Check that names, policy numbers, dates, email addresses are correctly passed

CRITERIA:
- 1: Critical information lost, essential data not transferred
- 2: Several contextual details missing or incomplete
- 3: Context partially preserved, minor information missing
- 4: Good preservation with small non-critical omissions
- 5: Context perfectly preserved and transferred to sub-agents

ORIGINAL REQUEST:
{inputs}

CONVERSATION:
{outputs}"""

context_preservation_evaluator = create_llm_as_judge(
    prompt=CONTEXT_PRESERVATION_PROMPT,
    system=SYSTEM_PROMPT,
    feedback_key="context_preservation",
    judge=JUDGE_LLM,
    choices=SCORE_CHOICES,
)


# =============================================================================
# WRAPPER FOR LANGSMITH
# =============================================================================
# Needed because dataset data structure (example.inputs, example.outputs)
# differs from what openevals expects (inputs, outputs as strings).
# This pattern is documented at: https://github.com/langchain-ai/openevals#evaluate

def format_conversation(conversation: list) -> str:
    """Format conversation as string for evaluator prompts."""
    if not conversation:
        return "[Empty conversation]"
    
    formatted_lines = []
    for msg in conversation:
        role = msg.get('role', 'unknown').upper()
        content = msg.get('content', '')
        formatted_lines.append(f"[{role}]: {content}")
    
    return "\n\n".join(formatted_lines)


def create_langsmith_evaluator(openevals_evaluator):
    """
    Wrapper to adapt openevals evaluators to LangSmith evaluate() format.

    feedback_key is already defined in the openevals evaluator and is extracted
    automatically from the result (documented best practice).
    """
    def wrapped_evaluator(run: Run, example: Example) -> dict:
        # Keys aligned with generate_threads_dataset.py
        conversation = example.outputs.get("messages", []) if example.outputs else []
        original_req = example.inputs.get("messages", "") if example.inputs else ""
        
        formatted_conv = format_conversation(conversation)
        
        try:
            result = openevals_evaluator(
                inputs=original_req,
                outputs=formatted_conv,
            )
            
            # Extract values from result (always a dict)
            score_value = result.get("score", result.get("value", 0))
            key = result.get("key", result.get("feedback_key", "unknown"))
            comment = result.get("comment", result.get("reasoning", ""))
            
            numeric_score = float(score_value) if isinstance(score_value, (int, float)) else 0.0
            
            return {
                "key": key,
                "score": numeric_score,
                "comment": str(comment) if comment else ""
            }
            
        except Exception as e:
            return {
                "key": "error",
                "score": 0.0,
                "comment": f"Error during evaluation: {str(e)}"
            }
    
    return wrapped_evaluator


# =============================================================================
# EVALUATOR CONFIGURATION
# =============================================================================

# List of LangSmith-wrapped evaluators
EVALUATORS = [
    create_langsmith_evaluator(request_fulfilled_evaluator),
    create_langsmith_evaluator(groundedness_evaluator),
    create_langsmith_evaluator(ambiguity_handling_evaluator),
    create_langsmith_evaluator(orchestration_evaluator),
    create_langsmith_evaluator(context_preservation_evaluator),
]

# Metric names for logging
METRIC_NAMES = [
    "request_fulfilled",
    "groundedness", 
    "ambiguity_handling",
    "orchestration",
    "context_preservation"
]

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run offline evaluation for the conversation dataset."""
    print("=" * 60)
    print("BLUEY CONVERSATION EVALUATION")
    
    # Check dataset
    print(f"Dataset '{DATASET_TO_EVALUATE}'")
    if SPLIT_TO_EVALUATE:
        print(f"Selected split: '{SPLIT_TO_EVALUATE}'")
    try:
        if not client.has_dataset(dataset_name=DATASET_TO_EVALUATE):
            print(f"[ERROR] Dataset not found: {DATASET_TO_EVALUATE}")
            print("Run generate_threads_dataset.py first\n")
            return
    except Exception as e:
        print(f"[ERROR] {e}\n")
        return
    
    # Filter examples by split if specified
    if SPLIT_TO_EVALUATE:
        examples_to_evaluate = list(client.list_examples(
            dataset_name=DATASET_TO_EVALUATE,
            splits=[SPLIT_TO_EVALUATE]
        ))
        if not examples_to_evaluate:
            print(f"[ERROR] No example found for split '{SPLIT_TO_EVALUATE}'")
            return
        print(f"Found {len(examples_to_evaluate)} examples in split '{SPLIT_TO_EVALUATE}'")
        data_source = examples_to_evaluate
    else:
        data_source = DATASET_TO_EVALUATE
    
    try:
        experiment_results = evaluate(
            lambda inputs: {}, 
            data=data_source,
            evaluators=EVALUATORS,
            experiment_prefix="bluey-eval-5-metrics-gpt-5",
            #experiment="bluey-eval-5-metriche-1-4d0853fc",
            max_concurrency=3,
            num_repetitions=3           
        )
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED")
        
        # Statistics
        try:
            results_df = experiment_results.to_pandas()
            n_examples = len(results_df)
            print(f"\n[STATS] Statistics on {n_examples} conversations (scale 1-5):")
            print("-" * 60)
            
            for metric in METRIC_NAMES:
                # Look for columns containing metric name
                cols = [c for c in results_df.columns if metric in c.lower()]
                for col in cols:
                    if results_df[col].dtype in ['int64', 'float64']:
                        scores = results_df[col].dropna()
                        if len(scores) > 0:
                            mean_val = scores.mean()
                            std_val = scores.std()
                            min_val = scores.min()
                            max_val = scores.max()
                            
                            print(f"  {metric}: {mean_val:.2f} avg ± {std_val:.2f} [min: {min_val:.1f}, max: {max_val:.1f}]")
                        break  # One column per metric
                        
        except Exception as e:
            print(f"\n[INFO] Statistics available on LangSmith (local error: {e})")
            
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()