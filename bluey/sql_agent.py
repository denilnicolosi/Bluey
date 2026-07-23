from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langsmith import uuid7
import json
import os
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateTable
from sqlalchemy import MetaData, Table
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.types import interrupt
from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    HITLRequest,
    ReviewConfig,
)
from pydantic import BaseModel, Field
from tracing import log_hitl_decision
from prompts import (
    SQL_GENERATE_QUERY_SYSTEM_PROMPT, 
    SQL_CHECK_QUERY_SYSTEM_PROMPT,
    SQL_EVALUATE_RESULT_SYSTEM_PROMPT
)

RETRY_QUERY_MAX = 5
_sql_engine_cache = None

# Base path relative to this file location (resolved to absolute)
_BASE_DIR = Path(__file__).parent.resolve()

def get_or_create_sql_engine() -> Engine:
    """
    Return an SQL engine instance (singleton pattern).

    Uses the DATABASE_PATH environment variable for the database path.
    If not specified, uses 'data/insurance.db' relative to this file's directory.

    Returns:
        Engine: SQLAlchemy engine instance.
    """
    global _sql_engine_cache
    
    if _sql_engine_cache is None:
        # Use absolute path based on file location
        default_db_path = _BASE_DIR / "data" / "insurance.db"
        db_path = os.getenv('DATABASE_PATH', str(default_db_path))
        # Ensure absolute path with posix slashes for SQLite URI format
        abs_db_path = Path(db_path).resolve().as_posix()    
        _sql_engine_cache = create_engine(
            "sqlite://",
            creator=lambda: sqlite3.connect(f"file:{abs_db_path}?mode=ro", uri=True)
        )
    
    return _sql_engine_cache



def create_sql_subgraph(model: BaseChatModel) -> CompiledStateGraph:
    """
    Create the SQL agent subgraph following LangGraph patterns.

    This subgraph implements a safe flow for SQL query execution:
    1. List available tables
    2. Retrieve schema for relevant tables
    3. Generate SQL query
    4. Validate query (read-only)
    5. Execute query with human-in-the-loop approval

    Args:
        model: LLM model to use for query generation.

    Returns:
        CompiledStateGraph: Compiled graph ready for execution.
    """

    engine = get_or_create_sql_engine()

    @tool
    def sql_db_list_tables() -> str:
        """Input is an empty string. Output is a comma-separated list of tables in the database."""
        inspector = inspect(engine)
        return ", ".join(inspector.get_table_names())

    class SchemaInput(BaseModel):
        table_names: str = Field(description="A comma-separated list of table names to get the schema for.")

    @tool(args_schema=SchemaInput)
    def sql_db_schema(table_names: str) -> str:
        """Get the schema (DDL statements) and a few sample rows for a comma-separated list of table names."""
        inspector = inspect(engine)
        all_tables = set(inspector.get_table_names())
        requested_tables = [t.strip() for t in table_names.split(",") if t.strip()]
        
        results = []
        metadata = MetaData()
        for table_name in requested_tables:
            if table_name not in all_tables:
                results.append(f"Table '{table_name}' not found.")
                continue
            try:
                table = Table(table_name, metadata, autoload_with=engine)
                ddl = str(CreateTable(table).compile(engine))
                
                with engine.connect() as conn:
                    sample_rows_query = text(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_rows_res = conn.execute(sample_rows_query)
                    cols = sample_rows_res.keys()
                    rows = sample_rows_res.fetchall()
                    
                    sample_str = ""
                    if rows:
                        sample_str += "\n/* Three sample rows: */\n"
                        sample_str += "\t".join(cols) + "\n"
                        for row in rows:
                            sample_str += "\t".join(str(val) for val in row) + "\n"
                    
                results.append(f"{ddl.strip()}{sample_str}")
            except Exception as e:
                results.append(f"Error getting schema for table '{table_name}': {str(e)}")
        return "\n\n".join(results)

    class QueryInput(BaseModel):
        query: str = Field(description="The SELECT SQL query to execute.")

    @tool(args_schema=QueryInput)
    def sql_db_query(query: str) -> str:
        """Execute a read-only SQL query on the database and return the results."""
        try:
            with engine.connect() as conn:
                res = conn.execute(text(query))
                cols = res.keys()
                rows = res.fetchall()
                if not rows:
                    return "Query returned no results."
                header = " | ".join(cols)
                divider = "-" * len(header)
                lines = [header, divider]
                for row in rows:
                    lines.append(" | ".join(str(val) for val in row))
                return "\n".join(lines)
        except Exception as e:
            return f"Error executing query: {str(e)}"

    get_schema_tool = sql_db_schema
    get_schema_node = ToolNode([get_schema_tool], name="get_schema")

    run_query_tool = sql_db_query

    
    # Custom state definition for retry counting
    class SQLAgentState(MessagesState):
        retry_count: int

 

    class GenerateQueryOutput(BaseModel):
        query: str = Field(description="SQL query to execute on the database.")
        explanation: str = Field(
            description="Explanation of why this query was generated.",
            max_length=500
        )

    
    @tool(
        name_or_callable=run_query_tool.name,
        description=run_query_tool.description,
        args_schema=run_query_tool.args_schema
    )
    def run_query_tool_with_interrupt(config: RunnableConfig, **tool_input):        
        action_request: ActionRequest = {
            "name": run_query_tool.name,
            "args": tool_input,
            "description": (
                "Approve, edit, or reject the following SQL query before execution"
            ),
        }
        review_config: ReviewConfig = {
            "action_name": run_query_tool.name,
            "allowed_decisions": ["approve", "edit", "reject"],
        }
        hitl_request: HITLRequest = {
            "action_requests": [action_request],
            "review_configs": [review_config],
        }

        response = interrupt(hitl_request)
        decisions = response.get("decisions") or []
        if decisions:
            decision = decisions[0]
            decision_type = decision.get("type")
            log_hitl_decision(
                agent="sql_agent",
                tool_name=run_query_tool.name,
                decision_type=decision_type or "unknown",
                comment=str(tool_input.get("query", ""))[:500],
            )
            if decision_type == "approve":
                return run_query_tool.invoke(tool_input, config)
            if decision_type == "edit":
                edited_action = decision.get("edited_action") or {}
                edited_args = edited_action.get("args") or {}
                tool_input = {**tool_input, **edited_args}
                return run_query_tool.invoke(tool_input, config)
            if decision_type == "reject":
                message = decision.get("message") or "Tool call rejected by reviewer."
                return {"error": message}
        return response

    run_query_node = ToolNode([run_query_tool_with_interrupt], name="run_query")
    list_tables_tool = sql_db_list_tables


    # Node: list_tables - Force table listing at the beginning
    def list_tables(state: SQLAgentState):
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": str(uuid7()),
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables to get the schema: {tool_message.content} choose the relevant ones for the request.")

        # Initialize retry_count to 0
        return {"messages": [tool_call_message, tool_message, response], "retry_count": 0}

    # Node: call_get_schema - Force schema retrieval for relevant tables
    def call_get_schema(state: SQLAgentState):
        llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate_query(state: SQLAgentState):
        system_message = {
            "role": "system",
            "content": SQL_GENERATE_QUERY_SYSTEM_PROMPT.format(
                dialect=engine.dialect.name,
                top_k=20,
            )
        }
        # Use structured output instead of tool binding
        structured_llm = model.with_structured_output(GenerateQueryOutput)
        # Invoke model
        try:
            generated_obj = structured_llm.invoke([system_message] + state["messages"])
            query = generated_obj.query
            explanation = generated_obj.explanation
            
            content = json.dumps({"query": query, "explanation": explanation})
            response = AIMessage(content=content)
            
        except Exception as e:
             response = AIMessage(content=f"Error generating query: {str(e)}")

        return {"messages": [response]}

    def check_query(state: SQLAgentState):
        """Validate query correctness and block data-modification operations."""
        last_message = state["messages"][-1]
        current_retries = state.get("retry_count", 0)
        
        query = ""
        # Try extracting query from content (JSON)
        try:
            data = json.loads(last_message.content)
            query = data.get("query", "")
        except:
             # Fallback or error
             pass
             
        if not query:
             return {
                "messages": [AIMessage(content="Error: No generated query to validate.")],
                "retry_count": current_retries + 1
            }
        
        # If query is valid, proceed with LLM-based validation
        system_message = {
            "role": "system",
            "content": SQL_CHECK_QUERY_SYSTEM_PROMPT.format(dialect=engine.dialect.name),
        }
        
        messages = [system_message] + state["messages"]

        llm_with_tools = model.bind_tools([run_query_tool_with_interrupt])
        response = llm_with_tools.invoke(messages)
        
        # If checker responds with text (rejection) instead of tool call, increment retry_count
        additional_updates = {}
        if not response.tool_calls:
             additional_updates["retry_count"] = current_retries + 1

        return {"messages": [response], **additional_updates}

    def evaluate_result(state: SQLAgentState):
        """Evaluate executed query result."""
        system_message = {
            "role": "system",
            "content": SQL_EVALUATE_RESULT_SYSTEM_PROMPT.format(dialect=engine.dialect.name),
        }
        response = model.invoke([system_message] + state["messages"])
        
        # If result requires RETRY, increment counter
        content = response.content if isinstance(response.content, str) else ""
        if "RETRY" in content or "ERROR" in content:
            return {
                "messages": [response],
                "retry_count": state.get("retry_count", 0) + 1
            }
        
        return {"messages": [response]}

    # Removed should_continue because flow is linear: generate -> check

    def should_run_query(state: SQLAgentState):
        """
        Decide whether to execute query or go back to generation.
        If latest message has a tool_call (run_query), proceed.
        Otherwise (error/text message), go back to generation if limit not exceeded.
        """
        last_message = state["messages"][-1]
        
        # If there is a tool call, this is a success -> run_query
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "run_query"
        
        # If there is no tool call, this is a failure/rejection.
        # Check retry limit
        if state.get("retry_count", 0) >= RETRY_QUERY_MAX:
            return END
        else:
            return "generate_query"
            
    def should_finish_after_eval(state: SQLAgentState):
        """
        Decide whether to finish or retry after evaluation.
        If latest message contains 'RETRY' or 'ERROR', return to generation (if below limit).
        Otherwise, finish.
        """
        last_message = state["messages"][-1]
        content = last_message.content if isinstance(last_message.content, str) else ""
        
        if "RETRY" in content or "ERROR" in content:
            # retry_count is already incremented in evaluate_result
            if state.get("retry_count", 0) >= RETRY_QUERY_MAX:
                return END
            return "generate_query"
            
        return END

    # Graph construction
    builder = StateGraph(SQLAgentState)
    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node("get_schema", get_schema_node)
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node("run_query", run_query_node)
    builder.add_node(evaluate_result)

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_edge("generate_query", "check_query")
    builder.add_edge("run_query", "evaluate_result")
    
    builder.add_conditional_edges(
        "check_query",
        should_run_query,
        ["run_query", "generate_query", END]
    )    
    builder.add_conditional_edges(
        "evaluate_result",
        should_finish_after_eval,
        ["generate_query", END]
    )

    return builder.compile(name="sql_agent")