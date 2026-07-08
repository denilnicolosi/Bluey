from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from typing import cast
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import GraphInterrupt
from langchain_core.tools import tool as tool_decorator
from prompts import *
from tools import *
import os
from middleware import dynamic_prompt_middleware

# Import for the SQL subgraph
from sql_agent import create_sql_subgraph


def _init_chat_model(model_name: str | None, **kwargs) -> BaseChatModel:
    # reasoning_effort va passato dentro model_kwargs, altrimenti LangChain genera un warning
    kwargs.setdefault("model_kwargs", {})["reasoning_effort"] = kwargs.pop("reasoning_effort", "low")
    return cast(BaseChatModel, init_chat_model(model_name, **kwargs))

class Agents:
    """
    Class for managing all specialized agents in the Bluey system.

    Implements the supervisor-worker pattern where a supervisor agent
    coordinates specialized agents (email, appointments, documentation, SQL)
    through the LangGraph framework.
    """
    debug: bool = False    
    
    def __init__(self, checkpointer=None):
        """
        Initialize the Bluey multi-agent system.

        Args:
            checkpointer: Checkpointer instance for state persistence.
                        If None, no persistence is used.
        """
        self.checkpointer = checkpointer
                
        # Initialize the SQL agent as a subgraph
        self.sql_agent = create_sql_subgraph(_init_chat_model(os.getenv('SQL_AGENT_MODEL'), temperature=0))
        
        # Initialize the other agents
        self.email_agent = create_agent(
            model=_init_chat_model(os.getenv('EMAIL_AGENT_MODEL'), temperature=0),
            tools=[send_email],
            system_prompt=EMAIL_AGENT_SYSTEM_PROMPT,
            name="email_agent",
            middleware=[ 
                HumanInTheLoopMiddleware( 
                    interrupt_on={"send_email": True}, 
                    description_prefix="Email sending in progress", 
                ), 
            ]
        )

        self.appointment_agent = create_agent(
            model=_init_chat_model(os.getenv('APPOINTMENT_AGENT_MODEL'), temperature=0),
            tools=[get_available_time_slots, create_calendar_event],
            middleware=[ 
                HumanInTheLoopMiddleware( 
                    interrupt_on={"create_calendar_event": True}, 
                    description_prefix="Appointment creation in progress", 
                ), 
                dynamic_prompt_middleware,
            ],
            system_prompt=APPOINTMENT_AGENT_SYSTEM_PROMPT,
            name="appointment_agent"
        )

        self.documentation_agent = create_agent(
            model=_init_chat_model(os.getenv('DOCUMENTATION_AGENT_MODEL'), temperature=0, reasoning_effort="minimal"),
            tools=[search_documentation],
            system_prompt=DOCUMENTATION_AGENT_SYSTEM_PROMPT,
            name="documentation_agent"
        )               

        # Wrap sub-agents as LangChain tools so the supervisor can call them explicitly
        self.subagent_tools = [
            self.wrap_agent_as_tool(
                agent=self.email_agent,
                name="call_email_agent",
                description=(
                    """Handles end-to-end email sending.
                    Args: 
                        'request': natural-language instruction specifying recipients, content, and goal of the email to send."""
                ),
            ),
            self.wrap_agent_as_tool(
                agent=self.appointment_agent,
                name="call_appointment_agent",
                description=(
                    """Handles calendar appointment booking and checks availability.
                    Args: 
                        'request': natural-language instruction with date, time, participants, and any planning details needed."""
                ),
            ),
            self.wrap_agent_as_tool(
                agent=self.documentation_agent,
                name="call_documentation_agent",
                description=(
                    """Searches for information in company documentation.
                    Args: 
                        'request': natural-language question clearly describing which information to retrieve or summarize."""
                ),
            ),
            self.wrap_agent_as_tool(
                agent=self.sql_agent,
                name="call_sql_agent",
                description=(
                    """Runs queries on the insurance portal database to retrieve information about customers, policies, and claims.
                    Args: 
                        'request': natural-language instruction describing data to extract or analysis to perform."""
                ),
            ),
        ]
        
    
        # Create the supervisor
        self.supervisor = create_agent(          
            model=_init_chat_model(os.getenv('SUPERVISOR_AGENT_MODEL'), temperature=0, reasoning_effort="medium"),
            tools=[
                *self.subagent_tools,
                export_to_excel                
            ],            
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            name="supervisor",
            checkpointer=self.checkpointer  # Enables state persistence for multi-turn conversations
        )        

    def wrap_agent_as_tool(
        self, 
        *, 
        agent: CompiledStateGraph, 
        name: str, 
        description: str
    ) -> BaseTool:
        """
        Allows the supervisor to invoke specialized agents as if they were standard tools, passing natural-language requests.

        Args:
            agent: Compiled agent to invoke.
            name: Unique tool name (e.g., 'call_email_agent').
            description: Tool description for the supervisor.

        Returns:
            BaseTool: LangChain tool that wraps the agent.
        """
        
        @tool_decorator(name, description=description)
        def agent_tool(request: str, config: RunnableConfig) -> str:
            """Routes a natural-language request to the corresponding sub-agent.

            The supervisor must include in the 'request' parameter all details
            required to complete the task (e.g., names, dates, email addresses, etc.).
            The sub-agent has no access to conversation history.
            """
            try:
                response = agent.invoke(
                    {"messages": [HumanMessage(content=request)]},
                    config=config
                )
                if response and "messages" in response and len(response["messages"]) > 0:
                    last_msg = response["messages"][-1]
                    if hasattr(last_msg, "content"):
                        return last_msg.content
                    elif hasattr(last_msg, "text"):
                        return last_msg.text
                    return str(last_msg)
                return "Error: Sub-agent returned an empty response."
            except GraphInterrupt:
                raise  # Let HITL interrupts propagate to the supervisor
            except Exception as e:
                return f"Error invoking sub-agent: {str(e)}"

        return agent_tool

# Export agents
agents = Agents()
email_agent = agents.email_agent
calendar_agent = agents.appointment_agent
doc_agent = agents.documentation_agent
sql_agent = agents.sql_agent
supervisor = agents.supervisor
