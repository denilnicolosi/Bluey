"""
Unit tests for agents in agents.py

Uses importlib.reload to handle the module-level Agents() instantiation.
Patches are applied at the source level (langchain.chat_models, langchain.agents,
sql_agent) so they're active when the module re-executes during reload.
"""
import pytest
import importlib
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage


class TestAgents:
    """Test the Agents class and its agent orchestration."""

    @pytest.fixture
    def agents_env(self):
        """Import the agents module with all LLM/API dependencies mocked.

        Yields (module, mocks_dict) so tests can inspect mock call counts, etc.
        """
        with patch('langchain.chat_models.init_chat_model') as mock_init_chat, \
             patch('langchain.agents.create_agent') as mock_create_agent, \
             patch('sql_agent.create_sql_subgraph') as mock_sql_subgraph:

            mock_model = MagicMock()
            mock_model.invoke.return_value = AIMessage(content="Mock response")
            mock_init_chat.return_value = mock_model

            mock_agent_graph = MagicMock()
            mock_create_agent.return_value = mock_agent_graph

            mock_sql_graph = MagicMock()
            mock_sql_subgraph.return_value = mock_sql_graph

            import agents as agents_mod
            importlib.reload(agents_mod)

            yield agents_mod, {
                'init_chat_model': mock_init_chat,
                'create_agent': mock_create_agent,
                'create_sql_subgraph': mock_sql_subgraph,
                'model': mock_model,
            }

    # ------------------------------------------------------------------
    # Initialization tests
    # ------------------------------------------------------------------

    def test_all_agents_are_created(self, agents_env):
        """Test that Agents() creates all five agent attributes."""
        mod, _ = agents_env
        instance = mod.Agents()

        assert instance.sql_agent is not None
        assert instance.email_agent is not None
        assert instance.appointment_agent is not None
        assert instance.documentation_agent is not None
        assert instance.supervisor is not None

    def test_sql_subgraph_is_called(self, agents_env):
        """Test that create_sql_subgraph is invoked during init."""
        mod, mocks = agents_env
        mocks['create_sql_subgraph'].reset_mock()
        mod.Agents()
        mocks['create_sql_subgraph'].assert_called_once()

    def test_create_agent_called_four_times(self, agents_env):
        """Test that create_agent is called 4 times (email + appointment + docs + supervisor)."""
        mod, mocks = agents_env
        mocks['create_agent'].reset_mock()
        mod.Agents()
        assert mocks['create_agent'].call_count == 4

    def test_init_chat_model_called_for_each_agent(self, agents_env):
        """Test that init_chat_model is called 5 times (one per agent type)."""
        mod, mocks = agents_env
        mocks['init_chat_model'].reset_mock()
        mod.Agents()
        # SQL + email + appointment + documentation + supervisor = 5
        assert mocks['init_chat_model'].call_count == 5

    def test_checkpointer_passed_to_supervisor(self, agents_env):
        """Test that the checkpointer parameter is forwarded to the supervisor agent."""
        mod, mocks = agents_env
        mocks['create_agent'].reset_mock()

        sentinel_checkpointer = object()
        mod.Agents(checkpointer=sentinel_checkpointer)

        # The supervisor is the last create_agent call
        supervisor_call = mocks['create_agent'].call_args_list[-1]
        assert supervisor_call.kwargs.get('checkpointer') is sentinel_checkpointer

    # ------------------------------------------------------------------
    # wrap_agent_as_tool tests
    # ------------------------------------------------------------------

    def test_wrap_agent_as_tool_returns_named_tool(self, agents_env):
        """Test that wrap_agent_as_tool creates a tool with the correct name and description."""
        mod, _ = agents_env
        instance = mod.Agents()

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="reply")]}

        tool = instance.wrap_agent_as_tool(
            agent=mock_agent,
            name="test_tool",
            description="A test tool description"
        )

        assert tool.name == "test_tool"
        assert "test tool description" in tool.description.lower()

    def test_wrap_agent_as_tool_delegates_to_agent(self, agents_env):
        """Test that invoking the wrapped tool delegates to the underlying agent."""
        mod, _ = agents_env
        instance = mod.Agents()

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Agent reply")]
        }

        tool = instance.wrap_agent_as_tool(
            agent=mock_agent,
            name="delegate_test",
            description="Test delegation"
        )

        result = tool.invoke({"request": "Do something"})

        # The agent should have been called with a HumanMessage
        mock_agent.invoke.assert_called_once()
        call_args = mock_agent.invoke.call_args[0][0]
        assert "messages" in call_args
        assert isinstance(call_args["messages"][0], HumanMessage)
        assert call_args["messages"][0].content == "Do something"

    def test_wrap_agent_as_tool_propagates_graph_interrupt(self, agents_env):
        """Test that GraphInterrupt is propagated and NOT swallowed."""
        from langgraph.errors import GraphInterrupt
        mod, _ = agents_env
        instance = mod.Agents()

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = GraphInterrupt()

        tool = instance.wrap_agent_as_tool(
            agent=mock_agent,
            name="interrupt_test",
            description="Test interrupt"
        )

        with pytest.raises(GraphInterrupt):
            tool.invoke({"request": "Do something"})

    def test_wrap_agent_as_tool_catches_generic_exceptions(self, agents_env):
        """Test that generic exceptions are caught and returned as error string."""
        mod, _ = agents_env
        instance = mod.Agents()

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = ValueError("Some validation error")

        tool = instance.wrap_agent_as_tool(
            agent=mock_agent,
            name="error_test",
            description="Test error handling"
        )

        result = tool.invoke({"request": "Do something"})
        assert "Error invoking sub-agent: Some validation error" in result

    def test_wrap_agent_as_tool_passes_config_through(self, agents_env):
        """Test that the tool wrapper passes config through to the sub-agent."""
        mod, _ = agents_env
        instance = mod.Agents()

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [HumanMessage(content="Sub-agent reply")]}

        tool = instance.wrap_agent_as_tool(
            agent=mock_agent,
            name="test_tool",
            description="Test config passthrough"
        )

        config = {"configurable": {"thread_id": "thread-123", "checkpoint_ns": "parent_ns"}}
        tool.invoke({"request": "Do something"}, config=config)
        
        mock_agent.invoke.assert_called_once()
        passed_config = mock_agent.invoke.call_args[1]["config"]
        assert passed_config["configurable"]["thread_id"] == "thread-123"
        assert passed_config["configurable"]["checkpoint_ns"] == "parent_ns"

    # ------------------------------------------------------------------
    # Subagent tools list
    # ------------------------------------------------------------------

    def test_subagent_tools_count(self, agents_env):
        """Test that subagent_tools contains exactly 4 tools."""
        mod, _ = agents_env
        instance = mod.Agents()
        assert len(instance.subagent_tools) == 4

    def test_subagent_tool_names(self, agents_env):
        """Test that the 4 subagent tools have the expected names."""
        mod, _ = agents_env
        instance = mod.Agents()
        names = {t.name for t in instance.subagent_tools}
        expected = {
            "call_email_agent",
            "call_appointment_agent",
            "call_documentation_agent",
            "call_sql_agent",
        }
        assert names == expected