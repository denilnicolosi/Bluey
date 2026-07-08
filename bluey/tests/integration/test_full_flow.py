"""
Integration tests for the Bluey system.

Smoke tests that verify the system initializes correctly and that
module-level exports are consistent. All LLM providers are mocked.
"""
import pytest
import importlib
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage


@pytest.mark.integration
class TestSystemInitialization:
    """Smoke tests for full system initialization."""

    @pytest.fixture(autouse=True)
    def _mock_llm_providers(self):
        """Mock all LLM/API providers so no real calls are made."""
        with patch('langchain.chat_models.init_chat_model') as mock_init, \
             patch('langchain.agents.create_agent') as mock_create, \
             patch('sql_agent.create_sql_subgraph') as mock_sql:

            mock_model = MagicMock()
            mock_model.invoke.return_value = AIMessage(content="Mock")
            mock_init.return_value = mock_model
            mock_create.return_value = MagicMock()
            mock_sql.return_value = MagicMock()

            yield {
                'init_chat_model': mock_init,
                'create_agent': mock_create,
                'create_sql_subgraph': mock_sql,
            }

    def _reload_agents(self):
        """Helper: reload the agents module so module-level code re-runs under mocks."""
        import agents as agents_mod
        importlib.reload(agents_mod)
        return agents_mod

    def test_module_imports_successfully(self):
        """Test that the agents module can be imported and all exports are set."""
        mod = self._reload_agents()

        assert mod.supervisor is not None
        assert mod.email_agent is not None
        assert mod.calendar_agent is not None
        assert mod.doc_agent is not None
        assert mod.sql_agent is not None

    def test_module_exports_match_instance(self):
        """Test that module-level exports point to the same objects as the Agents instance."""
        mod = self._reload_agents()

        instance = mod.agents  # The module-level Agents() instance
        assert mod.supervisor is instance.supervisor
        assert mod.email_agent is instance.email_agent
        assert mod.calendar_agent is instance.appointment_agent
        assert mod.doc_agent is instance.documentation_agent
        assert mod.sql_agent is instance.sql_agent

    def test_agents_class_is_accessible(self):
        """Test that the Agents class can be imported and instantiated."""
        mod = self._reload_agents()

        new_instance = mod.Agents()
        assert new_instance.supervisor is not None
        assert new_instance.subagent_tools is not None
        assert len(new_instance.subagent_tools) == 4