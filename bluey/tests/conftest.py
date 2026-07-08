"""Test configuration for Bluey LangGraph project.

Provides shared fixtures for unit and integration tests.
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from unittest.mock import MagicMock


@pytest.fixture
def sample_human_message():
    """Sample human message for testing."""
    return HumanMessage(content="Test query")


@pytest.fixture
def sample_ai_message():
    """Sample AI message for testing."""
    return AIMessage(content="Test response")


@pytest.fixture
def mock_model():
    """Mock LLM model for testing.

    Provides a MagicMock with .invoke(), .bind_tools(), and
    .with_structured_output() pre-configured so it can stand in
    for any BaseChatModel in unit tests.
    """
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="Mocked response")
    mock.bind_tools = MagicMock(return_value=mock)
    mock.with_structured_output = MagicMock(return_value=mock)
    return mock


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset global caches between tests to prevent cross-test pollution."""
    import sql_agent
    import tools
    sql_agent._sql_engine_cache = None
    tools._vectorstore_cache = None
    yield
    sql_agent._sql_engine_cache = None
    tools._vectorstore_cache = None


@pytest.fixture(autouse=True)
def _set_test_env_vars(monkeypatch):
    """Set minimal environment variables needed by the modules under test."""
    monkeypatch.setenv("DATABASE_PATH", ":memory:")
    monkeypatch.setenv("GOOGLE_CREDENTIALS_DIR", "./google_credentials")
    monkeypatch.setenv("GOOGLE_EMAIL_SENDER", "test@example.com")
    monkeypatch.setenv("GOOGLE_CALENDAR_ID", "primary")
    monkeypatch.setenv("GOOGLE_CALENDAR_TIMEZONE", "UTC")
    monkeypatch.setenv("DOCS_PATH", "./docs")
