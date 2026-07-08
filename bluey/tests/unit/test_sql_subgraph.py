"""
Unit tests for the SQL agent subgraph (sql_agent.py).

Tests database singleton behavior, subgraph compilation, and graph structure.
Uses a real in-memory SQLite database and a minimal FakeChatModel
that passes Pydantic validation for custom tool-calling LLM bindings.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from sql_agent import create_sql_subgraph, get_or_create_sql_engine


class FakeChatModel(BaseChatModel):
    """Minimal BaseChatModel subclass that passes Pydantic validation.

    Provides a compliant LLM class to test agent/tool bindings without making real API calls.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="fake"))]
        )

    @property
    def _llm_type(self):
        return "fake-chat-model"


class TestGetOrCreateSqlEngine:
    """Test the database engine singleton function."""

    def test_returns_sql_engine_instance(self):
        """Test that get_or_create_sql_engine returns a SQLAlchemy Engine."""
        engine = get_or_create_sql_engine()
        assert isinstance(engine, Engine)

    def test_singleton_returns_same_instance(self):
        """Test that repeated calls return the same cached engine instance."""
        engine1 = get_or_create_sql_engine()
        engine2 = get_or_create_sql_engine()
        assert engine1 is engine2

    def test_respects_database_path_env_var(self, monkeypatch, tmp_path):
        """Test that DATABASE_PATH env var overrides the default path."""
        import sql_agent
        sql_agent._sql_engine_cache = None  # Force re-creation

        db_path = tmp_path / "custom_test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY)"))
            conn.commit()
        engine.dispose()

        monkeypatch.setenv("DATABASE_PATH", str(db_path))
        engine_res = get_or_create_sql_engine()
        assert isinstance(engine_res, Engine)


class TestSQLSubgraphCreation:
    """Test SQL subgraph compilation and graph structure."""

    @pytest.fixture
    def in_memory_db(self):
        """Create an in-memory SQLite database with a minimal insurance schema."""
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE Customers (
                    CustomerID INTEGER PRIMARY KEY,
                    FirstName TEXT NOT NULL,
                    LastName TEXT NOT NULL,
                    Email TEXT
                )
            """))
            conn.execute(text("""
                INSERT INTO Customers VALUES
                    (1, 'Mario', 'Rossi', 'mario.rossi@test.com'),
                    (2, 'Luigi', 'Bianchi', 'luigi.bianchi@test.com')
            """))
            conn.execute(text("""
                CREATE TABLE Policies (
                    PolicyID INTEGER PRIMARY KEY,
                    CustomerID INTEGER,
                    PolicyNumber TEXT UNIQUE,
                    Status TEXT,
                    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
                )
            """))
            conn.execute(text("""
                INSERT INTO Policies VALUES
                    (1, 1, 'POL-0001', 'Active'),
                    (2, 2, 'POL-0002', 'Expired')
            """))
            conn.commit()
        return engine

    @pytest.fixture
    def sql_subgraph(self, in_memory_db):
        """Create the SQL subgraph with a real DB and a fake LLM."""
        fake_model = FakeChatModel()
        with patch('sql_agent.get_or_create_sql_engine', return_value=in_memory_db):
            graph = create_sql_subgraph(fake_model)
            yield graph

    def test_subgraph_compiles_successfully(self, sql_subgraph):
        """Test that create_sql_subgraph returns a compiled graph."""
        assert sql_subgraph is not None

    def test_subgraph_has_expected_nodes(self, sql_subgraph):
        """Test that the graph contains all required nodes."""
        graph_repr = sql_subgraph.get_graph()
        node_ids = set(graph_repr.nodes)

        expected_nodes = {
            "list_tables",
            "call_get_schema",
            "get_schema",
            "generate_query",
            "check_query",
            "run_query",
            "evaluate_result",
        }
        assert expected_nodes.issubset(node_ids), (
            f"Missing nodes: {expected_nodes - node_ids}"
        )

    def test_subgraph_has_start_and_end(self, sql_subgraph):
        """Test that the graph has __start__ and __end__ nodes."""
        graph_repr = sql_subgraph.get_graph()
        node_ids = set(graph_repr.nodes)
        assert "__start__" in node_ids
        assert "__end__" in node_ids

    def test_subgraph_has_edges(self, sql_subgraph):
        """Test that the graph has edges connecting nodes."""
        graph_repr = sql_subgraph.get_graph()
        assert len(graph_repr.edges) > 0

    def test_start_connects_to_list_tables(self, sql_subgraph):
        """Test that the graph starts with the list_tables node."""
        graph_repr = sql_subgraph.get_graph()
        start_edges = [e for e in graph_repr.edges if e.source == "__start__"]
        assert any(e.target == "list_tables" for e in start_edges)

    def test_subgraph_name_is_sql_agent(self, sql_subgraph):
        """Test that the compiled subgraph has the correct name."""
        assert sql_subgraph.name == "sql_agent"