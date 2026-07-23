# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `bluey/` (not the repo root).

```bash
cd bluey
pip install -r requirements.txt   # requirements_frozen.txt = exact pinned versions
langgraph dev                     # dev server on :2024, serves the graphs in langgraph.json
pytest                            # all tests (mocked LLMs, no API keys needed)
pytest -m unit                    # or -m integration
pytest tests/unit/test_tools.py::TestExportToExcel::test_export_dict -v
```

Evaluation (needs real API keys + LangSmith; scripts `os.chdir` to `bluey/` themselves, so they can be run from anywhere):

```bash
python evaluate/generate_threads_dataset.py   # run conversations, push to LangSmith dataset
python evaluate/evaluate_dataset.py           # LLM-as-judge scoring, 5 metrics
```

Regenerate synthetic DB: `python data/generate_db.py` (needs `faker`).
Google OAuth bootstrap: `python google_credentials/bootstrap_oauth.py` (writes `token.json`).

## Working directory & imports

`bluey/` is the import root. Modules import each other **flat** — `from prompts import *`, `import sql_agent` — never `from bluey.tools import ...`. Anything that imports `agents` must have `bluey/` on `sys.path` and, for relative paths in `.env`, `bluey/` as cwd. pytest gets this from `tests/__init__.py` + `pytest.ini` rootdir; the `evaluate/` scripts do it manually.

## Architecture

Supervisor–worker. `agents.py` builds four workers and wraps each as a LangChain tool (`wrap_agent_as_tool`) so the supervisor calls them like any other tool with a natural-language `request` string.

- **email / appointment / documentation agents** — `create_agent` (prebuilt ReAct), each with its own model from env, its own tools, its own prompt.
- **SQL agent** — hand-built `StateGraph` in `sql_agent.py`, not `create_agent`, because the flow must be forced: `list_tables → call_get_schema → get_schema → generate_query → check_query → run_query → evaluate_result`. Two conditional edges loop back to `generate_query`; `retry_count` in state, capped at `RETRY_QUERY_MAX = 5`.
- **Supervisor** — the only agent holding a checkpointer, and thus the only one with conversation history.

### Invariants worth not breaking

- **Sub-agents are stateless by design.** They receive only the `request` string the supervisor composes; no history is passed down. This is the core cost/latency decision of the thesis (see README "Design Decisions"), not an oversight.
- **The SQLite engine is read-only** — `sql_agent.get_or_create_sql_engine()` opens `file:...?mode=ro` through a custom `creator`. Do not "simplify" it to `create_engine("sqlite:///path")`; that silently removes the last defense against an LLM-generated DML statement.
- **`agents.py` instantiates `Agents()` at module scope** (bottom of file). Importing the module builds every LLM client and reads every model env var. Tests therefore use `importlib.reload(agents_mod)` inside `patch(...)` blocks, and patch **source** modules (`langchain.chat_models.init_chat_model`, `langchain.agents.create_agent`, `sql_agent.create_sql_subgraph`) — patching `agents.init_chat_model` does not survive the reload.
- **Global singletons:** `sql_agent._sql_engine_cache`, `tools._vectorstore_cache`. `tests/conftest.py` has an autouse fixture that nulls both before and after every test; a new cache needs adding there too.

### Human-in-the-loop — two different mechanisms

1. Worker agents: `HumanInTheLoopMiddleware(interrupt_on={"tool_name": True})` in `agents.py` (`send_email`, `create_calendar_event`).
2. SQL subgraph: manual `run_query_tool_with_interrupt` in `sql_agent.py` — wraps `sql_db_query`, raises `interrupt()` with a hand-built `HITLRequest`, and handles `approve` / `edit` / `reject` decisions itself. The middleware isn't available here because this is a raw `StateGraph`.

`wrap_agent_as_tool` catches all exceptions and returns them as strings **except** `GraphInterrupt`, which is re-raised so HITL pauses propagate to the supervisor. Keep that carve-out.

## Prompts

All in `prompts.py`, CO-STAR structure. Two independent brace conventions:

- SQL prompts are `.format(dialect=..., top_k=...)`ed at call time — literal JSON braces inside them are escaped as `{{ }}`.
- `{current_time}` / `{timezone}` are filled by `dynamic_prompt_middleware` (`middleware.py`). Only `APPOINTMENT_AGENT_SYSTEM_PROMPT` uses them, and only the appointment agent has that middleware attached. Adding the placeholder to another prompt without adding the middleware leaves it unsubstituted.

## Config

`bluey/.env` (template in `.env.example`). Per-agent model IDs (`SUPERVISOR_AGENT_MODEL`, `SQL_AGENT_MODEL`, …) use LangChain's `provider:model` form, so switching an agent to Ollama or Gemini is an env change, not a code change. Paths (`DATABASE_PATH`, `DOCS_PATH`, `OUTPUT_DIR`, `GOOGLE_CREDENTIALS_DIR`) are relative to `bluey/`.

Excel exports are served by `file_server.py` — a daemon `ThreadingHTTPServer` on `FILE_SERVER_PORT` (8765) lazily started by `get_output_dir()` / `get_download_url()`; a port collision is swallowed on the assumption the server is already up.
