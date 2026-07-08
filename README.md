# Bluey

**Bluey** is a multi-agent AI assistant built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain), designed to support back-office operators at an insurance company. It coordinates a team of specialized agents behind a single conversational interface where users interact with one assistant and the system intelligently routes tasks to the right agent under the hood.

---

## What It Does

Bluey acts as a unified back-office assistant capable of:

- **Querying an insurance database** : customers, policies, claims, payments, and more (via SQL agent)
- **Searching company documentation** : FAQs and operations manuals through semantic search (RAG with InMemoryVectorStore + OpenAI Embeddings)
- **Sending emails** : composing and sending professional emails via the Gmail API
- **Managing appointments** : checking calendar availability and creating events via Google Calendar API
- **Exporting data** : generating downloadable Excel files from query results

All actions that modify external state (sending emails, creating events, executing SQL queries) require **human-in-the-loop approval** before execution.

---

## Architecture

Bluey follows a **supervisor–worker** pattern using `create_agent` from `langchain.agents` and a custom LangGraph subgraph for the SQL agent:

```
                    ┌──────────────┐
   User ───────►    │  Supervisor  │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┬──────────────┐
            ▼              ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
     │   Email    │ │ Appointment│ │   Docs     │ │    SQL     │
     │   Agent    │ │   Agent    │ │   Agent    │ │   Agent    │
     └────────────┘ └────────────┘ └────────────┘ └────────────┘
```

- The **Supervisor** receives user messages, decides which specialist to invoke, and assembles a complete request with all necessary context (sub-agents have no access to conversation history).
- Each **worker agent** is created with `create_agent` and has its own tools, system prompt, LLM model (configurable via environment variables), and optional middleware.
- The **SQL Agent** is implemented as a custom LangGraph `StateGraph` subgraph with a multi-step pipeline: list tables → get schema → generate query → validate query → human approval → execute → evaluate result, with automatic retry logic (max 5 retries).
- **Human-in-the-loop** is enforced via `HumanInTheLoopMiddleware` on worker agents and `interrupt()` on the SQL subgraph.
- A custom `dynamic_prompt_middleware` (built with the `@dynamic_prompt` decorator from `langchain.agents.middleware`) injects runtime values (current date/time, timezone) into agent prompts.
- All prompts follow the **CO-STAR** structure (Context, Objective, Style, Tone, Audience, Response).

---

## Design Decisions

The architecture reflects a set of deliberate design choices, motivated and validated in the thesis work:

- **Supervisor pattern over alternatives** : Compared to peer-to-peer communication or fully decentralized multi-agent architectures, a hierarchical supervisor provides a single, natural conversational entry point: the user interacts with one assistant while internal iterations between agents stay completely hidden. Centralized coordination also makes the system's behavior more predictable and easier to debug, since the supervisor keeps the global view of the conversation.
- **Tool partitioning through specialization** : A single agent with access to every tool would have to choose among similar options, master each tool's exact format, and juggle multiple domains at once. Splitting the problem assigns intent understanding and routing to the supervisor, while each specialist masters a small, focused toolset within a well-defined domain. This reduced per-model complexity is what allows lighter, cheaper models to perform effectively.
- **Stateless sub-agents and context engineering** : Sub-agents never see the conversation history. The supervisor composes a self-contained request (task, context, known data, expected output) for each delegation. This cuts the tokens processed by each agent, lowering cost and latency, and guarantees that every agent only receives information relevant to its task.
- **Per-agent model choice (model agnosticism)** : Each agent's LLM is configured via environment variables: lightweight, fast models for narrow tasks and a stronger reasoning model for the supervisor. Privacy-sensitive agents can be pointed at local or self-hosted models. In the thesis evaluation, the SQL agent (which handles customer data) ran on Qwen3-coder via Ollama, avoiding the transmission of sensitive data to external cloud services.
- **Control vs. autonomy, per agent** : The design balances deterministic flow control against LLM autonomy at the level of each agent, prebuilt ReAct agents (`create_agent`) where autonomy is acceptable (email, calendar, documentation), and a custom `StateGraph` for SQL, where a controlled pipeline is required (forced schema discovery, double query validation, human approval, bounded retries). As defense in depth, the SQLite connection itself is opened in **read-only mode**, so no LLM-generated query can ever modify data.
- **Resilience through supervision** : The supervisor intercepts errors from specialized agents and applies recovery strategies, retrying with a reformulated request, asking the user for missing information, or delegating to a different agent. Retry limits (max 5 in the SQL subgraph) act as a circuit breaker against runaway loops.
- **Accepted trade-offs** : The supervisor is a single point of failure, each delegation adds an LLM round-trip (latency and cost), and trivial requests pay the full coordination overhead. These limitations are acceptable in a back-office context, where operators tolerate higher latency in exchange for response quality on complex, multi-step requests.

---

## Example Conversation

A single request that requires three specialists, transparently orchestrated (the user only ever sees the supervisor):

```
User: Customer Mario Rossi asked for a 30-minute appointment tomorrow at 3 PM
      to discuss his open claim. Book it and send him a confirmation email.

Supervisor ─► call_sql_agent("TASK: Retrieve email address and open claims
              for customer Mario Rossi. ...")
SQL Agent    [lists tables → reads schema → generates SELECT → validates]
             ⏸ operator approves query execution (HITL)
             ─► "mario.rossi@email.it - 1 open claim: SN001"

Supervisor ─► call_appointment_agent("TASK: Create 30-min event tomorrow 15:00
              with mario.rossi@email.it about claim SN001. ...")
Appointment  [checks availability]
Agent        ⏸ operator approves event creation (HITL)
             ─► "Event created, invitation sent"

Supervisor ─► call_email_agent("TASK: Confirmation email to mario.rossi@email.it
              for tomorrow 15:00 appointment about claim SN001. ...")
Email Agent  [composes a professional email]
             ⏸ operator approves sending (HITL)
             ─► "Email sent"

Supervisor ─► "Appointment booked for tomorrow at 15:00 and confirmation
              email sent to mario.rossi@email.it"
```

Note how the supervisor extracts the missing email address from the database on its own initiative, and how every state-changing action pauses for explicit operator approval.

---

## Project Structure

```
Bluey/
├── README.md
├── bluey/                        # Main application package
│   ├── agents.py                 # Agent orchestration (Supervisor + workers)
│   ├── sql_agent.py              # SQL agent subgraph (query generation, validation, execution)
│   ├── tools.py                  # LangChain tools (email, calendar, docs search, Excel export)
│   ├── prompts.py                # All system prompts (CO-STAR structured)
│   ├── middleware.py             # Custom middleware (dynamic_prompt_middleware for runtime prompt formatting)
│   ├── file_server.py            # Local HTTP file server for Excel downloads
│   ├── langgraph.json            # LangGraph deployment configuration
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # Environment variable template
│   ├── pytest.ini                # Test configuration
│   ├── data/
│   │   ├── insurance.db          # SQLite database (synthetic insurance data)
│   │   ├── generate_db.py        # Script to regenerate the database with Faker
│   │   ├── vectorstore.json      # Pre-generated vectorstore index
│   │   └── output/               # Directory for exported Excel files
│   ├── docs/
│   │   ├── FAQ.pdf               # Company FAQ document
│   │   └── Manuale.pdf           # Operations manual
│   ├── google_credentials/
│   │   └── bootstrap_oauth.py    # OAuth2 setup script for Gmail & Calendar APIs
│   ├── evaluate/
│   │   ├── evaluate_dataset.py           # LLM-as-judge evaluation (5 metrics)
│   │   ├── generate_threads_dataset.py   # Conversation dataset generator
│   │   ├── generate_thread_parallel_dataset.py  # Parallel dataset generator
│   │   └── evaluation_requests*.csv      # Test request sets
│   ├── tests/
│   │   ├── conftest.py           # Shared fixtures, singleton reset, env mocking
│   │   ├── unit/                 # Unit tests (agents, tools, SQL subgraph)
│   │   └── integration/         # Integration tests (full system flow)
│   └── .devcontainer/           # Dev container configuration
```

---

## Prerequisites

- **Python 3.13+**
- At least one LLM provider API key (OpenAI, Google Gemini, NVIDIA, or a local Ollama instance)
- Google Cloud credentials (for Gmail and Calendar features)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd Bluey/bluey
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `GEMINI_API_KEY` | Google Gemini API key (optional) |
| `NVIDIA_API_KEY` | NVIDIA API key (optional) |
| `SUPERVISOR_AGENT_MODEL` | Model for the supervisor (e.g., `openai:gpt-5-mini`) |
| `EMAIL_AGENT_MODEL` | Model for the email agent (e.g., `openai:gpt-5-nano`) |
| `APPOINTMENT_AGENT_MODEL` | Model for the appointment agent |
| `DOCUMENTATION_AGENT_MODEL` | Model for the documentation agent |
| `SQL_AGENT_MODEL` | Model for the SQL agent |
| `LANGSMITH_API_KEY` | LangSmith key for tracing (optional) |
| `LANGSMITH_ENDPOINT` | LangSmith endpoint URL |
| `LANGSMITH_PROJECT` | LangSmith project name |
| `DATABASE_PATH` | Path to the SQLite database |
| `DOCS_PATH` | Path to the documentation directory |
| `GOOGLE_CREDENTIALS_DIR` | Path to Google OAuth credentials directory |

### 3. Set up Google APIs (optional, for email & calendar)

To use the Gmail and Google Calendar features, you need to create OAuth2 credentials in the Google Cloud Console:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Search for and enable the **Gmail API** and the **Google Calendar API** in the API Library.
4. Set up the **OAuth consent screen**:
   - Choose **External** user type and fill in the required app information.
   - Add the scopes: `https://www.googleapis.com/auth/gmail.send` and `https://www.googleapis.com/auth/calendar`.
   - Add your own email address to the **Test users** list (crucial since the app is in "Testing" status).
5. Create credentials:
   - Go to **APIs & Services > Credentials**.
   - Click **+ Create Credentials** and select **OAuth client ID**.
   - Set the Application type to **Desktop app**.
   - Click **Create**, then download the generated JSON file.
6. Rename the downloaded file to `client_secret.json` and place it in the `bluey/google_credentials/` folder.

Once the file is in place, run the bootstrap script to authorize the app and generate `token.json`:

```bash
cd google_credentials
python bootstrap_oauth.py
```

Follow the browser OAuth2 flow to complete the authentication.

### 4. Generate the database (optional)

To regenerate the synthetic insurance database:

```bash
cd data
pip install faker
python generate_db.py
```

---

## Running

### With LangGraph CLI (recommended)

```bash
cd bluey
langgraph dev
```

This starts the LangGraph development server with the graphs defined in `langgraph.json`, exposing the Supervisor and all individual agents as API endpoints.

### Chat UI with Agent Chat

With the dev server running, you can talk to Bluey through the open-source [Agent Chat](https://agentchat.vercel.app/) frontend, no installation required:

1. Open [https://agentchat.vercel.app/](https://agentchat.vercel.app/)
2. Set **Deployment URL** to the local server address (default: `http://localhost:2024`)
3. Set **Assistant / Graph ID** to `bluey`

Agent Chat renders the conversation in real time, shows the supervisor's tool calls (i.e., the sub-agent invocations), and displays human-in-the-loop approval requests directly in chat, letting you **approve, edit, or reject** each critical action (SQL queries, emails, calendar events) before execution.

### Running tests

```bash
cd bluey
pytest
```

Unit tests run with mocked LLM providers and in-memory databases, no API keys required.

---

## Evaluation

The project includes an offline evaluation framework using **LLM-as-a-judge** with five metrics:

1. **Request Fulfilled** : Was the user's request completed?
2. **Groundedness** : Are responses grounded in facts (no hallucinations)?
3. **Ambiguity Handling** : Does the system ask for clarification when needed?
4. **Orchestration** : Were the right agents selected?
5. **Context Preservation** : Was context correctly transferred to sub-agents?

All metrics use a 1–5 numerical scale. Evaluation runs through [LangSmith](https://smith.langchain.com/) and [openevals](https://github.com/langchain-ai/openevals):

```bash
cd evaluate
python generate_threads_dataset.py   # Generate conversation dataset
python evaluate_dataset.py           # Run evaluation
```

The pipeline runs each request of a 30-example dataset (covering all agents, multi-agent collaboration, and deliberately ambiguous requests) against the live system, with an LLM-simulated user handling multi-turn interactions, automatic approval of HITL interrupts, and mocked Google APIs. Each conversation trace is then scored 3 times by the LLM judge to mitigate stochastic variability.

### Results

Two model configurations were benchmarked (results from the thesis evaluation):

| Metric | Economical models* | High-end models** |
|---|---|---|
| Request fulfilled | 4.61 | 4.74 |
| Groundedness | 4.81 | 4.86 |
| Ambiguity handling | 4.96 | 4.93 |
| Orchestration | 4.84 | 4.91 |
| Context preservation | 4.56 | 4.92 |
| **Overall average** | **4.75** | **4.87** |

Mean scores on a 1–5 scale. \*GPT-5 nano/mini + Qwen3-coder via Ollama (SQL agent) vs \*\*GPT-5 + GPT-5 Codex (SQL agent)

| Configuration | Avg. TTFT latency per turn | Avg. cost per turn |
|---|---|---|
| Economical models | 30 s | $0.005 |
| High-end models | 80 s (+167%) | $0.090 (+1700%) |

**Key finding:** a well-designed multi-agent architecture compensates for the limitations of weaker models through task specialization. The high-end configuration improves the average score by only **+0.12 points**, which does not justify ~18× higher costs and ~3× higher latency and the ability to run on lightweight models also enables deploying privacy-sensitive agents on local or self-hosted LLMs.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph, LangChain (`create_agent`) |
| LLM providers | OpenAI, Google Gemini, Ollama, NVIDIA and more |
| Database | SQLite + SQLAlchemy |
| Document search | InMemoryVectorStore + OpenAI Embeddings (RAG) |
| Email & Calendar | Google APIs (Gmail, Calendar) |
| Human-in-the-loop | `HumanInTheLoopMiddleware`, `interrupt()` |
| Evaluation | LangSmith, openevals |
| Testing | pytest |
