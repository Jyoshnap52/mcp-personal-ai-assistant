# MCP Personal AI Assistant
### Production Autonomous Agent with Dual MCP Protocol, Hybrid Retrieval (BM25 + pgvector), Human-in-the-Loop Safety, and Full Evaluation Telemetry

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Agent-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-FastMCP%202.3-purple.svg)](https://modelcontextprotocol.io/)
[![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Redis](https://img.shields.io/badge/Redis-Rate%20Limiting-DC382D.svg)](https://redis.io/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith%20Traced-00C49F.svg)](https://smith.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker%20Compose-Production-2496ED.svg)](https://www.docker.com/)

---

## Executive Summary

**MCP Personal AI Assistant** is an enterprise-grade autonomous agent built to demonstrate production LLM engineering: **multi-turn contextual memory**, **hybrid retrieval and semantic ranking**, **safe tool execution with Human-in-the-Loop (HITL) interrupts**, and **comprehensive evaluation telemetry** (both offline benchmarking and live LangSmith LLM-as-a-judge observability).

Instead of treating LLMs as naive prompt wrappers, this system implements an event-driven **LangGraph ReAct state machine** connected to two decoupled **Model Context Protocol (MCP)** servers over `stdio`, backed by PostgreSQL (`pgvector`) for long-term preference memory and Redis for sliding-window rate limiting.

---

## System Architecture

```text
                                 ┌──────────────────────────────────────────────┐
                                 │              Browser Client (UI)             │
                                 │       Cosmic Nebula Light/Dark Fusion        │
                                 └──────────────────────┬───────────────────────┘
                                                        │ HTTP / SSE Stream
                                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FastAPI Production Gateway                                     │
│  - HttpOnly Cookie Auth (PBKDF2-SHA256)      - Redis Sliding-Window Rate Limiter                     │
│  - Server-Sent Events (SSE) Streaming         - Token Estimator & Dynamic Context Guard               │
└───────────────────────────────────┬──────────────────────────────────┬───────────────────────────────┘
                                    │                                  │
                                    ▼                                  ▼
┌──────────────────────────────────────────────┐   ┌───────────────────────────────────────────────────┐
│              PostgreSQL + pgvector           │   │                  Redis Container                  │
│  - LangGraph Thread Checkpoints              │   │  - 30 req/min API Sliding Window Rate Limiting   │
│  - 1536-dim Cosine Memory Embeddings         │   │  - Temporary Result & Session Caching             │
└──────────────────────────────────┬───────────┘   └───────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 LangGraph ReAct Autonomous Agent                                     │
│                                                                                                      │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────┐     │
│     │ Node 1: Dynamic Context Assembly                                                         │     │
│     │ - Multi-turn Conversation Messages from Postgres Checkpointer                            │     │
│     │ - Active Repository Extraction (Zero Hardcoding)                                         │     │
│     │ - pgvector Semantic Preference Retrieval (limit=3, cosine distance)                       │     │
│     └────────────────────────────────────────────┬─────────────────────────────────────────────┘     │
│                                                  ▼                                                   │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────┐     │
│     │ Node 2: Reasoning & Tool Calling (Groq 120B / Llama 3)                                    │     │
│     │ - Structured Outputs via Pydantic Schemas                                                │     │
│     │ - LangSmith Tracing Spans                                                                │     │
│     └────────────────────────────────────────────┬─────────────────────────────────────────────┘     │
│                                                  ▼                                                   │
│                        Decision Gate: Does response require Tool Calls?                              │
│                                       /                      \                                       │
│                                     YES                      NO ──> Stream Final Response to Client  │
│                                     /                                                                │
│                                    ▼                                                                 │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────┐     │
│     │ Node 3: Safety & Human-in-the-Loop (HITL) Gate                                           │     │
│     │ - Read Operations (search_notes, list_issues, get_issue) ─────────> Safe / Auto-Execute  │     │
│     │ - Destructive Write Operations (create_comment) ──> PAUSE via LangGraph interrupt()      │     │
│     │   * Emits `approval_required` SSE event to Client                                        │     │
│     │   * Awaits User Click: [Approve & Execute] or [Reject]                                   │     │
│     └────────────────────────────────────────────┬─────────────────────────────────────────────┘     │
│                                                  ▼                                                   │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────┐     │
│     │ Node 4: Decoupled Model Context Protocol (MCP) Subprocesses                              │     │
│     │                                                                                          │     │
│     │  [MCP Server 1: GitHub API Client]        [MCP Server 2: Knowledge Base Engine]          │     │
│     │  - list_issues (Filtered PRs & limits)    - BM25Okapi Lexical Keyword Search             │     │
│     │  - get_issue (Issue + Thread Comments)    - In-Memory Startup Index                      │     │
│     │  - create_comment (Guarded Action)        - Sub-millisecond ADR/Runbook Retrieval        │     │
│     └──────────────────────────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Production Observability (LangSmith)                                 │
│  - Full Waterfall Latency Tracing (Groq queue vs generation vs MCP latency)                         │
│  - Tool-Level Telemetry via @traceable(run_type="tool")                                              │
│  - Token Consumption Accounting (Prompt vs Completion Tokens)                                        │
│  - LLM-as-a-Judge Architecture for Automated Hallucination & Groundedness Scoring                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Why This Architecture Matches Modern AI Engineering

### 1. Context Retention Without Hardcoding
- **Zero Hardcoded Repositories:** Unlike naive prototypes, the agent dynamically identifies target repositories from natural language or conversational memory, remembers them throughout the session thread, and proactively asks for clarification when missing.
- **pgvector Semantic Memory:** Automatically ingests and indexes user preferences (coding standards, workflow habits) into PostgreSQL using cosine distance embeddings, surfacing relevant context on every turn.
- **Context Window Protection:** Real-time token estimator monitors conversation density and triggers warnings when context usage reaches 75% capacity (6,144 / 8,192 tokens) to prevent prompt degradation.

### 2. Hybrid Retrieval (BM25 + Semantic Embeddings)
- **BM25 Lexical Precision (Notes MCP):** Company policies, architectural decision records (ADRs), and runbooks are indexed on startup using **BM25Okapi**. Lexical search eliminates semantic drift and excels at exact technical keyword matching (function names, error codes, policy numbers).
- **Semantic Vector Search (`pgvector`):** User preferences and past interactions are embedded to capture fuzzy conceptual meaning.
- **Cross-Source Synthesis:** The agent can synthesize across external tools (GitHub tickets) and internal knowledge (deployment runbooks) in a single unified reasoning cycle.

### 3. Human-in-the-Loop (HITL) Safety Guardrails
- **Destructive Action Isolation:** Read operations execute automatically. Write actions (`create_comment`) trigger an execution pause using LangGraph's native `interrupt()`.
- **Thread State Checkpointing:** The entire graph state is frozen in PostgreSQL. The client receives an interactive **Approval Card** over SSE. Once the user clicks *Approve* or *Reject*, the thread resumes from the exact checkpoint.

### 4. Enterprise Evaluation & Observability
- **Offline Deterministic Benchmarks (`evals/runner.py`):** 40 automated test scenarios verifying tool selection accuracy (100%), argument schema validity (100%), and task success (92.5%).
- **Adversarial Red-Teaming (`security/red_team_runner.py`):** Automated prompt injection and jailbreak suite verifying 0.0% attack success rate against strict Pydantic boundaries.
- **Online Observability (LangSmith):** Live end-to-end tracing capturing model latency, Groq queue times, token usage, tool spans via `@traceable(run_type="tool")`, and configured for asynchronous **LLM-as-a-Judge** scoring.

---

## Quantitative Evaluation Benchmark

All metrics below are reproducible via `evals/runner.py` and `security/red_team_runner.py`:

| Evaluation Dimension | Benchmark Result | Measurement Methodology |
| :--- | :---: | :--- |
| **Tool Selection Accuracy** | **100.0%** | Automated benchmark across 40 distinct intent scenarios |
| **Pydantic Argument Validity** | **100.0%** | Strict schema validation rate on MCP tool inputs |
| **Task Completion Success** | **92.5%** | Multi-step reasoning and synthesis verification |
| **Prompt Injection Defense** | **0.0% Attack Success** | Adversarial jailbreak and malicious instruction evaluation |
| **P50 Latency** | **2.4s** | Measured end-to-end including embedding and tool execution |
| **P99 Latency** | **19.6s** | Measured during multi-tool sequential fetches |
| **Cost Per Interaction** | **< $0.001** | High-throughput 120B parameter inference on Groq |

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Language & Runtime** | Python 3.14.4 (Strict type hints, PEP 695 generics, async/await) |
| **Agent Framework** | LangGraph 0.3+, LangChain Core, Pydantic v2 |
| **Model & Inference** | `openai/gpt-oss-120b` & Llama 3 via Groq API |
| **Tooling Protocol** | Model Context Protocol (`mcp==2.3.0`, `FastMCP`) over stdio |
| **Databases** | PostgreSQL 16 (`pgvector`), Redis 7 Alpine |
| **Web & API Gateway** | FastAPI, Uvicorn, Server-Sent Events (SSE), HTTP-only secure cookies |
| **Frontend UI** | Bespoke vanilla HTML5/JS (Cosmic Nebula & Starlight theme, marked.js) |
| **Observability** | LangSmith, Langfuse, Python `logging` audit trails |
| **DevOps & Containers** | Docker, Docker Compose, automated EC2 deployment scripts |
| **Code Quality** | Ruff (0 errors, 100% compliant), Pytest (8/8 unit & integration tests passing) |

---

## Project Structure

```text
mcp-assistant/
├── agent/                      # LangGraph ReAct state machine & tool bindings
│   ├── graph.py                # Graph compilation, checkpointer, and node functions
│   ├── tools.py                # Pydantic schemas and stdio MCP dispatchers
│   └── cli.py                  # Standalone interactive terminal client
├── api/                        # Production FastAPI REST & SSE Gateway
│   ├── main.py                 # /chat, /approve, /reject, /notes, /memories endpoints
│   ├── auth.py                 # PBKDF2 password hashing & HttpOnly session management
│   └── rate_limiter.py         # Redis sliding-window request throttling
├── mcp_servers/                # Decoupled Model Context Protocol servers
│   ├── github/                 # FastMCP GitHub server (list_issues, get_issue, create_comment)
│   └── notes/                  # FastMCP Notes server (BM25Okapi lexical retrieval)
├── memory/                     # Semantic vector memory layer
│   └── vector_memory.py        # pgvector cosine similarity store & embedding queries
├── security/                   # Hardened security & red-teaming utilities
│   ├── audit.py                # Structured JSON audit logger for restricted actions
│   └── red_team_runner.py      # Automated prompt injection test suite
├── evals/                      # Offline evaluation engine
│   ├── runner.py               # Deterministic benchmark runner across 40 scenarios
│   └── scenarios.json          # Benchmark dataset and expected trajectories
├── infra/                      # Infrastructure & deployment manifests
│   ├── init.sql                # PostgreSQL schema (pgvector extension, tables, indexes)
│   └── ec2_setup.sh            # Automated cloud provisioning & Docker setup script
├── notes_data/                 # Enterprise Markdown knowledge corpus (runbooks, ADRs)
├── tests/                      # Automated test suite
│   ├── test_github_client.py   # Unit tests for input validation, truncation, error mapping
│   └── test_health.py          # API health check tests
├── ui/                         # User Interface
│   └── index.html              # Cosmic Nebula & Starlight dual-tone console
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Local database orchestrator (Postgres + Redis)
├── docker-compose.prod.yml     # Multi-container production stack
└── pyproject.toml              # Build system, Ruff, and pytest configurations
```

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.14+ (or Python 3.11+)
- Docker & Docker Compose
- API Keys: `GROQ_API_KEY`, `GITHUB_TOKEN`, `LANGCHAIN_API_KEY`

### 2. Environment Configuration
Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_groq_key
GITHUB_TOKEN=ghp_your_github_token
GITHUB_ALLOWED_REPOS=owner/repo,Jyoshnap52/map-test-repo

# Databases
DATABASE_URL=postgresql://mcp:mcp_secure_pass@localhost:5433/mcp_assistant
REDIS_URL=redis://localhost:6379/0

# Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_langsmith_key
LANGCHAIN_PROJECT=mcp-assistant
```

### 3. Start Database Services
```zsh
docker compose up -d
```
*(Spins up PostgreSQL with `pgvector` on port `5433` and Redis on `6379`).*

### 4. Run Automated Tests & Code Quality Checks
```zsh
source .venv/bin/activate
ruff check .
pytest -v
```
Output:
```text
All checks passed!
============================== 8 passed in 3.35s ===============================
```

### 5. Launch the Assistant
```zsh
uvicorn api.main:app --reload --port 8001
```
Navigate to **`http://localhost:8001`** to enter the workspace!

---

## Production Cloud Deployment (AWS EC2 / DigitalOcean)

To run the complete production multi-container stack in the cloud:

```bash
# 1. SSH into cloud instance (Ubuntu 24.04 LTS)
git clone https://github.com/Jyoshnap52/mcp-personal-ai-assistant.git
cd mcp-personal-ai-assistant

# 2. Run automated server setup script
chmod +x infra/ec2_setup.sh
./infra/ec2_setup.sh

# 3. Launch isolated multi-container stack
docker compose -f docker-compose.prod.yml up -d --build
```

---

## License & Author

Authored by **Palli Jyoshna**  
Designed for scalable, resilient, and observable agentic workflows. Distributed under the MIT License.
