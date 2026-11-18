# MCP Assistant

An enterprise-grade, agentic AI assistant powered by the Model Context Protocol (MCP), LangGraph ReAct workflows, PostgreSQL with pgvector, and Redis.

## Problem
Modern AI applications need to connect autonomously to external data (internal policy documents) and write actions (GitHub issue tracking), while preventing hallucinations, protecting API keys, and ensuring human supervision before executing destructive or write operations.

## Architecture
- **Protocol Layer:** Dual Model Context Protocol (MCP) servers (`github` and `notes` with BM25 keyword search) communicating over `stdio`.
- **Agentic Core:** LangGraph ReAct state graph running Groq (`openai/gpt-oss-20b`) with tool-calling capabilities.
- **Safety & HITL Gate:** Human-in-the-Loop (HITL) gate that pauses execution via LangGraph `interrupt()` whenever write operations (`create_comment`) are attempted.
- **Persistence:** PostgreSQL checkpointer with persistent connection pooling to snapshot and resume paused conversation threads.
- **API & Streaming:** FastAPI server streaming real-time Server-Sent Events (SSE) across `/chat`, `/approve`, and `/reject` endpoints.
- **CI/CD Quality Gate:** Automated GitHub Actions workflow enforcing benchmark thresholds on all commits.

## Setup

```bash
# 1. Start databases (PostgreSQL & Redis)
make up

# 2. Run unit and integration tests
make test

# 3. Run code quality checks
make lint

# 4. Launch the live FastAPI server
make run
```

## Results

> **Note:** Every metric in this table is benchmarked and reproduced directly from scripts located in `evals/`.

| Metric | Baseline | Final | How measured |
|---|---|---|---|
| Tool-selection accuracy | 100.0% | 100.0% | `evals/runner.py` (40 scenarios) |
| Argument validity | 100.0% | 100.0% | Pydantic schema validation tests |
| Task success | 92.5% | 92.5% | End-to-end task scenario evals |
| Injection attack success rate | 0.0% | 0.0% | Allowlist & Pydantic schema rejection |
| Latency p50 | 5814 ms | 5814 ms | Automated benchmark run |
| Latency p95 | 10573 ms | 10573 ms | Automated benchmark run |
| Cost per request | < $0.001 | < $0.001 | Groq token estimation & Langfuse |

## Security
- **Principle of Least Privilege:** Fine-grained GitHub personal access tokens restricted strictly to Issues (Read/Write) on an explicit repository allowlist.
- **Network Isolation:** PostgreSQL and Redis ports bound strictly to `127.0.0.1` to prevent exposure over local Wi-Fi.
- **Error Sanitization:** All custom client exceptions strip raw headers and error payloads, preventing API token leakage in logs.
- **Context Length Protection:** Truncation of large external bodies (max 2000 characters) to prevent prompt bloat and denial-of-wallet.

## Limitations
- **BM25 Keyword Search:** Best for exact keyword retrieval; planned upgrade to hybrid search with `sentence-transformers` embeddings and ChromaDB vector store.
- **Single-Node Deployment:** Current PostgreSQL checkpointer runs as a single instance; distributed deployment would require a managed cluster.