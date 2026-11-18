import json
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.graph import app as agent_app
from api.auth import (
    COOKIE_NAME,
    authenticate_user,
    create_user,
    get_current_user,
    set_session_cookie,
)
from api.rate_limiter import check_rate_limit
from memory.vector_memory import VectorMemoryStore
from security.audit import AUDIT_LOG_PATH

load_dotenv()

app = FastAPI(title="MCP Assistant API with Auth & Human-in-the-Loop")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# --- Request Models ---


class AuthRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt or instruction.")
    thread_id: str | None = Field(default=None)


class ApprovalRequest(BaseModel):
    thread_id: str = Field(...)


class RejectionRequest(BaseModel):
    thread_id: str = Field(...)
    reason: str = Field(default="User declined action.")


class CreateNoteRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1)


# --- SSE Helper ---


# Context Window Token Limits (8,192 token window on Groq)
CONTEXT_WINDOW_LIMIT = 8192
WARNING_THRESHOLD_TOKENS = int(CONTEXT_WINDOW_LIMIT * 0.75)  # Warn at 75% capacity (6,144 tokens)


def estimate_tokens(messages: list) -> int:
    """Accurately estimates tokens in conversation history (~4 characters per token)."""
    total_chars = 0
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            total_chars += sum(len(str(c)) for c in content)
    return max(1, total_chars // 4)


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_graph_execution(input_data: Any, config: dict) -> AsyncGenerator[str, None]:
    try:
        for chunk in agent_app.stream(input_data, config, stream_mode="updates"):
            if "agent" in chunk:
                for msg in chunk["agent"].get("messages", []):
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield format_sse(
                                    "tool_call",
                                    {"tool": tc["name"], "args": tc["args"]},
                                )
                        if msg.content:
                            yield format_sse("message", {"content": msg.content})
            elif "tools" in chunk:
                for msg in chunk["tools"].get("messages", []):
                    if isinstance(msg, ToolMessage):
                        yield format_sse("tool_result", {"content": msg.content[:300]})

        state = agent_app.get_state(config)
        if state.next and state.tasks:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    for intr in task.interrupts:
                        yield format_sse(
                            "approval_required",
                            {
                                "thread_id": config["configurable"]["thread_id"],
                                "details": intr.value,
                            },
                        )
                    return

        # Only warn when the conversation context genuinely approaches the token limit (75% full)
        messages = state.values.get("messages", [])
        current_tokens = estimate_tokens(messages)
        if current_tokens >= WARNING_THRESHOLD_TOKENS:
            yield format_sse(
                "message",
                {
                    "content": (
                        f"\n\n---\n⚠️ **Context Window Alert "
                        f"({current_tokens:,} / {CONTEXT_WINDOW_LIMIT:,} tokens):** "
                        "Our conversation is approaching the model's memory capacity. "
                        "If you would like me to permanently store key notes or decisions "
                        "in your vector memory, tell me to **remember it** now, "
                        "or click **+ New Chat** to start a clean, fast session!"
                    )
                },
            )

        yield format_sse(
            "complete",
            {"thread_id": config["configurable"]["thread_id"], "status": "done"},
        )
    except Exception as e:
        yield format_sse("error", {"error": str(e)})


CurrentUser = Annotated[dict, Depends(get_current_user)]

# --- Public & Auth Routes ---

UI_PATH = Path(__file__).resolve().parents[1] / "ui" / "index.html"


@app.get("/")
def serve_ui():
    return FileResponse(
        UI_PATH,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/signup")
def signup(payload: AuthRequest, response: Response):
    user = create_user(payload.email, payload.password)
    set_session_cookie(response, user["user_id"], user["email"])
    return user


@app.post("/auth/login")
def login(payload: AuthRequest, response: Response):
    user = authenticate_user(payload.email, payload.password)
    set_session_cookie(response, user["user_id"], user["email"])
    return user


@app.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    response.headers["Clear-Site-Data"] = '"cookies"'
    return {"status": "logged_out"}


@app.get("/auth/me")
def me(user: CurrentUser):
    return user


# --- Protected AI Routes ---


@app.get("/memories")
def get_my_memories(user: CurrentUser):
    store = VectorMemoryStore()
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, memory_text, created_at FROM user_memories "
                "WHERE user_id = %s OR user_id = 'default_user' "
                "ORDER BY created_at DESC LIMIT 20;",
                (user["user_id"],),
            )
            rows = cur.fetchall()
    return {
        "user_id": user["user_id"],
        "memories": [{"id": r[0], "text": r[1], "created_at": r[2].isoformat()} for r in rows],
    }


NOTES_DIR = Path(__file__).resolve().parents[1] / "notes_data"
NOTES_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/notes")
def list_notes(user: CurrentUser):
    notes = []
    for p in NOTES_DIR.glob("*.md"):
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            content = ""
        notes.append(
            {
                "title": p.stem.replace("_", " ").title(),
                "filename": p.name,
                "snippet": content[:200] if content else "Empty document",
            }
        )
    return {"notes": notes}


@app.post("/notes")
def create_note(payload: CreateNoteRequest, user: CurrentUser):
    safe_title = (
        "".join(c for c in payload.title if c.isalnum() or c in ("-", "_")).strip() or "note"
    )
    note_path = NOTES_DIR / f"{safe_title}.md"
    note_path.write_text(payload.content, encoding="utf-8")
    return {"status": "created", "filename": note_path.name}


@app.get("/audit")
def get_audit_logs(user: CurrentUser):
    if not AUDIT_LOG_PATH.exists():
        return {"events": []}
    lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()[-25:]
    events = []
    for line in reversed(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"events": events}


@app.post("/chat", dependencies=[Depends(check_rate_limit)])
async def chat(request: ChatRequest, user: CurrentUser):
    raw_thread = request.thread_id or str(uuid.uuid4())
    full_thread = (
        raw_thread
        if raw_thread.startswith(f"{user['user_id']}:")
        else f"{user['user_id']}:{raw_thread}"
    )
    config = {
        "configurable": {
            "thread_id": full_thread,
            "user_id": user["user_id"],
        }
    }
    input_data = {"messages": [HumanMessage(content=request.message)]}
    return StreamingResponse(
        stream_graph_execution(input_data, config),
        media_type="text/event-stream",
    )


@app.post("/approve")
async def approve(request: ApprovalRequest, user: CurrentUser):
    raw_thread = request.thread_id
    full_thread = (
        raw_thread
        if raw_thread.startswith(f"{user['user_id']}:")
        else f"{user['user_id']}:{raw_thread}"
    )
    config = {"configurable": {"thread_id": full_thread, "user_id": user["user_id"]}}
    return StreamingResponse(
        stream_graph_execution(Command(resume={"approved": True}), config),
        media_type="text/event-stream",
    )


@app.post("/reject")
async def reject(request: RejectionRequest, user: CurrentUser):
    raw_thread = request.thread_id
    full_thread = (
        raw_thread
        if raw_thread.startswith(f"{user['user_id']}:")
        else f"{user['user_id']}:{raw_thread}"
    )
    config = {"configurable": {"thread_id": full_thread, "user_id": user["user_id"]}}
    return StreamingResponse(
        stream_graph_execution(
            Command(resume={"approved": False, "reason": request.reason}), config
        ),
        media_type="text/event-stream",
    )
