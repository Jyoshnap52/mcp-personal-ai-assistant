import os
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langsmith import traceable
from psycopg_pool import ConnectionPool
from typing_extensions import TypedDict

from agent.tools import (
    execute_create_comment,
    execute_get_issue,
    execute_list_issues,
    execute_save_preference,
    execute_search_notes,
    get_agent_tools,
)
from memory.vector_memory import VectorMemoryStore

load_dotenv()

SYSTEM_PROMPT = """You are an intelligent, highly capable enterprise AI assistant (like ChatGPT).
- Conversational Awareness: Always remember the context, repository, and issue numbers discussed
  earlier in the conversation. Never ask the user to repeat information they already gave you.
- Repository Detection: Whenever a user mentions a repository (by name, URL, or owner/repo),
  immediately lock onto that repository for all subsequent commands (issues, comments, summaries).
- If the user asks to view or manage issues without specifying which repository they want to work on
  (and none has been mentioned yet), politely ask: "Which repository would you like to inspect?
  Please share the repository link or owner/repo name."
- Document Search: Use 'search_notes' to find company policies, engineering guidelines, and HR docs.
- GitHub Actions:
  * Use 'list_issues' to retrieve open issues for the active repository.
  * Use 'get_issue' to view specific issue details and full comment threads.
  * Use 'create_comment' when instructed to post a comment.
- Answer clearly, helpfully, and with rich formatting (markdown tables, bullet points).
"""

RISKY_TOOLS = {"create_comment"}


# --- 1. Graph State Definition ---


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --- 2. Model & Tools ---

tools = get_agent_tools()
memory_store = VectorMemoryStore()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    groq_api_key=groq_api_key,
)
model_with_tools = llm.bind_tools(tools)


# --- 3. Node Functions ---


def get_langfuse_callback():
    """Returns Langfuse callback handler if keys are configured."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    if pk and not pk.startswith("pk-lf-your"):
        return [CallbackHandler()]
    return []


def call_model(state: AgentState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    """Calls Groq with history, freshly retrieved vector memories, and Langfuse tracing."""
    user_id = config.get("configurable", {}).get("user_id", "default_user")

    # Extract conversation messages and find latest user query for memory retrieval
    conv_messages = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    latest_query = ""
    for m in reversed(conv_messages):
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            latest_query = m.content
            break

    memories = []
    if latest_query.strip():
        try:
            memories = memory_store.retrieve_relevant_memories(
                user_id=user_id, query=latest_query, limit=3
            )
        except Exception:
            memories = []

    system_text = SYSTEM_PROMPT
    if memories:
        mem_bullets = "\n".join(f"- {m['memory']}" for m in memories)
        system_text += (
            f"\n\nRelevant user preferences retrieved from long-term memory:\n{mem_bullets}"
        )

    full_messages = [SystemMessage(content=system_text)] + conv_messages

    callbacks = get_langfuse_callback()
    if callbacks:
        response = model_with_tools.invoke(full_messages, config={"callbacks": callbacks})
    else:
        response = model_with_tools.invoke(full_messages)
    return {"messages": [response]}


def execute_tool_call(name: str, args: dict[str, Any]) -> str:
    """Dispatches a tool call to the respective executor with explicit LangSmith tool telemetry."""
    @traceable(run_type="tool", name=name)
    def _run(**kwargs):
        if name == "search_notes":
            return execute_search_notes(**kwargs)
        if name == "list_issues":
            return execute_list_issues(**kwargs)
        if name == "get_issue":
            return execute_get_issue(**kwargs)
        if name == "create_comment":
            return execute_create_comment(**kwargs)
        if name == "save_preference":
            return execute_save_preference(**kwargs)
        return f"Unknown tool: {name}"

    return _run(**args)


def safe_tool_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    """Executes tools, pausing via interrupt() if a risky tool is encountered."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        call_id = tc["id"]

        # If tool is marked RISKY -> Interrupt execution for human approval!
        if tool_name in RISKY_TOOLS:
            decision = interrupt(
                {
                    "action": tool_name,
                    "args": tool_args,
                    "message": (
                        f"Approval required to post comment on issue "
                        f"#{tool_args.get('issue_number')} in {tool_args.get('repo')}"
                    ),
                }
            )

            # Check decision passed when resumed: {"approved": True/False, "reason": ...}
            if isinstance(decision, dict) and decision.get("approved"):
                result = execute_tool_call(tool_name, tool_args)
            else:
                reason = (
                    decision.get("reason", "Action cancelled by user.")
                    if isinstance(decision, dict)
                    else "Action rejected by user."
                )
                result = f"Error: Tool execution was rejected by human operator. Reason: {reason}"
        else:
            # Safe tool: execute automatically
            result = execute_tool_call(tool_name, tool_args)

        tool_messages.append(ToolMessage(content=result, tool_call_id=call_id))

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# --- 4. Build and Compile Graph with Postgres Checkpointer ---


def get_agent_graph():
    """Builds and compiles graph using Postgres checkpointer with MemorySaver fallback."""
    db_url = os.getenv("DATABASE_URL", "")
    checkpointer = None

    if db_url and "localhost:5433" not in db_url and "127.0.0.1:5433" not in db_url:
        try:
            pool = ConnectionPool(
                conninfo=db_url,
                max_size=10,
                open=False,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            pool.open()
            with pool.connection() as test_conn:
                test_conn.execute("SELECT 1")
            pg_saver = PostgresSaver(pool)
            pg_saver.setup()
            checkpointer = pg_saver
        except Exception as e:
            print(f"PostgreSQL checkpointer unavailable ({e}). Falling back to MemorySaver.")
            checkpointer = MemorySaver()
    else:
        checkpointer = MemorySaver()

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", safe_tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


# Default compiled app instance
app = get_agent_graph()
