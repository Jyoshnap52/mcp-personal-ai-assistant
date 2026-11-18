import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import app
from agent.tools import metrics

load_dotenv()


def run_prompt(prompt_text: str):
    """Executes a prompt through the LangGraph agent and prints the step-by-step trace."""
    print("\n" + "=" * 70)
    print(f"👤 USER: {prompt_text}")
    print("=" * 70)

    # Invoke the agent graph
    state = app.invoke({"messages": [HumanMessage(content=prompt_text)]})

    # Display execution trajectory
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n⚙️ [TOOL CALL]: {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            # Print brief preview of tool output
            preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            print(f"📥 [TOOL RESULT]: {preview}")
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            print(f"\n🤖 ASSISTANT:\n{msg.content}")


def main():
    repo = os.getenv("GITHUB_REPO_ALLOWLIST", "your-repo")

    prompts = [
        # 1. Notes HR Policy Question
        "How many sick days do we get per year, and when does it accrue?",
        # 2. Notes Remote Work Question
        "What is our policy for remote work and the home office setup allowance?",
        # 3. GitHub List Issues Question
        f"List all open issues in the repository {repo}.",
        # 4. GitHub Get Issue Details Question
        f"Get the details for issue #2 in the repository {repo}.",
        # 5. Combined Question
        "What are our Python coding conventions regarding line length and tests?",
    ]

    print("\n🚀 STARTING LANGGRAPH AGENT TEST SUITE...")

    for i, p in enumerate(prompts, 1):
        print(f"\n--- TEST SCENARIO {i} OF {len(prompts)} ---")
        run_prompt(p)

    # Print Official Argument Validity Metric for README Scorecard
    print("\n" + "=" * 70)
    print("📊 AGENT EVALUATION METRICS REPORT")
    print("=" * 70)
    print(f"Total Tool Invocations:     {metrics.total_calls}")
    print(f"Valid Argument Calls:       {metrics.valid_calls}")
    print(f"Invalid Argument Calls:     {metrics.invalid_calls}")
    print(f"Argument Validity Rate:     {metrics.validity_rate}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
