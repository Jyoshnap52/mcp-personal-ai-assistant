import json
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import app
from agent.tools import metrics as tool_metrics

load_dotenv()

SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios.json"
RESULTS_PATH = Path(__file__).resolve().parent / "baseline_results.json"


def percentile(values: list[float], p: float) -> float:
    """Calculates the p-th percentile of a list of numbers."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    d = k - f
    return round(sorted_values[f] + d * (sorted_values[c] - sorted_values[f]), 2)


def run_eval_suite():
    if not SCENARIOS_PATH.exists():
        print(f"❌ Error: {SCENARIOS_PATH} not found.")
        return

    with open(SCENARIOS_PATH, encoding="utf-8") as f:
        scenarios = json.load(f)

    print("=" * 75)
    print(f"🧪 RUNNING BENCHMARK EVALUATION SUITE ({len(scenarios)} SCENARIOS)")
    print("=" * 75)

    tool_selection_correct = 0
    task_success_correct = 0
    latencies = []
    detailed_results = []

    start_valid_calls = tool_metrics.valid_calls
    start_total_calls = tool_metrics.total_calls

    for i, sc in enumerate(scenarios, 1):
        sc_id = sc["id"]
        question = sc["question"]
        expected_tool = sc["expected_tool"]
        expected_outcomes = sc.get("expected_outcome", [])

        # Unique thread per scenario for isolated checkpointing
        thread_id = f"eval-{sc_id}-{uuid.uuid4().hex[:6]}"
        config = {"configurable": {"thread_id": thread_id}}

        t0 = time.perf_counter()
        try:
            state = app.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            latencies.append(elapsed_ms)

            # Extract actual tool calls and final AI message
            actual_tools = []
            final_content = ""
            for msg in state["messages"]:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            actual_tools.append(tc["name"])
                    if msg.content:
                        final_content = msg.content

            # 1. Evaluate Tool Selection
            if expected_tool == "None":
                tool_match = len(actual_tools) == 0
            else:
                tool_match = expected_tool in actual_tools

            if tool_match:
                tool_selection_correct += 1

            # 2. Evaluate Task Success (keywords in final message or approval request)
            if expected_tool == "create_comment":
                # For risky tools, success means it triggered the approval interrupt
                outcome_match = (
                    "create_comment" in actual_tools or "approval" in final_content.lower()
                )
            else:
                outcome_match = all(kw.lower() in final_content.lower() for kw in expected_outcomes)

            if outcome_match:
                task_success_correct += 1

            status_icon = "✅" if (tool_match and outcome_match) else "⚠️"
            print(
                f"[{i:02d}/{len(scenarios):02d}] {status_icon} ID: {sc_id:<10} | "
                f"Tool: {str(actual_tools):<20} | Time: {elapsed_ms:>6.1f}ms"
            )

            detailed_results.append(
                {
                    "id": sc_id,
                    "question": question,
                    "expected_tool": expected_tool,
                    "actual_tools": actual_tools,
                    "tool_match": tool_match,
                    "outcome_match": outcome_match,
                    "latency_ms": elapsed_ms,
                }
            )

        except Exception as e:
            print(f"[{i:02d}/{len(scenarios):02d}] ❌ ID: {sc_id:<10} | Error: {e}")
            detailed_results.append(
                {
                    "id": sc_id,
                    "error": str(e),
                    "tool_match": False,
                    "outcome_match": False,
                }
            )

    # Summary Metrics Calculation
    total = len(scenarios)
    tool_acc = round((tool_selection_correct / total) * 100, 2)
    task_acc = round((task_success_correct / total) * 100, 2)

    total_eval_calls = tool_metrics.total_calls - start_total_calls
    valid_eval_calls = tool_metrics.valid_calls - start_valid_calls
    arg_validity = (
        round((valid_eval_calls / total_eval_calls) * 100, 2) if total_eval_calls > 0 else 100.0
    )

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_scenarios": total,
        "tool_selection_accuracy_pct": tool_acc,
        "argument_validity_pct": arg_validity,
        "task_success_pct": task_acc,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "detailed_results": detailed_results,
    }

    # Save to disk as official baseline
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("🏆 OFFICIAL BENCHMARK SCORECARD")
    print("=" * 75)
    print(f"Total Scenarios Evaluated:     {total}")
    print(f"Tool-Selection Accuracy:       {tool_acc}%")
    print(f"Argument Validity Rate:        {arg_validity}%")
    print(f"Task Success Rate:             {task_acc}%")
    print(f"Latency p50:                   {p50} ms")
    print(f"Latency p95:                   {p95} ms")
    print(f"Results Saved To:              {RESULTS_PATH}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_eval_suite()
