"""Public async entry point for invoking the compiled analysis graph."""
from typing import Any

from langsmith import traceable

from app.graph.graph import build_graph
from app.graph.state import AnalystGraphState


analysis_graph = build_graph()


@traceable(
    name="ai_analyst_langgraph_pipeline",
    run_type="chain",
    tags=["ai-analyst", "langgraph"],
    metadata={"component": "analysis-pipeline"},
)
async def run_analysis_pipeline(
    dataset_id: str,
    question: str,
    dataset_version: str | None = None,
    user_overrides: list[str] | None = None,
) -> dict[str, Any]:
    initial_state: AnalystGraphState = {
        "dataset_id": dataset_id,
        "question": question,
        "dataset_version": dataset_version,
        "user_overrides": user_overrides or [],
    }
    state = await analysis_graph.ainvoke(initial_state)

    if state.get("status") == "unanswerable":
        return {
            "status": "unanswerable",
            "clarification": state.get("clarification"),
            "plan": state.get("plan"),
        }

    if state.get("status") == "failed":
        return {
            "status": "failed",
            "plan": state.get("plan"),
            "caveats": state.get("all_caveats", []),
        }

    return {
        "status": "completed",
        "plan": state.get("plan"),
        "verified_results": state.get("verified_results", []),
        "report": state.get("final_report"),
        "caveats": state.get("all_caveats", []),
    }
