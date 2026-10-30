
from app.agents.followup_agent import (
    FollowUpDecision,
    resolve_followup,
)
from app.graph.state import AnalystGraphState


async def resolve_followup_node(
    state: AnalystGraphState,
) -> dict:
    history = state.get("conversation_history", [])

    # First question: no follow up resolution is needed.
    if not history:
        decision = FollowUpDecision(
            route="analysis",
            standalone_question=state["question"],
        )
    else:
        decision = await resolve_followup(
            question=state["question"],
            conversation_history=history,
            dataset_id=state["dataset_id"],
            dataset_version=state.get("dataset_version"),
        )

    updates = {
        "followup_decision": decision.model_dump(),
    }

    # Send the resolved question to the existing Planner.
    if decision.route == "analysis":
        updates["question"] = decision.standalone_question

    return updates


def finish_context_answer_node(
    state: AnalystGraphState,
) -> dict:
    decision = state["followup_decision"]

    # Preserve the existing ReporterOutput response shape.
    report = {
        "answer": decision["context_answer"],
        "key_findings": [],
        "caveats": [],
        "charts": [],
    }

    return {
        "final_report": report,
        "status": "completed",
    }


def finish_clarification_node(
    state: AnalystGraphState,
) -> dict:
    decision = state["followup_decision"]

    return {
        "clarification": decision["clarification_question"],
        "status": "clarification",
    }
