from app.graph.constants import MAX_ATTEMPTS
from app.graph.state import AnalystGraphState


def route_after_planner(state: AnalystGraphState) -> str:
    if not state.get("answerable"):
        return "finish_unanswerable"
    if not state.get("plan", {}).get("steps"):
        return "finish_empty_plan"
    return "prepare_step"


def route_after_prepare(state: AnalystGraphState) -> str:
    if state.get("status") == "steps_exhausted":
        return "finish_analysis"
    if state.get("status") == "step_missing_columns":
        return "skip_step"
    return "analyst"


def route_after_verifier(state: AnalystGraphState) -> str:
    status = state.get("verifier_output", {}).get("status")
    if status == "retry" and state.get("attempt", 0) < MAX_ATTEMPTS:
        return "analyst"
    return "record_step"


def route_after_record(state: AnalystGraphState) -> str:
    if state.get("step_index", 0) >= len(state.get("plan", {}).get("steps", [])):
        return "finish_analysis"
    return "prepare_step"


def route_after_finish_analysis(state: AnalystGraphState) -> str:
    return "reporter" if state.get("status") == "ready_to_report" else "end"


def route_after_followup(state: AnalystGraphState) -> str:
    route = state.get("followup_decision", {}).get("route")

    if route == "context":
        return "finish_context_answer"

    if route == "clarification":
        return "finish_clarification"

    return "load_context"