"""Graph construction only: nodes and routing logic live in separate modules."""
from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    analyst_node,
    finish_analysis_node,
    finish_empty_plan_node,
    finish_unanswerable_node,
    load_context_node,
    planner_node,
    prepare_step_node,
    record_step_node,
    reporter_node,
    skip_step_node,
    verifier_node,
)
from app.graph.routing import (
    route_after_finish_analysis,
    route_after_planner,
    route_after_prepare,
    route_after_record,
    route_after_verifier,
)
from app.graph.state import AnalystGraphState
from app.graph.followup_nodes import (
    resolve_followup_node,
    finish_context_answer_node,
    finish_clarification_node,
)

from app.graph.routing import route_after_followup

def build_graph():
    """Build and compile the AI Analyst StateGraph."""
    graph = StateGraph(AnalystGraphState)

    graph.add_node("load_context", load_context_node)
    graph.add_node("planner", planner_node)
    graph.add_node("prepare_step", prepare_step_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("record_step", record_step_node)
    graph.add_node("skip_step", skip_step_node)
    graph.add_node("finish_unanswerable", finish_unanswerable_node)
    graph.add_node("finish_empty_plan", finish_empty_plan_node)
    graph.add_node("finish_analysis", finish_analysis_node)
    graph.add_node("reporter", reporter_node)
    graph.add_node("resolve_followup", resolve_followup_node)
    graph.add_node("finish_context_answer", finish_context_answer_node)
    graph.add_node("finish_clarification", finish_clarification_node)

    graph.set_entry_point("resolve_followup")
    graph.add_edge("load_context", "planner")

    graph.add_conditional_edges(
        "resolve_followup",
        route_after_followup,
        {
            "load_context": "load_context",
            "finish_context_answer": "finish_context_answer",
            "finish_clarification": "finish_clarification",
        },
    )

    graph.add_edge("finish_context_answer", END)
    graph.add_edge("finish_clarification", END)
    
    graph.add_conditional_edges("planner", route_after_planner, {
        "finish_unanswerable": "finish_unanswerable",
        "finish_empty_plan": "finish_empty_plan",
        "prepare_step": "prepare_step",
    })
    graph.add_conditional_edges("prepare_step", route_after_prepare, {
        "finish_analysis": "finish_analysis",
        "skip_step": "skip_step",
        "analyst": "analyst",
    })
    graph.add_edge("analyst", "verifier")
    graph.add_conditional_edges("verifier", route_after_verifier, {
        "analyst": "analyst",
        "record_step": "record_step",
    })
    graph.add_conditional_edges("record_step", route_after_record, {
        "prepare_step": "prepare_step",
        "finish_analysis": "finish_analysis",
    })
    graph.add_conditional_edges("skip_step", route_after_record, {
        "prepare_step": "prepare_step",
        "finish_analysis": "finish_analysis",
    })
    graph.add_conditional_edges("finish_analysis", route_after_finish_analysis, {
        "reporter": "reporter",
        "end": END,
    })
    graph.add_edge("reporter", END)
    graph.add_edge("finish_unanswerable", END)
    graph.add_edge("finish_empty_plan", END)

    return graph.compile()
