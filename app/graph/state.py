from typing import Any, TypedDict


class AnalystGraphState(TypedDict, total=False):
    # Request inputs
    dataset_id: str
    question: str
    dataset_version: str | None
    user_overrides: list[str]

    # Dataset and context
    df: Any
    dataset_context: dict
    actual_version: str

    # Planner output
    plan: dict
    answerable: bool
    clarification: str | None

    # Current plan step
    step_index: int
    current_step: dict
    current_flags: list[dict]
    current_analyst_input: dict
    analyst_output: dict
    verifier_output: dict
    attempt: int
    missing_columns: list[str]

    # Accumulated outputs
    verified_results: list[dict]
    all_caveats: list[str]
    step_flags: dict[str, list[dict]]
    final_report: dict
    status: str

    # Conversation and follow up
    conversation_id: str | None
    conversation_history: list[dict]
    followup_decision: dict
