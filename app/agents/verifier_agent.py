
from typing import Any, Literal
from app.llm.prompts.verifier_prompt import VERIFIER_SYSTEM_PROMPT
from pydantic import BaseModel, Field
import json

from langchain_core.messages import HumanMessage, SystemMessage
from app.llm.service import get_llm_for_agent

class VerifierInput(BaseModel):
    user_question: str
    plan_step: dict[str, Any]
    code: str
    result_table: list[dict[str, Any]] = Field(default_factory=list)
    chart_spec: dict[str, Any] | None = None
    quality_flags: list[dict[str, Any]] = Field(default_factory=list)
    execution_error: str | None = None


class VerifierOutput(BaseModel):
    step_id: int
    status: Literal["pass", "retry", "caveat"]
    issues: list[str] = Field(default_factory=list)
    retry_instructions: str | None = None
    caveats: list[str] = Field(default_factory=list)


def validate_execution(
    verifier_input: VerifierInput,
) -> VerifierOutput | None:
    step_id = verifier_input.plan_step.get("id", -1)

    # An execution error must not be passed off as a valid result.
    if verifier_input.execution_error:
        return VerifierOutput(
            step_id=step_id,
            status="retry",
            issues=[verifier_input.execution_error],
            retry_instructions=(
                "Fix the execution error and regenerate the analysis code. "
                "Use only columns supplied in the analysis context."
            ),
        )

    # An empty result may be valid for some tasks, so request a retry
    # only when the plan expects a tabular result.
    if not verifier_input.result_table:
        return VerifierOutput(
            step_id=step_id,
            status="retry",
            issues=["The analysis returned no tabular results."],
            retry_instructions=(
                "Return a non-empty result table, or revise the step "
                "if an empty result is genuinely expected."
            ),
        )

    return None


async def verify_analysis(
    verifier_input: VerifierInput,
) -> VerifierOutput:
    # Run basic checks first. It runs the above check.
    deterministic_result = validate_execution(verifier_input)

    if deterministic_result is not None:
        return deterministic_result

    llm = get_llm_for_agent("verifier")
    structured_llm = llm.with_structured_output(VerifierOutput)

    payload = verifier_input.model_dump()

    response = await structured_llm.ainvoke([
        SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
        HumanMessage(
            content=json.dumps(payload, ensure_ascii=False, default=str)
        ),
    ])

    # Ensure the returned decision refers to the correct plan step.
    expected_id = verifier_input.plan_step.get("id", -1)

    if response.step_id != expected_id:
        raise ValueError(
            f"Verifier returned step_id={response.step_id}; "
            f"expected {expected_id}."
        )

    return response