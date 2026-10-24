import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.prompts.verifier_prompt import VERIFIER_SYSTEM_PROMPT
from app.llm.service import get_llm_for_agent
from app.analysis.compact import compact


class VerifierInput(BaseModel):
    user_question: str
    plan_step: dict[str, Any]
    code: str
    result_table: list[dict[str, Any]] = Field(default_factory=list)
    chart_spec: dict[str, Any] | None = None
    quality_flags: list[dict[str, Any]] = Field(default_factory=list)
    execution_error: str | None = None


# What the orchestrator and reporter receive (unchanged contract).
class VerifierOutput(BaseModel):
    step_id: int
    status: Literal["pass", "retry", "caveat"]
    issues: list[str] = Field(default_factory=list)
    retry_instructions: str | None = None
    caveats: list[str] = Field(default_factory=list)


# What the LLM generates (small).
class VerifierLLMOutput(BaseModel):
    status: Literal["pass", "retry", "caveat"]
    notes: list[Annotated[str, Field(max_length=160)]] = Field(
        default_factory=list, max_length=3,
        description="retry: problems. caveat: limitations. pass: empty.",
    )
    retry_instructions: str | None = Field(default=None, max_length=300)


def validate_execution(verifier_input: VerifierInput) -> VerifierOutput | None:
    step_id = verifier_input.plan_step.get("id", -1)

    if verifier_input.execution_error:
        return VerifierOutput(
            step_id=step_id,
            status="retry",
            issues=[verifier_input.execution_error[:300]],
            retry_instructions=(
                "Fix the execution error and regenerate the analysis code. "
                "Use only columns supplied in the analysis context."
            ),
        )

    if not verifier_input.result_table:
        return VerifierOutput(
            step_id=step_id,
            status="retry",
            issues=["The analysis returned no tabular results."],
            retry_instructions="Return a non-empty result table.",
        )

    return None


async def verify_analysis(verifier_input: VerifierInput) -> VerifierOutput:
    if (early := validate_execution(verifier_input)) is not None:
        return early

    step = verifier_input.plan_step
    step_id = step.get("id", -1)
    chart = verifier_input.chart_spec or {}

    payload = {
        "question": verifier_input.user_question,
        "step": {k: step.get(k) for k in ("goal", "columns", "method")},
        "code": verifier_input.code,
        "result": compact(verifier_input.result_table),
        "chart": chart.get("title") or chart.get("type"),
        "quality_flags": verifier_input.quality_flags,
    }

    llm = get_llm_for_agent("verifier").with_structured_output(VerifierLLMOutput)
    out = await llm.ainvoke([
        SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), default=str
        )),
    ])

    return VerifierOutput(
        step_id=step_id,  # from code, so the LLM can't get it wrong
        status=out.status,
        issues=out.notes if out.status == "retry" else [],
        caveats=out.notes if out.status == "caveat" else [],
        retry_instructions=out.retry_instructions if out.status == "retry" else None,
    )