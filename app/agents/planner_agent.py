
import json
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from app.llm.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from app.llm.service import get_llm_for_agent


class PlanStep(BaseModel):
    """One analytical operation for the Analyst to execute."""

    id: int = Field(description="Unique step number, starting from 1")
    goal: str = Field(description="What this step must determine")
    columns: list[str] = Field(
        description="Existing dataset columns needed for this step"
    )
    method: str = Field(
        description="Proposed analytical method, not executable code"
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="IDs of steps that must finish first"
    )


class AnalysisPlan(BaseModel):
    """Structured plan produced by the Planner."""

    answerable: bool = Field(
        description="Whether the question can be answered with the available data"
    )
    clarification: str | None = Field(
        default=None,
        description="Question to ask the user if clarification is necessary"
    )
    intent: Literal[
        "comparison",
        "trend",
        "correlation",
        "anomaly",
        "summary"
    ] = Field(description="Primary analytical intent")
    steps: list[PlanStep] = Field(
        description="Ordered analytical steps for the Analyst"
    )


async def plan(
    dataset_id: str,
    question: str,
    dataset_context: dict,
) -> AnalysisPlan:

    llm = get_llm_for_agent("planner")
    structured_llm = llm.with_structured_output(AnalysisPlan)

    planner_input = {
        "dataset_id": dataset_id,
        "user_question": question,
        "dataset_context": dataset_context,
    }

    response = await structured_llm.ainvoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(
            content=json.dumps(planner_input, ensure_ascii=False)
        ),
    ])

    return response
