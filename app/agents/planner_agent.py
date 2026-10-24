
import json
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from app.llm.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from app.llm.service import get_llm_for_agent


class PlanStep(BaseModel):
    id: int
    goal: str = Field(max_length=120)
    columns: list[str]
    method: str = Field(max_length=200, description="Exact statistic or aggregation, one sentence")
    depends_on: list[int] = Field(default_factory=list)

class AnalysisPlan(BaseModel):
    answerable: bool
    clarification: str | None = None
    intent: Literal["comparison", "trend", "correlation", "anomaly", "summary"]
    steps: list[PlanStep] = Field(max_length=3)



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

    if len(response.steps) > 3:
        response.steps = response.steps[:3]

    return response
