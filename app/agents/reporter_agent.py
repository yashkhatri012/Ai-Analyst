
from typing import Any

from pydantic import BaseModel, Field
from app.llm.prompts.reporter_prompt import REPORTER_SYSTEM_PROMPT

import json

from langchain_core.messages import HumanMessage, SystemMessage
from app.llm.service import get_llm_for_agent

class ReporterInput(BaseModel):
    user_question: str
    plan: dict[str, Any]
    verified_results: list[dict[str, Any]] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class KeyFinding(BaseModel):
    finding: str
    evidence: list[str] = Field(default_factory=list)


class ReporterOutput(BaseModel):
    answer: str
    key_findings: list[KeyFinding] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)



async def generate_report(
    reporter_input: ReporterInput,
) -> ReporterOutput:
    llm = get_llm_for_agent("reporter")
    structured_llm = llm.with_structured_output(ReporterOutput)

    payload = reporter_input.model_dump()

    response = await structured_llm.ainvoke(
        [
            SystemMessage(content=REPORTER_SYSTEM_PROMPT),
            HumanMessage(
                content=json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                )
            ),
        ]
    )

    return response
