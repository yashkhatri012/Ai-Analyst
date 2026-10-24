
import json
from typing import Any

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.prompts.reporter_prompt import REPORTER_SYSTEM_PROMPT
from app.llm.service import get_llm_for_agent

MAX_ROWS = 15
MAX_COLS = 8


class ReporterInput(BaseModel):
    user_question: str
    plan: dict[str, Any]
    verified_results: list[dict[str, Any]] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


# What the LLM generates: short, bounded, references only.
class KeyFinding(BaseModel):
    finding: str = Field(max_length=250)
    step_ids: list[int] = Field(description="Steps supporting this finding")


class ReporterLLMOutput(BaseModel):
    answer: str = Field(max_length=600, description="Direct answer, 2-4 sentences")
    key_findings: list[KeyFinding] = Field(max_length=4)


# What the API returns: LLM output + artifacts attached by code.
class ReporterOutput(ReporterLLMOutput):
    caveats: list[str] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)


def _round(v):
    return round(v, 4) if isinstance(v, float) else v


def compact_result(r: dict[str, Any]) -> dict[str, Any]:
    """Columnar + truncated: keys appear once instead of once per row."""
    table = r.get("result_table") or []
    cols = list(table[0].keys())[:MAX_COLS] if table else []
    chart = r.get("chart_spec") or {}
    return {
        "step_id": r["step_id"],
        "columns": cols,
        "rows": [[_round(row.get(c)) for c in cols] for row in table[:MAX_ROWS]],
        "total_rows": len(table),  # tells the LLM the table was truncated
        "chart": chart.get("title") or chart.get("type"),  # name only, not spec
        "issues": r.get("verification_issues", []),
    }


async def generate_report(reporter_input: ReporterInput) -> ReporterOutput:
    llm = get_llm_for_agent("reporter")
    structured_llm = llm.with_structured_output(ReporterLLMOutput)

    caveats = list(dict.fromkeys(reporter_input.caveats))  # dedupe, keep order

    payload = {
        "user_question": reporter_input.user_question,
        "intent": reporter_input.plan.get("intent"),
        "steps": [
            {"id": s["id"], "goal": s["goal"]}
            for s in reporter_input.plan.get("steps", [])
        ],
        "results": [compact_result(r) for r in reporter_input.verified_results],
        "caveats": caveats,  # sent once, to inform the answer
    }

    llm_out = await structured_llm.ainvoke([
        SystemMessage(content=REPORTER_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ])

    return ReporterOutput(
        **llm_out.model_dump(),
        caveats=caveats,  # from code, not the LLM
        charts=[
            {"step_id": r["step_id"], "spec": r["chart_spec"]}
            for r in reporter_input.verified_results
            if r.get("chart_spec")
        ],
    )