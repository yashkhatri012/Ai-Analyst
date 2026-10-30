
import json
from typing import Any, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.prompts.followup_prompt import FOLLOWUP_SYSTEM_PROMPT
from app.llm.service import get_llm_for_agent


class FollowUpDecision(BaseModel):
    route: Literal["context", "analysis", "clarification"]

    standalone_question: str = Field(
        description="The latest question rewritten to be self-contained."
    )

    context_answer: str = Field(
        default="",
        description=(
            "Direct answer grounded in previous verified results. "
            "Required for context route; empty for other routes."
        ),
    )

    clarification_question: str | None = Field(
        default=None,
        description="Question to ask if essential information is missing.",
    )


async def resolve_followup(
    question: str,
    conversation_history: list[dict[str, Any]],
    dataset_id: str,
    dataset_version: str | None,
) -> FollowUpDecision:
    llm = get_llm_for_agent("followup")
    structured_llm = llm.with_structured_output(FollowUpDecision)

    payload = {
        "question": question,
        "current_dataset": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
        },
        "conversation_history": conversation_history[-10:],
    }

    result = await structured_llm.ainvoke([
        SystemMessage(content=FOLLOWUP_SYSTEM_PROMPT),
        HumanMessage(
            content=json.dumps(payload, ensure_ascii=False, default=str)
        ),
    ])

    if result.route == "context" and not result.context_answer.strip():
        # Never return an empty answer as a successful context response.
        return result.model_copy(update={"route": "analysis"})

    if result.route == "clarification":
        if not result.clarification_question:
            return result.model_copy(update={"route": "analysis"})

    return result
