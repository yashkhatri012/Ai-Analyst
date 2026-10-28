import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.prompts.rewriter_prompt import REWRITER_SYSTEM_PROMPT
from app.llm.service import get_llm_for_agent
from app.session.store import Session


class RewriteResult(BaseModel):
    kind: Literal["refine", "history", "independent", "clarification"]
    standalone_question: str = Field(max_length=300)
    user_overrides: list[Annotated[str, Field(max_length=100)]] = Field(
        default_factory=list, max_length=3,
        description="Explicit user instructions about data handling.",
    )
    ask_user: str | None = Field(
        default=None, max_length=200,
        description="Set only if the reference is too ambiguous to resolve.",
    )


async def rewrite(question: str, session: Session) -> RewriteResult:
    # First turn: nothing to resolve, so skip the LLM call.
    if not session.turns:
        return RewriteResult(kind="independent", standalone_question=question)

    payload = {
        "new_question": question,
        "history": session.prompt_view(),
        "active_overrides": session.overrides,
    }
    llm = get_llm_for_agent("rewriter").with_structured_output(RewriteResult)
    return await llm.ainvoke([
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"))),
    ])