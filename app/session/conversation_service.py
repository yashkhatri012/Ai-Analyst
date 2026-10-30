
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import (
    create_conversation,
    get_conversation,
    get_conversation_history,
    get_next_turn_number,
    save_turn,
)
from app.graph.runner import run_analysis_pipeline

from sqlalchemy import update
from app.db.models import ConversationTurn


async def run_conversation_turn(
    session: AsyncSession,
    *,
    question: str,
    dataset_id: str,
    dataset_version: str | None = None,
    conversation_id: str | None = None,
    user_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Run one question within a persistent conversation."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Create or retrieve the conversation.
    if conversation_id is None:
        conversation = await create_conversation(session)
    else:
        try:
            conversation_uuid = uuid.UUID(conversation_id)
        except ValueError as exc:
            raise ValueError("Invalid conversation_id.") from exc

        conversation = await get_conversation(session, conversation_uuid)

        if conversation is None:
            raise ValueError("Conversation not found.")

    conversation_id = str(conversation.id)

    # 2. Load prior turns.
    history = await get_conversation_history(
        session,
        conversation.id,
        limit=10,
    )

    # 3. Reserve the next turn number.
    turn_number = await get_next_turn_number(
        session,
        conversation.id,
    )

    # 4. Save a pending turn before running the graph.
    turn = await save_turn(
        session,
        conversation_id=conversation.id,
        turn_number=turn_number,
        question=question,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        status="pending",
    )
    turn_id = turn.id

    try:
        # 5. Run the existing LangGraph pipeline.
        result = await run_analysis_pipeline(
            dataset_id=dataset_id,
            question=question,
            dataset_version=dataset_version,
            user_overrides=user_overrides,
            conversation_id=conversation_id,
            conversation_history=history,
        )

        status = result.get("status", "failed")

        # 6. Persist the answer and relevant analysis context.
        report = result.get("report") or {}
        answer = (
            report.get("answer")
            or result.get("clarification")
            or "The analysis did not produce an answer."
        )

        analysis_context = {
            "plan": result.get("plan"),
            "verified_results": result.get("verified_results", []),
            "report": report,
            "caveats": result.get("caveats", []),
        }

        turn.answer = answer
        turn.status = status
        turn.dataset_version = (
            result.get("dataset_version") or dataset_version
        )
        turn.analysis_context = analysis_context

        await session.commit()

        return {
            **result,
            "conversation_id": conversation_id,
            "turn_number": turn_number,
        }

    
    except Exception:
        await session.rollback()

        # Use the saved scalar ID, not an ORM attribute after rollback.
        async with session.begin():
            await session.execute(
                update(ConversationTurn)
                .where(ConversationTurn.id == turn_id)
                .values(
                    status="failed",
                    answer="The request failed during processing.",
                )
            )

        raise
