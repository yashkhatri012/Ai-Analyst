# This file contains functions for interacting with the database, 


import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, ConversationTurn


async def create_conversation(
    session: AsyncSession,
) -> Conversation:
    conversation = Conversation()
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def get_conversation(
    session: AsyncSession,
    conversation_id: uuid.UUID,
) -> Conversation | None:
    return await session.get(Conversation, conversation_id)


async def get_next_turn_number(
    session: AsyncSession,
    conversation_id: uuid.UUID,
) -> int:
    result = await session.execute(
        select(func.max(ConversationTurn.turn_number)).where(
            ConversationTurn.conversation_id == conversation_id
        )
    )
    return (result.scalar_one() or 0) + 1


async def save_turn(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    turn_number: int,
    question: str,
    dataset_id: str,
    dataset_version: str | None = None,
    answer: str | None = None,
    status: str = "pending",
    analysis_context: dict[str, Any] | None = None,
) -> ConversationTurn:
    turn = ConversationTurn(
        conversation_id=conversation_id,
        turn_number=turn_number,
        question=question,
        answer=answer,
        status=status,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        analysis_context=analysis_context,
    )

    session.add(turn)
    await session.commit()
    await session.refresh(turn)
    return turn


async def get_conversation_history(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    result = await session.execute(
        select(ConversationTurn)
        .where(
            ConversationTurn.conversation_id == conversation_id,
            ConversationTurn.status == "completed",
        )
        .order_by(ConversationTurn.turn_number.desc())
        .limit(limit)
    )

    turns = list(reversed(result.scalars().all()))

    return [
        {
            "turn_number": turn.turn_number,
            "question": turn.question,
            "answer": turn.answer,
            "dataset_id": turn.dataset_id,
            "dataset_version": turn.dataset_version,
            "analysis_context": turn.analysis_context,
        }
        for turn in turns
    ]
