
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.conversation import (
    SendMessageRequest,
    SendMessageResponse,
)
from app.db.database import get_db_session
from app.session.conversation_service import run_conversation_turn

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/messages", response_model=SendMessageResponse)
async def send_message(
    request: SendMessageRequest,
    session: AsyncSession = Depends(get_db_session),
) -> SendMessageResponse:
    try:
        result = await run_conversation_turn(
            session,
            question=request.question,
            dataset_id=request.dataset_id,
            dataset_version=request.dataset_version,
            conversation_id=request.conversation_id,
            user_overrides=request.user_overrides,
        )

        return SendMessageResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
