
from pydantic import BaseModel, Field


class SendMessageRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    dataset_id: str = Field(min_length=1)
    dataset_version: str | None = None
    conversation_id: str | None = None
    user_overrides: list[str] = Field(default_factory=list)


class SendMessageResponse(BaseModel):
    conversation_id: str
    turn_number: int
    status: str
    report: dict | None = None
    clarification: str | None = None
    plan: dict | None = None
    verified_results: list = Field(default_factory=list)
    caveats: list = Field(default_factory=list)
