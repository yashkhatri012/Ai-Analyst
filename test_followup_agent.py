
from unittest.mock import AsyncMock, patch



from app.agents.followup_agent import (
    FollowUpDecision,
    resolve_followup,
)



async def test_followup_can_answer_from_previous_results():
    decision = FollowUpDecision(
        route="context",
        standalone_question="Which correlation was strongest?",
        context_answer="Budget and revenue had the strongest correlation.",
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = decision

    with patch(
        "app.agents.followup_agent.get_llm_for_agent",
        return_value=type(
            "MockLLM",
            (),
            {
                "with_structured_output": lambda self, schema: mock_llm,
            },
        )(),
    ):
        result = await resolve_followup(
            question="Which was strongest?",
            conversation_history=[
                {
                    "question": "Analyze budget and revenue.",
                    "analysis_context": {
                        "verified_results": [
                            {"correlation": 0.82}
                        ]
                    },
                }
            ],
            dataset_id="dataset-1",
            dataset_version="v1",
        )

    assert result.route == "context"
    assert result.context_answer
