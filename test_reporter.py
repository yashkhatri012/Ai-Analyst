
import asyncio

from app.agents.reporter_agent import (
    ReporterInput,
    generate_report,
)


async def main():
    reporter_input = ReporterInput(
        user_question=(
            "Calculate the correlation between budget and revenue."
        ),
        plan={
            "intent": "correlation",
            "steps": [
                {
                    "id": 1,
                    "goal": "Calculate the correlation between budget and revenue",
                    "columns": ["budget", "revenue"],
                    "method": "Pearson correlation",
                }
            ],
        },
        verified_results=[
            {
                "step_id": 1,
                "status": "pass",
                "result_table": [
                    {"correlation": 0.9996366579014314}
                ],
                "chart_spec": None,
            }
        ],
        caveats=[],
    )

    output = await generate_report(reporter_input)

    print("\n--- Reporter Output ---")
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
