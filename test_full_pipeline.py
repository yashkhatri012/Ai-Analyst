
import asyncio

from app.agents.orchestrator import run_analysis_pipeline


async def main():
    dataset_id = input("Dataset ID: ").strip()
    question = input("Analysis question: ").strip()

    result = await run_analysis_pipeline(
        dataset_id=dataset_id,
        question=question,
    )

    import json
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
