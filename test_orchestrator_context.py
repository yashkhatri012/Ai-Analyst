
from app.storage.dataset_store import (
    get_dataset_metadata,
    load_dataset,
)
from app.analysis.context_builder import build_agent_context
from app.agents.planner_agent import plan


async def main():
    dataset_id = input("Enter your dataset ID: ").strip()

    metadata = get_dataset_metadata(dataset_id)
    version = metadata["latest_version"]
    df = load_dataset(dataset_id, version=version)
    context = build_agent_context(df)

    print("\nOriginal columns:")
    print(list(df.columns))

    print("\nContext column keys:")
    print(list(context.get("columns", {}).keys()))

    question = input("\nEnter an analysis question: ").strip()
    analysis_plan = await plan(
        dataset_id=dataset_id,
        question=question,
        dataset_context=context,
    )

    print("\nPlanner output:")
    print(analysis_plan.model_dump_json(indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
