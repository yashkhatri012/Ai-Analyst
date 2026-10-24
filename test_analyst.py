
import asyncio
import pandas as pd

from app.agents.analyst_agent import (
    AnalystInput,
    run_analysis,
)


async def main():
    # Sample dataset for testing.
    df = pd.DataFrame({
        "budget": [100, 200, 300, 400, 500],
        "revenue": [150, 260, 390, 510, 640],
        "genre": ["Drama", "Comedy", "Drama", "Action", "Comedy"],
    })

    # One step produced by the Planner.
    analyst_input = AnalystInput(
        dataset_id="test_dataset",
        dataset_version=1,
        step={
            "id": 1,
            "goal": "Calculate the correlation between budget and revenue",
            "columns": ["budget", "revenue"],
            "method": "Pearson correlation",
            "depends_on": [],
        },
        profile_slice={
            "columns": ["budget", "revenue"],
            "notes": "Both columns contain numeric values.",
        },
        sample_rows=df.head(5).to_dict(orient="records"),
    )

    output = await run_analysis(analyst_input, df)

    print("\n--- Analyst Output ---")
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
