
import asyncio

from app.agents.verifier_agent import (
    VerifierInput,
    verify_analysis,
)


async def main():
    verifier_input = VerifierInput(
        user_question="Calculate the correlation between budget and revenue.",
        plan_step={
            "id": 1,
            "goal": "Calculate the correlation between budget and revenue",
            "columns": ["budget", "revenue"],
            "method": "Pearson correlation",
            "depends_on": [],
        },
        code="""
sub_df = df[['budget', 'revenue']].dropna()
correlation_value = sub_df['budget'].corr(
    sub_df['revenue'],
    method='pearson'
)
result = pd.DataFrame({
    'correlation': [correlation_value]
})
""",
        result_table=[
            {"correlation": 0.9996366579014314}
        ],
        chart_spec=None,
        quality_flags=[],
        execution_error="KeyError: 'revenue'",
    )

    output = await verify_analysis(verifier_input)

    print("\n--- Verifier Output ---")
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
