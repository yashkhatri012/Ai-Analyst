
from typing import Any

import pandas as pd

from app.agents.planner_agent import plan, AnalysisPlan
from app.agents.analyst_agent import (
    AnalystInput,
    run_analysis,
)
from app.agents.verifier_agent import (
    VerifierInput,
    verify_analysis,
)
from app.agents.reporter_agent import (
    ReporterInput,
    generate_report,
)
from app.analysis.context_builder import build_agent_context
from app.storage.dataset_store import (
    load_dataset,
    get_dataset_metadata,
)



def build_profile_slice(
    dataset_context: dict,
    columns: list[str],
) -> dict:
    profiles = dataset_context.get("columns", {})
    selected = {}

    for column in columns:
        if column in profiles:
            selected[column] = profiles[column]
            continue

        other_columns = dataset_context.get("other_columns", {})
        compact_columns = other_columns.get("columns", {})
        compact_fields = other_columns.get("fields", [])

        if column in compact_columns:
            selected[column] = dict(
                zip(compact_fields, compact_columns[column])
            )

    return {
        "columns": selected,
        "notes": (
            "Column profiles may be compacted. "
            "Use only the supplied information."
        ),
    }

async def run_analysis_pipeline(
    dataset_id: str,
    question: str,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    # 1. Load the immutable dataset version.
    df = load_dataset(dataset_id, version=dataset_version)

    # 2. Build compact context for the Planner.
    dataset_context = build_agent_context(df)

    # 3. Ask the Planner to create an analysis plan.
    analysis_plan: AnalysisPlan = await plan(
        dataset_id=dataset_id,
        question=question,
        dataset_context=dataset_context,
    )

    # Stop if the question cannot be answered.
    if not analysis_plan.answerable:
        return {
            "status": "unanswerable",
            "clarification": analysis_plan.clarification,
            "plan": analysis_plan.model_dump(),
        }

    
    verified_results = []
    all_caveats = []
    
    metadata = get_dataset_metadata(dataset_id)

    actual_version = (
        dataset_version
        if dataset_version is not None
        else metadata["latest_version"]
    )


    for step in analysis_plan.steps:
        profile_slice = build_profile_slice(
            dataset_context,
            step.columns,
        )

            
        # Validate planned columns before running generated code.
        missing_columns = [
            column
            for column in step.columns
            if column not in df.columns
        ]

        if missing_columns:
            all_caveats.append(
                f"Step {step.id} references unavailable columns: "
                f"{missing_columns}"
            )
            continue

        analyst_input = AnalystInput(
            dataset_id=dataset_id,
            dataset_version=actual_version,
            step=step.model_dump(),
            profile_slice=profile_slice,
            sample_rows=df.loc[:, step.columns]
            .head(5)
            .to_dict(orient="records"),
        )


        verifier_output = None
        analyst_output = None

        for attempt in range(3):
            analyst_output = await run_analysis(analyst_input, df)

            verifier_input = VerifierInput(
                user_question=question,
                plan_step=step.model_dump(),
                code=analyst_output.code,
                result_table=analyst_output.result_table,
                chart_spec=analyst_output.chart_spec,
                quality_flags=[],
                execution_error=analyst_output.error
                if analyst_output.status == "error"
                else None,
            )

            verifier_output = await verify_analysis(verifier_input)

            if verifier_output.status != "retry":
                break

            if attempt < 2:
                analyst_input.previous_error = (
                    analyst_output.error
                    or "; ".join(verifier_output.issues)
                )
                analyst_input.retry_instructions = (
                    verifier_output.retry_instructions
                )

        if (
            analyst_output is None
            or verifier_output is None
            or verifier_output.status == "retry"
        ):
            all_caveats.append(
                f"Step {step.id} failed after three attempts."
            )
            continue

        if verifier_output.status == "caveat":
            all_caveats.extend(verifier_output.caveats)
            all_caveats.extend(verifier_output.issues)

        verified_results.append({
            "step_id": step.id,
            "status": verifier_output.status,
            "result_table": analyst_output.result_table,
            "chart_spec": analyst_output.chart_spec,
            "verification_issues": verifier_output.issues,
            "caveats": verifier_output.caveats,
        })

    if not verified_results:
        return {
            "status": "failed",
            "plan": analysis_plan.model_dump(),
            "caveats": all_caveats,
        }

    reporter_input = ReporterInput(
        user_question=question,
        plan=analysis_plan.model_dump(),
        verified_results=[
            result
            for result in verified_results
            if result["status"] == "pass"
        ],
        caveats=all_caveats,
    )

    report = await generate_report(reporter_input)

    return {
        "status": "completed",
        "plan": analysis_plan.model_dump(),
        "verified_results": verified_results,
        "report": report.model_dump(),
        "caveats": all_caveats,
    }

   