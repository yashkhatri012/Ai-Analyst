from typing import Any

from app.agents.planner_agent import plan, AnalysisPlan
from app.agents.analyst_agent import AnalystInput, run_analysis
from app.agents.verifier_agent import VerifierInput, verify_analysis
from app.agents.reporter_agent import ReporterInput, generate_report
from app.analysis.context_builder import build_agent_context
from app.storage.dataset_store import load_dataset, get_dataset_metadata
from app.graph.constants import MAX_ATTEMPTS
from app.graph.helpers import build_quality_flags, build_profile_slice, build_zero_assumption_caveat
from app.graph.state import AnalystGraphState


def load_context_node(state: AnalystGraphState) -> dict:
    """Load the selected immutable dataset version and build compact context."""
    dataset_id = state["dataset_id"]
    requested_version = state.get("dataset_version")
    df = load_dataset(dataset_id, version=requested_version)
    dataset_context = build_agent_context(df)
    metadata = get_dataset_metadata(dataset_id)
    actual_version = (
        requested_version
        if requested_version is not None
        else metadata["latest_version"]
    )

    return {
        "df": df,
        "dataset_context": dataset_context,
        "actual_version": actual_version,
        "step_index": 0,
        "attempt": 0,
        "verified_results": [],
        "all_caveats": [],
        "step_flags": {},
        "status": "context_loaded",
    }


async def planner_node(state: AnalystGraphState) -> dict:
    result: AnalysisPlan = await plan(
        dataset_id=state["dataset_id"],
        question=state["question"],
        dataset_context=state["dataset_context"],
    )
    return {
        "plan": result.model_dump(),
        "answerable": result.answerable,
        "clarification": result.clarification,
        "status": "planned" if result.answerable else "unanswerable",
    }


def prepare_step_node(state: AnalystGraphState) -> dict:
    """Validate and prepare one plan step for the Analyst."""
    steps = state["plan"].get("steps", [])
    index = state.get("step_index", 0)

    if index >= len(steps):
        return {"status": "steps_exhausted"}

    step = steps[index]
    columns = step.get("columns", [])
    flags = build_quality_flags(state["dataset_context"], columns)

    all_step_flags = dict(state.get("step_flags", {}))
    all_step_flags[str(step["id"])] = flags

    df = state["df"]
    missing_columns = [column for column in columns if column not in df.columns]

    if missing_columns:
        return {
            "current_step": step,
            "current_flags": flags,
            "step_flags": all_step_flags,
            "missing_columns": missing_columns,
            "status": "step_missing_columns",
        }

    profile_slice = build_profile_slice(state["dataset_context"], columns)
    profile_slice["quality_flags"] = flags

    previous_results = {
        result["step_id"]: result
        for result in state.get("verified_results", [])
    }
    dependency_results = {
        dependency_id: previous_results[dependency_id]
        for dependency_id in step.get("depends_on", [])
        if dependency_id in previous_results
    }

    # Keep this aligned with AnalystInput in app/agents/analyst_agent.py.
    analyst_input = AnalystInput(
        step=step,
        profile_slice=profile_slice,
        sample_rows=df.loc[:, columns].head(5).to_dict(orient="records"),
        dependency_results=dependency_results,
        user_overrides=state.get("user_overrides", []),
    )

    return {
        "current_step": step,
        "current_flags": flags,
        "step_flags": all_step_flags,
        "missing_columns": [],
        "current_analyst_input": analyst_input.model_dump(),
        "attempt": 0,
        "status": "step_ready",
    }


async def analyst_node(state: AnalystGraphState) -> dict:
    analyst_input = AnalystInput.model_validate(state["current_analyst_input"])
    output = await run_analysis(analyst_input, state["df"])
    return {
        "analyst_output": output.model_dump(),
        "attempt": state.get("attempt", 0) + 1,
        "status": "analyst_ran",
    }


async def verifier_node(state: AnalystGraphState) -> dict:
    step = state["current_step"]
    analyst = state["analyst_output"]

    verifier_input = VerifierInput(
        user_question=state["question"],
        plan_step=step,
        code=analyst.get("code", ""),
        result_table=analyst.get("result_table", []),
        chart_spec=analyst.get("chart_spec"),
        quality_flags=state.get("current_flags", []),
        execution_error=(
            analyst.get("error") if analyst.get("status") == "error" else None
        ),
    )
    output = await verify_analysis(verifier_input)

    analyst_input = dict(state["current_analyst_input"])
    if output.status == "retry":
        analyst_input["previous_code"] = analyst.get("code", "")
        analyst_input["previous_error"] = (
            analyst.get("error") or "; ".join(output.issues)
        )
        analyst_input["retry_instructions"] = output.retry_instructions

    return {
        "verifier_output": output.model_dump(),
        "current_analyst_input": analyst_input,
        "status": f"verified_{output.status}",
    }


def record_step_node(state: AnalystGraphState) -> dict:
    step = state["current_step"]
    analyst = state.get("analyst_output", {})
    verifier = state.get("verifier_output", {})
    caveats = list(state.get("all_caveats", []))
    results = list(state.get("verified_results", []))

    if verifier.get("status") == "retry":
        caveats.append(f"Step {step['id']} failed after {MAX_ATTEMPTS} attempts.")
    else:
        if verifier.get("status") == "caveat":
            caveats.extend(verifier.get("caveats", []))

        results.append({
            "step_id": step["id"],
            "status": verifier["status"],
            "result_table": analyst.get("result_table", []),
            "chart_spec": analyst.get("chart_spec"),
            "verification_issues": verifier.get("issues", []),
            "caveats": verifier.get("caveats", []),
        })

    return {
        "verified_results": results,
        "all_caveats": caveats,
        "step_index": state.get("step_index", 0) + 1,
        "status": "step_recorded",
    }


def skip_step_node(state: AnalystGraphState) -> dict:
    step = state["current_step"]
    caveats = list(state.get("all_caveats", []))
    caveats.append(
        f"Step {step['id']} references unavailable columns: "
        f"{state.get('missing_columns', [])}"
    )
    return {
        "all_caveats": caveats,
        "step_index": state.get("step_index", 0) + 1,
        "status": "step_skipped",
    }


def finish_unanswerable_node(state: AnalystGraphState) -> dict:
    return {"status": "unanswerable"}


def finish_empty_plan_node(state: AnalystGraphState) -> dict:
    caveats = list(state.get("all_caveats", []))
    caveats.append("The Planner returned no analysis steps.")
    return {"status": "failed", "all_caveats": caveats}


def finish_analysis_node(state: AnalystGraphState) -> dict:
    results = state.get("verified_results", [])
    caveats = list(state.get("all_caveats", []))

    if not results:
        return {"status": "failed", "all_caveats": caveats}

    assumption_caveat = build_zero_assumption_caveat(
        state.get("step_flags", {}),
        results,
    )
    if assumption_caveat and assumption_caveat not in caveats:
        caveats.append(assumption_caveat)

    return {"all_caveats": caveats, "status": "ready_to_report"}


async def reporter_node(state: AnalystGraphState) -> dict:
    results = state.get("verified_results", [])
    if not results:
        return {"status": "failed"}

    reporter_input = ReporterInput(
        user_question=state["question"],
        plan=state["plan"],
        verified_results=[
            result for result in results
            if result["status"] in ("pass", "caveat")
        ],
        caveats=list(state.get("all_caveats", [])),
    )
    report = await generate_report(reporter_input)

    return {
        "final_report": report.model_dump(),
        "status": "completed",
    }
