
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.llm.prompts.analyst_prompt import ANALYST_SYSTEM_PROMPT

import json
from app.sandbox.docker_runner import execute_analysis_code

from langchain_core.messages import SystemMessage, HumanMessage
from app.llm.service import get_llm_for_agent
from app.analysis.compact import compact 

MAX_ROWS = 20
MAX_STDOUT = 500


class AnalystInput(BaseModel):
    step: dict[str, Any]
    profile_slice: dict[str, Any]
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    dependency_results: dict[int, Any] = Field(default_factory=dict)
    previous_code: str | None = None
    previous_error: str | None = None
    retry_instructions: str | None = None


class AnalystOutput(BaseModel):
    """Result of executing one analysis step."""

    step_id: int

    status: Literal["success", "error"]

    code: str = ""
    result_table: list[dict[str, Any]] = Field(default_factory=list)
    chart_spec: dict[str, Any] | None = None

    stdout: str = ""
    error: str | None = None




async def generate_analysis_code(analyst_input: AnalystInput) -> str:
    import json

    from langchain_core.messages import SystemMessage, HumanMessage
    from app.llm.service import get_llm_for_agent

    llm = get_llm_for_agent("analyst")

    prompt_data = analyst_input.model_dump(exclude_none=True)
    prompt_data["step"] = {k: analyst_input.step[k] for k in ("goal", "columns", "method")}
    content = json.dumps(prompt_data, ensure_ascii=False, separators=(",", ":"), default=str)

    
    response = await llm.ainvoke([
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(
            content=content
        ),
    ])

    content = response.content

    # Gemini may return content as a list of blocks.
    if isinstance(content, str):
        code = content
    elif isinstance(content, list):
        text_blocks = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        code = "\n".join(text_blocks)
    else:
        raise ValueError(
            f"Unexpected LLM content type: {type(content).__name__}"
        )

    code = code.strip()

    if not code:
        raise ValueError("The Analyst returned empty code.")

    # Remove Markdown fences if the model includes them.
    if code.startswith("```"):
        lines = code.splitlines()
        if lines[-1].strip() == "```":
            code = "\n".join(lines[1:-1]).strip()

    return code



async def run_analysis(analyst_input: AnalystInput, df) -> AnalystOutput:
    step_id = analyst_input.step["id"]

    try:
        # Generate code using the Analyst LLM.
        code = await generate_analysis_code(analyst_input)

        # Execute the generated code inside Docker.
        execution = execute_analysis_code(code, df)

        # Convert the sandbox response to our output contract.
        table = [compact(r) for r in (execution.get("result_table") or [])]
        return AnalystOutput(
            step_id=step_id,
            status=execution["status"],
            code=code,
            result_table=table[:MAX_ROWS],
            total_rows=len(table),
            chart_spec=execution.get("chart_spec"),
            stdout=execution.get("stdout", "")[:MAX_STDOUT],
            error=execution.get("error"),
        )

    except Exception as exc:
        return AnalystOutput(
            step_id=step_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
