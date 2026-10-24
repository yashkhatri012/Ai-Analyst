
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


SANDBOX_IMAGE = "ai-analyst-sandbox"
TIMEOUT_SECONDS = 30


def execute_analysis_code(
    code: str,
    df: pd.DataFrame,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="ai_analyst_") as tmp:
        root = Path(tmp)
        input_dir = root / "input"
        data_dir = root / "data"

        input_dir.mkdir()
        data_dir.mkdir()

        # Save the generated code in a request file.
        request_path = input_dir / "request.json"
        request_path.write_text(
            json.dumps({"code": code}),
            encoding="utf-8",
        )

        # Save a temporary copy of the dataset.
        dataset_path = data_dir / "dataset.parquet"
        df.to_parquet(dataset_path, index=False)

        command = [
            "docker", "run",
            "--rm",
            "--network", "none",
            "--read-only",
            "--user", "10001:10001",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", "512m",
            "--cpus", "1",
            "--pids-limit", "64",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m",
            "--mount",
            f"type=bind,source={input_dir},target=/input,readonly",
            "--mount",
            f"type=bind,source={data_dir},target=/data,readonly",
            SANDBOX_IMAGE,
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "result_table": [],
                "chart_spec": None,
                "error": "Analysis timed out.",
            }

        if completed.returncode != 0:
            return {
                "status": "error",
                "result_table": [],
                "chart_spec": None,
                "error": completed.stderr[:5000]
                or f"Docker exited with code {completed.returncode}",
            }

        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "result_table": [],
                "chart_spec": None,
                "error": "Sandbox returned invalid JSON.",
            }

        return result
