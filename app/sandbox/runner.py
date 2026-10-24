
import json
import numpy as np
import pandas as pd


def make_json_safe(value):
    if isinstance(value, pd.DataFrame):
        return [
            {str(k): make_json_safe(v) for k, v in row.items()}
            for row in value.head(200).to_dict(orient="records")
        ]

    if isinstance(value, pd.Series):
        return {
            str(k): make_json_safe(v)
            for k, v in value.head(200).to_dict().items()
        }

    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if value is None or isinstance(value, (str, int, bool)):
        return value

    return str(value)

def main():
    # 1. Load the execution request
    with open("/input/request.json", encoding="utf-8") as f:
        request = json.load(f)

    # 2. Load the dataset
    df = pd.read_parquet("/data/dataset.parquet")

    # 3. Prepare the execution environment
    namespace = {
        "pd": pd,
        "np": np,
        "df": df,
        "result": None,
        "chart_spec": None,
    }

    # 4. Execute the Analyst's generated code
    try:
        code = request["code"]
        exec(compile(code, "<analyst_code>", "exec"), namespace)

        # 5. Collect the output
        result = namespace["result"]
        chart_spec = namespace["chart_spec"]

        # Convert a DataFrame into a list of dictionaries
        if isinstance(result, pd.DataFrame):
            result = result.head(200).to_dict(orient="records")

        # Convert a Series into a dictionary
        elif isinstance(result, pd.Series):
            result = result.head(200).to_dict()

        output = {
            "status": "success",
            "result_table": make_json_safe(result),
            "chart_spec": make_json_safe(chart_spec),
            "error": None,
        }

    except Exception as exc:
        output = {
            "status": "error",
            "result_table": [],
            "chart_spec": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    print(json.dumps(output, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
