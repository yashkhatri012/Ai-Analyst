
import json
import re
import pandas as pd

from app.storage.dataset_store import load_dataset, get_dataset_metadata
from app.analysis.profiler import profile_dataset

MAX_ROWS = 20
AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count", "nunique"}


def _safe(value, limit=200):
    """Sanitize dataset-derived text before returning it to an LLM."""
    if not isinstance(value, str):
        return value

    value = re.sub(r"[\x00-\x1f\x7f-\x9f]+", " ", value).strip()

    if re.search(
        r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above)"
        r"|system\s+prompt|new\s+instructions|you are now",
        value,
        re.I,
    ):
        return "<redacted: instruction-like text>"

    value = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "<email>", value)
    value = re.sub(r"https?://\S+|www\.\S+", "<url>", value, flags=re.I)
    value = re.sub(r"(?<!\w)\+?\d[\d\s()-]{8,}\d(?!\w)", "<phone>", value)

    return value[:limit] + ("…" if len(value) > limit else "")


def _clean(value):
    """Recursively make results JSON-safe and sanitize strings."""
    if isinstance(value, dict):
        return {str(_safe(k)): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if value is None:
        return None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        return _safe(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    return value


def _load(dataset_id, version):
    df = load_dataset(dataset_id, version)
    metadata = get_dataset_metadata(dataset_id)
    actual_version = version or metadata["latest_version"]
    return df, actual_version


def _result(tool, dataset_id, version, args, data, rows=None, total_rows=0):
    """Consistent result envelope with provenance and bounded table output."""
    return {
        "status": "success",
        "stdout": json.dumps(_clean(data), default=str),
        "tables": [{
            "rows": _clean((rows or [])[:MAX_ROWS]),
            "total_rows": total_rows,
        }] if rows is not None else [],
        "figures": [],
        "error": None,
        "provenance": {
            "tool": tool,
            "dataset_id": dataset_id,
            "dataset_version": version,
            "arguments": _clean(args),
        },
    }


def _filter(df, filters):
    for f in filters or []:
        col, op, value = f["column"], f["op"], f.get("value")
        if col not in df.columns:
            raise ValueError(f"Unknown column: {col}")

        s = df[col]
        if op == "eq":
            mask = s == value
        elif op == "ne":
            mask = s != value
        elif op in {"gt", "ge", "lt", "le"}:
            mask = {"gt": s.gt, "ge": s.ge, "lt": s.lt, "le": s.le}[op](value)
        elif op == "in":
            if not isinstance(value, list):
                raise ValueError("'in' requires a list value")
            mask = s.isin(value)
        elif op == "contains":
            mask = s.astype("string").str.contains(
                str(value), case=False, na=False, regex=False
            )
        elif op == "is_null":
            mask = s.isna()
        elif op == "not_null":
            mask = s.notna()
        else:
            raise ValueError(f"Unsupported filter operator: {op}")

        df = df.loc[mask.fillna(False)]
    return df


def _records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


def describe_columns(dataset_id, columns, version=None):
    df, actual = _load(dataset_id, version)
    if not columns or len(columns) > MAX_ROWS:
        raise ValueError("Provide 1–20 column names")
    unknown = set(columns) - set(df.columns)
    if unknown:
        raise ValueError(f"Unknown columns: {sorted(unknown)}")

    profile = profile_dataset(df)
    details = {c: profile["columns"][c] for c in columns}
    return _result(
        "describe_columns", dataset_id, actual, {"columns": columns}, details
    )


def sample_rows(dataset_id, columns=None, filters=None, n=10, version=None):
    if not 1 <= n <= MAX_ROWS:
        raise ValueError(f"n must be between 1 and {MAX_ROWS}")

    df, actual = _load(dataset_id, version)
    df = _filter(df, filters)  # Filter before selecting output columns.

    if columns:
        unknown = set(columns) - set(df.columns)
        if unknown:
            raise ValueError(f"Unknown columns: {sorted(unknown)}")
        df = df.loc[:, columns]

    rows = _records(df.head(n))
    return _result(
        "sample_rows", dataset_id, actual,
        {"columns": columns, "filters": filters, "n": n},
        {"rows_returned": len(rows), "total_matching_rows": len(df)},
        rows, len(df),
    )


def value_counts(dataset_id, column, top_n=10, version=None):
    if not 1 <= top_n <= MAX_ROWS:
        raise ValueError(f"top_n must be between 1 and {MAX_ROWS}")

    df, actual = _load(dataset_id, version)
    if column not in df.columns:
        raise ValueError(f"Unknown column: {column}")

    counts = df[column].value_counts(dropna=False)
    rows = [
        {"value": _clean(value), "count": int(count)}
        for value, count in counts.head(top_n).items()
    ]
    return _result(
        "value_counts", dataset_id, actual,
        {"column": column, "top_n": top_n},
        {"column": column, "distinct_values": len(counts)},
        rows, len(counts),
    )


def missingness_report(dataset_id, version=None):
    df, actual = _load(dataset_id, version)
    missing = df.isna()
    columns = [{
        "column": c,
        "missing_count": int(missing[c].sum()),
        "missing_percentage": round(float(missing[c].mean() * 100), 2),
    } for c in df.columns]

    co_missing = []
    cols = list(df.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = int((missing[a] & missing[b]).sum())
            a_missing = int(missing[a].sum())
            b_missing = int(missing[b].sum())

            if both and (both == a_missing or both == b_missing):
                co_missing.append({
                    "column_a": a,
                    "column_b": b,
                    "both_missing": both,
                    "note": "Every missing value in at least one column overlaps with the other",
                })
                if len(co_missing) >= MAX_ROWS:
                    break
        if len(co_missing) >= MAX_ROWS:
            break

    return _result(
        "missingness_report", dataset_id, actual, {},
        {
            "row_count": len(df),
            "columns": columns,
            "co_missing_patterns": co_missing,
        },
    )


def query_data(
    dataset_id, filters=None, group_by=None, aggregations=None,
    sort_by=None, ascending=True, limit=20, version=None,
):
    if not 1 <= limit <= MAX_ROWS:
        raise ValueError(f"limit must be between 1 and {MAX_ROWS}")

    df, actual = _load(dataset_id, version)
    df = _filter(df, filters)
    group_by, aggregations = group_by or [], aggregations or {}

    unknown = (set(group_by) | set(aggregations)) - set(df.columns)
    if unknown:
        raise ValueError(f"Unknown columns: {sorted(unknown)}")
    if any(func not in AGGREGATIONS for func in aggregations.values()):
        raise ValueError(f"Allowed aggregations: {sorted(AGGREGATIONS)}")
    if group_by and not aggregations:
        raise ValueError("Grouped queries require at least one aggregation")

    if group_by:
        result = df.groupby(group_by, dropna=False).agg(aggregations).reset_index()
    elif aggregations:
        result = pd.DataFrame([{
            col: df[col].agg(func) for col, func in aggregations.items()
        }])
    else:
        result = df

    if sort_by:
        if sort_by not in result.columns:
            raise ValueError(f"Unknown sort column: {sort_by}")
        result = result.sort_values(sort_by, ascending=ascending)

    rows = _records(result.head(limit))
    return _result(
        "query_data", dataset_id, actual,
        {
            "filters": filters, "group_by": group_by,
            "aggregations": aggregations, "sort_by": sort_by,
            "ascending": ascending, "limit": limit,
        },
        {"rows_returned": len(rows), "total_result_rows": len(result)},
        rows, len(result),
    )
