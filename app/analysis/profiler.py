# Returns statistics about the data


"""Generate statistical and semantic profiles for a Pandas DataFrame"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.analysis.semantic_types import (
    infer_semantic_type,
    
)

from app.analysis.helper import (
    _clean_number,
    _numeric_values,
    _datetime_values,
    _top_values
)
    

def _column_profile(
    series: pd.Series,
    column_name: str,
) -> dict[str, Any]:
    """Build a column profile using shared semantic inference."""
    total = len(series)
    missing_count = int(series.isna().sum())
    non_missing = series.dropna()
    unique_count = int(non_missing.nunique())

    # Shared with the quality engine.
    semantic = infer_semantic_type(series, column_name)
    semantic_type = semantic["semantic_type"]

    result: dict[str, Any] = {
        "dtype": str(series.dtype),
        "semantic_type": semantic_type,
        "semantic_type_confidence": semantic["confidence"],
        "inference_reason": semantic["inference_reason"],
        "missing_count": missing_count,
        "missing_percentage": (
            round(missing_count / total * 100, 2)
            if total
            else 0.0
        ),
        "unique_count": unique_count,
        "is_constant": unique_count <= 1,
        "is_all_null": missing_count == total,
    }

    # Numeric columns, including numeric strings.
    # Identifier columns are deliberately excluded.
    if semantic_type == "numeric":
        numeric = _numeric_values(series)

        result["statistics"] = None

        if not numeric.empty:
            q1 = numeric.quantile(0.25)
            median = numeric.quantile(0.50)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1 # Spread of 50% of the data, used for outlier detection.

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            # With IQR == 0 the fences collapse onto q1/q3 and every value
            # that differs from them would be flagged. The rule is not
            # informative there, so report 0 and mark it not applicable
            # (same behaviour as quality.py).
            iqr_rule_applicable = bool(np.isfinite(iqr) and iqr > 0)

            result["statistics"] = {
                "valid_numeric_count": int(len(numeric)),
                "numeric_parse_failure_count": int(
                    len(non_missing) - len(numeric)
                ),
                "mean": _clean_number(numeric.mean()),
                "median": _clean_number(median),
                "std": _clean_number(numeric.std()),
                "min": _clean_number(numeric.min()),
                "max": _clean_number(numeric.max()),
                "q1": _clean_number(q1),
                "q3": _clean_number(q3),
                "iqr": _clean_number(iqr),
                "zero_count": int((numeric == 0).sum()),
                "outlier_count_iqr": (
                    int(((numeric < lower) | (numeric > upper)).sum())
                    if iqr_rule_applicable
                    else 0
                ),
                "outlier_rule_applicable": iqr_rule_applicable,
            }

    elif semantic_type == "boolean":
        normalized = (
            non_missing.astype("string")
            .str.strip()
            .str.casefold()
        )

        result["value_counts"] = {
            str(value): int(count)
            for value, count in normalized.value_counts().items()
        }

    elif semantic_type == "datetime":
        dates = _datetime_values(series)

        result["datetime_statistics"] = None

        if not dates.empty:
            result["datetime_statistics"] = {
                "valid_datetime_count": int(len(dates)),
                "datetime_parse_failure_count": int(
                    len(non_missing) - len(dates)
                ),
                "min": str(dates.min()),
                "max": str(dates.max()),
                "unique_timestamps": int(dates.nunique()),
            }

    else:
        # Categorical, text, identifier, structured, and unknown columns.
        result["top_values"] = _top_values(series)

        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.StringDtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            string_values = series.astype("string")
            result["empty_string_count"] = int(
                (string_values.str.strip() == "").fillna(False).sum()
            )
        else:
            result["empty_string_count"] = 0

    return result


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a compact statistical and semantic dataset profile."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    if df.columns.duplicated().any():
        raise ValueError(
            "DataFrame contains duplicate column names; rename them "
            "before profiling."
        )

    columns = {
        str(column): _column_profile(df[column], str(column))
        for column in df.columns
    }

    semantic_counts = {
        semantic_type: sum(
            profile["semantic_type"] == semantic_type
            for profile in columns.values()
        )
        for semantic_type in (
            "numeric",
            "categorical",
            "datetime",
            "boolean",
            "identifier",
            "text",
            "structured",
            "unknown",
        )
    }

    all_null_columns = [
        name
        for name, profile in columns.items()
        if profile["is_all_null"]
    ]

    constant_columns = [
        name
        for name, profile in columns.items()
        if profile["is_constant"] and not profile["is_all_null"]
    ]

    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())

    return {
        "overview": {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_bytes": int(
                df.memory_usage(index=True, deep=True).sum()
            ),
            "missing_cells": total_missing,
            "missing_percentage": (
                round(total_missing / total_cells * 100, 2)
                if total_cells
                else 0.0
            ),
            "duplicate_rows": int(df.duplicated().sum()),
            **{
                f"{semantic_type}_columns": count
                for semantic_type, count in semantic_counts.items()
            },
        },
        "columns": columns,
        "quality": {
            "empty_rows": int(df.isna().all(axis=1).sum()),
            "constant_columns": constant_columns,
            "all_null_columns": all_null_columns,
        },
        "warnings": [],
    }