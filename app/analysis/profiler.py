# Returns statistics about the data


from typing import Any

import numpy as np
import pandas as pd


TOP_N = 3


def _clean_number(value: Any) -> Any:
    """Convert NumPy scalars and non finite numbers to JSON safe values"""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


def _column_profile(series: pd.Series) -> dict:
    total = len(series)
    missing = int(series.isna().sum())
    non_missing = series.dropna()
    unique_count = int(non_missing.nunique())

    result = {
        "dtype": str(series.dtype),
        "missing_count": missing,
        "missing_percentage": (
            round(missing / total * 100, 2) if total else 0.0
        ),
        "unique_count": unique_count,
        "is_constant": unique_count <= 1,
        "is_all_null": missing == total,
    }

    # Numeric columns, excluding boolean columns
    if pd.api.types.is_numeric_dtype(series) and not (
        pd.api.types.is_bool_dtype(series)
    ):
        result["semantic_type"] = "numeric"

        if not non_missing.empty:
            q1 = non_missing.quantile(0.25)
            median = non_missing.quantile(0.50)
            q3 = non_missing.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            result["statistics"] = {
                "mean": _clean_number(non_missing.mean()),
                "median": _clean_number(median),
                "std": _clean_number(non_missing.std()),
                "min": _clean_number(non_missing.min()),
                "max": _clean_number(non_missing.max()),
                "q1": _clean_number(q1),
                "q3": _clean_number(q3),
                "iqr": _clean_number(iqr),
                "zero_count": int((non_missing == 0).sum()),
                "outlier_count_iqr": int(
                    ((non_missing < lower) | (non_missing > upper)).sum()
                ),
            }
        else:
            result["statistics"] = None

    elif pd.api.types.is_bool_dtype(series):
        result["semantic_type"] = "boolean"
        result["value_counts"] = {
            str(key): int(value)
            for key, value in non_missing.value_counts().items()
        }

    elif pd.api.types.is_datetime64_any_dtype(series):
        result["semantic_type"] = "datetime"

        if not non_missing.empty:
            result["datetime_statistics"] = {
                "min": str(non_missing.min()),
                "max": str(non_missing.max()),
                "unique_timestamps": unique_count,
            }
        else:
            result["datetime_statistics"] = None

    else:
        # semantic classification can be
        # improved in a dedicated schema inference module
        avg_length = (
            non_missing.astype(str).str.len().mean()
            if not non_missing.empty
            else 0
        )

        result["semantic_type"] = (
            "text" if avg_length > 50 else "categorical"
        )

        result["top_values"] = [
            {
                "value": str(value),
                "count": int(count),
            }
            for value, count in non_missing.value_counts(
                dropna=True
            ).head(TOP_N).items()
        ]

        result["empty_string_count"] = int(
            (series.astype("string").str.strip() == "").sum()
        )

    return result


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a compact statistical profile of a DataFrame"""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    columns = {
        str(column): _column_profile(df[column])
        for column in df.columns
    }

    numeric_count = sum(
        1 for profile in columns.values()
        if profile["semantic_type"] == "numeric"
    )

    categorical_count = sum(
        1 for profile in columns.values()
        if profile["semantic_type"] == "categorical"
    )

    datetime_count = sum(
        1 for profile in columns.values()
        if profile["semantic_type"] == "datetime"
    )

    boolean_count = sum(
        1 for profile in columns.values()
        if profile["semantic_type"] == "boolean"
    )

    all_null_columns = [
        name for name, profile in columns.items()
        if profile["is_all_null"]
    ]

    constant_columns = [
        name for name, profile in columns.items()
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
                if total_cells else 0.0
            ),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": numeric_count,
            "categorical_columns": categorical_count,
            "datetime_columns": datetime_count,
            "boolean_columns": boolean_count,
        },
        "columns": columns,
        "quality": {
            "empty_rows": int(df.isna().all(axis=1).sum()),
            "constant_columns": constant_columns,
            "all_null_columns": all_null_columns,
        },
        "warnings": [],
    }
