"""Infer semantic types without modifying the source DataFrame"""

from __future__ import annotations

import pandas as pd
from typing import Any
from app.analysis.helper import (
    
    normalize_name,
    _sample,
    _result,
    parse_numeric,
    parse_datetimes,
    IDENTIFIER_NAME,
    BOOLEAN_NAME,
    DATE_PATTERN,
    
    NUMERIC_PARSE_THRESHOLD,
    DATE_PARSE_THRESHOLD,
    CATEGORICAL_UNIQUE_RATIO,
    DATE_NAME,
    MAX_CATEGORY_LENGTH
    
)



def infer_semantic_type(
    series: pd.Series,
    column_name: str | None = None,
) -> dict[str, Any]:
    """
    Infer a column's semantic type.

    Returns semantic_type, confidence, and inference_reason.
    It does not modify or convert the original series, just returns the above _result values.
    """
    name = normalize_name(str(column_name or series.name or ""))
    values = _sample(series)

    if values.empty:
        return _result("unknown", 0.0, "Column has no non null values")

    # Native types take precedence over string based inference.
    # Native types areint64, float64, bool, or datetime64
    if pd.api.types.is_bool_dtype(series):
        return _result("boolean", 1.0, "Native boolean dtype")

    if pd.api.types.is_datetime64_any_dtype(series):
        return _result("datetime", 1.0, "Native datetime dtype")

    if pd.api.types.is_numeric_dtype(series):
        if IDENTIFIER_NAME.search(name):
            return _result(
                "identifier", 0.98,
                "Numeric dtype and identifier like column name (id, uuid, key)",
            )
        return _result("numeric", 1.0, "Native numeric dtype")

    strings = values.astype("string").str.strip()
    strings = strings[strings.ne("")].dropna()

    if strings.empty:
        return _result("unknown", 0.0, "Only blank or null values")

    # Serialized JSON objects and arrays should not be treated as free text.
    structured_ratio = strings.str.startswith(("{", "[")).mean()
    if structured_ratio >= 0.80:
        return _result(
            "structured", 0.95,
            "Most sampled values look like serialized objects or arrays",
        )

    # Explicit boolean words are stronger evidence than a 0/1 pattern.
    normalized = strings.str.casefold()
    boolean_words = {
        "true", "false", "yes", "no", "y", "n", "on", "off"
    }
    if normalized.isin(boolean_words).mean() >= 0.95:
        return _result(
            "boolean", 0.95,
            "Values consistently match common boolean representations",
        )

    # Numeric strings, including negative and decimal values.
    numeric_values = parse_numeric(strings)
    numeric_ratio = float(numeric_values.notna().mean())

    if numeric_ratio >= NUMERIC_PARSE_THRESHOLD:
        if IDENTIFIER_NAME.search(name):
            return _result(
                "identifier", 0.95,
                "Numeric looking strings and identifier-like column name",
            )

        if (
            BOOLEAN_NAME.search(name)
            and set(normalized.unique()).issubset({"0", "1"})
        ):
            return _result(
                "boolean", 0.85,
                "Binary 0/1 values and boolean like column name",
            )

        return _result(
            "numeric", min(0.95, numeric_ratio),
            "Most sampled strings can be parsed as numbers",
        )

    # Date parsing is attempted only when values look date like or the
    # column name suggests a date, reducing accidental numeric conversions.
    date_like_ratio = float(
        strings.str.match(DATE_PATTERN).mean()
    )
    date_hint = bool(DATE_NAME.search(name))

    if date_like_ratio >= DATE_PARSE_THRESHOLD or date_hint:
        parsed_dates = parse_datetimes(strings)
        date_ratio = float(parsed_dates.notna().mean())

        if date_ratio >= DATE_PARSE_THRESHOLD:
            return _result(
                "datetime", date_ratio,
                "Most sampled values parse as dates or timestamps",
            )

    unique_ratio = strings.nunique() / len(strings)
    average_length = float(strings.str.len().mean())

    if (
        unique_ratio <= CATEGORICAL_UNIQUE_RATIO
        and average_length <= MAX_CATEGORY_LENGTH
    ):
        return _result(
            "categorical", 0.85,
            "Low unique value ratio and relatively short values",
        )

    if IDENTIFIER_NAME.search(name):
        return _result(
            "identifier", 0.80,
            "Identifier like column name with non numeric values",
        )

    return _result(
        "text", 0.80,
        "Values appear to be free text or high cardinality strings",
    )