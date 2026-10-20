
"""Infer semantic types without modifying the source DataFrame"""

import re
from typing import Any

import pandas as pd


SAMPLE_SIZE = 1000
NUMERIC_PARSE_THRESHOLD = 0.95
DATE_PARSE_THRESHOLD = 0.80
CATEGORICAL_UNIQUE_RATIO = 0.20
MAX_CATEGORY_LENGTH = 50

IDENTIFIER_NAME = re.compile(
    r"(^|[_\s-])(id|uuid|guid|identifier|index|key)($|[_\s-])",
    re.IGNORECASE,
)
DATE_NAME = re.compile(
    r"(^|[_\s-])(date|datetime|timestamp|time)($|[_\s-])",
    re.IGNORECASE,
)
BOOLEAN_NAME = re.compile(
    r"(^|[_\s-])(is|has|was|can|enabled|active|valid|flag)($|[_\s-])",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"^\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
)


def _sample(series: pd.Series) -> pd.Series:
    """Take a reproducible sample for inference on large columns."""
    values = series.dropna()

    if len(values) > SAMPLE_SIZE:
        values = values.sample(SAMPLE_SIZE, random_state=42)

    return values


def _result(
    semantic_type: str,
    confidence: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "semantic_type": semantic_type,
        "confidence": round(max(0.0, min(confidence, 1.0)), 2),
        "inference_reason": reason,
    }


def infer_semantic_type(
    series: pd.Series,
    column_name: str | None = None,
) -> dict[str, Any]:
    """
    Infer a column's semantic type.

    Returns semantic_type, confidence, and inference_reason.
    This function does not modify or convert the original series.
    """
    name = str(column_name or series.name or "")
    values = _sample(series)

    if values.empty:
        return _result("unknown", 0.0, "Column has no non-null values")

    # Native types take precedence over string-based inference.
    if pd.api.types.is_bool_dtype(series):
        return _result("boolean", 1.0, "Native boolean dtype")

    if pd.api.types.is_datetime64_any_dtype(series):
        return _result("datetime", 1.0, "Native datetime dtype")

    if pd.api.types.is_numeric_dtype(series):
        if IDENTIFIER_NAME.search(name):
            return _result(
                "identifier", 0.98,
                "Numeric dtype and identifier-like column name",
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
    numeric_values = pd.to_numeric(strings, errors="coerce")
    numeric_ratio = float(numeric_values.notna().mean())

    if numeric_ratio >= NUMERIC_PARSE_THRESHOLD:
        unique_ratio = strings.nunique() / len(strings)

        if IDENTIFIER_NAME.search(name):
            return _result(
                "identifier", 0.95,
                "Numeric-looking strings and identifier-like column name",
            )

        if (
            BOOLEAN_NAME.search(name)
            and set(normalized.unique()).issubset({"0", "1"})
        ):
            return _result(
                "boolean", 0.85,
                "Binary 0/1 values and boolean-like column name",
            )

        # A high-cardinality numeric string column with an identifier-like
        # name is more convincing than uniqueness alone.
        return _result(
            "numeric", min(0.95, numeric_ratio),
            "Most sampled strings can be parsed as numbers",
        )

    # Date parsing is attempted only when values look date-like or the
    # column name suggests a date, reducing accidental numeric conversions.
    date_like_ratio = float(
        strings.str.match(DATE_PATTERN).mean()
    )
    date_hint = bool(DATE_NAME.search(name))

    if date_like_ratio >= DATE_PARSE_THRESHOLD or date_hint:
        parsed_dates = pd.to_datetime(strings, errors="coerce")
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
            "Low unique-value ratio and relatively short values",
        )

    if IDENTIFIER_NAME.search(name):
        return _result(
            "identifier", 0.80,
            "Identifier-like column name with non-numeric values",
        )

    return _result(
        "text", 0.80,
        "Values appear to be free text or high-cardinality strings",
    )
