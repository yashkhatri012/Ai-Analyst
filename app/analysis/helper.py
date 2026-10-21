
import re
import pandas as pd
from typing import Any
import numpy as np


"""HELPER FUCNTIONS FOR SEMENTIC TYPES"""


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
    r"(^|[_\s-])(date|datetime|timestamp|time|dob"
    r"|created_at|updated_at|modified_at|deleted_at)($|[_\s-])",
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


def normalize_name(name: str) -> str:
    """
    Split camelCase / PascalCase / acronym boundaries with underscores so
    the name regexes work: 'customerId' -> 'customer_Id',
    'userID' -> 'user_ID', 'IDNumber' -> 'ID_Number'.
    """
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)


def is_identifier_name(name: str) -> bool:
    """True when the column name looks like an identifier (id, uuid, key...)."""
    return bool(IDENTIFIER_NAME.search(normalize_name(name)))


def parse_numeric(values: pd.Series) -> pd.Series:
    """Single numeric parser shared by all analysis modules."""
    # errors="coerce" converts invalid values into NaN (Not a Number)
    return pd.to_numeric(values, errors="coerce")


def parse_datetimes(values: pd.Series) -> pd.Series:
    """
    Single datetime parser shared by all analysis modules.

    format="mixed" parses each value independently. Without it, pandas 2+
    infers one format from the first value and turns every value in a
    different format into NaT, so modules would disagree about the same column.
    """
    return pd.to_datetime(values, errors="coerce", format="mixed")


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



"""HELPER FUCNTIONS FOR PROFILER"""



TOP_N = 3

def _clean_number(value: Any) -> Any:
    """Convert NumPy scalars and non-finite numbers to JSON safe values."""
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value):
        return None

    return value


def _numeric_values(series: pd.Series) -> pd.Series:
    """Convert a semantically numeric column into usable numeric values."""
    return parse_numeric(series).dropna()


def _datetime_values(series: pd.Series) -> pd.Series:
    """Parse datetime values, including columns stored as strings."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dropna()

    values = series.dropna().astype("string").str.strip()
    values = values[values.ne("")].dropna()

    if values.empty:
        return pd.Series(dtype="datetime64[ns]")

    return parse_datetimes(values).dropna()


def _top_values(series: pd.Series) -> list[dict[str, Any]]:
    """Return the most frequent non missing values."""
    return [
        {
            "value": str(value),
            "count": int(count),
        }
        for value, count in series.dropna().value_counts().head(TOP_N).items()
    ]

