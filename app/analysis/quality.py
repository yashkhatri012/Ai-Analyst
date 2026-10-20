
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


@dataclass(frozen=True)
class QualityConfig:
    """Configurable thresholds for generic quality checks."""

    missing_medium_pct: float = 5.0
    missing_high_pct: float = 30.0

    identifier_unique_ratio: float = 0.95
    numeric_parse_ratio: float = 0.95
    date_parse_ratio: float = 0.80

    categorical_max_unique_ratio: float = 0.20
    categorical_max_average_length: float = 50.0

    outlier_iqr_multiplier: float = 1.5
    minimum_outlier_count: int = 1

    examples_limit: int = 5
    category_groups_limit: int = 10


DEFAULT_CONFIG = QualityConfig()

IDENTIFIER_NAME_PATTERN = re.compile(
    r"(^|[_\W])(id|uuid|guid|index|key|identifier)([_\W]|$)",
    flags=re.IGNORECASE,
)

DATE_NAME_PATTERN = re.compile(
    r"(^|[_\W])(date|datetime|timestamp)([_\W]|$)",
    flags=re.IGNORECASE,
)

NUMERIC_TEXT_PATTERN = re.compile(
    r"""
    ^[+-]?
    (?:
        (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?
        |
        \.\d+
    )
    (?:[eE][+-]?\d+)?$
    """,
    flags=re.VERBOSE,
)

DATE_LIKE_PATTERN = re.compile(
    r"""
    \d{1,4}[-/]\d{1,2}[-/]\d{1,4}
    |
    \d{4}-\d{2}
    |
    \d{4}-\d{2}-\d{2}[T\s]
    """,
    flags=re.VERBOSE,
)


def _is_string_column(series: pd.Series) -> bool:
    return is_object_dtype(series) or is_string_dtype(series)


def _is_identifier_candidate(
    column: Any,
    series: pd.Series,
    config: QualityConfig,
) -> bool:
    """
    Infer likely identifiers from generic name patterns or sequential values.

    High cardinality alone is not sufficient: continuous measurements can
    legitimately have many distinct values.
    """
    name = str(column).strip()
    values = series.dropna()

    if values.empty or values.nunique() < 2:
        return False

    unique_ratio = values.nunique() / len(values)

    if unique_ratio < config.identifier_unique_ratio:
        return False

    if IDENTIFIER_NAME_PATTERN.search(name):
        return True

    # A secondary heuristic for unnamed sequential integer identifiers.
    if is_numeric_dtype(series) and not values.empty:
        numeric = pd.to_numeric(values, errors="coerce").dropna()

        if (
            len(numeric) == len(values)
            and (numeric % 1 == 0).all()
            and numeric.nunique() == len(numeric)
        ):
            ordered = numeric.sort_values().reset_index(drop=True)
            differences = ordered.diff().dropna()

            # Sequential integers are evidence, not proof, of an identifier.
            if (
                not differences.empty
                and (differences == 1).mean() >= 0.90
            ):
                return True

    return False


def _missing_severity(
    missing_percentage: float,
    config: QualityConfig,
) -> str:
    if missing_percentage >= config.missing_high_pct:
        return "high"
    if missing_percentage >= config.missing_medium_pct:
        return "medium"
    return "low"


def _looks_structured(series: pd.Series) -> bool:
    """Avoid interpreting serialized arrays/objects as categorical labels."""
    values = series.dropna().astype(str).str.strip().head(100)

    if values.empty:
        return False

    structured_ratio = values.str.startswith(
        ("[", "{", "(", "array(")
    ).mean()

    return bool(structured_ratio >= 0.70)


def _looks_categorical(
    series: pd.Series,
    config: QualityConfig,
) -> bool:
    """
    Conservative heuristic for low-cardinality string categories.

    This is an inference, not a guarantee that the column is categorical.
    """
    values = series.dropna().astype(str).str.strip()

    if len(values) < 2 or values.empty:
        return False

    if _looks_structured(series):
        return False

    unique_ratio = values.nunique() / len(values)
    average_length = float(values.str.len().mean())

    return (
        unique_ratio <= config.categorical_max_unique_ratio
        and average_length <= config.categorical_max_average_length
    )


def _numeric_parse_ratio(series: pd.Series) -> tuple[float, int]:
    """Return the parse ratio and number of non-null strings examined."""
    values = series.dropna().astype(str).str.strip()

    if values.empty:
        return 0.0, 0

    # Require a numeric-looking textual format before attempting conversion.
    numeric_format = values.map(
        lambda value: bool(NUMERIC_TEXT_PATTERN.fullmatch(value))
    )

    return float(numeric_format.mean()), len(values)


def _date_parse_ratio(
    column: Any,
    series: pd.Series,
) -> tuple[float, int, list[str]]:
    """
    Detect date-like strings using both parseability and format evidence.

    A column name can support detection, but cannot establish it by itself.
    """
    values = series.dropna().astype(str).str.strip()

    if values.empty:
        return 0.0, 0, []

    name_suggests_date = bool(DATE_NAME_PATTERN.search(str(column)))
    value_suggests_date = values.map(
        lambda value: bool(DATE_LIKE_PATTERN.search(value))
    )

    if not name_suggests_date and not value_suggests_date.any():
        return 0.0, len(values), []

    # Explicit format evidence reduces false positives from arbitrary strings.
    candidate_values = values[
        value_suggests_date
        if value_suggests_date.any()
        else values.index.isin(values.index)
    ]

    if candidate_values.empty:
        return 0.0, len(values), []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(candidate_values, errors="coerce")

    ratio = float(parsed.notna().mean())
    examples = parsed.dropna().astype(str).head(3).tolist()

    return ratio, len(candidate_values), examples


def _add_finding(
    findings: list[dict[str, Any]],
    *,
    rule_id: str,
    severity: str,
    title: str,
    columns: list[str],
    evidence: dict[str, Any],
    recommendation: str,
) -> None:
    findings.append(
        {
            "rule_id": rule_id,
            "severity": severity,
            "title": title,
            "columns": columns,
            "evidence": evidence,
            "recommendation": recommendation,
        }
    )


def _check_missing_values(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    config: QualityConfig,
) -> None:
    row_count = len(df)

    if row_count == 0:
        return

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count == 0:
            continue

        missing_percentage = round(missing_count / row_count * 100, 2)

        _add_finding(
            findings,
            rule_id="MISSING_VALUES",
            severity=_missing_severity(missing_percentage, config),
            title=f"Missing values in '{column}'",
            columns=[str(column)],
            evidence={
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
            },
            recommendation=(
                "Determine whether missingness is expected for this field "
                "and assess its impact before choosing an imputation, "
                "exclusion, or retention strategy."
            ),
        )


def _check_duplicate_rows(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
) -> None:
    if df.empty:
        return

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        return

    _add_finding(
        findings,
        rule_id="DUPLICATE_ROWS",
        severity="medium",
        title="Duplicate rows detected",
        columns=[],
        evidence={
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": round(
                duplicate_count / len(df) * 100, 2
            ),
        },
        recommendation=(
            "Inspect duplicate records and determine whether they are "
            "accidental duplicates or valid repeated observations."
        ),
    )


def _check_constant_columns(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
) -> None:
    for column in df.columns:
        series = df[column]
        unique_count = int(series.nunique(dropna=True))

        if series.isna().all():
            _add_finding(
                findings,
                rule_id="ALL_NULL_COLUMN",
                severity="medium",
                title=f"Column '{column}' contains only missing values",
                columns=[str(column)],
                evidence={"missing_count": int(series.isna().sum())},
                recommendation=(
                    "Verify whether this column is expected to contain "
                    "values before deciding whether it should be retained."
                ),
            )
        elif len(series) > 0 and unique_count <= 1:
            _add_finding(
                findings,
                rule_id="CONSTANT_COLUMN",
                severity="low",
                title=f"Column '{column}' has no variation",
                columns=[str(column)],
                evidence={"unique_non_null_values": unique_count},
                recommendation=(
                    "Assess whether this column contributes useful "
                    "information to the intended analysis."
                ),
            )


def _check_identifiers(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    config: QualityConfig,
) -> set[Any]:
    """
    Return likely identifier columns so they can be excluded from outlier checks.
    """
    identifiers: set[Any] = set()

    for column in df.columns:
        series = df[column]

        if not _is_identifier_candidate(column, series, config):
            continue

        identifiers.add(column)
        values = series.dropna()
        unique_ratio = values.nunique() / max(len(values), 1)

        _add_finding(
            findings,
            rule_id="POTENTIAL_IDENTIFIER",
            severity="low",
            title=f"Column '{column}' may be an identifier",
            columns=[str(column)],
            evidence={
                "unique_ratio": round(unique_ratio, 4),
                "unique_values": int(values.nunique()),
                "non_null_rows": int(len(values)),
                "detection_basis": (
                    "column_name_or_sequential_integer_pattern"
                ),
            },
            recommendation=(
                "Confirm whether this column identifies records. If it does, "
                "exclude it from analyses that interpret the identifier as "
                "a meaningful measurement."
            ),
        )

    return identifiers


def _check_numeric_outliers(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    identifier_columns: set[Any],
    config: QualityConfig,
) -> None:
    for column in df.columns:
        series = df[column]

        if column in identifier_columns or not is_numeric_dtype(series):
            continue

        values = series.dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        # No useful IQR fence when the distribution has zero spread.
        if pd.isna(iqr) or iqr <= 0:
            continue

        lower_bound = q1 - config.outlier_iqr_multiplier * iqr
        upper_bound = q3 + config.outlier_iqr_multiplier * iqr

        outliers = values[(values < lower_bound) | (values > upper_bound)]
        outlier_count = int(len(outliers))

        if outlier_count < config.minimum_outlier_count:
            continue

        _add_finding(
            findings,
            rule_id="POTENTIAL_OUTLIERS",
            severity="low",
            title=f"Potential outliers in '{column}'",
            columns=[str(column)],
            evidence={
                "outlier_count": outlier_count,
                "outlier_percentage": round(
                    outlier_count / len(values) * 100, 2
                ),
                "method": f"{config.outlier_iqr_multiplier} × IQR rule",
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "examples": outliers.head(config.examples_limit).tolist(),
            },
            recommendation=(
                "Inspect extreme observations in context. Statistical "
                "outliers are not necessarily errors and should not be "
                "removed automatically."
            ),
        )


def _check_numeric_text(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    identifier_columns: set[Any],
    config: QualityConfig,
) -> None:
    for column in df.columns:
        series = df[column]

        if column in identifier_columns or not _is_string_column(series):
            continue

        values = series.dropna().astype(str).str.strip()

        if values.empty:
            continue

        parse_ratio, checked_count = _numeric_parse_ratio(series)

        if parse_ratio < config.numeric_parse_ratio:
            continue

        # A textual column with numeric-looking values might be a code or ID.
        # This finding recommends inspection rather than automatic conversion.
        _add_finding(
            findings,
            rule_id="NUMERIC_STORED_AS_TEXT",
            severity="low",
            title=f"Column '{column}' may contain numbers stored as text",
            columns=[str(column)],
            evidence={
                "numeric_parse_ratio": round(parse_ratio, 4),
                "non_null_values_checked": checked_count,
                "examples": values.head(config.examples_limit).tolist(),
            },
            recommendation=(
                "Check the column's meaning, formatting, leading zeros, and "
                "units. Convert it to numeric only if it represents a "
                "quantitative measurement rather than a code or identifier."
            ),
        )


def _check_date_text(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    config: QualityConfig,
) -> None:
    for column in df.columns:
        series = df[column]

        if not _is_string_column(series):
            continue

        ratio, checked_count, examples = _date_parse_ratio(column, series)

        if ratio < config.date_parse_ratio:
            continue

        _add_finding(
            findings,
            rule_id="DATE_STORED_AS_TEXT",
            severity="low",
            title=f"Column '{column}' may contain dates stored as text",
            columns=[str(column)],
            evidence={
                "date_parse_ratio": round(ratio, 4),
                "values_checked": checked_count,
                "parsed_examples": examples,
            },
            recommendation=(
                "Validate the date format and intended timezone, then parse "
                "the column to a datetime dtype if appropriate."
            ),
        )


def _check_inconsistent_categories(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    config: QualityConfig,
) -> None:
    for column in df.columns:
        series = df[column]

        if not _is_string_column(series):
            continue

        if not _looks_categorical(series, config):
            continue

        values = series.dropna().astype(str)
        normalized_groups: dict[str, set[str]] = {}

        for value in values:
            normalized = value.strip().casefold()

            if normalized:
                normalized_groups.setdefault(normalized, set()).add(value)

        inconsistent_groups = [
            {
                "normalized_value": normalized,
                "observed_values": sorted(observed)[
                    :config.examples_limit
                ],
            }
            for normalized, observed in normalized_groups.items()
            if len(observed) > 1
        ]

        if not inconsistent_groups:
            continue

        _add_finding(
            findings,
            rule_id="INCONSISTENT_CATEGORIES",
            severity="low",
            title=f"Potentially inconsistent labels in '{column}'",
            columns=[str(column)],
            evidence={
                "groups": inconsistent_groups[:config.category_groups_limit],
                "groups_detected": len(inconsistent_groups),
            },
            recommendation=(
                "Review labels that differ only by capitalization or "
                "surrounding whitespace. Standardize them only if they "
                "represent the same category."
            ),
        )


def analyze_quality(
    df: pd.DataFrame,
    profile: dict[str, Any] | None = None,
    config: QualityConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """
    Run generic, deterministic quality checks on a DataFrame.

    `profile` is retained for API compatibility. These checks calculate the
    evidence they need directly from `df` to avoid stale or mismatched profile
    statistics.
    """
    del profile  # Reserved for future checks that can safely reuse the profile.

    findings: list[dict[str, Any]] = []

    _check_missing_values(df, findings, config)
    _check_duplicate_rows(df, findings)
    _check_constant_columns(df, findings)

    identifier_columns = _check_identifiers(df, findings, config)

    _check_numeric_outliers(
        df, findings, identifier_columns, config
    )
    _check_numeric_text(
        df, findings, identifier_columns, config
    )
    _check_date_text(df, findings, config)
    _check_inconsistent_categories(df, findings, config)

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    findings.sort(
        key=lambda item: (
            severity_rank.get(item["severity"], 3),
            item["rule_id"],
            item["columns"],
        )
    )

    severity_counts = {
        severity: sum(
            finding["severity"] == severity for finding in findings
        )
        for severity in ("high", "medium", "low")
    }

    return {
        "summary": {
            "total_findings": len(findings),
            "severity_counts": severity_counts,
        },
        "findings": findings,
    }
