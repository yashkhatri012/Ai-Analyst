"""Rule based data quality analysis for Pandas DataFrames using semantic_types"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.semantic_types import (
    infer_semantic_type,
    
)
from app.analysis.helper import (
    _clean_number,
    is_identifier_name,
    parse_datetimes,
    parse_numeric,
)


@dataclass(frozen=True)
class QualityConfig:
    """Configurable thresholds for quality checks"""

    missing_medium_pct: float = 5.0
    missing_high_pct: float = 30.0

    identifier_unique_ratio: float = 0.95
    numeric_parse_ratio: float = 0.95
    date_parse_ratio: float = 0.80

    categorical_max_unique_ratio: float = 0.20
    categorical_max_average_length: float = 50.0

    outlier_iqr_multiplier: float = 1.5
    minimum_outlier_count: int = 3

    max_examples: int = 5
    max_category_groups: int = 10

    # Zero checks are opt in because zero may be a valid value.
    suspicious_zero_columns: tuple[str, ...] = ()
    suspicious_zero_min_pct: float = 50.0


DEFAULT_CONFIG = QualityConfig()





def _severity_for_missing(
    percentage: float,
    config: QualityConfig,
) -> str | None:
    if percentage >= config.missing_high_pct:
        return "high"
    if percentage >= config.missing_medium_pct:
        return "medium"
    return None


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


def _sample_values(series: pd.Series, limit: int) -> list[Any]:
    """Return a small set of JSON safe example values."""
    values = series.dropna().drop_duplicates().head(limit)
    return [str(value) for value in values.tolist()]


def _is_identifier(
    name: str,
    series: pd.Series,
    semantic: dict[str, Any],
    config: QualityConfig,
) -> bool:
    """Use semantic inference, column names, and uniqueness as signals."""
    if semantic["semantic_type"] == "identifier":
        return True

    if is_identifier_name(name):
        return True

    non_missing = series.dropna()
    if non_missing.empty:
        return False

    # High uniqueness alone is not enough to classify a numeric measure
    # as an identifier. Require a name hint or string like values.
    if not pd.api.types.is_numeric_dtype(series):
        unique_ratio = non_missing.nunique() / len(non_missing)
        if unique_ratio >= config.identifier_unique_ratio:
            return True

    return False


def _parse_ratio(series: pd.Series, parser: str) -> tuple[float, int]:
    """Return parse success ratio and number of non empty values."""
    values = series.dropna().astype("string").str.strip()
    values = values[values.ne("")].dropna()

    if values.empty:
        return 0.0, 0

    if parser == "numeric":
        parsed = parse_numeric(values)
    elif parser == "datetime":
        parsed = parse_datetimes(values)
    else:
        raise ValueError(f"Unsupported parser: {parser}")

    return float(parsed.notna().mean()), len(values)


def _category_inconsistencies(
    series: pd.Series,
    config: QualityConfig,
) -> list[dict[str, Any]]:
    """
    Find text values that differ only by case or surrounding whitespace.

    Examples: 'Action', ' action', and 'ACTION'.
    """
    values = series.dropna().astype("string")
    values = values[values.str.strip().ne("")]

    if values.empty:
        return []

    groups: dict[str, set[str]] = {}

    for value in values.unique():
        original = str(value)
        normalized = original.strip().casefold()
        groups.setdefault(normalized, set()).add(original)

    inconsistent = [
        {
            "normalized_value": normalized,
            "variants": sorted(variants)[: config.max_examples],
            "variant_count": len(variants),
        }
        for normalized, variants in groups.items()
        if len(variants) > 1
    ]

    inconsistent.sort(
        key=lambda item: item["variant_count"],
        reverse=True,
    )

    return inconsistent[: config.max_category_groups]


def _check_missing_values(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    config: QualityConfig,
) -> None:
    row_count = len(df)

    if row_count == 0:
        return

    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())

        if missing_count == 0:
            continue

        # Fully empty columns are reported by ALL_NULL_COLUMN instead,
        # to avoid two findings for the same problem.
        if missing_count == row_count:
            continue

        missing_pct = missing_count / row_count * 100
        severity = _severity_for_missing(missing_pct, config)

        if severity is None:
            continue

        _add_finding(
            findings,
            rule_id="MISSING_VALUES",
            severity=severity,
            title=f"Missing values in '{column}'",
            columns=[str(column)],
            evidence={
                "missing_count": missing_count,
                "total_rows": row_count,
                "missing_percentage": round(missing_pct, 2),
                "examples_of_non_missing_values": _sample_values(
                    series, config.max_examples
                ),
            },
            recommendation=(
                "Determine whether these values are genuinely unknown, "
                "not applicable, or missing due to data collection. "
                "Choose an appropriate handling strategy before analysis."
            ),
        )


def _check_duplicates(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
) -> None:
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        return

    _add_finding(
        findings,
        rule_id="DUPLICATE_ROWS",
        severity="medium",
        title="Duplicate rows detected",
        columns=[str(column) for column in df.columns],
        evidence={
            "duplicate_row_count": duplicate_count,
            "total_rows": len(df),
            "duplicate_percentage": round(
                duplicate_count / len(df) * 100, 2
            ) if len(df) else 0.0,
        },
        recommendation=(
            "Inspect duplicate records and determine whether they are "
            "legitimate repeated observations or accidental duplicates. "
            "Do not remove them automatically."
        ),
    )


def _check_column_structure(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    config: QualityConfig,
) -> None:
    for column in df.columns:
        series = df[column]
        name = str(column)
        non_missing = series.dropna()

        if series.isna().all():
            _add_finding(
                findings,
                rule_id="ALL_NULL_COLUMN",
                severity="high",
                title=f"Column '{name}' contains only missing values",
                columns=[name],
                evidence={"row_count": len(series)},
                recommendation=(
                    "Verify whether this column was expected to contain "
                    "data. Populate it from a reliable source or remove it "
                    "if it has no analytical purpose."
                ),
            )
            continue

        if non_missing.nunique() <= 1:
            _add_finding(
                findings,
                rule_id="CONSTANT_COLUMN",
                severity="low",
                title=f"Column '{name}' has no variation",
                columns=[name],
                evidence={
                    "unique_non_missing_values": int(non_missing.nunique()),
                    "example_values": _sample_values(
                        non_missing, config.max_examples
                    ),
                },
                recommendation=(
                    "Check whether the constant value is expected. "
                    "A column without variation generally contributes "
                    "little to comparative or predictive analysis."
                ),
            )


def _check_text_parsing(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    semantics: dict[str, dict[str, Any]],
    config: QualityConfig,
) -> None:
    for column in df.columns:
        series = df[column]
        name = str(column)

        # Native numeric and datetime columns do not need text parsing.
        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            continue

        semantic = semantics[name]
        semantic_type = semantic["semantic_type"]

        # An identifier may contain digits, but converting it to a number
        # could destroy leading zeros or its meaning as a label.
        if semantic_type == "identifier":
            continue

        numeric_ratio, numeric_count = _parse_ratio(series, "numeric")

        if (
            numeric_count > 0
            and numeric_ratio >= config.numeric_parse_ratio
        ):
            _add_finding(
                findings,
                rule_id="NUMERIC_STORED_AS_TEXT",
                severity="medium",
                title=f"Numeric values in '{name}' are stored as text",
                columns=[name],
                evidence={
                    "parse_success_ratio": round(numeric_ratio, 3),
                    "non_empty_values_checked": numeric_count,
                    "semantic_type": semantic_type,
                    "semantic_type_confidence": semantic["confidence"],
                    "examples": _sample_values(
                        series, config.max_examples
                    ),
                },
                recommendation=(
                    "Validate the column's meaning and formatting before "
                    "converting it to a numeric dtype. Preserve identifiers, "
                    "leading zeros, currency conventions, and special codes."
                ),
            )
            continue

        date_ratio, date_count = _parse_ratio(series, "datetime")

        if (
            date_count > 0
            and date_ratio >= config.date_parse_ratio
            and semantic_type == "datetime"
        ):
            _add_finding(
                findings,
                rule_id="DATE_STORED_AS_TEXT",
                severity="low",
                title=f"Date values in '{name}' are stored as text",
                columns=[name],
                evidence={
                    "parse_success_ratio": round(date_ratio, 3),
                    "non_empty_values_checked": date_count,
                    "examples": _sample_values(
                        series, config.max_examples
                    ),
                },
                recommendation=(
                    "Validate date formats and convert the column to a "
                    "datetime dtype if date-based filtering, sorting, or "
                    "time-series analysis is required."
                ),
            )


def _check_numeric_outliers(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    identifiers: set[str],
    config: QualityConfig,
) -> None:
    for column in df.columns:
        name = str(column)
        series = df[column]

        if name in identifiers:
            continue

        if not pd.api.types.is_numeric_dtype(series):
            continue

        if pd.api.types.is_bool_dtype(series):
            continue

        values = series.dropna()

        # Very small samples do not support a useful outlier assessment.
        if len(values) < 4 or values.nunique() < 2:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        # If the IQR is zero, the conventional IQR rule is not informative.
        if not np.isfinite(iqr) or iqr <= 0:
            continue

        lower = q1 - config.outlier_iqr_multiplier * iqr
        upper = q3 + config.outlier_iqr_multiplier * iqr

        mask = (values < lower) | (values > upper)
        outliers = values[mask]

        if len(outliers) < config.minimum_outlier_count:
            continue

        _add_finding(
            findings,
            rule_id="POTENTIAL_NUMERIC_OUTLIERS",
            severity="low",
            title=f"Potential statistical outliers in '{name}'",
            columns=[name],
            evidence={
                "outlier_count": int(len(outliers)),
                "non_missing_values": int(len(values)),
                "outlier_percentage": round(
                    len(outliers) / len(values) * 100, 2
                ),
                "q1": _clean_number(q1),
                "q3": _clean_number(q3),
                "iqr": _clean_number(iqr),
                "lower_bound": _clean_number(lower),
                "upper_bound": _clean_number(upper),
                "examples": [
                    _clean_number(value)
                    for value in outliers.head(config.max_examples).tolist()
                ],
                "method": "IQR",
                "multiplier": config.outlier_iqr_multiplier,
            },
            recommendation=(
                "Inspect extreme values in context. They may be valid rare "
                "observations, skewed-distribution effects, or data errors; "
                "do not remove them solely because this rule flagged them."
            ),
        )


def _check_suspicious_zeros(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    config: QualityConfig,
) -> None:
    for column in config.suspicious_zero_columns:
        if column not in df.columns:
            continue

        series = df[column]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        if pd.api.types.is_bool_dtype(series):
            continue

        values = series.dropna()
        if values.empty:
            continue

        zero_count = int((values == 0).sum())
        zero_pct = zero_count / len(values) * 100

        if zero_count == 0 or zero_pct < config.suspicious_zero_min_pct:
            continue

        _add_finding(
            findings,
            rule_id="HIGH_ZERO_PROPORTION",
            severity="low",
            title=f"High proportion of zeros in '{column}'",
            columns=[str(column)],
            evidence={
                "zero_count": zero_count,
                "non_missing_values": int(len(values)),
                "zero_percentage": round(zero_pct, 2),
            },
            recommendation=(
                "Confirm whether zero means a genuine measured value, "
                "a valid absence, an unknown value, or a placeholder. "
                "Do not replace zeros without domain justification."
            ),
        )


def _check_category_inconsistency(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    semantics: dict[str, dict[str, Any]],
    config: QualityConfig,
) -> None:
    for column in df.columns:
        name = str(column)
        series = df[column]

        if not (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.StringDtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue

        semantic = semantics[name]
        if semantic["semantic_type"] not in {"categorical", "text"}:
            continue

        inconsistencies = _category_inconsistencies(series, config)

        if not inconsistencies:
            continue

        _add_finding(
            findings,
            rule_id="INCONSISTENT_TEXT_CATEGORIES",
            severity="low",
            title=f"Case or whitespace inconsistencies in '{name}'",
            columns=[name],
            evidence={
                "inconsistent_groups_found": len(inconsistencies),
                "groups": inconsistencies,
            },
            recommendation=(
                "Review the variants and normalize whitespace or letter "
                "case where they represent the same category. Avoid "
                "normalizing free text when case or spacing is meaningful."
            ),
        )


def analyze_quality(
    df: pd.DataFrame,
    profile: dict[str, Any] | None = None,
    config: QualityConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """
    Analyze a DataFrame and return structured data-quality findings.

    `profile` is accepted for compatibility with the API but is not used
    as the source of truth. All checks are computed from the current DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    if not isinstance(config, QualityConfig):
        raise TypeError("config must be a QualityConfig instance")

    if df.columns.duplicated().any():
        raise ValueError(
            "DataFrame contains duplicate column names; rename them "
            "before running quality analysis."
        )

    findings: list[dict[str, Any]] = []

    # An empty dataset must not look "clean" to the agent: report it.
    if df.empty or len(df.columns) == 0:
        _add_finding(
            findings,
            rule_id="EMPTY_DATASET",
            severity="high",
            title="Dataset has no rows or no columns",
            columns=[str(column) for column in df.columns],
            evidence={
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
            },
            recommendation=(
                "Check the source file, delimiter, header row, and any "
                "filters applied during ingestion. No further quality "
                "checks were run."
            ),
        )
        return {
            "summary": {
                "total_findings": 1,
                "severity_counts": {"high": 1, "medium": 0, "low": 0},
            },
            "findings": findings,
        }

    semantics: dict[str, dict[str, Any]] = {
        str(column): infer_semantic_type(df[column], str(column))
        for column in df.columns
    }

    identifiers = {
        str(column)
        for column in df.columns
        if _is_identifier(
            str(column),
            df[column],
            semantics[str(column)],
            config,
        )
    }

    _check_missing_values(df, findings, config)
    _check_duplicates(df, findings)
    _check_column_structure(df, findings, config)
    _check_text_parsing(df, findings, semantics, config)
    _check_numeric_outliers(df, findings, identifiers, config)
    _check_suspicious_zeros(df, findings, config)
    _check_category_inconsistency(df, findings, semantics, config)

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