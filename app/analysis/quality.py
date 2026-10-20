
from typing import Any

import pandas as pd


def _finding(
    rule_id: str,
    severity: str,
    title: str,
    columns: list[str],
    evidence: dict[str, Any],
    recommendation: str,
) -> dict:
    return {
        "rule_id": rule_id,
        "severity": severity,
        "title": title,
        "columns": columns,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def analyze_quality(
    df: pd.DataFrame,
    profile: dict,
) -> dict:
    """Detect data quality issues using a DataFrame and its profile"""

    findings = []

    rows = len(df)

    # 1. Missing values: reuse profiler statistics.
    for column, stats in profile["columns"].items():
        missing = stats["missing_count"]
        percentage = stats["missing_percentage"]

        if missing == 0:
            continue

        severity = (
            "high" if percentage >= 30
            else "medium" if percentage >= 5
            else "low"
        )

        findings.append(
            _finding(
                rule_id="MISSING_VALUES",
                severity=severity,
                title=f"Missing values in '{column}'",
                columns=[column],
                evidence={
                    "missing_count": missing,
                    "missing_percentage": percentage,
                },
                recommendation=(
                    "Investigate why values are missing and choose "
                    "an appropriate handling strategy."
                ),
            )
        )

    # 2. Duplicate rows: reuse profiler statistics.
    duplicate_rows = profile["overview"]["duplicate_rows"]

    if duplicate_rows:
        findings.append(
            _finding(
                rule_id="DUPLICATE_ROWS",
                severity="medium",
                title="Duplicate rows detected",
                columns=list(df.columns.astype(str)),
                evidence={
                    "duplicate_rows": duplicate_rows,
                    "duplicate_percentage": (
                        round(duplicate_rows / rows * 100, 2)
                        if rows else 0.0
                    ),
                },
                recommendation=(
                    "Determine whether these records are legitimate "
                    "repeated observations before removing them."
                ),
            )
        )

    # 3. Constant and all null columns: reuse profile.
    for column in profile["quality"]["constant_columns"]:
        findings.append(
            _finding(
                rule_id="CONSTANT_COLUMN",
                severity="low",
                title=f"Column '{column}' has no variation",
                columns=[column],
                evidence={"unique_non_null_values": 1},
                recommendation=(
                    "Consider excluding this column from analyses "
                    "that require variation."
                ),
            )
        )

    for column in profile["quality"]["all_null_columns"]:
        findings.append(
            _finding(
                rule_id="ALL_NULL_COLUMN",
                severity="high",
                title=f"Column '{column}' contains only missing values",
                columns=[column],
                evidence={"rows": rows},
                recommendation=(
                    "Verify whether this column is expected to contain "
                    "data; consider removing it if it is not useful."
                ),
            )
        )

    # 4. Potential identifier columns.
    # This is a heuristic, not a definitive classification.
    for column in df.columns:
        series = df[column]
        name = str(column).lower()

        if rows == 0 or series.isna().any():
            continue

        unique_ratio = series.nunique() / rows
        name_suggests_id = (
            name == "id"
            or name.endswith("_id")
            or name.endswith("id")
            or name in {"uuid", "index", "record_key"}
        )

        if name_suggests_id and unique_ratio >= 0.9:
            findings.append(
                _finding(
                    rule_id="POTENTIAL_IDENTIFIER",
                    severity="low",
                    title=f"Column '{column}' may be an identifier",
                    columns=[str(column)],
                    evidence={
                        "unique_ratio": round(unique_ratio, 4),
                        "unique_values": int(series.nunique()),
                        "rows": rows,
                    },
                    recommendation=(
                        "Check whether this column is a record identifier "
                        "rather than a meaningful predictive feature."
                    ),
                )
            )

    # 5. Outliers: reuse IQR counts from the profile.
    for column, stats in profile["columns"].items():
        numeric_stats = stats.get("statistics")

        if not numeric_stats:
            continue

        outlier_count = numeric_stats.get("outlier_count_iqr", 0)

        if outlier_count:
            findings.append(
                _finding(
                    rule_id="POTENTIAL_OUTLIERS",
                    severity="low",
                    title=f"Potential outliers in '{column}'",
                    columns=[column],
                    evidence={
                        "outlier_count": outlier_count,
                        "method": "1.5 × IQR rule",
                    },
                    recommendation=(
                        "Inspect the flagged values and determine whether "
                        "they are valid extreme observations or data errors."
                    ),
                )
            )

    # 6. Numeric looking strings.
    for column in df.columns:
        series = df[column]

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        non_missing = series.dropna()

        if non_missing.empty:
            continue

        # Avoid expensive inference on every row.
        sample = non_missing.head(1000)
        parsed = pd.to_numeric(sample, errors="coerce")
        numeric_ratio = parsed.notna().mean()

        if numeric_ratio >= 0.95:
            findings.append(
                _finding(
                    rule_id="NUMERIC_STORED_AS_TEXT",
                    severity="medium",
                    title=f"Column '{column}' may contain numbers stored as text",
                    columns=[str(column)],
                    evidence={
                        "sample_size": len(sample),
                        "numeric_parse_ratio": round(
                            float(numeric_ratio), 4
                        ),
                    },
                    recommendation=(
                        "Inspect formatting and missing-value tokens before "
                        "converting this column to a numeric type."
                    ),
                )
            )

    # 7. Inconsistent categorical labels.
    for column in df.columns:
        series = df[column]

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        values = series.dropna().astype(str)
        normalized = values.str.strip().str.casefold()

        # Report only when different original labels collapse
        # to the same normalized value.
        pairs = pd.DataFrame({
            "original": values,
            "normalized": normalized,
        })

        inconsistent = (
            pairs.groupby("normalized")["original"]
            .agg(lambda group: sorted(set(group)))
        )

        inconsistent = inconsistent[
            inconsistent.map(len) > 1
        ]

        if not inconsistent.empty:
            examples = [
                {
                    "normalized_value": str(key),
                    "observed_values": value[:5],
                }
                for key, value in inconsistent.head(5).items()
            ]

            findings.append(
                _finding(
                    rule_id="INCONSISTENT_CATEGORIES",
                    severity="low",
                    title=f"Potentially inconsistent labels in '{column}'",
                    columns=[str(column)],
                    evidence={
                        "groups": examples,
                        "groups_detected": len(inconsistent),
                    },
                    recommendation=(
                        "Review labels that differ only by capitalization "
                        "or surrounding whitespace before standardizing them."
                    ),
                )
            )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(
        key=lambda item: severity_order[item["severity"]]
    )

    counts = {
        level: sum(
            finding["severity"] == level for finding in findings
        )
        for level in severity_order
    }

    return {
        "summary": {
            "total_findings": len(findings),
            "severity_counts": counts,
        },
        "findings": findings,
    }
