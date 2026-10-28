from app.graph.constants import GLOBAL_RULES, DROP_EVIDENCE


def build_quality_flags(
    dataset_context: dict,
    columns: list[str],
    limit: int = 8,
) -> list[dict]:
    """Return relevant, compact quality findings for the selected columns."""
    selected_columns = set(columns)
    flags = []

    for finding in dataset_context.get("quality", {}).get("findings", []):
        if finding.get("rule_id") in GLOBAL_RULES:
            continue

        finding_columns = finding.get("columns", [])
        hit_columns = [column for column in finding_columns if column in selected_columns]
        if not hit_columns:
            continue

        evidence = finding.get("evidence", {})
        flags.append({
            "rule_id": finding.get("rule_id"),
            "severity": finding.get("severity"),
            "columns": hit_columns,
            "evidence": {
                key: value
                for key, value in evidence.items()
                if key not in DROP_EVIDENCE
            },
        })

    # The context builder is expected to sort findings by severity.
    return flags[:limit]


def build_profile_slice(dataset_context: dict, columns: list[str]) -> dict:
    """Build a compact profile containing only the requested columns."""
    profiles = dataset_context.get("columns", {})
    other_columns = dataset_context.get("other_columns", {})
    compact_columns = other_columns.get("columns", {})
    compact_fields = other_columns.get("fields", [])
    selected = {}

    for column in columns:
        if column in profiles:
            selected[column] = profiles[column]
        elif column in compact_columns:
            selected[column] = dict(zip(compact_fields, compact_columns[column]))

    return {
        "columns": selected,
        "notes": "Column profiles may be compacted. Use only the supplied information.",
    }


def build_zero_assumption_caveat(
    step_flags: dict[str, list[dict]],
    results: list[dict],
) -> str | None:
    """Create the zero-as-missing caveat only when a result indicates rows were excluded."""
    zero_columns = sorted({
        column
        for flags in step_flags.values()
        for flag in flags
        if flag.get("rule_id") == "HIGH_ZERO_PROPORTION"
        for column in flag.get("columns", [])
    })

    if not zero_columns:
        return None

    excluded = any(
        row.get("n_all") is not None
        and row.get("n") is not None
        and row["n"] < row["n_all"]
        for result in results
        for row in result.get("result_table", [])
    )
    if not excluded:
        return None

    return (
        f"Rows where {', '.join(zero_columns)} equal 0 were excluded, treating "
        "zeros as missing values. This is an assumption; the data does not "
        "confirm it. n is reported per pair."
    )
