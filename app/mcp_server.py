
import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.tools.inspect import (
    describe_columns as _describe_columns,
    sample_rows as _sample_rows,
    value_counts as _value_counts,
    missingness_report as _missingness_report,
    query_data as _query_data,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("ai-analyst")


def _call(function, **kwargs) -> str:
    """Keep tool errors inside a predictable JSON response."""
    try:
        return json.dumps(function(**kwargs), default=str)
    except Exception as exc:
        logger.exception("MCP tool failed: %s", function.__name__)
        return json.dumps({
            "status": "error",
            "stdout": "",
            "tables": [],
            "figures": [],
            "error": str(exc),
            "provenance": {
                "tool": function.__name__,
                "arguments": kwargs,
            },
        }, default=str)


@mcp.tool()
def describe_columns(
    dataset_id: str, columns: list[str], version: str | None = None
) -> str:
    """Get full profile statistics for selected dataset columns."""
    return _call(
        _describe_columns,
        dataset_id=dataset_id, columns=columns, version=version
    )


@mcp.tool()
def sample_rows(
    dataset_id: str,
    columns: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    n: int = 10,
    version: str | None = None,
) -> str:
    """Return up to 20 sanitized rows, optionally filtered."""
    return _call(
        _sample_rows,
        dataset_id=dataset_id, columns=columns,
        filters=filters, n=n, version=version
    )


@mcp.tool()
def value_counts(
    dataset_id: str, column: str, top_n: int = 10,
    version: str | None = None,
) -> str:
    """Return the most frequent values in a column."""
    return _call(
        _value_counts,
        dataset_id=dataset_id, column=column,
        top_n=top_n, version=version
    )


@mcp.tool()
def missingness_report(
    dataset_id: str, version: str | None = None
) -> str:
    """Report missing values and co-missing patterns by column."""
    return _call(
        _missingness_report,
        dataset_id=dataset_id, version=version
    )


@mcp.tool()
def query_data(
    dataset_id: str,
    filters: list[dict[str, Any]] | None = None,
    group_by: list[str] | None = None,
    aggregations: dict[str, str] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    limit: int = 20,
    version: str | None = None,
) -> str:
    """Filter, group, aggregate, and sort dataset rows without arbitrary code."""
    return _call(
        _query_data,
        dataset_id=dataset_id, filters=filters,
        group_by=group_by, aggregations=aggregations,
        sort_by=sort_by, ascending=ascending,
        limit=limit, version=version
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
