"""SQL analyses (DuckDB) over experiment tables.

The queries live in ``.sql`` files next to this module so they can be read, reviewed
and run in any SQL client, not only from Python.
"""

from __future__ import annotations

from importlib import resources

import duckdb
import pandas as pd

QUERIES = ("arm_summary", "segment_effects", "score_deciles")


def load_query(name: str) -> str:
    """Return the text of a bundled SQL query by name (without the ``.sql`` suffix)."""
    if name not in QUERIES:
        raise KeyError(f"unknown query {name!r}; choose from {QUERIES}")
    return resources.files(__package__).joinpath(f"{name}.sql").read_text(encoding="utf-8")


def run_query(name: str, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run a bundled query against in-memory DataFrames registered as tables.

    Args:
        name: Query name, one of ``QUERIES``.
        tables: Mapping from table name used in the query to a DataFrame.

    Returns:
        The query result as a DataFrame.
    """
    con = duckdb.connect(database=":memory:")
    try:
        for table_name, frame in tables.items():
            con.register(table_name, frame)
        return con.execute(load_query(name)).df()
    finally:
        con.close()
