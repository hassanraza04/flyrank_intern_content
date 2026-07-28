"""DuckDB feature-frame construction for the refresh opportunity capstone."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


WAREHOUSE = "hf://datasets/FlyRank/internship-warehouse"
MONTHS = (
    "2025-09", "2025-10", "2025-11", "2025-12", "2026-01",
    "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)
COHORTS = (
    ("2025-12", "2025-09-08", "2025-10-05", "2025-10-06", "2025-11-02", "2025-11-03", "2025-11-30"),
    ("2026-01", "2025-10-06", "2025-11-02", "2025-11-03", "2025-11-30", "2025-12-01", "2025-12-28"),
    ("2026-02", "2025-11-03", "2025-11-30", "2025-12-01", "2025-12-28", "2025-12-29", "2026-01-25"),
    ("2026-03", "2025-12-01", "2025-12-28", "2025-12-29", "2026-01-25", "2026-01-26", "2026-02-22"),
    ("2026-04", "2025-12-29", "2026-01-25", "2026-01-26", "2026-02-22", "2026-02-23", "2026-03-22"),
    ("2026-05", "2026-01-26", "2026-02-22", "2026-02-23", "2026-03-22", "2026-03-23", "2026-04-19"),
    ("2026-06-sealed", "2026-03-23", "2026-04-19", "2026-04-20", "2026-05-17", "2026-05-18", "2026-06-14"),
)


def create_connection(token: str) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection configured for the gated warehouse."""
    if not token:
        raise ValueError("A Hugging Face read token is required")
    connection = duckdb.connect()
    safe_token = token.replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{safe_token}')"
    )
    return connection


def daily_source() -> str:
    """Return explicit warehouse partitions so DuckDB never needs unsupported brace globs."""
    paths = ", ".join(
        f"'{WAREHOUSE}/fact_content_daily_performance/month={month}/*.parquet'"
        for month in MONTHS
    )
    return f"read_parquet([{paths}])"


def feature_sql() -> str:
    """Return SQL that aggregates past windows before defining a later outcome proxy."""
    cohort_values = ",\n        ".join(
        "(" + ", ".join(
            [f"'{cohort_id}'", *(f"DATE '{date_value}'" for date_value in dates)]
        ) + ")"
        for cohort_id, *dates in COHORTS
    )
    return f"""
WITH cohorts(
    cohort_id, previous_start, previous_end, current_start, current_end,
    outcome_start, outcome_end
) AS (
    VALUES
        {cohort_values}
),
eligible_clients AS (
    SELECT client_hash_id
    FROM read_parquet('{WAREHOUSE}/dim_clients.parquet')
    WHERE gsc_data_start <= DATE '2025-09-08'
),
windows AS (
    SELECT
        c.cohort_id,
        f.client_hash_id,
        f.content_hash_id,
        SUM(CASE WHEN f.report_date BETWEEN c.previous_start AND c.previous_end THEN f.gsc_impressions ELSE 0 END) AS previous_impressions,
        SUM(CASE WHEN f.report_date BETWEEN c.current_start AND c.current_end THEN f.gsc_impressions ELSE 0 END) AS current_impressions,
        SUM(CASE WHEN f.report_date BETWEEN c.outcome_start AND c.outcome_end THEN f.gsc_impressions ELSE 0 END) AS next_impressions,
        SUM(CASE WHEN f.report_date BETWEEN c.previous_start AND c.previous_end THEN f.gsc_clicks ELSE 0 END) AS previous_clicks,
        SUM(CASE WHEN f.report_date BETWEEN c.current_start AND c.current_end THEN f.gsc_clicks ELSE 0 END) AS current_clicks,
        AVG(CASE WHEN f.report_date BETWEEN c.previous_start AND c.previous_end THEN NULLIF(f.gsc_avg_position, 0) END) AS previous_avg_position,
        AVG(CASE WHEN f.report_date BETWEEN c.current_start AND c.current_end THEN NULLIF(f.gsc_avg_position, 0) END) AS current_avg_position,
        COUNT(*) FILTER (
            WHERE f.report_date BETWEEN c.current_start AND c.current_end
              AND f.gsc_impressions > 0
        ) AS current_active_days
    FROM {daily_source()} AS f
    JOIN cohorts AS c ON f.report_date BETWEEN c.previous_start AND c.outcome_end
    JOIN eligible_clients AS e USING (client_hash_id)
    GROUP BY 1, 2, 3
)
SELECT
    *,
    LN(1 + current_impressions) AS log_current_impressions,
    1.0 * (current_impressions - previous_impressions) / NULLIF(previous_impressions, 0) AS impression_change_pct,
    1.0 * (current_clicks - previous_clicks) / NULLIF(previous_clicks, 0) AS click_change_pct,
    100.0 * current_clicks / NULLIF(current_impressions, 0) AS current_ctr,
    current_avg_position - previous_avg_position AS position_change,
    CASE WHEN next_impressions < 0.80 * current_impressions THEN 1 ELSE 0 END AS is_declining_proxy
FROM windows
WHERE current_impressions >= 100
"""


def build_feature_frame(connection: duckdb.DuckDBPyConnection, cache_path: Path) -> pd.DataFrame:
    """Run the final remote aggregation once and cache the page-level result locally."""
    frame = connection.sql(feature_sql()).df()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame
