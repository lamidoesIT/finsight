"""
Gold layer: silver transactions -> a small star schema in DuckDB, plus marts.

  dim_date        one row per calendar day seen
  dim_category    one row per spending group the model can predict
  dim_statement   one row per statement file loaded
  fct_transaction the grain: one row per transaction line
  mart_*          pre-aggregated views for dashboards / SQL practice

Loads are idempotent: rows are keyed on txn_id and deleted-then-inserted per
statement, so re-running the same batch leaves the warehouse unchanged.
"""
from __future__ import annotations

import logging

import duckdb
import pandas as pd

from .config import DB_PATH, WAREHOUSE_DIR

log = logging.getLogger("finsight.load")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dim_category (
    category      VARCHAR PRIMARY KEY,
    is_spending   BOOLEAN
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_key      DATE PRIMARY KEY,
    year          INTEGER,
    month         INTEGER,
    day           INTEGER,
    month_name    VARCHAR,
    day_of_week   VARCHAR,
    is_weekend    BOOLEAN
);

CREATE TABLE IF NOT EXISTS dim_statement (
    statement_id  VARCHAR PRIMARY KEY,
    source_file   VARCHAR,
    file_sha256   VARCHAR,
    loaded_at     TIMESTAMP,
    n_rows        INTEGER,
    period_start  DATE,
    period_end    DATE
);

CREATE TABLE IF NOT EXISTS fct_transaction (
    txn_id            VARCHAR PRIMARY KEY,
    statement_id      VARCHAR,
    date_key          DATE,
    category          VARCHAR,
    description       VARCHAR,
    amount_zar        DOUBLE,
    direction         VARCHAR,
    txn_type          VARCHAR,
    confidence        DOUBLE,
    second_guess      VARCHAR,
    second_guess_prob DOUBLE,
    needs_review      BOOLEAN,
    balance_ok        BOOLEAN,
    page              INTEGER,
    line_no           INTEGER
);

CREATE TABLE IF NOT EXISTS run_log (
    run_id          VARCHAR,
    started_at      TIMESTAMP,
    finished_at     TIMESTAMP,
    files_loaded    INTEGER,
    files_skipped   INTEGER,
    files_failed    INTEGER,
    rows_loaded     INTEGER,
    status          VARCHAR,
    notes           VARCHAR
);
"""

MARTS_SQL = """
CREATE OR REPLACE VIEW mart_spend_by_category AS
SELECT
    f.category,
    COUNT(*)                                   AS n_transactions,
    ROUND(SUM(f.amount_zar), 2)                AS total_zar,
    ROUND(AVG(f.amount_zar), 2)                AS avg_zar,
    ROUND(MEDIAN(f.amount_zar), 2)             AS median_zar,
    ROUND(AVG(f.confidence), 3)                AS avg_confidence,
    SUM(CASE WHEN f.needs_review THEN 1 ELSE 0 END) AS n_needs_review
FROM fct_transaction f
WHERE f.direction = 'out'
GROUP BY f.category
ORDER BY total_zar DESC;

CREATE OR REPLACE VIEW mart_monthly_spend AS
SELECT
    d.year,
    d.month,
    d.month_name,
    f.category,
    COUNT(*)                    AS n_transactions,
    ROUND(SUM(f.amount_zar), 2) AS total_zar
FROM fct_transaction f
JOIN dim_date d ON d.date_key = f.date_key
WHERE f.direction = 'out'
GROUP BY d.year, d.month, d.month_name, f.category
ORDER BY d.year, d.month, total_zar DESC;

CREATE OR REPLACE VIEW mart_statement_summary AS
SELECT
    s.statement_id,
    s.source_file,
    s.period_start,
    s.period_end,
    COUNT(*) FILTER (WHERE f.direction = 'out')                      AS n_outflows,
    ROUND(SUM(f.amount_zar) FILTER (WHERE f.direction = 'out'), 2)   AS total_out_zar,
    ROUND(SUM(f.amount_zar) FILTER (WHERE f.direction = 'in'), 2)    AS total_in_zar,
    ROUND(AVG(f.confidence) FILTER (WHERE f.direction = 'out'), 3)   AS avg_confidence
FROM dim_statement s
JOIN fct_transaction f ON f.statement_id = s.statement_id
GROUP BY s.statement_id, s.source_file, s.period_start, s.period_end
ORDER BY s.statement_id;

CREATE OR REPLACE VIEW mart_review_queue AS
SELECT
    f.txn_id, f.statement_id, f.date_key, f.description,
    f.amount_zar, f.category, f.confidence,
    f.second_guess, f.second_guess_prob
FROM fct_transaction f
WHERE f.needs_review
ORDER BY f.confidence ASC;

CREATE OR REPLACE VIEW mart_weekday_spend AS
SELECT
    d.day_of_week,
    d.is_weekend,
    COUNT(*)                    AS n_transactions,
    ROUND(SUM(f.amount_zar), 2) AS total_zar
FROM fct_transaction f
JOIN dim_date d ON d.date_key = f.date_key
WHERE f.direction = 'out'
GROUP BY d.day_of_week, d.is_weekend
ORDER BY total_zar DESC;
"""


def connect(db_path=None):
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA_SQL)
    return con


def known_file_hashes(con) -> set[str]:
    rows = con.execute("SELECT file_sha256 FROM dim_statement").fetchall()
    return {r[0] for r in rows}


def _upsert_dim_category(con, categories):
    df = pd.DataFrame({
        "category": list(categories),
        "is_spending": [c != "Income" for c in categories],
    })
    con.register("_cat", df)
    con.execute("""
        INSERT INTO dim_category
        SELECT * FROM _cat
        WHERE category NOT IN (SELECT category FROM dim_category)
    """)
    con.unregister("_cat")


def _upsert_dim_date(con, dates):
    d = pd.to_datetime(pd.Series(sorted(set(dates))))
    if d.empty:
        return
    df = pd.DataFrame({
        "date_key": d.dt.date,
        "year": d.dt.year,
        "month": d.dt.month,
        "day": d.dt.day,
        "month_name": d.dt.strftime("%B"),
        "day_of_week": d.dt.strftime("%A"),
        "is_weekend": d.dt.day_of_week >= 5,
    })
    con.register("_dt", df)
    con.execute("""
        INSERT INTO dim_date
        SELECT * FROM _dt
        WHERE date_key NOT IN (SELECT date_key FROM dim_date)
    """)
    con.unregister("_dt")


def load(con, silver: pd.DataFrame, bronze: pd.DataFrame) -> int:
    """Replace-by-statement load. Returns the number of fact rows written."""
    if silver.empty:
        log.info("nothing to load")
        return 0

    _upsert_dim_category(con, sorted(silver["category"].unique()))
    _upsert_dim_date(con, silver["txn_date"])

    statements = silver["statement_id"].unique().tolist()
    con.execute(
        "DELETE FROM fct_transaction WHERE statement_id IN "
        f"({','.join('?' * len(statements))})", statements)
    con.execute(
        "DELETE FROM dim_statement WHERE statement_id IN "
        f"({','.join('?' * len(statements))})", statements)

    stmt_meta = (bronze.groupby("statement_id")
                 .agg(source_file=("source_file", "first"),
                      file_sha256=("file_sha256", "first"))
                 .reset_index())
    periods = (silver.groupby("statement_id")["txn_date"]
               .agg(period_start="min", period_end="max").reset_index())
    counts = silver.groupby("statement_id").size().rename("n_rows").reset_index()
    stmt_meta = stmt_meta.merge(periods, on="statement_id").merge(counts, on="statement_id")
    stmt_meta["loaded_at"] = pd.Timestamp.now()
    stmt_meta = stmt_meta[["statement_id", "source_file", "file_sha256",
                           "loaded_at", "n_rows", "period_start", "period_end"]]

    facts = pd.DataFrame({
        "txn_id": silver["txn_id"],
        "statement_id": silver["statement_id"],
        "date_key": silver["txn_date"],
        "category": silver["category"],
        "description": silver["description"],
        "amount_zar": silver["amount_zar"],
        "direction": silver["direction"],
        "txn_type": silver["txn_type"],
        "confidence": silver["confidence"],
        "second_guess": silver["second_guess"].astype("object"),
        "second_guess_prob": silver["second_guess_prob"],
        "needs_review": silver["needs_review"],
        "balance_ok": silver["balance_ok"],
        "page": silver["page"],
        "line_no": silver["line_no"],
    })

    con.register("_stmt", stmt_meta)
    con.register("_facts", facts)
    con.execute("INSERT INTO dim_statement SELECT * FROM _stmt")
    con.execute("INSERT INTO fct_transaction SELECT * FROM _facts")
    con.unregister("_stmt")
    con.unregister("_facts")

    con.execute(MARTS_SQL)
    log.info("loaded %d fact row(s) across %d statement(s)", len(facts), len(statements))
    return len(facts)


def write_parquet(bronze: pd.DataFrame, silver: pd.DataFrame, run_id: str):
    """Keep the layers on disk so a load can be replayed without re-parsing PDFs."""
    from .config import BRONZE_DIR, SILVER_DIR
    for folder, df, name in ((BRONZE_DIR, bronze, "bronze"), (SILVER_DIR, silver, "silver")):
        if df.empty:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}_{run_id}.parquet"
        df.to_parquet(path, index=False)
        log.info("wrote %s (%d rows)", path.relative_to(WAREHOUSE_DIR.parent), len(df))


def log_run(con, **kwargs):
    con.execute(
        "INSERT INTO run_log VALUES (?,?,?,?,?,?,?,?,?)",
        [kwargs["run_id"], kwargs["started_at"], kwargs["finished_at"],
         kwargs["files_loaded"], kwargs["files_skipped"], kwargs["files_failed"],
         kwargs["rows_loaded"], kwargs["status"], kwargs["notes"]])
