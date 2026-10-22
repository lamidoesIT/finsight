"""
Silver layer: bronze rows -> typed, cleaned, categorised transactions.

Decisions made here:
  * money in and money out are separated (only outflows get a spending category)
  * the trained model, not a rule table, assigns the category
  * every row keeps its confidence and the runner-up category, and rows under
    the confidence floor are flagged for review instead of being dropped
"""
from __future__ import annotations

import logging

import pandas as pd

import finsight_core
from .config import REVIEW_CONFIDENCE

log = logging.getLogger("finsight.transform")

SILVER_COLUMNS = [
    "txn_id", "statement_id", "source_file", "page", "line_no",
    "txn_date", "description", "amount_zar", "direction", "txn_type",
    "category", "confidence", "second_guess", "second_guess_prob",
    "needs_review", "balance_ok",
]


def _txn_id(row) -> str:
    """Deterministic row key, so re-running a statement overwrites rather than duplicates."""
    return f"{row['statement_id']}:{int(row['page'])}:{int(row['line_no'])}"


def transform(bronze: pd.DataFrame) -> pd.DataFrame:
    if bronze.empty:
        return pd.DataFrame(columns=SILVER_COLUMNS)

    df = bronze.copy()
    df["amount"] = pd.to_numeric(df["amount_raw"], errors="coerce")
    df["description_sa"] = df["description_raw"].fillna("").astype(str)
    df["date"] = df["txn_date_raw"]

    dropped = int(df["amount"].isna().sum())
    if dropped:
        log.warning("dropping %d row(s) with an unreadable amount", dropped)
    df = df.dropna(subset=["amount"])

    df["direction"] = df["amount"].apply(lambda a: "in" if a > 0 else "out")

    outflows = df[df["direction"] == "out"].copy()
    inflows = df[df["direction"] == "in"].copy()
    log.info("%d outflow(s) to categorise, %d inflow(s) left uncategorised",
             len(outflows), len(inflows))

    # Only money going out gets a spending category; income is not a "spend group".
    if not outflows.empty:
        scored = finsight_core.categorise(outflows)
        scored = scored.dropna(subset=["txn_date"])
    else:
        scored = pd.DataFrame()

    if not inflows.empty:
        inflows["txn_date"] = pd.to_datetime(inflows["date"], errors="coerce")
        inflows["amount_zar"] = inflows["amount"].abs().round(2)
        inflows["txn_type"] = inflows["description_sa"].apply(finsight_core.get_txn_type)
        for col, val in [("predicted_category", "Income"), ("confidence", pd.NA),
                         ("second_guess", pd.NA), ("second_guess_prob", pd.NA)]:
            inflows[col] = val
        inflows = inflows.dropna(subset=["txn_date"])

    combined = pd.concat([f for f in (scored, inflows) if not f.empty], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=SILVER_COLUMNS)

    combined["txn_id"] = combined.apply(_txn_id, axis=1)
    combined["needs_review"] = (
        pd.to_numeric(combined["confidence"], errors="coerce") < REVIEW_CONFIDENCE
    ).fillna(False)

    silver = pd.DataFrame({
        "txn_id": combined["txn_id"],
        "statement_id": combined["statement_id"],
        "source_file": combined["source_file"],
        "page": combined["page"].astype(int),
        "line_no": combined["line_no"].astype(int),
        "txn_date": pd.to_datetime(combined["txn_date"]).dt.date,
        "description": combined["description_sa"],
        "amount_zar": combined["amount_zar"].astype(float),
        "direction": combined["direction"],
        "txn_type": combined["txn_type"],
        "category": combined["predicted_category"].astype(str),
        "confidence": pd.to_numeric(combined["confidence"], errors="coerce"),
        "second_guess": combined["second_guess"].astype("object"),
        "second_guess_prob": pd.to_numeric(combined["second_guess_prob"], errors="coerce"),
        "needs_review": combined["needs_review"].astype(bool),
        "balance_ok": combined["balance_ok"].fillna(True).astype(bool),
    })

    # Same statement parsed twice in one batch -> keep one row per txn_id.
    before = len(silver)
    silver = silver.drop_duplicates(subset="txn_id", keep="last")
    if len(silver) != before:
        log.warning("removed %d duplicate txn_id row(s)", before - len(silver))

    return silver[SILVER_COLUMNS].sort_values(["statement_id", "page", "line_no"])
