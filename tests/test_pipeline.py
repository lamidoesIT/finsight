"""
Tests for the FINSIGHT batch pipeline.

Run from the project root:   pytest -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import load, quality, transform  # noqa: E402
from pipeline.extract import file_fingerprint, statement_id  # noqa: E402


def _bronze_row(**over):
    row = {
        "statement_id": "STMT001__abcd1234",
        "source_file": "STMT001.pdf",
        "file_sha256": "abcd1234" * 8,
        "page": 1,
        "line_no": 0,
        "txn_date_raw": pd.Timestamp("2023-06-01").date(),
        "description_raw": "WOOLWORTHS FOOD #1036",
        "amount_raw": -516.93,
        "balance_raw": 10000.0,
        "balance_ok": True,
    }
    row.update(over)
    return row


def _bronze(rows):
    return pd.DataFrame([_bronze_row(**r) for r in rows])


# ---------------------------------------------------------------- extract

def test_fingerprint_is_stable_and_content_based(tmp_path):
    a, b, c = tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    c.write_bytes(b"different")
    assert file_fingerprint(a) == file_fingerprint(b)
    assert file_fingerprint(a) != file_fingerprint(c)


def test_statement_id_includes_hash_prefix(tmp_path):
    p = tmp_path / "STMT009_highveld.pdf"
    p.write_bytes(b"x")
    sid = statement_id(p, "deadbeefcafe")
    assert sid == "STMT009_highveld__deadbeef"


# -------------------------------------------------------------- transform

def test_outflows_categorised_inflows_are_income():
    bronze = _bronze([
        {"line_no": 0, "amount_raw": -516.93, "description_raw": "WOOLWORTHS FOOD"},
        {"line_no": 1, "amount_raw": 28200.0, "description_raw": "SALARY PAYMENT"},
    ])
    silver = transform.transform(bronze)

    assert set(silver["direction"]) == {"out", "in"}
    income = silver[silver["direction"] == "in"].iloc[0]
    assert income["category"] == "Income"
    assert pd.isna(income["confidence"])

    spend = silver[silver["direction"] == "out"].iloc[0]
    assert spend["category"] != "Income"
    assert 0 <= spend["confidence"] <= 1


def test_amounts_are_positive_magnitudes():
    silver = transform.transform(_bronze([{"amount_raw": -250.0}]))
    assert (silver["amount_zar"] > 0).all()


def test_txn_id_is_deterministic_across_runs():
    bronze = _bronze([{"line_no": 0}, {"line_no": 1, "amount_raw": -12.0}])
    first = transform.transform(bronze)["txn_id"].tolist()
    second = transform.transform(bronze)["txn_id"].tolist()
    assert first == second
    assert len(set(first)) == len(first)


def test_unreadable_amount_is_dropped_not_crashed():
    bronze = _bronze([{"line_no": 0}, {"line_no": 1, "amount_raw": None}])
    silver = transform.transform(bronze)
    assert len(silver) == 1


def test_low_confidence_rows_are_flagged_for_review():
    silver = transform.transform(_bronze([
        {"line_no": 0, "description_raw": "ZZQX 4471 UNKNOWN REF"},
    ]))
    row = silver.iloc[0]
    assert row["needs_review"] == (row["confidence"] < 0.50)


# ---------------------------------------------------------------- quality

def test_quality_passes_on_clean_data():
    silver = transform.transform(_bronze([{"line_no": 0}, {"line_no": 1, "amount_raw": -40.0}]))
    checks = quality.run_checks(silver)
    quality.enforce(checks)  # must not raise
    assert all(c["passed"] for c in checks if c["fatal"])


def test_quality_blocks_duplicate_txn_ids():
    silver = transform.transform(_bronze([{"line_no": 0}]))
    silver = pd.concat([silver, silver], ignore_index=True)
    with pytest.raises(quality.QualityError):
        quality.enforce(quality.run_checks(silver))


def test_quality_blocks_empty_batch():
    with pytest.raises(quality.QualityError):
        quality.enforce(quality.run_checks(pd.DataFrame()))


def test_balance_failures_warn_but_do_not_block():
    silver = transform.transform(_bronze([
        {"line_no": 0, "balance_ok": False},
        {"line_no": 1, "amount_raw": -40.0, "balance_ok": False},
    ]))
    checks = quality.run_checks(silver)
    quality.enforce(checks)  # warns, does not raise
    balance = next(c for c in checks if c["check"] == "balance_crosscheck")
    assert balance["passed"] is False and balance["fatal"] is False


# ------------------------------------------------------------------- load

def test_load_is_idempotent(tmp_path):
    bronze = _bronze([{"line_no": 0}, {"line_no": 1, "amount_raw": -40.0}])
    silver = transform.transform(bronze)
    con = load.connect(tmp_path / "t.duckdb")
    try:
        load.load(con, silver, bronze)
        load.load(con, silver, bronze)          # same batch again
        n = con.execute("SELECT COUNT(*) FROM fct_transaction").fetchone()[0]
        stmts = con.execute("SELECT COUNT(*) FROM dim_statement").fetchone()[0]
        assert n == len(silver)
        assert stmts == 1
    finally:
        con.close()


def test_known_hashes_round_trip(tmp_path):
    bronze = _bronze([{"line_no": 0}])
    silver = transform.transform(bronze)
    con = load.connect(tmp_path / "t.duckdb")
    try:
        load.load(con, silver, bronze)
        assert bronze["file_sha256"].iloc[0] in load.known_file_hashes(con)
    finally:
        con.close()


def test_marts_are_queryable(tmp_path):
    bronze = _bronze([{"line_no": 0}, {"line_no": 1, "amount_raw": 5000.0}])
    silver = transform.transform(bronze)
    con = load.connect(tmp_path / "t.duckdb")
    try:
        load.load(con, silver, bronze)
        for view in ("mart_spend_by_category", "mart_monthly_spend",
                     "mart_statement_summary", "mart_review_queue", "mart_weekday_spend"):
            con.execute(f"SELECT * FROM {view}").df()
        # income must not be counted as spending
        spend = con.execute("SELECT category FROM mart_spend_by_category").df()
        assert "Income" not in spend["category"].tolist()
    finally:
        con.close()
