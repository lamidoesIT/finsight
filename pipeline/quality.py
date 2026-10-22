"""
Data-quality checks that run between silver and gold.

A check returns (name, passed, detail). Failures are either WARN (recorded and
the run continues) or FAIL (the load is abandoned so bad data never reaches the
warehouse). The point is that a silent wrong number is worse than a loud stop.
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import MAX_BALANCE_FAIL_RATE, REVIEW_CONFIDENCE

log = logging.getLogger("finsight.quality")


class QualityError(RuntimeError):
    pass


def _check(name, passed, detail, fatal=False):
    return {"check": name, "passed": bool(passed), "detail": detail, "fatal": fatal}


def run_checks(silver: pd.DataFrame) -> list[dict]:
    checks = []

    checks.append(_check("non_empty", not silver.empty,
                         f"{len(silver)} row(s) in silver", fatal=True))
    if silver.empty:
        return checks

    dupes = int(silver["txn_id"].duplicated().sum())
    checks.append(_check("unique_txn_id", dupes == 0,
                         f"{dupes} duplicate txn_id(s)", fatal=True))

    null_dates = int(silver["txn_date"].isna().sum())
    checks.append(_check("no_null_dates", null_dates == 0,
                         f"{null_dates} row(s) with no date", fatal=True))

    bad_amounts = int((silver["amount_zar"] <= 0).sum())
    checks.append(_check("positive_amounts", bad_amounts == 0,
                         f"{bad_amounts} row(s) with a non-positive amount", fatal=True))

    null_cat = int(silver["category"].isna().sum() + (silver["category"] == "").sum())
    checks.append(_check("every_row_categorised", null_cat == 0,
                         f"{null_cat} row(s) with no category", fatal=True))

    # Non-fatal signals: these say "look at this", not "stop".
    fail_rate = 1 - float(silver["balance_ok"].mean())
    checks.append(_check(
        "balance_crosscheck", fail_rate <= MAX_BALANCE_FAIL_RATE,
        f"{fail_rate:.1%} of rows failed the running-balance check "
        f"(threshold {MAX_BALANCE_FAIL_RATE:.0%})"))

    out = silver[silver["direction"] == "out"]
    if not out.empty:
        review_rate = float(out["needs_review"].mean())
        checks.append(_check(
            "model_confidence", review_rate <= 0.25,
            f"{review_rate:.1%} of outflows are below the {REVIEW_CONFIDENCE:.0%} "
            "confidence floor and are queued for review"))

    future = int((pd.to_datetime(silver["txn_date"]) > pd.Timestamp.now()).sum())
    checks.append(_check("no_future_dates", future == 0,
                         f"{future} row(s) dated in the future"))

    return checks


def report(checks: list[dict]) -> str:
    lines = []
    for c in checks:
        mark = "PASS" if c["passed"] else ("FAIL" if c["fatal"] else "WARN")
        lines.append(f"  [{mark}] {c['check']:<24} {c['detail']}")
    return "\n".join(lines)


def enforce(checks: list[dict]):
    fatal = [c for c in checks if c["fatal"] and not c["passed"]]
    if fatal:
        names = ", ".join(c["check"] for c in fatal)
        raise QualityError(f"aborting load, failed check(s): {names}")
    for c in checks:
        if not c["passed"]:
            log.warning("quality warning: %s - %s", c["check"], c["detail"])
