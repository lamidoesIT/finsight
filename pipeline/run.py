"""
FINSIGHT batch pipeline - orchestration and CLI.

    python -m pipeline.run                         # process the inbox
    python -m pipeline.run --full-refresh          # ignore what is already loaded
    python -m pipeline.run --input some/folder     # a different source folder
    python -m pipeline.run --dry-run               # extract + check, write nothing

Exit code is 0 on success, 1 when a quality check fails or nothing could be read.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

from . import extract, load, quality, transform
from .config import DB_PATH, INBOX_DIR

log = logging.getLogger("finsight.run")


def setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
    )


def run(inbox: Path, db_path: Path, full_refresh=False, dry_run=False) -> int:
    run_id = uuid.uuid4().hex[:12]
    started = pd.Timestamp.now()
    t0 = time.time()

    log.info("=" * 68)
    log.info("FINSIGHT pipeline run %s", run_id)
    log.info("inbox     : %s", inbox)
    log.info("warehouse : %s%s", db_path, "  (DRY RUN, nothing written)" if dry_run else "")
    log.info("=" * 68)

    if not inbox.is_dir():
        log.error("inbox folder does not exist: %s", inbox)
        return 1

    con = load.connect(db_path)
    try:
        known = set() if full_refresh else load.known_file_hashes(con)
        if known:
            log.info("%d statement(s) already in the warehouse will be skipped", len(known))

        # --- bronze ---------------------------------------------------
        log.info("--- extract (bronze) ---")
        bronze, rep = extract.extract_folder(inbox, known_hashes=known)
        for name, why in rep["failed"]:
            log.error("unreadable: %s - %s", name, why)

        if bronze.empty:
            log.warning("no new transactions to process")
            if not dry_run:
                load.log_run(con, run_id=run_id, started_at=started,
                             finished_at=pd.Timestamp.now(),
                             files_loaded=0, files_skipped=len(rep["skipped"]),
                             files_failed=len(rep["failed"]), rows_loaded=0,
                             status="empty", notes="no new files")
            return 0 if not rep["failed"] else 1

        # --- silver ---------------------------------------------------
        log.info("--- transform (silver) ---")
        silver = transform.transform(bronze)
        log.info("silver: %d row(s), %d statement(s)",
                 len(silver), silver["statement_id"].nunique())

        # --- quality --------------------------------------------------
        log.info("--- quality checks ---")
        checks = quality.run_checks(silver)
        print(quality.report(checks))
        quality.enforce(checks)

        if dry_run:
            log.info("dry run: skipping the load")
            print(_preview(silver))
            return 0

        # --- gold -----------------------------------------------------
        log.info("--- load (gold) ---")
        load.write_parquet(bronze, silver, run_id)
        rows = load.load(con, silver, bronze)

        warnings = [c["check"] for c in checks if not c["passed"]]
        load.log_run(con, run_id=run_id, started_at=started,
                     finished_at=pd.Timestamp.now(),
                     files_loaded=len(rep["loaded"]), files_skipped=len(rep["skipped"]),
                     files_failed=len(rep["failed"]), rows_loaded=rows,
                     status="ok" if not warnings else "ok_with_warnings",
                     notes=", ".join(warnings) or None)

        log.info("=" * 68)
        log.info("done in %.1fs - %d file(s) loaded, %d skipped, %d failed, %d fact row(s)",
                 time.time() - t0, len(rep["loaded"]), len(rep["skipped"]),
                 len(rep["failed"]), rows)
        print(_summary(con))
        return 0

    except quality.QualityError as e:
        log.error("%s", e)
        load.log_run(con, run_id=run_id, started_at=started,
                     finished_at=pd.Timestamp.now(), files_loaded=0, files_skipped=0,
                     files_failed=0, rows_loaded=0, status="failed", notes=str(e))
        return 1
    finally:
        con.close()


def _preview(silver: pd.DataFrame) -> str:
    cols = ["txn_date", "description", "amount_zar", "category", "confidence"]
    return "\nSample of what would be loaded:\n" + \
        silver[cols].head(10).to_string(index=False)


def _summary(con) -> str:
    stmts, rows = con.execute(
        "SELECT (SELECT COUNT(*) FROM dim_statement), (SELECT COUNT(*) FROM fct_transaction)"
    ).fetchone()
    top = con.execute("""
        SELECT category, n_transactions, total_zar, avg_confidence
        FROM mart_spend_by_category LIMIT 10
    """).df()
    review = con.execute("SELECT COUNT(*) FROM mart_review_queue").fetchone()[0]
    return (f"\nWarehouse now holds {rows} transaction(s) from {stmts} statement(s).\n"
            f"\nSpend by category:\n{top.to_string(index=False)}\n"
            f"\n{review} transaction(s) in the review queue "
            f"(SELECT * FROM mart_review_queue).\n")


def main(argv=None):
    p = argparse.ArgumentParser(description="FINSIGHT batch ETL: statement PDFs -> DuckDB")
    p.add_argument("--input", type=Path, default=INBOX_DIR, help="folder of statement PDFs")
    p.add_argument("--db", type=Path, default=DB_PATH, help="DuckDB warehouse file")
    p.add_argument("--full-refresh", action="store_true",
                   help="reprocess statements already in the warehouse")
    p.add_argument("--dry-run", action="store_true",
                   help="extract and check, but write nothing")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    setup_logging(args.verbose)
    return run(args.input, args.db, full_refresh=args.full_refresh, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
