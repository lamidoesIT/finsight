"""
Bronze layer: statement PDFs -> one raw transaction table.

Nothing is cleaned or interpreted here beyond what the parser returns. Each row
keeps its lineage (which file it came from, which page, which line) so anything
odd downstream can be traced back to a spot in a PDF.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from pdf_extraction import extract_transactions, ExtractionError

log = logging.getLogger("finsight.extract")

BRONZE_COLUMNS = [
    "statement_id", "source_file", "file_sha256", "page", "line_no",
    "txn_date_raw", "description_raw", "amount_raw", "balance_raw", "balance_ok",
]


def file_fingerprint(path: Path) -> str:
    """Content hash: the same statement re-dropped in the inbox is not loaded twice."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def statement_id(path: Path, digest: str) -> str:
    """Stable, readable surrogate key: filename stem + first 8 of the content hash."""
    return f"{path.stem}__{digest[:8]}"


def extract_one(path: Path) -> pd.DataFrame:
    """Parse a single PDF into bronze rows. Raises ExtractionError if unreadable."""
    digest = file_fingerprint(path)
    df = extract_transactions(str(path))

    out = pd.DataFrame({
        "statement_id": statement_id(path, digest),
        "source_file": path.name,
        "file_sha256": digest,
        "page": df["page"],
        "line_no": range(len(df)),
        "txn_date_raw": df["date"],
        "description_raw": df["description_sa"],
        "amount_raw": df["amount"],
        "balance_raw": df.get("balance"),
        "balance_ok": df.get("balance_ok", True),
    })
    return out[BRONZE_COLUMNS]


def extract_folder(inbox: Path, known_hashes: set[str] | None = None):
    """
    Parse every PDF in the inbox.

    Returns (bronze_dataframe, run_report). Files that fail to parse are reported
    rather than raised: one bad statement must not sink the whole batch.
    """
    known_hashes = known_hashes or set()
    frames, loaded, skipped, failed = [], [], [], []

    pdfs = sorted(p for p in inbox.glob("*.pdf"))
    log.info("found %d PDF(s) in %s", len(pdfs), inbox)

    for path in pdfs:
        try:
            digest = file_fingerprint(path)
            if digest in known_hashes:
                log.info("skip %s (already in warehouse)", path.name)
                skipped.append(path.name)
                continue
            df = extract_one(path)
            if df.empty:
                failed.append((path.name, "no transactions found"))
                log.warning("%s produced no rows", path.name)
                continue
            frames.append(df)
            loaded.append(path.name)
            log.info("parsed %-28s %3d rows", path.name, len(df))
        except ExtractionError as e:
            failed.append((path.name, str(e)))
            log.warning("%s could not be read: %s", path.name, e)
        except Exception as e:  # a malformed PDF must not kill the batch
            failed.append((path.name, f"{type(e).__name__}: {e}"))
            log.exception("%s failed unexpectedly", path.name)

    bronze = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=BRONZE_COLUMNS))
    report = {"loaded": loaded, "skipped": skipped, "failed": failed, "rows": len(bronze)}
    return bronze, report
