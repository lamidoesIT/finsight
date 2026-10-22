"""Paths and tunables for the FINSIGHT batch pipeline."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where statement PDFs are picked up from, and where the warehouse is written.
INBOX_DIR = Path(os.environ.get("FINSIGHT_INBOX", PROJECT_ROOT / "data" / "statements_inbox"))
WAREHOUSE_DIR = Path(os.environ.get("FINSIGHT_WAREHOUSE_DIR", PROJECT_ROOT / "warehouse"))
DB_PATH = Path(os.environ.get("FINSIGHT_WAREHOUSE", WAREHOUSE_DIR / "finsight.duckdb"))

# Parquet landing zone: every run writes bronze/silver so a load can be replayed
# without re-reading the PDFs (PDF parsing is by far the slowest step).
BRONZE_DIR = WAREHOUSE_DIR / "bronze"
SILVER_DIR = WAREHOUSE_DIR / "silver"

# Rows below this model confidence are flagged for human review rather than
# being trusted in the reporting marts.
REVIEW_CONFIDENCE = 0.50

# A run fails if more than this share of rows fail the balance cross-check.
MAX_BALANCE_FAIL_RATE = 0.20
