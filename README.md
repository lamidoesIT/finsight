---
title: FINSIGHT
emoji: 💸
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# FINSIGHT — Bank Statement Analyser

Upload a South African bank statement (PDF) and FINSIGHT extracts every transaction,
sorts each one into a spending category with a trained XGBoost model, and explains the
result in plain English — plus budget-vs-actual comparison and savings tracking.

**Live demo:** _(paste your Hugging Face Space URL here once deployed)_

> Demo only. Use the sample statements in `examples/`, not real personal statements.

---

## What it does

| Stage | How |
|---|---|
| Read the PDF | `pdfplumber` text extraction with a running-balance sanity check (`pdf_extraction.py`) |
| Feature engineering | Date parts, inferred purchase date, time-of-day, online flag, transaction type (`build_features`) |
| Categorise | TF-IDF on description + scaled numeric features → XGBoost multi-class classifier |
| Confidence | Top-1 probability, plus the runner-up category so low-confidence rows can be reviewed |
| Explain | Plain-language insights, donut / daily-stacked / confidence charts (Plotly) |
| Budget | Per-category budget, planned-vs-spent comparison, savings vs goal |
| Accounts | SQLite users with PBKDF2-hashed passwords, saved budgets, analysis history (`users.py`) |

**Stack:** Python · Gradio · XGBoost · scikit-learn · pandas · Plotly · pdfplumber · SQLite

---

## Run it on your own machine

You need **Python 3.10–3.12** installed.

### 1. Open a terminal in this folder

```bash
cd finsight-app
```

### 2. Create a virtual environment

A virtual environment keeps this project's packages separate from the rest of your computer.

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

### 3. Install the packages

```bash
pip install -r requirements.txt
```

This takes a few minutes the first time.

### 4. Start the app

```bash
python app.py
```

You'll see a line like `Running on local URL: http://127.0.0.1:7860`.
Open that address in your browser.

### 5. Use it

1. Create an account (any username/password — it's stored locally in `finsight_users.db`).
2. Upload one of the PDFs in `examples/`, or pick one from the samples row.
3. Press **"Show me where my money went."**
4. Set a budget on the **Budget** tab, then press **Compare**.

To stop the app, press `Ctrl+C` in the terminal.

---

## Batch ETL pipeline (statements → DuckDB warehouse)

The web app handles one statement at a time, interactively. The pipeline does the
other half: point it at a folder of statement PDFs and it loads them all into a
queryable warehouse.

```bash
python -m pipeline.run                    # process data/statements_inbox
python -m pipeline.run --dry-run          # parse + check, write nothing
python -m pipeline.run --full-refresh     # reprocess statements already loaded
python -m pipeline.run --input some/dir --db warehouse/other.duckdb
```

30 sample statements ship in `data/statements_inbox/`, so it runs out of the box.

### Layers

```
data/statements_inbox/*.pdf
        │
        ▼   extract.py        reuses pdf_extraction.py
   BRONZE   raw rows, untouched, with lineage (file, sha256, page, line_no)
        │                     → warehouse/bronze/*.parquet
        ▼   transform.py      reuses finsight_core.py (same model as the app)
   SILVER   typed + cleaned; inflow/outflow split; category + confidence
        │                     → warehouse/silver/*.parquet
        ▼   quality.py        8 checks; fatal ones abort before the warehouse is touched
    GOLD    star schema + marts in DuckDB
                              → warehouse/finsight.duckdb
```

The model and feature engineering live in **`finsight_core.py`**, imported by both
`app.py` and the pipeline. There is exactly one implementation of "how a transaction
becomes a category", so the app and the warehouse can never disagree.

### Warehouse schema

| Table | Grain |
|---|---|
| `fct_transaction` | one row per transaction line (the fact table) |
| `dim_date` | one row per calendar day, with weekday/weekend flags |
| `dim_category` | one row per spending group, `is_spending` flag |
| `dim_statement` | one row per statement file, with its period and row count |
| `run_log` | one row per pipeline run: counts, status, warnings |

Views for querying and dashboards:

`mart_spend_by_category` · `mart_monthly_spend` · `mart_statement_summary`
`mart_review_queue` · `mart_weekday_spend`

```sql
-- where the money goes
SELECT * FROM mart_spend_by_category;

-- transactions the model was unsure about, worst first
SELECT * FROM mart_review_queue LIMIT 20;

-- month-on-month movement in one category
SELECT year, month_name, total_zar
FROM mart_monthly_spend
WHERE category = 'Groceries'
ORDER BY year, month;
```

Query it from the CLI with `duckdb warehouse/finsight.duckdb`, from Python with
`duckdb.connect(...)`, or point Power BI / Tableau at the file via the DuckDB
ODBC driver.

### Design decisions worth knowing

- **Idempotent.** Statements are keyed by a SHA-256 of their contents, and rows by a
  deterministic `txn_id` (`statement_id:page:line_no`). Re-running the same inbox
  skips what is already loaded; `--full-refresh` replaces per statement rather than
  appending. Running the pipeline three times leaves 1 210 rows, not 3 630.
- **Bad files do not sink the batch.** An unreadable PDF is logged and reported;
  the other statements still load. Both deliberately broken test PDFs in the original
  repo are handled this way.
- **Income is not a spending category.** Only outflows go through the model;
  inflows are labelled `Income` with a null confidence and excluded from every
  spending mart.
- **Low confidence is surfaced, not hidden.** Rows under 50% confidence are flagged
  `needs_review` and collected in `mart_review_queue` instead of being silently
  trusted or dropped. On the 30 sample statements that is ~7% of transactions.
- **Quality gate before the load.** Duplicate keys, null dates, non-positive amounts
  and uncategorised rows abort the run. Balance-check failures, low confidence and
  future dates warn and are recorded in `run_log`.

### Tests

```bash
pytest -q        # 14 tests covering transform, quality gates and idempotency
```

---

## Deploy it (so it lives on the internet)

This is a **Python app with a trained model**, so it needs a server that runs Python.
GitHub Pages only serves static files, so it **cannot** host the app itself — but it can
host a landing page that embeds the running app. That's what `portfolio/index.html` does.

### Option A — Hugging Face Spaces (free, recommended)

1. Create a free account at <https://huggingface.co>.
2. Click your avatar → **New Space**.
   - **Space name:** `finsight`
   - **SDK:** Gradio
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public
3. On the Space page, open the **Files** tab → **Add file** → **Upload files**, and upload:
   `app.py`, `pdf_extraction.py`, `users.py`, `requirements.txt`,
   `finsight_model.joblib`, `logo.png`, `favicon.png`, `README.md`, and the `examples/` folder.
4. The Space builds itself (a few minutes). Your app is then live at
   `https://huggingface.co/spaces/<your-username>/finsight`.
5. Optional: Space **Settings → Persistent storage** keeps user accounts across restarts.
   Without it, the accounts database resets whenever the Space sleeps.

### Option B — Render / Railway / Fly.io

Any host that runs a Python web process works. Start command:

```bash
python app.py
```

Gradio reads the `PORT` environment variable on most of these hosts; if yours needs it
explicit, change the last line of `app.py` to:

```python
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), **_launch_kwargs)
```

---

## Put it on your GitHub and portfolio

### 1. Push the code to GitHub

```bash
git init
git add .
git commit -m "FINSIGHT: bank statement analyser"
git branch -M main
git remote add origin https://github.com/<your-username>/finsight.git
git push -u origin main
```

### 2. Publish the landing page

`portfolio/index.html` is a standalone page that describes the project and embeds the
live app in an iframe.

1. Open `portfolio/index.html` and replace `YOUR-USERNAME/finsight` with your real Space path
   (there is a single `SPACE_URL` line near the top of the file).
2. Copy `portfolio/index.html` into your portfolio website repo as e.g. `projects/finsight.html`,
   **or** publish it on its own with GitHub Pages: repo **Settings → Pages → Source: main / root**.

### 3. Or just embed the app in a page you already have

```html
<iframe
  src="https://<your-username>-finsight.hf.space"
  title="FINSIGHT"
  width="100%"
  height="900"
  style="border:1px solid #E2EBF0;border-radius:18px"
  allow="clipboard-write"
></iframe>
```

---

## Files

```
app.py                  Gradio UI: charts, insights, budget logic, accounts
finsight_core.py        Model + feature engineering (shared by the app and the pipeline)
pdf_extraction.py       PDF → transactions dataframe
users.py                SQLite accounts, PBKDF2 passwords, saved budgets, history
finsight_model.joblib   Trained bundle: tfidf + imputer + scaler + XGBoost + label encoder

pipeline/
  config.py             Paths and thresholds
  extract.py            Bronze: PDFs → raw rows with lineage
  transform.py          Silver: typed, cleaned, categorised
  quality.py            Data-quality gate (fatal vs warning)
  load.py               Gold: star schema + marts in DuckDB
  run.py                Orchestration and CLI

tests/test_pipeline.py  Pytest suite for the pipeline
data/statements_inbox/  30 sample statements the pipeline reads
warehouse/              Generated: DuckDB file + bronze/silver parquet (gitignored)
examples/               Sample statements for the web app
portfolio/index.html    Static landing page that embeds the live app
requirements.txt        Python dependencies
```

## Credits

Original FINSIGHT project and trained model by the Eduvos capstone group
(Hugging Face Space: `JenniferAnnKok/FINSIGHT`). This repository is a UI refresh and
deployment packaging of that work — the extraction, model and analysis logic are unchanged.

## License

MIT — see the original project for the model's terms.
