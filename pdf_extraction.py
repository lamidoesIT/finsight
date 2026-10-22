"""
FINSIGHT - pdf_extraction.py  (v2)

Extract transactions from a bank statement PDF into a DataFrame.

Approach (one method for every layout, no bank-specific code):
  1. pdfplumber gives every word on the page with its x/y position.
  2. Words are grouped into lines by their vertical position.
  3. The table header line (contains "Date", "Description" and "Balance") gives the column positions.
     Numeric columns are right-aligned, so a number belongs to the header whose right edge it lines up with.
  4. A line that starts with a date is a new transaction; a line without a date directly below it is a
     wrapped continuation of the description.
  5. Amount signs are confirmed with the running balance (previous balance + amount = balance).

Only text-based PDFs are supported; scanned statements (images) are rejected with a clear message.

v2: no text cleaning any more. The description is returned exactly as printed, in a column called
description_sa, because the v2 classifier is trained on the full statement text. Use
features.add_features() on the result to get the same extra columns as the training data.
"""

import re
from datetime import datetime

import pandas as pd
import pdfplumber

DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d %b", "%d/%m/%y", "%d-%m-%Y"]
MONEY_RE = re.compile(r"^\(?-?R?\s?\d{1,3}(?:[ ,]\d{3})*(?:\.\d{2})\)?(?:\s?(?:Cr|CR|Dr|DR))?$")
SMALL_INT_RE = re.compile(r"^-?\d{1,3}$")
THOUSANDS_TAIL_RE = re.compile(r"^\d{3}(?:[ ,]\d{3})*\.\d{2}(?:Cr|CR|Dr|DR)?$")
BALANCE_ROW_RE = re.compile(r"opening balance|balance brought forward|closing balance|balance carried forward", re.I)

IN_HEADERS = {"money in", "credits", "credit", "deposits"}
OUT_HEADERS = {"money out", "debits", "debit", "withdrawals"}
SIGNED_HEADERS = {"amount"}

LINE_TOLERANCE = 3      # pt: words closer than this vertically are on the same line
ALIGN_TOLERANCE = 6     # pt: how close a number's right edge must be to its column header's right edge
TABLE_END_GAP = 25      # pt: a vertical gap bigger than this ends the table on the page


class ExtractionError(Exception):
    pass


def parse_money(text):
    """'1 234.56' / '1,234.56Cr' / '-591.13' / '(20.00)' -> signed float (Cr = +, Dr = -)."""
    t = text.strip().replace("R", "", 1) if text.strip().startswith("R") else text.strip()
    sign = 1
    m = re.search(r"(Cr|CR|Dr|DR)$", t)
    if m:
        sign = -1 if m.group(1).lower() == "dr" else 1
        t = t[: m.start()]
    if t.startswith("(") and t.endswith(")"):
        sign, t = -1, t[1:-1]
    if t.startswith("-"):
        sign, t = -sign, t[1:]
    return sign * float(t.replace(" ", "").replace(",", ""))


def parse_date(text, year_hint):
    for fmt in DATE_FORMATS:
        try:
            d = datetime.strptime(text, fmt)
            if "%Y" not in fmt and "%y" not in fmt:
                d = d.replace(year=year_hint)
            return d.date()
        except ValueError:
            continue
    return None


def _group_lines(words):
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(w["top"] - lines[-1]["top"]) <= LINE_TOLERANCE:
            lines[-1]["words"].append(w)
        else:
            lines.append({"top": w["top"], "bottom": w["bottom"], "words": [w]})
        lines[-1]["bottom"] = max(lines[-1]["bottom"], w["bottom"])
    for ln in lines:
        ln["words"].sort(key=lambda w: w["x0"])
    return lines


def _merge_split_numbers(words):
    """'15' + '978.00' (space used as thousands separator) -> '15 978.00'."""
    out = []
    for w in words:
        if (out and SMALL_INT_RE.match(out[-1]["text"]) and THOUSANDS_TAIL_RE.match(w["text"])
                and 0 <= w["x0"] - out[-1]["x1"] < 4):
            prev = out.pop()
            w = {**w, "text": prev["text"] + " " + w["text"], "x0": prev["x0"]}
        out.append(w)
    return out


def _header_columns(line):
    """Merge header words that sit close together ('Money' 'In') into column labels with positions."""
    cols = []
    for w in line["words"]:
        if cols and w["x0"] - cols[-1]["x1"] < 6:
            cols[-1]["label"] += " " + w["text"]
            cols[-1]["x1"] = w["x1"]
        else:
            cols.append({"label": w["text"], "x0": w["x0"], "x1": w["x1"]})
    for c in cols:
        c["key"] = c["label"].strip().lower()
    return cols


def _is_header(line):
    texts = {w["text"].lower() for w in line["words"]}
    return {"date", "description", "balance"} <= texts


def _year_hint(first_page_text):
    """Year for dates printed without one ('02 Sep'). Prefer the statement period line, and ignore
    numbers that only look like years (e.g. the last 4 digits of an account number: ******2060)."""
    text = first_page_text or ""
    this_year = datetime.now().year
    period = [ln for ln in text.splitlines() if re.search(r"period|statement date|from|to", ln, re.I)]
    for chunk in period + [text]:
        years = [int(y) for y in re.findall(r"(?<![\d*])(20\d{2})(?![\d])", chunk) if 2000 <= int(y) <= this_year + 1]
        if years:
            return max(set(years), key=years.count)
    return this_year


def extract_transactions(pdf_path):
    """Return a DataFrame: page, date, description_sa, amount, balance, balance_ok."""
    rows, opening_balance = [], None
    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            raise ExtractionError("The PDF has no pages.")
        first_text = pdf.pages[0].extract_text() or ""
        if not first_text.strip():
            raise ExtractionError("No text found. Scanned (image) statements are not supported.")
        year = _year_hint(first_text)

        for page_no, page in enumerate(pdf.pages, start=1):
            lines = _group_lines(page.extract_words(x_tolerance=1.5, y_tolerance=2))
            header_idx = next((i for i, ln in enumerate(lines) if _is_header(ln)), None)
            if header_idx is None:
                continue
            cols = _header_columns(lines[header_idx])
            desc_col = next(c for c in cols if c["key"] == "description")
            num_cols = [c for c in cols if c["x0"] > desc_col["x0"]]
            prev_bottom = lines[header_idx]["bottom"]

            for ln in lines[header_idx + 1:]:
                if ln["top"] - prev_bottom > TABLE_END_GAP:
                    break                                   # end of the table on this page
                prev_bottom = ln["bottom"]
                words = _merge_split_numbers(ln["words"])

                date_words = [w for w in words if w["x1"] <= desc_col["x0"] - 1]
                values, desc_words = {}, []
                for w in words:
                    if w in date_words:
                        continue
                    col = None
                    if MONEY_RE.match(w["text"]):
                        col = min(num_cols, key=lambda c: abs(c["x1"] - w["x1"]))
                        if abs(col["x1"] - w["x1"]) > ALIGN_TOLERANCE:
                            col = None
                    if col is not None:
                        values[col["key"]] = w["text"]
                    else:
                        desc_words.append(w["text"])

                date_text = " ".join(w["text"] for w in date_words)
                txn_date = parse_date(date_text, year) if date_text else None
                description = " ".join(desc_words)

                if txn_date is None:
                    if rows and description and not values:  # wrapped description line
                        rows[-1]["description_sa"] += " " + description
                    continue
                if BALANCE_ROW_RE.search(description):
                    if "balance" in values and opening_balance is None:
                        opening_balance = parse_money(values["balance"])
                    continue

                amount = None
                for key, text in values.items():
                    if key in IN_HEADERS:
                        amount = abs(parse_money(text))
                    elif key in OUT_HEADERS:
                        amount = -abs(parse_money(text))
                    elif key in SIGNED_HEADERS:
                        amount = parse_money(text)
                rows.append({"page": page_no, "date": txn_date, "description_sa": description,
                             "amount": amount,
                             "balance": parse_money(values["balance"]) if "balance" in values else None})

    if not rows:
        raise ExtractionError("No transaction table found (expected a header with Date, Description and Balance).")

    df = pd.DataFrame(rows)
    # Confirm / correct amount signs with the running balance.
    # A single 'Amount' column often prints debits without a minus sign.
    prev = opening_balance
    signs_ok = []
    for i, r in df.iterrows():
        ok = False
        if prev is not None and r["balance"] is not None and r["amount"] is not None:
            delta = round(r["balance"] - prev, 2)
            if abs(abs(delta) - abs(r["amount"])) < 0.01:
                df.at[i, "amount"] = delta
                ok = True
        signs_ok.append(ok)
        prev = r["balance"] if r["balance"] is not None else prev
    df["balance_ok"] = signs_ok
    df.attrs["opening_balance"] = opening_balance
    return df
