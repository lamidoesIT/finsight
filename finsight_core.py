"""
FINSIGHT core: loading the trained model, feature engineering and prediction.

This is the single source of truth for how a transaction becomes a category.
Both the Gradio web app (app.py) and the batch ETL pipeline (pipeline/) import
from here, so the interactive app and the warehouse can never drift apart.

The feature rules below are the same ones used when the model was trained.
"""
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

HERE = Path(__file__).parent
MODEL_PATH = HERE / "finsight_model.joblib"

TEXT_FEATURES = ["description_sa", "txn_type"]
NUMERIC_FEATURES = [
    "amount_zar", "stmt_year", "stmt_month", "stmt_day",
    "purchase_year", "purchase_month", "purchase_day",
    "days_to_process", "has_purchase_date",
    "hour", "minute", "has_time", "is_online",
]

LOW_CONFIDENCE = 0.50
HIGH_CONFIDENCE = 0.80

_bundle = joblib.load(MODEL_PATH)
tfidf_D = _bundle["tfidf"]
imputer_D = _bundle["imputer"]
scaler_D = _bundle["scaler"]
xgb_model_D = _bundle["model"]
label_encoder_D = _bundle["label_encoder"]

# Every spending group the model can predict.
CATEGORIES = [str(c) for c in label_encoder_D.classes_]


def infer_purchase_date(row):
    """Some descriptions carry the original purchase date; recover it when it looks sane."""
    if pd.isna(row["txn_date"]):
        return pd.NaT
    match = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", row["description_sa"])
    if not match:
        return pd.NaT
    month, day = int(match.group(1)), int(match.group(2))
    try:
        purchase_dt = pd.Timestamp(year=row["txn_date"].year, month=month, day=day)
    except ValueError:
        return pd.NaT
    if purchase_dt > row["txn_date"]:
        purchase_dt -= pd.DateOffset(years=1)
    gap = (row["txn_date"] - purchase_dt).days
    return purchase_dt if 0 <= gap <= 10 else pd.NaT


def get_txn_type(text):
    """Coarse transaction type read off the description wording."""
    text = text.lower()
    if "atm" in text or "withdrawal" in text:
        return "atm"
    if "recurring" in text:
        return "recurring"
    if ("point of sale" in text or "pos " in text or "purchase" in text
            or "debit card" in text or "checkcard" in text or "check card" in text):
        return "card"
    if ("ach" in text or "ppd" in text or "web id" in text or "bill pay" in text
            or "autopay" in text or "payment" in text):
        return "bill_payment"
    if "transfer" in text or "xfer" in text:
        return "transfer"
    return "other"


def build_features(transactions):
    """Raw extracted transactions -> the feature frame the model expects."""
    f = transactions.copy().reset_index(drop=True)
    f["txn_date"] = pd.to_datetime(f["date"], errors="coerce")
    f["description_sa"] = f["description_sa"].fillna("").astype(str)

    f["stmt_year"] = f["txn_date"].dt.year
    f["stmt_month"] = f["txn_date"].dt.month
    f["stmt_day"] = f["txn_date"].dt.day

    purchase_dates = pd.to_datetime(f.apply(infer_purchase_date, axis=1), errors="coerce")
    f["purchase_year"] = purchase_dates.dt.year
    f["purchase_month"] = purchase_dates.dt.month
    f["purchase_day"] = purchase_dates.dt.day
    f["days_to_process"] = (f["txn_date"] - purchase_dates).dt.days
    f["has_purchase_date"] = purchase_dates.notna().astype(int)

    time_parts = f["description_sa"].str.extract(r"\b(\d{1,2}):(\d{2})\s*(AM|PM)?\b", flags=re.IGNORECASE)
    hour = pd.to_numeric(time_parts[0], errors="coerce")
    minute = pd.to_numeric(time_parts[1], errors="coerce")
    ampm = time_parts[2].str.upper()
    hour = hour.where(~((ampm == "PM") & (hour < 12)), hour + 12)
    hour = hour.where(~((ampm == "AM") & (hour == 12)), 0)
    f["hour"], f["minute"] = hour, minute
    f["has_time"] = f["hour"].notna().astype(int)

    f["is_online"] = (
        f["description_sa"]
        .str.contains(r"www|https?:|\.com|\bcom\b|\.net\b|\.co\b", case=False, regex=True)
        .astype(int)
    )
    f["txn_type"] = f["description_sa"].apply(get_txn_type)
    f["amount_zar"] = pd.to_numeric(f["amount"], errors="coerce").abs().round(2)
    return f


def predict_categories(f):
    """Adds predicted_category, confidence, second_guess and second_guess_prob."""
    text = f[TEXT_FEATURES].fillna("").astype(str).agg(" ".join, axis=1)
    X_tfidf = tfidf_D.transform(text)
    X_num = scaler_D.transform(imputer_D.transform(f[NUMERIC_FEATURES].copy()))
    X = hstack([X_tfidf, csr_matrix(X_num)]).tocsr()

    proba = xgb_model_D.predict_proba(X)
    order = np.argsort(proba, axis=1)
    best, second = order[:, -1], order[:, -2]
    rows = np.arange(len(proba))
    f = f.copy()
    f["predicted_category"] = label_encoder_D.inverse_transform(best.astype(int))
    f["confidence"] = proba[rows, best]
    f["second_guess"] = label_encoder_D.inverse_transform(second.astype(int))
    f["second_guess_prob"] = proba[rows, second]
    return f


def categorise(transactions):
    """Convenience: raw transactions -> features + predictions in one call."""
    return predict_categories(build_features(transactions))
