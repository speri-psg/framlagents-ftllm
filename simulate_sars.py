"""
simulate_sars.py — Generate synthetic SAR labels for alerted customers.

Approach:
  1. Join PSG alerts → ss_segmentation_data on subject_id = account_id
  2. Aggregate to customer level (one customer may hold multiple accounts)
  3. Compute a SAR score via logistic function over alert features + noise
  4. Threshold to hit ~10% SAR filing rate (realistic for AML programs)
  5. Write docs/sar_simulation.csv

Fixed seed (RANDOM_SEED) makes output reproducible but re-runnable.
"""

import os
import numpy as np
import pandas as pd
from scipy.special import expit   # sigmoid

RANDOM_SEED  = 42
SAR_RATE     = 0.10   # target ~10% of alerted customers flagged as SAR

ALERTS_PATH  = "C:/Users/Aaditya/Downloads/PSG_Alert_Report_env2_Nov3_PSG_Alerts_11112025.csv"
SEG_PATH     = os.path.join(os.path.dirname(__file__), "docs", "ss_segmentation_data.csv")
OUT_PATH     = os.path.join(os.path.dirname(__file__), "docs", "sar_simulation.csv")

SEG_COLS = [
    "account_id", "customer_id", "customer_type", "smart_segment_id",
    "avg_num_trxns", "avg_weekly_trxn_amt", "trxn_amt_monthly",
    "trxn_count", "total_trxn_amt", "cashout_count",
]

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data...")
alerts = pd.read_csv(ALERTS_PATH)
seg    = pd.read_csv(SEG_PATH, low_memory=False, usecols=SEG_COLS)

# ── 2. Join alerts → seg on subject_id = account_id ──────────────────────────
merged = alerts.merge(seg, left_on="subject_id", right_on="account_id", how="left")

# ── 3. Aggregate per customer ─────────────────────────────────────────────────
# Sum/max alert-level info; keep seg features from the first matching account
cust = (
    merged
    .groupby("customer_id", as_index=False)
    .agg(
        customer_type    = ("customer_type",     "first"),
        smart_segment_id = ("smart_segment_id",  "first"),
        alert_count      = ("id",                "count"),
        total_alert_amt  = ("amount",            "sum"),
        max_alert_amt    = ("amount",            "max"),
        avg_num_trxns    = ("avg_num_trxns",     "first"),
        avg_weekly_trxn_amt = ("avg_weekly_trxn_amt", "first"),
        trxn_amt_monthly = ("trxn_amt_monthly",  "first"),
        total_trxn_amt   = ("total_trxn_amt",    "first"),
        cashout_count    = ("cashout_count",     "first"),
    )
)
print(f"Unique alerted customers: {len(cust)}")
print(f"  INDIVIDUAL: {(cust.customer_type=='INDIVIDUAL').sum()}")
print(f"  BUSINESS:   {(cust.customer_type=='BUSINESS').sum()}")

# ── 4. Compute SAR score ──────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)

def zscore(s):
    """Robust z-score using median/IQR to avoid outlier distortion."""
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    return (s - med) / (iqr + 1e-9)

# Features weighted by AML typology risk:
#   - high total alert amount       → strong signal
#   - multiple alerts on same cust  → structuring / velocity indicator
#   - high monthly transaction amt  → volume risk
#   - high avg weekly transactions  → frequency risk
z = (
    1.5 * zscore(cust["total_alert_amt"].fillna(0))
  + 1.2 * zscore(cust["alert_count"].fillna(0))
  + 0.8 * zscore(cust["trxn_amt_monthly"].fillna(0))
  + 0.5 * zscore(cust["avg_num_trxns"].fillna(0))
  + rng.normal(0, 1.0, len(cust))   # noise so it isn't purely deterministic
)

cust["sar_score"] = expit(z)

# ── 5. Threshold to hit target SAR rate ───────────────────────────────────────
threshold = cust["sar_score"].quantile(1 - SAR_RATE)
cust["is_sar"] = (cust["sar_score"] >= threshold).astype(int)

actual_rate = cust["is_sar"].mean()
print(f"\nSAR threshold: {threshold:.4f}")
print(f"SAR rate: {actual_rate:.1%}  ({cust['is_sar'].sum()} / {len(cust)} customers)")
print(f"  SAR by segment:")
print(cust.groupby("customer_type")["is_sar"].agg(["sum","count","mean"]).round(3))

# ── 6. Save ───────────────────────────────────────────────────────────────────
out_cols = [
    "customer_id", "customer_type", "smart_segment_id",
    "alert_count", "total_alert_amt", "max_alert_amt",
    "avg_num_trxns", "avg_weekly_trxn_amt", "trxn_amt_monthly",
    "sar_score", "is_sar",
]
cust[out_cols].to_csv(OUT_PATH, index=False)
print(f"\nSaved to: {OUT_PATH}")
print(cust[out_cols].head(5).to_string())
