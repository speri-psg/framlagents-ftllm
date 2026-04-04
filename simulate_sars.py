"""
simulate_sars.py — Generate synthetic SAR labels for alerted customers.

Approach:
  1. Load rule_sweep_data.csv (output of prepare_rule_sweep_data.py) as the
     authoritative alerted-customer population — these are the customers
     identified from condition_details, covering all 11 AML rules.
  2. Aggregate per customer: max trigger_amt, max z_score, alert_count, etc.
  3. Join to ss_segmentation_data for smart_segment_id, customer_type,
     and additional profile features.
  4. Compute a SAR score via logistic function over alert features + noise.
  5. Threshold to hit ~10% SAR filing rate (realistic for AML programs).
  6. Write docs/sar_simulation.csv

Fixed seed (RANDOM_SEED) makes output reproducible but re-runnable.
"""

import os
import numpy as np
import pandas as pd
def expit(x):
    """Sigmoid / logistic function (numerically stable)."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

RANDOM_SEED  = 42
SAR_RATE     = 0.10   # target ~10% of alerted customers flagged as SAR

_HERE        = os.path.dirname(__file__)
SS_DIR       = os.path.join(_HERE, "ss_files")
SWEEP_PATH   = os.path.join(_HERE, "docs", "rule_sweep_data.csv")
SEG_PATH     = os.path.join(_HERE, "docs", "ss_segmentation_data.csv")
TXNS_PATH    = os.path.join(SS_DIR, "aml_s_transactions.csv")
ACCT_PATH    = os.path.join(SS_DIR, "aml_s_account_relationship.csv")
OUT_PATH     = os.path.join(_HERE, "docs", "sar_simulation.csv")

SEG_COLS = [
    "customer_id", "customer_type", "smart_segment_id",
    "avg_num_trxns", "avg_weekly_trxn_amt", "trxn_amt_monthly",
    "total_trxn_amt", "cashout_count",
]

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data...")
sweep = pd.read_csv(SWEEP_PATH)
seg   = pd.read_csv(SEG_PATH, low_memory=False, usecols=SEG_COLS)

print(f"Rule sweep data: {len(sweep)} rows, {sweep['customer_id'].nunique()} unique customers")
print(f"Rules covered:   {sorted(sweep['risk_factor'].unique())}")

# ── 2. Aggregate per customer ─────────────────────────────────────────────────
# A customer can appear under multiple rules; aggregate to one row per customer.
cust = (
    sweep
    .groupby("customer_id", as_index=False)
    .agg(
        alert_count   = ("risk_factor",   "count"),      # number of rule hits
        rule_count    = ("risk_factor",   "nunique"),     # distinct rules fired
        max_trigger   = ("trigger_amt",   "max"),         # highest trigger amount
        max_z_score   = ("z_score",       lambda s: s.dropna().max() if s.notna().any() else np.nan),
        max_txn_count = ("txn_count",     lambda s: s.dropna().max() if s.notna().any() else np.nan),
    )
)
print(f"\nCustomers after aggregation: {len(cust)}")

# ── 3. Join segmentation features ─────────────────────────────────────────────
# Take first matching account per customer for profile features.
seg_cust = (
    seg
    .dropna(subset=["customer_id"])
    .groupby("customer_id", as_index=False)
    .agg(
        customer_type    = ("customer_type",        "first"),
        smart_segment_id = ("smart_segment_id",     "first"),
        avg_num_trxns    = ("avg_num_trxns",         "first"),
        avg_weekly_trxn_amt = ("avg_weekly_trxn_amt","first"),
        trxn_amt_monthly = ("trxn_amt_monthly",     "first"),
        total_trxn_amt   = ("total_trxn_amt",       "first"),
        cashout_count    = ("cashout_count",         "first"),
    )
)

cust = cust.merge(seg_cust, on="customer_id", how="left")

matched = cust["customer_type"].notna().sum()
print(f"Matched to segmentation data: {matched} / {len(cust)} customers")
print(f"  INDIVIDUAL: {(cust['customer_type']=='INDIVIDUAL').sum()}")
print(f"  BUSINESS:   {(cust['customer_type']=='BUSINESS').sum()}")
print(f"  Unmatched:  {(cust['customer_type'].isna()).sum()}")

# ── 3b. Fill zero/null trxn_amt_monthly from actual transaction data ──────────
# ss_segmentation_data has many zeros for alerted customers; compute the true
# average monthly transaction total per customer from aml_s_transactions.
zeros_before = int((cust["trxn_amt_monthly"].fillna(0) == 0).sum())
print(f"\ntrxn_amt_monthly zeros/nulls before fix: {zeros_before} / {len(cust)}")

print("Computing monthly averages from transaction data...")
txns = pd.read_csv(TXNS_PATH, low_memory=False, usecols=["subject_id", "amount", "timestamp"])
acct = pd.read_csv(ACCT_PATH, usecols=["account_id", "customer_id"])

txns["timestamp"] = pd.to_datetime(txns["timestamp"])
txns["month"] = txns["timestamp"].dt.to_period("M")

# Restrict to accounts belonging to our alerted customers only (performance)
alert_cust_ids = set(cust["customer_id"].dropna())
alert_acct_ids = set(
    acct.loc[acct["customer_id"].isin(alert_cust_ids), "account_id"]
)
txns_alert = txns[txns["subject_id"].isin(alert_acct_ids)].copy()

# Sum all transaction amounts per account per month, then average across months
monthly = (
    txns_alert
    .groupby(["subject_id", "month"], as_index=False)["amount"]
    .sum()
    .rename(columns={"amount": "monthly_total"})
)
avg_monthly = (
    monthly
    .groupby("subject_id", as_index=False)["monthly_total"]
    .mean()
    .rename(columns={"monthly_total": "avg_monthly_amt"})
)

# Aggregate to customer level (sum across all accounts)
avg_monthly = avg_monthly.merge(acct, left_on="subject_id", right_on="account_id", how="left")
cust_monthly = (
    avg_monthly
    .groupby("customer_id", as_index=False)["avg_monthly_amt"]
    .sum()
)

# Fill only where trxn_amt_monthly is zero or null
cust = cust.merge(cust_monthly, on="customer_id", how="left")
mask_zero = cust["trxn_amt_monthly"].fillna(0) == 0
cust.loc[mask_zero, "trxn_amt_monthly"] = cust.loc[mask_zero, "avg_monthly_amt"]
cust = cust.drop(columns=["avg_monthly_amt"])

zeros_after = int((cust["trxn_amt_monthly"].fillna(0) == 0).sum())
print(f"trxn_amt_monthly zeros/nulls after fix:  {zeros_after} / {len(cust)}")
print(f"Filled {zeros_before - zeros_after} customers from transaction data")

# ── 4. Compute SAR score ──────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)

def zscore(s):
    """Robust z-score using median/IQR to avoid outlier distortion."""
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    return (s - med) / (iqr + 1e-9)

# Features weighted by AML typology risk:
#   - high trigger amount          → amount threshold exceeded by a lot
#   - high statistical z-score     → very anomalous compared to own history
#   - multiple distinct rules fired → breadth of suspicious activity
#   - high monthly transaction amt → volume risk
#   - high cashout ratio           → structuring / cash typology risk
z = (
    1.5 * zscore(cust["max_trigger"].fillna(0))
  + 1.2 * zscore(cust["max_z_score"].fillna(0))
  + 1.0 * zscore(cust["rule_count"].fillna(0))
  + 0.8 * zscore(cust["trxn_amt_monthly"].fillna(0))
  + 0.5 * zscore(cust["cashout_count"].fillna(0))
  + rng.normal(0, 1.0, len(cust))   # noise so it isn't purely deterministic
)

cust["sar_score"] = expit(z)

# ── 5. Threshold to hit target SAR rate ───────────────────────────────────────
threshold = cust["sar_score"].quantile(1 - SAR_RATE)
cust["is_sar"] = (cust["sar_score"] >= threshold).astype(int)

actual_rate = cust["is_sar"].mean()
print(f"\nSAR threshold: {threshold:.4f}")
print(f"SAR rate: {actual_rate:.1%}  ({cust['is_sar'].sum()} / {len(cust)} customers)")
print(f"\nSAR by segment:")
print(cust.groupby("customer_type")["is_sar"].agg(["sum","count","mean"]).round(3))

# ── 6. Save ───────────────────────────────────────────────────────────────────
out_cols = [
    "customer_id", "customer_type", "smart_segment_id",
    "alert_count", "rule_count", "max_trigger", "max_z_score",
    "avg_num_trxns", "avg_weekly_trxn_amt", "trxn_amt_monthly",
    "total_trxn_amt", "cashout_count",
    "sar_score", "is_sar",
]
cust[out_cols].to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(cust)} rows to: {OUT_PATH}")

# ── 7. Coverage check ─────────────────────────────────────────────────────────
print("\nSAR coverage by rule:")
labeled = sweep.merge(
    cust[["customer_id", "is_sar"]].rename(columns={"is_sar": "sar_label"}),
    on="customer_id", how="left"
)
rule_cov = (
    labeled
    .groupby("risk_factor")
    .apply(lambda g: pd.Series({
        "customers": g["customer_id"].nunique(),
        "SAR": int(g.drop_duplicates("customer_id")["sar_label"].sum()),
        "FP":  int((g.drop_duplicates("customer_id")["sar_label"] == 0).sum()),
    }))
)
print(rule_cov.to_string())
