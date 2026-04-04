"""
prepare_rule_sweep_data.py — Pre-compute per-customer rule metrics for FP/FN sweep

Join chain (authoritative):
  condition_details[Transaction ID]  (strip trailing "N")
      -> aml_s_transactions[id]      (amount, subject_id, timestamp, type, direction)
      -> aml_s_account_relationship  (subject_id -> customer_id)
      -> sar_simulation              (SAR label)
  aml_s_customers                    (age for Elder Abuse)

Output: docs/rule_sweep_data.csv
  One row per (customer_id, risk_factor).

Run once (or after data changes):
    python prepare_rule_sweep_data.py
"""

import os
import pandas as pd
import numpy as np
from config import SAR_CSV

_HERE = os.path.dirname(os.path.abspath(__file__))

SS_DIR   = os.path.join(_HERE, "ss_files")
DOCS_DIR = os.path.join(_HERE, "docs")
OUT_CSV  = os.path.join(DOCS_DIR, "rule_sweep_data.csv")

# ── Load base tables ──────────────────────────────────────────────────────────

print("Loading base tables …")
sar      = pd.read_csv(SAR_CSV)
acct_rel = pd.read_csv(os.path.join(SS_DIR, "aml_s_account_relationship.csv"))
cust_raw = pd.read_csv(os.path.join(SS_DIR, "aml_s_customers.csv"), low_memory=False)
txns_raw = pd.read_csv(os.path.join(SS_DIR, "aml_s_transactions.csv"), low_memory=False)
cd_raw   = pd.read_csv(os.path.join(SS_DIR, "PSG_Alert_Report_env2_Nov3_condition_details.csv"))

txns_raw["timestamp"] = pd.to_datetime(txns_raw["timestamp"])

# Reference date: end of alert generation window
REF_DATE = pd.Timestamp("2025-11-03")

# ── Join condition_details → aml_s_transactions via Transaction ID ────────────
# Transaction IDs in condition_details have a trailing "N" (e.g. "612318N").
# Strip it to match the numeric id in aml_s_transactions.

cd_raw["txn_id"] = pd.to_numeric(
    cd_raw["Transaction ID"].astype(str).str.rstrip("N"), errors="coerce"
)

# Pull only the columns we need from aml_s_transactions
txn_cols = txns_raw[["id", "subject_id", "amount", "timestamp",
                      "transaction_type", "cash_direction"]].copy()

cd = cd_raw.merge(txn_cols, left_on="txn_id", right_on="id", how="left",
                  suffixes=("_cd", ""))

# Resolve subject_id → customer_id via account_relationship
cd = cd.merge(
    acct_rel[["account_id", "customer_id"]],
    left_on="subject_id", right_on="account_id",
    how="left",
)

print(f"  Condition-details rows: {len(cd):,}  |  matched to transactions: {cd['txn_id'].notna().sum():,}")

# ── Load full transaction history for all alerted customers ───────────────────
# (profile computation: 12-month history, rolling windows)

alert_cust_ids = set(cd["customer_id"].dropna())
alert_acct_ids = set(acct_rel.loc[acct_rel["customer_id"].isin(alert_cust_ids), "account_id"])

txns = txns_raw[txns_raw["subject_id"].isin(alert_acct_ids)].copy()
txns = txns.merge(
    acct_rel[["account_id", "customer_id"]],
    left_on="subject_id", right_on="account_id",
    how="left",
)

print(f"  Alerted customers: {len(alert_cust_ids):,}  |  their transactions: {len(txns):,}")

results = []   # collect one dict per (customer_id, risk_factor)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 1 — Activity Deviation
#   Two sub-rules by Condition ID:
#     4ddd5cd7 -> Outgoing ACH   >= $50K AND >= 5 std devs above 12-month mean
#     d1e0f845 -> Outgoing Check >= $50K AND >= 2 std devs above 12-month mean
#
#   trigger_amt = monthly SUM of matching transactions from aml_s_transactions
#                 (via Transaction ID in condition_details, stripped of trailing N)
# ═══════════════════════════════════════════════════════════════════════════════

print("\nComputing Activity Deviation metrics …")

ACT_COND_META = {
    "4ddd5cd7-bc83-36aa-909c-4e660f57c830": ("ACH",   "CashOut", 5),
    "d1e0f845-bc1b-368c-b8c8-49ab0b9e486b": ("Check", "CashOut", 2),
}

# Filter condition_details to Activity Deviation only; already joined to transactions
act_cd = cd[
    (cd["Risk Factors"] == "Activity Deviation") &
    (cd["Condition ID"].isin(ACT_COND_META))
].copy()
act_cd["month"] = act_cd["timestamp"].dt.to_period("M")

# trigger_amt = monthly sum of the triggering transaction type per (customer, condition, month)
act_monthly = (
    act_cd.groupby(["customer_id", "Condition ID", "month"])["amount"]
    .sum()
    .reset_index()
    .rename(columns={"amount": "monthly_sum"})
)
# Keep the peak month per (customer, condition)
act_trigger = (
    act_monthly.sort_values("monthly_sum", ascending=False)
    .drop_duplicates(subset=["customer_id", "Condition ID"])
    .rename(columns={"monthly_sum": "trigger_amt", "month": "trigger_month"})
)

# 12-month profile history per transaction type from full alerted-customer transactions
def _monthly_profile(txn_type, direction="CashOut"):
    hist = txns[
        (txns["transaction_type"] == txn_type) &
        (txns["cash_direction"] == direction)
    ].copy()
    hist["month"] = hist["timestamp"].dt.to_period("M")
    return (
        hist.groupby(["customer_id", "month"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "monthly_amt"})
    )

_profile_cache = {}

def _get_monthly_profile(txn_type, direction="CashOut"):
    key = (txn_type, direction)
    if key not in _profile_cache:
        _profile_cache[key] = _monthly_profile(txn_type, direction)
    return _profile_cache[key]

def _act_dev_stats(cust_id, trigger_month, txn_type, direction="CashOut"):
    """Return (trigger_month_total, prior_mean, prior_std) from aml_s_transactions.

    trigger_month_total = actual full monthly sum from aml_s_transactions for the
    trigger month (more complete than summing condition_details rows, which may only
    include a subset of the month's transactions).
    prior_mean / prior_std = 12-month profile from months BEFORE the trigger month.
    """
    src = _get_monthly_profile(txn_type, direction)
    cust_src = src[src["customer_id"] == cust_id]

    # Full monthly total from aml_s_transactions for the trigger month
    trigger_row = cust_src[cust_src["month"] == trigger_month]
    trigger_total = float(trigger_row["monthly_amt"].iloc[0]) if not trigger_row.empty else 0.0

    # Prior 12-month profile (exclude trigger month)
    hist = cust_src[cust_src["month"] < trigger_month]["monthly_amt"]
    if len(hist) == 0:
        return trigger_total, 0.0, 0.0
    mean = float(hist.mean())
    std  = float(hist.std(ddof=1)) if len(hist) > 1 else 0.0
    return trigger_total, mean, std

act_rows = []
for _, row in act_trigger.iterrows():
    txn_type, direction, _ = ACT_COND_META[row["Condition ID"]]
    trigger_total, mean, std = _act_dev_stats(
        row["customer_id"], row["trigger_month"], txn_type, direction
    )
    # Use the full monthly total from aml_s_transactions as trigger_amt
    # (falls back to condition_details sum if the customer has no aml_s_transactions data)
    trigger_amt = trigger_total if trigger_total > 0 else row["trigger_amt"]
    z = (trigger_amt - mean) / std if std > 0 else np.nan
    act_rows.append({
        "customer_id":  row["customer_id"],
        "risk_factor":  f"Activity Deviation ({txn_type})",
        "trigger_amt":  round(trigger_amt, 2),
        "profile_mean": round(mean, 2),
        "profile_std":  round(std, 2),
        "z_score":      round(z, 3) if not np.isnan(z) else np.nan,
    })

# A customer can appear in both sub-rules — keep one row per (customer, risk_factor)
act_df = (
    pd.DataFrame(act_rows)
    .sort_values("trigger_amt", ascending=False)
    .drop_duplicates(subset=["customer_id", "risk_factor"])
    .reset_index(drop=True)
)
for rf, grp in act_df.groupby("risk_factor"):
    print(f"  {rf}: {len(grp)} alerted customers")
results.append(act_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 2 — Elder Abuse
#   Condition: age >= 60  AND  14-day outgoing >= $5K  AND  >= 3 σ above 90-day mean
#   Sweep params: age_threshold, amount_threshold, z_threshold
# ═══════════════════════════════════════════════════════════════════════════════

print("Computing Elder Abuse metrics …")

# trigger_amt = max outgoing transaction amount from the triggering transactions
# trigger_date = date of the latest triggering transaction
elder_cd = cd[cd["Risk Factors"] == "Elder Abuse"].copy()

elder_trigger = (
    elder_cd.sort_values("timestamp")
    .groupby("customer_id")
    .agg(trigger_amt=("amount", "max"), trigger_date=("timestamp", "last"))
    .reset_index()
)

# 2b. Customer age (from birthdate)
cust_raw["birthdate"] = pd.to_datetime(cust_raw["birthdate"], errors="coerce")
cust_age = cust_raw[["id", "birthdate"]].rename(columns={"id": "customer_id"})
cust_age["age"] = ((REF_DATE - cust_age["birthdate"]).dt.days / 365.25).round(1)

elder_trigger = elder_trigger.merge(cust_age[["customer_id", "age"]], on="customer_id", how="left")

# 2c. 90-day preceding rolling 14-day outgoing sums
#     For each customer, compute 14-day rolling outgoing totals across the 90 days
#     before the trigger date, then take mean and std of those totals.

def _elder_profile(cust_id, trigger_date):
    window_end   = trigger_date - pd.Timedelta(days=1)
    window_start = window_end   - pd.Timedelta(days=90)

    cust_txns = txns[
        (txns["customer_id"] == cust_id) &
        (txns["cash_direction"] == "CashOut") &
        (txns["timestamp"] >= window_start) &
        (txns["timestamp"] <= window_end)
    ][["timestamp", "amount"]].sort_values("timestamp")

    if cust_txns.empty:
        return 0.0, 0.0

    # Compute 14-day rolling sums by day
    cust_txns = cust_txns.set_index("timestamp")
    daily = cust_txns["amount"].resample("D").sum().fillna(0)
    rolling_14d = daily.rolling(14, min_periods=1).sum()
    mean = float(rolling_14d.mean())
    std  = float(rolling_14d.std(ddof=1)) if len(rolling_14d) > 1 else 0.0
    return mean, std

elder_rows = []
for _, row in elder_trigger.iterrows():
    mean, std = _elder_profile(row["customer_id"], row["trigger_date"])
    z = (row["trigger_amt"] - mean) / std if std > 0 else np.nan
    elder_rows.append({
        "customer_id":     row["customer_id"],
        "risk_factor":     "Elder Abuse",
        "trigger_amt":     round(row["trigger_amt"], 2),
        "age":             row["age"] if not pd.isna(row["age"]) else np.nan,
        "profile_mean":    round(mean, 2),
        "profile_std":     round(std, 2),
        "z_score":         round(z, 3) if not np.isnan(z) else np.nan,
    })

elder_df = pd.DataFrame(elder_rows)
print(f"  Elder Abuse: {len(elder_df)} alerted customers")
results.append(elder_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 3 — Velocity Single
#   Condition: ≥1 pair (CashIn, CashOut) within 14 days,
#              CashOut = 90–110% of CashIn, pair total ≥ $20 K
#   Sweep params: pair_total_threshold, ratio_tolerance (e.g. ±10%)
# ═══════════════════════════════════════════════════════════════════════════════

print("Computing Velocity Single metrics …")

# Transactions already joined from aml_s_transactions via Transaction ID
vs_cd = cd[cd["Risk Factors"] == "Velocity Single"].copy()

# For each alert, find the best CashIn + CashOut pair within 14 days
vs_rows = []

for alert_id, group in vs_cd.groupby("Alert ID"):
    ins  = group[group["cash_direction"] == "CashIn"].copy()
    outs = group[group["cash_direction"] == "CashOut"].copy()

    if ins.empty or outs.empty:
        best_amt = group["amount"].max()
        cust = group["customer_id"].iloc[0]
        vs_rows.append({"customer_id": cust, "risk_factor": "Velocity Single",
                        "pair_total": float(best_amt) * 2, "in_amt": float(best_amt),
                        "out_amt": float(best_amt), "ratio": 1.0})
        continue

    best_total = 0.0
    best_in = best_out = best_ratio = 0.0
    best_cust = ins["customer_id"].iloc[0]

    for _, r_in in ins.iterrows():
        for _, r_out in outs.iterrows():
            days_apart = abs((r_out["timestamp"] - r_in["timestamp"]).days)
            if days_apart > 14:
                continue
            if r_in["amount"] <= 0:
                continue
            ratio = r_out["amount"] / r_in["amount"]
            pair_total = r_in["amount"] + r_out["amount"]
            if pair_total > best_total:
                best_total = pair_total
                best_in    = r_in["amount"]
                best_out   = r_out["amount"]
                best_ratio = ratio
                best_cust  = r_in["customer_id"]

    if best_total == 0:
        best_total = group["amount"].max() * 2
        best_cust  = group["customer_id"].iloc[0]

    vs_rows.append({
        "customer_id": best_cust,
        "risk_factor": "Velocity Single",
        "pair_total":  round(best_total, 2),
        "in_amt":      round(best_in, 2),
        "out_amt":     round(best_out, 2),
        "ratio":       round(best_ratio, 4),
    })

vs_df = pd.DataFrame(vs_rows)

# If a customer appears in multiple alerts, keep the worst (highest pair total)
vs_df = (
    vs_df.sort_values("pair_total", ascending=False)
    .drop_duplicates(subset="customer_id")
    .reset_index(drop=True)
)
print(f"  Velocity Single: {len(vs_df)} alerted customers")
results.append(vs_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 4 — Detect Excessive Transaction Activity
#   Condition: 5-day sum of incoming Cash + Check > $10 K
#   Sweep params: amount_threshold, time_window_days
# ═══════════════════════════════════════════════════════════════════════════════

print("Computing Detect Excessive metrics …")

det_cd = cd[cd["Risk Factors"] == "Detect Excessive Transaction Activity"].copy()

det_trigger = (
    det_cd.sort_values("timestamp")
    .groupby("customer_id")
    .agg(trigger_amt=("amount", "max"), trigger_date=("timestamp", "last"))
    .reset_index()
)

# Compute max rolling N-day Cash + Check CashIn sum for each customer
# across multiple time windows, using a window ending on their trigger date.
WINDOWS = [3, 7, 10, 14]   # 5-day is the current condition (trigger_amt)

cash_check_in = txns[
    (txns["transaction_type"].isin(["Cash", "Check"])) &
    (txns["cash_direction"] == "CashIn")
].copy()

def _max_rolling_sum(cust_id, trigger_date, window_days):
    end   = trigger_date
    start = end - pd.Timedelta(days=90)   # look back up to 90 days for context
    cust_txns = cash_check_in[
        (cash_check_in["customer_id"] == cust_id) &
        (cash_check_in["timestamp"] >= start) &
        (cash_check_in["timestamp"] <= end)
    ][["timestamp", "amount"]].sort_values("timestamp")
    if cust_txns.empty:
        return 0.0
    cust_txns = cust_txns.set_index("timestamp")
    daily = cust_txns["amount"].resample("D").sum().fillna(0)
    rolling = daily.rolling(window_days, min_periods=1).sum()
    return float(rolling.max())

det_rows = []
for _, row in det_trigger.iterrows():
    entry = {"customer_id": row["customer_id"], "risk_factor": "Detect Excessive Transaction Activity",
             "trigger_amt": round(row["trigger_amt"], 2)}
    for w in WINDOWS:
        entry[f"max_rolling_{w}d"] = round(_max_rolling_sum(row["customer_id"], row["trigger_date"], w), 2)
    det_rows.append(entry)

det_df = pd.DataFrame(det_rows)
print(f"  Detect Excessive: {len(det_df)} alerted customers")
results.append(det_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 5 — Structuring
#   4 sub-conditions (In/Out x 2-day/3-day windows)
#   Condition: N qualifying days within M-day window, each day's Cash total in [$floor, $ceiling]
#   Sweep params: daily_floor, days_required
# ═══════════════════════════════════════════════════════════════════════════════

print("Computing Structuring metrics …")

STRUCT_COND_META = {
    "335fa537-0909-395d-a696-6f41827dcbeb": ("Incoming", "CashIn",  3, 14, 3_000, 40_000),
    "a1c62d82-d13c-361b-8f4e-ca91bc2f7fc5": ("Outgoing", "CashOut", 3, 14, 7_000, 30_000),
    "dbd58444-05ad-3edd-adc7-4393ecbcb43c": ("Incoming", "CashIn",  2, 10, 3_000, 30_000),
    "e943d83e-3f82-307f-81ed-b7a7bcd0743e": ("Outgoing", "CashOut", 2, 10, 3_000, 30_000),
}

struct_cd = cd[
    (cd["Risk Factors"] == "Structuring") &
    (cd["Condition ID"].isin(STRUCT_COND_META))
].copy()

struct_rows = []
for cond_id, grp in struct_cd.groupby("Condition ID"):
    direction, cash_dir, days_req, window, daily_floor, daily_ceiling = STRUCT_COND_META[cond_id]
    rf_name = f"Structuring ({direction} Cash)"
    # Per customer: max single-day Cash amount and days observed
    daily = (
        grp.groupby(["customer_id", grp["timestamp"].dt.date])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "daily_amt", "timestamp": "date"})
    )
    for cust_id, cust_daily in daily.groupby("customer_id"):
        qualifying = cust_daily[
            (cust_daily["daily_amt"] >= daily_floor) &
            (cust_daily["daily_amt"] <= daily_ceiling)
        ]
        struct_rows.append({
            "customer_id":    cust_id,
            "risk_factor":    rf_name,
            "trigger_amt":    round(float(qualifying["daily_amt"].max()) if not qualifying.empty else grp[grp["customer_id"]==cust_id]["amount"].max(), 2),
            "days_observed":  int(len(qualifying)),
            "days_required":  days_req,
            "time_window":    window,
            "daily_floor":    daily_floor,
            "daily_ceiling":  daily_ceiling,
        })

struct_df = (
    pd.DataFrame(struct_rows)
    .sort_values("trigger_amt", ascending=False)
    .drop_duplicates(subset=["customer_id", "risk_factor"])
    .reset_index(drop=True)
)
for rf, grp in struct_df.groupby("risk_factor"):
    print(f"  {rf}: {len(grp)} alerted customers")
results.append(struct_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 6 — CTR Client
#   Condition: Cash + Currency Exchange in/out total > $10K
#   Sweep param: floor_amount
# ═══════════════════════════════════════════════════════════════════════════════

print("Computing CTR Client metrics …")

ctr_cd = cd[cd["Risk Factors"] == "CTR Client"].copy()

ctr_trigger = (
    ctr_cd.groupby("customer_id")
    .agg(trigger_amt=("amount", "sum"))
    .reset_index()
)
ctr_trigger["risk_factor"] = "CTR Client"
ctr_df = ctr_trigger.reset_index(drop=True)
print(f"  CTR Client: {len(ctr_df)} alerted customers")
results.append(ctr_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 7 — Burst in Originator Activity
#   Condition: external account, 5-day sum Incoming Wire+ACH >= $5K,
#              >= 3 transactions, >= 2 distinct counterparties
#   Sweep param: floor_amount
# ═══════════════════════════════════════════════════════════════════════════════

print("Computing Burst in Originator Activity metrics …")

burst_orig_cd = cd[cd["Risk Factors"] == "Burst in Originator Activity"].copy()

burst_orig_trigger = (
    burst_orig_cd.groupby("customer_id")
    .agg(trigger_amt=("amount", "sum"), txn_count=("txn_id", "count"))
    .reset_index()
)
burst_orig_trigger["risk_factor"] = "Burst in Originator Activity"
print(f"  Burst in Originator Activity: {len(burst_orig_trigger)} alerted customers")
results.append(burst_orig_trigger)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 8 — Burst in Beneficiary Activity
#   Condition: external account, 5-day sum Outgoing Wire+ACH >= $5K,
#              >= 3 transactions, >= 2 distinct counterparties
#   Sweep param: floor_amount
# ═══════════════════════════════════════════════════════════════════════════════

print("Computing Burst in Beneficiary Activity metrics …")

burst_ben_cd = cd[cd["Risk Factors"] == "Burst in Beneficiary Activity"].copy()

burst_ben_trigger = (
    burst_ben_cd.groupby("customer_id")
    .agg(trigger_amt=("amount", "sum"), txn_count=("txn_id", "count"))
    .reset_index()
)
burst_ben_trigger["risk_factor"] = "Burst in Beneficiary Activity"
print(f"  Burst in Beneficiary Activity: {len(burst_ben_trigger)} alerted customers")
results.append(burst_ben_trigger)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 9 — Risky International Transfer
#   Two sub-conditions: single Wire >= $300K or >= $500K to/from risky country
#   Sweep param: floor_amount
# ═══════════════════════════════════════════════════════════════════════════════

print("Computing Risky International Transfer metrics …")

RISKY_COND_META = {
    "adab623b-1343-307f-80d8-58d005376ad9": 300_000,
    "36cc7f20-126c-3e40-94e7-737ac7486547": 500_000,
}

risky_cd = cd[cd["Risk Factors"] == "Risky International Transfer"].copy()

risky_rows = []
for _, row in risky_cd.iterrows():
    risky_rows.append({
        "customer_id": row["customer_id"],
        "risk_factor": "Risky International Transfer",
        "trigger_amt": round(float(row["amount"]), 2),
        "floor_current": RISKY_COND_META.get(row["Condition ID"], 300_000),
    })

risky_df = (
    pd.DataFrame(risky_rows)
    .sort_values("trigger_amt", ascending=False)
    .drop_duplicates("customer_id")
    .reset_index(drop=True)
)
print(f"  Risky International Transfer: {len(risky_df)} alerted customers")
results.append(risky_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Combine + join SAR labels
# ═══════════════════════════════════════════════════════════════════════════════

print("\nCombining and joining SAR labels …")

combined = pd.concat(results, ignore_index=True, sort=False)
combined = combined.merge(
    sar[["customer_id", "is_sar", "smart_segment_id", "customer_type"]],
    on="customer_id",
    how="left",
)

combined.to_csv(OUT_CSV, index=False)
print(f"\nSaved to {OUT_CSV}  ({len(combined):,} rows)")

# Summary
print("\nSummary by risk_factor:")
for rf, grp in combined.groupby("risk_factor"):
    n   = len(grp)
    sar_n = int(grp["is_sar"].sum()) if "is_sar" in grp else "?"
    fp_n  = int((grp["is_sar"] == 0).sum()) if "is_sar" in grp else "?"
    null_n = int(grp["is_sar"].isna().sum()) if "is_sar" in grp else "?"
    print(f"  {rf:<42} | customers={n:>4} | SAR={sar_n:>3} | FP={fp_n:>4} | no_label={null_n:>4}")
