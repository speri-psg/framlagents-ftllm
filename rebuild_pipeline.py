"""
rebuild_pipeline.py — Full rebuild pipeline:

  Step 1: Aggregate ds_segmentation_data.csv at the customer level
          (sum transaction metrics across all accounts — a customer with
          5 accounts contributes combined metrics to their risk profile)

  Step 2: Rebuild sar_simulation.csv for alerted customers
          (using aggregated metrics for more accurate SAR scoring)

  Step 3: Simulate FN customers — non-alerted but truly suspicious
          (customers who flew under the radar; target ~200)

  Step 4: Merge alerted SARs + FN customers into final sar_simulation.csv

  Step 5: Build docs/custs_accts_txns_alerts.csv
          - Alerted customers  : ALERT=Yes, FP=Yes/No, FN=Yes/No
          - FN-only customers  : ALERT=No,  FP=No,     FN=Yes

Outputs:
  docs/sar_simulation.csv          (updated with FN customers included)
  docs/custs_accts_txns_alerts.csv (rebuilt from real data)
"""

import os
import numpy as np
import pandas as pd
from scipy.special import expit

RANDOM_SEED  = 42
SAR_RATE     = 0.10    # ~10% SAR rate among alerted customers
FN_TARGET    = 200     # non-alerted truly suspicious customers to add

_HERE        = os.path.dirname(os.path.abspath(__file__))
SEG_PATH     = os.path.join(_HERE, "docs", "ds_segmentation_data.csv")
SAR_OUT      = os.path.join(_HERE, "docs", "sar_simulation.csv")
ALERTS_OUT   = os.path.join(_HERE, "docs", "custs_accts_txns_alerts.csv")
PSG_ALERTS   = "C:/Users/Aaditya/Downloads/PSG_Alert_Report_env2_Nov3_PSG_Alerts_11112025.csv"

rng = np.random.default_rng(RANDOM_SEED)

def zscore(s):
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    return (s - med) / (iqr + 1e-9)

def sar_score(df):
    """Logistic SAR probability based on transaction features."""
    z = (
        1.5 * zscore(df["total_alert_amt"].fillna(0))
      + 1.2 * zscore(df["alert_count"].fillna(0))
      + 0.8 * zscore(df["trxn_amt_monthly"].fillna(0))
      + 0.5 * zscore(df["avg_num_trxns"].fillna(0))
      + rng.normal(0, 1.0, len(df))
    )
    return expit(z)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Aggregate ds_segmentation_data at customer level
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Aggregating ds_segmentation_data at customer level")
print("=" * 60)

seg = pd.read_csv(SEG_PATH, low_memory=False)
print(f"  Raw rows (account level): {len(seg):,}")

# Transaction metrics: sum across all accounts per customer
SUM_COLS   = ["CURRENT_BALANCE", "trxn_count", "total_trxn_amt",
              "cashin_count", "cashout_count", "avg_num_trxns",
              "avg_weekly_trxn_amt", "trxn_amt_monthly", "loan_amount"]
# Take max for age-related account fields
MAX_COLS   = ["ACCT_AGE_YEARS"]
# Customer-level demographics — same across all accounts, just take first
FIRST_COLS = ["AGE", "GENDER", "marital_status", "occupation", "CITIZENSHIP",
              "INCOME", "nationality_country_id", "RESIDENCY_COUNTRY",
              "pep", "314b", "negative_news", "OFAC", "exempt_cdd",
              "exempt_sanctions", "naics", "sic_code", "entity",
              "customer_type", "dynamic_segment",
              "AGE_CATEGORY", "INCOME_BAND", "INCOME_BAND",
              # Take primary account info from highest-balance account
              "ACCOUNT_TYPE", "product_name", "status", "ACCOUNT_AGE_CATEGORY"]

agg_dict = {}
for c in SUM_COLS:
    if c in seg.columns:
        agg_dict[c] = "sum"
for c in MAX_COLS:
    if c in seg.columns:
        agg_dict[c] = "max"
for c in FIRST_COLS:
    if c in seg.columns:
        agg_dict[c] = "first"
# Number of accounts per customer
agg_dict["account_id"] = "count"

# Sort by CURRENT_BALANCE descending so "first" picks the primary account
seg_sorted = seg.sort_values("CURRENT_BALANCE", ascending=False)
cust_agg = seg_sorted.groupby("customer_id", as_index=False).agg(agg_dict)
cust_agg = cust_agg.rename(columns={"account_id": "account_count"})

# Re-compute avg_trxn_amt at customer level
cust_agg["avg_trxn_amt"] = (
    cust_agg["total_trxn_amt"] / cust_agg["trxn_count"].replace(0, np.nan)
).fillna(0)

# Re-compute cashin_ratio at customer level
total_trxns = cust_agg["cashin_count"] + cust_agg["cashout_count"]
cust_agg["cashin_ratio"] = (
    cust_agg["cashin_count"] / total_trxns.replace(0, np.nan)
).fillna(0)

print(f"  Customer-level rows: {len(cust_agg):,}")
print(f"  Avg accounts per customer: {cust_agg['account_count'].mean():.1f}")
print(f"  INDIVIDUAL: {(cust_agg['customer_type']=='INDIVIDUAL').sum():,}")
print(f"  BUSINESS:   {(cust_agg['customer_type']=='BUSINESS').sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Rebuild SAR simulation for alerted customers
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2: Rebuilding SAR simulation (alerted customers)")
print("=" * 60)

psg = pd.read_csv(PSG_ALERTS)
print(f"  PSG alerts: {len(psg):,} transactions")

# Join: PSG alert subject_id → ss_segmentation account_id → customer_id
# (seg has both account_id and customer_id — look up account_id from original seg)
acct_cust = seg[["account_id", "customer_id"]].drop_duplicates()
psg_cust = psg.merge(acct_cust, left_on="subject_id", right_on="account_id", how="left")
print(f"  Matched to customer_id: {psg_cust['customer_id'].notna().sum():,} / {len(psg_cust):,}")

# Aggregate alerts at customer level
alert_agg = (
    psg_cust.groupby("customer_id", as_index=False)
    .agg(
        alert_count     = ("id",     "count"),
        total_alert_amt = ("amount", "sum"),
        max_alert_amt   = ("amount", "max"),
    )
)
print(f"  Unique alerted customers: {len(alert_agg):,}")

# Join alert aggregates to customer-level seg data
alerted = alert_agg.merge(
    cust_agg[["customer_id", "customer_type", "dynamic_segment",
              "avg_num_trxns", "avg_weekly_trxn_amt", "trxn_amt_monthly",
              "total_trxn_amt", "cashout_count"]],
    on="customer_id", how="left"
)

alerted["sar_score"] = sar_score(alerted)
n_sar_target = max(1, int(round(SAR_RATE * len(alerted))))
top_sar_idx  = alerted["sar_score"].nlargest(n_sar_target).index
alerted["is_sar"] = 0
alerted.loc[top_sar_idx, "is_sar"] = 1
alerted["source"] = "alerted"

n_sar = alerted["is_sar"].sum()
print(f"  SAR rate: {n_sar}/{len(alerted)} = {n_sar/len(alerted):.1%}")
print(f"  INDIVIDUAL SARs: {alerted[alerted['customer_type']=='INDIVIDUAL']['is_sar'].sum()}")
print(f"  BUSINESS SARs:   {alerted[alerted['customer_type']=='BUSINESS']['is_sar'].sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Simulate FN customers (non-alerted but truly suspicious)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Simulating FN customers (non-alerted suspicious)")
print("=" * 60)

alerted_ids = set(alerted["customer_id"])
non_alerted = cust_agg[~cust_agg["customer_id"].isin(alerted_ids)].copy()
print(f"  Non-alerted customer pool: {len(non_alerted):,}")

# FN customers should have metrics in the 20th–80th percentile of alerted
# population — they're suspicious but transact below alert-level volumes
p20_monthly = alerted["trxn_amt_monthly"].quantile(0.20)
p80_monthly = alerted["trxn_amt_monthly"].quantile(0.80)
p20_trxns   = alerted["avg_num_trxns"].quantile(0.20)
p80_trxns   = alerted["avg_num_trxns"].quantile(0.80)

fn_pool = non_alerted[
    (non_alerted["trxn_amt_monthly"] >= p20_monthly) &
    (non_alerted["trxn_amt_monthly"] <= p80_monthly) &
    (non_alerted["avg_num_trxns"]    >= p20_trxns) &
    (non_alerted["avg_num_trxns"]    <= p80_trxns)
].copy()
print(f"  FN pool (20th-80th pct metric range): {len(fn_pool):,}")

# Score within the pool and take the top FN_TARGET by SAR score
fn_pool["alert_count"]     = 0
fn_pool["total_alert_amt"] = 0
fn_pool["max_alert_amt"]   = 0
fn_pool["sar_score"]       = sar_score(fn_pool)

fn_customers = fn_pool.nlargest(FN_TARGET, "sar_score").copy()
fn_customers["is_sar"] = 1
fn_customers["source"] = "fn_non_alerted"
print(f"  FN customers selected: {len(fn_customers):,}")
print(f"  INDIVIDUAL FN: {(fn_customers['customer_type']=='INDIVIDUAL').sum()}")
print(f"  BUSINESS FN:   {(fn_customers['customer_type']=='BUSINESS').sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Merge and save sar_simulation.csv
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4: Saving sar_simulation.csv")
print("=" * 60)

SAR_COLS = ["customer_id", "customer_type", "dynamic_segment",
            "alert_count", "total_alert_amt", "max_alert_amt",
            "avg_num_trxns", "avg_weekly_trxn_amt", "trxn_amt_monthly",
            "sar_score", "is_sar", "source"]

sar_final = pd.concat([
    alerted[SAR_COLS],
    fn_customers[SAR_COLS]
], ignore_index=True)

sar_final.to_csv(SAR_OUT, index=False)
total_sars = sar_final["is_sar"].sum()
print(f"  Total rows: {len(sar_final):,}  (alerted: {len(alerted):,}, FN: {len(fn_customers):,})")
print(f"  Total SARs: {total_sars}  (alerted SARs: {n_sar}, FN customers: {len(fn_customers):,})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build custs_accts_txns_alerts.csv
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5: Building custs_accts_txns_alerts.csv")
print("=" * 60)

# Join alerted + FN customers back to customer-level aggregated features
all_subjects = sar_final.merge(
    cust_agg[[
        "customer_id", "ACCOUNT_TYPE", "product_name", "ACCT_AGE_YEARS",
        "CURRENT_BALANCE", "AGE", "AGE_CATEGORY", "GENDER", "CITIZENSHIP",
        "RESIDENCY_COUNTRY", "occupation", "INCOME", "naics",
        "nationality_country_id", "negative_news", "OFAC", "314b",
        "customer_type", "dynamic_segment", "INCOME_BAND",
        "avg_num_trxns", "avg_weekly_trxn_amt", "trxn_amt_monthly",
        "account_count",
    ]],
    on="customer_id", suffixes=("_sar", ""), how="left"
)

# Resolve any duplicate column names from the merge
for col in ["customer_type", "dynamic_segment", "avg_num_trxns",
            "avg_weekly_trxn_amt", "trxn_amt_monthly"]:
    sar_col = col + "_sar"
    if sar_col in all_subjects.columns:
        all_subjects[col] = all_subjects[col].fillna(all_subjects[sar_col])
        all_subjects.drop(columns=[sar_col], inplace=True)

# Derive ALERT / FP / FN flags
#   Alerted customers:   ALERT=Yes
#   FN-only customers:   ALERT=No
#   FP = alerted AND not truly suspicious (is_sar=0)
#   FN = truly suspicious (is_sar=1) — whether alerted or not
all_subjects["ALERT"] = all_subjects["source"].apply(
    lambda s: "Yes" if s == "alerted" else "No"
)
all_subjects["FP"] = all_subjects.apply(
    lambda r: "Yes" if r["source"] == "alerted" and r["is_sar"] == 0 else "No", axis=1
)
all_subjects["FN"] = all_subjects["is_sar"].apply(lambda x: "Yes" if x == 1 else "No")

# Build final output matching original schema
out = pd.DataFrame()
out["ACCOUNT_NUMBER"]       = all_subjects["customer_id"]   # customer-level: use customer_id
out["CUSTOMER_TYPE"]        = all_subjects["customer_type"]
out["ACCOUNT_TYPE"]         = all_subjects["ACCOUNT_TYPE"]
out["PRODUCT_NAME"]         = all_subjects["product_name"]
out["OPEN_DATE"]            = ""
out["ACCT_AGE_YEARS"]       = all_subjects["ACCT_AGE_YEARS"]
out["CURRENT_BALANCE"]      = all_subjects["CURRENT_BALANCE"]
out["CUSTOMER_ID"]          = all_subjects["customer_id"]
out["STATE"]                = ""
out["COUNTRY"]              = all_subjects["RESIDENCY_COUNTRY"]
out["BIRTH_DATE"]           = ""
out["AGE"]                  = all_subjects["AGE"]
out["AGE_CATEGORY"]         = all_subjects["AGE_CATEGORY"]
out["GENDER"]               = all_subjects["GENDER"]
out["CITIZENSHIP"]          = all_subjects["CITIZENSHIP"]
out["RESIDENCY_COUNTRY"]    = all_subjects["RESIDENCY_COUNTRY"]
out["OCCUPATION"]           = all_subjects["occupation"]
out["INCOME"]               = all_subjects["INCOME"]
out["NAICS"]                = all_subjects["naics"]
out["AVG_TRXNS_WEEK"]       = all_subjects["avg_num_trxns"]
out["AVG_TRXN_AMT"]         = all_subjects["avg_weekly_trxn_amt"]
out["TRXN_AMT_MONTHLY"]     = all_subjects["trxn_amt_monthly"]
out["REGISTRATION_COUNTRY"] = all_subjects["nationality_country_id"]
out["REGISTRATION_DATE"]    = ""
out["ACCT_OPEN_CHANNEL"]    = ""
out["NNM"]                  = all_subjects["negative_news"].apply(
                                  lambda x: "Yes" if x == 1 else "No")
out["OFAC"]                 = all_subjects["OFAC"].apply(
                                  lambda x: "Yes" if x == 1 else "No")
out["314b"]                 = all_subjects["314b"].apply(
                                  lambda x: "Yes" if x == 1 else "No")
out["ALERT"]                = all_subjects["ALERT"]
out["FP"]                   = all_subjects["FP"]
out["FN"]                   = all_subjects["FN"]

out.to_csv(ALERTS_OUT, sep="\t", index=False)

print(f"  Total rows:  {len(out):,}")
print(f"  ALERT=Yes:   {(out['ALERT']=='Yes').sum():,}")
print(f"  ALERT=No:    {(out['ALERT']=='No').sum():,}  (FN-only customers)")
print(f"  FP=Yes:      {(out['FP']=='Yes').sum():,}")
print(f"  FN=Yes:      {(out['FN']=='Yes').sum():,}")
print(f"  INDIVIDUAL:  {(out['CUSTOMER_TYPE']=='INDIVIDUAL').sum():,}")
print(f"  BUSINESS:    {(out['CUSTOMER_TYPE']=='BUSINESS').sum():,}")
print(f"\nSaved to: {ALERTS_OUT}")
print(f"SAR sim: {SAR_OUT}")
print("\nDone.")
