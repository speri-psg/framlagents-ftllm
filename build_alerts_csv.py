"""
build_alerts_csv.py — Rebuild docs/custs_accts_txns_alerts.csv from:
  - docs/ss_segmentation_data.csv  (transaction features for all customers)
  - docs/sar_simulation.csv        (which customers were alerted / filed SARs)

Output schema matches the original file so application.py needs no changes.

Threshold analysis logic:
  - ALERT = Yes for all 4,835 alerted customers
  - FP    = Yes where is_sar=0  (alerted but not truly suspicious)
  - FN    = Yes where is_sar=1  (truly suspicious — missed when threshold rises above metric)

At the lowest threshold  → FP = 4,351 (all non-SAR customers alert), FN = 0
As threshold rises       → FP decreases, FN increases
At the highest threshold → FP = 0, FN = 484
"""

import os
import pandas as pd

_HERE    = os.path.dirname(os.path.abspath(__file__))
SEG_PATH = os.path.join(_HERE, "docs", "ss_segmentation_data.csv")
SAR_PATH = os.path.join(_HERE, "docs", "sar_simulation.csv")
OUT_PATH = os.path.join(_HERE, "docs", "custs_accts_txns_alerts.csv")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading ss_segmentation_data.csv ...")
seg = pd.read_csv(SEG_PATH, low_memory=False)

print("Loading sar_simulation.csv ...")
sar = pd.read_csv(SAR_PATH, usecols=["customer_id", "is_sar"])

# ── Join: keep only alerted customers, one row per customer ──────────────────
# ss_segmentation_data has one row per account — deduplicate to one row per
# customer (keep row with highest avg_num_trxns as most representative account)
seg_dedup = (
    seg.sort_values("avg_num_trxns", ascending=False)
       .drop_duplicates(subset="customer_id", keep="first")
)
print(f"ss_segmentation_data: {len(seg):,} rows -> {len(seg_dedup):,} unique customers")

df = seg_dedup.merge(sar, on="customer_id", how="inner")
print(f"Alerted customers joined: {len(df):,}")
print(f"  INDIVIDUAL: {(df['customer_type']=='INDIVIDUAL').sum():,}")
print(f"  BUSINESS:   {(df['customer_type']=='BUSINESS').sum():,}")

# ── Build FP / FN / ALERT flags ───────────────────────────────────────────────
df["ALERT"] = "Yes"
df["FP"]    = df["is_sar"].apply(lambda x: "No"  if x == 1 else "Yes")
df["FN"]    = df["is_sar"].apply(lambda x: "Yes" if x == 1 else "No")

# ── Rename / select columns to match original schema ─────────────────────────
out = pd.DataFrame()
out["ACCOUNT_NUMBER"]      = df["account_id"]
out["CUSTOMER_TYPE"]       = df["customer_type"]
out["ACCOUNT_TYPE"]        = df["ACCOUNT_TYPE"]
out["PRODUCT_NAME"]        = df["product_name"]
out["OPEN_DATE"]           = df.get("open_date", pd.Series("", index=df.index))
out["ACCT_AGE_YEARS"]      = df["ACCT_AGE_YEARS"]
out["CURRENT_BALANCE"]     = df["CURRENT_BALANCE"]
out["CUSTOMER_ID"]         = df["customer_id"]
out["STATE"]               = ""   # not available in ss_segmentation_data
out["COUNTRY"]             = df["RESIDENCY_COUNTRY"]
out["BIRTH_DATE"]          = ""   # not available
out["AGE"]                 = df["AGE"]
out["AGE_CATEGORY"]        = df["AGE_CATEGORY"]
out["GENDER"]              = df["GENDER"]
out["CITIZENSHIP"]         = df["CITIZENSHIP"]
out["RESIDENCY_COUNTRY"]   = df["RESIDENCY_COUNTRY"]
out["OCCUPATION"]          = df["occupation"]
out["INCOME"]              = df["INCOME"]
out["NAICS"]               = df["naics"]
out["AVG_TRXNS_WEEK"]      = df["avg_num_trxns"]
out["AVG_TRXN_AMT"]        = df["avg_weekly_trxn_amt"]
out["TRXN_AMT_MONTHLY"]    = df["trxn_amt_monthly"]
out["REGISTRATION_COUNTRY"]= df["nationality_country_id"]
out["REGISTRATION_DATE"]   = ""
out["ACCT_OPEN_CHANNEL"]   = df.get("ACCOUNT_AGE_CATEGORY", pd.Series("", index=df.index))
out["NNM"]                 = df["negative_news"].apply(lambda x: "Yes" if x == 1 else "No")
out["OFAC"]                = df["OFAC"].apply(lambda x: "Yes" if x == 1 else "No")
out["314b"]                = df["314b"].apply(lambda x: "Yes" if x == 1 else "No")
out["ALERT"]               = df["ALERT"]
out["FP"]                  = df["FP"]
out["FN"]                  = df["FN"]

# ── Write tab-separated (matches original) ────────────────────────────────────
out.to_csv(OUT_PATH, sep="\t", index=False)
print(f"\nSaved {len(out):,} rows to {OUT_PATH}")

# ── Sanity check ──────────────────────────────────────────────────────────────
print(f"\nALERT=Yes: {(out['ALERT']=='Yes').sum():,}")
print(f"FP=Yes:    {(out['FP']=='Yes').sum():,}  (non-SAR alerted)")
print(f"FN=Yes:    {(out['FN']=='Yes').sum():,}  (SAR alerted — missed when threshold raised)")
print(f"smart_segment_id check:")
print(f"  INDIVIDUAL (id=1): {(df['smart_segment_id']==1).sum():,}")
print(f"  BUSINESS   (id=0): {(df['smart_segment_id']==0).sum():,}")
