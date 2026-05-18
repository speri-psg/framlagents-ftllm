"""
network_features.py — Network Graph Feature Extraction

Reads docs/aml_transactions.csv, builds a directed transaction graph using networkx,
and computes per-customer (sender) features that capture laundering network behaviour.

Output: docs/network_features.csv
Columns (all per sender_customer_id):
    out_degree              — number of distinct external accounts received from this customer
    in_degree               — number of distinct external accounts that sent to this customer (via receiver lookup)
    unique_banks_sent_to    — number of distinct receiver_aba values
    unique_banks_rcvd_from  — number of distinct sender_aba values seen in reverse legs (always INTERNAL_ABA here,
                              so this is kept for schema completeness and equals 1 for all active senders)
    structuring_score       — fraction of outgoing txns with amount in [$8K, $10K)
    fan_out_score           — max outgoing txns to distinct accounts within any 72-hour window / total txns
    txn_velocity            — outgoing txns per active day (days with ≥1 txn)
    cross_bank_ratio        — fraction of outgoing txns going to EU / non-USD banks
    total_out_txns          — raw count of outgoing transactions
    total_out_amt           — total outgoing amount USD-equivalent
    max_single_txn          — largest single outgoing transaction
    avg_txn_amt             — mean outgoing transaction amount

Run:
    python network_features.py
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta

STRUCTURING_LOW  = 8_000
STRUCTURING_HIGH = 10_000   # CTR threshold
EU_CURRENCIES    = {"EUR", "GBP", "CHF"}   # non-USD → cross-border flag


def _fan_out_score(group: pd.DataFrame) -> float:
    """
    For each 72-hour rolling window, count distinct receiver accounts.
    Return (max distinct receivers in any 72h window) / total_txns.
    """
    if len(group) == 0:
        return 0.0
    group = group.sort_values("txn_dt")
    dates = group["txn_dt"].values
    receivers = group["receiver_account_id"].values
    n = len(dates)
    max_in_window = 1
    for i in range(n):
        window_end = dates[i] + np.timedelta64(72, "h")
        mask = (dates >= dates[i]) & (dates <= window_end)
        distinct = len(set(receivers[mask]))
        if distinct > max_in_window:
            max_in_window = distinct
    return round(max_in_window / n, 4)


def compute_network_features(txn_path: str) -> pd.DataFrame:
    df = pd.read_csv(txn_path, parse_dates=["txn_date"])
    df["txn_dt"] = df["txn_date"].astype("datetime64[ns]")

    # ── Build directed graph: sender_account_id → receiver_account_id ──────────
    # Nodes labelled by account; edge weight = transaction amount
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src = row["sender_account_id"]
        dst = row["receiver_account_id"]
        amt = row["amount"]
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += amt
            G[src][dst]["count"]  += 1
        else:
            G.add_edge(src, dst, weight=amt, count=1)

    # Map sender_account_id → sender_customer_id
    acct_to_cid = (
        df[["sender_account_id", "sender_customer_id"]]
        .drop_duplicates("sender_account_id")
        .set_index("sender_account_id")["sender_customer_id"]
        .to_dict()
    )

    # ── Per-customer feature extraction ────────────────────────────────────────
    records = []
    grouped = df.groupby("sender_customer_id")

    for cid, grp in grouped:
        sender_acct = grp["sender_account_id"].iloc[0]

        # Out-degree: distinct receiver accounts
        out_degree = grp["receiver_account_id"].nunique()

        # Unique banks sent to
        unique_banks_sent_to = grp["receiver_aba"].nunique()

        # Structuring score: fraction of txns in [$8K, $10K)
        struct_mask = (grp["amount"] >= STRUCTURING_LOW) & (grp["amount"] < STRUCTURING_HIGH)
        structuring_score = round(struct_mask.sum() / len(grp), 4) if len(grp) > 0 else 0.0

        # Fan-out score (72h window)
        fan_out = _fan_out_score(grp)

        # Transaction velocity: txns per active day
        active_days = grp["txn_date"].dt.date.nunique()
        txn_velocity = round(len(grp) / max(active_days, 1), 4)

        # Cross-bank ratio: EU/non-USD fraction
        cross_mask = grp["currency"].isin(EU_CURRENCIES)
        cross_bank_ratio = round(cross_mask.sum() / len(grp), 4) if len(grp) > 0 else 0.0

        # Amount stats
        total_out_txns = len(grp)
        total_out_amt  = round(grp["amount"].sum(), 2)
        max_single_txn = round(grp["amount"].max(), 2)
        avg_txn_amt    = round(grp["amount"].mean(), 2)

        # In-degree for this customer's sender account (how many unique sources send to it)
        # In our synthetic data all flows are outgoing from internal accounts,
        # so in_degree from graph perspective = predecessors of sender_acct node
        in_degree = G.in_degree(sender_acct) if sender_acct in G else 0

        # Unique banks received from (predecessors' bank ABA — structural placeholder)
        unique_banks_rcvd_from = 1 if in_degree > 0 else 0

        records.append({
            "customer_id":            cid,
            "out_degree":             out_degree,
            "in_degree":              in_degree,
            "unique_banks_sent_to":   unique_banks_sent_to,
            "unique_banks_rcvd_from": unique_banks_rcvd_from,
            "structuring_score":      structuring_score,
            "fan_out_score":          fan_out,
            "txn_velocity":           txn_velocity,
            "cross_bank_ratio":       cross_bank_ratio,
            "total_out_txns":         total_out_txns,
            "total_out_amt":          total_out_amt,
            "max_single_txn":         max_single_txn,
            "avg_txn_amt":            avg_txn_amt,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    _HERE    = os.path.dirname(os.path.abspath(__file__))
    txn_path = os.path.join(_HERE, "docs", "aml_transactions.csv")
    out_path = os.path.join(_HERE, "docs", "network_features.csv")

    if not os.path.exists(txn_path):
        raise FileNotFoundError(
            f"Transaction file not found: {txn_path}\n"
            "Run generate_aml_transactions.py first."
        )

    print(f"[network_features] Reading {txn_path} ...")
    feat_df = compute_network_features(txn_path)
    feat_df.to_csv(out_path, index=False)
    print(f"[network_features] {len(feat_df):,} customers | features: {list(feat_df.columns[1:])}")
    print(f"[network_features] Saved -> {out_path}")
