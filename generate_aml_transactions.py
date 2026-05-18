"""
generate_aml_transactions.py

Generates synthetic AML transaction data rooted in existing DF_SAR customer stats.
Produces docs/aml_transactions.csv with individual ACH/Wire records including
real ABA routing numbers for US and European correspondent banks.

Run:
    python generate_aml_transactions.py
"""

import os
import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── ABA Institution Pool ──────────────────────────────────────────────────────

US_BANKS = [
    ("JPMorgan Chase",   "021000021", "USD"),
    ("Bank of America",  "026009593", "USD"),
    ("Wells Fargo",      "121042882", "USD"),
    ("Citibank",         "021000089", "USD"),
    ("US Bank",          "091000022", "USD"),
    ("TD Bank",          "011103093", "USD"),
    ("PNC Bank",         "043000096", "USD"),
    ("Capital One",      "056073502", "USD"),
    ("Truist",           "061000104", "USD"),
    ("Regions Bank",     "062000019", "USD"),
    ("Fifth Third",      "042000314", "USD"),
    ("KeyBank",          "041001039", "USD"),
    ("M&T Bank",         "022000046", "USD"),
    ("Santander USA",    "231372691", "USD"),
    ("BMO Harris",       "071025661", "USD"),
    ("HSBC USA",         "021001088", "USD"),
    ("Citizens Bank",    "011500010", "USD"),
    ("Ally Bank",        "124003116", "USD"),
    ("Goldman Sachs",    "124085066", "USD"),
    ("Western Alliance", "122400779", "USD"),
]

EU_BANKS = [
    ("Deutsche Bank",    "026003780", "EUR"),
    ("Barclays",         "026002561", "GBP"),
    ("BNP Paribas",      "026007689", "EUR"),
    ("Societe Generale", "026008073", "EUR"),
    ("Credit Suisse",    "026009557", "CHF"),
    ("UBS",              "026007993", "CHF"),
    ("ING Bank",         "026009632", "EUR"),
    ("ABN AMRO",         "026009464", "EUR"),
    ("Rabobank",         "026010420", "EUR"),
    ("RBS",              "026005092", "GBP"),
    ("Lloyds Bank",      "026002794", "GBP"),
    ("UniCredit",        "026007262", "EUR"),
    ("Intesa Sanpaolo",  "026009413", "EUR"),
    ("Banco Santander",  "026009444", "EUR"),
    ("BBVA",             "026009180", "EUR"),
    ("Commerzbank",      "026003089", "EUR"),
    ("Raiffeisen Bank",  "026007508", "EUR"),
    ("Nordea",           "026010786", "EUR"),
    ("DNB Bank",         "026013444", "EUR"),
    ("Societe Generale", "026008073", "EUR"),
]

ALL_BANKS        = US_BANKS + EU_BANKS
INTERNAL_ABA     = "021000999"   # fictitious ABA for our monitored institution
INTERNAL_BANK    = "First National Demo Bank"
SIM_START        = datetime(2023, 1, 1)
SIM_END          = datetime(2023, 12, 31)
SIM_DAYS         = (SIM_END - SIM_START).days
STRUCTURING_HIGH = 9_900   # just below $10K CTR threshold
STRUCTURING_LOW  = 8_000


def _random_account():
    """Generate a realistic 10-digit account number."""
    return str(random.randint(1_000_000_000, 9_999_999_999))


def _random_date(start=SIM_START, end=SIM_END):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _pick_bank(eu_weight=0.15):
    """Pick a random external institution. EU banks have eu_weight probability."""
    if random.random() < eu_weight:
        return random.choice(EU_BANKS)
    return random.choice(US_BANKS)


def _txn_type(bank):
    """ACH for domestic USD, Wire for large or EU transactions."""
    _, _, currency = bank
    if currency != "USD":
        return "Wire"
    return random.choice(["ACH", "ACH", "Wire"])   # 2:1 ACH bias for domestic


# ── External account pool ─────────────────────────────────────────────────────
# Pre-generate a pool of external accounts so some accounts appear repeatedly
# (layering targets) while most appear once (smurfing destinations).

N_REPEAT_ACCOUNTS = 200    # accounts that appear multiple times (intermediaries)
N_ONEOFF_ACCOUNTS = 5_000  # single-use destination accounts

_repeat_pool = [
    {"acct": _random_account(), "bank": random.choice(ALL_BANKS)}
    for _ in range(N_REPEAT_ACCOUNTS)
]
_oneoff_pool = [
    {"acct": _random_account(), "bank": random.choice(ALL_BANKS)}
    for _ in range(N_ONEOFF_ACCOUNTS)
]


def _external_account(prefer_repeat=False):
    if prefer_repeat or random.random() < 0.2:
        return random.choice(_repeat_pool)
    return random.choice(_oneoff_pool)


# ── Transaction generators ────────────────────────────────────────────────────

def _make_txn(txn_date, sender_customer_id, sender_account_id,
              receiver_acct, receiver_bank, amount, pattern_type, is_suspicious):
    bank_name, aba, currency = receiver_bank
    txn_type = _txn_type(receiver_bank)
    return {
        "txn_id":               str(uuid.uuid4())[:12].upper(),
        "txn_date":             txn_date.strftime("%Y-%m-%d"),
        "txn_type":             txn_type,
        "sender_customer_id":   sender_customer_id,
        "sender_account_id":    sender_account_id,
        "sender_aba":           INTERNAL_ABA,
        "sender_bank_name":     INTERNAL_BANK,
        "receiver_account_id":  receiver_acct,
        "receiver_aba":         aba,
        "receiver_bank_name":   bank_name,
        "amount":               round(amount, 2),
        "currency":             currency,
        "is_suspicious":        int(is_suspicious),
        "pattern_type":         pattern_type,
    }


def generate_normal_transactions(customer_id, account_id, avg_weekly_amt, avg_num_trxns):
    """Generate routine non-suspicious transactions for a normal customer."""
    txns = []
    # Estimate weekly transaction count (Poisson)
    weekly_count = max(1, int(round(avg_num_trxns)))
    n_weeks = 52
    for w in range(n_weeks):
        week_start = SIM_START + timedelta(weeks=w)
        for _ in range(np.random.poisson(weekly_count)):
            amount = max(10, np.random.lognormal(
                mean=np.log(max(avg_weekly_amt / weekly_count, 100)),
                sigma=0.5
            ))
            ext = _external_account()
            txn_date = week_start + timedelta(days=random.randint(0, 6))
            if txn_date > SIM_END:
                continue
            txns.append(_make_txn(
                txn_date, customer_id, account_id,
                ext["acct"], ext["bank"],
                amount, "NORMAL", False
            ))
    return txns


def generate_fanout(customer_id, account_id, base_amt):
    """Fan-Out: one account rapidly sends to many receivers in a short window."""
    txns = []
    n_legs = random.randint(5, 12)
    start  = _random_date(SIM_START, SIM_END - timedelta(days=5))
    total  = base_amt * random.uniform(3, 8)
    for i in range(n_legs):
        amt      = total / n_legs * random.uniform(0.7, 1.3)
        ext      = _external_account(prefer_repeat=False)
        txn_date = start + timedelta(hours=random.randint(0, 72))
        if txn_date > SIM_END:
            continue
        txns.append(_make_txn(
            txn_date, customer_id, account_id,
            ext["acct"], ext["bank"],
            amt, "FAN-OUT", True
        ))
    return txns


def generate_structuring(customer_id, account_id):
    """Structuring: multiple transactions just below $10K CTR threshold."""
    txns = []
    n    = random.randint(3, 8)
    start = _random_date(SIM_START, SIM_END - timedelta(days=14))
    for i in range(n):
        amt      = random.uniform(STRUCTURING_LOW, STRUCTURING_HIGH)
        ext      = _external_account()
        txn_date = start + timedelta(days=random.randint(0, 10))
        if txn_date > SIM_END:
            continue
        txns.append(_make_txn(
            txn_date, customer_id, account_id,
            ext["acct"], ext["bank"],
            amt, "STRUCTURING", True
        ))
    return txns


def generate_layering(customer_id, account_id, intermediary_accounts, base_amt):
    """Layering: funds flow through repeat intermediary accounts before final destination."""
    txns  = []
    start = _random_date(SIM_START, SIM_END - timedelta(days=20))
    amt   = base_amt * random.uniform(2, 5)
    for hop, intermediary in enumerate(intermediary_accounts[:random.randint(2, 5)]):
        amt      *= random.uniform(0.85, 0.98)   # slight reduction each hop (fees)
        txn_date  = start + timedelta(days=hop * random.randint(1, 4))
        if txn_date > SIM_END:
            continue
        txns.append(_make_txn(
            txn_date, customer_id, account_id,
            intermediary["acct"], intermediary["bank"],
            amt, "LAYERING", True
        ))
    # Final leg to a one-off destination
    ext = _external_account(prefer_repeat=False)
    txns.append(_make_txn(
        start + timedelta(days=15), customer_id, account_id,
        ext["acct"], ext["bank"],
        amt * 0.9, "LAYERING", True
    ))
    return txns


def generate_rapid_movement(customer_id, account_id, base_amt):
    """Rapid movement: large amount sent and received within a short window."""
    txns  = []
    start = _random_date(SIM_START, SIM_END - timedelta(days=7))
    amt   = base_amt * random.uniform(4, 10)
    # Outgoing burst
    for i in range(random.randint(3, 6)):
        ext      = _external_account(prefer_repeat=True)
        txn_date = start + timedelta(hours=random.randint(0, 48))
        if txn_date > SIM_END:
            continue
        txns.append(_make_txn(
            txn_date, customer_id, account_id,
            ext["acct"], ext["bank"],
            amt / 4, "RAPID-MOVEMENT", True
        ))
    return txns


# ── Main generation ───────────────────────────────────────────────────────────

def generate(sar_csv_path, out_path):
    df_sar = pd.read_csv(sar_csv_path)
    print(f"[generate] Loaded {len(df_sar):,} customers | SARs={int(df_sar['is_sar'].sum()):,}")

    # Pre-build intermediary pool from repeat accounts (for layering hops)
    intermediary_pool = _repeat_pool[:50]

    all_txns = []
    for _, row in df_sar.iterrows():
        cid         = row["customer_id"]
        account_id  = _random_account()
        is_sar      = int(row["is_sar"])
        avg_amt     = max(float(row.get("avg_weekly_trxn_amt", 1000)), 100)
        avg_trxns   = max(float(row.get("avg_num_trxns", 1)), 0.1)

        # Every customer gets some normal transactions
        txns = generate_normal_transactions(cid, account_id, avg_amt, avg_trxns)

        # SAR customers get laundering patterns injected
        if is_sar:
            patterns = random.choices(
                ["fanout", "structuring", "layering", "rapid"],
                weights=[0.35, 0.30, 0.20, 0.15],
                k=random.randint(1, 2)
            )
            for pattern in patterns:
                if pattern == "fanout":
                    txns += generate_fanout(cid, account_id, avg_amt)
                elif pattern == "structuring":
                    txns += generate_structuring(cid, account_id)
                elif pattern == "layering":
                    txns += generate_layering(cid, account_id, intermediary_pool, avg_amt)
                elif pattern == "rapid":
                    txns += generate_rapid_movement(cid, account_id, avg_amt)

        all_txns.extend(txns)

    df_out = pd.DataFrame(all_txns).sort_values("txn_date").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)

    n_suspicious = int(df_out["is_suspicious"].sum())
    print(f"[generate] {len(df_out):,} transactions | Suspicious={n_suspicious:,} ({100*n_suspicious/len(df_out):.1f}%)")
    print(f"[generate] Saved -> {out_path}")
    return df_out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import SAR_CSV
    _HERE    = os.path.dirname(os.path.abspath(__file__))
    sar_path = SAR_CSV
    out_path = os.path.join(_HERE, "docs", "aml_transactions.csv")
    generate(sar_path, out_path)
