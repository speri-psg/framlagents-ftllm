"""
generate_synthetic_data.py — Generate synthetic AML customer/account/transaction data.

Produces aria_synth/ directory with:
  aria_customers.csv
  aria_accounts.csv
  aria_account_relationships.csv
  aria_transactions.csv
  aria_alerts.csv               (condition details + SAR labels)

Covers all 11 rules in rule_catalogue.json plus extended typologies:
  1.  Activity Deviation (ACH)
  2.  Activity Deviation (Check)
  3.  Activity Deviation (Wire)          [new]
  4.  Elder Abuse
  5.  Velocity Single
  6.  Velocity Multiple / Fan-out        [new]
  7.  Detect Excessive Transaction Activity
  8.  Structuring (Incoming Cash)
  9.  Structuring (Outgoing Cash)
  10. CTR Client
  11. Burst in Originator Activity
  12. Burst in Beneficiary Activity
  13. Risky International Transfer
  14. Funnel Account                     [new]
  15. Round-trip / Loan-back             [new]
  16. Human Trafficking Indicators       [new]

Usage:
    python generate_synthetic_data.py [--customers N] [--seed S] [--out DIR]

    --customers : total synthetic customer count (default 5000)
    --seed      : random seed for reproducibility (default 42)
    --out       : output directory (default aria_synth)
"""

import argparse
import os
import uuid
import random
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

REF_DATE   = datetime(2026, 3, 31)
START_DATE = datetime(2025, 3, 31)

EDUCATION_LEVELS = [
    "Less than High School",
    "High School / GED",
    "Some College",
    "Associate's Degree",
    "Bachelor's Degree",
    "Master's Degree",
    "Doctoral / Professional",
]
# US Census approximate distribution for adults 25+
EDUCATION_PROBS = [0.10, 0.27, 0.20, 0.08, 0.22, 0.10, 0.03]

# Education → income multiplier
EDUCATION_INCOME_MULT = {
    "Less than High School":    0.65,
    "High School / GED":        0.80,
    "Some College":             0.90,
    "Associate's Degree":       0.95,
    "Bachelor's Degree":        1.10,
    "Master's Degree":          1.28,
    "Doctoral / Professional":  1.55,
}

# Occupation → base annual income (US median approximations)
OCCUPATIONS_INDIVIDUAL = {
    "Retail Associate":           35_000,
    "Security Guard":             38_000,
    "Food Service Worker":        32_000,
    "Administrative Assistant":   44_000,
    "Medical Assistant":          40_000,
    "Chef":                       48_000,
    "Social Worker":              52_000,
    "Construction Worker":        55_000,
    "Truck Driver":               57_000,
    "Teacher":                    62_000,
    "Sales Representative":       60_000,
    "Office Manager":             62_000,
    "Plumber":                    65_000,
    "Electrician":                68_000,
    "Graphic Designer":           57_000,
    "Accountant":                 78_000,
    "Nurse":                      78_000,
    "Marketing Manager":          85_000,
    "Physical Therapist":         95_000,
    "Financial Analyst":          95_000,
    "Software Engineer":         115_000,
    "Physician":                 220_000,
    "Attorney":                  130_000,
}
# Sampling weights — more blue-collar/service than professional (reflects workforce distribution)
_OCC_KEYS   = list(OCCUPATIONS_INDIVIDUAL.keys())
_OCC_PROBS_RAW = [
    0.08, 0.04, 0.07,   # Retail, Security, Food Service
    0.07, 0.05, 0.05,   # Admin, Med Asst, Chef
    0.05, 0.06, 0.06,   # Social, Construction, Truck
    0.06, 0.05, 0.04,   # Teacher, Sales, Office Mgr
    0.04, 0.04, 0.04,   # Plumber, Electrician, Graphic
    0.04, 0.05, 0.03,   # Accountant, Nurse, Mktg Mgr
    0.03, 0.03, 0.05,   # PT, Fin Analyst, SW Eng
    0.02, 0.02,         # Physician, Attorney
]
_OCC_PROBS = [p / sum(_OCC_PROBS_RAW) for p in _OCC_PROBS_RAW]

OCCUPATIONS_BUSINESS = [
    "Business Owner", "Director", "CEO", "CFO", "Controller",
    "Operations Manager", "General Manager", "Vice President",
]

COUNTRIES_HIGH_RISK = ["Iran", "North Korea", "Cuba", "Syria", "Russia",
                       "Venezuela", "Myanmar", "Afghanistan", "Belarus", "Sudan"]
COUNTRIES_MED_RISK  = ["Nigeria", "Mexico", "Colombia", "Pakistan", "UAE",
                       "Turkey", "China", "Ukraine", "Kazakhstan", "Egypt"]
COUNTRIES_LOW_RISK  = ["United States", "Canada", "United Kingdom", "Germany",
                       "France", "Australia", "Japan", "Netherlands", "Sweden", "Switzerland"]

NAICS_BUSINESS = [
    "522110",  # Commercial Banking
    "441110",  # New Car Dealers
    "722511",  # Full-Service Restaurants
    "236220",  # Commercial Building Construction
    "423860",  # Transportation Equipment Wholesale
    "531110",  # Lessors of Residential Buildings
    "561320",  # Temporary Staffing Agencies
    "711211",  # Sports Teams and Clubs
    "713210",  # Casinos (except Casino Hotels)
    "812990",  # All Other Personal Services
]

ACCOUNT_PRODUCTS = {
    "Checking":            ["PREFERRED CHECKING", "PREMIER CHECKING", "ICHECKING", "Checking Account"],
    "Savings":             ["POD Savings Account", "Trust Savings Account", "Money Market Savings"],
    "Loan":                ["LN CR- 60M", "Fixed 30 Yr 1st Mortgage", "Auto Loan", "Line of Credit"],
    "Certificate Deposit": ["Certificate Account 1 Year", "Certificate Account 6 Month"],
    "Credit Card":         ["SR Visa Signature Rewards", "PP Visa Platinum Rewards"],
    "Foreign Currency":    ["Foreign Currency Account"],
}

TXN_TYPES_NORMAL = ["Debit Card", "Credit Card", "ACH", "Check", "Misc Deposit",
                    "Misc Debit", "Wire", "Misc Credit", "Misc Withdrawal"]

SAR_RATE_BY_RULE = {
    "Activity Deviation (ACH)":             0.15,
    "Activity Deviation (Check)":           0.12,
    "Activity Deviation (Wire)":            0.18,
    "Elder Abuse":                          0.17,
    "Velocity Single":                      0.16,
    "Velocity Multiple":                    0.20,
    "Detect Excessive Transaction Activity":0.13,
    "Structuring (Incoming Cash)":          0.14,
    "Structuring (Outgoing Cash)":          0.13,
    "CTR Client":                           0.10,
    "Burst in Originator Activity":         0.15,
    "Burst in Beneficiary Activity":        0.14,
    "Risky International Transfer":         0.22,
    "Funnel Account":                       0.25,
    "Round-trip":                           0.28,
    "Human Trafficking Indicators":         0.30,
}

# Customers per rule (alerted population)
ALERTED_PER_RULE = {
    "Activity Deviation (ACH)":             300,
    "Activity Deviation (Check)":           200,
    "Activity Deviation (Wire)":            150,
    "Elder Abuse":                          400,
    "Velocity Single":                      250,
    "Velocity Multiple":                    200,
    "Detect Excessive Transaction Activity":350,
    "Structuring (Incoming Cash)":          300,
    "Structuring (Outgoing Cash)":          250,
    "CTR Client":                           400,
    "Burst in Originator Activity":         200,
    "Burst in Beneficiary Activity":        200,
    "Risky International Transfer":         150,
    "Funnel Account":                       150,
    "Round-trip":                           100,
    "Human Trafficking Indicators":         100,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cust_id(n):   return f"aria_cust_{n:06d}"
def _acct_id(n):   return f"aria_acct_{n:06d}"
def _txn_id(n):    return n
def _alert_id(n):  return f"aria_alert_{n:05d}"
def _cond_id(rule): return str(uuid.uuid5(uuid.NAMESPACE_DNS, rule))

def _rand_date(start, end, rng):
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta)))

def _fmt_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ── Generators ────────────────────────────────────────────────────────────────

def _sample_age(rng):
    """
    Age distribution for banking customers — bell-shaped peaking 38-52,
    min 18, max 92. Uses a mixture: working-age bulk + elder tail.
    """
    while True:
        age = rng.normal(loc=44, scale=15)
        if 18 <= age <= 92:
            return round(age, 1)


def _marital_status(age, rng):
    """Marital status weighted by age bracket (US Census proportions)."""
    if age < 30:
        probs = [0.60, 0.30, 0.06, 0.01, 0.03]   # Single, Married, Divorced, Widowed, Separated
    elif age < 45:
        probs = [0.25, 0.55, 0.14, 0.02, 0.04]
    elif age < 60:
        probs = [0.13, 0.52, 0.20, 0.09, 0.06]
    else:
        probs = [0.07, 0.43, 0.14, 0.32, 0.04]
    return rng.choice(["Single", "Married", "Divorced", "Widowed", "Separated"], p=probs)


def _sample_country(rng):
    """
    Citizenship for a US bank customer population.
    ~83% domestic, ~12% other low-risk, ~4% medium-risk, ~1% high-risk.
    """
    tier = rng.choice(["US", "low", "med", "high"], p=[0.83, 0.12, 0.04, 0.01])
    if tier == "US":
        return "United States"
    elif tier == "low":
        others = [c for c in COUNTRIES_LOW_RISK if c != "United States"]
        return rng.choice(others)
    elif tier == "med":
        return rng.choice(COUNTRIES_MED_RISK)
    else:
        return rng.choice(COUNTRIES_HIGH_RISK)


def _sample_income(occupation, education, age, rng):
    """
    Income correlated with occupation base, education multiplier, and age curve.
    Age peaks at 48-55 then tapers (retirement/reduced hours).
    """
    base      = OCCUPATIONS_INDIVIDUAL.get(occupation, 55_000)
    edu_mult  = EDUCATION_INCOME_MULT.get(education, 1.0)
    # Age income curve: rises through 30s-40s, peaks ~50, tapers after 65
    if age < 25:
        age_mult = 0.60
    elif age < 35:
        age_mult = 0.80
    elif age < 50:
        age_mult = 1.00
    elif age < 60:
        age_mult = 1.08
    elif age < 70:
        age_mult = 0.90   # partial retirement
    else:
        age_mult = 0.65   # pension/SS income
    raw = base * edu_mult * age_mult * rng.uniform(0.80, 1.25)
    return max(18_000, int(raw))


def generate_customers(n_individual, n_business, rng):
    rows = []
    cid  = 1

    # Individual customers
    for _ in range(n_individual):
        age       = _sample_age(rng)
        birth     = REF_DATE - timedelta(days=int(age * 365.25))
        education = rng.choice(EDUCATION_LEVELS, p=EDUCATION_PROBS)
        occupation = rng.choice(_OCC_KEYS, p=_OCC_PROBS)
        income    = _sample_income(occupation, education, age, rng)
        country   = _sample_country(rng)

        rows.append({
            "aria_customer_id":           _cust_id(cid),
            "aria_entity":                False,
            "aria_is_customer":           True,
            "aria_birthdate":             _fmt_dt(birth),
            "aria_age":                   age,
            "aria_gender":                rng.choice(["M", "F", "U"], p=[0.489, 0.489, 0.022]),
            "aria_marital_status":        _marital_status(age, rng),
            "aria_education":             education,
            "aria_occupation":            occupation,
            "aria_gross_annual_income":   income,
            "aria_citizenship":           country,
            "aria_residency_country":     country,
            "aria_nationality":           country,
            "aria_naics":                 None,
            "aria_pep":                   bool(rng.random() < 0.005),
            "aria_negative_news":         bool(rng.random() < 0.01),
            "aria_ofac":                  bool(rng.random() < 0.002),
            "aria_sanctions_hit":         bool(rng.random() < 0.003),  # EU/UN/OFAC combined
            "aria_subpoena":              bool(rng.random() < 0.005),  # active legal hold
            "aria_exempt_cdd":            False,
            "aria_exempt_sanctions":      False,
            "aria_internal_employee":     bool(rng.random() < 0.02),
            "aria_customer_type":         "Individual",
        })
        cid += 1

    # Business customers (entities)
    for _ in range(n_business):
        country = _sample_country(rng)
        # Business revenue: log-normal — most are small, some large
        revenue = max(100_000, int(np.exp(rng.normal(np.log(400_000), 1.2))))
        rows.append({
            "aria_customer_id":           _cust_id(cid),
            "aria_entity":                True,
            "aria_is_customer":           True,
            "aria_birthdate":             None,
            "aria_age":                   None,
            "aria_gender":                None,
            "aria_marital_status":        None,
            "aria_education":             None,
            "aria_occupation":            rng.choice(OCCUPATIONS_BUSINESS),
            "aria_gross_annual_income":   revenue,
            "aria_citizenship":           country,
            "aria_residency_country":     country,
            "aria_nationality":           country,
            "aria_naics":                 rng.choice(NAICS_BUSINESS),
            "aria_pep":                   False,
            "aria_negative_news":         bool(rng.random() < 0.02),
            "aria_ofac":                  bool(rng.random() < 0.005),
            "aria_sanctions_hit":         bool(rng.random() < 0.008),  # businesses higher exposure
            "aria_subpoena":              bool(rng.random() < 0.010),  # businesses more likely
            "aria_exempt_cdd":            bool(rng.random() < 0.05),
            "aria_exempt_sanctions":      False,
            "aria_internal_employee":     False,
            "aria_customer_type":         "Business",
        })
        cid += 1

    return pd.DataFrame(rows)


def _open_channel(open_dt, rng):
    """
    Account opening channel weighted by year — older accounts skew in-person,
    newer accounts skew mobile/online.
    """
    yr = open_dt.year
    if yr < 2015:
        probs = [0.70, 0.20, 0.08, 0.02, 0.00]  # In-Person, Phone, Online, Mobile, Referral
    elif yr < 2018:
        probs = [0.50, 0.15, 0.22, 0.10, 0.03]
    elif yr < 2021:
        probs = [0.30, 0.10, 0.28, 0.27, 0.05]
    else:
        probs = [0.15, 0.05, 0.25, 0.48, 0.07]
    return rng.choice(["In-Person", "Phone", "Online", "Mobile", "Referral"], p=probs)


def _rand_open_date(rng):
    """
    Account open date with recency bias: exponential-ish weight toward recent years.
    Produces more accounts opened in the last 5 years, a long tail back to 2006.
    """
    # Sample year with weighted probabilities (sums to 1)
    years  = list(range(2006, 2026))
    # Linearly increasing weight: 2006=1, 2025=20 → normalised
    raw_w  = [y - 2005 for y in years]
    total  = sum(raw_w)
    probs  = [w / total for w in raw_w]
    yr     = int(rng.choice(years, p=probs))
    # Random month/day within that year (cap at 2025-12-31)
    start  = datetime(yr, 1, 1)
    end    = datetime(min(yr, 2025), 12, 31)
    return _rand_date(start, end, rng)


def generate_accounts(customers_df, rng):
    acct_rows = []
    rel_rows  = []
    acct_id   = 1

    cust_list = customers_df["aria_customer_id"].tolist()
    cust_map  = customers_df.set_index("aria_customer_id").to_dict("index")
    indiv_custs = [c for c in cust_list if not cust_map[c]["aria_entity"]]

    for cid in cust_list:
        is_entity = cust_map[cid]["aria_entity"]

        if is_entity:
            # Business: always Checking, sometimes Savings + Loan + Foreign Currency
            types = ["Checking"]
            if rng.random() < 0.50:
                types.append("Savings")
            if rng.random() < 0.35:
                types.append("Loan")
            if rng.random() < 0.10:
                types.append("Credit Card")
            if rng.random() < 0.05:
                types.append("Foreign Currency")
        else:
            # Individual: always Checking; add others probabilistically
            types = ["Checking"]
            if rng.random() < 0.55:
                types.append("Savings")
            if rng.random() < 0.30:
                types.append("Credit Card")
            if rng.random() < 0.15:
                types.append("Loan")
            if rng.random() < 0.10:
                types.append("Certificate Deposit")
            if rng.random() < 0.02:
                types.append("Foreign Currency")

        for acct_type in types:
            aid     = _acct_id(acct_id)
            open_dt = _rand_open_date(rng)
            product = rng.choice(ACCOUNT_PRODUCTS.get(acct_type, ["Unknown"]))
            channel = _open_channel(open_dt, rng)

            # Balance correlated loosely with account age (older → more accumulated)
            acct_age_days = (REF_DATE - open_dt).days
            age_factor    = min(acct_age_days / 365, 10)           # cap at 10 years effect
            base_balance  = max(0, int(rng.exponential(15_000 + age_factor * 3_000)))
            balance       = base_balance

            loan_release = loan_amount = loan_maturity = None
            if acct_type == "Loan":
                loan_amount   = max(10_000, int(rng.exponential(80_000)))
                loan_release  = _fmt_dt(open_dt)
                loan_maturity = _fmt_dt(open_dt + timedelta(days=int(rng.integers(365, 3650))))
                balance       = 0   # loan accounts don't hold a deposit balance

            acct_rows.append({
                "aria_account_id":          aid,
                "aria_account_type":        acct_type,
                "aria_product_name":        product,
                "aria_open_date":           _fmt_dt(open_dt),
                "aria_open_channel":        channel,
                "aria_status":              "ACTIVE" if rng.random() < 0.95 else "CLOSED",
                "aria_initial_balance":     0,
                "aria_current_balance":     balance,
                "aria_loan_amount":         loan_amount,
                "aria_loan_release_date":   loan_release,
                "aria_loan_maturity_date":  loan_maturity,
                # filled in post-hoc after transactions are generated
                "aria_peak_activity_period": None,
            })

            # Primary relationship
            if acct_type == "Loan":
                rel_rows.append({"aria_account_id": aid, "aria_customer_id": cid,
                                  "aria_relationship_type": "BORROWER"})
                if rng.random() < 0.2 and len(cust_list) > 1:
                    co = rng.choice(cust_list)
                    if co != cid:
                        rel_rows.append({"aria_account_id": aid, "aria_customer_id": co,
                                          "aria_relationship_type": "GUARANTOR"})
            elif is_entity:
                rel_rows.append({"aria_account_id": aid, "aria_customer_id": cid,
                                  "aria_relationship_type": "OWNER"})
                n_conductors = int(rng.integers(1, 4))
                conductors = rng.choice(
                    indiv_custs, size=min(n_conductors, len(indiv_custs), 3), replace=False
                )
                for conductor in conductors:
                    rel_rows.append({"aria_account_id": aid, "aria_customer_id": conductor,
                                      "aria_relationship_type": "AUTHORIZED_USER"})
            else:
                rel_rows.append({"aria_account_id": aid, "aria_customer_id": cid,
                                  "aria_relationship_type": "HOLDER"})
                if rng.random() < 0.15 and len(cust_list) > 1:
                    joint = rng.choice(cust_list)
                    if joint != cid:
                        rel_rows.append({"aria_account_id": aid, "aria_customer_id": joint,
                                          "aria_relationship_type": "HOLDER"})
                if rng.random() < 0.10 and len(cust_list) > 1:
                    bene = rng.choice(cust_list)
                    rel_rows.append({"aria_account_id": aid, "aria_customer_id": bene,
                                      "aria_relationship_type": "BENEFICIARY"})

            acct_id += 1

    return pd.DataFrame(acct_rows), pd.DataFrame(rel_rows)


def _baseline_transactions(cust_id, acct_id, open_date, rng, txn_id_start):
    """Generate 12 months of normal low-frequency transactions for a customer."""
    rows = []
    tid  = txn_id_start
    start = max(open_date, REF_DATE - timedelta(days=365))
    d = start
    while d < REF_DATE:
        if rng.random() < 0.15:   # ~15% chance of a transaction per day
            amount = max(10, int(rng.exponential(2_000)))
            direction = "CashOut" if rng.random() < 0.7 else "CashIn"
            txn_type = rng.choice(["Debit Card", "ACH", "Check", "Misc Deposit", "Misc Debit"])
            rows.append({
                "aria_transaction_id":   tid,
                "aria_timestamp":        _fmt_dt(d + timedelta(hours=int(rng.integers(8, 20)))),
                "aria_cash_direction":   direction,
                "aria_transaction_type": txn_type,
                "aria_amount":           amount,
                "aria_subject_id":       acct_id,
                "aria_customer_id":      cust_id,
                "aria_channel_id":       int(rng.choice([1, 2, 4])),
                "aria_currency_id":      1,
                "aria_country_id":       267,
                "aria_other_party_id":   None,
                "aria_is_anomalous":     False,
                "aria_rule_trigger":     None,
            })
            tid += 1
        d += timedelta(days=1)
    return rows, tid


# ── Rule-specific transaction injectors ──────────────────────────────────────
#
# Each injector receives a LIST of (acct_id, rel_type) tuples representing all
# monitored accounts for the customer. Transactions are spread across accounts
# so that no single account breaches the threshold — only the customer-level
# aggregate triggers the rule. This reflects how real AML engines work.
#
# Helper: build a transaction row
def _txn(tid, dt, direction, txn_type, amount, cust_id, acct_id,
         rule_trigger, channel=1, country_id=267, other_party=None):
    return {
        "aria_transaction_id":   tid,
        "aria_timestamp":        _fmt_dt(dt),
        "aria_cash_direction":   direction,
        "aria_transaction_type": txn_type,
        "aria_amount":           amount,
        "aria_subject_id":       acct_id,
        "aria_customer_id":      cust_id,
        "aria_channel_id":       channel,
        "aria_currency_id":      1,
        "aria_country_id":       country_id,
        "aria_other_party_id":   other_party,
        "aria_is_anomalous":     True,
        "aria_rule_trigger":     rule_trigger,
    }


def _split_across_accounts(total_amt, acct_ids, rng):
    """Split a total amount across N accounts with random proportions."""
    if len(acct_ids) == 1:
        return {acct_ids[0]: total_amt}
    weights = rng.dirichlet(np.ones(len(acct_ids)))
    return {aid: round(total_amt * w) for aid, w in zip(acct_ids, weights)}


def _inject_activity_deviation(cust_id, acct_ids, txn_type, rng, tid,
                                profile_mean=10_000, z_multiplier=6):
    """
    Monthly spike >> profile mean across ALL monitored accounts.
    Each account alone may look normal; the customer-level monthly aggregate
    is >> threshold.
    """
    rows = []
    spike_date  = REF_DATE - timedelta(days=int(rng.integers(30, 90)))
    spike_total = profile_mean * z_multiplier * rng.uniform(1.0, 2.0)
    splits      = _split_across_accounts(spike_total, acct_ids, rng)
    for aid, acct_total in splits.items():
        n_txns = int(rng.integers(2, 5))
        per_txn = acct_total / n_txns
        for i in range(n_txns):
            d = spike_date + timedelta(days=i * 3)
            rows.append(_txn(tid, d, "CashOut", txn_type,
                             round(per_txn * rng.uniform(0.8, 1.2)),
                             cust_id, aid, f"Activity Deviation ({txn_type})"))
            tid += 1
    return rows, tid, round(spike_total)


def _inject_elder_abuse(cust_id, acct_ids, rng, tid):
    """
    14-day outgoing burst spread across accounts — customer-level aggregate
    exceeds floor and z-threshold vs 90-day profile.
    """
    rows = []
    burst_start = REF_DATE - timedelta(days=int(rng.integers(14, 60)))
    target_14d  = rng.uniform(5_000, 80_000)
    splits      = _split_across_accounts(target_14d, acct_ids, rng)
    for aid, acct_total in splits.items():
        n_days = int(rng.integers(3, 14))
        per_day = acct_total / n_days
        for i in range(n_days):
            d = burst_start + timedelta(days=i)
            rows.append(_txn(tid, d, "CashOut",
                             rng.choice(["Check", "Wire", "ACH"]),
                             round(per_day * rng.uniform(0.5, 1.5)),
                             cust_id, aid, "Elder Abuse"))
            tid += 1
    return rows, tid, round(target_14d)


def _inject_velocity_single(cust_id, acct_ids, rng, tid):
    """
    Pass-through: CashIn on one account, CashOut on another (or same),
    out ~90-110% of in, within 14 days. Customer-level in+out pair triggers.
    """
    rows = []
    in_date  = REF_DATE - timedelta(days=int(rng.integers(14, 60)))
    out_date = in_date + timedelta(days=int(rng.integers(1, 14)))
    in_amt   = round(rng.uniform(20_000, 200_000))
    ratio    = rng.uniform(0.90, 1.10)
    out_amt  = round(in_amt * ratio)
    # CashIn on first account, CashOut on second (if available) — cross-account pass-through
    in_acct  = acct_ids[0]
    out_acct = acct_ids[1] if len(acct_ids) > 1 else acct_ids[0]
    rows.append(_txn(tid, in_date,  "CashIn",  "Wire", in_amt,  cust_id, in_acct,  "Velocity Single"))
    tid += 1
    rows.append(_txn(tid, out_date, "CashOut", "Wire", out_amt, cust_id, out_acct, "Velocity Single"))
    tid += 1
    return rows, tid, in_amt + out_amt, ratio


def _inject_velocity_multiple(cust_id, acct_ids, rng, tid):
    """
    Fan-out: one large CashIn, multiple CashOuts to distinct counterparties
    spread across accounts within 14 days.
    """
    rows = []
    in_date = REF_DATE - timedelta(days=int(rng.integers(14, 60)))
    in_amt  = round(rng.uniform(30_000, 300_000))
    in_acct = acct_ids[0]
    rows.append(_txn(tid, in_date, "CashIn", "Wire", in_amt, cust_id, in_acct, "Velocity Multiple"))
    tid += 1
    n_outs    = int(rng.integers(3, 8))
    remaining = in_amt
    for i in range(n_outs):
        out_date = in_date + timedelta(days=int(rng.integers(1, 14)))
        out_amt  = round(remaining / (n_outs - i) * rng.uniform(0.8, 1.1))
        out_amt  = min(out_amt, remaining)
        remaining -= out_amt
        out_acct = rng.choice(acct_ids)
        rows.append(_txn(tid, out_date, "CashOut", "Wire", out_amt, cust_id, out_acct,
                         "Velocity Multiple", other_party=tid * 1000 + i))
        tid += 1
    return rows, tid, in_amt


def _inject_detect_excessive(cust_id, acct_ids, rng, tid):
    """
    5-day burst of incoming Cash+Check spread across accounts.
    Customer-level 5-day sum > $10K threshold.
    """
    rows = []
    burst_start = REF_DATE - timedelta(days=int(rng.integers(5, 60)))
    target_5d   = rng.uniform(10_001, 60_000)
    splits      = _split_across_accounts(target_5d, acct_ids, rng)
    for aid, acct_total in splits.items():
        per_day = acct_total / 5
        for i in range(5):
            d = burst_start + timedelta(days=i)
            rows.append(_txn(tid, d, "CashIn",
                             rng.choice(["Cash", "Check"]),
                             round(per_day * rng.uniform(0.8, 1.2)),
                             cust_id, aid, "Detect Excessive Transaction Activity"))
            tid += 1
    return rows, tid, round(target_5d)


def _inject_structuring(cust_id, acct_ids, direction_flag, rng, tid):
    """
    Repeated cash amounts each below $10K across accounts — individually
    innocuous but customer-level pattern shows deliberate structuring.
    """
    rows  = []
    rf    = f"Structuring ({'Incoming' if direction_flag == 'CashIn' else 'Outgoing'} Cash)"
    start = REF_DATE - timedelta(days=int(rng.integers(14, 45)))
    n_days = int(rng.integers(3, 7))
    for i in range(n_days):
        d    = start + timedelta(days=i * int(rng.integers(1, 4)))
        # Each day: split a sub-$10K amount across accounts — each acct gets < $5K
        day_target = rng.uniform(8_000, 9_900)
        splits = _split_across_accounts(day_target, acct_ids, rng)
        for aid, amt in splits.items():
            rows.append(_txn(tid, d, direction_flag, "Cash", round(amt),
                             cust_id, aid, rf))
            tid += 1
    return rows, tid, n_days


def _inject_ctr(cust_id, acct_ids, rng, tid):
    """
    Cash transaction(s) that aggregate > $10K across customer accounts on same day.
    Each account may be < $10K; customer total crosses CTR threshold.
    """
    rows = []
    d         = REF_DATE - timedelta(days=int(rng.integers(1, 30)))
    direction = rng.choice(["CashIn", "CashOut"])
    target    = round(rng.uniform(10_001, 50_000))
    splits    = _split_across_accounts(target, acct_ids, rng)
    for aid, amt in splits.items():
        rows.append(_txn(tid, d, direction, "Cash", round(amt), cust_id, aid, "CTR Client"))
        tid += 1
    return rows, tid, target


def _inject_burst(cust_id, acct_ids, direction_flag, rng, tid):
    """
    5-day burst of Wire/ACH from/to multiple distinct counterparties,
    spread across customer accounts. Customer-level sum + counterparty count triggers.
    """
    rows  = []
    rf    = "Burst in Originator Activity" if direction_flag == "CashIn" else "Burst in Beneficiary Activity"
    start = REF_DATE - timedelta(days=int(rng.integers(5, 60)))
    n_txns = int(rng.integers(3, 8))
    total  = 0
    for i in range(n_txns):
        d    = start + timedelta(days=i)
        amt  = round(rng.uniform(1_000, 50_000))
        total += amt
        aid  = rng.choice(acct_ids)
        rows.append(_txn(tid, d, direction_flag, rng.choice(["Wire", "ACH"]),
                         amt, cust_id, aid, rf, other_party=tid * 100 + i))
        tid += 1
    return rows, tid, total, n_txns


def _inject_risky_international(cust_id, acct_ids, rng, tid):
    """
    Large wire(s) to/from high/medium risk country across customer accounts.
    Customer-level aggregate triggers the rule.
    """
    rows    = []
    country = rng.choice(COUNTRIES_HIGH_RISK + COUNTRIES_MED_RISK)
    d       = REF_DATE - timedelta(days=int(rng.integers(1, 60)))
    target  = round(rng.uniform(300_000, 2_000_000))
    splits  = _split_across_accounts(target, acct_ids, rng)
    direction = rng.choice(["CashIn", "CashOut"])
    for aid, amt in splits.items():
        row = _txn(tid, d, direction, "Wire", round(amt), cust_id, aid,
                   "Risky International Transfer", country_id=None)
        row["aria_counterparty_country"] = country
        rows.append(row)
        tid += 1
    return rows, tid, target


def _inject_funnel(cust_id, acct_ids, rng, tid):
    """
    Many small inflows from distinct counterparties across accounts,
    followed by one large outflow — customer-level funnel pattern.
    """
    rows     = []
    start    = REF_DATE - timedelta(days=int(rng.integers(30, 90)))
    n_in     = int(rng.integers(5, 15))
    total_in = 0
    for i in range(n_in):
        d   = start + timedelta(days=i * 2)
        amt = round(rng.uniform(500, 9_000))
        total_in += amt
        aid = rng.choice(acct_ids)
        rows.append(_txn(tid, d, "CashIn", rng.choice(["ACH", "Wire", "Cash"]),
                         amt, cust_id, aid, "Funnel Account",
                         other_party=tid * 50 + i))
        tid += 1
    # Single large outflow — concentrated on one account
    out_acct = acct_ids[0]
    out_date = start + timedelta(days=n_in * 2 + 1)
    rows.append(_txn(tid, out_date, "CashOut", "Wire",
                     round(total_in * rng.uniform(0.85, 0.98)),
                     cust_id, out_acct, "Funnel Account", country_id=None))
    tid += 1
    return rows, tid, total_in


def _inject_roundtrip(cust_id, acct_ids, rng, tid):
    """
    Funds leave on one account and return (net ~same amount) on another —
    cross-account round-trip only visible at customer level.
    """
    rows  = []
    start = REF_DATE - timedelta(days=int(rng.integers(30, 90)))
    amt   = round(rng.uniform(50_000, 500_000))
    out_acct = acct_ids[0]
    in_acct  = acct_ids[1] if len(acct_ids) > 1 else acct_ids[0]
    rows.append(_txn(tid, start, "CashOut", "Wire",
                     round(amt * rng.uniform(0.95, 1.05)),
                     cust_id, out_acct, "Round-trip", country_id=None))
    tid += 1
    return_date = start + timedelta(days=int(rng.integers(7, 30)))
    rows.append(_txn(tid, return_date, "CashIn", "Wire",
                     round(amt * rng.uniform(0.95, 1.05)),
                     cust_id, in_acct, "Round-trip", country_id=None))
    tid += 1
    return rows, tid, amt


def _inject_human_trafficking(cust_id, acct_ids, rng, tid):
    """
    Frequent small cash deposits across accounts + hotel/transport payments.
    Customer-level pattern: daily cash inflows + hospitality spend.
    """
    rows   = []
    start  = REF_DATE - timedelta(days=int(rng.integers(30, 90)))
    n_days = int(rng.integers(14, 30))
    total  = 0
    for i in range(n_days):
        d   = start + timedelta(days=i)
        amt = round(rng.uniform(200, 1_500))
        total += amt
        aid = rng.choice(acct_ids)
        rows.append(_txn(tid, d, "CashIn", "Cash", amt, cust_id, aid,
                         "Human Trafficking Indicators"))
        tid += 1
        if rng.random() < 0.5:
            hotel_acct = rng.choice(acct_ids)
            rows.append(_txn(tid, d + timedelta(hours=6),
                             "CashOut", "Debit Card",
                             round(rng.uniform(50, 300)),
                             cust_id, hotel_acct,
                             "Human Trafficking Indicators", channel=2))
            tid += 1
    return rows, tid, total


# ── Relationship type definitions ─────────────────────────────────────────────

# All relationship types that give a customer access to or benefit from account funds.
# Used to determine which accounts are in scope for AML monitoring per customer.
MONITORED_REL_TYPES = {
    "HOLDER",           # Primary account owner
    "AUTHORIZED_USER",  # Authorized to transact (business conductors, card users)
    "SIGNERS",          # Can sign on the account
    "POA",              # Power of Attorney — can transact on behalf of another
    "CUSTODIAN",        # Controls assets for minor/incapacitated person
    "TRUSTEE",          # Controls trust assets
    "TRUSTOR",          # Funded the trust — source of funds
    "BENEFICIARY",      # Receives proceeds
    "BORROWER",         # Loan account — monitor for loan-back schemes
    "GUARANTOR",        # Co-signed — financial exposure
    "OWNER",            # Business entity owns the account
    "CONSERVATOR",      # Court-appointed controller
    "PARENT",           # Parent entity relationship
}

# Rule → preferred relationship types for account selection (in priority order).
# The injector picks the first account where the customer has one of these relationships.
RULE_REL_PREFERENCE = {
    # Personal activity rules — prefer accounts customer directly controls
    "Elder Abuse":                          ["HOLDER", "POA", "CUSTODIAN", "TRUSTEE", "SIGNERS"],
    "Velocity Single":                      ["HOLDER", "SIGNERS", "POA", "AUTHORIZED_USER"],
    "Velocity Multiple":                    ["HOLDER", "SIGNERS", "POA", "AUTHORIZED_USER"],
    "Detect Excessive Transaction Activity":["HOLDER", "SIGNERS", "AUTHORIZED_USER"],
    "Structuring (Incoming Cash)":          ["HOLDER", "SIGNERS", "AUTHORIZED_USER", "OWNER"],
    "Structuring (Outgoing Cash)":          ["HOLDER", "SIGNERS", "AUTHORIZED_USER", "OWNER"],
    "CTR Client":                           ["HOLDER", "SIGNERS", "AUTHORIZED_USER", "OWNER"],
    "Human Trafficking Indicators":         ["HOLDER", "SIGNERS"],
    "Round-trip":                           ["HOLDER", "TRUSTEE", "TRUSTOR", "SIGNERS"],
    # Business/wire-heavy rules — prefer business account relationships
    "Activity Deviation (ACH)":             ["OWNER", "AUTHORIZED_USER", "SIGNERS", "HOLDER"],
    "Activity Deviation (Check)":           ["OWNER", "AUTHORIZED_USER", "SIGNERS", "HOLDER"],
    "Activity Deviation (Wire)":            ["OWNER", "AUTHORIZED_USER", "SIGNERS", "HOLDER"],
    "Burst in Originator Activity":         ["OWNER", "AUTHORIZED_USER", "SIGNERS", "HOLDER"],
    "Burst in Beneficiary Activity":        ["OWNER", "AUTHORIZED_USER", "SIGNERS", "HOLDER"],
    "Risky International Transfer":         ["OWNER", "AUTHORIZED_USER", "HOLDER", "TRUSTEE"],
    "Funnel Account":                       ["OWNER", "AUTHORIZED_USER", "HOLDER", "SIGNERS"],
}

# Default fallback if rule not in map
_DEFAULT_REL_PREF = ["HOLDER", "OWNER", "AUTHORIZED_USER", "SIGNERS", "POA",
                     "CUSTODIAN", "TRUSTEE", "BENEFICIARY", "BORROWER", "GUARANTOR"]


def _select_account(cid, rule, rel_by_cust, acct_type_map, rng):
    """
    Pick the most appropriate account for a customer to inject rule transactions on.

    Strategy:
      1. Get all accounts the customer has a monitored relationship with
      2. Filter to preferred relationship types for this rule (in priority order)
      3. Among matching accounts, prefer Checking/Savings over Loan/CD
      4. Fall back through preference tiers until an account is found
    """
    cust_rels = rel_by_cust.get(cid, [])  # list of (acct_id, rel_type)
    if not cust_rels:
        return None, None

    pref_types = RULE_REL_PREFERENCE.get(rule, _DEFAULT_REL_PREF)
    liquid_types = {"Checking", "Savings", "Foreign Currency", "Money Market Savings"}

    # Build lookup: rel_type -> list of acct_ids
    by_rel = {}
    for acct_id, rel_type in cust_rels:
        if rel_type in MONITORED_REL_TYPES:
            by_rel.setdefault(rel_type, []).append(acct_id)

    # Walk preference list — pick liquid account first within each tier
    for rel_type in pref_types:
        candidates = by_rel.get(rel_type, [])
        if not candidates:
            continue
        # Prefer liquid accounts (Checking/Savings)
        liquid = [a for a in candidates if acct_type_map.get(a) in liquid_types]
        pool   = liquid if liquid else candidates
        chosen = rng.choice(pool)
        return chosen, rel_type

    # Last resort: any monitored account
    all_accts = [a for rel_type, accts in by_rel.items() for a in accts]
    if all_accts:
        chosen = rng.choice(all_accts)
        return chosen, "any"

    return None, None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(n_customers, seed, out_dir):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating synthetic AML data — {n_customers} customers, seed={seed}")

    # Split customers: 80% Individual, 20% Business
    n_indiv = int(n_customers * 0.80)
    n_biz   = n_customers - n_indiv

    print(f"  Customers: {n_indiv} Individual, {n_biz} Business")
    customers_df = generate_customers(n_indiv, n_biz, rng)
    accounts_df, relationships_df = generate_accounts(customers_df, rng)
    print(f"  Accounts: {len(accounts_df)}  |  Relationships: {len(relationships_df)}")

    # Build lookup structures
    cust_info    = customers_df.set_index("aria_customer_id").to_dict("index")
    acct_type_map = accounts_df.set_index("aria_account_id")["aria_account_type"].to_dict()

    # customer → list of (acct_id, rel_type) for all monitored relationships
    rel_by_cust = {}
    for _, row in relationships_df.iterrows():
        cid = row["aria_customer_id"]
        if row["aria_relationship_type"] in MONITORED_REL_TYPES:
            rel_by_cust.setdefault(cid, []).append(
                (row["aria_account_id"], row["aria_relationship_type"])
            )

    all_txns   = []
    all_alerts = []
    txn_id     = 1
    alert_id   = 1

    # Baseline transactions — generated per account, attributed to the primary
    # relationship holder (HOLDER > OWNER > AUTHORIZED_USER in priority)
    print("  Generating baseline transactions …")
    primary_rel_priority = ["HOLDER", "OWNER", "AUTHORIZED_USER", "BORROWER",
                            "SIGNERS", "POA", "TRUSTEE", "BENEFICIARY", "GUARANTOR"]
    for _, acct in accounts_df.iterrows():
        acct_rels = relationships_df[
            relationships_df["aria_account_id"] == acct["aria_account_id"]
        ]
        if acct_rels.empty:
            continue
        # Pick primary customer for baseline generation (highest-priority rel type)
        cid = None
        for preferred_rel in primary_rel_priority:
            match = acct_rels[acct_rels["aria_relationship_type"] == preferred_rel]
            if not match.empty:
                cid = match.iloc[0]["aria_customer_id"]
                break
        if cid is None:
            cid = acct_rels.iloc[0]["aria_customer_id"]

        open_date = datetime.strptime(acct["aria_open_date"], "%Y-%m-%d %H:%M:%S")
        baseline, txn_id = _baseline_transactions(cid, acct["aria_account_id"], open_date, rng, txn_id)
        all_txns.extend(baseline)

    # Alerted customers — inject rule-specific transactions on the right account
    print("  Injecting rule-triggered transactions …")
    all_cust_ids = customers_df["aria_customer_id"].tolist()
    elder_custs  = [
        c for c in all_cust_ids
        if cust_info[c].get("aria_age") is not None and cust_info[c]["aria_age"] >= 60
    ]

    sar_rows = {}  # cust_id -> is_sar

    def _sar_prob(cid, base_rate):
        """
        Risk-adjust the base SAR rate for a customer using their risk flags and
        country tier. Higher-risk customers are significantly more likely to result
        in a SAR filing — this matches real AML analyst behaviour.

        Multipliers (cumulative, capped at 0.95):
          OFAC hit        : 5.0x  — near-certain SAR
          Subpoena        : 3.5x  — legal obligation to file
          Sanctions hit   : 3.0x  — EU/UN/OFAC combined match
          PEP             : 2.0x  — enhanced scrutiny
          High-risk country: 2.0x — jurisdiction risk
          Negative news   : 1.5x  — adverse media
          Med-risk country: 1.3x  — elevated jurisdiction risk
        """
        info = cust_info.get(cid, {})
        mult = 1.0
        if info.get("aria_ofac"):            mult *= 5.0
        if info.get("aria_subpoena"):        mult *= 3.5
        if info.get("aria_sanctions_hit"):   mult *= 3.0
        if info.get("aria_pep"):             mult *= 2.0
        if info.get("aria_negative_news"):   mult *= 1.5
        country = info.get("aria_citizenship", "")
        if country in COUNTRIES_HIGH_RISK:   mult *= 2.0
        elif country in COUNTRIES_MED_RISK:  mult *= 1.3
        return min(base_rate * mult, 0.95)

    for rule, n_alerted in ALERTED_PER_RULE.items():
        sar_rate = SAR_RATE_BY_RULE[rule]
        cond_id  = _cond_id(rule)

        # Pool restriction: Elder Abuse only applies to age >= 60
        pool = elder_custs if rule == "Elder Abuse" else all_cust_ids
        n    = min(n_alerted, len(pool))
        alerted = rng.choice(pool, size=n, replace=False).tolist()

        skipped = 0
        for cid in alerted:
            # Select the primary account for alert attribution (highest-priority rel type)
            primary_acct_id, rel_used = _select_account(cid, rule, rel_by_cust, acct_type_map, rng)
            if primary_acct_id is None:
                skipped += 1
                continue

            # All monitored accounts for this customer — injectors spread across them
            cust_accts = [aid for aid, _ in rel_by_cust.get(cid, [])
                          if _ in MONITORED_REL_TYPES]
            if not cust_accts:
                cust_accts = [primary_acct_id]
            # Deduplicate while preserving order (primary account first)
            seen = set()
            deduped = []
            for a in ([primary_acct_id] + cust_accts):
                if a not in seen:
                    seen.add(a)
                    deduped.append(a)
            cust_accts = deduped

            is_sar = bool(rng.random() < _sar_prob(cid, sar_rate))
            sar_rows[cid] = is_sar

            injected_txns = []
            alert_meta    = {}

            if rule in ("Activity Deviation (ACH)", "Activity Deviation (Check)", "Activity Deviation (Wire)"):
                ttype = {"Activity Deviation (ACH)": "ACH",
                         "Activity Deviation (Check)": "Check",
                         "Activity Deviation (Wire)": "Wire"}[rule]
                profile_mean = float(rng.uniform(5_000, 30_000))
                injected_txns, txn_id, trigger_amt = _inject_activity_deviation(
                    cid, cust_accts, ttype, rng, txn_id, profile_mean)
                alert_meta = {"trigger_amt": trigger_amt, "z_score": round(rng.uniform(5, 15), 2),
                              "profile_mean": profile_mean}

            elif rule == "Elder Abuse":
                injected_txns, txn_id, trigger_amt = _inject_elder_abuse(cid, cust_accts, rng, txn_id)
                age = cust_info[cid].get("aria_age", 65)
                alert_meta = {"trigger_amt": trigger_amt, "age": age,
                              "z_score": round(rng.uniform(3, 10), 2)}

            elif rule == "Velocity Single":
                injected_txns, txn_id, pair_total, ratio = _inject_velocity_single(cid, cust_accts, rng, txn_id)
                alert_meta = {"pair_total": pair_total, "ratio": round(ratio, 4)}

            elif rule == "Velocity Multiple":
                injected_txns, txn_id, in_amt = _inject_velocity_multiple(cid, cust_accts, rng, txn_id)
                alert_meta = {"pair_total": in_amt, "n_counterparties": len(injected_txns) - 1}

            elif rule == "Detect Excessive Transaction Activity":
                injected_txns, txn_id, trigger_amt = _inject_detect_excessive(cid, cust_accts, rng, txn_id)
                alert_meta = {"trigger_amt": trigger_amt}

            elif rule == "Structuring (Incoming Cash)":
                injected_txns, txn_id, n_days = _inject_structuring(cid, cust_accts, "CashIn", rng, txn_id)
                alert_meta = {"trigger_amt": round(rng.uniform(3_000, 9_900)),
                              "days_observed": n_days}

            elif rule == "Structuring (Outgoing Cash)":
                injected_txns, txn_id, n_days = _inject_structuring(cid, cust_accts, "CashOut", rng, txn_id)
                alert_meta = {"trigger_amt": round(rng.uniform(7_000, 9_900)),
                              "days_observed": n_days}

            elif rule == "CTR Client":
                injected_txns, txn_id, trigger_amt = _inject_ctr(cid, cust_accts, rng, txn_id)
                alert_meta = {"trigger_amt": trigger_amt}

            elif rule == "Burst in Originator Activity":
                injected_txns, txn_id, total, n_txns = _inject_burst(cid, cust_accts, "CashIn", rng, txn_id)
                alert_meta = {"trigger_amt": total, "txn_count": n_txns}

            elif rule == "Burst in Beneficiary Activity":
                injected_txns, txn_id, total, n_txns = _inject_burst(cid, cust_accts, "CashOut", rng, txn_id)
                alert_meta = {"trigger_amt": total, "txn_count": n_txns}

            elif rule == "Risky International Transfer":
                injected_txns, txn_id, trigger_amt = _inject_risky_international(cid, cust_accts, rng, txn_id)
                alert_meta = {"trigger_amt": trigger_amt}

            elif rule == "Funnel Account":
                injected_txns, txn_id, total_in = _inject_funnel(cid, cust_accts, rng, txn_id)
                alert_meta = {"trigger_amt": total_in}

            elif rule == "Round-trip":
                injected_txns, txn_id, amt = _inject_roundtrip(cid, cust_accts, rng, txn_id)
                alert_meta = {"trigger_amt": amt}

            elif rule == "Human Trafficking Indicators":
                injected_txns, txn_id, total = _inject_human_trafficking(cid, cust_accts, rng, txn_id)
                alert_meta = {"trigger_amt": total}

            all_txns.extend(injected_txns)

            # Alert record (primary_acct_id = highest-priority account for attribution)
            created = REF_DATE - timedelta(days=int(rng.integers(1, 30)))
            alert_rec = {
                "aria_alert_id":        _alert_id(alert_id),
                "aria_customer_id":     cid,
                "aria_account_id":      primary_acct_id,
                "aria_relationship_type": rel_used,   # which relationship triggered monitoring
                "aria_risk_factor":     rule,
                "aria_condition_id":    cond_id,
                "aria_created_on":      _fmt_dt(created),
                "aria_state":           "OPEN",
                "aria_is_sar":          int(is_sar),
                "aria_customer_type":   cust_info[cid]["aria_customer_type"],
            }
            alert_rec.update(alert_meta)
            all_alerts.append(alert_rec)
            alert_id += 1

        if skipped:
            print(f"    [{rule}] skipped {skipped} customers with no eligible account")

    # Compute aria_peak_activity_period per account from actual transaction timestamps
    print("  Computing peak activity periods …")
    txns_df = pd.DataFrame(all_txns)
    if not txns_df.empty:
        txns_df["_hour"] = pd.to_datetime(txns_df["aria_timestamp"]).dt.hour
        def _period(h):
            if  6 <= h < 12: return "Morning"
            if 12 <= h < 16: return "Afternoon"
            if 16 <= h < 20: return "Evening"
            return "Night"
        txns_df["_period"] = txns_df["_hour"].map(_period)
        # Most common period per account
        peak = (txns_df.groupby("aria_subject_id")["_period"]
                .agg(lambda s: s.value_counts().idxmax())
                .rename("aria_peak_activity_period"))
        accounts_df = accounts_df.set_index("aria_account_id")
        accounts_df["aria_peak_activity_period"] = peak
        accounts_df["aria_peak_activity_period"] = accounts_df["aria_peak_activity_period"].fillna("Morning")
        accounts_df = accounts_df.reset_index()
    else:
        accounts_df["aria_peak_activity_period"] = "Morning"

    # Save
    print("  Saving files …")
    customers_df.to_csv(os.path.join(out_dir, "aria_customers.csv"), index=False)
    accounts_df.to_csv(os.path.join(out_dir, "aria_accounts.csv"), index=False)
    relationships_df.to_csv(os.path.join(out_dir, "aria_account_relationships.csv"), index=False)

    txns_df = txns_df.drop(columns=["_hour", "_period"], errors="ignore")
    # aria_counterparty_country is only populated for Risky International Transfer rows
    if "aria_counterparty_country" not in txns_df.columns:
        txns_df["aria_counterparty_country"] = None
    txns_df.to_csv(os.path.join(out_dir, "aria_transactions.csv"), index=False)

    alerts_df = pd.DataFrame(all_alerts)
    alerts_df.to_csv(os.path.join(out_dir, "aria_alerts.csv"), index=False)

    print(f"\nDone. Output in {out_dir}/")
    print(f"  aria_customers.csv               : {len(customers_df):>6,} rows")
    print(f"  aria_accounts.csv                : {len(accounts_df):>6,} rows")
    print(f"  aria_account_relationships.csv   : {len(relationships_df):>6,} rows")
    print(f"  aria_transactions.csv            : {len(txns_df):>6,} rows")
    print(f"  aria_alerts.csv                  : {len(alerts_df):>6,} rows")

    print("\nAlert summary by rule:")
    for rf, grp in alerts_df.groupby("aria_risk_factor"):
        n    = len(grp)
        sars = int(grp["aria_is_sar"].sum())
        fp   = n - sars
        print(f"  {rf:<42} | alerted={n:>4} | SAR={sars:>3} | FP={fp:>4} | precision={100*sars/n:.1f}%")

    print("\nRelationship type distribution in alerts:")
    for rtype, grp in alerts_df.groupby("aria_relationship_type"):
        print(f"  {rtype:<20} : {len(grp):>4} alerts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic AML data")
    parser.add_argument("--customers", type=int, default=5000,
                        help="Total number of synthetic customers (default 5000)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--out",       default="aria_synth",
                        help="Output directory (default aria_synth)")
    args = parser.parse_args()
    main(args.customers, args.seed, args.out)
