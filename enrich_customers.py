"""
Enrich ss_files_anon/aml_s_customers.csv with realistic values:

1. Risk flags (all customers):  exempt_cdd, exempt_sanctions, sic_code, soc_code,
   pep, subpeona, is_314b, negative_news, ofac
   → at least 5% of customers get ≥1 flag; distribute across flags; 2-3 flags allowed

2. residency_country_id → 15 country IDs, ~85% US, 2 FinCEN high-risk countries

3. gross_annual_income → individuals only (entity=False), realistic log-normal dist

4. gender → fill NULLs for individuals (keep existing A/B/C)

5. marital_status → individuals only

6. occupation → individuals only, normalise to 15 canonical occupations, fill NULLs
"""

import csv, random, math, os
from copy import deepcopy

random.seed(42)

SRC = r"C:\Users\Aaditya\PycharmProjects\framlagents-ftllm\ss_files_anon\aml_s_customers.csv"
DST = r"C:\Users\Aaditya\PycharmProjects\framlagents-ftllm\ss_files_anon\aml_s_customers.csv"

# ── 1. load ────────────────────────────────────────────────────────────────────
with open(SRC, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

total = len(rows)
print(f"Loaded {total} customers")

# ── helpers ────────────────────────────────────────────────────────────────────
def is_individual(r):
    return r["entity"].strip().lower() == "false"

def is_business(r):
    return r["entity"].strip().lower() == "true"

# ── 2. Risk flags ──────────────────────────────────────────────────────────────
# 15 high-risk SIC codes (money services, gambling, pawn, jewelry, crypto, bars)
HIGH_RISK_SIC = [
    "5813",   # Bars / Taverns
    "6099",   # Financial Services NEC (check cashing, money transmission)
    "7993",   # Video Gambling / Amusement
    "5094",   # Jewelry, Watches, Precious Metals
    "7011",   # Hotels & Motels
    "5511",   # Auto Dealers (new cars, cash-intensive)
    "6153",   # Short-term business credit / pawn shops
    "5912",   # Drug Stores / Pharmacies
    "4812",   # Telephone Communications
    "7389",   # Services to Buildings / misc business services
]

# High-risk SOC codes (financial sales, real-estate, general managers)
HIGH_RISK_SOC = [
    "41-3031",   # Securities / commodity sales agents
    "13-2099",   # Financial specialists NEC
    "11-1021",   # General Managers
    "41-9099",   # Sales / related workers NEC
    "13-1041",   # Compliance Officers
]

# Boolean risk flags and their approximate prevalence (OFAC handled separately)
BOOL_FLAGS = [
    ("pep",               0.018),   # politically exposed person
    ("negative_news",     0.030),   # negative news hit
    ("subpeona",          0.008),   # received subpoena
    ("is_314b",           0.006),   # 314b information sharing flag
    ("exempt_cdd",        0.010),   # CDD exemption
    ("exempt_sanctions",  0.005),   # sanctions exemption
]

# OFAC: comprehensive-sanction countries → 100% flagged
#        targeted-sanction countries     → % chance flagged
# Source: OFAC SDN list (18,732 entries, build_ofac_db.py)
OFAC_COMPREHENSIVE = {"Iran", "North Korea", "Cuba", "Syria"}
OFAC_TARGETED_RATE = {
    "Myanmar":   0.25,   # BURMA-EO14014 — targeted, not blanket
    "Russia":    0.10,   # RUSSIA-EO14024 — targeted
    "Belarus":   0.15,   # BELARUS-EO14038
    "Venezuela": 0.08,
}

# Mark customers for risk flags; target overall ≥10% have at least one flag
flagged_indices = set()

# Assign each bool flag independently
for flag, rate in BOOL_FLAGS:
    n = max(1, int(total * rate))
    candidates = random.sample(range(total), n)
    for i in candidates:
        rows[i][flag] = "True"
        flagged_indices.add(i)

# NOTE: OFAC flag is set AFTER nationality is assigned (step 3 below).
# We store a placeholder here so the column exists; step 3b sets it properly.

# sic_code: assign to ~5% of businesses + ~2% of individuals
biz_idx = [i for i, r in enumerate(rows) if is_business(r)]
ind_idx  = [i for i, r in enumerate(rows) if is_individual(r)]

for i in random.sample(biz_idx, max(1, int(len(biz_idx) * 0.05))):
    rows[i]["sic_code"] = random.choice(HIGH_RISK_SIC)
    flagged_indices.add(i)

for i in random.sample(ind_idx, max(1, int(len(ind_idx) * 0.02))):
    rows[i]["sic_code"] = random.choice(HIGH_RISK_SIC)
    flagged_indices.add(i)

# soc_code: assign to ~3% of individuals
for i in random.sample(ind_idx, max(1, int(len(ind_idx) * 0.03))):
    rows[i]["soc_code"] = random.choice(HIGH_RISK_SOC)
    flagged_indices.add(i)

pct_flagged = len(flagged_indices) / total * 100
print(f"Flagged customers: {len(flagged_indices)} ({pct_flagged:.1f}%)")

# ── 3. residency, nationality, citizenship ────────────────────────────────────
#
# AML rationale:
#   residency  = where the customer lives and banks → overwhelmingly US for a US bank;
#                a small slice for legitimate expats (visa holders, green card holders).
#                NO FATF blacklist/greylist countries — those customers cannot open
#                a US bank account with an Iranian or North Korean address.
#
#   nationality = country of origin / passport held → this is the actual AML risk signal.
#                 A customer can live in the US (residency=US) but hold an Iranian or
#                 North Korean passport. Banks screen this combination specifically.
#
#   citizenship  = same concept as nationality, stored as a readable string.
#                  Populated to match nationality_country_id.
#
# Country ID scheme:
#   - Existing system IDs for countries already in the source data (267=USA, 104=Myanmar…)
#   - ISO 3166-1 numeric codes for countries not yet in the internal range
#     (364=Iran, 408=North Korea, 643=Russia, 804=Ukraine, 760=Syria, 332=Haiti)
#
# FATF risk classification (Feb 2026):
#   Blacklist:          Iran (364), North Korea (408), Myanmar (104)
#   Greylist:           Syria (760), Haiti (332), Cameroon (120)
#   FinCEN primary:     Iran, North Korea

# (id, res_weight, nat_weight, name)
#   res_weight  → residency pool   (US-heavy, no blacklist/greylist)
#   nat_weight  → nationality pool (more diverse, includes high-risk)
COUNTRIES = [
    (267,  9300,  7200, "United States"),
    (156,   130,   320, "China"),
    (50,    100,   260, "Bangladesh"),
    (152,    90,   210, "Chile"),
    (170,    80,   190, "Colombia"),
    (108,    70,   160, "Burundi"),
    (231,    60,   160, "Ethiopia"),
    (643,    50,   180, "Russia"),
    (804,    50,   170, "Ukraine"),
    (104,     0,   160, "Myanmar"),        # FATF Blacklist
    (364,     0,   130, "Iran"),           # FATF Blacklist / FinCEN Primary Concern
    (120,     0,   110, "Cameroon"),       # FATF Greylist
    (332,     0,    90, "Haiti"),          # FATF Greylist
    (408,     0,    60, "North Korea"),    # FATF Blacklist / FinCEN Primary Concern
    (760,     0,    60, "Syria"),          # FATF Greylist
]

COUNTRY_NAME = {cid: name for cid, _, _, name in COUNTRIES}

res_pool = []
nat_pool = []
for cid, rw, nw, _ in COUNTRIES:
    res_pool.extend([cid] * rw)
    nat_pool.extend([cid] * nw)

for r in rows:
    # residency — US-centric, no high-risk jurisdictions
    rcid = random.choice(res_pool)
    r["residency_country_id"] = str(rcid)
    r["residency_country_name"] = COUNTRY_NAME[rcid]

    # nationality — diverse, includes FATF blacklist/greylist
    ncid = random.choice(nat_pool)
    r["nationality_country_id"] = str(ncid)

    # citizenship — readable string matching nationality
    r["citizenship"] = COUNTRY_NAME[ncid]

# Insert residency_country_name into fieldnames after residency_country_id
fieldnames = list(fieldnames)
if "residency_country_name" not in fieldnames:
    idx = fieldnames.index("residency_country_id")
    fieldnames.insert(idx + 1, "residency_country_name")

print("residency, nationality, citizenship assigned.")

# ── 3b. OFAC flag — systematic by citizenship ─────────────────────────────────
# Comprehensive-sanction countries: all customers flagged (Iran, DPRK, Cuba, Syria)
# Targeted-sanction countries: probabilistic rate
ofac_hit = 0
for r in rows:
    ctz = r.get("citizenship", "")
    if ctz in OFAC_COMPREHENSIVE:
        r["ofac"] = "True"
        ofac_hit += 1
    elif ctz in OFAC_TARGETED_RATE:
        if random.random() < OFAC_TARGETED_RATE[ctz]:
            r["ofac"] = "True"
            ofac_hit += 1
    # else: leave as previously set by random flag assignment

print(f"OFAC flags set: {ofac_hit} systematic + prior random = "
      f"{sum(1 for r in rows if r['ofac']=='True')} total")

# ── 4. gross_annual_income (individuals only) ──────────────────────────────────
# Log-normal: median ~$55k, most between $25k-$200k, tail to $750k
def sample_income():
    # lognormal with mu=log(55000), sigma=0.7
    raw = random.lognormvariate(math.log(55000), 0.7)
    return max(12000, min(750000, int(round(raw / 1000) * 1000)))

for r in rows:
    if is_individual(r):
        r["gross_annual_income"] = str(sample_income())

print("gross_annual_income assigned for individuals.")

# ── 5. gender (individuals only, fill NULLs) ──────────────────────────────────
# Existing codes: A, B, C — keep as-is; fill NULL with A or B (50/50)
for r in rows:
    if is_individual(r) and r["gender"].strip().upper() in ("NULL", ""):
        r["gender"] = random.choice(["A", "B"])

print("gender NULLs filled for individuals.")

# ── 6. marital_status (individuals only) ──────────────────────────────────────
MARITAL = [
    ("Single",    35),
    ("Married",   45),
    ("Divorced",  12),
    ("Widowed",    5),
    ("Separated",  3),
]
marital_pool = []
for status, weight in MARITAL:
    marital_pool.extend([status] * weight)

for r in rows:
    if is_individual(r):
        r["marital_status"] = random.choice(marital_pool)

print("marital_status assigned for individuals.")

# ── 7. occupation (individuals only) ──────────────────────────────────────────
# 15 canonical occupations with realistic distribution weights
OCCUPATIONS = [
    ("Student",               12),
    ("Engineer",              10),
    ("Teacher",                8),
    ("Homemaker",              8),
    ("Manager",                7),
    ("Software Engineer",      6),
    ("Accountant",             5),
    ("Sales Representative",   5),
    ("Registered Nurse",       5),
    ("Retired",                7),
    ("Real Estate Agent",      4),
    ("Attorney",               3),
    ("Construction Worker",    6),
    ("Financial Analyst",      3),
    ("Business Owner",         5),
    # Remainder - catch-all for existing misc values already in the data
]

# Normalisation map: collapse existing varied cases to canonical
NORM_MAP = {
    # Student variants
    "student": "Student", "STUDENT": "Student",
    # Engineer variants
    "engineer": "Engineer", "ENGINEER": "Engineer",
    "software engineer": "Software Engineer", "SOFTWARE ENGINEER": "Software Engineer",
    # Teacher variants
    "teacher": "Teacher", "TEACHER": "Teacher",
    # Homemaker variants
    "homemaker": "Homemaker", "HOMEMAKER": "Homemaker",
    # Manager variants
    "manager": "Manager", "MANAGER": "Manager",
    # Accountant
    "accountant": "Accountant", "ACCOUNTANT": "Accountant",
    # Nurse
    "rn": "Registered Nurse", "RN": "Registered Nurse",
    # Retired
    "retired": "Retired", "RETIRED": "Retired",
    # Sales
    "sales": "Sales Representative", "SALES": "Sales Representative",
    # Owner → Business Owner
    "owner": "Business Owner", "OWNER": "Business Owner",
    # Lawyer → Attorney
    "lawyer": "Attorney", "LAWYER": "Attorney",
}

occ_pool = []
for occ, weight in OCCUPATIONS:
    occ_pool.extend([occ] * weight)

for r in rows:
    if is_individual(r):
        existing = r["occupation"].strip()
        if existing in NORM_MAP:
            r["occupation"] = NORM_MAP[existing]
        elif existing.upper() == "NULL" or existing == "":
            r["occupation"] = random.choice(occ_pool)
        # else keep as-is (other specific occupations already in data)

print("occupation normalised and NULLs filled for individuals.")

# ── 8. write output ────────────────────────────────────────────────────────────
with open(DST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[done] Written {len(rows)} rows to {DST}")

# ── 9. summary stats ──────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────")
for flag, _ in BOOL_FLAGS:
    ct = sum(1 for r in rows if r[flag].strip().lower() == "true")
    print(f"  {flag}: {ct} ({ct/total*100:.1f}%)")

ct_sic = sum(1 for r in rows if r["sic_code"] not in ("NULL",""))
ct_soc = sum(1 for r in rows if r["soc_code"] not in ("NULL",""))
print(f"  sic_code set: {ct_sic} ({ct_sic/total*100:.1f}%)")
print(f"  soc_code set: {ct_soc} ({ct_soc/total*100:.1f}%)")

res_dist = {}
for r in rows:
    v = r["residency_country_id"]
    res_dist[v] = res_dist.get(v, 0) + 1
print(f"\n  Residency countries: {len(res_dist)}")
for cid, cnt in sorted(res_dist.items(), key=lambda x: -x[1]):
    tag = " ← FinCEN high-risk" if cid in ("98","156") else ""
    print(f"    country_id={cid}: {cnt} ({cnt/total*100:.1f}%){tag}")

inc_vals = [int(r["gross_annual_income"]) for r in rows
            if is_individual(r) and r["gross_annual_income"] not in ("NULL","")]
if inc_vals:
    print(f"\n  Income (individuals): min={min(inc_vals):,}  "
          f"median={sorted(inc_vals)[len(inc_vals)//2]:,}  max={max(inc_vals):,}")

occ_dist = {}
for r in rows:
    if is_individual(r):
        occ_dist[r["occupation"]] = occ_dist.get(r["occupation"],0)+1
top_occ = sorted(occ_dist.items(), key=lambda x:-x[1])[:16]
print(f"\n  Top occupations (individuals):")
for o, c in top_occ:
    print(f"    {o}: {c}")
