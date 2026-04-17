"""
build_ofac_db.py — Parse OFAC SDN list and save as clean CSV + SQLite.

Downloads from:
  https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV
  https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ALT.CSV

Outputs:
  data/ofac_sdn.csv      — cleaned SDN entries with program group + risk fields
  data/ofac_sdn.db       — SQLite for fast querying in the app
"""

import csv, re, os, sqlite3, subprocess, sys

SDN_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV"
ALT_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ALT.CSV"
SDN_RAW = "C:/tmp/sdn.csv"
ALT_RAW = "C:/tmp/sdn_alt.csv"
OUT_CSV = "data/ofac_sdn.csv"
OUT_DB  = "data/ofac_sdn.db"

os.makedirs("data", exist_ok=True)

# ── 1. Download if not cached ──────────────────────────────────────────────────
def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        print(f"[cache] {dest}")
        return
    print(f"[download] {url}")
    subprocess.run(["curl", "-s", "--max-time", "60", "-k", "-L", url, "-o", dest],
                   check=True)

download(SDN_URL, SDN_RAW)
download(ALT_URL, ALT_RAW)

# ── 2. Parse alt names: ent_num → [aliases] ───────────────────────────────────
aliases = {}  # ent_num → list of alias names
with open(ALT_RAW, encoding="utf-8", errors="replace") as f:
    for row in csv.reader(f):
        if len(row) >= 4:
            ent = row[0].strip()
            name = row[3].strip()
            if name and name != "-0-":
                aliases.setdefault(ent, []).append(name)

# ── 3. Sanctions program → country / risk group mapping ───────────────────────
PROGRAM_MAP = {
    # Comprehensive embargoes (highest risk — blanket prohibition)
    "IRAN":             ("Iran",        "Comprehensive"),
    "IRAN-EO13846":     ("Iran",        "Comprehensive"),
    "IRAN-EO13902":     ("Iran",        "Comprehensive"),
    "IRAN-HR":          ("Iran",        "Comprehensive"),
    "IFSR":             ("Iran",        "Comprehensive"),
    "NPWMD":            ("Iran",        "Comprehensive"),   # WMD proliferators (mostly Iran/DPRK)
    "DPRK":             ("North Korea", "Comprehensive"),
    "DPRK2":            ("North Korea", "Comprehensive"),
    "DPRK3":            ("North Korea", "Comprehensive"),
    "DPRK4":            ("North Korea", "Comprehensive"),
    "CUBA":             ("Cuba",        "Comprehensive"),
    "SYRIA":            ("Syria",       "Comprehensive"),
    # Targeted / sectoral sanctions
    "RUSSIA-EO14024":   ("Russia",      "Targeted"),
    "UKRAINE-EO13662":  ("Russia",      "Targeted"),
    "UKRAINE-EO13660":  ("Ukraine",     "Targeted"),
    "BELARUS-EO14038":  ("Belarus",     "Targeted"),
    "BURMA-EO14014":    ("Myanmar",     "Targeted"),
    "VENEZUELA-EO13850":("Venezuela",   "Targeted"),
    "VENEZUELA":        ("Venezuela",   "Targeted"),
    "IRAQ2":            ("Iraq",        "Targeted"),
    "BALKANS":          ("Balkans",     "Targeted"),
    "DRCONGO":          ("DRC",         "Targeted"),
    # Functional programs (not country-specific)
    "SDGT":             ("Global",      "Terrorism"),
    "SDNTK":            ("Global",      "Narcotics"),
    "SDNT":             ("Global",      "Narcotics"),
    "ILLICIT-DRUGS-EO14059": ("Global", "Narcotics"),
    "TCO":              ("Global",      "TCO"),
    "GLOMAG":           ("Global",      "Human Rights"),
    "CYBER2":           ("Global",      "Cyber"),
    "PAARSSR-EO13894":  ("Global",      "Counter-terrorism"),
    "IRGC":             ("Iran",        "Comprehensive"),
}

def classify_program(prog_raw):
    """Return (country, sanction_type) for a raw program string."""
    prog = prog_raw.strip().upper()
    for key, val in PROGRAM_MAP.items():
        if key in prog:
            return val
    return ("Other", "Other")

def extract_dob(remarks):
    m = re.search(r"DOB\s+(\d{1,2}\s+\w+\s+\d{4}|\d{4})", remarks, re.I)
    return m.group(1) if m else ""

def extract_pob(remarks):
    m = re.search(r"POB\s+([^;]+)", remarks, re.I)
    return m.group(1).strip() if m else ""

def extract_nationality(remarks):
    m = re.search(r"nationality\s+([A-Za-z ,]+?)(?:;|\.|$)", remarks, re.I)
    return m.group(1).strip() if m else ""

# ── 4. Parse SDN entries ───────────────────────────────────────────────────────
print("[parse] SDN list...")
records = []
with open(SDN_RAW, encoding="utf-8", errors="replace") as f:
    for row in csv.reader(f):
        if len(row) < 4:
            continue
        ent_num  = row[0].strip()
        name     = row[1].strip().strip('"')
        sdn_type = row[2].strip()   # individual / vessel / aircraft / -0-
        program  = row[3].strip()
        remarks  = row[-1].strip() if len(row) >= 12 else ""

        country, sanction_type = classify_program(program)

        records.append({
            "ent_num":        ent_num,
            "name":           name,
            "sdn_type":       sdn_type if sdn_type != "-0-" else "entity",
            "program_raw":    program,
            "country":        country,
            "sanction_type":  sanction_type,
            "dob":            extract_dob(remarks),
            "pob":            extract_pob(remarks),
            "nationality":    extract_nationality(remarks),
            "aliases":        "|".join(aliases.get(ent_num, [])),
            "remarks":        remarks,
        })

print(f"  Parsed {len(records)} entries")

# ── 5. Write CSV ───────────────────────────────────────────────────────────────
FIELDS = ["ent_num","name","sdn_type","program_raw","country","sanction_type",
          "dob","pob","nationality","aliases","remarks"]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for r in records:
        w.writerow({k: r[k] for k in FIELDS})

print(f"[wrote] {OUT_CSV}")

# ── 6. Write SQLite ────────────────────────────────────────────────────────────
if os.path.exists(OUT_DB):
    os.remove(OUT_DB)

con = sqlite3.connect(OUT_DB)
cur = con.cursor()
cur.execute("""
    CREATE TABLE sdn (
        ent_num       TEXT,
        name          TEXT,
        sdn_type      TEXT,
        program_raw   TEXT,
        country       TEXT,
        sanction_type TEXT,
        dob           TEXT,
        pob           TEXT,
        nationality   TEXT,
        aliases       TEXT,
        remarks       TEXT
    )
""")
cur.execute("CREATE INDEX idx_name    ON sdn(name)")
cur.execute("CREATE INDEX idx_country ON sdn(country)")
cur.execute("CREATE INDEX idx_type    ON sdn(sanction_type)")

cur.executemany(
    "INSERT INTO sdn VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    [(r["ent_num"], r["name"], r["sdn_type"], r["program_raw"],
      r["country"], r["sanction_type"], r["dob"], r["pob"],
      r["nationality"], r["aliases"], r["remarks"]) for r in records]
)

# ── Phonetic index for fuzzy name screening ────────────────────────────────────
print("[phonetic] Building Double Metaphone index...")
try:
    import jellyfish
    import re as _re
    _NOISE = {"AL", "EL", "BIN", "BEN", "ABU", "UL", "UR", "JR", "SR", "II", "III"}
    _STRIP = _re.compile(r"[^A-Za-z0-9 ]")

    def _tokens(name):
        clean = _STRIP.sub(" ", name).upper()
        return [t for t in clean.split() if len(t) > 1 and t not in _NOISE]

    def _codes(name):
        codes = set()
        for tok in _tokens(name):
            p, s = jellyfish.double_metaphone(tok)
            if p: codes.add(p)
            if s: codes.add(s)
        return codes

    cur.execute("""
        CREATE TABLE sdn_phonetic (
            ent_num TEXT,
            code    TEXT
        )
    """)
    cur.execute("CREATE INDEX idx_phonetic_code ON sdn_phonetic(code)")

    phonetic_rows = []
    for r in records:
        all_names = [r["name"]] + [a.strip() for a in r["aliases"].split("|") if a.strip()]
        seen_codes = set()
        for nm in all_names:
            for code in _codes(nm):
                if code not in seen_codes:
                    seen_codes.add(code)
                    phonetic_rows.append((r["ent_num"], code))

    cur.executemany("INSERT INTO sdn_phonetic VALUES (?,?)", phonetic_rows)
    print(f"  Phonetic rows: {len(phonetic_rows):,}")
except ImportError:
    print("  WARNING: jellyfish not installed — skipping phonetic index (pip install jellyfish)")

con.commit()
con.close()
print(f"[wrote] {OUT_DB}")

# ── 7. Summary ────────────────────────────────────────────────────────────────
import collections
by_country = collections.Counter(r["country"] for r in records)
by_type    = collections.Counter(r["sanction_type"] for r in records)
by_sdn     = collections.Counter(r["sdn_type"] for r in records)

print("\n--- SDN by country/program ---")
for k,v in by_country.most_common(20):
    print(f"  {k}: {v}")
print("\n--- SDN by sanction type ---")
for k,v in by_type.most_common():
    print(f"  {k}: {v}")
print("\n--- SDN by entity type ---")
for k,v in by_sdn.most_common():
    print(f"  {k}: {v}")
