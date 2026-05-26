"""
PRR_ROR.py — Adapter to run PRR and ROR on the FAERS flat parquet dataset.

Builds the contingency table expected by vigipy from the flat (report × drug × reaction)
parquet format and runs both disproportionality analyses.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vigipy.PRR import prr
from vigipy.ROR import ror
from vigipy.utils import Container


# ── 1. CONFIG ─────────────────────────────────────────────────────────────────

PARQUET_PATH = "data/faers_flat_deduped.parquet"   # or faers_flat.parquet

# drug_characterization == "1" → suspect drug (the one being evaluated)
# Set to None to include all roles (suspect + concomitant + interacting)
DRUG_ROLE_FILTER = "1"

# Column names in your parquet — adjust if yours differ
COL_REPORT   = "safetyreportid"
COL_DRUG     = "drug_name"
COL_REACTION = "reaction_pt"
COL_ROLE     = "drug_characterization"


# ── 2. LOAD & OPTIONALLY FILTER ───────────────────────────────────────────────

df = pd.read_parquet(PARQUET_PATH)

# Drop rows with missing drug or reaction — they can't contribute to any cell
df = df.dropna(subset=[COL_DRUG, COL_REACTION])

if DRUG_ROLE_FILTER is not None:
    df = df[df[COL_ROLE].astype(str) == str(DRUG_ROLE_FILTER)]

# Normalise to upper-case strings (your data already looks upper-case, but just in case)
df[COL_DRUG]     = df[COL_DRUG].str.strip().str.upper()
df[COL_REACTION] = df[COL_REACTION].str.strip().str.upper()


# ── 3. DEDUPLICATE AT REPORT LEVEL ───────────────────────────────────────────
# Each (report, drug, reaction) triplet should count once.
# If you loaded the deduped file this is mostly already done, but be safe.

df = df.drop_duplicates(subset=[COL_REPORT, COL_DRUG, COL_REACTION])


# ── 4. BUILD THE CONTINGENCY COUNTS ──────────────────────────────────────────

# N — total number of unique reports in the dataset
N = df[COL_REPORT].nunique()

# n11 — reports mentioning drug X AND reaction Y  (the co-occurrence count)
n11 = (
    df.groupby([COL_DRUG, COL_REACTION])[COL_REPORT]
    .nunique()
    .reset_index()
    .rename(columns={COL_REPORT: "events",
                     COL_DRUG:   "product_name",
                     COL_REACTION: "ae_name"})
)

# n1j — total unique reports mentioning drug X  (across any reaction)
drug_counts = (
    df.groupby(COL_DRUG)[COL_REPORT]
    .nunique()
    .reset_index()
    .rename(columns={COL_REPORT: "product_aes", COL_DRUG: "product_name"})
)

# ni1 — total unique reports mentioning reaction Y  (across any drug)
ae_counts = (
    df.groupby(COL_REACTION)[COL_REPORT]
    .nunique()
    .reset_index()
    .rename(columns={COL_REPORT: "count_across_brands", COL_REACTION: "ae_name"})
)

# Merge everything into one flat table — one row per (drug, reaction) pair
data = (
    n11
    .merge(drug_counts, on="product_name", how="left")
    .merge(ae_counts,   on="ae_name",      how="left")
)

data = data[data["product_aes"] >= 50]

print(f"Total unique reports (N): {N:,}")
print(f"Unique drug-reaction pairs: {len(data):,}")
print(data.head())


# ── 5. BUILD THE CONTAINER OBJECT ────────────────────────────────────────────

container = Container()
container.data = data   # DataFrame with: events, product_name, ae_name,
                        #                 product_aes, count_across_brands
container.N    = N


# ── 6. RUN PRR ────────────────────────────────────────────────────────────────

prr_results = prr(
    container,
    relative_risk=1,
    min_events=3,            # ignore pairs with fewer than x reports
    decision_metric="fdr",
    decision_thres=0.05,
    ranking_statistic="p_value",
    expected_method="mantel-haentzel",
)

print(f"\n=== PRR: {prr_results.num_signals} signals detected ===")
print(prr_results.signals.head(20).to_string(index=False))
prr_results.all_signals.to_csv("data/prr_results.csv", index=False)


# ── 7. RUN ROR ────────────────────────────────────────────────────────────────

ror_results = ror(
    container,
    relative_risk=1,
    min_events=3,            # ignore pairs with fewer than x reports
    decision_metric="fdr",
    decision_thres=0.05,
    ranking_statistic="p_value",
    expected_method="mantel-haentzel",
)

print(f"\n=== ROR: {ror_results.num_signals} signals detected ===")
print(ror_results.signals.head(20).to_string(index=False))
ror_results.all_signals.to_csv("data/ror_results.csv", index=False)