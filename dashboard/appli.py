"""
dashboard/appli.py

Disproportionality analysis pipeline on FAERS data.

Drug name resolution is delegated to src/match_drug.py:
    resolve_drug_name() — four-step cascade:
        0. Local brand → INN dictionary (offline)
        1. Exact match against Parquet drug index
        2. rapidfuzz fuzzy match (token_set_ratio ≥ 75)
        3. Mistral AI fallback → INN → fuzzy match

Age input: user types age in years → automatically mapped to age_stratum.

AE ranking: sorted by composite confidence score computed across 4 algorithms.
Icons per AE:
    ✅  green tick   — event already validated by openFDA (KNOWN)
    ⚠️  triangle     — Weber effect risk (green / orange / red)
"""

import os
import json
import time
import duckdb
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Pipeline imports
# run_signals.py and src/ are copied into /app by Dockerfile
# ---------------------------------------------------------------------------
from run_signals import run_pipeline
from match_drug import resolve_drug_name

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drug Safety Signal Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — clinical, minimal, precise
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:           #F4F6F9;
        --surface:      #FFFFFF;
        --surface-2:    #F0F3F8;
        --border:       #DDE3ED;
        --border-light: #EEF1F6;
        --text:         #111827;
        --text-2:       #4B5563;
        --text-3:       #9CA3AF;
        --accent:       #1D4ED8;
        --accent-light: #EFF6FF;
        --accent-mid:   #BFDBFE;
        --green:        #065F46;
        --green-bg:     #ECFDF5;
        --green-border: #A7F3D0;
        --orange:       #92400E;
        --orange-bg:    #FFFBEB;
        --orange-border:#FDE68A;
        --red:          #7F1D1D;
        --red-bg:       #FEF2F2;
        --red-border:   #FECACA;
        --radius:       8px;
        --radius-lg:    12px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp { background: var(--bg); }

    /* ── Page title ── */
    .page-title {
        font-size: 30px;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }
    .page-subtitle {
        font-size: 22px;
        color: var(--text-3);
        margin-bottom: 28px;
        font-weight: 400;
    }

    /* ── Section header ── */
    .section-header {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-3);
        margin: 28px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-light);
    }

    /* ── AE signal card ── */
    .signal-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 14px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 14px;
        transition: border-color 0.15s;
    }
    .signal-card:hover { border-color: var(--accent-mid); }

    /* Rank */
    .rank-badge {
        font-size: 11px;
        font-weight: 600;
        color: var(--text-3);
        font-family: 'JetBrains Mono', monospace;
        min-width: 26px;
        text-align: center;
    }

    /* AE name */
    .ae-name {
        flex: 1;
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
        line-height: 1.3;
    }

    /* Algo pills */
    .algo-pill {
        display: inline-block;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 2px 6px;
        border-radius: 4px;
        background: var(--accent-light);
        color: var(--accent);
        margin-right: 3px;
        margin-top: 4px;
    }

    /* Confidence bar */
    .conf-wrap { width: 100px; }
    .conf-track {
        background: var(--border-light);
        border-radius: 3px;
        height: 5px;
        width: 100%;
    }
    .conf-fill {
        height: 5px;
        border-radius: 3px;
    }
    .conf-label {
        font-size: 10px;
        color: var(--text-3);
        margin-top: 3px;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Icon with hover tooltip */
    .icon-wrap {
        position: relative;
        font-size: 17px;
        min-width: 28px;
        text-align: center;
        cursor: default;
        user-select: none;
        display: inline-block;
    }
    .icon-wrap .tip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        background: #1F2937;
        color: #F9FAFB;
        font-size: 11px;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        line-height: 1.45;
        padding: 6px 10px;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.18);
        z-index: 9999;
        white-space: normal;
        max-width: 180px;
        text-align: left;
        transition: opacity 0.15s;
        pointer-events: none;
    }
    .icon-wrap .tip::after {
        content: '';
        position: absolute;
        top: 100%; left: 50%;
        transform: translateX(-50%);
        border: 5px solid transparent;
        border-top-color: #1F2937;
    }
    .icon-wrap:hover .tip {
        visibility: visible;
        opacity: 1;
    }

    /* Chip for drug match */
    .match-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--accent-light);
        color: var(--accent);
        font-size: 12px;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 999px;
        margin-bottom: 8px;
        border: 1px solid var(--accent-mid);
    }

    /* Popup / tooltip boxes */
    .popup-known {
        background: var(--green-bg);
        border: 1px solid var(--green-border);
        border-radius: var(--radius);
        padding: 12px 16px;
        color: var(--green);
        font-size: 13px;
        line-height: 1.5;
    }
    .popup-new {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: var(--radius);
        padding: 12px 16px;
        color: #78350F;
        font-size: 13px;
        line-height: 1.5;
    }
    .popup-weber-low {
        background: var(--green-bg);
        border: 1px solid var(--green-border);
        border-radius: var(--radius);
        padding: 12px 16px;
        color: var(--green);
        font-size: 13px;
        line-height: 1.5;
    }
    .popup-weber-medium {
        background: var(--orange-bg);
        border: 1px solid var(--orange-border);
        border-radius: var(--radius);
        padding: 12px 16px;
        color: var(--orange);
        font-size: 13px;
        line-height: 1.5;
    }
    .popup-weber-high {
        background: var(--red-bg);
        border: 1px solid var(--red-border);
        border-radius: var(--radius);
        padding: 12px 16px;
        color: var(--red);
        font-size: 13px;
        line-height: 1.5;
    }

    /* Metric box */
    .metric-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 14px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: var(--text);
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 11px;
        color: var(--text-3);
        margin-top: 2px;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Run button override */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--accent) !important;
        border: none !important;
        border-radius: var(--radius) !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        padding: 10px 28px !important;
        font-size: 14px !important;
    }

    /* Info box */
    .info-box {
        background: var(--accent-light);
        border: 1px solid var(--accent-mid);
        border-radius: var(--radius);
        padding: 10px 14px;
        font-size: 10px
        color: var(--accent);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR              = Path(__file__).resolve().parent / "data"
PARQUET_PATH          = DATA_DIR / "faers_flat_deduped.parquet"
SIGNALS_FULL_PATH     = DATA_DIR / "signals_full.parquet"
DRUG_INDEX_CACHE_PATH = DATA_DIR / "drug_index_cache.json"

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
DEFAULT_ALGORITHMS = ["prr", "ror", "bcpnn", "mgps"]
DEFAULT_MIN_A  = 10      
DEFAULT_FDR    = 0.05   # ok così
DEFAULT_EB05   = 2.5    # era 2.0 — MGPS più conservativo
DEFAULT_IC     = 0.5    # era 0.0 — BCPNN richiede IC025 > 0.5

SEX_OPTIONS = ["All", "Female", "Male"]

WEBER_COLORS = {
    "LOW":    ("🟢", "popup-weber-low",    "Low Risk"),
    "MEDIUM": ("🟡", "popup-weber-medium", "Moderate Risk"),
    "HIGH":   ("🔴", "popup-weber-high",   "High Risk"),
}


# ============================================================================
# Drug index
# ============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_drug_index() -> list[str]:
    sorted_path = DATA_DIR / "faers_sorted.parquet"
    if sorted_path.exists():
        try:
            import pyarrow.parquet as pq
            pf      = pq.ParquetFile(str(sorted_path))
            meta    = pf.metadata
            col_idx = pf.schema_arrow.get_field_index("drug_name")
            names   = set()
            for i in range(meta.num_row_groups):
                rg   = meta.row_group(i)
                col  = rg.column(col_idx)
                stat = col.statistics
                if stat and stat.has_min_max:
                    names.add(stat.min)
                    names.add(stat.max)
            names.discard(None)
            names.discard("")
            if names:
                return sorted(names)
        except Exception:
            pass

    flat_path = DATA_DIR / "faers_flat_deduped.parquet"
    if flat_path.exists():
        try:
            con = duckdb.connect()
            con.execute("SET memory_limit = '1GB'")
            con.execute("SET threads = 2")
            result = con.execute(
                f"SELECT DISTINCT drug_name FROM '{flat_path}' "
                f"WHERE drug_name IS NOT NULL AND drug_name != ''"
            ).fetchall()
            con.close()
            names = [r[0] for r in result if r[0]]
            if names:
                return names
        except Exception:
            pass

    return []


# ============================================================================
# Age → stratum
# ============================================================================

def age_to_stratum(age_input: str) -> str | None:
    if not age_input or age_input.strip() == "":
        return None

    s = age_input.strip().lower()

    mapping = {
        "pediatric": "pediatric", "child": "pediatric", "children": "pediatric",
        "infant": "pediatric", "neonatal": "pediatric",
        "adult": "adult",
        "geriatric": "geriatric", "elderly": "geriatric",
        "all": None, "": None,
    }
    if s in mapping:
        return mapping[s]

    try:
        age = int(s)
        if age < 18:
            return "pediatric"
        elif age < 65:
            return "adult"
        else:
            return "geriatric"
    except ValueError:
        return None


# ============================================================================
# Where clause builder
# ============================================================================

def build_where_extra(
    sex: str,
    age_stratum: str | None,
    comedication_resolved: str | None,
) -> str | None:
    clauses = []

    if sex and sex.lower() not in ("all", ""):
        clauses.append(f"sex = '{sex.strip().lower()}'")

    if age_stratum:
        clauses.append(f"age_stratum = '{age_stratum}'")

    if comedication_resolved:
        clauses.append(
            f"""safetyreportid IN (
                SELECT DISTINCT safetyreportid
                FROM '{PARQUET_PATH}'
                WHERE drug_name = '{comedication_resolved}'
            )"""
        )

    return " AND ".join(clauses) if clauses else None


def count_unknown_age_excluded() -> int | None:
    try:
        con = duckdb.connect()
        result = con.execute(
            f"SELECT COUNT(DISTINCT safetyreportid) FROM '{PARQUET_PATH}'"
            f" WHERE age_stratum = 'unknown'"
        ).fetchone()
        return result[0] if result else None
    except Exception:
        return None


def count_total_reports() -> int | None:
    try:
        con = duckdb.connect()
        result = con.execute(
            f"SELECT COUNT(DISTINCT safetyreportid) FROM '{PARQUET_PATH}'"
        ).fetchone()
        return result[0] if result else None
    except Exception:
        return None


# ============================================================================
# Confidence score
# ============================================================================

def compute_confidence_score(
    validated_df: pd.DataFrame,
    algo_results: dict,
    ct: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if validated_df is None or len(validated_df) == 0:
        return pd.DataFrame()

    rr_index: dict[str, float] = {}
    if ct is not None and not ct.empty and {"pt", "a", "b"}.issubset(ct.columns):
        ct_rr = ct.copy()
        denom = (ct_rr["a"] + ct_rr["b"]).replace(0, np.nan)
        ct_rr["reporting_rate"] = ct_rr["a"] / denom
        rr_index = ct_rr.set_index("pt")["reporting_rate"].to_dict()

    ae_set = set(validated_df["ae_name"].unique())
    rows = []

    for ae in ae_set:
        algos_positive  = []
        signal_strengths = []

        if "prr" in algo_results and algo_results["prr"] is not None:
            df_prr = algo_results["prr"]
            row = df_prr[df_prr["ae_name"] == ae]
            if not row.empty and row.iloc[0].get("signal_positive", False):
                algos_positive.append("PRR")
                prr_val = row.iloc[0].get("PRR", 1)
                if prr_val > 0:
                    signal_strengths.append(np.log(max(prr_val, 1)))

        if "ror" in algo_results and algo_results["ror"] is not None:
            df_ror = algo_results["ror"]
            row = df_ror[df_ror["ae_name"] == ae]
            if not row.empty and row.iloc[0].get("signal_positive", False):
                algos_positive.append("ROR")
                ror_val = row.iloc[0].get("ROR", 1)
                if ror_val > 0:
                    signal_strengths.append(np.log(max(ror_val, 1)))

        if "bcpnn" in algo_results and algo_results["bcpnn"] is not None:
            df_bc = algo_results["bcpnn"]
            row = df_bc[df_bc["ae_name"] == ae]
            if not row.empty and row.iloc[0].get("signal_positive", False):
                algos_positive.append("BCPNN")
                ic_val = row.iloc[0].get("IC", 0)
                signal_strengths.append(max(ic_val, 0))

        if "mgps" in algo_results and algo_results["mgps"] is not None:
            df_mg = algo_results["mgps"]
            row = df_mg[df_mg["ae_name"] == ae]
            if not row.empty and row.iloc[0].get("signal_positive", False):
                algos_positive.append("MGPS")
                eb_val = row.iloc[0].get("EBGM", row.iloc[0].get("EB05", 0))
                signal_strengths.append(np.log(max(eb_val, 1)))

        n_algos       = len(algos_positive)
        avg_strength  = np.mean(signal_strengths) if signal_strengths else 0.0
        reporting_rate = rr_index.get(ae, 0.0) or 0.0

        val_row    = validated_df[validated_df["ae_name"] == ae]
        val_status = val_row.iloc[0].get("validation_status", "") if not val_row.empty else ""

        rows.append({
            "ae_name":             ae,
            "n_algorithms":        n_algos,
            "algorithms_positive": algos_positive,
            "avg_strength":        avg_strength,
            "reporting_rate":      reporting_rate,
            "validation_status":   val_status,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    max_strength = result["avg_strength"].max()
    result["strength_norm"] = result["avg_strength"] / max_strength if max_strength > 0 else 0.0

    max_rr = result["reporting_rate"].max()
    result["rr_norm"] = result["reporting_rate"] / max_rr if max_rr > 0 else 0.0

    result["confidence_score"] = (
        0.30 * (result["n_algorithms"] / 4.0) +
        0.30 * result["strength_norm"] +
        0.40 * result["rr_norm"]
    ) * 100

    result["confidence_score"] = result["confidence_score"].round(1)

    return result.sort_values("confidence_score", ascending=False).reset_index(drop=True)


# ============================================================================
# AE list renderer
# ============================================================================

def render_ae_list(
    ranked_df: pd.DataFrame,
    weber_risk: str | None,
    limit: int | None = None,
):
    if ranked_df.empty:
        st.info("No signals detected with the current parameters.")
        return

    display_df = ranked_df if limit is None else ranked_df.head(limit)

    weber_emoji, _, _ = WEBER_COLORS.get(
        (weber_risk or "").upper(),
        ("⬜", "", "N/A"),
    )

    # Tooltip texts
    WEBER_TIP = {
        "LOW":    "🟢 Low bias risk \u2014 stable reporting over time.",
        "MEDIUM": "🟡 Moderate bias risk \u2014 early-phase spike detected. Interpret with caution.",
        "HIGH":   "🔴 High bias risk \u2014 reports clustered post-approval. May reflect notoriety, not pharmacology.",
    }
    weber_tip = WEBER_TIP.get((weber_risk or "").upper(), "Weber check not run.")

    for idx, row in display_df.iterrows():
        ae       = row["ae_name"]
        score    = row["confidence_score"]
        algos    = row.get("algorithms_positive", [])
        val_st   = row.get("validation_status", "")
        is_known = val_st == "KNOWN"

        pills_html = "".join(
            f'<span class="algo-pill">{a}</span>' for a in algos
        )

        bar_color = (
            "#1D4ED8" if score >= 60
            else ("#D97706" if score >= 35 else "#DC2626")
        )
        bar_html = f"""
        <div class="conf-wrap">
          <div class="conf-track">
            <div class="conf-fill"
                 style="width:{score}%;background:{bar_color};"></div>
          </div>
          <div class="conf-label">{score}/100</div>
        </div>"""

        fda_icon = "✅" if is_known else "🔍"
        fda_tip  = (
            "✅ In FDA label \u2014 known adverse event."
            if is_known else
            "🔍 Not in FDA label \u2014 potentially new signal."
        )

        card_html = f"""
        <div class="signal-card">
          <div class="rank-badge">#{idx + 1}</div>
          <div style="flex:1">
            <div class="ae-name">{ae}</div>
            <div>{pills_html}</div>
          </div>
          {bar_html}
          <div class="icon-wrap">
            {fda_icon}
            <div class="tip">{fda_tip}</div>
          </div>
          <div class="icon-wrap">
            {weber_emoji}
            <div class="tip">{weber_tip}</div>
          </div>
        </div>"""

        st.markdown(card_html, unsafe_allow_html=True)


# ============================================================================
# Config builder
# ============================================================================

def build_config_from_ui(
    target_drug, where_extra, min_a, algorithms,
    fdr_threshold, eb05_threshold, ic_threshold,
    openfda_api_key, validate_label, check_weber, weber_approval_override,
) -> dict:
    return {
        "target_drug":             target_drug.strip().upper(),
        "where_extra":             where_extra,
        "min_a":                   min_a,
        "algorithms":              algorithms,
        "fdr_threshold":           fdr_threshold,
        "eb05_threshold":          eb05_threshold,
        "ic_threshold":            ic_threshold,
        "openfda_api_key":         openfda_api_key or None,
        "validate_label":          validate_label,
        "check_weber":             check_weber,
        "weber_approval_override": weber_approval_override,
    }


# ============================================================================
# Pipeline execution + display
# ============================================================================

def run_and_display(
    config: dict,
    age_stratum: str | None,
    ae_limit: int | None,
):
    if not config["target_drug"]:
        st.warning("Please enter a drug name.")
        return

    if age_stratum:
        n_unknown = count_unknown_age_excluded()
        n_total = count_total_reports()
        if n_unknown and n_total:
            st.markdown(
                f'<div class="info-box">Age filter active (<strong>{age_stratum}</strong>): '
                f'<strong>{n_total - n_unknown:,}</strong> reports included for analysis. </div>',
                unsafe_allow_html=True,
            )
            st.write("")

    # ── Progress bar setup ────────────────────────────────────────────────
    progress_bar  = st.progress(0, text="Initialising…")
    status_text   = st.empty()

    def _step(pct: int, msg: str):
        progress_bar.progress(pct, text=msg)
        status_text.caption(msg)

    # ── Contingency table cache ───────────────────────────────────────────
    ct_key = (
        config["target_drug"],
        config.get("where_extra") or "",
        config["min_a"],
    )
    cached_ct = None
    if st.session_state.get("_ct_key") == ct_key:
        cached_ct = st.session_state.get("_ct_cache")
        if cached_ct is not None:
            st.caption(
                f"Contingency table loaded from cache ({len(cached_ct)} pairs) "
                f"— only algorithms are re-run."
            )

    # ── Run pipeline with stepped progress ────────────────────────────────
    _step(5, "Building contingency table…" if cached_ct is None else "Loading cached contingency table…")

    try:
        _step(20, "Running disproportionality algorithms (PRR, ROR, BCPNN, MGPS)…")
        results = run_pipeline(config, precomputed_ct=cached_ct)
        _step(70, "Validating signals against FDA label…")
        time.sleep(0.05)   # brief pause so user can read the step
        _step(85, "Applying Weber effect check…")
        time.sleep(0.05)
        _step(95, "Computing confidence scores…")
        time.sleep(0.05)
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Pipeline error: {e}")
        return

    # ── Update CT cache ───────────────────────────────────────────────────
    if cached_ct is None and results.get("error") != "no_data":
        st.session_state["_ct_cache"] = results["contingency_table"]
        st.session_state["_ct_key"]   = ct_key

    _step(100, "Done.")
    time.sleep(0.3)
    progress_bar.empty()
    status_text.empty()

    if results.get("error") == "no_data":
        st.error(
            f"No drug–event pairs found for **{config['target_drug']}** "
            f"with the current filters."
        )
        return

    ct        = results["contingency_table"]
    validated = results.get("validated", pd.DataFrame())
    weber     = results.get("weber_check", {})
    summary   = results.get("run_summary", {})
    weber_risk = weber.get("weber_risk") or summary.get("weber_risk")

    algo_results = {k: results.get(k) for k in ["prr", "ror", "bcpnn", "mgps"]}

    # ── Run summary metrics ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Run Summary</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    def _metric(col, value, label):
        col.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _metric(m1, config["target_drug"], "Drug")
    _metric(m2, f"{len(ct):,}", "CT Pairs")
    _metric(m3, summary.get("total_elapsed_s", "—"), "Time (s)")
    _metric(m4, config["where_extra"] or "None", "Filter")

    if summary.get("algorithm_results"):
        st.write("")
        algo_rows = [
            {
                "Algorithm": a.upper(),
                "Positive signals": s["positive"],
                "Evaluated pairs": s["total_pairs"],
            }
            for a, s in summary["algorithm_results"].items()
        ]
        st.dataframe(
            pd.DataFrame(algo_rows),
            use_container_width=True,
            hide_index=True,
        )

    # ── AE list ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Detected Adverse Events</div>', unsafe_allow_html=True)

    ranked = compute_confidence_score(validated, algo_results, ct=ct)

    if not ranked.empty:
        render_ae_list(ranked, weber_risk=weber_risk, limit=ae_limit)
    else:
        st.info("No positive signals detected with the current parameters.")

    # ── Expanders ─────────────────────────────────────────────────────────
    with st.expander("📋 Full validated signals table"):
        if validated is not None and len(validated) > 0:
            st.dataframe(validated, use_container_width=True)
            st.download_button(
                "Download signals (CSV)",
                data=validated.to_csv(index=False).encode("utf-8"),
                file_name=f"{config['target_drug']}_signals.csv",
                mime="text/csv",
            )
        else:
            st.info("No validated signals.")

    with st.expander("⏱ Weber effect — detail"):
        if weber:
            st.json(weber)
        else:
            st.info("Weber check disabled or no data available.")

    with st.expander("🔢 Contingency table"):
        st.dataframe(ct, use_container_width=True)
        st.download_button(
            "Download CT (CSV)",
            data=ct.to_csv(index=False).encode("utf-8"),
            file_name=f"{config['target_drug']}_ct.csv",
            mime="text/csv",
        )


# ============================================================================
# PAGE HEADER
# ============================================================================

st.markdown('<div class="page-title">Drug Safety Signal Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Disproportionality analysis on FDA FAERS data</div>',
    unsafe_allow_html=True,
)

# ============================================================================
# INPUT PARAMETERS
# ============================================================================

st.markdown('<div class="section-header">Insert Parameters</div>', unsafe_allow_html=True)

drug_index = load_drug_index()

col1, col2, col3, col4 = st.columns(4)

with col1:
    drug_input_raw = st.selectbox(
        "Drug",
        options=drug_index if drug_index else [],
        index=None,
        placeholder="Type a drug name…",
        accept_new_options=True,
        key="drug_input",
    )

with col2:
    sex_input = st.selectbox(
        "Sex",
        options=SEX_OPTIONS,
        index=0,
        key="sex_input",
        label_visibility="visible",
    )

with col3:
    age_input_raw = st.text_input(
        "Age",
        placeholder="Insert age",
        key="age_input",
    )

with col4:
    comed_input_raw = st.selectbox(
        "Co-medication",
        options=drug_index if drug_index else [],
        index=None,
        placeholder="Optional…",
        accept_new_options=True,
        key="comedication_input",
    )

# ── Advanced parameters ───────────────────────────────────────────────────

with st.expander("Advanced parameters"):
    adv1, adv2 = st.columns(2)

    with adv1:
        algorithms = st.multiselect(
            "Algorithms",
            ["prr", "ror", "bcpnn", "mgps"],
            default=DEFAULT_ALGORITHMS,
            key="algorithms_input",
        )
        min_a = st.number_input(
            "Min occurrences (a)",
            min_value=1,
            value=DEFAULT_MIN_A,
            step=1,
            key="min_a_input",
        )
        fdr_threshold = st.number_input(
            "FDR threshold (PRR/ROR)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_FDR,
            step=0.01,
            key="fdr_input",
        )

    with adv2:
        eb05_threshold = st.number_input(
            "EB05 threshold (MGPS)",
            min_value=0.0,
            value=DEFAULT_EB05,
            step=0.1,
            key="eb05_input",
        )
        ic_threshold = st.number_input(
            "IC025 threshold (BCPNN)",
            value=DEFAULT_IC,
            step=0.1,
            key="ic_input",
        )
        weber_override = st.text_input(
            "Approval year override (Weber)",
            placeholder="e.g. 2007",
            key="weber_override_input",
        )
        mistral_key = st.text_input(
            "Mistral API key (drug name fallback)",
            type="password",
            key="mistral_key_input",
        )

    openfda_api_key    = st.text_input(
        "openFDA API key (optional)",
        type="password",
        key="api_key_input",
    )
    validate_label_cb = st.checkbox("FDA label validation", value=True, key="validate_input")
    check_weber_cb    = st.checkbox("Weber effect check",   value=True, key="weber_check_input")

# ── AE display limit ───────────────────────────────────────────────────────

ae_limit_map   = {"Top 5": 5, "Top 10": 10, "Top 20": 20, "All": None}
ae_limit_label = st.radio(
    "Show AEs",
    options=list(ae_limit_map.keys()),
    index=1,
    horizontal=True,
    key="ae_limit_input",
)
ae_limit = ae_limit_map[ae_limit_label]

# ============================================================================
# RUN BUTTON
# ============================================================================

st.write("")
if st.button("▶  Run analysis", type="primary"):

    if not drug_input_raw:
        st.warning("Please enter a drug name.")
        st.stop()

    # ── Resolve drug name ────────────────────────────────────────────────
    resolved_drug, drug_method, drug_score, drug_err = resolve_drug_name(
        drug_input_raw, drug_index, mistral_key or None
    )

    if drug_err:
        st.warning(f"Drug name resolution: {drug_err}")

    if drug_method == "fuzzy":
        st.markdown(
            f'<span class="match-chip"> Found corresponding ACTIVE SUBSTANCE "{drug_input_raw}" → '
            f'<strong>{resolved_drug}</strong> '
            f'(fuzzy match · {drug_score:.0f}/100)</span>',
            unsafe_allow_html=True,
        )
    elif drug_method == "mistral":
        st.markdown(
            f'<span class="match-chip"> Found corresponding ACTIVE SUBSTANCE: "{drug_input_raw}" → '
            f'<strong>{resolved_drug}</strong> (via Mistral AI)</span>',
            unsafe_allow_html=True,
        )
    elif drug_method == "passthrough":
        st.warning(
            f"No reliable match found for '{drug_input_raw}'. "
            f"Proceeding with the raw input — if no pairs are found, "
            f"try the English INN or active substance name."
        )

    # ── Resolve co-medication ────────────────────────────────────────────
    resolved_comed = None
    if comed_input_raw and str(comed_input_raw).strip():
        resolved_comed, comed_method, comed_score, comed_err = resolve_drug_name(
            str(comed_input_raw), drug_index, mistral_key or None
        )
        if comed_err:
            st.warning(f"Co-medication resolution: {comed_err}")
        if comed_method == "fuzzy" and comed_score < 100:
            st.markdown(
                f'<span class="match-chip">Found Co-med ACTIVE SUBSTANCE: "{comed_input_raw} " → '
                f'<strong>{resolved_comed}</strong> ({comed_score:.0f}/100)</span>',
                unsafe_allow_html=True,
            )
        elif comed_method == "mistral":
            st.markdown(
                f'<span class="match-chip"> Found Co-med ACTIVE SUBSTANCE: "{comed_input_raw}" → '
                f'<strong>{resolved_comed}</strong> (via Mistral AI)</span>',
                unsafe_allow_html=True,
            )
        elif comed_method == "passthrough":
            st.warning(
                f"No reliable match found for co-medication '{comed_input_raw}'. "
                f"Co-medication filter will not be applied."
            )
            resolved_comed = None

    # ── Resolve age ──────────────────────────────────────────────────────
    age_stratum = age_to_stratum(age_input_raw)
    if age_input_raw.strip() and age_stratum:
        st.markdown(
            f'<span class="match-chip">Age "{age_input_raw}" → '
            f'<strong>{age_stratum}</strong></span>',
            unsafe_allow_html=True,
        )
    elif age_input_raw.strip() and not age_stratum:
        st.warning(
            f"Age input '{age_input_raw}' not recognised —> no age filter applied."
        )

    # ── Build where clause ───────────────────────────────────────────────
    where_extra = build_where_extra(sex_input, age_stratum, resolved_comed)

    # ── Build config ─────────────────────────────────────────────────────
    weber_override_int = (
        int(weber_override.strip())
        if weber_override.strip().isdigit()
        else None
    )

    config = build_config_from_ui(
        target_drug=resolved_drug,
        where_extra=where_extra,
        min_a=min_a,
        algorithms=algorithms,
        fdr_threshold=fdr_threshold,
        eb05_threshold=eb05_threshold,
        ic_threshold=ic_threshold,
        openfda_api_key=openfda_api_key or None,
        validate_label=validate_label_cb,
        check_weber=check_weber_cb,
        weber_approval_override=weber_override_int,
    )

    run_and_display(config, age_stratum=age_stratum, ae_limit=ae_limit)