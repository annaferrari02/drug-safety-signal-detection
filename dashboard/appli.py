"""
dashboard/appli.py

Pipeline di disproportionality analysis su dati FAERS.

Flusso risoluzione nome farmaco:
    1. rapidfuzz — match fuzzy contro i drug_name nel Parquet
    2. fallback Mistral AI — se score < soglia, chiede il principio attivo in inglese

Age input: l'utente scrive l'età in anni → automaticamente mappata a age_stratum.

Ranking AE: ordinato per "confidence score" composito calcolato sui 4 algoritmi.
Icone per ogni AE:
    ✅ tick verde  — evento già validato da openFDA (KNOWN)
    ⚠️ triangolo   — Weber effect risk (verde/arancione/rosso)
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
# Import pipeline — run_signals.py viene copiato in /app dal Dockerfile
# ---------------------------------------------------------------------------
from run_signals import run_pipeline

# ---------------------------------------------------------------------------
# Configurazione pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drug Safety Signal Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS personalizzato — palette clinica, pulita
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Palette */
    :root {
        --bg: #F7F9FC;
        --card: #FFFFFF;
        --border: #E2E8F0;
        --text: #1A202C;
        --muted: #718096;
        --accent: #2B6CB0;
        --accent-light: #EBF4FF;
        --green: #276749;
        --green-bg: #F0FFF4;
        --orange: #C05621;
        --orange-bg: #FFFAF0;
        --red: #9B2335;
        --red-bg: #FFF5F5;
    }

    .stApp { background: var(--bg); }

    /* Card generica */
    .signal-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    /* Rank badge */
    .rank-badge {
        font-size: 11px;
        font-weight: 700;
        color: var(--muted);
        min-width: 28px;
        text-align: center;
    }

    /* Nome AE */
    .ae-name {
        flex: 1;
        font-size: 14px;
        font-weight: 600;
        color: var(--text);
    }

    /* Confidence bar */
    .conf-bar-wrap {
        width: 120px;
    }
    .conf-bar-bg {
        background: var(--border);
        border-radius: 4px;
        height: 6px;
        width: 100%;
    }
    .conf-bar-fill {
        height: 6px;
        border-radius: 4px;
        background: var(--accent);
    }
    .conf-label {
        font-size: 10px;
        color: var(--muted);
        margin-top: 2px;
        text-align: right;
    }

    /* Icone */
    .icon-btn {
        font-size: 18px;
        cursor: pointer;
        text-decoration: none;
    }

    /* Pill algoritmi */
    .algo-pill {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 999px;
        background: var(--accent-light);
        color: var(--accent);
        margin-right: 3px;
    }

    /* Header sezione */
    .section-header {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 24px 0 10px 0;
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
    }

    /* Tooltip box */
    .tooltip-known {
        background: var(--green-bg);
        border: 1px solid #C6F6D5;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: var(--green);
        margin-bottom: 8px;
    }
    .tooltip-weber-low    { background: var(--green-bg);  border: 1px solid #C6F6D5; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: var(--green);  margin-bottom: 8px; }
    .tooltip-weber-medium { background: var(--orange-bg); border: 1px solid #FEEBC8; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: var(--orange); margin-bottom: 8px; }
    .tooltip-weber-high   { background: var(--red-bg);    border: 1px solid #FED7D7; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: var(--red);    margin-bottom: 8px; }

    /* Drug match chip */
    .match-chip {
        display: inline-block;
        background: #EBF4FF;
        color: #2B6CB0;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 999px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
DATA_DIR               = Path(__file__).resolve().parent / "data"
PARQUET_PATH           = DATA_DIR / "faers_flat_deduped.parquet"
SIGNALS_FULL_PATH      = DATA_DIR / "signals_full.parquet"
DRUG_INDEX_CACHE_PATH  = DATA_DIR / "drug_index_cache.json"

# ---------------------------------------------------------------------------
# Costanti UI
# ---------------------------------------------------------------------------
DEFAULT_ALGORITHMS  = ["prr", "ror", "bcpnn", "mgps"]
DEFAULT_MIN_A       = 3
DEFAULT_FDR         = 0.05
DEFAULT_EB05        = 2.0
DEFAULT_IC          = 0.0

SUGGESTED_DRUGS = [
    "LAPATINIB","CAPECITABINE","TRASTUZUMAB","PACLITAXEL","DOCETAXEL",
    "TAMOXIFEN","LETROZOLE","CISPLATIN","DOXORUBICIN","METFORMIN",
]
SEX_OPTIONS = ["All", "Female", "Male"]


# ============================================================================
# UTILITÀ: indice farmaci dal Parquet
# ============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_drug_index() -> list[str]:
    """
    Carica tutti i drug_name distinti dal Parquet.

    Usa faers_sorted.parquet se disponibile — DuckDB legge le statistiche
    per row group senza scansionare il file completo, evitando OOM.
    Fallback su faers_flat_deduped.parquet, poi su SUGGESTED_DRUGS.
    """
    # Preferisce faers_sorted.parquet: le statistiche per colonna permettono
    # a DuckDB di raccogliere i drug_name distinti leggendo solo i metadati
    # dei row group invece di scansionare tutto il file.
    # Strategia 1: legge i valori min/max per row group dai metadati di
    # faers_sorted.parquet — zero scan dei dati, solo lettura metadati.
    # Poiché il file è ordinato per drug_name, ogni row group ha min==max
    # per i row group mono-farmaco, e i valori min/max coprono tutti i farmaci.
    sorted_path = DATA_DIR / "faers_sorted.parquet"
    if sorted_path.exists():
        try:
            import pyarrow.parquet as pq
            pf       = pq.ParquetFile(str(sorted_path))
            meta     = pf.metadata
            col_idx  = pf.schema_arrow.get_field_index("drug_name")
            names    = set()
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

    # Strategia 2: DuckDB con memoria limitata su faers_flat_deduped
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

    return SUGGESTED_DRUGS


# ============================================================================
# RISOLUZIONE NOME FARMACO: rapidfuzz → fallback Mistral
# ============================================================================

def resolve_drug_name(user_input: str, drug_index: list[str], mistral_key: str | None = None) -> tuple[str, str, float]:
    """
    Risolve il nome farmaco inserito dall'utente al drug_name nel Parquet.

    Returns
    -------
    (resolved_name, method, score)
        method: "exact" | "fuzzy" | "mistral" | "passthrough"
        score: 0–100 (confidenza del match fuzzy; 100 per exact/mistral)
    """
    from rapidfuzz import process, fuzz

    query = user_input.strip().upper()

    # 1. Match esatto
    if query in drug_index:
        return query, "exact", 100.0

    # 2. rapidfuzz — token_set_ratio gestisce sinonimi parziali meglio di ratio semplice
    result = process.extractOne(query, drug_index, scorer=fuzz.token_set_ratio)
    if result:
        match, score, _ = result
        if score >= 75:
            return match, "fuzzy", float(score)

    # 3. Fallback Mistral — due step separati:
    #    a) Mistral identifica l'INN in inglese (es. "ketoprofene" → "KETOPROFEN")
    #       Prompt piccolo, nessuna lista, solo conoscenza farmacologica.
    #    b) rapidfuzz matcha l'INN sul drug_index reale del Parquet
    #       Garantisce che il nome restituito esista esattamente nel dataset.
    if mistral_key:
        try:
            from src.text_to_sql import resolve_drug_name_mistral
            inn = resolve_drug_name_mistral(user_input, api_key=mistral_key)
            if inn:
                # Step b: match dell'INN sul dataset reale
                result_b = process.extractOne(inn, drug_index, scorer=fuzz.token_set_ratio)
                if result_b and result_b[1] >= 75:
                    return result_b[0], "mistral", float(result_b[1])
        except Exception:
            pass

    # 4. Passthrough — nessun match affidabile, passa la query grezza
    return query, "passthrough", 0.0


def resolve_comedication(user_input: str, drug_index: list[str], mistral_key: str | None = None) -> tuple[str, str, float]:
    """Stesso flow di resolve_drug_name per la co-medicazione."""
    return resolve_drug_name(user_input, drug_index, mistral_key)


# ============================================================================
# MAPPING ETÀ → AGE STRATUM
# ============================================================================

def age_to_stratum(age_input: str) -> str | None:
    """
    Converte un input testuale età in age_stratum.
    Accetta: numero intero (anni), oppure testo libero ("adult", "pediatric", ...).
    """
    if not age_input or age_input.strip() == "":
        return None

    s = age_input.strip().lower()

    # Testo diretto
    mapping = {
        "pediatric": "pediatric", "child": "pediatric", "children": "pediatric",
        "infant": "pediatric", "neonatal": "pediatric", "bambino": "pediatric",
        "adult": "adult", "adulto": "adult",
        "geriatric": "geriatric", "elderly": "geriatric", "anziano": "geriatric",
        "all": None, "tutti": None, "": None,
    }
    if s in mapping:
        return mapping[s]

    # Numero di anni
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
# COSTRUZIONE WHERE EXTRA
# ============================================================================

def build_where_extra(sex: str, age_stratum: str | None, comedication_resolved: str | None) -> str | None:
    clauses = []

    if sex and sex.lower() not in ("all", "tutti", ""):
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
            f"SELECT COUNT(DISTINCT safetyreportid) FROM '{PARQUET_PATH}' WHERE age_stratum = 'unknown'"
        ).fetchone()
        return result[0] if result else None
    except Exception:
        return None


# ============================================================================
# CONFIDENCE SCORE — ranking degli AE
# ============================================================================

def compute_confidence_score(validated_df: pd.DataFrame, algo_results: dict) -> pd.DataFrame:
    """
    Calcola un confidence score composito per ogni AE rilevato come segnale positivo
    da almeno un algoritmo.

    Logica:
        1. Conteggio algoritmi concordanti  (0–4) — peso 40%
           Quanti dei 4 algoritmi classificano l'AE come signal_positive?

        2. Forza del segnale normalizzata    (0–1) — peso 60%
           Media delle metriche principali normalizzate:
               PRR  → log(PRR) / log(PRR_max)
               ROR  → log(ROR) / log(ROR_max)
               BCPNN → IC / IC_max
               MGPS  → EBGM / EBGM_max   (o EB05)

    Score finale: 0–100, arrotondato a 1 decimale.

    Returns
    -------
    DataFrame con colonne: ae_name, confidence_score, n_algorithms,
                           algorithms_positive, validation_status, weber_risk
    """
    if validated_df is None or len(validated_df) == 0:
        return pd.DataFrame()

    # Raccogli tutti gli AE con almeno un segnale positivo
    ae_set = set(validated_df["ae_name"].unique())

    rows = []
    for ae in ae_set:
        algos_positive = []
        signal_strengths = []

        # PRR
        if "prr" in algo_results and algo_results["prr"] is not None:
            df_prr = algo_results["prr"]
            row = df_prr[df_prr["ae_name"] == ae]
            if not row.empty and row.iloc[0].get("signal_positive", False):
                algos_positive.append("PRR")
                prr_val = row.iloc[0].get("PRR", 1)
                if prr_val > 0:
                    signal_strengths.append(np.log(max(prr_val, 1)))

        # ROR
        if "ror" in algo_results and algo_results["ror"] is not None:
            df_ror = algo_results["ror"]
            row = df_ror[df_ror["ae_name"] == ae]
            if not row.empty and row.iloc[0].get("signal_positive", False):
                algos_positive.append("ROR")
                ror_val = row.iloc[0].get("ROR", 1)
                if ror_val > 0:
                    signal_strengths.append(np.log(max(ror_val, 1)))

        # BCPNN
        if "bcpnn" in algo_results and algo_results["bcpnn"] is not None:
            df_bc = algo_results["bcpnn"]
            row = df_bc[df_bc["ae_name"] == ae]
            if not row.empty and row.iloc[0].get("signal_positive", False):
                algos_positive.append("BCPNN")
                ic_val = row.iloc[0].get("IC", 0)
                signal_strengths.append(max(ic_val, 0))

        # MGPS
        if "mgps" in algo_results and algo_results["mgps"] is not None:
            df_mg = algo_results["mgps"]
            row = df_mg[df_mg["ae_name"] == ae]
            if not row.empty and row.iloc[0].get("signal_positive", False):
                algos_positive.append("MGPS")
                eb_val = row.iloc[0].get("EBGM", row.iloc[0].get("EB05", 0))
                signal_strengths.append(np.log(max(eb_val, 1)))

        n_algos = len(algos_positive)
        avg_strength = np.mean(signal_strengths) if signal_strengths else 0.0

        # Validazione label
        val_row = validated_df[validated_df["ae_name"] == ae]
        val_status = val_row.iloc[0].get("validation_status", "") if not val_row.empty else ""

        rows.append({
            "ae_name":            ae,
            "n_algorithms":       n_algos,
            "algorithms_positive": algos_positive,
            "avg_strength":       avg_strength,
            "validation_status":  val_status,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Normalizza avg_strength su 0–1
    max_strength = result["avg_strength"].max()
    if max_strength > 0:
        result["strength_norm"] = result["avg_strength"] / max_strength
    else:
        result["strength_norm"] = 0.0

    # Score composito: 40% concordanza algoritmi + 60% forza segnale
    result["confidence_score"] = (
        0.40 * (result["n_algorithms"] / 4.0) +
        0.60 * result["strength_norm"]
    ) * 100

    result["confidence_score"] = result["confidence_score"].round(1)

    return result.sort_values("confidence_score", ascending=False).reset_index(drop=True)


# ============================================================================
# RENDERING LISTA AE CON ICONE
# ============================================================================

WEBER_COLORS = {
    "LOW":    ("🟢", "tooltip-weber-low",    "Bias minimo"),
    "MEDIUM": ("🟠", "tooltip-weber-medium", "Bias moderato"),
    "HIGH":   ("🔴", "tooltip-weber-high",   "Bias elevato"),
}

def render_ae_list(ranked_df: pd.DataFrame, weber_risk: str | None, limit: int | None = None):
    """
    Renderizza la lista ordinata degli AE con confidence bar e icone interattive.
    """
    if ranked_df.empty:
        st.info("Nessun segnale rilevato con i parametri correnti.")
        return

    display_df = ranked_df if limit is None else ranked_df.head(limit)

    # Determina icone Weber globali (rischio uguale per tutti gli AE della stessa run)
    weber_emoji, weber_css, weber_label = WEBER_COLORS.get(
        (weber_risk or "").upper(), ("⬜", "tooltip-weber-low", "N/D")
    )

    for i, row in display_df.iterrows():
        ae      = row["ae_name"]
        score   = row["confidence_score"]
        n_algo  = row["n_algorithms"]
        algos   = row.get("algorithms_positive", [])
        val_st  = row.get("validation_status", "")
        is_known = val_st == "KNOWN"

        # Pill algoritmi
        pills_html = "".join(f'<span class="algo-pill">{a}</span>' for a in algos)

        # Confidence bar
        bar_color = "#2B6CB0" if score >= 60 else ("#C05621" if score >= 35 else "#9B2335")
        bar_html = f"""
        <div class="conf-bar-wrap">
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{score}%;background:{bar_color};"></div>
            </div>
            <div class="conf-label">{score}/100</div>
        </div>"""

        # Icona validazione FDA
        fda_icon = "✅" if is_known else ""

        # Card HTML
        card_html = f"""
        <div class="signal-card">
            <div class="rank-badge">#{i+1}</div>
            <div style="flex:1">
                <div class="ae-name">{ae}</div>
                <div style="margin-top:4px">{pills_html}</div>
            </div>
            {bar_html}
            <div style="font-size:20px;min-width:30px;text-align:center">{fda_icon}</div>
            <div style="font-size:20px;min-width:30px;text-align:center">{weber_emoji}</div>
        </div>"""

        st.markdown(card_html, unsafe_allow_html=True)

        # Expander per i dettagli delle icone
        with st.expander("", expanded=False):
            col_fda, col_weber = st.columns(2)

            with col_fda:
                if is_known:
                    st.markdown(f"""
                    <div class="tooltip-known">
                        <strong>✅ Evento avverso scientificamente validato</strong><br>
                        Questo evento avverso è riportato nel bugiardino ufficiale FDA
                        per questo farmaco. La sua associazione è riconosciuta dalla
                        letteratura regolatoria e confermata da studi clinici.
                    </div>""", unsafe_allow_html=True)
                elif val_st == "POTENTIALLY_NEW":
                    st.markdown("""
                    <div style="background:#FFFFF0;border:1px solid #FAF089;border-radius:8px;
                                padding:10px 14px;font-size:13px;color:#744210;margin-bottom:8px">
                        <strong>🔍 Segnale potenzialmente nuovo</strong><br>
                        Questo evento avverso <em>non</em> compare nel bugiardino FDA.
                        Il segnale è rilevato statisticamente ma non ha ancora
                        una base scientifica consolidata nella letteratura regolatoria.
                        Richiede ulteriore indagine clinica.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.caption("Validazione FDA non disponibile per questo AE.")

            with col_weber:
                if weber_risk:
                    weber_text = {
                        "LOW":    "Il rischio di Weber effect è basso. I report sono distribuiti "
                                  "uniformemente nel tempo, senza picchi nelle fasi iniziali "
                                  "post-approvazione. Il segnale è statisticamente affidabile.",
                        "MEDIUM": "Rischio moderato di Weber effect. È presente una concentrazione "
                                  "di report nelle fasi iniziali post-approvazione che potrebbe "
                                  "sovrastimare il segnale. Interpretare con cautela.",
                        "HIGH":   "⚠️ Rischio elevato di Weber effect. I report si concentrano "
                                  "fortemente nei primi anni post-approvazione, il che suggerisce "
                                  "un bias significativo. Il segnale potrebbe riflettere un effetto "
                                  "di notorietà iniziale piuttosto che un rischio reale.",
                    }.get((weber_risk or "").upper(), "Dati Weber non disponibili.")

                    st.markdown(f"""
                    <div class="{weber_css}">
                        <strong>{weber_emoji} Weber effect — {weber_label}</strong><br>
                        {weber_text}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.caption("Weber check non eseguito.")


# ============================================================================
# CONFIG
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
# ESECUZIONE PIPELINE + DISPLAY
# ============================================================================

def run_and_display(
    config: dict,
    age_stratum: str | None,
    ae_limit: int | None,
):
    if not config["target_drug"]:
        st.warning("Inserisci almeno il nome del farmaco.")
        return

    # Avviso report esclusi per filtro età
    if age_stratum:
        n_unknown = count_unknown_age_excluded()
        if n_unknown:
            st.info(
                f"Filtro età attivo (`{age_stratum}`): **{n_unknown:,}** report "
                f"senza età registrata (`age_stratum = 'unknown'`) vengono esclusi "
                f"sia dal gruppo target che dal baseline."
            )

    # ── Cache contingency table ───────────────────────────────────────────
    # La CT dipende solo da (target_drug, where_extra, min_a).
    # Se questi tre parametri non cambiano, la CT viene riutilizzata
    # dalla session_state e il passo DuckDB viene saltato completamente.
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
                f"⚡ Contingency table dalla cache ({len(cached_ct)} coppie) "
                f"— solo gli algoritmi vengono rieseguiti."
            )

    # ── Esecuzione pipeline ───────────────────────────────────────────────
    spinner_msg = (
        "Eseguendo algoritmi in parallelo (CT dalla cache)…"
        if cached_ct is not None
        else "Costruendo la contingency table e calcolando i segnali…"
    )
    with st.spinner(spinner_msg):
        try:
            results = run_pipeline(config, precomputed_ct=cached_ct)
        except Exception as e:
            st.error(f"Errore durante l'esecuzione: {e}")
            return

    # ── Aggiorna cache CT se appena ricalcolata ───────────────────────────
    if cached_ct is None and results.get("error") != "no_data":
        st.session_state["_ct_cache"] = results["contingency_table"]
        st.session_state["_ct_key"]   = ct_key

    if results.get("error") == "no_data":
        st.error(f"Nessuna coppia trovata per **{config['target_drug']}** con questi filtri.")
        return

    ct        = results["contingency_table"]
    validated = results.get("validated", pd.DataFrame())
    weber     = results.get("weber_check", {})
    summary   = results.get("run_summary", {})
    weber_risk = weber.get("weber_risk") or summary.get("weber_risk")

    # Raccoglie risultati per algoritmo per il confidence score
    algo_results = {k: results.get(k) for k in ["prr", "ror", "bcpnn", "mgps"]}

    # ── Run summary ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Riepilogo run</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Farmaco",    config["target_drug"])
    m2.metric("Coppie CT",  len(ct))
    m3.metric("Tempo (s)",  summary.get("total_elapsed_s", "—"))
    m4.metric("Filtro",     config["where_extra"] or "Nessuno")

    if summary.get("algorithm_results"):
        algo_rows = [
            {"Algoritmo": a.upper(),
             "Segnali positivi": s["positive"],
             "Coppie valutate": s["total_pairs"]}
            for a, s in summary["algorithm_results"].items()
        ]
        st.dataframe(pd.DataFrame(algo_rows), use_container_width=True, hide_index=True)

    # ── Segnali AE ordinati per confidence ────────────────────────────────
    st.markdown('<div class="section-header">Adverse Events rilevati</div>', unsafe_allow_html=True)

    ranked = compute_confidence_score(validated, algo_results)

    if not ranked.empty:
        render_ae_list(ranked, weber_risk=weber_risk, limit=ae_limit)
    else:
        st.info("Nessun segnale positivo rilevato con i parametri correnti.")

    # ── Validazione label FDA (tabella completa) ───────────────────────────
    with st.expander("📋 Tabella completa segnali validati"):
        if validated is not None and len(validated) > 0:
            st.dataframe(validated, use_container_width=True)
            st.download_button(
                "Scarica segnali (CSV)",
                data=validated.to_csv(index=False).encode("utf-8"),
                file_name=f"{config['target_drug']}_signals.csv",
                mime="text/csv",
            )
        else:
            st.info("Nessun segnale validato.")

    # ── Weber detail ───────────────────────────────────────────────────────
    with st.expander("⏱ Weber effect — dettaglio"):
        if weber:
            st.json(weber)
        else:
            st.info("Weber check disabilitato o nessun dato disponibile.")

    # ── Contingency table ──────────────────────────────────────────────────
    with st.expander("🔢 Contingency table"):
        st.dataframe(ct, use_container_width=True)
        st.download_button(
            "Scarica CT (CSV)",
            data=ct.to_csv(index=False).encode("utf-8"),
            file_name=f"{config['target_drug']}_ct.csv",
            mime="text/csv",
        )


# ============================================================================
# HEADER
# ============================================================================

st.title("Drug Safety Signal Detection")
st.markdown(
    "Disproportionality analysis su dati FAERS — rilevamento segnali di sicurezza farmaci."
)

# ============================================================================
# INPUT PARAMETRI
# ============================================================================

st.markdown('<div class="section-header">Parametri di analisi</div>', unsafe_allow_html=True)

# Carica indice farmaci (lazy, cached)
drug_index = load_drug_index()

col1, col2, col3, col4 = st.columns(4)

with col1:
    drug_input_raw = st.selectbox(
        "Farmaco",
        options=SUGGESTED_DRUGS,
        index=None,
        placeholder="es. LAPATINIB, ibuprofen, brufen…",
        accept_new_options=True,
        key="drug_input",
    )

with col2:
    sex_input = st.selectbox(
        "Sesso",
        options=SEX_OPTIONS,
        index=0,
        key="sex_input",
    )

with col3:
    age_input_raw = st.text_input(
        "Età (anni o fascia)",
        placeholder="es. 45 → adult | 10 → pediatric | 70 → geriatric",
        key="age_input",
    )

with col4:
    comed_input_raw = st.selectbox(
        "Co-medicazione",
        options=SUGGESTED_DRUGS,
        index=None,
        placeholder="opzionale, es. CAPECITABINE",
        accept_new_options=True,
        key="comedication_input",
    )

# ── Parametri avanzati ──────────────────────────────────────────────────────
with st.expander("Parametri avanzati"):
    adv1, adv2 = st.columns(2)

    with adv1:
        algorithms = st.multiselect(
            "Algoritmi", ["prr","ror","bcpnn","mgps"],
            default=DEFAULT_ALGORITHMS, key="algorithms_input",
        )
        min_a = st.number_input(
            "Min occorrenze (a)", min_value=1, value=DEFAULT_MIN_A, step=1, key="min_a_input",
        )
        fdr_threshold = st.number_input(
            "Soglia FDR (PRR/ROR)", min_value=0.0, max_value=1.0,
            value=DEFAULT_FDR, step=0.01, key="fdr_input",
        )

    with adv2:
        eb05_threshold = st.number_input(
            "Soglia EB05 (MGPS)", min_value=0.0, value=DEFAULT_EB05, step=0.1, key="eb05_input",
        )
        ic_threshold = st.number_input(
            "Soglia IC025 (BCPNN)", value=DEFAULT_IC, step=0.1, key="ic_input",
        )
        weber_override = st.text_input(
            "Anno approvazione (override Weber)", placeholder="es. 2007", key="weber_override_input",
        )
        mistral_key = st.text_input(
            "Mistral API key (fallback drug name)", type="password", key="mistral_key_input",
        )

    openfda_api_key = st.text_input(
        "openFDA API key (opzionale)", type="password", key="api_key_input",
    )
    validate_label_cb = st.checkbox("Validazione label FDA", value=True, key="validate_input")
    check_weber_cb    = st.checkbox("Weber effect check",    value=True, key="weber_check_input")

# ── Filtro visualizzazione AE ────────────────────────────────────────────────
ae_limit_map = {"Top 5": 5, "Top 10": 10, "Top 20": 20, "Tutti": None}
ae_limit_label = st.radio(
    "Mostra AE",
    options=list(ae_limit_map.keys()),
    index=1,
    horizontal=True,
    key="ae_limit_input",
)
ae_limit = ae_limit_map[ae_limit_label]

# ============================================================================
# BOTTONE ESEGUI
# ============================================================================

if st.button("▶ Esegui analisi", type="primary"):

    # ── Risolvi nome farmaco ─────────────────────────────────────────────
    if not drug_input_raw:
        st.warning("Inserisci il nome del farmaco.")
        st.stop()

    resolved_drug, drug_method, drug_score = resolve_drug_name(
        drug_input_raw, drug_index, mistral_key or None
    )

    if drug_method == "fuzzy":
        st.markdown(
            f'<span class="match-chip">🔍 "{drug_input_raw}" → <strong>{resolved_drug}</strong> '
            f'(fuzzy match, score {drug_score:.0f}/100)</span>',
            unsafe_allow_html=True,
        )
    elif drug_method == "mistral":
        st.markdown(
            f'<span class="match-chip">🤖 "{drug_input_raw}" → <strong>{resolved_drug}</strong> '
            f'(via Mistral AI)</span>',
            unsafe_allow_html=True,
        )
    elif drug_method == "passthrough":
        st.warning(
            f"Nessun match affidabile trovato per '{drug_input_raw}'. "
            f"Procedo con il nome inserito — se non vengono trovate coppie, "
            f"prova il nome in inglese o il principio attivo."
        )

    # ── Risolvi co-medicazione ───────────────────────────────────────────
    resolved_comed = None
    if comed_input_raw and str(comed_input_raw).strip():
        resolved_comed, comed_method, comed_score = resolve_comedication(
            str(comed_input_raw), drug_index, mistral_key or None
        )
        if comed_method == "fuzzy" and comed_score < 100:
            st.markdown(
                f'<span class="match-chip">🔍 Co-med: "{comed_input_raw}" → '
                f'<strong>{resolved_comed}</strong> (score {comed_score:.0f}/100)</span>',
                unsafe_allow_html=True,
            )
        elif comed_method == "mistral":
            st.markdown(
                f'<span class="match-chip">🤖 Co-med: "{comed_input_raw}" → '
                f'<strong>{resolved_comed}</strong> (via Mistral AI)</span>',
                unsafe_allow_html=True,
            )

    # ── Risolvi età ──────────────────────────────────────────────────────
    age_stratum = age_to_stratum(age_input_raw)
    if age_input_raw.strip() and age_stratum:
        st.markdown(
            f'<span class="match-chip">👤 Età "{age_input_raw}" → '
            f'<strong>{age_stratum}</strong></span>',
            unsafe_allow_html=True,
        )
    elif age_input_raw.strip() and not age_stratum:
        st.warning(f"Input età '{age_input_raw}' non riconosciuto — nessun filtro applicato.")

    # ── Costruisci where_extra ───────────────────────────────────────────
    where_extra = build_where_extra(sex_input, age_stratum, resolved_comed)

    # ── Costruisci config ────────────────────────────────────────────────
    weber_override_int = int(weber_override.strip()) if weber_override.strip().isdigit() else None

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


# ============================================================================
# PANNELLO PANORAMICO
# ============================================================================

st.divider()
st.markdown('<div class="section-header">Panoramica segnali storici</div>', unsafe_allow_html=True)

if SIGNALS_FULL_PATH.exists():
    try:
        signals_full = pd.read_parquet(SIGNALS_FULL_PATH)
        st.caption(f"{len(signals_full)} segnali totali registrati nelle run precedenti.")
        st.dataframe(signals_full, use_container_width=True)
        st.download_button(
            "Scarica signals_full (CSV)",
            data=signals_full.to_csv(index=False).encode("utf-8"),
            file_name="signals_full.csv",
            mime="text/csv",
            key="download_signals_full",
        )
    except Exception as e:
        st.error(f"Errore nella lettura di signals_full.parquet: {e}")
else:
    st.info(
        "Nessun signals_full.parquet trovato. "
        "Esegui almeno un'analisi per popolare il pannello panoramico."
    )