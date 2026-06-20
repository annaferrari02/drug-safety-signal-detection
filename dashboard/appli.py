import streamlit as st
import pandas as pd
from pathlib import Path

from run_signals import run_pipeline  # copiato in /app dal Dockerfile, vedi note

st.set_page_config(page_title="Drug Safety Signal Detection", layout="wide")

st.title("Drug Safety Signal Detection")
st.markdown("Configura i parametri e avvia la pipeline di disproportionality analysis sui dati FAERS.")


# ---------------------------------------------------------------------------
# Path del parquet con tutti i segnali storici/aggregati, scritto da
# run_pipeline() stessa (vedi SIGNALS_OUT in run_signals.py) ad ogni run.
# Risolve a /app/data/signals_full.parquet nel container, coerente con
# il volume ./data:/app/data del docker-compose.
# ---------------------------------------------------------------------------
SIGNALS_FULL_PATH = Path(__file__).resolve().parent / "data" / "signals_full.parquet"


# ---------------------------------------------------------------------------
# Default per i parametri avanzati
# ---------------------------------------------------------------------------
DEFAULT_ALGORITHMS = ["prr", "ror", "bcpnn", "mgps"]
DEFAULT_MIN_A = 3
DEFAULT_FDR_THRESHOLD = 0.05
DEFAULT_EB05_THRESHOLD = 2.0
DEFAULT_IC_THRESHOLD = 0.0

# Liste statiche di suggerimento per i blocchi DRUG e CO-MEDICATION.
# TODO: personalizza con i farmaci piu rilevanti per il tuo caso d'uso.
SUGGESTED_DRUGS = [
    "LAPATINIB",
    "CAPECITABINE",
    "TRASTUZUMAB",
    "PACLITAXEL",
    "DOCETAXEL",
    "TAMOXIFEN",
    "LETROZOLE",
    "CISPLATIN",
    "DOXORUBICIN",
    "METFORMIN",
]

SEX_OPTIONS = ["Tutti", "Female", "Male"]
AGE_OPTIONS = ["Tutti", "Pediatric", "Adult", "Geriatric"]


# ---------------------------------------------------------------------------
# Helper: costruisce where_extra a partire dai 4 blocchi UI.
# NB: la subquery di co-medication referenzia faers_flat_deduped.parquet,
# cioe PARQUET_PATH dentro run_signals.py. Qui usiamo lo stesso path
# assoluto che risolve run_signals.py nel container (/app/data/...).
# ---------------------------------------------------------------------------
COMEDICATION_PARQUET_PATH = Path(__file__).resolve().parent / "data" / "faers_flat_deduped.parquet"


def build_where_extra(sex: str, age_stratum: str, comedication: str) -> str | None:
    clauses = []

    if sex and sex != "Tutti":
        clauses.append(f"sex = '{sex.strip().lower()}'")

    if age_stratum and age_stratum != "Tutti":
        clauses.append(f"age_stratum = '{age_stratum.strip().lower()}'")

    if comedication and comedication.strip():
        comed = comedication.strip().upper()
        clauses.append(
            f"""safetyreportid IN (
                SELECT DISTINCT safetyreportid
                FROM '{COMEDICATION_PARQUET_PATH}'
                WHERE drug_name = '{comed}'
            )"""
        )

    if not clauses:
        return None
    return " AND ".join(clauses)


def count_unknown_age_excluded(age_stratum: str) -> int | None:
    """
    Se e stato applicato un filtro su age_stratum (diverso da 'Tutti' / vuoto),
    conta quanti report con safetyreportid distinto hanno age_stratum = 'unknown'
    nell'intero dataset.
    """
    if not age_stratum or age_stratum == "Tutti":
        return None

    import duckdb
    query = f"""
        SELECT COUNT(DISTINCT safetyreportid)
        FROM '{COMEDICATION_PARQUET_PATH}'
        WHERE age_stratum = 'unknown'
    """
    try:
        con = duckdb.connect()
        result = con.execute(query).fetchone()
        return result[0] if result else None
    except Exception:
        return None


def build_config_from_ui(
    target_drug: str,
    where_extra: str | None,
    min_a: int,
    algorithms: list,
    fdr_threshold: float,
    eb05_threshold: float,
    ic_threshold: float,
    openfda_api_key: str | None,
    validate_label: bool,
    check_weber: bool,
    weber_approval_override: int | None,
) -> dict:
    return {
        "target_drug": target_drug.strip().upper(),
        "where_extra": where_extra,
        "min_a": min_a,
        "algorithms": algorithms,
        "fdr_threshold": fdr_threshold,
        "eb05_threshold": eb05_threshold,
        "ic_threshold": ic_threshold,
        "openfda_api_key": openfda_api_key or None,
        "validate_label": validate_label,
        "check_weber": check_weber,
        "weber_approval_override": weber_approval_override,
    }


# ---------------------------------------------------------------------------
# Esecuzione pipeline + visualizzazione risultati
# ---------------------------------------------------------------------------
def run_and_display(config: dict, age_stratum_for_unknown_check: str | None = None):
    if not config["target_drug"]:
        st.warning("Inserisci almeno il nome del drug prima di eseguire.")
        return

    st.subheader("Parametri estratti")
    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Drug", config["target_drug"])
    pcol2.metric("Min occorrenze (a)", config["min_a"])
    pcol3.metric("Filtro", config["where_extra"] or "Nessuno")

    unknown_excluded = count_unknown_age_excluded(age_stratum_for_unknown_check)
    if unknown_excluded is not None:
        st.info(
            f"Filtro eta attivo ('{age_stratum_for_unknown_check}'): "
            f"**{unknown_excluded:,}** report con eta non registrata (`age_stratum = 'unknown'`) "
            f"sono esclusi sia dal gruppo target che dalla baseline di confronto, "
            f"poiche non e possibile attribuirli con certezza a nessuna fascia."
        )

    with st.spinner("Eseguo la pipeline completa..."):
        try:
            results = run_pipeline(config)
        except Exception as e:
            st.error(f"Errore durante l'esecuzione della pipeline: {e}")
            return

    if results.get("error") == "no_data":
        st.error(f"Nessuna coppia trovata per '{config['target_drug']}' con questi filtri.")
        return

    # --- Contingency table ---
    ct = results["contingency_table"]
    st.subheader(f"Contingency table — {len(ct)} coppie (drug, PT)")
    st.dataframe(ct, use_container_width=True)
    st.download_button(
        "Scarica contingency table (CSV)",
        data=ct.to_csv(index=False).encode("utf-8"),
        file_name=f"{config['target_drug']}_contingency.csv",
        mime="text/csv",
        key="download_ct",
    )

    # --- Risultati per algoritmo ---
    st.subheader("Risultati per algoritmo")
    algo_tabs = st.tabs([a.upper() for a in config["algorithms"]])
    for tab, algo in zip(algo_tabs, config["algorithms"]):
        with tab:
            algo_df = results.get(algo)
            if algo_df is None:
                st.info(f"Nessun risultato per {algo.upper()} (algoritmo fallito o non eseguito).")
            elif len(algo_df) == 0:
                st.info(f"Nessun segnale rilevato da {algo.upper()}.")
            else:
                st.dataframe(algo_df, use_container_width=True)
                st.download_button(
                    f"Scarica {algo.upper()} (CSV)",
                    data=algo_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{config['target_drug']}_{algo}.csv",
                    mime="text/csv",
                    key=f"download_{algo}",
                )

    # --- Validazione label FDA ---
    st.subheader("Validazione label FDA")
    validated = results.get("validated")
    if validated is not None and len(validated) > 0:
        st.dataframe(validated, use_container_width=True)
    else:
        st.info("Nessun segnale validato o validazione disabilitata.")

    # --- Weber effect check ---
    st.subheader("Weber effect check")
    weber = results.get("weber_check")
    if weber:
        st.json(weber)
    else:
        st.info("Weber check disabilitato o nessun dato disponibile.")

    # --- Run summary ---
    st.subheader("Run summary")
    summary = results.get("run_summary", {})

    st.metric("Tempo totale (s)", summary.get("total_elapsed_s", "—"))

    algo_results_summary = summary.get("algorithm_results", {})
    if algo_results_summary:
        rows = [
            {"algoritmo": algo.upper(), "positivi": stats.get("positive"), "coppie_valutate": stats.get("total_pairs")}
            for algo, stats in algo_results_summary.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if summary.get("validation"):
        v = summary["validation"]
        vcol1, vcol2, vcol3 = st.columns(3)
        vcol1.metric("KNOWN", v.get("KNOWN", 0))
        vcol2.metric("POTENTIALLY NEW", v.get("POTENTIALLY_NEW", 0))
        vcol3.metric("NO LABEL", v.get("NO_LABEL", 0))

    if summary.get("weber_risk"):
        st.metric("Weber effect risk", summary["weber_risk"])

    st.success(
        f"I segnali validati di questa run sono stati aggiunti a "
        f"`signals_full.parquet` per il pannello panoramico qui sotto."
    )


# ---------------------------------------------------------------------------
# 4 blocchi di input — dropdown di suggerimenti + possibilita di scrivere
# un valore libero (accept_new_options richiede Streamlit >= 1.40)
# ---------------------------------------------------------------------------
st.subheader("Parametri di analisi")

col1, col2, col3, col4 = st.columns(4)

with col1:
    target_drug = st.selectbox(
        "Drug",
        options=SUGGESTED_DRUGS,
        index=None,
        placeholder="es. LAPATINIB",
        accept_new_options=True,
        key="drug_input",
    )

with col2:
    sex = st.selectbox(
        "Sex",
        options=SEX_OPTIONS,
        index=0,
        accept_new_options=True,
        key="sex_input",
    )

with col3:
    age_stratum = st.selectbox(
        "Age",
        options=AGE_OPTIONS,
        index=0,
        accept_new_options=True,
        key="age_input",
    )

with col4:
    comedication = st.selectbox(
        "Co-medication",
        options=SUGGESTED_DRUGS,
        index=None,
        placeholder="opzionale, es. CAPECITABINE",
        accept_new_options=True,
        key="comedication_input",
    )


# ---------------------------------------------------------------------------
# Parametri avanzati
# ---------------------------------------------------------------------------
with st.expander("Parametri avanzati"):
    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        algorithms = st.multiselect(
            "Algoritmi",
            ["prr", "ror", "bcpnn", "mgps"],
            default=DEFAULT_ALGORITHMS,
            key="algorithms_input",
        )
        min_a = st.number_input("Min occorrenze (a)", min_value=1, value=DEFAULT_MIN_A, step=1, key="min_a_input")
        fdr_threshold = st.number_input(
            "Soglia FDR (PRR/ROR)", min_value=0.0, max_value=1.0,
            value=DEFAULT_FDR_THRESHOLD, step=0.01, key="fdr_input",
        )

    with adv_col2:
        eb05_threshold = st.number_input(
            "Soglia EB05 (MGPS)", min_value=0.0, value=DEFAULT_EB05_THRESHOLD, step=0.1, key="eb05_input",
        )
        ic_threshold = st.number_input(
            "Soglia IC025 (BCPNN)", value=DEFAULT_IC_THRESHOLD, step=0.1, key="ic_input",
        )
        weber_override = st.text_input(
            "Anno approvazione (override Weber, opzionale)", placeholder="es. 2007", key="weber_override_input",
        )

    openfda_api_key = st.text_input("openFDA API key (opzionale)", type="password", key="api_key_input")
    validate_label = st.checkbox("Validazione label FDA", value=True, key="validate_input")
    check_weber = st.checkbox("Weber effect check", value=True, key="weber_check_input")


# ---------------------------------------------------------------------------
# Bottone esecuzione
# ---------------------------------------------------------------------------
if st.button("Esegui"):
    where_extra = build_where_extra(sex, age_stratum, comedication or "")
    weber_override_int = int(weber_override) if weber_override.strip().isdigit() else None

    config = build_config_from_ui(
        target_drug=target_drug or "",
        where_extra=where_extra,
        min_a=min_a,
        algorithms=algorithms,
        fdr_threshold=fdr_threshold,
        eb05_threshold=eb05_threshold,
        ic_threshold=ic_threshold,
        openfda_api_key=openfda_api_key,
        validate_label=validate_label,
        check_weber=check_weber,
        weber_approval_override=weber_override_int,
    )

    run_and_display(config, age_stratum_for_unknown_check=age_stratum)


# ---------------------------------------------------------------------------
# Pannello panoramico — legge signals_full.parquet (storico cumulativo di
# tutti i segnali validati nelle run precedenti, scritto da run_pipeline())
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Panoramica segnali storici (signals_full.parquet)")

if SIGNALS_FULL_PATH.exists():
    try:
        signals_full = pd.read_parquet(SIGNALS_FULL_PATH)
        st.caption(f"{len(signals_full)} segnali totali registrati finora.")
        st.dataframe(signals_full, use_container_width=True)
        st.download_button(
            "Scarica signals_full (CSV)",
            data=signals_full.to_csv(index=False).encode("utf-8"),
            file_name="signals_full.csv",
            mime="text/csv",
            key="download_signals_full",
        )
    except Exception as e:
        st.error(f"Errore durante la lettura di signals_full.parquet: {e}")
else:
    st.info(
        "Nessun signals_full.parquet trovato ancora. "
        "Esegui almeno un'analisi per popolare il pannello panoramico."
    )