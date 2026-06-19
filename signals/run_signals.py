#script that orchestrates the logic of the signals, calling 
#the functions from src 
"""
run_signals.py

Orchestrazione della pipeline di signal detection.
Restituisce i risultati in memoria come dizionario di DataFrame,
senza scrivere file temporanei su disco.

Le uniche scritture su disco sono:
    data/label_cache.json    — cache delle chiamate API openFDA label (costose)
    data/approval_cache.json — cache degli anni di approvazione FDA (weber_check)

Utilizzo attuale (senza dashboard):
    results = run_pipeline(get_run_config())

Utilizzo futuro (con dashboard):
    config  = build_config_from_ui(user_input)   # parametri dal form Streamlit
    results = run_pipeline(config)               # stessa funzione, nessuna modifica
    # risultati disponibili in st.session_state per tutta la sessione

Struttura del dict restituito da run_pipeline():
    {
        "config":             dict            — parametri usati
        "contingency_table":  pd.DataFrame    — CT 2x2
        "prr":                pd.DataFrame | None
        "ror":                pd.DataFrame | None
        "bcpnn":              pd.DataFrame | None
        "mgps":               pd.DataFrame | None
        "validated":          pd.DataFrame    — segnali con validazione label FDA
        "weber_check":        dict            — risultato check Weber effect
        "run_summary":        dict            — metadata (tempi, conteggi)
    }
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contingency_table import build_contingency_table, qc_contingency_table
from src.signals import compute_prr, compute_ror, compute_bcpnn, compute_mgps
from src.validate_label import validate_signals, validation_summary
from src.weber_check import check_weber_effect, weber_summary          # <-- NUOVO

PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "faers_flat_deduped.parquet"
CONFIG_FILE  = Path(__file__).resolve().parent.parent / "run_config.json"


# ── CONFIGURAZIONE ───────────────────────────────────────────────────────────
# get_run_config() è l'unico punto che cambierà con l'integrazione della dashboard.

def get_run_config() -> dict:
    """
    Restituisce la configurazione del run corrente.

    Ordine di priorità:
        1. CONFIG_FILE (run_config.json) se presente — utile per test ripetibili
        2. Valori di default hardcoded — utile per sviluppo e CI

    Struttura del dict restituito:
    {
        "target_drug":            str        — nome farmaco in uppercase, es. "LAPATINIB"
        "where_extra":            str | None — filtro SQL per stratificazione
        "min_a":                  int        — soglia minima cella a della CT (default 3)
        "algorithms":             list       — sottoinsieme di ["prr", "ror", "bcpnn", "mgps"]
        "fdr_threshold":          float      — soglia FDR per PRR e ROR (default 0.05)
        "eb05_threshold":         float      — soglia EB05 per MGPS (default 2.0)
        "ic_threshold":           float      — soglia IC025 per BCPNN (default 0.0)
        "openfda_api_key":        str | None — chiave API openFDA opzionale
        "validate_label":         bool       — se True esegue validazione label FDA
        "check_weber":            bool       — se True esegue Weber effect check
        "weber_approval_override":int | None — anno di approvazione manuale (bypassa API)
    }

    TODO (dashboard): sostituire con build_config_from_ui(user_input) che riceve
    i parametri dal form Streamlit e restituisce lo stesso dict.
    """
    defaults = {
        "target_drug":             "LAPATINIB",
        "where_extra":             None,
        "min_a":                   3,
        "algorithms":              ["prr", "ror", "bcpnn", "mgps"],
        "fdr_threshold":           0.05,
        "eb05_threshold":          2.0,
        "ic_threshold":            0.0,
        "openfda_api_key":         None,
        "validate_label":          True,
        "check_weber":             True,       # <-- NUOVO
        "weber_approval_override": None,       # <-- NUOVO
    }

    if CONFIG_FILE.exists():
        print(f"  [CONFIG] Carico parametri da {CONFIG_FILE}")
        user_config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        defaults.update(user_config)
    else:
        print(f"  [CONFIG] {CONFIG_FILE} non trovato, uso valori di default")

    defaults["target_drug"] = defaults["target_drug"].upper().strip()
    return defaults


# ── LOGICA DI ESECUZIONE ─────────────────────────────────────────────────────
# Queste funzioni non cambieranno con l'integrazione della dashboard.

def run_algorithm(name: str, ct: pd.DataFrame, config: dict) -> pd.DataFrame | None:
    """
    Esegue un singolo algoritmo di disproportionality analysis.
    Gestisce le eccezioni in isolamento: se un algoritmo fallisce,
    gli altri continuano.
    """
    fn_map = {
        "prr":   lambda: compute_prr(ct,
                     min_events=config["min_a"],
                     decision_thres=config["fdr_threshold"]),
        "ror":   lambda: compute_ror(ct,
                     min_events=config["min_a"],
                     decision_thres=config["fdr_threshold"]),
        "bcpnn": lambda: compute_bcpnn(ct,
                     min_events=config["min_a"],
                     ic_threshold=config["ic_threshold"]),
        "mgps":  lambda: compute_mgps(ct,
                     min_events=config["min_a"],
                     eb05_threshold=config["eb05_threshold"]),
    }

    if name not in fn_map:
        print(f"  [WARN] Algoritmo '{name}' non riconosciuto, saltato")
        return None

    try:
        t0      = time.time()
        result  = fn_map[name]()
        elapsed = time.time() - t0
        n_pos   = result["signal_positive"].sum()
        print(f"  [{name.upper()}] {len(result)} coppie, "
              f"{n_pos} segnali positivi — {elapsed:.1f}s")
        return result

    except Exception as e:
        print(f"  [ERR] {name.upper()} fallito: {e}")
        return None


def build_validated_union(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Unisce i segnali positivi di tutti gli algoritmi in un unico DataFrame
    con una colonna 'algorithm' che indica la provenienza di ogni riga.
    Un AE rilevato da più algoritmi appare più volte (una per algoritmo).
    """
    frames = []
    for algo_name, df in results.items():
        if df is None:
            continue
        positive = df[df["signal_positive"] == True].copy()
        positive["algorithm"] = algo_name
        frames.append(positive)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_run_summary(
        config:        dict,
        algo_results:  dict[str, pd.DataFrame],
        validated:     pd.DataFrame,
        weber:         dict,
        total_elapsed: float,
) -> dict:
    """
    Costruisce il dizionario di metadata del run.
    In futuro il dashboard può mostrare questi dati nella UI
    senza dover rieseguire la pipeline.
    """
    algo_stats = {}
    for name, df in algo_results.items():
        if df is not None:
            algo_stats[name] = {
                "total_pairs": len(df),
                "positive":    int(df["signal_positive"].sum()),
            }

    val_stats = {}
    if len(validated) > 0 and "validation_status" in validated.columns:
        vc = validated["validation_status"].value_counts().to_dict()
        val_stats = {
            "KNOWN":           vc.get("KNOWN", 0),
            "POTENTIALLY_NEW": vc.get("POTENTIALLY_NEW", 0),
            "NO_LABEL":        vc.get("NO_LABEL", 0),
        }

    return {
        "run_timestamp":     datetime.now().isoformat(),
        "target_drug":       config["target_drug"],
        "where_extra":       config["where_extra"],
        "min_a":             config["min_a"],
        "algorithms":        config["algorithms"],
        "thresholds": {
            "fdr":  config["fdr_threshold"],
            "eb05": config["eb05_threshold"],
            "ic":   config["ic_threshold"],
        },
        "algorithm_results": algo_stats,
        "validation":        val_stats,
        "weber_risk":        weber.get("weber_risk"),           # <-- NUOVO
        "total_elapsed_s":   round(total_elapsed, 1),
    }


def run_pipeline(config: dict) -> dict:
    """
    Esegue la pipeline completa e restituisce tutti i risultati in memoria.

    Questa è la funzione principale da chiamare — sia nella modalità attuale
    (CLI / script) che in futuro dalla dashboard Streamlit.

    Parameters
    ----------
    config : dict — output di get_run_config() o di build_config_from_ui()

    Returns
    -------
    dict con chiavi:
        config, contingency_table, prr, ror, bcpnn, mgps,
        validated, weber_check, run_summary
    """
    t_start = time.time()

    print(f"\n  Drug target    : {config['target_drug']}")
    print(f"  Stratificazione: {config['where_extra'] or 'globale'}")
    print(f"  Algoritmi      : {config['algorithms']}")

    # STEP 1: Contingency table
    print("\n=== Contingency table ===")
    ct = build_contingency_table(
        parquet_path=str(PARQUET_PATH),
        target_drug=config["target_drug"],
        min_a=config["min_a"],
        where_extra=config["where_extra"],
    )

    if len(ct) == 0:
        print(f"  [ERR] Nessuna coppia trovata per '{config['target_drug']}'.")
        return {"config": config, "error": "no_data"}

    qc_contingency_table(ct, label=config["target_drug"])

    # STEP 2: Signal detection
    print("\n=== Signal detection ===")
    algo_results = {}
    for algo in config["algorithms"]:
        algo_results[algo] = run_algorithm(algo, ct, config)

    # STEP 3: Validazione label FDA
    print("\n=== Validazione label FDA ===")
    validated = build_validated_union(algo_results)

    if config["validate_label"] and len(validated) > 0:
        validated = validate_signals(
            signals_df=validated,
            drug_name=config["target_drug"].lower(),
            api_key=config["openfda_api_key"],
        )
        validation_summary(validated)
    else:
        print("  [SKIP] Validazione disabilitata o nessun segnale positivo")

    # STEP 4: Weber effect check                                    # <-- NUOVO
    print("\n=== Weber effect check ===")
    weber = {}
    if config.get("check_weber", True):
        weber = check_weber_effect(
            parquet_path=str(PARQUET_PATH),
            target_drug=config["target_drug"],
            api_key=config.get("openfda_api_key"),
            approval_year_override=config.get("weber_approval_override"),
        )
        weber_summary(weber)
    else:
        print("  [SKIP] Weber check disabilitato")

    # STEP 5: Run summary
    print("\n=== Run summary ===")
    run_summary = build_run_summary(
        config=config,
        algo_results=algo_results,
        validated=validated,
        weber=weber,
        total_elapsed=time.time() - t_start,
    )

    print(f"\n=== Completato in {run_summary['total_elapsed_s']}s ===")

    return {
        "config":            config,
        "contingency_table": ct,
        **algo_results,         # prr, ror, bcpnn, mgps — ognuno None se fallito
        "validated":         validated,
        "weber_check":       weber,            # <-- NUOVO
        "run_summary":       run_summary,
    }


if __name__ == "__main__":
    config  = get_run_config()
    results = run_pipeline(config)

    summary = results["run_summary"]
    print("\n=== RIEPILOGO ===")
    for algo, stats in summary.get("algorithm_results", {}).items():
        print(f"  {algo.upper():<6}: {stats['positive']} segnali positivi "
              f"su {stats['total_pairs']} coppie valutate")
    if summary.get("validation"):
        v = summary["validation"]
        print(f"\n  Validazione label:")
        print(f"    KNOWN          : {v.get('KNOWN', 0)}")
        print(f"    POTENTIALLY NEW: {v.get('POTENTIALLY_NEW', 0)}")
        print(f"    NO LABEL       : {v.get('NO_LABEL', 0)}")
    if summary.get("weber_risk"):
        print(f"\n  Weber effect risk: {summary['weber_risk']}")