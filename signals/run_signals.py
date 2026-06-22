"""
run_signals.py  —  pipeline di signal detection (versione ottimizzata)

Ottimizzazioni rispetto alla versione precedente
─────────────────────────────────────────────────
1. Algoritmi in parallelo
   PRR / ROR / BCPNN / MGPS girano con ThreadPoolExecutor(max_workers=4).
   Il tempo totale degli algoritmi = tempo del più lento (MGPS), non la somma.

2. Validazione label + Weber in parallelo
   Le due chiamate HTTP a openFDA sono indipendenti e girano in parallelo
   con ThreadPoolExecutor(max_workers=2).
   Cache JSON su disco (label_cache.json, approval_cache.json) già presente:
   al secondo run queste chiamate sono gratuite (sub-millisecondo).

3. run_pipeline() accetta precomputed_ct
   Se la contingency table è già in session_state (stesso farmaco + filtri),
   il passo DuckDB viene saltato completamente.
   Chiamata da appli.py:
       run_pipeline(config, precomputed_ct=st.session_state["ct"])

4. Timing granulare
   Ogni step stampa il proprio tempo. Utile per individuare il collo di
   bottiglia reale sulla macchina specifica.

Struttura del dict restituito da run_pipeline() — invariata:
    {
        "config":             dict
        "contingency_table":  pd.DataFrame
        "prr":                pd.DataFrame | None
        "ror":                pd.DataFrame | None
        "bcpnn":              pd.DataFrame | None
        "mgps":               pd.DataFrame | None
        "validated":          pd.DataFrame
        "weber_check":        dict
        "run_summary":        dict
    }
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contingency_table import build_contingency_table, qc_contingency_table
from src.signals import compute_prr, compute_ror, compute_bcpnn, compute_mgps
from src.validate_label import validate_signals, validation_summary
from src.weber_check import check_weber_effect, weber_summary

PARQUET_PATH = Path(__file__).resolve().parent / "data" / "faers_sorted.parquet"
CONFIG_FILE  = Path(__file__).resolve().parent / "run_config.json"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────

def get_run_config() -> dict:
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
        "check_weber":             True,
        "weber_approval_override": None,
    }
    if CONFIG_FILE.exists():
        print(f"  [CONFIG] Carico parametri da {CONFIG_FILE}")
        defaults.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    else:
        print(f"  [CONFIG] {CONFIG_FILE} non trovato, uso valori di default")

    defaults["target_drug"] = defaults["target_drug"].upper().strip()
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Contingency table
# ─────────────────────────────────────────────────────────────────────────────

def _build_ct(config: dict) -> pd.DataFrame:
    """Costruisce la contingency table via DuckDB. Il passo più pesante della pipeline."""
    return build_contingency_table(
        parquet_path=str(PARQUET_PATH),
        target_drug=config["target_drug"],
        min_a=config["min_a"],
        where_extra=config["where_extra"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Algoritmi (eseguiti in parallelo)
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm(name: str, ct: pd.DataFrame, config: dict) -> pd.DataFrame | None:
    """
    Esegue un singolo algoritmo in isolamento.
    Un'eccezione in un algoritmo non blocca gli altri.
    """
    fn_map = {
        "prr":   lambda: compute_prr(
                     ct,
                     min_events=config["min_a"],
                     decision_thres=config["fdr_threshold"]),
        "ror":   lambda: compute_ror(
                     ct,
                     min_events=config["min_a"],
                     decision_thres=config["fdr_threshold"]),
        "bcpnn": lambda: compute_bcpnn(
                     ct,
                     min_events=config["min_a"],
                     ic_threshold=config["ic_threshold"]),
        "mgps":  lambda: compute_mgps(
                     ct,
                     min_events=config["min_a"],
                     eb05_threshold=config["eb05_threshold"]),
        # force_priors=True è già il default di compute_mgps:
        # salta l'ottimizzazione numerica dei prior (la parte più lenta di MGPS)
    }

    if name not in fn_map:
        print(f"  [WARN] Algoritmo '{name}' non riconosciuto, saltato")
        return None

    t0 = time.time()
    try:
        result  = fn_map[name]()
        elapsed = time.time() - t0
        n_pos   = int(result["signal_positive"].sum())
        print(f"  [{name.upper():5}] {len(result):>5} coppie · {n_pos:>4} positivi · {elapsed:.2f}s")
        return result
    except Exception as e:
        print(f"  [ERR] {name.upper()} fallito ({time.time()-t0:.2f}s): {e}")
        return None


def _run_algorithms_parallel(ct: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame | None]:
    """
    Esegue tutti gli algoritmi configurati in parallelo.
    Ordine dei risultati: determinato da as_completed, non dall'ordine di input.
    """
    algo_results: dict[str, pd.DataFrame | None] = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_algorithm, name, ct, config): name
            for name in config["algorithms"]
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                algo_results[name] = future.result()
            except Exception as e:
                print(f"  [ERR] Future {name} fallita: {e}")
                algo_results[name] = None

    return algo_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Unione segnali positivi
# ─────────────────────────────────────────────────────────────────────────────

def build_validated_union(results: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """
    Unisce i segnali positivi di tutti gli algoritmi.
    Ogni AE appare una riga per algoritmo che lo ha rilevato.
    """
    frames = []
    for algo_name, df in results.items():
        if df is None:
            continue
        positive = df[df["signal_positive"] == True].copy()
        positive["algorithm"] = algo_name
        frames.append(positive)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4+5 — Validazione label FDA + Weber (eseguiti in parallelo)
# ─────────────────────────────────────────────────────────────────────────────

def _run_validation(validated_union: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Valida i segnali positivi contro il bugiardino openFDA."""
    if not config.get("validate_label", True) or len(validated_union) == 0:
        print("  [SKIP] Validazione label disabilitata o nessun segnale positivo")
        return validated_union

    t0 = time.time()
    result = validate_signals(
        signals_df=validated_union,
        drug_name=config["target_drug"].lower(),
        api_key=config.get("openfda_api_key"),
    )
    print(f"  [LABEL ] Validazione completata in {time.time()-t0:.2f}s")
    validation_summary(result)
    return result


def _run_weber(config: dict) -> dict:
    """Esegue il Weber effect check."""
    if not config.get("check_weber", True):
        print("  [SKIP] Weber check disabilitato")
        return {}

    t0 = time.time()
    result = check_weber_effect(
        parquet_path=str(PARQUET_PATH),
        target_drug=config["target_drug"],
        api_key=config.get("openfda_api_key"),
        approval_year_override=config.get("weber_approval_override"),
    )
    print(f"  [WEBER ] Weber check completato in {time.time()-t0:.2f}s")
    weber_summary(result)
    return result


def _run_validation_and_weber_parallel(
    validated_union: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Esegue validazione label FDA e Weber check in parallelo.
    Entrambi fanno chiamate HTTP a openFDA — sono I/O-bound e indipendenti.
    Il guadagno è netto al primo run; i run successivi usano cache JSON locale
    e sono già sub-secondo indipendentemente dal parallelismo.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_label = executor.submit(_run_validation, validated_union, config)
        fut_weber = executor.submit(_run_weber, config)

        validated = fut_label.result()
        weber     = fut_weber.result()

    return validated, weber


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Run summary
# ─────────────────────────────────────────────────────────────────────────────

def build_run_summary(
    config:        dict,
    algo_results:  dict[str, pd.DataFrame | None],
    validated:     pd.DataFrame,
    weber:         dict,
    total_elapsed: float,
    ct_from_cache: bool = False,
) -> dict:
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
        "ct_from_cache":     ct_from_cache,          # flag utile per il dashboard
        "thresholds": {
            "fdr":  config["fdr_threshold"],
            "eb05": config["eb05_threshold"],
            "ic":   config["ic_threshold"],
        },
        "algorithm_results": algo_stats,
        "validation":        val_stats,
        "weber_risk":        weber.get("weber_risk"),
        "total_elapsed_s":   round(total_elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    config: dict,
    precomputed_ct: pd.DataFrame | None = None,
) -> dict:
    """
    Esegue la pipeline completa di signal detection.

    Parameters
    ----------
    config : dict
        Output di get_run_config() o build_config_from_ui().
    precomputed_ct : pd.DataFrame | None
        Se fornita, salta il calcolo della contingency table (DuckDB).
        Usare quando target_drug + where_extra + min_a non sono cambiati
        rispetto al run precedente (cache in appli.py via st.session_state).

    Returns
    -------
    dict con chiavi:
        config, contingency_table, prr, ror, bcpnn, mgps,
        validated, weber_check, run_summary
    """
    t_start = time.time()
    ct_from_cache = False

    print(f"\n{'='*55}")
    print(f"  Farmaco        : {config['target_drug']}")
    print(f"  Stratificazione: {config['where_extra'] or 'globale'}")
    print(f"  Algoritmi      : {config['algorithms']}")
    print(f"{'='*55}")

    # ── STEP 1: Contingency table ─────────────────────────────────────────
    if precomputed_ct is not None:
        ct = precomputed_ct
        ct_from_cache = True
        print(f"\n[CT    ] Riutilizzata dalla cache — {len(ct)} coppie (0.00s)")
    else:
        print("\n[CT    ] Costruzione contingency table...")
        t0 = time.time()
        ct = _build_ct(config)
        print(f"[CT    ] {len(ct)} coppie trovate — {time.time()-t0:.2f}s")

    if len(ct) == 0:
        print(f"  [ERR] Nessuna coppia trovata per '{config['target_drug']}'.")
        return {"config": config, "error": "no_data"}

    qc_contingency_table(ct, label=config["target_drug"])

    # ── STEP 2: Algoritmi in parallelo ───────────────────────────────────
    print("\n[ALGOS ] Esecuzione parallela algoritmi...")
    t0 = time.time()
    algo_results = _run_algorithms_parallel(ct, config)
    print(f"[ALGOS ] Completati in {time.time()-t0:.2f}s (parallelo)")

    # ── STEP 3: Unione segnali positivi ──────────────────────────────────
    validated_union = build_validated_union(algo_results)
    print(f"\n[UNION ] {len(validated_union)} segnali positivi totali (tutte le sorgenti)")

    # ── STEP 4+5: Validazione label + Weber in parallelo ─────────────────
    print("\n[API   ] Validazione label FDA + Weber check in parallelo...")
    t0 = time.time()
    validated, weber = _run_validation_and_weber_parallel(validated_union, config)
    print(f"[API   ] Completati in {time.time()-t0:.2f}s (parallelo)")

    # ── STEP 6: Summary ───────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    run_summary = build_run_summary(
        config=config,
        algo_results=algo_results,
        validated=validated,
        weber=weber,
        total_elapsed=total_elapsed,
        ct_from_cache=ct_from_cache,
    )

    print(f"\n{'='*55}")
    print(f"  TOTALE: {run_summary['total_elapsed_s']}s "
          f"({'CT dalla cache' if ct_from_cache else 'CT ricalcolata'})")
    print(f"{'='*55}\n")

    # ── Scrivi su disco per il pannello panoramico ────────────────────────
    SIGNALS_OUT = Path(__file__).resolve().parent / "data" / "signals_latest.parquet"
    if len(validated) > 0:
        validated.to_parquet(SIGNALS_OUT, index=False)
        print(f"  [OK] {SIGNALS_OUT.name} scritto ({len(validated)} righe)")

    return {
        "config":            config,
        "contingency_table": ct,
        **algo_results,
        "validated":         validated,
        "weber_check":       weber,
        "run_summary":       run_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

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