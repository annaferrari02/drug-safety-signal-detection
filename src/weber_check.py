"""
src/weber_check.py

Detection del Weber effect per il farmaco target inserito dall'utente.

Il Weber effect descrive la tendenza a una sovra-segnalazione di eventi avversi
nelle fasi immediatamente successive all'approvazione di un farmaco, con un picco
tipicamente entro i primi 2 anni. Questo bias può gonfiare artificialmente le
metriche di disproportionality (PRR, ROR, BCPNN, MGPS), portando a segnali che
riflettono l'interesse mediatico/clinico per il nuovo farmaco più che un reale
eccesso di eventi avversi.

Riferimento: Overstreet et al. (2014), Drug Safety 37:283-294.
    https://doi.org/10.1007/s40264-014-0150-2
    NB: lo studio suggerisce che nel FAERS moderno il Weber effect classico è
    meno pronunciato rispetto a quanto descritto da Weber (1984), ma rimane
    un bias da documentare e comunicare all'utente.

Struttura parallela a validate_label.py:
    - fetch_approval_year()     ↔  fetch_label()
    - compute_weber_metrics()   ↔  extract_safety_sections()
    - check_weber_effect()      ↔  validate_signals()
    - weber_summary()           ↔  validation_summary()

Output (dict) restituito da check_weber_effect():
    {
        "drug":                  str,
        "approval_year":         int | None,
        "approval_source":       str,        # "openfda" | "manual" | "unknown"
        "years_in_dataset":      int,
        "quarters_analyzed":     int,
        "total_reports":         int,
        "early_phase_ratio":     float,      # % report nei primi 2 anni post-approval
        "quarterly_trend":       list[dict], # serie temporale per grafici dashboard
        "trend_slope":           float,      # pendenza regressione lineare (report/quarter)
        "peak_quarter_offset":   int | None, # quarter con più report (0 = primo quarter)
        "weber_risk":            "LOW" | "MODERATE" | "HIGH",
        "risk_reasons":          list[str],  # motivazioni leggibili per la dashboard
        "warning_message":       str,
        "check_timestamp":       str,
    }

Integrazione in run_signals.py:
    from src.weber_check import check_weber_effect, weber_summary

    weber = check_weber_effect(
        parquet_path=str(PARQUET_PATH),
        target_drug=config["target_drug"],
        api_key=config.get("openfda_api_key"),
    )
    weber_summary(weber)

    # Il dict viene aggiunto al risultato finale di run_pipeline():
    return { ..., "weber_check": weber }
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import requests

# ── Path assoluto per la cache (coerente con validate_label.py) ─────────────
APPROVAL_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "approval_cache.json"

# Endpoint openFDA per le approvazioni (drugsatfda)
DRUGSFDA_URL = "https://api.fda.gov/drug/drugsfda.json"

# ── Soglie per la classificazione del rischio ────────────────────────────────
# Derivate dalla letteratura e dalla logica del Weber effect:
#
#   early_phase_ratio: proporzione di report nei primi 2 anni post-approvazione
#       >0.55 → sovra-rappresentazione early-phase sospetta
#       >0.70 → forte concentrazione, alta probabilità di Weber effect
#
#   years_in_dataset: anni di dati disponibili per il farmaco
#       <3  → dataset troppo corto per distinguere trend da rumore
#       <5  → finestra temporale limitata, cauto
#
#   peak_quarter_offset: quarter (dalla approvazione) con il picco di report
#       0-3 → picco nel primo anno: classico Weber
#       4-7 → picco nel secondo anno: Weber moderato
#
#   trend_slope_normalized: pendenza della regressione dopo il picco
#       fortemente negativa → declino rapido dopo il picco (Weber classico)

THRESHOLDS = {
    "early_ratio_high":     0.70,
    "early_ratio_moderate": 0.50,
    "min_years_low":        5,
    "min_years_moderate":   3,
    "peak_window_high":     4,   # quarter 0-3 = primo anno
    "peak_window_moderate": 8,   # quarter 4-7 = secondo anno
    "min_reports":          30,  # sotto questa soglia i ratio sono instabili
}


# ── Cache approvazioni ───────────────────────────────────────────────────────

def _load_approval_cache() -> dict:
    if APPROVAL_CACHE_PATH.exists():
        return json.loads(APPROVAL_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_approval_cache(cache: dict) -> None:
    APPROVAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    APPROVAL_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Fetch anno di approvazione FDA ──────────────────────────────────────────

def fetch_approval_year(
        drug_name: str,
        api_key:   Optional[str] = None,
        cache:     Optional[dict] = None,
) -> tuple[Optional[int], str]:
    """
    Recupera l'anno della prima approvazione FDA per un farmaco tramite
    l'endpoint openFDA drugsatfda (drugs@FDA database).

    Cerca la submission di tipo ORIG (originale) con status AP (approved)
    e prende la data più antica tra quelle disponibili.

    Parameters
    ----------
    drug_name : nome del farmaco (qualsiasi case; viene lowercasato per la query)
    api_key   : chiave API openFDA opzionale
    cache     : dizionario in memoria { drug_name: (year, source) | None }

    Returns
    -------
    (approval_year: int | None, source: str)
        source è "openfda" se trovato via API, "unknown" altrimenti.
    """
    key = drug_name.lower().strip()

    if cache is not None and key in cache:
        cached = cache[key]
        return (cached["year"], cached["source"])

    params = {
        "search": (
            f'openfda.generic_name:"{key}" '
            f'OR openfda.brand_name:"{key}"'
        ),
        "limit": 5,
    }
    if api_key:
        params["api_key"] = api_key

    approval_year  = None
    source         = "unknown"

    try:
        resp = requests.get(DRUGSFDA_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        earliest = None
        for product in data.get("results", []):
            for sub in product.get("submissions", []):
                # Cerca approvazioni originali (non supplementari)
                if sub.get("submission_type") != "ORIG":
                    continue
                if sub.get("submission_status") != "AP":
                    continue
                date_str = sub.get("submission_status_date", "")
                if len(date_str) >= 4:
                    try:
                        year = int(date_str[:4])
                        if earliest is None or year < earliest:
                            earliest = year
                    except ValueError:
                        continue

        if earliest is not None:
            approval_year = earliest
            source        = "openfda"
            print(f"  [API] '{drug_name}' → approvazione FDA: {approval_year}")
        else:
            print(f"  [WARN] Anno di approvazione non trovato per '{drug_name}'")

    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "?")
        print(f"  [ERR] drugsatfda HTTP {status} per '{drug_name}': {e}")
    except Exception as e:
        print(f"  [ERR] fetch_approval_year '{drug_name}': {e}")

    if cache is not None:
        cache[key] = {"year": approval_year, "source": source}

    return (approval_year, source)


# ── Analisi temporale sul Parquet ────────────────────────────────────────────

def compute_weber_metrics(
        parquet_path:  str,
        target_drug:   str,
        approval_year: Optional[int],
) -> dict:
    """
    Calcola le metriche temporali di distribuzione dei report FAERS per il
    farmaco target, necessarie alla classificazione del rischio Weber.

    Usa DuckDB direttamente sul Parquet senza caricare dati in pandas,
    coerentemente con il resto della pipeline.

    Parameters
    ----------
    parquet_path  : path al Parquet deduplicato
    target_drug   : nome farmaco in uppercase
    approval_year : anno di approvazione FDA (None se non disponibile)

    Returns
    -------
    dict con:
        total_reports        : int
        years_in_dataset     : int   — span anni min→max nel dataset
        first_year           : int
        last_year            : int
        quarterly_counts     : list[dict]  — { year, quarter, reports, quarter_offset }
        early_phase_ratio    : float | None
        peak_quarter_offset  : int | None
        trend_slope          : float | None  — da regressione lineare semplice
    """
    con = duckdb.connect()
    p   = parquet_path

    # Totale report e span temporale per il farmaco
    totals = con.execute(f"""
        SELECT
            COUNT(DISTINCT safetyreportid)      AS total_reports,
            MIN(receive_year)                   AS first_year,
            MAX(receive_year)                   AS last_year
        FROM '{p}'
        WHERE drug_name = '{target_drug}'
          AND receive_year IS NOT NULL
    """).fetchone()

    if totals is None or totals[0] == 0:
        con.close()
        return {
            "total_reports":     0,
            "years_in_dataset":  0,
            "first_year":        None,
            "last_year":         None,
            "quarterly_counts":  [],
            "early_phase_ratio": None,
            "peak_quarter_offset": None,
            "trend_slope":       None,
        }

    total_reports = int(totals[0])
    first_year    = int(totals[1])
    last_year     = int(totals[2])
    years_in_dataset = last_year - first_year + 1

    # Distribuzione per quarter (report unici per trimestre)
    quarterly_raw = con.execute(f"""
        SELECT
            receive_year                        AS year,
            CAST(receive_quarter[-1] AS INT)    AS q,
            COUNT(DISTINCT safetyreportid)      AS reports
        FROM '{p}'
        WHERE drug_name = '{target_drug}'
          AND receive_year IS NOT NULL
          AND receive_quarter IS NOT NULL
        GROUP BY receive_year, receive_quarter
        ORDER BY receive_year, q
    """).df()

    con.close()

    # Calcola quarter_offset dalla data di approvazione
    # Se approval_year non disponibile, usa first_year come riferimento
    ref_year = approval_year if approval_year is not None else first_year

    quarterly_counts = []
    for _, row in quarterly_raw.iterrows():
        year     = int(row["year"])
        q        = int(row["q"])
        reports  = int(row["reports"])
        # Offset in quarter dalla prima quarter del ref_year
        offset   = (year - ref_year) * 4 + (q - 1)
        quarterly_counts.append({
            "year":           year,
            "quarter":        q,
            "quarter_label":  f"{year}Q{q}",
            "reports":        reports,
            "quarter_offset": offset,
        })

    # Early-phase ratio: report nei primi 2 anni post-approvazione
    # (offset 0-7 = primi 8 quarter = 2 anni)
    early_phase_ratio = None
    if approval_year is not None and total_reports > 0:
        early_reports = sum(
            e["reports"] for e in quarterly_counts
            if 0 <= e["quarter_offset"] <= 7
        )
        early_phase_ratio = early_reports / total_reports

    # Quarter con il picco di segnalazione
    peak_quarter_offset = None
    if quarterly_counts:
        peak_entry = max(quarterly_counts, key=lambda x: x["reports"])
        peak_quarter_offset = peak_entry["quarter_offset"]

    # Regressione lineare semplice: reports ~ quarter_offset
    # per quantificare il trend complessivo (slope negativa = declino nel tempo)
    trend_slope = None
    if len(quarterly_counts) >= 4:
        xs = [e["quarter_offset"] for e in quarterly_counts]
        ys = [e["reports"]        for e in quarterly_counts]
        n  = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num    = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den    = sum((x - mean_x) ** 2 for x in xs)
        if den > 0:
            trend_slope = round(num / den, 4)

    return {
        "total_reports":       total_reports,
        "years_in_dataset":    years_in_dataset,
        "first_year":          first_year,
        "last_year":           last_year,
        "quarterly_counts":    quarterly_counts,
        "early_phase_ratio":   early_phase_ratio,
        "peak_quarter_offset": peak_quarter_offset,
        "trend_slope":         trend_slope,
    }


# ── Classificazione del rischio ──────────────────────────────────────────────

def _classify_weber_risk(
        metrics:       dict,
        approval_year: Optional[int],
) -> tuple[str, list[str]]:
    """
    Classifica il rischio Weber in tre livelli sulla base delle metriche
    calcolate da compute_weber_metrics().

    Logica multi-criterio: ogni criterio contribuisce a un punteggio.
    Il livello finale è il massimo tra tutti i criteri valutati.

    Criteri (in ordine di peso):
        1. early_phase_ratio   — peso maggiore: misura diretta della concentrazione
        2. years_in_dataset    — dato troppo corto → incertezza intrinseca
        3. peak_quarter_offset — picco nel primo anno è il marker classico di Weber
        4. approval_year None  — impossibile calcolare i ratio senza riferimento

    Returns
    -------
    (risk_level: str, reasons: list[str])
    """
    T       = THRESHOLDS
    reasons = []
    scores  = []  # "LOW", "MODERATE", "HIGH" per ogni criterio

    total_reports       = metrics["total_reports"]
    years_in_dataset    = metrics["years_in_dataset"]
    early_phase_ratio   = metrics["early_phase_ratio"]
    peak_quarter_offset = metrics["peak_quarter_offset"]

    # ── Criterio 0: dati insufficienti ───────────────────────────────────────
    if total_reports < T["min_reports"]:
        reasons.append(
            f"Soli {total_reports} report disponibili: i ratio sono instabili "
            f"con campioni ridotti (soglia minima: {T['min_reports']})"
        )
        scores.append("HIGH")

    # ── Criterio 1: early-phase ratio ────────────────────────────────────────
    if early_phase_ratio is not None:
        pct = f"{early_phase_ratio:.0%}"
        if early_phase_ratio >= T["early_ratio_high"]:
            reasons.append(
                f"{pct} dei report è concentrato nei primi 2 anni post-approvazione "
                f"(soglia HIGH: ≥{T['early_ratio_high']:.0%}): forte indicatore di Weber effect"
            )
            scores.append("HIGH")
        elif early_phase_ratio >= T["early_ratio_moderate"]:
            reasons.append(
                f"{pct} dei report è concentrato nei primi 2 anni post-approvazione "
                f"(soglia MODERATE: ≥{T['early_ratio_moderate']:.0%}): potenziale Weber effect"
            )
            scores.append("MODERATE")
        else:
            reasons.append(
                f"{pct} dei report nei primi 2 anni: distribuzione temporale accettabile"
            )
            scores.append("LOW")
    else:
        # Anno di approvazione sconosciuto → impossibile calcolare il ratio
        reasons.append(
            "Anno di approvazione FDA non disponibile: impossibile calcolare "
            "l'early-phase ratio. Interpretare i segnali con cautela."
        )
        scores.append("MODERATE")

    # ── Criterio 2: finestra temporale disponibile ───────────────────────────
    if years_in_dataset < T["min_years_moderate"]:
        reasons.append(
            f"Solo {years_in_dataset} anno/i di dati disponibili: "
            f"finestra troppo corta per distinguere Weber effect da trend reale "
            f"(minimo consigliato: {T['min_years_moderate']} anni)"
        )
        scores.append("HIGH")
    elif years_in_dataset < T["min_years_low"]:
        reasons.append(
            f"{years_in_dataset} anni di dati: finestra temporale limitata "
            f"(consigliati ≥{T['min_years_low']} anni per stabilità)"
        )
        scores.append("MODERATE")
    else:
        reasons.append(
            f"{years_in_dataset} anni di dati: finestra temporale adeguata"
        )
        scores.append("LOW")

    # ── Criterio 3: posizione del picco di segnalazione ──────────────────────
    if peak_quarter_offset is not None and approval_year is not None:
        if 0 <= peak_quarter_offset < T["peak_window_high"]:
            reasons.append(
                f"Il picco di segnalazione cade nel quarter {peak_quarter_offset + 1} "
                f"dalla approvazione (primo anno): firma classica del Weber effect"
            )
            scores.append("HIGH")
        elif T["peak_window_high"] <= peak_quarter_offset < T["peak_window_moderate"]:
            reasons.append(
                f"Il picco di segnalazione cade nel quarter {peak_quarter_offset + 1} "
                f"dalla approvazione (secondo anno): possibile Weber effect moderato"
            )
            scores.append("MODERATE")
        else:
            reasons.append(
                f"Il picco di segnalazione cade nel quarter {peak_quarter_offset + 1} "
                f"dalla approvazione: distribuzione compatibile con segnalazione stabile"
            )
            scores.append("LOW")

    # ── Livello finale: il più alto tra tutti i criteri ──────────────────────
    priority = {"HIGH": 2, "MODERATE": 1, "LOW": 0}
    final_risk = max(scores, key=lambda s: priority[s]) if scores else "LOW"

    return final_risk, reasons


def _build_warning_message(risk: str, drug_name: str) -> str:
    messages = {
        "HIGH": (
            f"⚠️  ATTENZIONE — Rischio Weber effect ALTO per {drug_name}. "
            "I segnali rilevati potrebbero essere sovrastimati a causa di una "
            "sovra-segnalazione nelle fasi iniziali post-approvazione. "
            "Interpretare i risultati con estrema cautela e confrontarli con "
            "letteratura clinica indipendente."
        ),
        "MODERATE": (
            f"⚡ CAUTELA — Rischio Weber effect MODERATO per {drug_name}. "
            "La distribuzione temporale dei report presenta alcune caratteristiche "
            "compatibili con il Weber effect. I segnali potrebbero essere "
            "parzialmente influenzati da bias di segnalazione early-phase."
        ),
        "LOW": (
            f"✅ Rischio Weber effect BASSO per {drug_name}. "
            "La distribuzione temporale dei report appare stabile. "
            "I segnali rilevati non mostrano evidenti distorsioni legate "
            "alla sovra-segnalazione post-approvazione."
        ),
    }
    return messages[risk]


# ── Funzione principale ──────────────────────────────────────────────────────

def check_weber_effect(
        parquet_path: str,
        target_drug:  str,
        api_key:      Optional[str] = None,
        approval_year_override: Optional[int] = None,
) -> dict:
    """
    Esegue il check completo del Weber effect per il farmaco target.

    Flusso:
        1. Recupera anno di approvazione FDA via openFDA drugsatfda
           (con cache su disco per evitare chiamate ripetute)
        2. Calcola le metriche temporali dal Parquet via DuckDB
        3. Classifica il rischio in LOW / MODERATE / HIGH
        4. Restituisce un dict completo per la dashboard

    Parameters
    ----------
    parquet_path           : path al Parquet deduplicato
    target_drug            : nome farmaco in uppercase (es. "LAPATINIB")
    api_key                : chiave API openFDA opzionale
    approval_year_override : anno di approvazione hardcoded (bypassa l'API,
                             utile per test o farmaci non su drugsatfda)

    Returns
    -------
    dict — struttura documentata nel modulo docstring
    """
    drug_name = target_drug.upper().strip()
    print(f"\n=== Weber effect check: {drug_name} ===")

    # ── Step 1: anno di approvazione ─────────────────────────────────────────
    disk_cache = _load_approval_cache()
    mem_cache  = {k: v for k, v in disk_cache.items()}

    if approval_year_override is not None:
        approval_year  = approval_year_override
        approval_source = "manual"
        print(f"  [CONFIG] Anno di approvazione override: {approval_year}")
    else:
        approval_year, approval_source = fetch_approval_year(
            drug_name=drug_name,
            api_key=api_key,
            cache=mem_cache,
        )
        _save_approval_cache(mem_cache)

    # ── Step 2: metriche temporali ───────────────────────────────────────────
    metrics = compute_weber_metrics(
        parquet_path=parquet_path,
        target_drug=drug_name,
        approval_year=approval_year,
    )

    if metrics["total_reports"] == 0:
        return {
            "drug":                drug_name,
            "approval_year":       approval_year,
            "approval_source":     approval_source,
            "years_in_dataset":    0,
            "quarters_analyzed":   0,
            "total_reports":       0,
            "early_phase_ratio":   None,
            "quarterly_trend":     [],
            "trend_slope":         None,
            "peak_quarter_offset": None,
            "weber_risk":          "HIGH",
            "risk_reasons":        ["Nessun report trovato nel dataset per questo farmaco."],
            "warning_message":     f"⚠️  Nessun dato disponibile per {drug_name} nel dataset.",
            "check_timestamp":     datetime.now().isoformat(),
        }

    # ── Step 3: classificazione ──────────────────────────────────────────────
    weber_risk, risk_reasons = _classify_weber_risk(
        metrics=metrics,
        approval_year=approval_year,
    )

    warning_message = _build_warning_message(weber_risk, drug_name)

    # ── Step 4: costruzione output ───────────────────────────────────────────
    result = {
        "drug":                drug_name,
        "approval_year":       approval_year,
        "approval_source":     approval_source,
        "years_in_dataset":    metrics["years_in_dataset"],
        "first_year":          metrics["first_year"],
        "last_year":           metrics["last_year"],
        "quarters_analyzed":   len(metrics["quarterly_counts"]),
        "total_reports":       metrics["total_reports"],
        "early_phase_ratio":   (
            round(metrics["early_phase_ratio"], 4)
            if metrics["early_phase_ratio"] is not None else None
        ),
        "quarterly_trend":     metrics["quarterly_counts"],  # lista completa per grafici
        "trend_slope":         metrics["trend_slope"],
        "peak_quarter_offset": metrics["peak_quarter_offset"],
        "weber_risk":          weber_risk,
        "risk_reasons":        risk_reasons,
        "warning_message":     warning_message,
        "check_timestamp":     datetime.now().isoformat(),
    }

    return result


# ── Summary stampabile (equivalente di validation_summary) ──────────────────

def weber_summary(result: dict) -> None:
    """
    Stampa un riepilogo leggibile del Weber effect check.
    Chiamata da run_signals.py dopo check_weber_effect().
    """
    if not result:
        print("  [WARN] Nessun risultato Weber check disponibile.")
        return


    print(f"\n  Farmaco         : {result['drug']}")
    print(f"  Approvazione FDA: {result['approval_year'] or 'non disponibile'} "
          f"({result['approval_source']})")
    print(f"  Dati disponibili: {result['first_year']}–{result['last_year']} "
          f"({result['years_in_dataset']} anni, {result['quarters_analyzed']} quarter)")
    print(f"  Report totali   : {result['total_reports']}")
    if result["early_phase_ratio"] is not None:
        print(f"  Early-phase ratio: {result['early_phase_ratio']:.1%} "
              f"(primi 2 anni post-approvazione)")
    print(f"\n  {icon} Weber risk: {result['weber_risk']}")
    for reason in result["risk_reasons"]:
        print(f"    • {reason}")
    print(f"\n  {result['warning_message']}")


# ── Esecuzione diretta (test / debug) ────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    PARQUET = str(Path(__file__).resolve().parent.parent / "data" / "faers_flat_deduped.parquet")
    DRUG    = sys.argv[1].upper() if len(sys.argv) > 1 else "LAPATINIB"

    result = check_weber_effect(
        parquet_path=PARQUET,
        target_drug=DRUG,
    )
    weber_summary(result)

    # Salva il risultato per ispezione
    out = Path(__file__).resolve().parent.parent / "data" / f"weber_{DRUG.lower()}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\nSalvato: {out}")