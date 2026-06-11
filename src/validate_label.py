"""
src/validate_label.py

Validazione dei segnali rilevati dagli algoritmi di disproportionality analysis
contro il bugiardino ufficiale FDA (openFDA drug/label API).

Per ogni coppia (drug, AE) rilevata come segnale positivo, classifica:
    KNOWN          → l'AE compare nel bugiardino (segnale già riconosciuto)
    POTENTIALLY_NEW → l'AE NON compare nel bugiardino (candidato a segnale nuovo)
    NO_LABEL        → nessun label trovato per quel farmaco su openFDA

Input atteso: DataFrame con colonne ae_name, product_name (output di signals.py).
Output: stesso DataFrame con colonne aggiuntive di validazione.
"""

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


BASE_URL = "https://api.fda.gov/drug/label.json"

# Sezioni del bugiardino rilevanti per la sicurezza, in ordine di priorità
SAFETY_SECTIONS = [
    "adverse_reactions",
    "warnings",
    "warnings_and_cautions",
    "boxed_warning",
    "precautions",
    "contraindications",
    "drug_interactions",
    "use_in_specific_populations",
]

# Path della cache locale dei label: evita chiamate API ridondanti tra run
LABEL_CACHE_PATH = Path("data/label_cache.json")


def _load_label_cache() -> dict:
    """Carica la cache dei label da disco, o restituisce un dizionario vuoto."""
    if LABEL_CACHE_PATH.exists():
        return json.loads(LABEL_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_label_cache(cache: dict) -> None:
    """Persiste la cache dei label su disco."""
    LABEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABEL_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def fetch_label(
        drug_name: str,
        api_key:   Optional[str] = None,
        cache:     Optional[dict] = None,
) -> Optional[dict]:
    """
    Recupera il bugiardino più recente di un farmaco tramite openFDA drug/label API.
    Cerca prima per nome generico, poi per nome commerciale.

    Usa la cache in memoria (dict passato come argomento) per evitare chiamate
    duplicate nello stesso run. La cache su disco è gestita da validate_signals().

    Parameters
    ----------
    drug_name : nome del farmaco in qualsiasi case (viene lowercasato per la query)
    api_key   : chiave API openFDA opzionale (240 req/min senza, 1000 con)
    cache     : dizionario in memoria { drug_name: response_dict | None }

    Returns
    -------
    dict con i risultati openFDA, oppure None se non trovato o errore.
    """
    key = drug_name.lower().strip()

    if cache is not None and key in cache:
        return cache[key]

    params = {
        "search": (
            f'openfda.generic_name:"{key}" '
            f'OR openfda.brand_name:"{key}"'
        ),
        "limit": 1,
    }
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        total = data["meta"]["results"]["total"]
        print(f"  [API] '{drug_name}' → {total} label trovati")
        result = data if data.get("results") else None

    except requests.exceptions.HTTPError as e:
        status = getattr(resp, "status_code", "?")
        if status == 404:
            print(f"  [WARN] Nessun label trovato per '{drug_name}'")
        else:
            print(f"  [ERR] HTTP {status}: {e}")
        result = None

    except Exception as e:
        print(f"  [ERR] {drug_name}: {e}")
        result = None

    if cache is not None:
        cache[key] = result

    return result


def extract_safety_sections(label_record: dict) -> dict:
    """
    Estrae le sezioni di sicurezza da un record label openFDA.
    Ogni sezione è una lista di stringhe nell'API: le concatena e porta in lowercase
    per la ricerca testuale successiva.

    Returns { nome_sezione: testo_lowercase }
    """
    sections = {}
    for field in SAFETY_SECTIONS:
        content = label_record.get(field)
        if content:
            sections[field] = " ".join(content).lower()
    return sections


def search_ae_in_label(ae_term: str, sections: dict) -> dict:
    """
    Cerca un termine AE nelle sezioni di sicurezza del bugiardino con substring match.

    I termini MedDRA PT (es. "DIARRHOEA") vengono portati in lowercase per il match.
    La ricerca è volutamente permissiva: un match parziale è sufficiente per
    classificare il segnale come KNOWN (approccio conservativo).

    Returns
    -------
    dict con:
        found        : bool  — trovato in almeno una sezione
        sections_hit : list  — nomi delle sezioni dove compare
        snippets     : dict  — { sezione: estratto contestuale ±150 char }
    """
    ae_lower = ae_term.lower().strip()
    result   = {"found": False, "sections_hit": [], "snippets": {}}

    for section_name, text in sections.items():
        if ae_lower in text:
            result["found"] = True
            result["sections_hit"].append(section_name)
            # Estratto contestuale per ispezione manuale
            idx     = text.find(ae_lower)
            start   = max(0, idx - 150)
            end     = min(len(text), idx + len(ae_lower) + 150)
            result["snippets"][section_name] = "…" + text[start:end] + "…"

    return result


def validate_signals(
        signals_df:  pd.DataFrame,
        drug_name:   str,
        api_key:     Optional[str] = None,
        only_positive: bool = True,
) -> pd.DataFrame:
    """
    Valida i segnali rilevati dagli algoritmi contro il bugiardino openFDA.

    Aggiunge tre colonne al DataFrame di input:
        validation_status : "KNOWN" | "POTENTIALLY_NEW" | "NO_LABEL"
        sections_hit      : lista delle sezioni del bugiardino dove compare l'AE
        label_snippet     : estratto testuale contestuale (prima sezione trovata)

    Parameters
    ----------
    signals_df   : DataFrame output di compute_prr/ror/bcpnn/mgps con colonne
                   ae_name, product_name, signal_positive
    drug_name    : nome del farmaco target (usato per la query API)
    api_key      : chiave API openFDA opzionale
    only_positive: se True (default), valida solo le righe con signal_positive=True,
                   lascia NO_LABEL per le negative (risparmia chiamate API)

    Returns
    -------
    signals_df con le colonne di validazione aggiunte
    """
    df = signals_df.copy()

    # Inizializza le colonne di output
    df["validation_status"] = "NO_LABEL"
    df["sections_hit"]      = [[] for _ in range(len(df))]
    df["label_snippet"]     = ""

    # Carica cache da disco per non ripetere chiamate tra run diversi
    disk_cache = _load_label_cache()
    mem_cache  = {}  # cache in memoria per questo run

    # Prepopola la cache in memoria con i dati su disco
    for k, v in disk_cache.items():
        mem_cache[k] = v

    # Recupera il label una volta sola per il drug target
    label_data = fetch_label(drug_name, api_key=api_key, cache=mem_cache)

    if label_data is None or not label_data.get("results"):
        print(f"  [WARN] Nessun label disponibile per '{drug_name}', validazione saltata")
        _save_label_cache(mem_cache)
        return df

    label_record = label_data["results"][0]
    sections     = extract_safety_sections(label_record)

    # Filtra le righe da validare
    mask = df["signal_positive"] if only_positive else pd.Series(True, index=df.index)
    rows_to_validate = df[mask]

    print(f"  Validando {len(rows_to_validate)} segnali positivi contro label '{drug_name}'...")

    for idx, row in rows_to_validate.iterrows():
        ae_term    = str(row["ae_name"])
        ae_result  = search_ae_in_label(ae_term, sections)

        df.at[idx, "validation_status"] = "KNOWN" if ae_result["found"] else "POTENTIALLY_NEW"
        df.at[idx, "sections_hit"]      = ae_result["sections_hit"]

        # Salva lo snippet della prima sezione trovata (per il dashboard)
        if ae_result["sections_hit"]:
            first_sec = ae_result["sections_hit"][0]
            df.at[idx, "label_snippet"] = ae_result["snippets"][first_sec][:300]

        # Rate limit: 240 req/min senza API key. Il label è una sola chiamata,
        # ma il sleep protegge in caso di chiamate batch per più drug.
        time.sleep(0.05)

    # Persiste la cache aggiornata su disco
    _save_label_cache(mem_cache)

    return df


def validation_summary(df: pd.DataFrame) -> None:
    """
    Stampa un riepilogo della validazione: conteggi per status e top segnali nuovi.
    """
    if "validation_status" not in df.columns:
        print("  [WARN] Colonna validation_status non presente, esegui validate_signals() prima.")
        return

    validated = df[df["signal_positive"] == True]
    known     = (validated["validation_status"] == "KNOWN").sum()
    new       = (validated["validation_status"] == "POTENTIALLY_NEW").sum()
    no_label  = (validated["validation_status"] == "NO_LABEL").sum()

    print(f"\n  Segnali positivi validati : {len(validated)}")
    print(f"    KNOWN          : {known}  ({100*known/len(validated):.1f}%)" if len(validated) else "")
    print(f"    POTENTIALLY NEW: {new}  ({100*new/len(validated):.1f}%)"  if len(validated) else "")
    print(f"    NO LABEL       : {no_label}")

    # Mostra i top segnali potenzialmente nuovi ordinati per forza del segnale
    potentially_new = validated[validated["validation_status"] == "POTENTIALLY_NEW"] \
                    .drop_duplicates(subset=["ae_name"])
    if len(potentially_new) > 0:
        # Ordina per la prima colonna metrica disponibile tra quelle degli algoritmi
        sort_col = next(
            (c for c in ["EB05", "IC_lower_bound", "PRR", "ROR"] if c in potentially_new.columns),
            "events"
        )
        print(f"\n  Top segnali POTENTIALLY NEW (ordinati per {sort_col}):")
        top = potentially_new.nlargest(10, sort_col)[["ae_name", sort_col, "events"]]
        print(top.to_string(index=False))


if __name__ == "__main__":
    # Demo con i segnali di lapatinib dal paper Cerbito et al. 2026
    # In produzione questo blocco non viene eseguito: validate_signals()
    # viene chiamata da run_signals.py dopo compute_prr/ror/bcpnn/mgps.

    import sys
    sys.path.insert(0, "..")
    from src.contingency_table import build_contingency_table
    from src.signals import compute_mgps

    PARQUET = "data/faers_flat_deduped.parquet"
    DRUG    = "LAPATINIB"

    print(f"Costruzione contingency table per {DRUG}...")
    ct = build_contingency_table(PARQUET, DRUG, min_a=3)

    print("Calcolo segnali MGPS...")
    signals = compute_mgps(ct)

    print("Validazione contro label FDA...")
    validated = validate_signals(signals, drug_name="lapatinib")

    validation_summary(validated)

    # Salva il risultato completo
    out_path = "data/validated_signals_lapatinib.parquet"
    validated.to_parquet(out_path, index=False)
    print(f"\nSalvato: {out_path}")