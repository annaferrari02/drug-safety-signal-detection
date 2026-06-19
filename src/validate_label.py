"""
src/validate_label.py

Validazione dei segnali rilevati dagli algoritmi di disproportionality analysis
contro il bugiardino ufficiale FDA (openFDA drug/label API).

Per ogni coppia (drug, AE) rilevata come segnale positivo classifica:
    KNOWN          → l'AE compare nel bugiardino: segnale già riconosciuto ufficialmente
    POTENTIALLY_NEW → l'AE NON compare nel bugiardino: evidenza solo statistica,
                      nessuna base scientifica consolidata nella letteratura regolatoria
    NO_LABEL        → nessun label trovato su openFDA per quel farmaco

Tutti i segnali (positivi e negativi) vengono classificati, non solo i positivi.
I segnali negativi ricevono sempre NO_LABEL (non è rilevante validarli).

Correzioni rispetto alla versione precedente:
    - word-boundary match (re.search) invece di substring match per evitare
      falsi KNOWN (es. "pain" che matcha dentro "abdominal pain")
    - dizionario di normalizzazione UK→US per i termini MedDRA PT
      (es. DIARRHOEA → diarrhea, HAEMORRHAGE → hemorrhage)
    - fetch dei primi 3 label invece di 1 solo, unione delle sezioni di sicurezza
    - LABEL_CACHE_PATH usa Path(__file__) per funzionare da qualsiasi working dir
    - only_positive rimosso: tutti i segnali positivi vengono classificati,
      i negativi restano NO_LABEL per costruzione
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


BASE_URL = "https://api.fda.gov/drug/label.json"

# Path assoluto indipendente dalla working directory
LABEL_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "label_cache.json"

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

# Mapping UK→US per i termini MedDRA PT più comuni in farmacovigilanza.
# I label FDA sono in inglese americano; i PT di FAERS/MedDRA sono in inglese britannico.
# Senza questo mapping termini come DIARRHOEA vengono classificati erroneamente
# come POTENTIALLY_NEW perché non matchano "diarrhea" nel label.
UK_TO_US: dict[str, str] = {
    "diarrhoea":            "diarrhea",
    "haemorrhage":          "hemorrhage",
    "haemorrhagic":         "hemorrhagic",
    "haematuria":           "hematuria",
    "haemoglobin":          "hemoglobin",
    "haemolysis":           "hemolysis",
    "haemolytic":           "hemolytic",
    "haematopoietic":       "hematopoietic",
    "oedema":               "edema",
    "anaemia":              "anemia",
    "anaesthetic":          "anesthetic",
    "dyspnoea":             "dyspnea",
    "hypokalaemia":         "hypokalemia",
    "hyperkalaemia":        "hyperkalemia",
    "hyponatraemia":        "hyponatremia",
    "hypernatraemia":       "hypernatremia",
    "hypocalcaemia":        "hypocalcemia",
    "hypercalcaemia":       "hypercalcemia",
    "foetal":               "fetal",
    "foetus":               "fetus",
    "tumour":               "tumor",
    "behaviour":            "behavior",
    "colour":               "color",
    "labour":               "labor",
    "leukopenia":           "leukopenia",  # già uguale, per completezza
    "neutropenia":          "neutropenia",
    "thrombocytopenia":     "thrombocytopenia",
    "ischaemia":            "ischemia",
    "ischaemic":            "ischemic",
    "oesophageal":          "esophageal",
    "oesophagus":           "esophagus",
    "diarrhoeal":           "diarrheal",
    "paediatric":           "pediatric",
    "gynaecological":       "gynecological",
    "orthopaedic":          "orthopedic",
}


def _normalize_ae_term(ae_term: str) -> list[str]:
    """
    Restituisce lista di varianti del termine AE da cercare nel label.
    Include il termine originale (lowercase) e la variante US se esiste nel mapping.
    Entrambe le varianti vengono cercate con OR logico: basta un match per KNOWN.
    """
    lower = ae_term.lower().strip()
    variants = [lower]
    if lower in UK_TO_US:
        variants.append(UK_TO_US[lower])
    # Gestisce anche composti: "peripheral oedema" → "peripheral edema"
    us_variant = lower
    for uk, us in UK_TO_US.items():
        if uk in us_variant:
            us_variant = us_variant.replace(uk, us)
    if us_variant != lower and us_variant not in variants:
        variants.append(us_variant)
    return variants


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
    Recupera i primi 3 bugiardini per un farmaco tramite openFDA drug/label API.
    Cerca per nome generico OR nome commerciale.

    Recuperare più label (limit=3) garantisce che le sezioni di sicurezza
    siano più complete, specialmente per farmaci con più formulazioni.

    Parameters
    ----------
    drug_name : nome del farmaco (qualsiasi case; viene lowercasato per la query)
    api_key   : chiave API openFDA opzionale (240 req/min senza, 1000 con)
    cache     : dizionario in memoria { drug_name: response_dict | None }

    Returns
    -------
    dict con i risultati openFDA (fino a 3 label), oppure None se non trovato.
    """
    key = drug_name.lower().strip()

    if cache is not None and key in cache:
        return cache[key]

    params = {
        "search": (
            f'openfda.generic_name:"{key}" '
            f'OR openfda.brand_name:"{key}"'
        ),
        "limit": 3,  # unione di più label → sezioni più complete
    }
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        total = data["meta"]["results"]["total"]
        print(f"  [API] '{drug_name}' → {total} label trovati, uso i primi {len(data.get('results', []))}")
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


def extract_safety_sections(label_records: list[dict]) -> dict:
    """
    Estrae e unisce le sezioni di sicurezza da una lista di record label openFDA.
    Unire più record garantisce copertura maggiore per farmaci con più formulazioni.

    Per ogni sezione, il testo di tutti i record viene concatenato e portato
    in lowercase per la ricerca testuale successiva.

    Parameters
    ----------
    label_records : lista di record label (tipicamente 1-3 risultati dall'API)

    Returns
    -------
    { nome_sezione: testo_unificato_lowercase }
    """
    sections: dict[str, str] = {}
    for record in label_records:
        for field in SAFETY_SECTIONS:
            content = record.get(field)
            if content:
                new_text = " ".join(content).lower()
                if field in sections:
                    sections[field] += " " + new_text
                else:
                    sections[field] = new_text
    return sections


def search_ae_in_label(ae_term: str, sections: dict) -> dict:
    """
    Cerca un termine AE nelle sezioni di sicurezza del bugiardino.

    Usa word-boundary match (re.search con \\b) invece di substring match per
    evitare falsi KNOWN: il termine "pain" non deve matchare dentro "abdominal pain",
    ma "abdominal pain" deve matchare se cercato esplicitamente.

    Cerca tutte le varianti UK/US del termine (via _normalize_ae_term): basta
    un match in una variante per classificare come KNOWN.

    Parameters
    ----------
    ae_term  : termine MedDRA PT (es. "DIARRHOEA", "PERIPHERAL OEDEMA")
    sections : dict { nome_sezione: testo_lowercase } da extract_safety_sections()

    Returns
    -------
    dict con:
        found        : bool  — trovato in almeno una sezione con almeno una variante
        sections_hit : list  — nomi delle sezioni dove compare
        snippets     : dict  — { sezione: estratto contestuale ±150 char }
        matched_term : str   — variante effettivamente trovata (UK o US)
    """
    variants = _normalize_ae_term(ae_term)
    result   = {"found": False, "sections_hit": [], "snippets": {}, "matched_term": ""}

    for section_name, text in sections.items():
        for variant in variants:
            pattern = re.compile(r'\b' + re.escape(variant) + r'\b')
            match = pattern.search(text)
            if match and section_name not in result["sections_hit"]:
                result["found"]       = True
                result["matched_term"] = variant
                result["sections_hit"].append(section_name)
                # Estratto contestuale per ispezione nella dashboard
                idx   = match.start()
                start = max(0, idx - 150)
                end   = min(len(text), idx + len(variant) + 150)
                result["snippets"][section_name] = "…" + text[start:end] + "…"
                break  # una variante trovata in questa sezione è sufficiente

    return result


def validate_signals(
        signals_df: pd.DataFrame,
        drug_name:  str,
        api_key:    Optional[str] = None,
) -> pd.DataFrame:
    """
    Valida tutti i segnali positivi contro il bugiardino openFDA e assegna
    un'etichetta a ogni coppia (drug, AE).

    Logica di classificazione:
        signal_positive=True  + AE trovato nel label  → KNOWN
            Il segnale è già riconosciuto ufficialmente dalla FDA.
            C'è base scientifica e regolatoria consolidata.

        signal_positive=True  + AE NON trovato        → POTENTIALLY_NEW
            Il segnale emerge solo dai dati statistici (FAERS).
            Non ha ancora una base scientifica ufficiale: va trattato con
            cautela e richiederebbe ulteriore indagine clinica.

        signal_positive=False  (qualsiasi)             → NO_SIGNAL
            Nessun segnale rilevato dagli algoritmi: non pertinente alla
            classificazione di sicurezza.

    Nota: i segnali negativi NON vengono validati contro il label perché
    l'assenza di segnale statistico rende irrilevante la presenza nel bugiardino.

    Parameters
    ----------
    signals_df : DataFrame con colonne ae_name, signal_positive (output di signals.py)
    drug_name  : nome del farmaco target (usato per la query API)
    api_key    : chiave API openFDA opzionale

    Returns
    -------
    signals_df con colonne aggiuntive:
        validation_status : "KNOWN" | "POTENTIALLY_NEW" | "NO_SIGNAL"
        sections_hit      : lista sezioni del label dove compare l'AE (solo KNOWN)
        label_snippet     : estratto testuale contestuale, prima sezione (solo KNOWN)
        matched_term      : variante UK/US effettivamente trovata nel label
    """
    df = signals_df.copy()

    # Inizializza le colonne di output
    df["validation_status"] = "NO_SIGNAL"
    df["sections_hit"]      = [[] for _ in range(len(df))]
    df["label_snippet"]     = ""
    df["matched_term"]      = ""

    # Carica cache da disco
    disk_cache = _load_label_cache()
    mem_cache  = {k: v for k, v in disk_cache.items()}

    # Recupera i label per il drug target
    label_data = fetch_label(drug_name, api_key=api_key, cache=mem_cache)

    if label_data is None or not label_data.get("results"):
        print(f"  [WARN] Nessun label disponibile per '{drug_name}', validazione saltata")
        # I segnali positivi restano senza classificazione: li marchiamo esplicitamente
        df.loc[df["signal_positive"] == True, "validation_status"] = "NO_LABEL"
        _save_label_cache(mem_cache)
        return df

    # Unisce le sezioni di sicurezza di tutti i label recuperati
    label_records = label_data["results"]
    sections      = extract_safety_sections(label_records)

    if not sections:
        print(f"  [WARN] Label trovato per '{drug_name}' ma nessuna sezione di sicurezza estratta")
        df.loc[df["signal_positive"] == True, "validation_status"] = "NO_LABEL"
        _save_label_cache(mem_cache)
        return df

    # Valida solo i segnali positivi
    positive_mask = df["signal_positive"] == True
    rows_to_validate = df[positive_mask]

    print(f"  Validando {len(rows_to_validate)} segnali positivi contro label '{drug_name}'...")
    print(f"  Sezioni disponibili: {list(sections.keys())}")

    for idx, row in rows_to_validate.iterrows():
        ae_term   = str(row["ae_name"])
        ae_result = search_ae_in_label(ae_term, sections)

        if ae_result["found"]:
            df.at[idx, "validation_status"] = "KNOWN"
            df.at[idx, "sections_hit"]      = ae_result["sections_hit"]
            df.at[idx, "matched_term"]      = ae_result["matched_term"]
            if ae_result["sections_hit"]:
                first_sec = ae_result["sections_hit"][0]
                df.at[idx, "label_snippet"] = ae_result["snippets"][first_sec][:300]
        else:
            df.at[idx, "validation_status"] = "POTENTIALLY_NEW"

    _save_label_cache(mem_cache)
    return df


def validation_summary(df: pd.DataFrame) -> None:
    """
    Stampa un riepilogo della validazione con conteggi per status
    e lista dei top segnali potenzialmente nuovi.
    """
    if "validation_status" not in df.columns:
        print("  [WARN] Colonna validation_status non presente, esegui validate_signals() prima.")
        return

    positive = df[df["signal_positive"] == True]
    if len(positive) == 0:
        print("  Nessun segnale positivo da validare.")
        return

    known     = (positive["validation_status"] == "KNOWN").sum()
    new       = (positive["validation_status"] == "POTENTIALLY_NEW").sum()
    no_label  = (positive["validation_status"] == "NO_LABEL").sum()
    total_pos = len(positive)

    print(f"\n  Segnali positivi totali   : {total_pos}")
    print(f"    KNOWN (già nel label)   : {known}  ({100*known/total_pos:.1f}%)")
    print(f"    POTENTIALLY NEW         : {new}  ({100*new/total_pos:.1f}%)")
    if no_label:
        print(f"    NO_LABEL (API fallita)  : {no_label}")

    potentially_new = positive[positive["validation_status"] == "POTENTIALLY_NEW"] \
                        .drop_duplicates(subset=["ae_name"])
    if len(potentially_new) > 0:
        sort_col = next(
            (c for c in ["EB05", "IC_lower_bound", "PRR", "ROR"] if c in potentially_new.columns),
            "events"
        )
        print(f"\n  Top segnali POTENTIALLY NEW (ordinati per {sort_col}):")
        top = potentially_new.nlargest(10, sort_col)[["ae_name", sort_col, "events"]]
        print(top.to_string(index=False))
        print(f"\n  ⚠  I segnali POTENTIALLY NEW emergono da evidenza statistica (FAERS)")
        print(f"     ma non hanno ancora una base scientifica regolatoria consolidata.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.contingency_table import build_contingency_table
    from src.signals import compute_mgps

    PARQUET = str(Path(__file__).resolve().parent.parent / "data" / "faers_flat_deduped.parquet")
    DRUG    = "LAPATINIB"

    print(f"Costruzione contingency table per {DRUG}...")
    ct = build_contingency_table(PARQUET, DRUG, min_a=3)

    print("Calcolo segnali MGPS...")
    signals = compute_mgps(ct)

    print("Validazione contro label FDA...")
    validated = validate_signals(signals, drug_name="lapatinib")

    validation_summary(validated)

    out_path = Path(__file__).resolve().parent.parent / "data" / "validated_signals_lapatinib.parquet"
    validated.to_parquet(str(out_path), index=False)
    print(f"\nSalvato: {out_path}")