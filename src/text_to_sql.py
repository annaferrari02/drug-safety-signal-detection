"""
src/text_to_sql.py

Due responsabilità distinte:

1. resolve_drug_name_mistral(user_input, drug_list, api_key) → str | None
   Chiede a Mistral di mappare l'input dell'utente a uno dei principi attivi
   nella lista `drug_list` (caricata da principi_attivi.json).
   Restituisce il nome in UPPERCASE esattamente come appare nella lista,
   oppure None se nessun match è affidabile.

   Questa funzione è il fallback chiamato da appli.py quando rapidfuzz
   non trova un match con score >= 75 sul Parquet reale.
   L'output è una stringa che appli.py usa direttamente come target_drug.

2. parse_nl_to_params(user_input, api_key) → dict
   Funzione originale: estrae tutti i parametri di analisi da una query
   in linguaggio naturale (target_drug + where_extra + min_a).
   Usata per flussi NLP avanzati (query tipo "mostrami le reazioni
   cardiache nelle donne anziane che prendono anche warfarin").

Coerenza con appli.py:
   - resolve_drug_name_mistral() restituisce esattamente lo stesso tipo
     di valore che appli.py si aspetta dal blocco "mistral" in resolve_drug_name():
     una stringa UPPERCASE o None.
   - L'API key viene passata come parametro (non letta da env) per permettere
     ad appli.py di passare quella inserita dall'utente nel form.
   - Il client OpenAI viene istanziato dentro le funzioni (non a livello modulo)
     per evitare crash se MISTRAL_API_KEY non è in .env.
"""

import json
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Lista principi attivi — caricata dal JSON alla root del progetto.
# Il path è relativo a questo file (src/text_to_sql.py → ../principi_attivi.json).
# ---------------------------------------------------------------------------
_PRINCIPI_ATTIVI_PATH = Path(__file__).resolve().parent.parent / "principi_attivi.json"

def load_principi_attivi() -> list[str]:
    """
    Carica la lista dei principi attivi da principi_attivi.json.

    Path atteso (sia in locale che nel container Docker):
        <project_root>/principi_attivi.json
    dove <project_root> = due livelli sopra src/text_to_sql.py.

    Fallback a lista vuota con warning se il file non esiste —
    in quel caso resolve_drug_name_mistral restituisce sempre None.
    """
    if _PRINCIPI_ATTIVI_PATH.exists():
        data = json.loads(_PRINCIPI_ATTIVI_PATH.read_text(encoding="utf-8"))
        return data
    import warnings
    warnings.warn(
        f"[text_to_sql] principi_attivi.json non trovato in {_PRINCIPI_ATTIVI_PATH}. "
        f"Il fallback Mistral non potrà risolvere i nomi farmaco. "
        f"Aggiungi il file alla root del progetto e assicurati che il Dockerfile "
        f"lo includa con: COPY principi_attivi.json .",
        stacklevel=2,
    )
    return []

PRINCIPI_ATTIVI: list[str] = load_principi_attivi()


# ---------------------------------------------------------------------------
# Helper: costruisce il client Mistral on-demand
# ---------------------------------------------------------------------------
def _make_client(api_key: str):
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
    )


# ============================================================================
# 1. RISOLUZIONE NOME FARMACO — fallback Mistral per appli.py
# ============================================================================

_DRUG_RESOLUTION_SYSTEM = """You are a pharmacology expert with deep knowledge of drug names,
brand names, INN (International Nonproprietary Names), and trade names in all languages.

Your only job is to identify the English INN (International Nonproprietary Name) of the drug
the user is referring to, regardless of the language, spelling, or brand name used.

Steps you must follow internally:
1. Identify the drug the user means (brand name, foreign spelling, abbreviation, etc.)
2. Return its English INN in UPPERCASE

Examples:
  "ketoprofene"    → KETOPROFEN
  "brufen"         → IBUPROFEN
  "tachipirina"    → ACETAMINOPHEN
  "aspirine"       → ASPIRIN
  "cardioaspirina" → ASPIRIN
  "voltaren"       → DICLOFENAC
  "moment"         → IBUPROFEN
  "paracetamolo"   → ACETAMINOPHEN
  "ibuprofene"     → IBUPROFEN
  "cortisone"      → HYDROCORTISONE
  "lasix"          → FUROSEMIDE

Output rules:
- Return ONLY the English INN in UPPERCASE
- No explanation, no punctuation, no extra words
- If you cannot identify the drug with confidence, return exactly: NONE
"""

def resolve_drug_name_mistral(
    user_input: str,
    drug_list: list[str] | None = None,   # non usato, mantenuto per compatibilità
    api_key: str | None = None,
    retries: int = 3,
) -> str | None:
    """
    Identifica il principio attivo INN in inglese del farmaco inserito dall'utente.

    Mistral si occupa solo di traduzione/identificazione farmacologica —
    restituisce l'INN in inglese (es. "ketoprofene" → "KETOPROFEN").
    Il matching ortografico contro il dataset reale è delegato a rapidfuzz
    in appli.py, che riceve l'INN e lo confronta con drug_index del Parquet.

    Vantaggi rispetto al passare la lista completa a Mistral:
    - Prompt piccolo e veloce (nessuna lista nel contesto)
    - Nessuna allucinazione sul nome esatto: rapidfuzz garantisce il match
    - Funziona anche per farmaci non nella lista originale principi_attivi.json

    Returns
    -------
    str | None — INN in inglese UPPERCASE, oppure None se non identificato
    """
    import os

    resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not resolved_key:
        return None

    client = _make_client(resolved_key)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="mistral-small-latest",
                temperature=0,
                messages=[
                    {"role": "system", "content": _DRUG_RESOLUTION_SYSTEM},
                    {"role": "user",   "content": user_input.strip()},
                ],
            )
            result = response.choices[0].message.content.strip().upper()

            if result == "NONE" or not result:
                return None

            # Restituisce l'INN — rapidfuzz in appli.py farà il match sul dataset
            return result

        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None

    return None


# ============================================================================
# 2. PARSING QUERY NL COMPLETA — estrae tutti i parametri per run_pipeline
# ============================================================================

_NL_TO_PARAMS_SYSTEM = """You are an expert in pharmacovigilance and the FDA FAERS database.

The user will ask a question in natural language about drug safety signals.
Your job is to extract the parameters needed to run a disproportionality analysis.

Respond ONLY with a valid JSON object, no explanation, no markdown fences. All fields required.

Schema:
{
    "target_drug":   string  — drug name in UPPERCASE, chosen from the allowed list below,
    "min_a":         integer — minimum cell-a threshold (default: 3),
    "where_extra":   string | null — SQL WHERE clause without the WHERE keyword
}

Rules for target_drug:
- Always map the drug name to the closest entry in the allowed list below.
- Use UPPERCASE exactly as it appears in the list.
- If uncertain, pick the most specific match.

Allowed drug names:
{drug_list}

Available columns for where_extra:
- sex: 'male' or 'female'
- age_stratum: 'pediatric', 'adult', 'geriatric'
- age_years: integer
- receive_year: integer
- receive_quarter: string (e.g. '2015Q1')
- serious: boolean
- outcome_death, outcome_lifethreat, outcome_hosp, outcome_disab: boolean
- reporter_country: string
- reporter_qual: string
- drug_name: use a subquery on safetyreportid for comedication filters
- reaction_pt: string (MedDRA preferred term)

Example output:
{
  "target_drug": "IBUPROFEN",
  "min_a": 3,
  "where_extra": "sex = 'female' AND age_stratum = 'adult'"
}

If no filter applies, set where_extra to null.
"""

def parse_nl_to_params(
    user_input: str,
    api_key: str | None = None,
    drug_list: list[str] | None = None,
    retries: int = 3,
) -> dict:
    """
    Estrae i parametri di analisi da una query in linguaggio naturale.

    A differenza di resolve_drug_name_mistral(), questa funzione gestisce
    query complesse tipo "mostrami le reazioni nelle donne anziane che prendono
    anche warfarin" producendo anche where_extra e min_a.

    Parameters
    ----------
    user_input : str  — query in linguaggio naturale
    api_key    : str | None — chiave Mistral (fallback su env MISTRAL_API_KEY)
    drug_list  : list[str] | None — lista principi attivi (default: PRINCIPI_ATTIVI)
    retries    : int — tentativi in caso di rate limit

    Returns
    -------
    dict con chiavi: target_drug (str), min_a (int), where_extra (str | None)

    Raises
    ------
    ValueError  — se Mistral non risponde con JSON valido dopo tutti i tentativi
    RuntimeError — se la API key non è disponibile
    """
    import os

    resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "Mistral API key non trovata. "
            "Passala come parametro api_key o impostala in MISTRAL_API_KEY."
        )

    active_list = drug_list if drug_list is not None else PRINCIPI_ATTIVI
    list_str    = "\n".join(f"- {name}" for name in active_list)
    system_prompt = _NL_TO_PARAMS_SYSTEM.format(drug_list=list_str)

    client = _make_client(resolved_key)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="mistral-small-latest",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_input.strip()},
                ],
            )
            raw = response.choices[0].message.content.strip()
            # Strip eventuali fence markdown che Mistral a volte aggiunge
            raw = raw.strip("```json").strip("```").strip()
            params = json.loads(raw)

            # Normalizza target_drug
            if "target_drug" in params:
                params["target_drug"] = params["target_drug"].strip().upper()

            # Garantisce min_a con default
            params.setdefault("min_a", 3)

            return params

        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise ValueError(f"Mistral non ha restituito JSON valido: {raw!r}")

        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


# ============================================================================
# Funzione di compatibilità con il codice esistente (nl_to_contingency_table)
# ============================================================================

def nl_to_contingency_table(user_input: str, api_key: str | None = None):
    """
    Compatibilità con il codice esistente.
    Parsea la query NL ed esegue build_contingency_table().
    """
    from src.contingency_table import build_contingency_table

    params = parse_nl_to_params(user_input, api_key=api_key)
    print(f"  [NL→CT] Parametri estratti: {params}")

    return build_contingency_table(
        parquet_path="data/faers_flat_deduped.parquet",
        target_drug=params["target_drug"],
        min_a=params.get("min_a", 3),
        where_extra=params.get("where_extra"),
    )