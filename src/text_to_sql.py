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
    Fallback a lista vuota se il file non esiste.
    """
    if _PRINCIPI_ATTIVI_PATH.exists():
        return json.loads(_PRINCIPI_ATTIVI_PATH.read_text(encoding="utf-8"))
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

Your only job is to map a drug name provided by the user to the single best match
from the following list of allowed drug names. The list is exhaustive — you must
choose from it and only from it.

Rules:
- Return ONLY the exact string from the list, in UPPERCASE, with no explanation.
- If the user provides a brand name (e.g. "Brufen", "Tachipirina", "Voltaren"),
  map it to the corresponding entry in the list.
- If the user provides a name in another language (Italian, French, Spanish, etc.),
  translate and map it to the correct entry.
- If multiple entries are plausible, pick the most specific one
  (e.g. prefer "IBUPROFEN" over "IBUPROPHEN" for "brufen").
- If no entry in the list matches the input at all, return exactly: NONE
- Never return anything other than one entry from the list or NONE.

Allowed drug names:
{drug_list}
"""

def resolve_drug_name_mistral(
    user_input: str,
    drug_list: list[str] | None = None,
    api_key: str | None = None,
    retries: int = 3,
) -> str | None:
    """
    Mappa il nome farmaco inserito dall'utente a un principio attivo nella lista.

    Chiamata da appli.py come fallback quando rapidfuzz non trova un match
    con score >= 75. Restituisce il nome esatto dalla lista (UPPERCASE) o None.

    Parameters
    ----------
    user_input : str
        Nome farmaco inserito dall'utente (qualsiasi lingua, qualsiasi case).
    drug_list : list[str] | None
        Lista dei principi attivi validi. Se None, usa PRINCIPI_ATTIVI
        caricata da principi_attivi.json.
    api_key : str | None
        Chiave API Mistral. Se None, tenta di leggerla da MISTRAL_API_KEY in env.
    retries : int
        Numero di tentativi in caso di rate limit (429).

    Returns
    -------
    str | None
        Nome esatto dalla lista in UPPERCASE, oppure None se nessun match.
    """
    import os

    resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not resolved_key:
        return None

    active_list = drug_list if drug_list is not None else PRINCIPI_ATTIVI
    if not active_list:
        return None

    # Formatta la lista come stringa numerata per il prompt
    list_str = "\n".join(f"- {name}" for name in active_list)
    system_prompt = _DRUG_RESOLUTION_SYSTEM.format(drug_list=list_str)

    client = _make_client(resolved_key)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="mistral-small-latest",
                temperature=0,   # deterministico: vogliamo sempre lo stesso match
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_input.strip()},
                ],
            )
            result = response.choices[0].message.content.strip().upper()

            # Validazione: il risultato deve essere nella lista o "NONE"
            if result == "NONE":
                return None
            if result in [d.upper() for d in active_list]:
                # Restituisce la versione esatta dalla lista (case originale)
                for name in active_list:
                    if name.upper() == result:
                        return name
            # Se Mistral ha restituito qualcosa di non riconosciuto, None
            return None

        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)   # backoff: 1s, 2s, 4s
                continue
            return None   # in caso di errore non bloccante, None è il fallback sicuro

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