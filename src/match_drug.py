"""
src/match_drug.py

Drug name resolution pipeline.

Resolution flow (in order):
    0. Local brand-name → INN dictionary  (offline, zero-latency)
    1. Exact match against the Parquet drug index
    2. rapidfuzz fuzzy match (token_set_ratio ≥ 75)
    3. Mistral AI fallback → INN → fuzzy match on index (score ≥ 75)
    4. Passthrough (raw input returned as-is, with optional error message)

Public API
----------
resolve_drug_name(user_input, drug_index, mistral_key=None)
    → (resolved_name: str, method: str, score: float, error: str | None)

    method values:
        "exact"       — exact match in the index
        "fuzzy"       — rapidfuzz match
        "mistral"     — Mistral-resolved INN then fuzzy-matched
        "passthrough" — no reliable match found

    score: 0–100 (0 for exact / mistral / passthrough when not meaningful)
"""

import os
import time
from typing import Optional

# ─── Local brand-name → INN dictionary ─────────────────────────────────────
# Covers common European / Italian brand names and alternative spellings.
# Keys: UPPERCASE. Values: INN in UPPERCASE as found in FAERS.

_BRAND_TO_INN: dict[str, str] = {
    # Paracetamol / Acetaminophen
    "TACHIPIRINA":      "PARACETAMOL",
    "EFFERALGAN":       "PARACETAMOL",
    "PANADOL":          "PARACETAMOL",
    "TYLENOL":          "ACETAMINOPHEN",
    "ACET":             "ACETAMINOPHEN",
    "PARACETAMOLO":     "PARACETAMOL",

    # Ibuprofen
    "BRUFEN":           "IBUPROFEN",
    "MOMENT":           "IBUPROFEN",
    "NUROFEN":          "IBUPROFEN",
    "ADVIL":            "IBUPROFEN",
    "IBUPROFENE":       "IBUPROFEN",

    # Aspirin
    "ASPIRINA":         "ASPIRIN",
    "ASPIRINE":         "ASPIRIN",
    "CARDIOASPIRINA":   "ASPIRIN",
    "CARDIOASPIRIN":    "ASPIRIN",
    "BAYER":            "ASPIRIN",

    # Diclofenac
    "VOLTAREN":         "DICLOFENAC",
    "VOLTAROL":         "DICLOFENAC",
    "DICLOREUM":        "DICLOFENAC",
    "DICLOFENACO":      "DICLOFENAC",

    # Ketoprofen
    "KETOPROFENE":      "KETOPROFEN",
    "ORUDIS":           "KETOPROFEN",
    "FASTUM":           "KETOPROFEN",

    # Furosemide
    "LASIX":            "FUROSEMIDE",

    # PPIs / Omeprazole family
    "LOSEC":            "OMEPRAZOLE",
    "PRILOSEC":         "OMEPRAZOLE",
    "NEXIUM":           "ESOMEPRAZOLE",
    "PANTORC":          "PANTOPRAZOLE",
    "PARIET":           "RABEPRAZOLE",

    # Statins
    "LIPITOR":          "ATORVASTATIN",
    "ZOCOR":            "SIMVASTATIN",
    "CRESTOR":          "ROSUVASTATIN",
    "PRAVACHOL":        "PRAVASTATIN",

    # Antihypertensives
    "NORVASC":          "AMLODIPINE",
    "ZESTRIL":          "LISINOPRIL",
    "PRINIVIL":         "LISINOPRIL",
    "COVERSYL":         "PERINDOPRIL",
    "TRITACE":          "RAMIPRIL",
    "DIOVAN":           "VALSARTAN",
    "COZAAR":           "LOSARTAN",
    "TENORMIN":         "ATENOLOL",
    "CONCOR":           "BISOPROLOL",

    # Antibiotics
    "AUGMENTIN":        "AMOXICILLIN",
    "ZIMOX":            "AMOXICILLIN",
    "ZITHROMAX":        "AZITHROMYCIN",
    "KLACID":           "CLARITHROMYCIN",
    "CIPROXIN":         "CIPROFLOXACIN",
    "FLAGYL":           "METRONIDAZOLE",

    # Antidiabetics
    "GLUCOPHAGE":       "METFORMIN",
    "JANUVIA":          "SITAGLIPTIN",
    "LANTUS":           "INSULIN GLARGINE",
    "HUMALOG":          "INSULIN LISPRO",
    "NOVOLOG":          "INSULIN ASPART",

    # Anticoagulants
    "COUMADIN":         "WARFARIN",
    "SINTROM":          "ACENOCOUMAROL",
    "PRADAXA":          "DABIGATRAN",
    "XARELTO":          "RIVAROXABAN",
    "ELIQUIS":          "APIXABAN",

    # Corticosteroids
    "DELTACORTENE":     "PREDNISONE",
    "MEDROL":           "METHYLPREDNISOLONE",
    "BENTELAN":         "BETAMETHASONE",

    # Oncology (common in FAERS)
    "HERCEPTIN":        "TRASTUZUMAB",
    "AVASTIN":          "BEVACIZUMAB",
    "GLEEVEC":          "IMATINIB",
    "GLIVEC":           "IMATINIB",
    "TAXOL":            "PACLITAXEL",
    "TAXOTERE":         "DOCETAXEL",
    "XELODA":           "CAPECITABINE",
    "ZOMETA":           "ZOLEDRONIC ACID",
    "NEUPOGEN":         "FILGRASTIM",
    "KEYTRUDA":         "PEMBROLIZUMAB",
    "OPDIVO":           "NIVOLUMAB",

    # Immunosuppressants
    "PROGRAF":          "TACROLIMUS",
    "SANDIMMUN":        "CICLOSPORIN",
    "CELLCEPT":         "MYCOPHENOLATE MOFETIL",

    # Neurological / Psychiatric
    "LYRICA":           "PREGABALIN",
    "NEURONTIN":        "GABAPENTIN",
    "ZOLOFT":           "SERTRALINE",
    "PROZAC":           "FLUOXETINE",
    "LEXAPRO":          "ESCITALOPRAM",
    "CIPRALEX":         "ESCITALOPRAM",
    "RISPERDAL":        "RISPERIDONE",
    "ZYPREXA":          "OLANZAPINE",
    "SEROQUEL":         "QUETIAPINE",
    "XANAX":            "ALPRAZOLAM",
    "VALIUM":           "DIAZEPAM",
    "TAVOR":            "LORAZEPAM",
    "RIVOTRIL":         "CLONAZEPAM",
    "DEPAKINE":         "VALPROIC ACID",
    "TEGRETOL":         "CARBAMAZEPINE",
    "KEPPRA":           "LEVETIRACETAM",

    # Respiratory
    "VENTOLIN":         "ALBUTEROL",
    "SALBUTAMOLO":      "ALBUTEROL",
    "SYMBICORT":        "BUDESONIDE",
    "SERETIDE":         "FLUTICASONE",
    "SPIRIVA":          "TIOTROPIUM",
    "SINGULAR":         "MONTELUKAST",

    # Other frequently reported in FAERS
    "DIFLUCAN":         "FLUCONAZOLE",
    "TAMIFLU":          "OSELTAMIVIR",
    "PLAVIX":           "CLOPIDOGREL",
    "NEXPLANON":        "ETONOGESTREL",
}

# ─── Mistral system prompt ───────────────────────────────────────────────────

_MISTRAL_SYSTEM_PROMPT = (
    "You are a pharmacology expert. Your only job is to identify the English INN "
    "(International Nonproprietary Name) of the drug the user refers to, regardless "
    "of language, brand name, spelling, or abbreviation.\n\n"
    "If the drug is referred to in a foreign language, use your knowledge to map it "
    "to the English INN (e.g. 'oki task' → KETOPROFEN).\n\n"
    "Return ONLY the English INN in UPPERCASE. No explanation, no punctuation, "
    "no extra words. If you cannot identify the drug with confidence, return: NONE\n\n"
    "Examples:\n"
    "  brufen         → IBUPROFEN\n"
    "  tachipirina    → ACETAMINOPHEN\n"
    "  cardioaspirina → ASPIRIN\n"
    "  voltaren       → DICLOFENAC\n"
    "  lasix          → FUROSEMIDE\n"
    "  moment         → IBUPROFEN\n"
    "  ketoprofene    → KETOPROFEN\n"
    "  aspirine       → ASPIRIN"
)

# ─── Fuzzy match threshold ───────────────────────────────────────────────────

_FUZZY_THRESHOLD = 75   # token_set_ratio score (0–100)


# ─── Internal helpers ────────────────────────────────────────────────────────

def _local_brand_lookup(query_upper: str) -> Optional[str]:
    """
    Exact key lookup in _BRAND_TO_INN, then prefix scan for names with
    dosage suffixes (e.g. "TACHIPIRINA 500MG" → "PARACETAMOL").
    """
    if query_upper in _BRAND_TO_INN:
        return _BRAND_TO_INN[query_upper]
    for brand, inn in _BRAND_TO_INN.items():
        if query_upper.startswith(brand):
            return inn
    return None


def _fuzzy_match(query: str, drug_index: list[str]) -> tuple[Optional[str], float]:
    """
    Returns (best_match, score) if score ≥ _FUZZY_THRESHOLD, else (None, score).
    Uses rapidfuzz token_set_ratio for robustness against word-order variants.
    """
    from rapidfuzz import process, fuzz
    result = process.extractOne(query, drug_index, scorer=fuzz.token_set_ratio)
    if result:
        match, score, _ = result
        if score >= _FUZZY_THRESHOLD:
            return match, float(score)
    return None, float(result[1]) if result else 0.0


def _call_mistral_inn(
    user_input: str,
    api_key: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Calls Mistral AI to resolve any drug name/brand/spelling to its English INN.

    Parameters
    ----------
    user_input : str
        Raw drug name as entered by the user.
    api_key : str | None
        Explicit API key, or None → falls back to MISTRAL_API_KEY env var.

    Returns
    -------
    (inn, error)
        inn   : UPPERCASE INN string on success, None otherwise.
        error : Human-readable error string, or None if no error occurred.
                Returns (None, None) silently when no key is available at all
                (not an error condition — Mistral is an optional fallback).
    """
    resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not resolved_key:
        return None, None  # no key available — silent skip

    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package not installed — Mistral fallback unavailable."

    client = OpenAI(api_key=resolved_key, base_url="https://api.mistral.ai/v1")

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="mistral-small-latest",
                temperature=0,
                messages=[
                    {"role": "system", "content": _MISTRAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_input.strip()},
                ],
            )
            result = resp.choices[0].message.content.strip().upper()
            if result == "NONE" or not result:
                return None, None
            return result, None
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return None, f"Mistral API error: {err}"

    return None, "Mistral did not respond after 3 attempts."


# ─── Public API ──────────────────────────────────────────────────────────────

def resolve_drug_name(
    user_input: str,
    drug_index: list[str],
    mistral_key: Optional[str] = None,
) -> tuple[str, str, float, Optional[str]]:
    """
    Resolves a user-entered drug name to the exact drug_name present in the
    Parquet index, following a four-step cascade.

    Parameters
    ----------
    user_input  : str        Raw input from the UI (any language/casing).
    drug_index  : list[str]  List of canonical drug_name values from the Parquet.
    mistral_key : str | None Explicit Mistral API key; falls back to env var.

    Returns
    -------
    (resolved_name, method, score, error)

    resolved_name : str
        The matched drug_name (UPPERCASE), or the raw input on passthrough.
    method : "exact" | "fuzzy" | "mistral" | "passthrough"
    score  : float  0–100 (rapidfuzz score where applicable, else 100 or 0)
    error  : str | None  — error/warning message to surface in the UI, or None
    """
    query = user_input.strip().upper()

    # Step 0 — Local brand dictionary (offline, zero-latency)
    local_inn = _local_brand_lookup(query)
    if local_inn:
        # Validate the INN against the index with an exact or fuzzy check
        if local_inn in drug_index:
            return local_inn, "exact", 100.0, None
        match, score = _fuzzy_match(local_inn, drug_index)
        if match:
            return match, "fuzzy", score, None
        # INN known locally but not in this dataset — fall through to Mistral

    # Step 1 — Exact match
    if query in drug_index:
        return query, "exact", 100.0, None

    # Step 2 — Fuzzy match
    match, score = _fuzzy_match(query, drug_index)
    if match:
        return match, "fuzzy", score, None

    # Step 3 — Mistral AI fallback
    inn, err = _call_mistral_inn(user_input, mistral_key)
    if err:
        return query, "passthrough", 0.0, err
    if inn:
        match_b, score_b = _fuzzy_match(inn, drug_index)
        if match_b:
            return match_b, "mistral", score_b, None
        # INN resolved but not found in dataset
        return query, "passthrough", 0.0, (
            f"Mistral resolved to **{inn}** but no match found in the dataset. "
            f"Try using the drug's full English INN directly."
        )

    # Step 4 — No reliable match
    return query, "passthrough", 0.0, None