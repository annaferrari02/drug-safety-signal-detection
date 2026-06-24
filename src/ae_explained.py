"""
src/ae_explained.py

Adverse Event explanation module.

Public functions:
    explain_ae(ae_name, drug_name, api_key) -> str | None
    medlineplus_url(ae_name) -> str
"""

import os
import re
import urllib.parse
import urllib.request
from typing import Optional

# ── Cache ────────────────────────────────────────────────────────────────────
_explain_cache: dict[tuple[str, str], Optional[str]] = {}
_url_cache: dict[str, str] = {}

# ── British → American spelling variants (and vice versa) ───────────────────
# Only terms likely to appear as MedDRA preferred terms.
_SPELLING_VARIANTS: list[tuple[str, str]] = [
    ("diarrhoea",       "diarrhea"),
    ("haemorrhage",     "hemorrhage"),
    ("haematuria",      "hematuria"),
    ("haemoglobin",     "hemoglobin"),
    ("oedema",          "edema"),
    ("anaemia",         "anemia"),
    ("leukaemia",       "leukemia"),
    ("dyspnoea",        "dyspnea"),
    ("ischaemia",       "ischemia"),
    ("foetal",          "fetal"),
    ("gynaecological",  "gynecological"),
    ("hypokalaemia",    "hypokalemia"),
    ("hyperkalaemia",   "hyperkalemia"),
    ("hypocalcaemia",   "hypocalcemia"),
    ("hypercalcaemia",  "hypercalcemia"),
    ("coagulopathy",    "coagulopathy"),   # same, just in case
    ("tumour",          "tumor"),
    ("colour",          "color"),
    ("labour",          "labor"),
    ("behaviour",       "behavior"),
    ("anaesthesia",     "anesthesia"),
    ("paediatric",      "pediatric"),
    ("oesophageal",     "esophageal"),
    ("orthopaedic",     "orthopedic"),
]

# Build reverse map too (American → British)
_VARIANT_MAP: dict[str, str] = {}
for brit, amer in _SPELLING_VARIANTS:
    _VARIANT_MAP[brit] = amer
    _VARIANT_MAP[amer] = brit


def _to_slug(term: str) -> str:
    """Lowercase, remove punctuation, collapse spaces → single slug word."""
    clean = re.sub(r"[^\w\s]", "", term.lower()).strip()
    return clean.replace(" ", "")


def _url_exists(url: str, timeout: float = 3.0) -> bool:
    """HEAD request to check if a URL returns 200. Fails fast."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _direct_url(slug: str) -> str:
    return f"https://medlineplus.gov/{slug}.html"


def _search_url(ae_name: str) -> str:
    query = urllib.parse.quote_plus(ae_name.lower())
    return f"https://medlineplus.gov/search/?query={query}"


def medlineplus_url(ae_name: str) -> str:
    """
    Return the best available MedlinePlus URL for an adverse event term.

    Resolution order:
        1. Direct page URL with original slug           → /fatigue.html
        2. Direct page URL with spelling variant        → /diarrhea.html
        3. Search URL fallback (always works)           → /search/?query=...

    URL existence is verified with a HEAD request (3 s timeout).
    Results are cached per ae_name to avoid duplicate network calls.
    """
    if ae_name in _url_cache:
        return _url_cache[ae_name]

    words = re.sub(r"[^\w\s]", "", ae_name.lower()).strip().split()

    # Multi-word terms: slug is unreliable → go straight to search
    if len(words) > 1:
        url = _search_url(ae_name)
        _url_cache[ae_name] = url
        return url

    # Single-word term: try direct page
    slug = words[0]
    direct = _direct_url(slug)
    if _url_exists(direct):
        _url_cache[ae_name] = direct
        return direct

    # Try spelling variant
    alt_slug = _VARIANT_MAP.get(slug)
    if alt_slug:
        alt_url = _direct_url(alt_slug)
        if _url_exists(alt_url):
            _url_cache[ae_name] = alt_url
            return alt_url

    # Fallback: search URL (always valid)
    url = _search_url(ae_name)
    _url_cache[ae_name] = url
    return url


# ── Mistral explanation ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a clinical pharmacologist writing for healthcare professionals. "
    "Given an adverse event (MedDRA preferred term) and a drug, write exactly "
    "2 sentences: one explaining what the adverse event is, one describing its "
    "known or suspected relationship with the drug. "
    "Be factual, concise, and neutral. No markdown, no bullet points, plain text only."
)


def _build_user_prompt(ae_name: str, drug_name: str) -> str:
    return f"Adverse event: {ae_name}\nDrug: {drug_name}"


def explain_ae(
    ae_name: str,
    drug_name: str,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Generate a 2-sentence clinical explanation via Mistral AI.
    Returns None silently on any failure.
    """
    cache_key = (ae_name.upper(), drug_name.upper())
    if cache_key in _explain_cache:
        return _explain_cache[cache_key]

    resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not resolved_key:
        _explain_cache[cache_key] = None
        return None

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=resolved_key,
            base_url="https://api.mistral.ai/v1",
            timeout=8.0,
        )
        resp = client.chat.completions.create(
            model="mistral-small-latest",
            temperature=0.3,
            max_tokens=120,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(ae_name, drug_name)},
            ],
        )
        text = resp.choices[0].message.content.strip()
        _explain_cache[cache_key] = text if text else None
        return _explain_cache[cache_key]

    except Exception:
        _explain_cache[cache_key] = None
        return None