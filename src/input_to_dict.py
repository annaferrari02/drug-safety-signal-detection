import os
import json
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["MISTRAL_API_KEY"],
    base_url="https://api.mistral.ai/v1"
)

SYSTEM_PROMPT = """You are an expert in pharmacovigilance and the FDA FAERS database.

The user will ask a question in natural language about drug safety signals.
Your job is to extract the parameters needed to run a disproportionality analysis.

Respond ONLY with a valid JSON object, no explanation, no markdown fences. All fields are required.

Schema:
{
    "target_drug":             string  — drug name (e.g. "LAPATINIB")
    "where_extra":             string | null — SQL filter for stratification (no WHERE keyword)
    "min_a":                   integer — minimum cell-a threshold for contingency table (default: 3)
    "algorithms":              array   — subset of ["prr", "ror", "bcpnn", "mgps"] (default: all four)
    "fdr_threshold":           float   — FDR threshold for PRR and ROR (default: 0.05)
    "eb05_threshold":          float   — EB05 threshold for MGPS (default: 2.0)
    "ic_threshold":            float   — IC025 threshold for BCPNN (default: 0.0)
    "openfda_api_key":         string | null — optional openFDA API key
    "validate_label":          boolean — run FDA label validation (default: false)
    "check_weber":             boolean — run Weber effect check (default: false)
    "weber_approval_override": integer | null — manual drug approval year, bypasses API lookup
}

Available columns for where_extra filters:
- 'safetyreportid': ID of the adverse event report
- 'receivedate': Date when the report was received
- 'receive_year': Year when the report was received
- 'receive_quarter': Quarter when the report was received
- 'serious': Indicates if the adverse event is serious
- 'outcome_death': Indicates if the adverse event resulted in death
- 'outcome_lifethreat': Indicates if the adverse event resulted in life-threatening condition
- 'outcome_hosp': Indicates if the adverse event resulted in hospitalization
- 'outcome_disab': Indicates if the adverse event resulted in disability
- 'reporter_country': Country of the reporter
- 'reporter_qual': Qualification of the reporter
- 'age_years': Age of the patient in years
- 'age_stratum': Age stratum of the patient — 'pediatric', 'adult', 'geriatric'
- 'sex': Sex of the patient — 'male' or 'female'
- 'drug_name': Name of the drug (use subquery on safetyreportid for comedication filters)
- 'drug_name_source': Source of the drug name
- 'drug_characterization': Characterization of the drug
- 'drug_indication': Indication for which the drug was used
- 'reaction_pt': Preferred term for the adverse reaction

Example output:
{
  "target_drug": "LAPATINIB",
  "where_extra": "sex = 'female' AND age_stratum = 'adult'",
  "min_a": 3,
  "algorithms": ["prr", "ror", "bcpnn", "mgps"],
  "fdr_threshold": 0.05,
  "eb05_threshold": 2.0,
  "ic_threshold": 0.0,
  "openfda_api_key": null,
  "validate_label": false,
  "check_weber": false,
  "weber_approval_override": null
}"""


def parse_nl_to_run_config(user_input: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_input},
                ],
                response_format={"type": "json_object"},  # enforces JSON output
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"Rate limited. Retrying in {wait}s... (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            raise


def nl_to_run_config(user_input: str) -> dict:
    """Parse natural language input and return the full analysis parameter dict."""
    run_config = parse_nl_to_run_config(user_input)
    print(f"Parametri estratti: {json.dumps(run_config, indent=2)}")
    return run_config


# def nl_to_contingency_table(user_input: str):
#     """Convenience wrapper: parse NL input and directly call build_contingency_table."""
#     from src.contingency_table import build_contingency_table

#     run_config = parse_nl_to_run_config(user_input)
#     print(f"Parametri estratti: {json.dumps(run_config, indent=2)}")

#     return build_contingency_table(
#         parquet_path="data/faers_flat_deduped.parquet",
#         target_drug=run_config["target_drug"],
#         min_a=params.get("min_a", 3),
#         where_extra=params.get("where_extra"),
#     )