import os
import json
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()  # carica la chiave API da .env

client = OpenAI(
    api_key=os.environ["MISTRAL_API_KEY"],
    base_url="https://api.mistral.ai/v1"
)

SYSTEM_PROMPT = """You are an expert in pharmacovigilance and the FDA FAERS database.

The user will ask a question in natural language about drug safety signals.
Your job is to extract the parameters needed to call the function build_contingency_table().

The function signature is:
    build_contingency_table(
        parquet_path: str,       # always "data/faers_flat_deduped.parquet"
        target_drug: str,        # drug name in UPPERCASE
        pt_col: str,             # always "reaction_pt"
        min_a: int,              # minimum co-occurrence count, default 3
        where_extra: str | None  # optional SQL filter clause (without WHERE keyword)
    )

Available columns for where_extra filters:

- 'safetyreportid': ID of the adverse event report,
- 'receivedate': Date when the report was received,
- 'receive_year': Year when the report was received,
- 'receive_quarter': Quarter when the report was received,
- 'serious': Indicates if the adverse event is serious,
- 'outcome_death': Indicates if the adverse event resulted in death,
- 'outcome_lifethreat': Indicates if the adverse event resulted in life-threatening condition,
- 'outcome_hosp': Indicates if the adverse event resulted in hospitalization,
- 'outcome_disab': Indicates if the adverse event resulted in disability,
- 'reporter_country': Country of the reporter,
- 'reporter_qual': Qualification of the reporter,
- 'age_years': Age of the patient in years,
- 'age_stratum': Age stratum of the patient, 'pediatric', 'adult', 'geriatric',
- 'sex': Sex of the patient, 'male' or 'female'
- 'drug_name': Name of the drug, for comedication filters (use a subquery on safetyreportid)
- 'drug_name_source': Source of the drug name,
- 'drug_characterization': Characterization of the drug,
- 'drug_indication': Indication for which the drug was used,
- 'reaction_pt': Preferred term for the adverse reaction

Respond ONLY with a valid JSON object, no explanation, no markdown. Example:
{
  "target_drug": "LAPATINIB",
  "min_a": 3,
  "where_extra": "sex = 'female' AND age_stratum = 'adult'"
}

If no filter applies, set where_extra to null.

The database has the following columns:
 'safetyreportid': ID of the adverse event report,
 'receivedate': Date when the report was received,
 'receive_year': Year when the report was received,
 'receive_quarter': Quarter when the report was received,
 'serious': Indicates if the adverse event is serious,
 'outcome_death': Indicates if the adverse event resulted in death,
 'outcome_lifethreat': Indicates if the adverse event resulted in life-threatening condition,
 'outcome_hosp': Indicates if the adverse event resulted in hospitalization,
 'outcome_disab': Indicates if the adverse event resulted in disability,
 'reporter_country': Country of the reporter,
 'reporter_qual': Qualification of the reporter,
 'age_years': Age of the patient in years,
 'age_stratum': Age stratum of the patient,
 'sex': Sex of the patient,
 'drug_name': Name of the drug,
 'drug_name_source': Source of the drug name,
 'drug_characterization': Characterization of the drug,
 'drug_indication': Indication for which the drug was used,
 'reaction_pt': Preferred term for the adverse reaction



- YOUR_TABLE_NAME (column1, column2, ...)  # <-- adatta al tuo schema
"""


def parse_nl_to_params(user_input: str) -> dict:
    response = client.chat.completions.create(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def nl_to_contingency_table(user_input: str):
    from src.contingency_table import build_contingency_table
    params = parse_nl_to_params(user_input)
    print(f"Parametri estratti: {params}")
    return build_contingency_table(
        parquet_path="data/faers_flat_deduped.parquet",
        target_drug=params["target_drug"],
        min_a=params.get("min_a", 3),
        where_extra=params.get("where_extra")
    )
