"""
run_flatten.py

Appiattisce i file JSON FAERS scaricati da run_download.py in un Parquet flat
(una riga per coppia drug × reaction), poi deduplica i follow-up report.

Correzioni rispetto alla versione precedente:
    - Parsing streaming con ijson: carica un report alla volta invece dell'intero
      JSON, eliminando il collo di bottiglia di memoria sul parsing
    - Scrittura Parquet in append per quarter con ParquetWriter: scrive su disco
      dopo ogni quarter invece di accumulare tutto in all_rows, eliminando il
      picco di memoria finale di pd.DataFrame(all_rows)

Output in data/:
    faers_flat.parquet          — dataset flat pre-deduplicazione
    faers_flat_deduped.parquet  — dataset pulito, input degli step successivi
    deduplication_report.txt    — report QC della deduplicazione
"""

import platform
import re
from datetime import datetime
from pathlib import Path

import duckdb
import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR        = Path("data")
RAW_DIR         = DATA_DIR / "raw"
PARQUET_FLAT    = DATA_DIR / "faers_flat.parquet"
PARQUET_DEDUPED = DATA_DIR / "faers_flat_deduped.parquet"
DEDUP_REPORT    = DATA_DIR / "deduplication_report.txt"

# Numero di report da accumulare in memoria prima di scrivere un batch su disco.
# Valore più alto = meno scritture = più veloce, ma più RAM usata per batch.
# 50_000 è un buon compromesso: ~200-400 MB RAM per batch a seconda dei report.
BATCH_SIZE = 50_000


# FUNZIONI DI NORMALIZZAZIONE (invariate da run_ingestion.py)

def parse_age_to_years(age_val, age_unit_code):
    """Converte l'età nel campo FAERS in anni float, gestendo tutte le unità."""
    if age_val is None:
        return None
    try:
        val = float(str(age_val).strip())
    except (ValueError, TypeError):
        return None
    unit = str(age_unit_code).strip() if age_unit_code else "801"
    conversions = {"801": 1.0, "802": 1/12, "803": 1/52.18,
                   "804": 1/365.25, "805": 1/8766}
    return val * conversions.get(unit, 1.0)


def age_to_stratum(age_years):
    """Mappa l'età in anni al gruppo demografico usato nei filtri DuckDB."""
    if age_years is None:
        return "unknown"
    if age_years < 18:
        return "pediatric"
    if age_years < 65:
        return "adult"
    return "geriatric"


def parse_sex(sex_val):
    """Normalizza il campo sesso ai valori attesi dai filtri DuckDB."""
    if sex_val is None:
        return "unknown"
    s = str(sex_val).strip().upper()
    if s in ("1", "M", "MALE"):
        return "male"
    if s in ("2", "F", "FEMALE"):
        return "female"
    return "unknown"


def parse_date(date_str):
    """Normalizza le date FAERS (vari formati) al formato ISO YYYY-MM-DD."""
    if not date_str:
        return None
    s = str(date_str).strip().replace("-", "")
    try:
        if len(s) == 8:
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        if len(s) == 6:
            return f"{s[:4]}-{s[4:6]}-01"
        if len(s) == 4:
            return f"{s}-01-01"
    except Exception:
        pass
    return None


def extract_drug_name_with_source(drug: dict) -> tuple:
    """
    Estrae il nome del farmaco con fallback chain in ordine di affidabilità:
      1. activesubstance.activesubstancename  (era moderna 2019+)
      2. medicinalproduct                     (tutte le ere, nome commerciale)
      3. openfda.generic_name[0]              (era legacy 2004-2012)
      4. openfda.brand_name[0]               (fallback commerciale legacy)
      5. openfda.substance_name[0]           (ultimo tentativo)
    """
    active = drug.get("activesubstance") or {}
    if isinstance(active, dict):
        name = str(active.get("activesubstancename") or "").upper().strip()
        if name:
            return name, "activesubstance"

    name = str(drug.get("medicinalproduct") or "").upper().strip()
    if name:
        return name, "medicinalproduct"

    openfda = drug.get("openfda") or {}
    if isinstance(openfda, dict):
        for field in ("generic_name", "brand_name", "substance_name"):
            val = openfda.get(field)
            if val:
                if isinstance(val, list) and val:
                    name = str(val[0]).upper().strip()
                elif isinstance(val, str):
                    name = val.upper().strip()
                if name:
                    return name, f"openfda.{field}"

    return "", "missing"


def flatten_report(report: dict) -> list[dict]:
    """
    Appiattisce un singolo report FAERS in una lista di righe,
    una per ogni coppia (drug × reaction).
    """
    rows = []

    report_id    = report.get("safetyreportid")
    receive_date = parse_date(report.get("receivedate") or report.get("receiptdate"))
    serious      = report.get("serious")

    outcome_death = report.get("seriousnessdeath", "0")
    outcome_lt    = report.get("seriousnesslifethreatening", "0")
    outcome_hosp  = report.get("seriousnesshospitalization", "0")
    outcome_disab = report.get("seriousnessdisabling", "0")

    primary_source   = report.get("primarysource") or {}
    reporter_country = primary_source.get("reportercountry")
    reporter_qual    = primary_source.get("qualification")

    patient     = report.get("patient") or {}
    age_years   = parse_age_to_years(
                      patient.get("patientonsetage"),
                      patient.get("patientonsetageunit"))
    age_stratum = age_to_stratum(age_years)
    sex         = parse_sex(patient.get("patientsex"))

    drugs = patient.get("drug", [])
    if not isinstance(drugs, list):
        drugs = [drugs] if drugs else []

    reactions = patient.get("reaction", [])
    if not isinstance(reactions, list):
        reactions = [reactions] if reactions else []

    reaction_pts = [
        r.get("reactionmeddrapt", "").upper().strip()
        for r in reactions
        if isinstance(r, dict) and r.get("reactionmeddrapt")
    ]

    for drug in drugs:
        if not isinstance(drug, dict):
            continue

        char                        = str(drug.get("drugcharacterization", "") or "").strip()
        drug_name, drug_name_source = extract_drug_name_with_source(drug)
        indication                  = str(drug.get("drugindication") or "").upper().strip()

        for reaction_pt in reaction_pts:
            rows.append({
                "safetyreportid":        str(report_id) if report_id else None,
                "receivedate":           receive_date,
                "receive_year":          int(receive_date[:4]) if receive_date else None,
                "receive_quarter": (
                    f"{receive_date[:4]}Q{(int(receive_date[5:7])-1)//3+1}"
                    if receive_date and len(receive_date) >= 7 else None
                ),
                "serious":               str(serious) if serious else None,
                "outcome_death":         str(outcome_death),
                "outcome_lifethreat":    str(outcome_lt),
                "outcome_hosp":          str(outcome_hosp),
                "outcome_disab":         str(outcome_disab),
                "reporter_country":      reporter_country,
                "reporter_qual":         str(reporter_qual) if reporter_qual else None,
                "age_years":             age_years,
                "age_stratum":           age_stratum,
                "sex":                   sex,
                "drug_name":             drug_name,
                "drug_name_source":      drug_name_source,
                "drug_characterization": char,
                "drug_indication":       indication if indication else None,
                "reaction_pt":           reaction_pt,
            })

    return rows


# STEP 1: FLATTENING STREAMING → faers_flat.parquet

def run_flatten(json_files: list[Path]) -> None:
    """
    Parsea i file JSON in streaming (ijson) e scrive il Parquet in append
    per batch (ParquetWriter). La RAM usata in ogni momento è quella di
    BATCH_SIZE report, non dell'intero dataset.

    Flusso:
        ijson legge un report alla volta dal JSON
        → flatten_report() lo trasforma in righe
        → ogni BATCH_SIZE report, scrive un batch su disco e libera la memoria
        → alla fine del file, scrive il batch residuo
    """
    writer      = None  # ParquetWriter aperto al primo batch
    schema      = None  # schema PyArrow fissato al primo batch
    total_rows  = 0
    total_files = len(json_files)

    for i, path in enumerate(json_files, 1):
        quarter    = path.stem  # es. "2007q1" dal nome file "2007q1.json"
        batch_rows = []
        n_reports  = 0

        print(f"  [{i}/{total_files}] Parsing {quarter}...", flush=True)

        # ijson legge il JSON in streaming: tiene in RAM solo il report corrente
        with open(path, "rb") as f:
            for report in ijson.items(f, "results.item"):
                batch_rows.extend(flatten_report(report))
                n_reports += 1

                # Ogni BATCH_SIZE report, scarica su disco e libera la memoria
                if n_reports % BATCH_SIZE == 0:
                    writer, schema = _write_batch(batch_rows, writer, schema)
                    total_rows    += len(batch_rows)
                    batch_rows     = []  # libera la memoria del batch

        # Scrive il batch residuo (ultimi report del file, < BATCH_SIZE)
        if batch_rows:
            writer, schema  = _write_batch(batch_rows, writer, schema)
            total_rows     += len(batch_rows)

        quarter_rows = n_reports  # solo per il log
        print(f"     {n_reports:,} reports processati", flush=True)

    if writer:
        writer.close()

    print(f"\n  Totale righe scritte: {total_rows:,}")
    print(f"  Salvato: {PARQUET_FLAT} ({PARQUET_FLAT.stat().st_size / 1e6:.0f} MB)")

    # QC post-flattening
    _qc_flat()


def _write_batch(
        rows:   list[dict],
        writer: pq.ParquetWriter | None,
        schema: pa.Schema | None,
) -> tuple[pq.ParquetWriter, pa.Schema]:
    """
    Converte una lista di righe in una PyArrow Table e la appende al Parquet.
    Apre il ParquetWriter al primo batch e lo riusa per tutti i successivi,
    garantendo schema consistente in tutto il file.
    """
    df    = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer is None:
        # Primo batch: apre il file e fissa lo schema
        schema = table.schema
        writer = pq.ParquetWriter(str(PARQUET_FLAT), schema)

    # Cast al schema del primo batch per evitare type mismatch tra quarter
    writer.write_table(table.cast(schema))
    return writer, schema


def _qc_flat() -> None:
    """QC post-flattening: distribuzione drug_name_source e missing per quarter."""
    con = duckdb.connect()
    P   = str(PARQUET_FLAT)

    print("\n  === Distribuzione drug_name_source ===")
    print(con.execute(f"""
        SELECT drug_name_source,
               COUNT(*) AS n_rows,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM '{P}'
        GROUP BY drug_name_source
        ORDER BY n_rows DESC
    """).df().to_string(index=False))

    print("\n  === Missing drug_name > 5% per quarter ===")
    result = con.execute(f"""
        SELECT receive_quarter,
               COUNT(*) AS total,
               SUM(CASE WHEN drug_name = '' OR drug_name IS NULL THEN 1 ELSE 0 END) AS missing,
               ROUND(100.0 * SUM(CASE WHEN drug_name = '' OR drug_name IS NULL
                                 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_missing
        FROM '{P}'
        GROUP BY receive_quarter
        HAVING pct_missing > 5
        ORDER BY pct_missing DESC
    """).df()
    print(result.to_string(index=False) if len(result) > 0 else "  Nessun quarter con missing > 5%")
    con.close()


# STEP 2: DEDUPLICAZIONE → faers_flat_deduped.parquet

def run_deduplication() -> None:
    """
    Deduplica i follow-up report e i duplicati da overlap tra file bulk.
    Logica: per ogni safetyreportid, mantieni solo le righe con receivedate massima.
    Le righe con safetyreportid NULL (dati legacy) passano invariate.
    DuckDB opera direttamente sul Parquet senza caricare nulla in RAM pandas.
    """
    con    = duckdb.connect()
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pre = con.execute(f"""
        SELECT
            COUNT(*)                                                       AS total_rows,
            COUNT(DISTINCT safetyreportid)                                AS distinct_ids,
            COALESCE(SUM(CASE WHEN safetyreportid IS NULL THEN 1 END), 0) AS null_rows
        FROM '{PARQUET_FLAT}'
    """).fetchone()
    total_rows, distinct_ids, null_rows = pre

    dup_reports = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT safetyreportid
            FROM '{PARQUET_FLAT}'
            WHERE safetyreportid IS NOT NULL
            GROUP BY safetyreportid
            HAVING COUNT(DISTINCT receivedate) > 1
        )
    """).fetchone()[0]

    rows_in_dup_reports = con.execute(f"""
        SELECT COUNT(*) FROM '{PARQUET_FLAT}'
        WHERE safetyreportid IN (
            SELECT safetyreportid
            FROM '{PARQUET_FLAT}'
            WHERE safetyreportid IS NOT NULL
            GROUP BY safetyreportid
            HAVING COUNT(DISTINCT receivedate) > 1
        )
    """).fetchone()[0]

    quarter_breakdown_pre = con.execute(f"""
        SELECT receive_quarter,
               COUNT(*) AS total_rows,
               COUNT(DISTINCT safetyreportid) AS distinct_ids
        FROM '{PARQUET_FLAT}'
        WHERE safetyreportid IS NOT NULL
        GROUP BY receive_quarter
        ORDER BY receive_quarter
    """).df()

    print(f"  Righe totali (pre):        {total_rows:>10,}")
    print(f"  Report ID distinti:        {distinct_ids:>10,}")
    print(f"  Righe NULL report_id:      {null_rows:>10,}")
    print(f"  Report con >1 receivedate: {dup_reports:>10,}")

    con.execute(f"""
    COPY (
        WITH latest AS (
            SELECT safetyreportid, MAX(receivedate) AS max_date
            FROM '{PARQUET_FLAT}'
            WHERE safetyreportid IS NOT NULL
            GROUP BY safetyreportid
        )
        SELECT f.*
        FROM '{PARQUET_FLAT}' f
        JOIN latest l
          ON f.safetyreportid = l.safetyreportid
         AND f.receivedate    = l.max_date

        UNION ALL

        SELECT *
        FROM '{PARQUET_FLAT}'
        WHERE safetyreportid IS NULL
    )
    TO '{PARQUET_DEDUPED}' (FORMAT PARQUET)
    """)

    post = con.execute(f"""
        SELECT
            COUNT(*)                                                       AS total_rows,
            COUNT(DISTINCT safetyreportid)                                AS distinct_ids,
            COALESCE(SUM(CASE WHEN safetyreportid IS NULL THEN 1 END), 0) AS null_rows
        FROM '{PARQUET_DEDUPED}'
    """).fetchone()
    total_rows2, distinct_ids2, null_rows2 = post

    residual_dups = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT safetyreportid
            FROM '{PARQUET_DEDUPED}'
            WHERE safetyreportid IS NOT NULL
            GROUP BY safetyreportid
            HAVING COUNT(DISTINCT receivedate) > 1
        )
    """).fetchone()[0]

    quarter_breakdown_post = con.execute(f"""
        SELECT receive_quarter,
               COUNT(*) AS total_rows,
               COUNT(DISTINCT safetyreportid) AS distinct_ids
        FROM '{PARQUET_DEDUPED}'
        WHERE safetyreportid IS NOT NULL
        GROUP BY receive_quarter
        ORDER BY receive_quarter
    """).df()

    rows_removed = total_rows - total_rows2
    pct_removed  = 100 * rows_removed / total_rows if total_rows > 0 else 0

    print(f"\n  Righe totali (post):       {total_rows2:>10,}")
    print(f"  Righe rimosse:             {rows_removed:>10,}  ({pct_removed:.1f}%)")
    print(f"  Residual dups check:       {residual_dups}  "
          f"{'✓ OK' if residual_dups == 0 else '✗ ATTENZIONE'}")

    # Tabella comparativa per quarter
    merged = quarter_breakdown_pre.merge(
        quarter_breakdown_post, on="receive_quarter", suffixes=("_pre", "_post")
    )
    merged["rows_removed"] = merged["total_rows_pre"] - merged["total_rows_post"]
    merged["pct_removed"]  = (100 * merged["rows_removed"] / merged["total_rows_pre"]).round(1)

    col_q   = max(len("Quarter"), merged["receive_quarter"].str.len().max())
    col_pre = max(len("Righe pre"), 10)
    col_pos = max(len("Righe post"), 10)
    col_rem = max(len("Rimosse"), 8)
    col_pct = max(len("% rim."), 6)

    def row_fmt(q, pre, post, rem, pct):
        return (f"  {str(q):<{col_q}}  {str(pre):>{col_pre}}  "
                f"{str(post):>{col_pos}}  {str(rem):>{col_rem}}  {str(pct):>{col_pct}}")

    header = row_fmt("Quarter", "Righe pre", "Righe post", "Rimosse", "% rim.")
    sep    = "  " + "-"*col_q + "  " + "-"*col_pre + "  " + "-"*col_pos + \
             "  " + "-"*col_rem + "  " + "-"*col_pct

    table_lines = [header, sep]
    for _, r in merged.iterrows():
        table_lines.append(row_fmt(
            r["receive_quarter"],
            f"{int(r['total_rows_pre']):,}",
            f"{int(r['total_rows_post']):,}",
            f"{int(r['rows_removed']):,}",
            f"{r['pct_removed']}%",
        ))
    table_lines.append(sep)
    table_lines.append(row_fmt(
        "TOTALE",
        f"{total_rows:,}",
        f"{total_rows2:,}",
        f"{rows_removed:,}",
        f"{pct_removed:.1f}%",
    ))

    report_text = f"""
================================================================================
FAERS DEDUPLICATION REPORT
================================================================================
Data esecuzione  : {run_ts}
Python / DuckDB  : {platform.python_version()} / {duckdb.__version__}
File input       : {PARQUET_FLAT}
File output      : {PARQUET_DEDUPED}

--------------------------------------------------------------------------------
1. CONTESTO E MOTIVAZIONE
--------------------------------------------------------------------------------
  a) Follow-up report: stesso safetyreportid, receivedate più recente.
     → Teniamo solo la versione più recente.
  b) Duplicati veri: safetyreportid diversi, stesso evento.
     → Non gestiti (richiedono algoritmi probabilistici, fuori scope).
  c) Overlap tra file bulk: un report Q1 finisce nel file Q2.
     → Gestito con la stessa logica del caso (a).

--------------------------------------------------------------------------------
2. RISULTATI
--------------------------------------------------------------------------------
  Righe totali (pre)          : {total_rows:>12,}
  safetyreportid distinti     : {distinct_ids:>12,}
  Report con >1 receivedate   : {dup_reports:>12,}
  Righe in report duplicati   : {rows_in_dup_reports:>12,}

  Righe totali (post)         : {total_rows2:>12,}
  Righe rimosse               : {rows_removed:>12,}  ({pct_removed:.1f}% del totale)
  Righe NULL report_id        : {null_rows2:>12,}   (invariate, come atteso)

  Verifica residui: {residual_dups} {'✓ PASS' if residual_dups == 0 else '✗ FAIL — investigare'}

  Breakdown per quarter:

{chr(10).join(table_lines)}

--------------------------------------------------------------------------------
3. LIMITAZIONI DOCUMENTATE
--------------------------------------------------------------------------------
  - I duplicati veri (caso b) non sono gestiti. Impatto stimato < 5%.
  - Le righe con safetyreportid NULL (legacy pre-2012) non sono deduplicabili
    e rappresentano il {100*null_rows/total_rows:.1f}% del dataset.

================================================================================
END OF REPORT
================================================================================
""".strip()

    DEDUP_REPORT.write_text(report_text, encoding="utf-8")
    print(f"\n  Report QC salvato: {DEDUP_REPORT}")
    con.close()


# MAIN

def main():
    # Legge la lista dei file scaricati dal manifest scritto da run_download.py
    manifest_path = RAW_DIR / "downloaded.txt"

    if not manifest_path.exists():
        # Fallback: usa tutti i .json presenti in data/raw/
        print("  [WARN] downloaded.txt non trovato, uso tutti i .json in data/raw/")
        json_files = sorted(RAW_DIR.glob("*.json"))
    else:
        quarters   = manifest_path.read_text(encoding="utf-8").splitlines()
        json_files = [RAW_DIR / f"{q}.json" for q in quarters
                      if (RAW_DIR / f"{q}.json").exists()]

    if not json_files:
        print("[ERROR] Nessun file JSON trovato in data/raw/. Esegui prima run_download.py.")
        return

    print(f"=== STEP 1: Flattening {len(json_files)} file → faers_flat.parquet ===")
    run_flatten(json_files)

    print("\n=== STEP 2: Deduplicazione → faers_flat_deduped.parquet ===")
    run_deduplication()

    print("\n=== FLATTEN COMPLETATO ===")
    print(f"  Output: {PARQUET_DEDUPED}")


if __name__ == "__main__":
    main()