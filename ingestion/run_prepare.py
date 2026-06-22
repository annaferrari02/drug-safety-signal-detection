"""
ingestion/run_prepare.py

Pre-aggregazione one-shot dei datamart di supporto alla contingency table.
Va eseguito UNA SOLA VOLTA dopo run_flatten.py (o rebuild_deduped()).
Può essere rieseguito ogni volta che il dataset viene aggiornato.

Output in data/:
    marginals_cubed.parquet       — cubo OLAP sex × age_stratum × reaction_pt
                                    (~pochi MB, sostituisce il full-scan da 2.5 GB
                                    per filtri fissi sex/age)
    drug_inverted_index.parquet   — indice invertito drug_name → [safetyreportid]
                                    (rende istantanea l'intersezione per co-medication)

Tempi attesi (dataset ~20 GB flat, ~2.5 GB deduped):
    marginals_cubed       :  30-90s (window function su tutto il Parquet)
    drug_inverted_index   :  20-60s (GROUP BY + list aggregation)
"""

import sys
import time
from pathlib import Path

import duckdb

DATA_DIR = Path("/app/data")
PARQUET_DEDUPED = DATA_DIR / "faers_flat_deduped.parquet"
CUBED_OUT       = DATA_DIR / "marginals_cubed.parquet"
INDEX_OUT       = DATA_DIR / "drug_inverted_index.parquet"

# DuckDB tuning — adatta se il container ha più/meno RAM
DUCKDB_THREADS = 4
DUCKDB_MEMORY  = "3GB"


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads = {DUCKDB_THREADS}")
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY}'")
    con.execute("SET preserve_insertion_order = false")
    tmp = DATA_DIR / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp}'")
    return con


def build_marginals_cubed(con: duckdb.DuckDBPyConnection) -> None:
    """
    Cubo OLAP: per ogni combinazione (sex, age_stratum, reaction_pt) calcola:
        n_pt_stratum  — report con quel PT dentro lo strato
        n_stratum     — totale report in quello strato (tutti i PT)

    La window function SUM(...) OVER(PARTITION BY sex, age_stratum) permette
    di calcolare n_stratum in un solo passaggio, senza un secondo GROUP BY.

    Nota: usiamo COUNT(DISTINCT safetyreportid) perché un report può comparire
    più volte nel Parquet flat (una riga per coppia drug×reaction).
    """
    print("\n[CUBO ] Costruzione marginals_cubed.parquet...", flush=True)
    t0 = time.time()

    # Strati fissi supportati — devono corrispondere ai valori reali nel dataset
    # 'unknown' viene escluso perché i filtri UI lo ignorano già.
    VALID_SEX    = ("'male'", "'female'")
    VALID_AGE    = ("'pediatric'", "'adult'", "'geriatric'")

    con.execute(f"""
        COPY (
            WITH base AS (
                SELECT
                    sex,
                    age_stratum,
                    reaction_pt,
                    safetyreportid
                FROM read_parquet('{PARQUET_DEDUPED}')
                WHERE sex         IN ({', '.join(VALID_SEX)})
                  AND age_stratum IN ({', '.join(VALID_AGE)})
                  AND reaction_pt IS NOT NULL
                  AND reaction_pt != ''
            ),
            counts AS (
                SELECT
                    sex,
                    age_stratum,
                    reaction_pt,
                    COUNT(DISTINCT safetyreportid) AS n_pt_stratum
                FROM base
                GROUP BY sex, age_stratum, reaction_pt
            )
            SELECT
                sex,
                age_stratum,
                reaction_pt,
                n_pt_stratum,
                SUM(n_pt_stratum) OVER (PARTITION BY sex, age_stratum)
                    AS n_stratum
            FROM counts
        )
        TO '{CUBED_OUT}'
        (FORMAT 'PARQUET', COMPRESSION 'SNAPPY')
    """)

    size_mb = CUBED_OUT.stat().st_size / 1e6
    print(f"[CUBO ] Completato in {time.time()-t0:.1f}s — {size_mb:.1f} MB", flush=True)


def build_drug_inverted_index(con: duckdb.DuckDBPyConnection) -> None:
    """
    Indice invertito: drug_name → array di safetyreportid unici.

    Permette di calcolare l'intersezione di due insiemi di report
    (target drug ∩ co-medication) via list_intersect() di DuckDB,
    evitando un full-scan con subquery IN (...).

    Il tipo INT[] (lista di interi) è nativo in DuckDB/Parquet e viene
    letto back in Python come lista NumPy/Python standard.
    """
    print("\n[IDX  ] Costruzione drug_inverted_index.parquet...", flush=True)
    t0 = time.time()

    con.execute(f"""
        COPY (
            SELECT
                drug_name,
                list(DISTINCT safetyreportid) AS report_ids
            FROM read_parquet('{PARQUET_DEDUPED}')
            WHERE drug_name IS NOT NULL
              AND drug_name != ''
            GROUP BY drug_name
        )
        TO '{INDEX_OUT}'
        (FORMAT 'PARQUET', COMPRESSION 'SNAPPY')
    """)

    size_mb = INDEX_OUT.stat().st_size / 1e6
    print(f"[IDX  ] Completato in {time.time()-t0:.1f}s — {size_mb:.1f} MB", flush=True)


def verify_outputs() -> bool:
    """Verifica veloce che i file siano stati scritti e non siano vuoti."""
    ok = True
    for path in (CUBED_OUT, INDEX_OUT):
        if not path.exists() or path.stat().st_size < 1000:
            print(f"[ERR  ] {path.name} mancante o troppo piccolo!", flush=True)
            ok = False
        else:
            con = duckdb.connect()
            n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
            con.close()
            print(f"[OK   ] {path.name} — {n:,} righe", flush=True)
    return ok


def main() -> None:
    if not PARQUET_DEDUPED.exists():
        print(f"[ERR  ] Dataset non trovato: {PARQUET_DEDUPED}")
        print("        Esegui prima run_flatten.py (o update_ingestion.py).")
        sys.exit(1)

    size_gb = PARQUET_DEDUPED.stat().st_size / 1e9
    print(f"\n{'='*55}")
    print(f"  run_prepare.py — pre-aggregazione datamart")
    print(f"  Dataset: {PARQUET_DEDUPED.name} ({size_gb:.2f} GB)")
    print(f"  Output : {DATA_DIR}")
    print(f"{'='*55}")

    t_total = time.time()
    con = _connect()

    build_marginals_cubed(con)
    build_drug_inverted_index(con)

    con.close()

    print(f"\n{'='*55}")
    print(f"  Verifica output...")
    ok = verify_outputs()
    elapsed = time.time() - t_total
    print(f"\n  Tempo totale: {elapsed:.1f}s")
    print(f"  Stato: {'✓ OK' if ok else '✗ ERRORI — controlla i log'}")
    print(f"{'='*55}\n")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()