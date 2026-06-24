#script for trimestral update of the dataset (when openFDA is updated)

"""
ingestion/run_prepare.py

Pre-aggregazione one-shot dei datamart di supporto alla contingency table.
Va eseguito UNA SOLA VOLTA dopo run_flatten.py (o rebuild_deduped()).
Può essere rieseguito ogni volta che il dataset viene aggiornato.

Output in data/:
    faers_sorted.parquet          — dataset flat ordinato per drug_name con row
                                    group da 50k righe. Usato da Route A e D.
                                    DuckDB salta interi row group grazie alle
                                    statistiche min/max per colonna (pruning).

    marginals_global.parquet      — margini pre-calcolati sull'intero dataset:
                                    n_total (report distinti) + n_pt per ogni
                                    reaction_pt. Usato da Route A.

    marginals_cubed.parquet       — cubo OLAP sex × age_stratum × reaction_pt
                                    (~pochi MB). Usato da Route B per filtri
                                    fissi sex/age senza full-scan.

    drug_inverted_index.parquet   — indice invertito drug_name → [safetyreportid]
                                    (rende istantanea l'intersezione per
                                    co-medication). Usato da Route C.

Tempi attesi (dataset ~2.5 GB deduped):
    faers_sorted          :  60-120s (sort globale, spill su disco)
    marginals_global      :  20-40s  (COUNT DISTINCT su tutto il Parquet)
    marginals_cubed       :  30-90s  (window function su tutto il Parquet)
    drug_inverted_index   :  20-60s  (GROUP BY + list aggregation)

Esecuzione:
    docker compose run --profile pipeline ingestion python run_prepare.py
    docker compose run --profile pipeline ingestion python run_prepare.py --force
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

DATA_DIR        = Path("/app/data")
PARQUET_DEDUPED = DATA_DIR / "faers_flat_deduped.parquet"
PARQUET_SORTED  = DATA_DIR / "faers_sorted.parquet"
MARGINALS_OUT   = DATA_DIR / "marginals_global.parquet"
CUBED_OUT       = DATA_DIR / "marginals_cubed.parquet"
INDEX_OUT       = DATA_DIR / "drug_inverted_index.parquet"

ROW_GROUP_SIZE = 50_000   # righe per row group in faers_sorted.parquet
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


def _needs_rebuild(source: Path, *targets: Path) -> bool:
    """
    Restituisce True se almeno uno dei target non esiste o è più vecchio
    della sorgente.
    """
    if not source.exists():
        raise FileNotFoundError(
            f"{source} non trovato. "
            "Esegui prima run_flatten.py per produrre il Parquet deduplicato."
        )
    source_mtime = source.stat().st_mtime
    for t in targets:
        if not t.exists() or t.stat().st_mtime < source_mtime:
            return True
    return False


# ── Route A (1/2): faers_sorted.parquet ──────────────────────────────────────

def build_sorted_parquet(con: duckdb.DuckDBPyConnection) -> None:
    """
    Legge faers_flat_deduped.parquet, ordina per drug_name e scrive
    faers_sorted.parquet con row group da ROW_GROUP_SIZE righe.

    Usa COPY TO di DuckDB invece di caricare tutto in una Arrow Table:
    DuckDB gestisce internamente lo spill su disco durante il sort,
    evitando il picco di RAM che causava OOM con fetch_arrow_table().
    """
    print("\n[SORTED] Ordinamento e scrittura per drug_name...", flush=True)
    t0 = time.time()

    con.execute(f"""
        COPY (
            SELECT *
            FROM '{PARQUET_DEDUPED}'
            ORDER BY drug_name NULLS LAST
        )
        TO '{PARQUET_SORTED}'
        (
            FORMAT PARQUET,
            ROW_GROUP_SIZE {ROW_GROUP_SIZE},
            COMPRESSION SNAPPY
        )
    """)

    size_mb = PARQUET_SORTED.stat().st_size / 1e6
    print(f"[SORTED] Scritto {PARQUET_SORTED.name} ({size_mb:.0f} MB) "
          f"in {time.time()-t0:.1f}s", flush=True)


# ── Route A (2/2): marginals_global.parquet ───────────────────────────────────

def build_marginals_global(con: duckdb.DuckDBPyConnection) -> None:
    """
    Calcola i margini globali da faers_flat_deduped.parquet.

    Schema output:
        pt        : str    — reaction_pt (MedDRA preferred term)
        n_pt      : int64  — safetyreportid distinti con questa reazione
        n_total   : int64  — safetyreportid distinti nell'intero dataset

    n_total è replicato su ogni riga per semplicità di join in DuckDB
    (evita un cross join separato nella query CT).
    """
    print("\n[MARG ] Calcolo marginals_global.parquet...", flush=True)
    t0 = time.time()

    n_total = con.execute(f"""
        SELECT COUNT(DISTINCT safetyreportid) AS n
        FROM '{PARQUET_DEDUPED}'
        WHERE safetyreportid IS NOT NULL
    """).fetchone()[0]

    print(f"[MARG ] n_total = {n_total:,} report distinti", flush=True)

    con.execute(f"""
        COPY (
            SELECT
                reaction_pt                        AS pt,
                COUNT(DISTINCT safetyreportid)     AS n_pt,
                {n_total}                          AS n_total
            FROM '{PARQUET_DEDUPED}'
            WHERE reaction_pt IS NOT NULL
              AND reaction_pt != ''
              AND safetyreportid IS NOT NULL
            GROUP BY reaction_pt
            ORDER BY n_pt DESC
        )
        TO '{MARGINALS_OUT}'
        (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    n_pts = pq.read_metadata(str(MARGINALS_OUT)).num_rows
    print(f"[MARG ] {n_pts:,} PT distinti — scritto {MARGINALS_OUT.name} "
          f"in {time.time()-t0:.1f}s", flush=True)


# ── Route B: marginals_cubed.parquet ─────────────────────────────────────────

def build_marginals_cubed(con: duckdb.DuckDBPyConnection) -> None:
    """
    Cubo OLAP: per ogni combinazione (sex, age_stratum, reaction_pt) calcola:
        n_pt_stratum  — report con quel PT dentro lo strato
        n_stratum     — totale report in quello strato (tutti i PT)

    La window function SUM(...) OVER(PARTITION BY sex, age_stratum) permette
    di calcolare n_stratum in un solo passaggio, senza un secondo GROUP BY.
    """
    print("\n[CUBO ] Costruzione marginals_cubed.parquet...", flush=True)
    t0 = time.time()

    VALID_SEX = ("'male'", "'female'")
    VALID_AGE = ("'pediatric'", "'adult'", "'geriatric'")

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
        (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    size_mb = CUBED_OUT.stat().st_size / 1e6
    print(f"[CUBO ] Completato in {time.time()-t0:.1f}s — {size_mb:.1f} MB", flush=True)


# ── Route C: drug_inverted_index.parquet ──────────────────────────────────────

def build_drug_inverted_index(con: duckdb.DuckDBPyConnection) -> None:
    """
    Indice invertito: drug_name → array di safetyreportid unici.

    Permette di calcolare l'intersezione di due insiemi di report
    (target drug ∩ co-medication) via list_intersect() di DuckDB,
    evitando un full-scan con subquery IN (...).
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
        (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    size_mb = INDEX_OUT.stat().st_size / 1e6
    print(f"[IDX  ] Completato in {time.time()-t0:.1f}s — {size_mb:.1f} MB", flush=True)


# ── Verifica output ───────────────────────────────────────────────────────────

def verify_outputs() -> bool:
    """Verifica veloce che tutti i file siano stati scritti e non siano vuoti."""
    ok = True
    for path in (PARQUET_SORTED, MARGINALS_OUT, CUBED_OUT, INDEX_OUT):
        if not path.exists() or path.stat().st_size < 1000:
            print(f"[ERR  ] {path.name} mancante o troppo piccolo!", flush=True)
            ok = False
        else:
            con = duckdb.connect()
            n = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{path}')"
            ).fetchone()[0]
            con.close()
            size_mb = path.stat().st_size / 1e6
            print(f"[OK   ] {path.name} — {n:,} righe, {size_mb:.0f} MB", flush=True)
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if not PARQUET_DEDUPED.exists():
        print(f"[ERR  ] Dataset non trovato: {PARQUET_DEDUPED}")
        print("        Esegui prima run_flatten.py (o update_ingestion.py).")
        sys.exit(1)

    needs = force or _needs_rebuild(
        PARQUET_DEDUPED,
        PARQUET_SORTED,
        MARGINALS_OUT,
        CUBED_OUT,
        INDEX_OUT,
    )

    if not needs:
        print(
            f"[PREPARE] Tutti i file sono aggiornati rispetto a "
            f"{PARQUET_DEDUPED.name}. Usa --force per rigenerare."
        )
        return

    size_gb = PARQUET_DEDUPED.stat().st_size / 1e9
    t_start = time.time()

    print(f"\n{'='*55}")
    print(f"  run_prepare.py — pre-aggregazione datamart")
    print(f"  Dataset: {PARQUET_DEDUPED.name} ({size_gb:.2f} GB)")
    print(f"  Output : {DATA_DIR}")
    print(f"{'='*55}")

    con = _connect()

    build_sorted_parquet(con)       # Route A (1/2) — deve venire prima di cubed/index
    build_marginals_global(con)     # Route A (2/2)
    build_marginals_cubed(con)      # Route B
    build_drug_inverted_index(con)  # Route C

    con.close()

    print(f"\n{'='*55}")
    print(f"  Verifica output...")
    ok = verify_outputs()
    elapsed = time.time() - t_start
    print(f"\n  Tempo totale: {elapsed:.1f}s")
    print(f"  Stato: {'✓ OK' if ok else '✗ ERRORI — controlla i log'}")
    print(f"{'='*55}\n")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara i file ottimizzati per la contingency table (tutte le Route)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rigenera anche se i file sono già aggiornati"
    )
    args = parser.parse_args()
    main(force=args.force)