"""
src/contingency_table.py — versione ottimizzata

Ottimizzazioni rispetto alla versione precedente:
1. DuckDB con thread e memoria configurabili (default: 4 thread, 2GB)
2. View unica sul Parquet — il file viene aperto una sola volta
   invece di 4 volte (una per CTE). Su dataset grandi questo dimezza
   il tempo di I/O.
3. Parametri duckdb_threads e duckdb_memory passati da run_signals.py
"""

import duckdb
import pandas as pd


def build_contingency_table(
        parquet_path:    str,
        target_drug:     str,
        pt_col:          str = "reaction_pt",
        min_a:           int = 3,
        where_extra:     str = None,
        duckdb_threads:  int = 2,
        duckdb_memory:   str = "1500MB",
) -> pd.DataFrame:
    """
    Costruisce la contingency table 2x2 per tutte le coppie (target_drug, PT).

    Logica delle 4 celle (standard disproportionality analysis):
        a = report con target_drug E questo PT
        b = report con target_drug E altri PT
        c = report con altri drug  E questo PT
        d = report con altri drug  E altri PT
        n = totale report nel background

    Parameters
    ----------
    parquet_path   : path al Parquet deduplicato
    target_drug    : nome farmaco in UPPERCASE
    pt_col         : colonna reazioni MedDRA PT
    min_a          : soglia minima cella a
    where_extra    : clausola SQL per stratificazione (senza WHERE)
    duckdb_threads : thread DuckDB (default 4)
    duckdb_memory  : memoria DuckDB (default "2GB")

    Returns
    -------
    pd.DataFrame con colonne [drug, pt, a, b, c, d, n]
    """
    base_filter = f"drug_name = '{target_drug}'"

    if where_extra:
        full_filter = f"{base_filter} AND {where_extra}"
        bg_where    = f"WHERE {where_extra}"
    else:
        full_filter = base_filter
        bg_where    = ""

    # Connessione con configurazione esplicita per ambienti container
    con = duckdb.connect()
    con.execute(f"SET threads = {duckdb_threads}")
    con.execute(f"SET memory_limit = '{duckdb_memory}'")
    con.execute("SET preserve_insertion_order = false")  # libera ~20% memoria
    con.execute("SET temp_directory = '/app/data/duckdb_tmp'")  # spill su disco se OOM
    import os; os.makedirs("/app/data/duckdb_tmp", exist_ok=True)

    # View unica — il Parquet viene aperto una sola volta
    # Tutte le CTE sotto referenziano 'faers' invece di riaprire il file
    con.execute(f"CREATE VIEW faers AS SELECT * FROM read_parquet('{parquet_path}')")

    query = f"""
    WITH

    target_reports AS (
        SELECT DISTINCT safetyreportid AS report_id
        FROM faers
        WHERE {full_filter}
    ),

    background_reports AS (
        SELECT DISTINCT safetyreportid AS report_id
        FROM faers
        {bg_where}
    ),

    totals AS (
        SELECT COUNT(DISTINCT report_id) AS n
        FROM background_reports
    ),

    drug_marginal AS (
        SELECT COUNT(DISTINCT report_id) AS n_drug
        FROM target_reports
    ),

    target_pts AS (
        SELECT DISTINCT p.safetyreportid AS report_id, p.{pt_col} AS pt
        FROM faers p
        JOIN target_reports t ON p.safetyreportid = t.report_id
        WHERE p.drug_name = '{target_drug}'
          AND p.{pt_col} IS NOT NULL
          AND p.{pt_col} != ''
    ),

    pair_counts AS (
        SELECT pt, COUNT(DISTINCT report_id) AS a
        FROM target_pts
        GROUP BY pt
    ),

    pt_marginal AS (
        SELECT p.{pt_col} AS pt, COUNT(DISTINCT p.safetyreportid) AS n_pt
        FROM faers p
        JOIN background_reports b ON p.safetyreportid = b.report_id
        WHERE p.{pt_col} IS NOT NULL
          AND p.{pt_col} != ''
        GROUP BY p.{pt_col}
    )

    SELECT
        '{target_drug}'                          AS drug,
        pc.pt                                    AS pt,
        pc.a                                     AS a,
        (dm.n_drug - pc.a)                       AS b,
        (pm.n_pt   - pc.a)                       AS c,
        (t.n - dm.n_drug - pm.n_pt + pc.a)       AS d,
        t.n                                      AS n
    FROM pair_counts   pc
    JOIN pt_marginal   pm ON pc.pt = pm.pt
    CROSS JOIN drug_marginal dm
    CROSS JOIN totals        t
    WHERE pc.a >= {min_a}
      AND (dm.n_drug - pc.a)                 >= 0
      AND (pm.n_pt   - pc.a)                 >= 0
      AND (t.n - dm.n_drug - pm.n_pt + pc.a) >= 0
    ORDER BY pc.a DESC
    """

    result = con.execute(query).df()
    con.close()
    return result


def qc_contingency_table(df: pd.DataFrame, label: str = "") -> None:
    """
    Stampa un report di sanità sulla contingency table.
    Verifica: a+b+c+d == n, nessuna cella negativa, distribuzione di a.
    """
    if df.empty:
        print(f"  [QC] {label}: DataFrame vuoto")
        return

    print(f"\n[{label}] QC — {len(df)} coppie (drug, PT)")

    ok = ((df["a"] + df["b"] + df["c"] + df["d"]) == df["n"]).all()
    print(f"  a+b+c+d == n : {'OK' if ok else 'FAIL'}")

    neg = {col: int((df[col] < 0).sum()) for col in ["a", "b", "c", "d"]}
    print(f"  Celle negative: " + "  ".join(f"{k}={v}" for k, v in neg.items()))

    print(f"  a — min={df['a'].min()}  median={df['a'].median():.0f}  "
          f"max={df['a'].max()}  sum={df['a'].sum()}")
    print(f"  n — valore unico: {df['n'].nunique() == 1}  ({df['n'].iloc[0]})")

    print(f"  Top 5 PT per a:")
    print(df.nlargest(5, "a")[["pt", "a"]].to_string(index=False))