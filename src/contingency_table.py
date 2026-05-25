import duckdb
import pandas as pd
from pathlib import Path

PARQUET_PATH = "data/faers_flat.parquet"

# FUNZIONE CORE

def build_contingency_table(
        parquet_path : str,
        target_drug  : str,
        pt_col       : str  = "reaction_pt",
        min_a        : int  = 3,
        where_extra  : str  = None,
) -> pd.DataFrame:
    """
    Costruisce la contingency table 2x2 per tutte le coppie (drug, PT)
    del target_drug, usando DuckDB direttamente sul Parquet.

    Parameters
    ----------
    parquet_path : str
        Path al Parquet deduplificato (output di 02_flatten).
    target_drug : str
        Nome del drug target in uppercase, es. "LAPATINIB".
    pt_col : str
        Colonna delle reazioni, default "reaction_pt".
    min_a : int
        Soglia minima per la cella a (default 3, criterio letteratura).
    where_extra : str | None
        Clausola WHERE aggiuntiva per la stratificazione, senza il WHERE.
        Esempi:
            "sex = 'female'"
            "age_stratum = 'geriatric'"
            "sex = 'female' AND age_stratum = 'adult'"
            "safetyreportid IN (
                SELECT DISTINCT safetyreportid
                FROM 'data/faers_flat.parquet'
                WHERE drug_name = 'CAPECITABINE'
             )"

    Returns
    -------
    pd.DataFrame con colonne [drug, pt, a, b, c, d, n]
    """

    # Componi il filtro WHERE
    base_filter = f"drug_name = '{target_drug}'"
    if where_extra:
        full_filter = f"{base_filter} AND {where_extra}"
        bg_filter   = where_extra          # background: stessa stratificazione, tutti i drug
    else:
        full_filter = base_filter
        bg_filter   = None                 # background: tutto il dataset

    bg_where = f"WHERE {bg_filter}" if bg_filter else ""

    query = f"""
    WITH

    -- ── Sottoinsieme: report che contengono il target drug (+ eventuali filtri) ──
    target_reports AS (
        SELECT DISTINCT safetyreportid AS report_id
        FROM '{parquet_path}'
        WHERE {full_filter}
    ),

    -- ── Background: TUTTI i report nel sottoinsieme demografico ─────────────────
    -- (se non c'è stratificazione demografica = tutti i report del dataset)
    background_reports AS (
        SELECT DISTINCT safetyreportid AS report_id
        FROM '{parquet_path}'
        {bg_where}
    ),

    -- ── N: totale report nel background ─────────────────────────────────────────
    totals AS (
        SELECT COUNT(DISTINCT report_id) AS n
        FROM background_reports
    ),

    -- ── a+b: report con target_drug nel background ───────────────────────────────
    drug_marginal AS (
        SELECT COUNT(DISTINCT report_id) AS n_drug
        FROM target_reports
    ),

    -- ── Reazioni del target drug (per calcolare a e b) ───────────────────────────
    target_pts AS (
        SELECT DISTINCT p.safetyreportid AS report_id, p.{pt_col} AS pt
        FROM '{parquet_path}' p
        JOIN target_reports t ON p.safetyreportid = t.report_id
        WHERE p.drug_name = '{target_drug}'
          AND p.{pt_col} IS NOT NULL
          AND p.{pt_col} != ''
    ),

    -- ── a: report con target_drug E questo PT ────────────────────────────────────
    pair_counts AS (
        SELECT pt, COUNT(DISTINCT report_id) AS a
        FROM target_pts
        GROUP BY pt
    ),

    -- ── a+c: report nel background che hanno questo PT (qualsiasi drug) ──────────
    pt_marginal AS (
        SELECT p.{pt_col} AS pt, COUNT(DISTINCT p.safetyreportid) AS n_pt
        FROM '{parquet_path}' p
        JOIN background_reports b ON p.safetyreportid = b.report_id
        WHERE p.{pt_col} IS NOT NULL
          AND p.{pt_col} != ''
        GROUP BY p.{pt_col}
    )

    SELECT
        '{target_drug}'              AS drug,
        pc.pt                        AS pt,
        pc.a                         AS a,
        (dm.n_drug    - pc.a)        AS b,
        (pm.n_pt      - pc.a)        AS c,
        (t.n - dm.n_drug - pm.n_pt + pc.a) AS d,
        t.n                          AS n
    FROM pair_counts  pc
    JOIN pt_marginal  pm ON pc.pt = pm.pt
    CROSS JOIN drug_marginal dm
    CROSS JOIN totals        t
    WHERE pc.a >= {min_a}
      AND (dm.n_drug - pc.a)                    >= 0
      AND (pm.n_pt   - pc.a)                    >= 0
      AND (t.n - dm.n_drug - pm.n_pt + pc.a)    >= 0
    ORDER BY pc.a DESC
    """

    return duckdb.connect().execute(query).df()


# QC: verifica sanità delle 4 celle

 
def qc_contingency_table(df: pd.DataFrame, label: str = "") -> None:
    """
    Stampa un report di sanità sulla contingency table:
        - a + b + c + d == n per ogni riga
        - nessuna cella negativa
        - distribuzione di a
    """
    tag = f"[{label}] " if label else ""
 
    # Verifica a + b + c + d = n
    check = df["a"] + df["b"] + df["c"] + df["d"]
    bad   = (check != df["n"]).sum()
 
    print(f"\n{tag}QC Report — {len(df)} coppie (drug, PT)")
    print(f"  a+b+c+d == n : {'OK' if bad == 0 else f'FAIL ({bad} righe)'}")
    print(f"  Celle negative: a={( df['a']<0).sum()} b={(df['b']<0).sum()} "
          f"c={(df['c']<0).sum()} d={(df['d']<0).sum()}")
    print(f"  a  — min={df['a'].min()}  median={df['a'].median():.0f}  "
          f"max={df['a'].max()}  sum={df['a'].sum()}")
    print(f"  n  — valore unico: {df['n'].nunique()==1}  "
          f"({df['n'].iloc[0] if len(df) else 'n/a'})")
    print(f"  Top 5 PT per a:\n{df[['pt','a']].head(5).to_string(index=False)}")