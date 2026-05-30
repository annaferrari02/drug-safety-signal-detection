import duckdb
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

PARQUET_PATH = "data/faers_flat.parquet"

def build_contingency_table_datestrat(
        parquet_path : str,
        target_drug  : str,
        pt_col       : str                            = "reaction_pt",
        min_a        : int                            = 3,
        where_extra  : str                            = None,
        date_range   : Optional[Tuple[str, str]]      = None,   # <-- NUOVO
        date_col     : str                            = "receivedate",  # <-- NUOVO
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
    date_range : tuple(str, str) | None
        Range di date (inclusivo) nel formato ('YYYY-MM-DD', 'YYYY-MM-DD').
        Esempio: ('2015-01-01', '2020-12-31')
    date_col : str
        Colonna della data nel Parquet, default "receivedate".

    Returns
    -------
    pd.DataFrame con colonne [drug, pt, a, b, c, d, n]
    """

    # ── Costruisci il filtro data ────────────────────────────────────────────────
    def build_date_filter(date_range, date_col):
        if date_range is None:
            return None
        start, end = date_range
        return f"{date_col} BETWEEN '{start}' AND '{end}'"

    date_filter = build_date_filter(date_range, date_col)

    # ── Combina tutti i filtri ───────────────────────────────────────────────────
    filters = [f"drug_name = '{target_drug}'"]
    if date_filter:
        filters.append(date_filter)
    if where_extra:
        filters.append(where_extra)

    full_filter = " AND ".join(filters)

    # background: stessa stratificazione demografica + date, senza drug filter
    bg_filters = []
    if date_filter:
        bg_filters.append(date_filter)
    if where_extra:
        bg_filters.append(where_extra)

    bg_filter = " AND ".join(bg_filters) if bg_filters else None
    bg_where  = f"WHERE {bg_filter}" if bg_filter else ""

    query = f"""
    WITH

    -- ── Sottoinsieme: report che contengono il target drug (+ eventuali filtri) ──
    target_reports AS (
        SELECT DISTINCT safetyreportid AS report_id
        FROM '{parquet_path}'
        WHERE {full_filter}
    ),

    -- ── Background: TUTTI i report nel sottoinsieme (demografico + date) ─────────
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


# def qc_contingency_table(df: pd.DataFrame, label: str = "") -> None:
#     tag = f"[{label}] " if label else ""
#     check = df["a"] + df["b"] + df["c"] + df["d"]
#     bad   = (check != df["n"]).sum()

#     print(f"\n{tag}QC Report — {len(df)} coppie (drug, PT)")
#     print(f"  a+b+c+d == n : {'OK' if bad == 0 else f'FAIL ({bad} righe)'}")
#     print(f"  Celle negative: a={( df['a']<0).sum()} b={(df['b']<0).sum()} "
#           f"c={(df['c']<0).sum()} d={(df['d']<0).sum()}")
#     print(f"  a  — min={df['a'].min()}  median={df['a'].median():.0f}  "
#           f"max={df['a'].max()}  sum={df['a'].sum()}")
#     print(f"  n  — valore unico: {df['n'].nunique()==1}  "
#           f"({df['n'].iloc[0] if len(df) else 'n/a'})")
#     print(f"  Top 5 PT per a:\n{df[['pt','a']].head(5).to_string(index=False)}")