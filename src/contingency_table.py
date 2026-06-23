"""
src/contingency_table.py — versione dual-route

Route di esecuzione in base al tipo di filtro:

    ROUTE A — nessun filtro (FAST PATH)
        faers_sorted.parquet + marginals_global.parquet
        Row group pruning su drug_name + join su marginali pre-calcolati
        Tempo: ~2-5s

    ROUTE B — filtri fissi (sex / age_stratum)
        marginals_cubed.parquet + faers_sorted.parquet (solo target drug)
        Push-down sul cubo per i marginali dello strato, pruning per il drug
        Tempo: <1s

    ROUTE C — co-medication (subquery su drug_name)
        drug_inverted_index.parquet + faers_sorted.parquet
        list_intersect() in RAM, poi scan solo sui report dell'intersezione
        Tempo: ~5-8s (prima volta), poi coperto da session_state cache

    ROUTE D — fallback (filtri dinamici non gestiti)
        faers_sorted.parquet con full scan + where_extra
        Comportamento identico alla versione originale con where_extra
        Tempo: ~60-90s

Fallback globale: se faers_sorted.parquet o marginals_global.parquet non
esistono, cade su faers_flat_deduped.parquet (parquet_path) con full scan.
"""

import os
import re
import duckdb
import pandas as pd
from pathlib import Path

_DATA_DIR       = Path("/app/data")
_SORTED_PARQUET = _DATA_DIR / "faers_sorted.parquet"
_MARGINALS      = _DATA_DIR / "marginals_global.parquet"
_CUBED          = _DATA_DIR / "marginals_cubed.parquet"
_INDEX          = _DATA_DIR / "drug_inverted_index.parquet"

_FIXED_COLS = {"sex", "age_stratum"}


def _make_con(threads: int, memory: str) -> duckdb.DuckDBPyConnection:
    tmp_dir = str(_DATA_DIR / "duckdb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET memory_limit = '{memory}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{tmp_dir}'")
    return con


# ─── Routing ────────────────────────────────────────────────────────────────

def _detect_route(where_extra: str | None) -> str:
    if where_extra is None:
        return "global"
    we = where_extra.strip()
    if re.search(r"safetyreportid\s+IN\s*\(", we, re.IGNORECASE):
        return "index"
    mentioned = {c.lower() for c in re.findall(r"\b([a-z_]+)\b\s*(?:=|IN\b|LIKE\b)", we, re.IGNORECASE)}
    if mentioned and mentioned.issubset(_FIXED_COLS):
        return "cubed"
    return "full"


# ─── Route A: sorted + marginals_global (fast path) ─────────────────────────

def _build_ct_global(
        target_drug: str, pt_col: str, min_a: int,
        threads: int, memory: str,
) -> pd.DataFrame:
    con = _make_con(threads, memory)
    sp = str(_SORTED_PARQUET)
    mp = str(_MARGINALS)

    query = f"""
    WITH
    drug_pt_pairs AS (
        SELECT DISTINCT safetyreportid AS report_id, {pt_col} AS pt
        FROM '{sp}'
        WHERE drug_name = '{target_drug}'
          AND {pt_col} IS NOT NULL AND {pt_col} != ''
    ),
    pair_counts AS (
        SELECT pt, COUNT(DISTINCT report_id) AS a
        FROM drug_pt_pairs
        GROUP BY pt
    ),
    drug_total AS (
        SELECT COUNT(DISTINCT report_id) AS n_drug
        FROM drug_pt_pairs
    ),
    marginals AS (
        SELECT pt, n_pt, n_total
        FROM '{mp}'
    )
    SELECT
        '{target_drug}'                              AS drug,
        pc.pt                                        AS pt,
        pc.a                                         AS a,
        (dt.n_drug   - pc.a)                         AS b,
        (mg.n_pt     - pc.a)                         AS c,
        (mg.n_total  - dt.n_drug - mg.n_pt + pc.a)   AS d,
        mg.n_total                                   AS n
    FROM pair_counts  pc
    JOIN marginals    mg ON pc.pt = mg.pt
    CROSS JOIN drug_total dt
    WHERE pc.a >= {min_a}
      AND (dt.n_drug   - pc.a)                    >= 0
      AND (mg.n_pt     - pc.a)                    >= 0
      AND (mg.n_total  - dt.n_drug - mg.n_pt + pc.a) >= 0
    ORDER BY pc.a DESC
    """
    result = con.execute(query).df()
    con.close()
    return result


# ─── Route B: cubo OLAP per filtri fissi ────────────────────────────────────

def _build_cube_filter(where_extra: str) -> str:
    """
    Estrae dal where_extra solo i predicati su sex e age_stratum,
    riscrivendoli in forma pulita per il cubo (niente newline, niente subquery).
    Restituisce una stringa "WHERE ..." oppure "" se nessun predicato trovato.
    """
    parts = []
    m_sex = re.search(r"sex\s*=\s*'([^']+)'", where_extra, re.IGNORECASE)
    if m_sex:
        parts.append(f"sex = '{m_sex.group(1)}'")
    m_age = re.search(r"age_stratum\s*=\s*'([^']+)'", where_extra, re.IGNORECASE)
    if m_age:
        parts.append(f"age_stratum = '{m_age.group(1)}'")
    return ("WHERE " + " AND ".join(parts)) if parts else ""


def _build_ct_cubed(
        target_drug: str, pt_col: str, min_a: int,
        where_extra: str, threads: int, memory: str,
) -> pd.DataFrame:
    con = _make_con(threads, memory)
    sp = str(_SORTED_PARQUET)
    cp = str(_CUBED)

    # Clausola WHERE sicura per il cubo: solo sex/age_stratum, senza newline
    # né subquery che il cubo non conosce.
    cube_where = _build_cube_filter(where_extra)

    # n_stratum nel cubo è il totale per sex×age_stratum.
    # Se filtriamo su entrambi → una sola riga per PT → SUM = valore corretto.
    # Se filtriamo solo su age_stratum → due righe per PT (male + female)
    #   → SUM(n_pt_stratum) aggrega correttamente i PT counts
    #   → SUM(DISTINCT n_stratum) sommerebbe valori distinti, ma n_stratum
    #     varia per sesso, quindi serve SUM dei totali di strato distinti.
    # Soluzione: calcoliamo n_stratum_total separatamente come somma dei
    # totali unici per ciascuna combinazione sex×age presente nel filtro.

    query = f"""
    WITH
    -- Marginali PT aggregati sullo strato richiesto
    -- SUM per gestire il caso "solo age_stratum" (righe male+female sommate)
    stratum_marginals AS (
        SELECT
            reaction_pt             AS pt,
            SUM(n_pt_stratum)       AS n_pt_stratum
        FROM '{cp}'
        {cube_where}
        GROUP BY reaction_pt
    ),
    -- Totale report nello strato: somma degli n_stratum distinti per sex×age.
    -- MAX(n_stratum) per ogni combinazione sex×age_stratum, poi somma.
    -- Questo funziona sia con filtro su entrambi (1 combo) sia solo su age (2 combo).
    stratum_total AS (
        SELECT SUM(max_n) AS n_stratum
        FROM (
            SELECT sex, age_stratum, MAX(n_stratum) AS max_n
            FROM '{cp}'
            {cube_where}
            GROUP BY sex, age_stratum
        )
    ),
    -- Solo le righe del target drug nello strato (row group pruning)
    drug_pt_pairs AS (
        SELECT DISTINCT safetyreportid AS report_id, {pt_col} AS pt
        FROM '{sp}'
        WHERE drug_name = '{target_drug}'
          AND {where_extra}
          AND {pt_col} IS NOT NULL AND {pt_col} != ''
    ),
    pair_counts AS (
        SELECT pt, COUNT(DISTINCT report_id) AS a
        FROM drug_pt_pairs
        GROUP BY pt
    ),
    drug_total AS (
        SELECT COUNT(DISTINCT report_id) AS n_drug
        FROM drug_pt_pairs
    )
    SELECT
        '{target_drug}'                                          AS drug,
        pc.pt                                                    AS pt,
        pc.a                                                     AS a,
        (dt.n_drug         - pc.a)                               AS b,
        (sm.n_pt_stratum   - pc.a)                               AS c,
        (st.n_stratum - dt.n_drug - sm.n_pt_stratum + pc.a)      AS d,
        st.n_stratum                                             AS n
    FROM pair_counts       pc
    JOIN stratum_marginals sm ON pc.pt = sm.pt
    CROSS JOIN drug_total  dt
    CROSS JOIN stratum_total st
    WHERE pc.a >= {min_a}
      AND (dt.n_drug         - pc.a)                           >= 0
      AND (sm.n_pt_stratum   - pc.a)                           >= 0
      AND (st.n_stratum - dt.n_drug - sm.n_pt_stratum + pc.a)  >= 0
    ORDER BY pc.a DESC
    """
    result = con.execute(query).df()
    con.close()
    return result


# ─── Route C: indice invertito per co-medication ────────────────────────────

def _extract_comedication_drug(where_extra: str) -> str | None:
    m = re.search(r"drug_name\s*=\s*'([^']+)'", where_extra, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _build_ct_index(
        target_drug: str, pt_col: str, min_a: int,
        where_extra: str, threads: int, memory: str,
) -> pd.DataFrame:
    co_drug = _extract_comedication_drug(where_extra)
    if co_drug is None:
        print(f"  [CT  ] Co-medication non riconosciuta — fallback Route D")
        return _build_ct_full(target_drug, pt_col, min_a, where_extra, threads, memory)

    con = _make_con(threads, memory)
    ip = str(_INDEX)
    sp = str(_SORTED_PARQUET)

    # Verifica che il co-drug esista nell'indice
    check = con.execute(f"""
        SELECT COUNT(*) FROM '{ip}' WHERE drug_name = '{co_drug}'
    """).fetchone()[0]
    if check == 0:
        con.close()
        print(f"  [CT  ] '{co_drug}' non nell'indice — fallback Route D")
        return _build_ct_full(target_drug, pt_col, min_a, where_extra, threads, memory)

    query = f"""
    WITH
    -- 1. IDs di tutti i report che contengono il co-farmaco (Popolazione di Background dello Strato)
    co_reports AS (
        SELECT UNNEST(report_ids) AS report_id 
        FROM '{ip}' 
        WHERE drug_name = '{co_drug}'
    ),
    -- 2. Intersezione in RAM per calcolare il totale del target drug nello strato
    intersection AS (
        SELECT list_intersect(
            (SELECT report_ids FROM '{ip}' WHERE drug_name = '{target_drug}'),
            (SELECT report_ids FROM '{ip}' WHERE drug_name = '{co_drug}')
        ) AS valid_ids
    ),
    target_co_reports AS (
        SELECT UNNEST(valid_ids) AS report_id FROM intersection
    ),
    -- 3. Totali complessivi dello strato co-medication
    totals AS (
        SELECT COUNT(*) AS n FROM co_reports
    ),
    drug_total AS (
        SELECT COUNT(*) AS n_drug FROM target_co_reports
    ),
    -- 4. Conteggio Cella A: Solo i report con entrambi i farmaci e il PT specifico
    target_pts AS (
        SELECT DISTINCT p.safetyreportid AS report_id, p.{pt_col} AS pt
        FROM '{sp}' p
        JOIN target_co_reports v ON p.safetyreportid = v.report_id
        WHERE p.drug_name = '{target_drug}'
          AND p.{pt_col} IS NOT NULL AND p.{pt_col} != ''
    ),
    pair_counts AS (
        SELECT pt, COUNT(DISTINCT report_id) AS a
        FROM target_pts GROUP BY pt
    ),
    -- 5. Marginali PT nello strato: Tutti i casi di quell'AE tra chi prende il co-farmaco
    pt_marginal AS (
        SELECT p.{pt_col} AS pt, COUNT(DISTINCT p.safetyreportid) AS n_pt
        FROM '{sp}' p
        JOIN co_reports v ON p.safetyreportid = v.report_id
        WHERE p.{pt_col} IS NOT NULL AND p.{pt_col} != ''
        GROUP BY p.{pt_col}
    )
    SELECT
        '{target_drug}'                          AS drug,
        pc.pt                                    AS pt,
        pc.a                                     AS a,
        (dt.n_drug - pc.a)                       AS b,
        (pm.n_pt   - pc.a)                       AS c,
        (t.n - dt.n_drug - pm.n_pt + pc.a)       AS d,
        t.n                                      AS n
    FROM pair_counts  pc
    JOIN pt_marginal  pm ON pc.pt = pm.pt
    CROSS JOIN drug_total dt
    CROSS JOIN totals     t
    WHERE pc.a >= {min_a}
      AND (dt.n_drug - pc.a)                 >= 0
      AND (pm.n_pt   - pc.a)                 >= 0
      AND (t.n - dt.n_drug - pm.n_pt + pc.a) >= 0
    ORDER BY pc.a DESC
    """
    result = con.execute(query).df()
    con.close()
    return result


# ─── Route D: full scan con where_extra su faers_sorted ─────────────────────

def _build_ct_full(
        target_drug: str, pt_col: str, min_a: int,
        where_extra: str | None, threads: int, memory: str,
) -> pd.DataFrame:
    """Full scan su faers_sorted.parquet — comportamento originale con where_extra."""
    sp = str(_SORTED_PARQUET)

    if where_extra:
        full_filter = f"drug_name = '{target_drug}' AND {where_extra}"
        bg_where    = f"WHERE {where_extra}"
    else:
        full_filter = f"drug_name = '{target_drug}'"
        bg_where    = ""

    con = _make_con(threads, memory)
    query = f"""
    WITH
    drug_pt_pairs AS (
        SELECT DISTINCT safetyreportid AS report_id, {pt_col} AS pt
        FROM '{sp}'
        WHERE {full_filter}
          AND {pt_col} IS NOT NULL AND {pt_col} != ''
    ),
    pair_counts AS (
        SELECT pt, COUNT(DISTINCT report_id) AS a
        FROM drug_pt_pairs GROUP BY pt
    ),
    drug_total AS (
        SELECT COUNT(DISTINCT report_id) AS n_drug FROM drug_pt_pairs
    ),
    strat_n_total AS (
        SELECT COUNT(DISTINCT safetyreportid) AS n
        FROM '{sp}' {bg_where}
    ),
    strat_pt_marginal AS (
        SELECT {pt_col} AS pt, COUNT(DISTINCT safetyreportid) AS n_pt
        FROM '{sp}'
        WHERE {(where_extra + " AND ") if where_extra else ""}{pt_col} IS NOT NULL AND {pt_col} != ''
        GROUP BY {pt_col}
    )
    SELECT
        '{target_drug}'                            AS drug,
        pc.pt                                      AS pt,
        pc.a                                       AS a,
        (dt.n_drug  - pc.a)                        AS b,
        (pm.n_pt    - pc.a)                        AS c,
        (nt.n - dt.n_drug - pm.n_pt + pc.a)        AS d,
        nt.n                                       AS n
    FROM pair_counts       pc
    JOIN strat_pt_marginal pm ON pc.pt = pm.pt
    CROSS JOIN drug_total  dt
    CROSS JOIN strat_n_total nt
    WHERE pc.a >= {min_a}
      AND (dt.n_drug  - pc.a)                   >= 0
      AND (pm.n_pt    - pc.a)                   >= 0
      AND (nt.n - dt.n_drug - pm.n_pt + pc.a)   >= 0
    ORDER BY pc.a DESC
    """
    result = con.execute(query).df()
    con.close()
    return result


# ─── Interfaccia pubblica ────────────────────────────────────────────────────

def build_contingency_table(
        parquet_path:   str,
        target_drug:    str,
        pt_col:         str = "reaction_pt",
        min_a:          int = 3,
        where_extra:    str = None,
        duckdb_threads: int = 4,
        duckdb_memory:  str = "4GB",
) -> pd.DataFrame:
    """
    Costruisce la contingency table 2×2 per tutte le coppie (target_drug, PT).

    Routing automatico:
        None         → Route A: sorted + marginals_global (~2-5s)
        sex/age      → Route B: cubo OLAP (<1s)
        co-medication→ Route C: indice invertito (~5-8s)
        altro        → Route D: full scan su faers_sorted (~60-90s)

    Fallback globale: se faers_sorted o marginals_global non esistono,
    usa parquet_path con full scan (comportamento pre-ottimizzazione).
    """
    # Normalizza where_extra: collassa newline e spazi multipli.
    # Difende da stringhe con \n letterali che spezzano il parser SQL di DuckDB.
    if where_extra:
        where_extra = " ".join(where_extra.split())

    # Fallback se i file ottimizzati non esistono
    if not _SORTED_PARQUET.exists() or not _MARGINALS.exists():
        print(f"  [CT  ] File ottimizzati non trovati — fallback su {parquet_path}")
        return _build_ct_full_legacy(parquet_path, target_drug, pt_col, min_a,
                                     where_extra, duckdb_threads, duckdb_memory)

    route = _detect_route(where_extra)
    route_labels = {
        "global": "A — sorted + marginals_global",
        "cubed":  "B — cubo OLAP (marginals_cubed)",
        "index":  "C — indice invertito (co-medication)",
        "full":   "D — full scan faers_sorted con filtro",
    }
    print(f"  [CT  ] Route {route_labels[route]}", flush=True)

    if route == "global":
        return _build_ct_global(target_drug, pt_col, min_a, duckdb_threads, duckdb_memory)

    if route == "cubed":
        if _CUBED.exists():
            return _build_ct_cubed(target_drug, pt_col, min_a, where_extra,
                                    duckdb_threads, duckdb_memory)
        print(f"  [CT  ] marginals_cubed.parquet non trovato — fallback Route D")
        print(f"         Esegui ingestion/run_prepare.py per abilitare Route B.")

    if route == "index":
        if _INDEX.exists():
            return _build_ct_index(target_drug, pt_col, min_a, where_extra,
                                    duckdb_threads, duckdb_memory)
        print(f"  [CT  ] drug_inverted_index.parquet non trovato — fallback Route D")
        print(f"         Esegui ingestion/run_prepare.py per abilitare Route C.")

    # Route D
    return _build_ct_full(target_drug, pt_col, min_a, where_extra,
                           duckdb_threads, duckdb_memory)


def _build_ct_full_legacy(
        parquet_path: str, target_drug: str, pt_col: str, min_a: int,
        where_extra: str | None, threads: int, memory: str,
) -> pd.DataFrame:
    """Fallback su faers_flat_deduped.parquet — pre-ottimizzazione."""
    base_filter = f"drug_name = '{target_drug}'"
    full_filter = f"{base_filter} AND {where_extra}" if where_extra else base_filter
    bg_where    = f"WHERE {where_extra}" if where_extra else ""

    con = _make_con(threads, memory)
    con.execute(f"CREATE VIEW faers AS SELECT * FROM read_parquet('{parquet_path}')")
    query = f"""
    WITH
    target_reports AS (
        SELECT DISTINCT safetyreportid AS report_id FROM faers WHERE {full_filter}
    ),
    background_reports AS (
        SELECT DISTINCT safetyreportid AS report_id FROM faers {bg_where}
    ),
    totals      AS (SELECT COUNT(DISTINCT report_id) AS n     FROM background_reports),
    drug_marginal AS (SELECT COUNT(DISTINCT report_id) AS n_drug FROM target_reports),
    target_pts  AS (
        SELECT DISTINCT p.safetyreportid AS report_id, p.{pt_col} AS pt
        FROM faers p JOIN target_reports t ON p.safetyreportid = t.report_id
        WHERE p.drug_name = '{target_drug}'
          AND p.{pt_col} IS NOT NULL AND p.{pt_col} != ''
    ),
    pair_counts AS (SELECT pt, COUNT(DISTINCT report_id) AS a FROM target_pts GROUP BY pt),
    pt_marginal AS (
        SELECT p.{pt_col} AS pt, COUNT(DISTINCT p.safetyreportid) AS n_pt
        FROM faers p JOIN background_reports b ON p.safetyreportid = b.report_id
        WHERE p.{pt_col} IS NOT NULL AND p.{pt_col} != ''
        GROUP BY p.{pt_col}
    )
    SELECT '{target_drug}' AS drug, pc.pt AS pt, pc.a AS a,
           (dm.n_drug - pc.a) AS b, (pm.n_pt - pc.a) AS c,
           (t.n - dm.n_drug - pm.n_pt + pc.a) AS d, t.n AS n
    FROM pair_counts pc
    JOIN pt_marginal pm ON pc.pt = pm.pt
    CROSS JOIN drug_marginal dm CROSS JOIN totals t
    WHERE pc.a >= {min_a}
      AND (dm.n_drug - pc.a) >= 0 AND (pm.n_pt - pc.a) >= 0
      AND (t.n - dm.n_drug - pm.n_pt + pc.a) >= 0
    ORDER BY pc.a DESC
    """
    result = con.execute(query).df()
    con.close()
    return result


def qc_contingency_table(df: pd.DataFrame, label: str = "") -> None:
    if df.empty:
        print(f"  [QC] {label}: DataFrame vuoto")
        return
    print(f"\n[{label}] QC — {len(df)} coppie (drug, PT)")
    ok = ((df["a"] + df["b"] + df["c"] + df["d"]) == df["n"]).all()
    print(f"  a+b+c+d == n : {'OK' if ok else 'FAIL'}")
    neg = {col: int((df[col] < 0).sum()) for col in ["a", "b", "c", "d"]}
    print(f"  Celle negative: " + "  ".join(f"{k}={v}" for k, v in neg.items()))
    print(f"  a — min={df['a'].min()}  median={df['a'].median():.0f}  max={df['a'].max()}")
    print(f"  n — valore unico: {df['n'].nunique() == 1}  ({df['n'].iloc[0]})")
    print(f"  Top 5 PT per a:")
    print(df.nlargest(5, "a")[["pt", "a"]].to_string(index=False))