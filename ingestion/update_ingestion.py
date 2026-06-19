#script for trimestral update of the dataset (when openFDA is updated)

"""
update_ingestion.py

Aggiornamento incrementale del dataset FAERS.
Controlla il manifest openFDA, scarica i quarter mancanti,
li appende al Parquet flat e ricostruisce il Parquet deduplicato.

Scheduling consigliato: cron trimestrale (openFDA rilascia dati ogni ~3 mesi)
    0 6 1 */3 * python update_ingestion.py

Flusso:
    1. Fetch manifest openFDA → lista quarter disponibili
    2. Confronto con downloaded.txt → quarter nuovi (delta)
    3. Per ogni quarter nuovo: download + flatten → Parquet temp
    4. Append Parquet temp → faers_flat.parquet (DuckDB, zero RAM pandas)
    5. Ricostruzione faers_flat_deduped.parquet (DuckDB su tutto il flat)
    6. Aggiornamento downloaded.txt
    7. (Opzionale) invalidazione label_cache.json se nuovi drug rilevati
"""

import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import duckdb
import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import urllib3

# Import delle funzioni di ingestion già testate
from run_flatten import (
    flatten_report, _write_batch,
    PARQUET_FLAT, PARQUET_DEDUPED, DEDUP_REPORT,
    RAW_DIR, DATA_DIR, BATCH_SIZE,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MANIFEST_URL  = "https://api.fda.gov/download.json"
DOWNLOADED_TXT = RAW_DIR / "downloaded.txt"
UPDATE_LOG     = DATA_DIR / "update_log.jsonl"   # log append-only degli update


#  1. CHECK MANIFEST 
def fetch_available_quarters() -> dict[str, dict]:
    """
    Scarica il manifest openFDA e restituisce tutti i quarter disponibili.

    Returns
    -------
    { quarter_id: { "file": url, "size_mb": float, "records": int } }
    """
    manifest = requests.get(MANIFEST_URL, verify=False, timeout=30).json()
    partitions = manifest["results"]["drug"]["event"]["partitions"]

    available = {}
    for p in partitions:
        url   = p["file"]
        parts = url.split("/")
        quarter   = parts[-2]          # es. "2024q3"
        filename  = parts[-1]
        # Prende solo il primo file per quarter (come run_download.py)
        if "0001-of" in filename and quarter not in available:
            available[quarter] = {
                "file":     url,
                "size_mb":  float(p["size_mb"]),
                "records":  p["records"],
            }

    return available


def load_downloaded_quarters() -> set[str]:
    """Legge i quarter già scaricati da downloaded.txt."""
    if not DOWNLOADED_TXT.exists():
        return set()
    return set(DOWNLOADED_TXT.read_text(encoding="utf-8").splitlines())


def find_new_quarters(
        available:   dict[str, dict],
        downloaded:  set[str],
) -> list[tuple[str, dict]]:
    """
    Calcola il delta: quarter disponibili su openFDA ma non ancora scaricati.
    Ordina cronologicamente per garantire l'append in ordine temporale.
    """
    new = {q: info for q, info in available.items() if q not in downloaded}

    def sort_key(item):
        m = re.search(r"(\d{4})q(\d)", item[0])
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return sorted(new.items(), key=sort_key)


# 2. DOWNLOAD 
def download_quarter(quarter: str, url: str) -> Path | None:
    """
    Scarica e decomprime un singolo quarter.
    Identico a run_download.download_and_extract(), isolato qui
    per non importare il modulo di download completo.
    """
    out_path = RAW_DIR / f"{quarter}.json"
    if out_path.exists():
        print(f"  [SKIP] {quarter} già presente in data/raw/")
        return out_path

    print(f"  [DOWNLOAD] {quarter}...", flush=True)
    try:
        resp      = requests.get(url, verify=False, timeout=300, stream=True)
        resp.raise_for_status()
        zip_bytes = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_bytes) as zf:
            json_name = [n for n in zf.namelist() if n.endswith(".json")][0]
            out_path.write_bytes(zf.read(json_name))
        print(f"  [OK] {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path
    except Exception as e:
        print(f"  [ERR] {quarter}: {e}")
        return None


# 3. FLATTEN → PARQUET TEMP 

def flatten_to_temp(json_path: Path) -> Path | None:
    """
    Appiattisce un singolo JSON quarter in un Parquet temporaneo.
    Riusa flatten_report() e _write_batch() da run_flatten.py.

    Usa un file temporaneo invece di appendere direttamente al flat principale:
    se il processo si interrompe a metà, il flat principale rimane integro.
    """
    temp_path = DATA_DIR / f"_temp_{json_path.stem}.parquet"
    writer    = None
    schema    = None
    batch     = []
    n_reports = 0

    print(f"  [FLATTEN] {json_path.name}...", flush=True)
    try:
        with open(json_path, "rb") as f:
            for report in ijson.items(f, "results.item"):
                batch.extend(flatten_report(report))
                n_reports += 1
                if n_reports % BATCH_SIZE == 0:
                    writer, schema = _write_batch(batch, writer, schema,
                                                  output_path=temp_path)
                    batch = []

        if batch:
            writer, schema = _write_batch(batch, writer, schema,
                                          output_path=temp_path)
        if writer:
            writer.close()

        print(f"     {n_reports:,} report → {temp_path.name} "
              f"({temp_path.stat().st_size / 1e6:.0f} MB)")
        return temp_path

    except Exception as e:
        print(f"  [ERR] flatten {json_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return None
    
# ── 4. APPEND AL FLAT PRINCIPALE ──────────────────────────────────────────────

def append_to_flat(temp_path: Path) -> bool:
    """
    Appende il Parquet temporaneo al faers_flat.parquet principale.

    Strategia: DuckDB legge entrambi i file e scrive un nuovo flat
    con COPY ... TO. Nessun dato caricato in RAM pandas.

    Alternativa più rapida se lo schema è identico: concatenare i file
    PyArrow direttamente (pq.write_table con append=True).
    Usiamo DuckDB per robustezza sullo schema (cast automatico).
    """
    if not PARQUET_FLAT.exists():
        # Prima volta: rinomina il temp come flat principale
        temp_path.rename(PARQUET_FLAT)
        print(f"  [INIT] flat principale creato da {temp_path.name}")
        return True

    backup_path = PARQUET_FLAT.with_suffix(".backup.parquet")
    print(f"  [APPEND] {temp_path.name} → {PARQUET_FLAT.name}...", flush=True)

    try:
        # Lettura schema dal flat esistente per il cast
        existing_schema = pq.read_schema(PARQUET_FLAT)

        con = duckdb.connect()
        # Scrittura atomica: prima in un file nuovo, poi rename
        new_flat = PARQUET_FLAT.with_suffix(".new.parquet")
        con.execute(f"""
            COPY (
                SELECT * FROM '{PARQUET_FLAT}'
                UNION ALL
                SELECT * FROM '{temp_path}'
            )
            TO '{new_flat}' (FORMAT PARQUET)
        """)
        con.close()

        # Rename atomico: se fallisce qui, il flat originale è intatto
        PARQUET_FLAT.rename(backup_path)
        new_flat.rename(PARQUET_FLAT)
        backup_path.unlink()
        temp_path.unlink()

        size_mb = PARQUET_FLAT.stat().st_size / 1e6
        print(f"  [OK] flat aggiornato ({size_mb:.0f} MB)")
        return True

    except Exception as e:
        print(f"  [ERR] append fallito: {e}")
        # Rollback: ripristina il backup se esiste
        if backup_path.exists() and not PARQUET_FLAT.exists():
            backup_path.rename(PARQUET_FLAT)
        return False


# ── 5. RICOSTRUZIONE DEDUPED ──────────────────────────────────────────────────

def rebuild_deduped() -> bool:
    """
    Ricostruisce faers_flat_deduped.parquet dal flat aggiornato.

    Non è possibile deduplicare incrementalmente perché un nuovo quarter
    può contenere follow-up (stesso safetyreportid, receivedate più recente)
    di report già presenti nel dataset — quindi è necessario passare su tutto.

    DuckDB esegue questa query senza caricare nulla in RAM Python:
    lavora direttamente sul Parquet tramite memory-mapped I/O.
    Su dataset da 20-30 GB impiega tipicamente 3-8 minuti.
    """
    print("  [DEDUP] Ricostruzione faers_flat_deduped.parquet...", flush=True)
    temp_deduped = PARQUET_DEDUPED.with_suffix(".new.parquet")

    try:
        con = duckdb.connect()
        con.execute("SET memory_limit = '4GB'")
        con.execute("SET temp_directory = 'data/duckdb_tmp'")
        Path("data/duckdb_tmp").mkdir(exist_ok=True)

        t0 = datetime.now()
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

                SELECT * FROM '{PARQUET_FLAT}'
                WHERE safetyreportid IS NULL
            )
            TO '{temp_deduped}' (FORMAT PARQUET)
        """)
        elapsed = (datetime.now() - t0).seconds

        con.close()

        # Rename atomico
        old_deduped = PARQUET_DEDUPED.with_suffix(".old.parquet")
        if PARQUET_DEDUPED.exists():
            PARQUET_DEDUPED.rename(old_deduped)
        temp_deduped.rename(PARQUET_DEDUPED)
        if old_deduped.exists():
            old_deduped.unlink()

        size_mb = PARQUET_DEDUPED.stat().st_size / 1e6
        print(f"  [OK] deduped ricostruito ({size_mb:.0f} MB, {elapsed}s)")
        return True

    except Exception as e:
        print(f"  [ERR] rebuild deduped fallito: {e}")
        if temp_deduped.exists():
            temp_deduped.unlink()
        return False


# ── 6. LOG + STATO ────────────────────────────────────────────────────────────

def update_downloaded_txt(new_quarters: list[str]) -> None:
    existing = load_downloaded_quarters()
    updated  = sorted(existing | set(new_quarters))
    DOWNLOADED_TXT.write_text("\n".join(updated), encoding="utf-8")


def append_update_log(entry: dict) -> None:
    """Log append-only in JSONL: ogni run aggiunge una riga."""
    import json
    with open(UPDATE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n=== FAERS UPDATE CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    available  = fetch_available_quarters()
    downloaded = load_downloaded_quarters()
    new_quarters = find_new_quarters(available, downloaded)

    if not new_quarters:
        print("  Nessun nuovo quarter disponibile. Dataset aggiornato.")
        append_update_log({
            "timestamp": datetime.now().isoformat(),
            "status": "no_update",
            "available": len(available),
            "downloaded": len(downloaded),
        })
        return

    print(f"\n  {len(new_quarters)} nuovo/i quarter da scaricare:")
    for q, info in new_quarters:
        print(f"    {q}  ({info['size_mb']} MB, ~{info['records']:,} record)")

    success = []
    failed  = []

    for quarter, info in new_quarters:
        print(f"\n--- {quarter} ---")

        # 1. Download
        json_path = download_quarter(quarter, info["file"])
        if not json_path:
            failed.append(quarter)
            continue

        # 2. Flatten → temp Parquet
        temp_path = flatten_to_temp(json_path)
        if not temp_path:
            failed.append(quarter)
            continue

        # 3. Append al flat principale
        if not append_to_flat(temp_path):
            failed.append(quarter)
            continue

        # 4. Aggiorna downloaded.txt subito (idempotenza: se crasha al prossimo
        #    quarter, questo non viene riscaricato)
        update_downloaded_txt([quarter])
        success.append(quarter)

        # Opzionale: rimuovi il JSON raw per liberare spazio
        # json_path.unlink()

    # 5. Ricostruisce il deduped una volta sola alla fine (non per ogni quarter)
    if success:
        print(f"\n=== Ricostruzione deduped ({len(success)} quarter aggiunti) ===")
        rebuild_deduped()

    # Log finale
    append_update_log({
        "timestamp":      datetime.now().isoformat(),
        "status":         "updated" if success else "failed",
        "new_quarters":   success,
        "failed_quarters": failed,
        "total_available": len(available),
        "total_downloaded": len(downloaded) + len(success),
    })

    print(f"\n=== UPDATE COMPLETATO ===")
    print(f"  Aggiunti : {success}")
    if failed:
        print(f"  Falliti  : {failed}")


if __name__ == "__main__":
    main()