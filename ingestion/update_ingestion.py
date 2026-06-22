#script for trimestral update of the dataset (when openFDA is updated)

"""
update_ingestion.py

Aggiornamento incrementale del dataset FAERS.
Controlla il manifest openFDA, scarica i quarter mancanti,
li appende al Parquet flat e ricostruisce il Parquet deduplicato.

Flusso:
    1. Fetch manifest openFDA → lista quarter disponibili
    2. Confronto con downloaded.txt → quarter nuovi (delta)
    3. Per ogni quarter nuovo: download + flatten → Parquet temp
    4. Append Parquet temp → faers_flat.parquet (DuckDB, zero RAM pandas)
    5. Ricostruzione faers_flat_deduped.parquet (DuckDB su tutto il flat)
    6. Rigenerazione file ottimizzati: faers_sorted.parquet + marginals_global.parquet
    7. Aggiornamento downloaded.txt
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

from run_flatten import (
    flatten_report, _write_batch,
    PARQUET_FLAT, PARQUET_DEDUPED, DEDUP_REPORT,
    RAW_DIR, DATA_DIR, BATCH_SIZE,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MANIFEST_URL   = "https://api.fda.gov/download.json"
DOWNLOADED_TXT = RAW_DIR / "downloaded.txt"
UPDATE_LOG     = DATA_DIR / "update_log.jsonl"


# ── 1. CHECK MANIFEST ─────────────────────────────────────────────────────────

def fetch_available_quarters() -> dict[str, dict]:
    manifest   = requests.get(MANIFEST_URL, verify=False, timeout=30).json()
    partitions = manifest["results"]["drug"]["event"]["partitions"]

    available = {}
    for p in partitions:
        url      = p["file"]
        parts    = url.split("/")
        quarter  = parts[-2]
        filename = parts[-1]
        if "0001-of" in filename and quarter not in available:
            available[quarter] = {
                "file":     url,
                "size_mb":  float(p["size_mb"]),
                "records":  p["records"],
            }
    return available


def load_downloaded_quarters() -> set[str]:
    if not DOWNLOADED_TXT.exists():
        return set()
    return set(DOWNLOADED_TXT.read_text(encoding="utf-8").splitlines())


def find_new_quarters(
        available:  dict[str, dict],
        downloaded: set[str],
) -> list[tuple[str, dict]]:
    new = {q: info for q, info in available.items() if q not in downloaded}

    def sort_key(item):
        m = re.search(r"(\d{4})q(\d)", item[0])
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return sorted(new.items(), key=sort_key)


# ── 2. DOWNLOAD ───────────────────────────────────────────────────────────────

def download_quarter(quarter: str, url: str) -> Path | None:
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


# ── 3. FLATTEN → PARQUET TEMP ─────────────────────────────────────────────────

def flatten_to_temp(json_path: Path) -> Path | None:
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
    if not PARQUET_FLAT.exists():
        temp_path.rename(PARQUET_FLAT)
        print(f"  [INIT] flat principale creato da {temp_path.name}")
        return True

    backup_path = PARQUET_FLAT.with_suffix(".backup.parquet")
    print(f"  [APPEND] {temp_path.name} → {PARQUET_FLAT.name}...", flush=True)

    try:
        new_flat = PARQUET_FLAT.with_suffix(".new.parquet")
        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT * FROM '{PARQUET_FLAT}'
                UNION ALL
                SELECT * FROM '{temp_path}'
            )
            TO '{new_flat}' (FORMAT PARQUET)
        """)
        con.close()

        PARQUET_FLAT.rename(backup_path)
        new_flat.rename(PARQUET_FLAT)
        backup_path.unlink()
        temp_path.unlink()

        print(f"  [OK] flat aggiornato ({PARQUET_FLAT.stat().st_size / 1e6:.0f} MB)")
        return True

    except Exception as e:
        print(f"  [ERR] append fallito: {e}")
        if backup_path.exists() and not PARQUET_FLAT.exists():
            backup_path.rename(PARQUET_FLAT)
        return False


# ── 5. RICOSTRUZIONE DEDUPED ──────────────────────────────────────────────────

def rebuild_deduped() -> bool:
    print("  [DEDUP] Ricostruzione faers_flat_deduped.parquet...", flush=True)
    temp_deduped = PARQUET_DEDUPED.with_suffix(".new.parquet")

    try:
        con = duckdb.connect()
        con.execute("SET memory_limit = '4GB'")
        con.execute("SET preserve_insertion_order = false")
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

        old_deduped = PARQUET_DEDUPED.with_suffix(".old.parquet")
        if PARQUET_DEDUPED.exists():
            PARQUET_DEDUPED.rename(old_deduped)
        temp_deduped.rename(PARQUET_DEDUPED)
        if old_deduped.exists():
            old_deduped.unlink()

        print(f"  [OK] deduped ricostruito ({PARQUET_DEDUPED.stat().st_size / 1e6:.0f} MB, {elapsed}s)")
        return True

    except Exception as e:
        print(f"  [ERR] rebuild deduped fallito: {e}")
        if temp_deduped.exists():
            temp_deduped.unlink()
        return False


# ── 6. RIGENERAZIONE FILE OTTIMIZZATI ─────────────────────────────────────────

def rebuild_optimized_files() -> None:
    """
    Rigenera faers_sorted.parquet e marginals_global.parquet dopo ogni update.
    Questi file sono usati da build_contingency_table() per accelerare la CT.
    Senza questa chiamata, il dashboard continuerebbe a usare dati stale.
    """
    print("\n  [PREPARE] Rigenerazione file ottimizzati...", flush=True)
    try:
        from run_prepare import main as prepare_main
        prepare_main(force=True)
    except Exception as e:
        print(f"  [WARN] run_prepare fallito: {e}")
        print(f"  [WARN] Il dashboard userà il fallback full scan fino alla prossima esecuzione.")
        print(f"  [WARN] Rigenera manualmente con: python run_prepare.py --force")


# ── 7. LOG + STATO ────────────────────────────────────────────────────────────

def update_downloaded_txt(new_quarters: list[str]) -> None:
    existing = load_downloaded_quarters()
    updated  = sorted(existing | set(new_quarters))
    DOWNLOADED_TXT.write_text("\n".join(updated), encoding="utf-8")


def append_update_log(entry: dict) -> None:
    import json
    with open(UPDATE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n=== FAERS UPDATE CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    available    = fetch_available_quarters()
    downloaded   = load_downloaded_quarters()
    new_quarters = find_new_quarters(available, downloaded)

    if not new_quarters:
        print("  Nessun nuovo quarter disponibile. Dataset aggiornato.")
        append_update_log({
            "timestamp":  datetime.now().isoformat(),
            "status":     "no_update",
            "available":  len(available),
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

        json_path = download_quarter(quarter, info["file"])
        if not json_path:
            failed.append(quarter)
            continue

        temp_path = flatten_to_temp(json_path)
        if not temp_path:
            failed.append(quarter)
            continue

        if not append_to_flat(temp_path):
            failed.append(quarter)
            continue

        # Aggiorna subito: se crasha al quarter successivo, questo non viene
        # riscaricato (idempotenza)
        update_downloaded_txt([quarter])
        success.append(quarter)

    if success:
        print(f"\n=== Ricostruzione deduped ({len(success)} quarter aggiunti) ===")
        if rebuild_deduped():
            # Rigenera i file ottimizzati solo se il deduped è andato a buon fine
            rebuild_optimized_files()

    append_update_log({
        "timestamp":        datetime.now().isoformat(),
        "status":           "updated" if success else "failed",
        "new_quarters":     success,
        "failed_quarters":  failed,
        "total_available":  len(available),
        "total_downloaded": len(downloaded) + len(success),
    })

    print(f"\n=== UPDATE COMPLETATO ===")
    print(f"  Aggiunti : {success}")
    if failed:
        print(f"  Falliti  : {failed}")


if __name__ == "__main__":
    main()