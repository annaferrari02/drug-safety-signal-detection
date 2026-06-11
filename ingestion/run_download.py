"""
run_download.py

Scarica i file raw FAERS dall'API openFDA e li salva in data/raw/ come JSON.
Questo script è separato da run_flatten.py: i dati scaricati persistono su disco
e il download può essere saltato se i file sono già presenti (idempotente).

Output in data/raw/:
    <quarter>.json    — un file per ogni partizione scaricata
                        es. 2007q1.json, 2008q1.json, ...

Configurazione:
    TARGET_QUARTERS = None   → scarica tutti i 1700+ file (full dataset)
    TARGET_QUARTERS = {...}  → scarica solo i quarter nel set (mock dataset)
"""

import io
import re
import zipfile
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_DIR = Path("data")
RAW_DIR  = DATA_DIR / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Sostituire con None per scaricare l'intero FAERS
TARGET_QUARTERS = None


def extract_year_quarter(url: str) -> tuple:
    """Estrae (anno, quarter) dall'URL per ordinare i file cronologicamente."""
    match = re.search(r"/(\d{4})q(\d)/", url)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, 0)


def select_partitions(all_partitions: list) -> list:
    """
    Seleziona i file da scaricare dall'elenco completo delle partizioni FAERS.
    Se TARGET_QUARTERS è None, restituisce tutte le partizioni normalizzate.
    Altrimenti filtra per i quarter nel set e prende solo il primo file di ognuno.
    """
    if TARGET_QUARTERS is None:
        selected = []
        for p in all_partitions:
            url     = p["file"]
            parts   = url.split("/")
            quarter = parts[-2]
            selected.append({
                "quarter": quarter,
                "file":    url,
                "size_mb": float(p["size_mb"]),
                "records": p["records"],
            })
        print(f"[INFO] Modalità full dataset: {len(selected)} file totali")
        return selected

    selected = []
    for p in all_partitions:
        url      = p["file"]
        parts    = url.split("/")
        quarter  = parts[-2]
        filename = parts[-1]
        if quarter in TARGET_QUARTERS and "0001-of" in filename:
            selected.append({
                "quarter": quarter,
                "file":    url,
                "size_mb": float(p["size_mb"]),
                "records": p["records"],
            })

    print(f"[INFO] Mock dataset: {len(selected)} file selezionati su {len(all_partitions)} totali")
    return selected


def download_and_extract(entry: dict) -> Path | None:
    """
    Scarica un file zip FAERS, estrae il JSON interno e lo salva in RAW_DIR.
    Salta il download se il file è già presente (idempotente).
    """
    url      = entry["file"]
    quarter  = entry["quarter"]
    out_path = RAW_DIR / f"{quarter}.json"

    if out_path.exists():
        print(f"  [SKIP] {quarter} già presente")
        return out_path

    print(f"  [DOWNLOAD] {quarter} ({entry['size_mb']} MB zip, ~{entry['records']:,} records)...")
    try:
        response  = requests.get(url, verify=False, timeout=300, stream=True)
        response.raise_for_status()
        zip_bytes = io.BytesIO(response.content)

        with zipfile.ZipFile(zip_bytes) as zf:
            json_filename = [n for n in zf.namelist() if n.endswith(".json")][0]
            json_bytes    = zf.read(json_filename)

        out_path.write_bytes(json_bytes)
        print(f"  [OK] {out_path} ({out_path.stat().st_size / 1e6:.1f} MB decompressi)")
        return out_path

    except Exception as e:
        print(f"  [ERROR] {quarter}: {e}")
        return None


def main():
    print("=== DOWNLOAD FILE FAERS ===")
    manifest       = requests.get("https://api.fda.gov/download.json", verify=False).json()
    all_partitions = manifest["results"]["drug"]["event"]["partitions"]

    all_partitions_sorted = sorted(
        all_partitions, key=lambda p: extract_year_quarter(p["file"])
    )

    selected   = select_partitions(all_partitions_sorted)
    downloaded = []

    for entry in selected:
        path = download_and_extract(entry)
        if path:
            downloaded.append(entry["quarter"])

    print(f"\n=== DOWNLOAD COMPLETATO: {len(downloaded)}/{len(selected)} file ===")

    # Scrive la lista dei quarter scaricati su disco: run_flatten.py la legge
    # per sapere quali file processare senza dover interrogare il manifest FDA
    manifest_path = RAW_DIR / "downloaded.txt"
    manifest_path.write_text("\n".join(sorted(downloaded)), encoding="utf-8")
    print(f"  Manifest scritto: {manifest_path}")


if __name__ == "__main__":
    main()