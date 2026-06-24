"""
main.py — Drug Safety Signal Detection - One-shot launcher

Usage:
    python main.py

Steps performed automatically:
    1. Launches Docker Desktop + Check prerequisites (Docker running, .env with MISTRAL_API_KEY)
    2. Download pre-computed Parquet files from HF
    3. Build Docker containers  (docker compose build)
    4. Launch the dashboard     (docker compose up dashboard)
    5. Open the browser at http://localhost:8501
"""

import os
import sys
import time
import shutil
import subprocess
import webbrowser
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' is not installed.")
    print("        Run: pip install requests")
    sys.exit(1)


HF_REPO_ID  = "annaferrari02/drug-safety-faers"          # ← your HF dataset repo
HF_BRANCH   = "main"
DATA_DIR    = Path(__file__).parent / "data"
DASHBOARD_URL = "http://localhost:8501"

# Files to download from HF (path inside the repo → local destination)
FILES = {
    "faers_flat_deduped.parquet":   DATA_DIR / "faers_flat_deduped.parquet",
    "faers_sorted.parquet":         DATA_DIR / "faers_sorted.parquet",
    "marginals_global.parquet":     DATA_DIR / "marginals_global.parquet",
    "marginals_cubed.parquet":      DATA_DIR / "marginals_cubed.parquet",
    "drug_inverted_index.parquet":  DATA_DIR / "drug_inverted_index.parquet",
}

def hf_url(filename: str) -> str:
    """Direct download URL for a file in a public HF dataset repo."""
    return (
        f"https://huggingface.co/datasets/{HF_REPO_ID}"
        f"/resolve/{HF_BRANCH}/{filename}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"

def banner(text: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {BOLD}{text}{RESET}")
    print(f"{'─' * width}")

def ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg: str)  -> None: print(f"  {RED}✗{RESET}  {msg}")
def info(msg: str) -> None: print(f"     {msg}")

def abort(msg: str) -> None:
    err(msg)
    sys.exit(1)

#prerequisiti

def _start_docker_desktop() -> None:
    import platform
    system = platform.system()
    
    if system == "Windows":
        docker_path = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Docker/Docker/Docker Desktop.exe"
        if not docker_path.exists():
            abort("Docker Desktop non trovato. Installalo da https://www.docker.com/products/docker-desktop")
        subprocess.Popen([str(docker_path)])
    
    elif system == "Darwin":  # macOS
        subprocess.Popen(["open", "-a", "Docker"])
    
    elif system == "Linux":
        # Su Linux non esiste Docker Desktop, il daemon si avvia come servizio
        subprocess.run(["sudo", "systemctl", "start", "docker"], capture_output=True)
    
    else:
        abort(f"Sistema operativo non supportato: {system}")

    info("Attendo che Docker sia pronto...")
    for _ in range(40):
        r = subprocess.run(["docker", "info"], capture_output=True)
        if r.returncode == 0:
            ok("Docker avviato")
            return
        time.sleep(2)
    
    abort("Docker non risponde dopo 80 secondi. Avvialo manualmente e riprova.")

def check_prerequisites() -> None:
    banner("Step 1 · Checking prerequisites")

    # Docker CLI available
    if not shutil.which("docker"):
        abort("Docker not found. Install Docker Desktop: https://www.docker.com/products/docker-desktop")
    ok("Docker CLI found")

    # Docker daemon running
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        abort("Docker daemon is not running. Please start Docker Desktop and try again.")
    ok("Docker daemon is running")

    # docker compose available (v2: 'docker compose'; v1: 'docker-compose')
    compose_cmd = _get_compose_cmd()
    if compose_cmd is None:
        abort("'docker compose' not found. Update Docker Desktop to a recent version.")
    ok(f"Docker Compose found ({' '.join(compose_cmd)})")

    # .env file present
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        abort(
            ".env file not found.\n"
            "        Create one with:\n\n"
            "            echo MISTRAL_API_KEY=your_key_here > .env\n"
            "                You can obtain your Mistral API key here: https://admin.mistral.ai/organization/api-keys"
        )

    # MISTRAL_API_KEY in .env
    env_text = env_path.read_text(encoding="utf-8")
    if "MISTRAL_API_KEY" not in env_text or "MISTRAL_API_KEY=" not in env_text:
        abort("MISTRAL_API_KEY not found in .env.\n"
              "        Add: MISTRAL_API_KEY=your_key_here"
              "             You can obtain your Mistral API key here: https://admin.mistral.ai/organization/api-keys")
    
    # Check it's not empty
    for line in env_text.splitlines():
        if line.startswith("MISTRAL_API_KEY="):
            value = line.split("=", 1)[1].strip()
            if not value or value in ("your_key_here", ""):
                abort("MISTRAL_API_KEY is empty in .env. Add your Mistral API key." \
                "           You can obtain your Mistral API key here: https://admin.mistral.ai/organization/api-keys")
            break
    ok(".env with MISTRAL_API_KEY found")

    # data/ directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ok(f"Data directory ready: {DATA_DIR}")


def _get_compose_cmd() -> list[str] | None:
    """Returns the docker compose command as a list, or None if not found."""
    # Docker Compose v2 (plugin)
    r = subprocess.run(["docker", "compose", "version"], capture_output=True)
    if r.returncode == 0:
        return ["docker", "compose"]
    # Docker Compose v1 (standalone)
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return None

#download Data Mart (parquet files) from HF 

def download_data() -> None:
    banner("Step 2 · Downloading data from Hugging Face")

    if all(dest.exists() for dest in FILES.values()):
        ok("All data files already present in /data/ — skipping download")
        return

    info(f"Repository: {HF_REPO_ID}")

    for filename, dest_path in FILES.items():
        if dest_path.exists():
            size_mb = dest_path.stat().st_size / 1e6
            ok(f"{filename} already present ({size_mb:.0f} MB) — skipping")
            continue

        url = hf_url(filename)
        info(f"Downloading {filename} …")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                abort(
                    f"{filename} not found at {url}\n"
                    f"        Make sure you have uploaded the files to:\n"
                    f"        https://huggingface.co/datasets/{HF_REPO_ID}"
                )
            abort(f"HTTP error downloading {filename}: {e}")
        except requests.exceptions.ConnectionError:
            abort("Cannot reach Hugging Face. Check your internet connection.")

        total_bytes = int(response.headers.get("content-length", 0))
        chunk_size  = 8 * 1024 * 1024  # 8 MB chunks

        tmp_path = dest_path.with_suffix(".tmp")
        try:
            if HAS_TQDM and total_bytes:
                with tqdm(
                    total=total_bytes,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f"  {filename[:40]}",
                    ncols=70,
                ) as pbar, open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                # Fallback progress without tqdm
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_bytes:
                            pct = downloaded / total_bytes * 100
                            print(f"\r     {pct:5.1f}%  {downloaded/1e6:.0f}/{total_bytes/1e6:.0f} MB", end="", flush=True)
                if total_bytes:
                    print()  # newline after progress

            tmp_path.rename(dest_path)
            size_mb = dest_path.stat().st_size / 1e6
            ok(f"{filename} downloaded ({size_mb:.0f} MB)")

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            abort(f"Download failed for {filename}: {e}")

#build containers with docker desktop 

def build_containers() -> None:
    banner("Step 3 : Building Docker containers")
    info("This may take a few minutes the first time…")

    compose_cmd = _get_compose_cmd()
    project_dir = Path(__file__).parent

    result = subprocess.run(
        [*compose_cmd, "build", "--quiet", "dashboard"],
        cwd=project_dir,
    )
    if result.returncode != 0:
        abort(
            "docker compose build failed.\n"
            "        Check the output above for errors.\n"
            "        Common causes: missing requirements.txt, build context too large."
        )
    ok("Dashboard image built successfully")

#LAUNCH dashboard 

def launch_dashboard() -> None:
    banner("Step 4 · Launching dashboard")

    compose_cmd = _get_compose_cmd()
    project_dir = Path(__file__).parent

    # Stop any existing instance first (ignore errors)
    subprocess.run(
        [*compose_cmd, "stop", "dashboard"],
        cwd=project_dir,
        capture_output=True,
    )

    info("Starting dashboard container…")
    proc = subprocess.Popen(
        [*compose_cmd, "up", "dashboard"],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Wait until Streamlit is ready
    ok_markers = ["You can now view your Streamlit app", "Network URL", "Local URL"]
    err_markers = ["Error", "error", "Traceback"]
    ready = False
    startup_log: list[str] = []

    print("  Waiting for Streamlit to start", end="", flush=True)
    try:
        for line in proc.stdout:
            startup_log.append(line.rstrip())
            print(".", end="", flush=True)

            if any(m in line for m in ok_markers):
                ready = True
                print()  # newline
                break
            if proc.poll() is not None:
                print()
                break
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        proc.terminate()
        sys.exit(0)

    if not ready:
        warn("Could not confirm Streamlit started. Check logs below:")
        for l in startup_log[-20:]:
            print(f"     {l}")
        warn(f"Try opening {DASHBOARD_URL} manually in a few seconds.")
    else:
        ok(f"Dashboard is running at {DASHBOARD_URL}")

    # Open browser
    time.sleep(1.5)
    try:
        webbrowser.open(DASHBOARD_URL)
        ok("Browser opened")
    except Exception:
        info(f"Open your browser at: {DASHBOARD_URL}")

    # Keep the process running and stream logs
    banner("Dashboard running — press Ctrl+C to stop")
    try:
        for line in proc.stdout:
            # Only show warnings/errors to keep output clean
            stripped = line.strip()
            if stripped and any(k in stripped for k in ("WARNING", "ERROR", "Error", "Traceback")):
                warn(stripped)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  Stopping dashboard…")
        proc.terminate()
        subprocess.run(
            [*compose_cmd, "stop", "dashboard"],
            cwd=project_dir,
            capture_output=True,
        )
        ok("Dashboard stopped.")

#MAIN 

def main() -> None:
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Drug Safety Signal Detection — Setup & Launch{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    check_prerequisites()
    download_data()
    build_containers()
    launch_dashboard()


if __name__ == "__main__":
    main()