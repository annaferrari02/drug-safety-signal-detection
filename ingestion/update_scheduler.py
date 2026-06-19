# update_scheduler.py — gira dentro il container ingestion
import time
import subprocess
from datetime import datetime

def next_quarter_date():
    """Restituisce i mesi di inizio quarter: gennaio, aprile, luglio, ottobre."""
    now = datetime.now()
    # Prossimo mese di rilascio FDA (circa 2-3 mesi dopo fine quarter)
    release_months = [1, 4, 7, 10]
    for month in release_months:
        if now.month < month or (now.month == month and now.day < 15):
            return now.replace(month=month, day=15, hour=6, minute=0, second=0)
    # Prossimo anno
    return now.replace(year=now.year+1, month=1, day=15, hour=6, minute=0, second=0)

while True:
    target = next_quarter_date()
    wait_s = (target - datetime.now()).total_seconds()
    print(f"Prossimo check: {target.strftime('%Y-%m-%d')} (tra {wait_s/3600:.0f}h)")
    time.sleep(wait_s)
    subprocess.run(["python", "update_ingestion.py"], check=True)