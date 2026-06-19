# Immagine base Python slim — minimale, senza GUI o tool non necessari
FROM python:3.13-slim

WORKDIR /app

# Copia prima solo requirements per sfruttare la cache di Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia vigipy locale (non disponibile su PyPI)
COPY vigipy/ ./vigipy/

# Copia src/ per le funzioni condivise
COPY src/ ./src/

# Copia i due script di ingestion
COPY ingestion/run_download.py .
COPY ingestion/run_flatten.py .
COPY ingestion/update_ingestion.py .
COPY ingestion/update_scheduler.py .

RUN mkdir -p data/raw

# Esegue download e flatten in sequenza
CMD ["sh", "-c", "python run_download.py && python run_flatten.py"]
