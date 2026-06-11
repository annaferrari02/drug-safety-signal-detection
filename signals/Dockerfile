# Immagine base Python slim — minimale, senza GUI o tool non necessari
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia vigipy locale (non disponibile su PyPI)
COPY vigipy/ ./vigipy/

# Copia src/ per le funzioni condivise
COPY src/ ./src/

# Copia lo script di entrypoint
COPY signals/run_signals.py .

RUN mkdir -p data

CMD ["python", "run_signals.py"]