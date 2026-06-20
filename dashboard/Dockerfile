# Immagine base Python slim — minimale, senza GUI o tool non necessari
FROM python:3.13-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia vigipy locale (non disponibile su PyPI) — necessaria perché
# run_signals.py (importato sotto) usa gli algoritmi PRR/ROR/BCPNN/MGPS
COPY vigipy/ ./vigipy/

# Copia src/ per le funzioni condivise (es. text_to_sql, build_contingency_table)
COPY src/ ./src/

# Copia run_signals.py nella stessa posizione (root /app) usata dal
# container 'signals', cosi che Path(__file__).resolve().parent risolva
# a /app sia qui che li, coerente con il volume ./data:/app/data
COPY signals/run_signals.py .

# Copia lo script Streamlit
COPY dashboard/appli.py .

RUN mkdir -p data

# Espone la porta Streamlit
EXPOSE 8501

# --server.address=0.0.0.0 necessario per rendere Streamlit raggiungibile
# dall'esterno del container (altrimenti ascolta solo su localhost interno)
CMD ["streamlit", "run", "appli.py", "--server.address=0.0.0.0", "--server.port=8501"]