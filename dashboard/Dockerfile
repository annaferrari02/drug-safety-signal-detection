# Immagine base Python slim — minimale, senza GUI o tool non necessari
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia vigipy locale (non disponibile su PyPI)
COPY vigipy/ ./vigipy/

# Copia src/ per le funzioni condivise
COPY src/ ./src/

# Copia lo script Streamlit
COPY dashboard/app.py .

RUN mkdir -p data

# Espone la porta Streamlit
EXPOSE 8501

# --server.address=0.0.0.0 necessario per rendere Streamlit raggiungibile
# dall'esterno del container (altrimenti ascolta solo su localhost interno)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]