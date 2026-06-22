FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vigipy/ ./vigipy/
COPY src/ ./src/
COPY ingestion/run_download.py .
COPY ingestion/run_flatten.py .
COPY ingestion/run_prepare.py .
COPY ingestion/update_ingestion.py .
COPY ingestion/update_scheduler.py .

RUN mkdir -p data/raw

# Pipeline completa: download → flatten → prepare (sort + marginals)
CMD ["sh", "-c", "python run_download.py && python run_flatten.py && python run_prepare.py"]
