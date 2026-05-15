FROM eclipse-temurin:17-jdk-jammy

# Installa Python 3.13
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/opt/java/openjdk
ENV PATH=$PATH:$JAVA_HOME/bin

# Installa uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
ENV PYSPARK_PYTHON=/app/.venv/bin/python3

COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY main.py .