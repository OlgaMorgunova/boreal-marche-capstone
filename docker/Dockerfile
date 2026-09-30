# syntax=docker/dockerfile:1
# Serving image for the Boréal Marché API (D4 will pin the base image by digest and report sizes).

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 BOREAL_ROOT=/app BOREAL_MODEL_SOURCE=local
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY params.yaml ./
COPY artifacts/model.joblib artifacts/model_meta.json artifacts/rules.json ./artifacts/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)" || exit 1
CMD ["uvicorn", "boreal_marche.api:app", "--host", "0.0.0.0", "--port", "8000"]
