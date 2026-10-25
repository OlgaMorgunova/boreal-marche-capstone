# syntax=docker/dockerfile:1.7
# Serving image for the Boréal Marché API (Heads A, B, C + monitoring). Design notes: D4 §4.
#
#   docker build -f docker/Dockerfile -t boreal-api:local --build-arg GIT_SHA=$(git rev-parse HEAD) .
#
# Base pinned by digest (python:3.11-slim, multi-arch index, pulled 2026-10-24): a tag can move, a
# digest cannot, so a rebuild months later produces the same OS layer. Update deliberately.
ARG PYTHON_IMAGE=python:3.11-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377

# ---- builder: compilers/pip caches stay here, only /opt/venv and /opt/models move on ----------
FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
# Dependencies first (own layer): code edits do not reinstall 60 wheels.
COPY docker/requirements-api.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements-api.txt \
    # Test suites shipped inside wheels are dead weight at runtime (pandas/scipy/sklearn: ~100 MB).
    # Not numpy's: numpy.testing (imported by scipy/sklearn) needs numpy/_core/tests.
    && find /opt/venv/lib/python3.11/site-packages -depth -type d \
         \( -name tests -o -name __pycache__ \) -not -path "*/numpy/*" -exec rm -rf {} +
# The multilingual embedding model is baked in: no download at start-up, works without egress.
ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN /opt/venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBED_MODEL}', cache_dir='/opt/models')"
COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps . \
    && find /opt/venv -depth -type d -name __pycache__ -exec rm -rf {} +

# ---- runtime -----------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
# libgomp: OpenMP runtime for XGBoost and LightGBM. Non-root user with a fixed UID.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BOREAL_ROOT=/app \
    BOREAL_MODEL_SOURCE=local \
    BOREAL_EMBED_CACHE=/opt/models \
    HF_HUB_OFFLINE=1 \
    DATABASE_URL=sqlite:////home/appuser/boreal.db
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models
COPY params.yaml ./
COPY artifacts/model.joblib artifacts/model_meta.json artifacts/rules.json \
     artifacts/reference_profile.json artifacts/demo_customers.json artifacts/catalogue.json \
     ./artifacts/
COPY artifacts/index ./artifacts/index
# Last, so a new commit only rebuilds this tiny layer; /version reports both values.
ARG GIT_SHA=unknown
ARG IMAGE_TAG=local
ENV GIT_SHA=${GIT_SHA} IMAGE_TAG=${IMAGE_TAG}
USER appuser
EXPOSE 8000
# /ready (not /health): the container is healthy only once model, rules and index are loaded.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)" || exit 1
CMD ["uvicorn", "boreal_marche.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
