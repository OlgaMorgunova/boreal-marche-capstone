# syntax=docker/dockerfile:1.7
# Streamlit interface. Talks to the API over HTTP only (API_URL); no model code inside.
#
#   docker build -f docker/frontend.Dockerfile -t boreal-ui:local .
ARG PYTHON_IMAGE=python:3.11-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY docker/requirements-frontend.txt /build/
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r /build/requirements-frontend.txt \
    && find /opt/venv/lib/python3.11/site-packages -depth -type d \
         \( -name tests -o -name __pycache__ \) -not -path "*/numpy/*" -exec rm -rf {} +

FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --create-home --uid 10001 appuser
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BOREAL_ROOT=/app \
    API_URL=http://api:8000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# "About" section content: model card, data provenance, Law 25 notice, architecture diagram.
COPY MODEL_CARD.md DATA_CARD.md GOVERNANCE_LOI25.md ./
COPY reports/figures/architecture.png ./reports/figures/
COPY frontend ./frontend
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)" || exit 1
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
