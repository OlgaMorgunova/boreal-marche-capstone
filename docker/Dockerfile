FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY src ./src
COPY tests ./tests

RUN pip install --no-cache-dir -e .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "boreal_marche.api:app", "--host", "0.0.0.0", "--port", "8000"]
