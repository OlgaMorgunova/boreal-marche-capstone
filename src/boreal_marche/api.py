from fastapi import FastAPI

app = FastAPI(title="Boréal Marché API", version="0.1.0")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/version")
def version() -> dict[str, str]:
    return {"version": "0.1.0"}
