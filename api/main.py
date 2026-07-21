"""FastAPI application serving the portfolio API and the built frontend."""

from fastapi import FastAPI

app = FastAPI(title="Crypto Portfolio Tracker API", version="1.0.0")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
