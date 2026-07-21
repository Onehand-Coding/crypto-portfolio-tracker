"""FastAPI application serving the portfolio API and the built frontend."""

from fastapi import FastAPI

from api.routes import portfolio

app = FastAPI(title="Crypto Portfolio Tracker API", version="1.0.0")
app.include_router(portfolio.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
