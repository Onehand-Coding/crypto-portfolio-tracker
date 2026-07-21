"""FastAPI application serving the portfolio API and the built frontend."""

from fastapi import FastAPI

from api.routes import capital, portfolio, sync

app = FastAPI(title="Crypto Portfolio Tracker API", version="1.0.0")
app.include_router(portfolio.router)
app.include_router(capital.router)
app.include_router(sync.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
