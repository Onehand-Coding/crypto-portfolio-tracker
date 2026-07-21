"""FastAPI application serving the portfolio API and the built frontend.

One process, one port. In development Vite proxies /api here; in production
this serves the built bundle too. Neither arrangement needs CORS.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import capital, portfolio, sync

app = FastAPI(title="Crypto Portfolio Tracker API", version="1.0.0")

app.include_router(portfolio.router)
app.include_router(capital.router)
app.include_router(sync.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> FileResponse:
        # Registered after the routers, so /api/* has already matched. An
        # unmatched /api path must 404 rather than silently return the SPA.
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(FRONTEND_DIST / "index.html")
