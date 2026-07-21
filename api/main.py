"""FastAPI application serving the portfolio API and the built frontend.

One process, one port. In development Vite proxies /api here; in production
this serves the built bundle too. Neither arrangement needs CORS.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from api.routes import capital, portfolio, screens, strategy, sync, wallets

app = FastAPI(title="Crypto Portfolio Tracker API", version="1.0.0")

app.include_router(portfolio.router)
app.include_router(capital.router)
app.include_router(sync.router)
app.include_router(wallets.router)
app.include_router(strategy.router)
# Registered last of the API routers: it owns the generic /api/assets/{symbol}
# path, which must not shadow a more specific route added later.
app.include_router(screens.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def _is_api_path(full_path: str) -> bool:
    """True for anything that was meant to reach the API.

    The captured path arrives normalised inconsistently: "/api/x" as "api/x",
    but "//api/x" as "/api/x" and a bare "/api" as "api". A base URL built with
    a trailing slash produces the double-slash form, so matching only "api/"
    would hand those requests the SPA -- HTML with a 200, which is exactly the
    failure this guard exists to prevent.
    """
    stripped = full_path.strip("/")
    return stripped == "api" or stripped.startswith("api/")


def _wants_a_file(full_path: str) -> bool:
    """True for requests that name an asset rather than a client-side route.

    The test is an extension on the last segment, and nothing else. An earlier
    version also treated everything under "assets/" as a file, which broke the
    client-side route /assets/:symbol -- the asset-detail page 404'd because it
    shares a prefix with the bundle directory. Hashed bundles always carry .js
    or .css, so the extension test alone is enough to make a stale bundle
    reference 404 honestly instead of arriving as HTML with a 200.
    """
    return "." in full_path.rsplit("/", 1)[-1]


def _dist_file(dist: Path, full_path: str) -> Optional[Path]:
    """Resolve a request path to a real file inside dist, or None.

    full_path is attacker-controlled and gets joined onto a filesystem path,
    which is precisely how "../../.env" ends up served. Resolving first and
    then confirming containment is what makes that impossible.
    """
    root = dist.resolve()
    candidate = (root / full_path).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate if candidate.is_file() else None


def mount_spa(target: FastAPI, dist: Path) -> bool:
    """Serve the built frontend from `dist`. Returns False if there is no build.

    Takes its app and directory as arguments so tests can exercise it against a
    temporary dist. Reading the real one meant the tests passed vacuously
    wherever no build existed -- dist/ is gitignored, so that is every fresh
    clone and every CI run.
    """
    # Guarded on index.html, not the directory: an interrupted `vite build` can
    # leave a dist/ that exists but has no entry point, and serving a missing
    # file raises at request time. Absent or partial build -> API only, still up.
    index = dist / "index.html"
    if not index.is_file():
        return False

    @target.api_route("/{full_path:path}", methods=["GET", "HEAD"])
    def serve_spa(full_path: str) -> FileResponse:
        # Registered after the routers, so a real /api route has already
        # matched. An unmatched one must 404: a fetch that silently receives
        # HTML fails far away from its cause.
        if _is_api_path(full_path):
            raise HTTPException(status_code=404, detail="Not found")
        # Any real file in dist -- hashed bundles, favicon.svg, robots.txt, a
        # service worker -- is served as itself.
        file = _dist_file(dist, full_path)
        if file is not None:
            return FileResponse(file)
        if _wants_a_file(full_path):
            raise HTTPException(status_code=404, detail="Not found")
        # Only genuine client-side routes fall through to the SPA, so a refresh
        # on /sync works while a missing asset still fails honestly.
        return FileResponse(index)

    return True


mount_spa(app, FRONTEND_DIST)
