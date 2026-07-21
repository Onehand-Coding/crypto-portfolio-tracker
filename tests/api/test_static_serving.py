"""Static serving, exercised against a temporary dist.

These tests build their own frontend directory rather than reading the real
one. frontend/dist is gitignored, so tests that depend on a build present pass
vacuously in every fresh clone and every CI run -- the route under test is not
even registered there.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import app, mount_spa


@pytest.fixture
def dist(tmp_path):
    """A minimal build: an entry point, a hashed asset, and a root file."""
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>SPA</title>")
    (root / "assets" / "index-abc123.js").write_text("console.log('bundle')")
    (root / "favicon.svg").write_text("<svg/>")
    return root


@pytest.fixture
def client(dist):
    served = FastAPI()

    @served.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    assert mount_spa(served, dist) is True
    return TestClient(served)


def test_serves_the_spa_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SPA" in response.text


def test_serves_a_client_side_deep_link_as_the_spa(client):
    """A refresh on /sync must not 404 -- the route only exists in the browser."""
    response = client.get("/sync")
    assert response.status_code == 200
    assert "SPA" in response.text


def test_serves_a_hashed_asset_as_itself_not_the_spa(client):
    response = client.get("/assets/index-abc123.js")
    assert response.status_code == 200
    assert "bundle" in response.text
    assert "javascript" in response.headers["content-type"]


def test_serves_a_dist_root_file_as_itself(client):
    """Regression: everything outside /assets used to fall through to
    index.html, so favicon.svg -- and any robots.txt or service worker added
    later -- was served as HTML with a 200."""
    response = client.get("/favicon.svg")
    assert response.status_code == 200
    assert "<svg/>" in response.text
    assert "html" not in response.headers["content-type"]


def test_a_missing_asset_404s_rather_than_returning_html(client):
    """A stale bundle reference should fail loudly, not arrive as HTML with a
    200 and break later on MIME type."""
    response = client.get("/assets/index-deleted.js")
    assert response.status_code == 404


@pytest.mark.parametrize("path", ["/assets/BTC", "/assets/ETH", "/sync", "/rebalance"])
def test_client_side_routes_reach_the_spa(client, path):
    """/assets/:symbol is a real client route and shares its prefix with the
    bundle directory. Treating everything under assets/ as a file made the
    asset-detail page 404 in the browser while every test still passed."""
    response = client.get(path)
    assert response.status_code == 200
    assert "SPA" in response.text


def test_unknown_api_path_returns_404_not_the_spa(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert "SPA" not in response.text


@pytest.mark.parametrize("path", ["/api", "/api/", "//api/nope"])
def test_api_prefix_variants_never_receive_the_spa(client, path):
    """A base URL built with a trailing slash produces "//api/...", which the
    naive startswith("api/") check let through as HTML.

    Requested as an absolute URL: httpx reads a bare "//api/nope" as a
    protocol-relative URL and rewrites it to host "api", so the relative form
    never reaches the server at all.
    """
    response = client.request("GET", "http://testserver" + path)
    assert response.status_code == 404


def test_real_api_routes_still_resolve_behind_the_catch_all(client):
    assert client.get("/api/health").json() == {"status": "ok"}


@pytest.mark.parametrize("path", [
    "/../pyproject.toml",
    "/assets/../../pyproject.toml",
    "/....//pyproject.toml",
])
def test_path_traversal_cannot_escape_dist(client, path):
    """full_path is joined onto a filesystem path; containment is what stops
    it resolving to .env."""
    response = client.get(path)
    assert response.status_code in (200, 404)
    assert "[project]" not in response.text


def test_head_on_the_spa_is_not_405(client):
    """Uptime monitors and some proxies probe with HEAD. FastAPI's APIRoute,
    unlike Starlette's Route, does not add it automatically."""
    assert client.head("/").status_code == 200


def test_no_build_leaves_the_api_up(tmp_path):
    """An absent or interrupted build must not take the JSON API down with it."""
    empty = FastAPI()

    @empty.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    assert mount_spa(empty, tmp_path / "nonexistent") is False
    assert TestClient(empty).get("/api/health").json() == {"status": "ok"}


def test_dist_without_an_index_does_not_crash_at_import(tmp_path):
    """Regression: the old guard checked the directory and then mounted
    dist/assets unconditionally, so a partial build raised RuntimeError on
    import and the whole app -- API included -- failed to start."""
    partial = tmp_path / "dist"
    partial.mkdir()
    assert mount_spa(FastAPI(), partial) is False


def test_the_real_app_exposes_health():
    """Smoke test on the actual app object, whatever its build state."""
    assert TestClient(app).get("/api/health").json() == {"status": "ok"}
