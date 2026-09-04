"""Start the React UI on http://127.0.0.1:8000.

Run from the project root: the config resolves data/ relative to the working
directory, so launching from elsewhere reads a different (empty) cache.
"""

import os

# Headless matplotlib: pin the backend before any local import pulls in
# pyplot (api -> visualizations imports it at module scope).
os.environ.setdefault("MPLBACKEND", "Agg")

import uvicorn

from api.main import FRONTEND_DIST

if __name__ == "__main__":
    # Without this the server still starts and the API works, but every page
    # request 404s with no indication that a build is what's missing.
    if not (FRONTEND_DIST / "index.html").is_file():
        print("No frontend build found. Run:  npm --prefix frontend run build")
        print("Starting the API only.\n")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
