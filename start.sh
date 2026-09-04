#!/bin/bash
# Root start script - runs the React UI.
#
# Unlike the sibling projects, this app is ONE process in production: FastAPI
# serves the built React bundle, so there is no separate frontend server and no
# CORS. Pass --dev for the two-process arrangement (Vite on 5173 proxying /api
# to 8000), which gives hot reload while developing the frontend.
#
#   ./start.sh          build the bundle, serve everything from :8000
#   ./start.sh --dev    uvicorn --reload on :8000 + Vite on :5173
#   ./start.sh --skip-build   serve the existing bundle without rebuilding

set -uo pipefail

# Headless matplotlib: no display on a server
export MPLBACKEND=Agg

DEV=false
SKIP_BUILD=false
for arg in "$@"; do
    case "$arg" in
        --dev) DEV=true ;;
        --skip-build) SKIP_BUILD=true ;;
        -h|--help)
            cat <<'USAGE'
Runs the React UI for the Crypto Portfolio Tracker.

In production this is ONE process: FastAPI serves the built React bundle,
so there is no separate frontend server and no CORS.

  ./start.sh                build the bundle, serve everything from :8000
  ./start.sh --dev          uvicorn --reload on :8000 + Vite on :5173
  ./start.sh --skip-build   serve the existing bundle without rebuilding
USAGE
            exit 0 ;;
        *)
            echo "Unknown option: $arg (try --help)"
            exit 1 ;;
    esac
done

echo "📊 Starting Crypto Portfolio Tracker..."

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"

# The config resolves data/ relative to the working directory, so a run started
# from elsewhere reads a different (empty) database and cache.
cd "$ROOT_DIR" || { echo "❌ Cannot enter $ROOT_DIR"; exit 1; }

# ==================================
# Cleanup: Kill any existing processes (gracefully)
# ==================================
echo "🧹 Cleaning up existing processes..."

PID_FILE="$ROOT_DIR/.crypto_tracker.pids"
API_PID=""
FRONTEND_PID=""

graceful_kill() {
    local pid=$1
    local name=$2
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "   Sending SIGTERM to $name (PID: $pid)..."
        kill -15 "$pid" 2>/dev/null
        for _ in 1 2 3 4 5; do
            sleep 1
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "   ✓ $name stopped gracefully"
                return 0
            fi
        done
        echo "   ⚠ Force-stopping $name..."
        kill -9 "$pid" 2>/dev/null
    fi
    return 0
}

if [ -f "$PID_FILE" ]; then
    # shellcheck disable=SC1090
    source "$PID_FILE"
    [ -n "$API_PID" ] && graceful_kill "$API_PID" "api"
    [ -n "$FRONTEND_PID" ] && graceful_kill "$FRONTEND_PID" "frontend"
    rm -f "$PID_FILE"
fi

# Fallback if the PID file was lost. Matched narrowly on this app's own
# uvicorn target so it cannot take down an unrelated server on the machine.
pkill -15 -f 'uvicorn.*api\.main:app' 2>/dev/null
pkill -15 -f 'api\.main:app' 2>/dev/null
$DEV && pkill -15 -f "vite.*crypto-portfolio-tracker" 2>/dev/null

sleep 1
echo "✅ Cleanup complete"
echo ""

# ==================================
# Preflight
# ==================================
if ! command -v uv >/dev/null 2>&1; then
    echo "❌ uv not found. Install it: https://docs.astral.sh/uv/"
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "❌ npm not found. The React UI needs Node."
    exit 1
fi

# Deliberately only a warning, and never auto-generated: this .env holds real
# Binance API credentials. There is nothing safe to invent here.
if [ ! -f "$ROOT_DIR/.env" ]; then
    echo "⚠️  No .env found. The app will start, but syncing needs Binance"
    echo "   credentials. Copy .env.example to .env and fill it in."
    echo ""
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "📥 Installing frontend dependencies..."
    npm --prefix "$FRONTEND_DIR" install || { echo "❌ npm install failed"; exit 1; }
    echo ""
fi

# ==================================
# Shutdown
# ==================================
cleanup() {
    echo ""
    echo "👋 Shutting down..."
    [ -n "$API_PID" ] && graceful_kill "$API_PID" "api"
    [ -n "$FRONTEND_PID" ] && graceful_kill "$FRONTEND_PID" "frontend"
    rm -f "$PID_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ==================================
# Start
# ==================================
if [ "$DEV" = true ]; then
    echo "🔧 Dev mode: Vite serves the frontend and proxies /api to :8000"
    echo ""

    echo "📦 Starting API (auto-reload)..."
    uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload &
    API_PID=$!

    sleep 2

    echo "🎨 Starting Vite..."
    npm --prefix "$FRONTEND_DIR" run dev &
    FRONTEND_PID=$!

    UI_URL="http://localhost:5173"
else
    if [ "$SKIP_BUILD" = false ]; then
        echo "🔨 Building the frontend..."
        # Aborting on a failed build is the point: serving a stale bundle looks
        # like a working app showing figures from whenever it last built.
        if ! npm --prefix "$FRONTEND_DIR" run build; then
            echo ""
            echo "❌ Build failed. Not starting the server -- the existing bundle"
            echo "   is stale and would silently serve old code."
            exit 1
        fi
        echo ""
    elif [ ! -f "$FRONTEND_DIR/dist/index.html" ]; then
        echo "❌ --skip-build passed but no build exists yet. Run without it first."
        exit 1
    fi

    echo "📦 Starting the server..."
    uv run python run_ui.py &
    API_PID=$!

    UI_URL="http://localhost:8000"
fi

cat > "$PID_FILE" << EOF
API_PID=$API_PID
FRONTEND_PID=$FRONTEND_PID
EOF

sleep 2

# Confirm it actually came up rather than printing a URL for a dead process.
if ! kill -0 "$API_PID" 2>/dev/null; then
    echo ""
    echo "❌ The API exited on startup. Scroll up for the traceback."
    rm -f "$PID_FILE"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UI:       $UI_URL"
echo "  API:      http://localhost:8000/api/health"
echo "  API Docs: http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Sync is the only action that contacts Binance."
echo "Press Ctrl+C to stop."
echo ""

wait $API_PID $FRONTEND_PID
