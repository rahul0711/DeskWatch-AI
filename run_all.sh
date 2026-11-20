#!/usr/bin/env bash
# Starts the whole project for local development, in one terminal:
#   .NET backend    (CCTV_Backend_Project)         -> :$DB_API_PORT   logins, roles, people, reports (MySQL)
#   attendance API  (app/web/attendance_server.py) -> :$ATTENDANCE_PORT
#   camera live view (app/web/server.py)           -> :$CAMERAS_PORT
#   React frontend  (Vite dev server)              -> :$FRONTEND_PORT
# Ctrl+C stops all four.
#   bash run_all.sh
# Override ports with e.g. ATTENDANCE_PORT=8005 bash run_all.sh
# (default 8003, not 8001 -- the Face_Testing_Website project already uses 8001).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

ATTENDANCE_PORT="${ATTENDANCE_PORT:-8003}"
CAMERAS_PORT="${CAMERAS_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
DB_API_PORT="${DB_API_PORT:-5075}"
DOTNET_DIR="$PROJECT_DIR/CCTV_Backend_Project/CCTV_Backend_Project"
# Built here, not in bin/Debug, so the copy of appsettings.json in bin/ is left alone.
DOTNET_OUT="$PROJECT_DIR/.dotnet-run"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ -x .venv/bin/python ] || fail ".venv missing -- run: bash deploy/setup.sh"
[ -f .env ] || fail ".env missing in $PROJECT_DIR"
[ -d frontend/node_modules ] || (cd frontend && npm install --no-audit --no-fund)
command -v dotnet >/dev/null || fail "dotnet SDK missing -- install .NET 8 SDK"

for port in "$DB_API_PORT" "$ATTENDANCE_PORT" "$CAMERAS_PORT" "$FRONTEND_PORT"; do
  if [ -n "$(ss -Hltn "sport = :$port")" ]; then
    fail "port $port is already in use (see: ss -ltnp | grep :$port)"
  fi
done

echo "Building .NET backend..."
dotnet build "$DOTNET_DIR/CCTV_Backend_Project.csproj" -c Release -o "$DOTNET_OUT" --nologo -v quiet \
  || fail ".NET backend build failed"

# Use this project's venv explicitly, whatever venv the shell has active.
PY="$PROJECT_DIR/.venv/bin/python"
export LD_LIBRARY_PATH="$(bash deploy/cuda_lib_path.sh)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Kills the whole process group (uvicorn workers, npm's node child, the sed
# log prefixers), not just the PIDs of the pipelines started below.
cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping..."
  trap '' TERM
  kill -TERM 0 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Run from the project folder so it reads CCTV_Backend_Project/appsettings.json.
(cd "$DOTNET_DIR" && exec dotnet "$DOTNET_OUT/CCTV_Backend_Project.dll" --urls "http://0.0.0.0:$DB_API_PORT") \
  2>&1 | sed -u 's/^/[dotnet]     /' &

"$PY" -m uvicorn app.web.attendance_server:app --host 0.0.0.0 --port "$ATTENDANCE_PORT" \
  2>&1 | sed -u 's/^/[attendance] /' &

"$PY" -m uvicorn app.web.server:app --host 0.0.0.0 --port "$CAMERAS_PORT" \
  2>&1 | sed -u 's/^/[cameras]    /' &

(cd frontend && API_TARGET="http://127.0.0.1:$ATTENDANCE_PORT" \
  DB_API_TARGET="http://127.0.0.1:$DB_API_PORT" \
  CAMERAS_TARGET="http://127.0.0.1:$CAMERAS_PORT" \
  exec npm run dev -- --host --port "$FRONTEND_PORT" --strictPort) \
  2>&1 | sed -u 's/^/[frontend]   /' &

echo "Frontend:       http://localhost:$FRONTEND_PORT"
echo ".NET backend:   http://localhost:$DB_API_PORT/api/db/health"
echo "Attendance API: http://localhost:$ATTENDANCE_PORT"
echo "Camera view:    http://localhost:$CAMERAS_PORT"
echo "Press Ctrl+C to stop everything."

# Exit (and stop the rest) as soon as any one of them dies.
wait -n
echo "A process exited -- shutting down the others."
