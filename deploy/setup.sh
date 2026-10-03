#!/usr/bin/env bash
# Builds everything the backend needs, as your normal user (NOT root):
# .venv + Python deps, the React frontend, and sanity checks for .env and
# the AdaFace model. Safe to re-run. Called automatically by install.sh.
#   bash deploy/setup.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python3}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ "$EUID" -ne 0 ] || fail "Run setup.sh as your normal user, not root."
command -v "$PYTHON" >/dev/null || fail "$PYTHON not found. Install Python 3.10+."

# nvm only puts node/npm on PATH in interactive shells, not under sudo.
if ! command -v npm >/dev/null && [ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]; then
  set +u
  # shellcheck disable=SC1091
  . "${NVM_DIR:-$HOME/.nvm}/nvm.sh"
  set -u
fi
command -v npm >/dev/null || fail "npm not found. Install Node.js 20+ (needed to build the frontend)."

[ -f .env ] || fail ".env missing in $PROJECT_DIR. Copy it from the original machine (or from 'CCTV Face AI Camera Pipeline.md')."
[ -f models/adaface_ir101_webface12m.onnx ] || fail "models/adaface_ir101_webface12m.onnx missing (~260MB, not in git). Copy it from the original machine, or generate it with scripts/export_adaface_onnx.py."

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtualenv..."
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip -q

# paddlepaddle/paddleocr are only used by the legacy OCR mode, never by the
# attendance server, and have no wheels for newer Pythons -- skip them.
echo "Installing Python dependencies (first run downloads several GB)..."
REQ_TMP="$(mktemp)"
trap 'rm -f "$REQ_TMP"' EXIT
grep -vE '^(paddlepaddle|paddleocr)([=<> ]|$)' requirements.txt > "$REQ_TMP"
.venv/bin/pip install -r "$REQ_TMP"

# insightface depends on the CPU "onnxruntime" wheel, which overwrites the
# files of onnxruntime-gpu (same module name) and silently disables CUDA.
if .venv/bin/pip show onnxruntime >/dev/null 2>&1; then
  echo "Removing CPU onnxruntime that shadows onnxruntime-gpu..."
  .venv/bin/pip uninstall -y onnxruntime
  ORT_GPU_REQ="$(grep -E '^onnxruntime-gpu' requirements.txt | awk '{print $1}')"
  .venv/bin/pip install --force-reinstall --no-deps "${ORT_GPU_REQ:-onnxruntime-gpu}"
fi

echo "Building frontend..."
(cd frontend && npm install --no-audit --no-fund && npm run build)

echo
echo "Setup complete. ONNX Runtime providers:"
LD_LIBRARY_PATH="$(bash deploy/cuda_lib_path.sh)" .venv/bin/python -c \
  "import onnxruntime as o; print(' ', o.get_available_providers())"
echo "(No CUDAExecutionProvider = no NVIDIA GPU/driver here; it still runs on CPU, just slower.)"
