#!/usr/bin/env bash
# One-time local setup on macOS / Linux (Windows: scripts/setup_local.ps1, or WSL).
#   bash scripts/setup_local.sh
# Needs Node 22 (see .nvmrc) and Python 3.11+ on PATH. Creates .venv, installs the Python tools,
# the site's npm packages and Playwright's Chromium, then runs the fast checks once.
set -euo pipefail
cd "$(dirname "$0")/.."

node_major=$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)
if [ "$node_major" -lt 22 ]; then echo "Node 22+ is required (found: $(node -v 2>/dev/null || echo none)); try: nvm install 22" >&2; exit 1; fi
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), sys.version' || { echo "Python 3.11+ is required" >&2; exit 1; }

echo "== Python tools (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements-dev.txt

echo "== Site packages"
(cd site && npm ci && npx playwright install chromium)

echo "== Generated files"
.venv/bin/python scripts/build_exercises.py
.venv/bin/python scripts/build_outline.py --check

echo "== Fast checks"
(cd site && npx tsc --noEmit -p . && npx vitest run)
.venv/bin/python -m pytest -q tests

cat <<'MSG'

Done. Daily use:
  source .venv/bin/activate          # python, pytest, scripts/*.py
  cd site && npm run dev             # http://localhost:4321/zh/  (Pyodide loads from jsDelivr)
See docs/LOCAL_DEV.md for the workflow, verify scripts and lessons learned.
MSG
