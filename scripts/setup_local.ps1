# One-time local setup on Windows (PowerShell). Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1
# Needs Node 22 (see .nvmrc) and Python 3.11+ on PATH.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$nodeMajor = [int](node -p "process.versions.node.split('.')[0]")
if ($nodeMajor -lt 22) { throw "Node 22+ is required (found $(node -v))" }
python -c "import sys; assert sys.version_info >= (3, 11), sys.version"

Write-Host "== Python tools (.venv)"
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python -m pip install -q --upgrade pip
.\.venv\Scripts\python -m pip install -q -r requirements-dev.txt

Write-Host "== Site packages"
Push-Location site
npm ci
npx playwright install chromium
Pop-Location

Write-Host "== Generated files and fast checks"
.\.venv\Scripts\python scripts\build_exercises.py
.\.venv\Scripts\python scripts\build_outline.py --check
Push-Location site
npx tsc --noEmit -p .
npx vitest run
Pop-Location
.\.venv\Scripts\python -m pytest -q tests

Write-Host "`nDone. Daily use: .\.venv\Scripts\Activate.ps1; cd site; npm run dev  ->  http://localhost:4321/zh/"
Write-Host "See docs\LOCAL_DEV.md for the workflow, verify scripts and lessons learned."
