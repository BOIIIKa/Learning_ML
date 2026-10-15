#!/usr/bin/env bash
# Optional: serve Pyodide from the dev server instead of jsDelivr (offline work, or a network that
# blocks the CDN). Downloads the full Pyodide 0.29.3 release (~300 MB) once into .cache/ and links it.
#   bash scripts/fetch_pyodide.sh
#   cd site && PUBLIC_PYODIDE_URL=/pyodide/ npm run dev
# Remove the link (rm site/public/pyodide) before `npm run build`, or the build copies 300 MB into dist/.
set -euo pipefail
cd "$(dirname "$0")/.."
V=0.29.3
mkdir -p .cache
if [ ! -d .cache/pyodide ]; then
  curl -L --fail -o .cache/pyodide.tar.bz2 "https://github.com/pyodide/pyodide/releases/download/$V/pyodide-$V.tar.bz2"
  tar -xjf .cache/pyodide.tar.bz2 -C .cache
  rm .cache/pyodide.tar.bz2
fi
ln -sfn "$PWD/.cache/pyodide" site/public/pyodide
echo "Linked site/public/pyodide -> .cache/pyodide. Start the dev server with PUBLIC_PYODIDE_URL=/pyodide/."
