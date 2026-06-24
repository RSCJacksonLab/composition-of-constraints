#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

cd "${ROOT}"

echo "[check] shell syntax"
find experiments scripts -type f \( -name '*.sh' -o -name 'run.sh' \) -print0 |
  while IFS= read -r -d '' file; do
    bash -n "${file}"
  done

echo "[check] python syntax"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

for root in [Path("experiments"), Path("analyses"), Path("scripts")]:
    for path in root.rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("python syntax ok")
PY

echo "[check] notebook JSON"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json

for path in [*Path("analyses").glob("*/notebooks/*.ipynb"), *Path("experiments").glob("*/source/figure_notebooks_rev/*.ipynb")]:
    json.loads(path.read_text(encoding="utf-8"))
print("notebooks ok")
PY

echo "[check] complete"
