from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import matplotlib

from update_figure4_boundary_enrichment_exports import NOTEBOOK_DIR, SYSTEM_SPECS


matplotlib.use("Agg")

SUMMARY_NOTEBOOK = NOTEBOOK_DIR / "Figure_4_boundary_enrichment_summary.ipynb"
EXPORT_MARKER = "# Boundary enrichment export"


def run_code_cells(notebook_path: Path, *, stop_after_marker: str | None = None) -> dict:
    data = json.loads(notebook_path.read_text())
    namespace = {
        "__name__": "__main__",
        "display": lambda *args, **kwargs: None,
    }
    sys.modules.setdefault("py3Dmol", types.ModuleType("py3Dmol"))

    old_cwd = Path.cwd()
    os.chdir(NOTEBOOK_DIR)
    try:
        for idx, cell in enumerate(data["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue

            print(f"[{notebook_path.name}] Running cell {idx}", flush=True)
            start = time.time()
            exec(compile(source, f"{notebook_path.name}::cell_{idx}", "exec"), namespace)
            elapsed = time.time() - start
            print(f"[{notebook_path.name}] Cell {idx} finished in {elapsed:.2f}s", flush=True)

            if stop_after_marker and stop_after_marker in source:
                break
    finally:
        os.chdir(old_cwd)

    return namespace


def main() -> None:
    overall_start = time.time()

    for spec in SYSTEM_SPECS:
        notebook_path = NOTEBOOK_DIR / spec["notebook"]
        run_code_cells(notebook_path, stop_after_marker=EXPORT_MARKER)

    run_code_cells(SUMMARY_NOTEBOOK)

    elapsed = time.time() - overall_start
    print(f"Finished Figure 4 boundary enrichment export workflow in {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
