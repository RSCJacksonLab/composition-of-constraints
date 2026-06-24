from pathlib import Path
import json
import math
import pickle
import shutil

import numpy as np
import pandas as pd


def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Promoted {src} -> {dst}")

outdir = PAPER_PROCESSED_DIR / "experimental_boundaries"
outdir.mkdir(parents=True, exist_ok=True)
if "min_summary_df" in globals() and isinstance(min_summary_df, pd.DataFrame):
    min_summary_df.to_json(outdir / "lycov_min_operator_summary.json", orient="records", indent=2)
if "proxy_summaries" in globals():
    trimmed = []
    for row in proxy_summaries:
        trimmed.append({k: v for k, v in row.items() if k != "fitted"})
    _write_json(trimmed, outdir / "lycov_min_operator_proxy_summaries.json")
source_dir = Path.cwd() / "boundary_enrichment_results"
_copy_if_exists(source_dir / "lycov_boundary_enrichment.json", outdir / "lycov_boundary_enrichment.json")
_copy_if_exists(PAPER_DATA_FILES / "lycov_combination_antibodies" / "lycov_combination_antibodies.lsbundle", outdir / "lycov_combination_antibodies.lsbundle")
