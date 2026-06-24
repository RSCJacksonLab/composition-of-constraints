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

outdir = PAPER_PROCESSED_DIR / "stability_dms"
outdir.mkdir(parents=True, exist_ok=True)

for name in [
    "sv_limiter_domain_df",
    "sv_limiter_assoc_df",
    "sv_limiter_boundary_dirichlet_domain_df",
    "sv_limiter_boundary_dirichlet_edge_df",
    "sv_limiter_boundary_dirichlet_summary_df",
    "svd_spectral_overlap_domain_df",
    "svd_spectral_overlap_stats_df",
]:
    value = globals().get(name)
    if isinstance(value, pd.DataFrame):
        value.to_csv(outdir / f"{name}.csv", index=False)

for base in [PAPER_OUTPUT_DIR / "figures" / "SI_figures" / "SI_figure_DMS", PAPER_OUTPUT_DIR / "figures" / "figure_2"]:
    if base.exists():
        for src in base.glob("sv*.csv"):
            _copy_if_exists(src, outdir / src.name)
        for src in base.glob("*limiter*.csv"):
            _copy_if_exists(src, outdir / src.name)
