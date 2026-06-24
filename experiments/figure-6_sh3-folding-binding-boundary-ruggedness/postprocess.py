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
summary = {}
for key in [
    "obs_boundary_edge_share", "obs_boundary_energy_share", "obs_energy_share_enrichment",
    "p_energy_share_fixed", "p_enrichment_fixed", "moran_I", "p_value",
]:
    if key in globals():
        summary[key] = globals()[key]
_write_json(summary, outdir / "sh3_boundary_energy_summary.json")
for name in ["latent_resample_tmap_df", "eps_scan_df", "tmap_df"]:
    value = globals().get(name)
    if isinstance(value, pd.DataFrame):
        value.to_csv(outdir / f"sh3_{name}.csv", index=False)
for src_name, dst_name in {
    "sh3_tmap_df.csv": "sh3_regime_tmap_summary.csv",
    "sh3_eps_scan_df.csv": "sh3_epsilon_tmap_scan.csv",
    "sh3_latent_resample_tmap_df.csv": "sh3_limiter_resampling_summary.csv",
}.items():
    _copy_if_exists(outdir / src_name, outdir / dst_name)
