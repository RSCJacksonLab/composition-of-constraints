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

outdir = PAPER_PROCESSED_DIR / "diffusion_scale_validation"
outdir.mkdir(parents=True, exist_ok=True)

if "replicate_dict" in globals():
    _write_json(replicate_dict, outdir / "nk_tmap_results.json")
nk_df = globals().get("df_nk")
if not isinstance(nk_df, pd.DataFrame):
    fallback_df = globals().get("df")
    if isinstance(fallback_df, pd.DataFrame) and {"K", "t_map"}.issubset(fallback_df.columns):
        nk_df = fallback_df
if isinstance(nk_df, pd.DataFrame):
    nk_df.to_csv(outdir / "nk_tmap_results.csv", index=False)
if "df_eig" in globals() and isinstance(df_eig, pd.DataFrame):
    df_eig.to_csv(outdir / "nk_eigenmode_tmap.csv", index=False)

stats_payload = {}
if isinstance(nk_df, pd.DataFrame):
    from scipy.stats import spearmanr
    stats_payload["nk_K_spearman"] = dict(zip(["rho", "p_value"], map(float, spearmanr(nk_df["K"], nk_df["t_map"]))))
if "df_eig" in globals() and isinstance(df_eig, pd.DataFrame):
    from scipy.stats import spearmanr
    stats_payload["nk_eigenmode_spearman"] = dict(zip(["rho", "p_value"], map(float, spearmanr(df_eig["eig_index"], df_eig["t_map"]))))
_write_json(stats_payload, outdir / "nk_statistics.json")
