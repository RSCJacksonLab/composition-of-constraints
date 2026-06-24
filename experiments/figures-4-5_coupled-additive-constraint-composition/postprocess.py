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

outdir = PAPER_PROCESSED_DIR / "constraint_composition"
outdir.mkdir(parents=True, exist_ok=True)

def _safe_csv(frame, path):
    frame = frame.copy()
    for col in ["nk_landscape"]:
        if col in frame.columns:
            frame = frame.drop(columns=[col])
    frame.to_csv(path, index=False)

if "nk_results" in globals():
    with (outdir / "nk_min_composition_results.pkl").open("wb") as handle:
        pickle.dump(nk_results, handle)
if "df" in globals() and isinstance(df, pd.DataFrame):
    _safe_csv(df, outdir / "nk_min_composition_results.csv")
for name in ["edge_energy_summary", "summary_df", "product_results_df", "soft_results_df", "softmin_df", "product_df"]:
    value = globals().get(name)
    if isinstance(value, pd.DataFrame):
        _safe_csv(value, outdir / f"{name}.csv")
if "edge_energy_summary" in globals() and isinstance(edge_energy_summary, pd.DataFrame):
    _safe_csv(edge_energy_summary, outdir / "boundary_switching_energy_summary.csv")
elif "summary_df" in globals() and isinstance(summary_df, pd.DataFrame):
    _safe_csv(summary_df, outdir / "boundary_switching_energy_summary.csv")
control_frames = []
if "softmin_df" in globals() and isinstance(softmin_df, pd.DataFrame):
    control_frames.append(softmin_df.assign(control="softmin"))
if "product_df" in globals() and isinstance(product_df, pd.DataFrame):
    control_frames.append(product_df.assign(control="product"))
if control_frames:
    _safe_csv(pd.concat(control_frames, ignore_index=True), outdir / "softmin_product_control_summary.csv")
_write_json({k: v for k, v in globals().items() if k.endswith("_results") and k != "nk_results"}, outdir / "composition_auxiliary_results.json")
