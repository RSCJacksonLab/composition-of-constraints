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

if "results" in globals():
    with (outdir / "alisim_tmap_by_rep_and_eig.pkl").open("wb") as handle:
        pickle.dump(results, handle)
    _write_json(results, outdir / "alisim_tmap_by_rep_and_eig.json")
if "df" in globals() and isinstance(df, pd.DataFrame):
    df.to_csv(outdir / "alisim_tmap_table.csv", index=False)
