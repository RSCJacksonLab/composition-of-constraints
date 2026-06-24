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

if "results" in globals():
    with (outdir / "megascale_folding_tmap.pkl").open("wb") as handle:
        pickle.dump(results, handle)
    _write_json(results, outdir / "megascale_folding_tmap.json")
if "dms_tmap_df" in globals() and isinstance(dms_tmap_df, pd.DataFrame):
    dms_tmap_df.to_csv(outdir / "megascale_folding_tmap_table.csv", index=False)

for src in [
    PAPER_DATA_FILES / "protein_gym" / "DMS_assays_substitutions" / "folding_permutation_short.df",
    PAPER_DATA_FILES / "protein_gym" / "DMS_assays_substitutions" / "folding_permutation_long.df",
]:
    _copy_if_exists(src, outdir / src.name)

for src in (PAPER_OUTPUT_DIR / "figures" / "SI_figures" / "SI_figure_DMS").glob("spectral_*.csv"):
    _copy_if_exists(src, outdir / src.name)
for src in (PAPER_OUTPUT_DIR / "figures" / "SI_figures" / "SI_figure_DMS").glob("*spectral*.csv"):
    _copy_if_exists(src, outdir / src.name)

for src_name, dst_name in {
    "spectral_partition_boundary_tmap_domain_summary.csv": "spectral_partition_domain_summary.csv",
    "spectral_cut_distance_profile.csv": "spectral_cut_shell_profiles.csv",
    "spectral_cut_cheeger_enrichment_summary.csv": "spectral_cheeger_summary.csv",
}.items():
    _copy_if_exists(outdir / src_name, outdir / dst_name)
