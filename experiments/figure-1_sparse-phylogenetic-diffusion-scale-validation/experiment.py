#!/usr/bin/env python3
"""Publication experiment runner.

Experiment code below is copied from Figure_1.ipynb cells 0,24,26-27,31-35 in
the copied source/figure_notebooks_rev snapshot. This script replaces legacy `.ipynb` cell execution while preserving the same
scientific operations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from paper_runtime import prepare_native_experiment, run_postprocess

ENV = prepare_native_experiment(__file__)
SCRIPT_DIR = ENV["script_dir"]
PROJECT_ROOT = ENV["project_root"]
OUTPUT_DIR = ENV["output_dir"]
WORK_DIR = ENV["work_dir"]
SOURCE_ROOT = ENV["source_root"]
NOTEBOOK_DIR = ENV["notebook_dir"]
PAPER_PROJECT_ROOT = ENV["project_root"]
PAPER_OUTPUT_DIR = ENV["output_dir"]
PAPER_PROCESSED_DIR = ENV["processed_dir"]
PAPER_DATA_FILES = ENV["data_files"]
PAPER_ALISIM_RESULTS = ENV.get("alisim_results")

ALISIM_FASTA_SUFFIXES = {".fa", ".fasta", ".fas", ".faa"}


def _is_aligned_alisim_fasta(path):
    name = path.name.lower()
    return (
        path.is_file()
        and path.suffix.lower() in ALISIM_FASTA_SUFFIXES
        and "unaligned" not in name
    )


def _discover_aligned_alisim_fastas(path):
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob("*") if _is_aligned_alisim_fasta(p))


def _replicate_id_from_path(path):
    stem = Path(path).stem
    match = re.search(r"(?:replicate|rep|sim)[_-]?0*([0-9]+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"([0-9]+)", stem)
    if match:
        return match.group(1).lstrip("0") or "0"
    return stem


def _resolve_alisim_results_dir():
    candidates = [
        os.environ.get("PAPER_ALISIM_RESULTS"),
        PAPER_ALISIM_RESULTS,
        PROJECT_ROOT / "data" / "alisim_results",
        PROJECT_ROOT / "data" / "raw" / "alisim_results",
        PROJECT_ROOT / "alisim_results",
        SOURCE_ROOT / "alisim_results",
    ]
    checked = []
    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        checked.append(path)
        if _discover_aligned_alisim_fastas(path):
            return path

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise RuntimeError(
        "No aligned AliSim FASTA files found. Expected files named "
        "*.fa, *.fasta, *.fas, or *.faa, excluding names containing "
        "'unaligned', in one of:\n"
        f"{checked_text}\n\n"
        "Place the publication data bundle at data/alisim_results, keep the "
        "project-record layout at data/raw/alisim_results, or set "
        "PAPER_ALISIM_RESULTS=/path/to/alisim_results."
    )

def display(*args, **kwargs):
    return None

os.chdir(NOTEBOOK_DIR)
print("[paper-exp] Running copied notebook-derived experiment code", flush=True)

# --- Begin copied code from Figure_1.ipynb cells 0,24,26-27,31-35 ---

# %% [Figure_1.ipynb cell 0]
import fitness_landscape as fl
from fitness_landscape.utils import fasta_to_prot20_sequences
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.transforms.eigenmode import eigenmode_decomposition
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
import pickle
import json
from tqdm import tqdm
import networkx as nx
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from scipy.stats import norm
import matplotlib as mpl

# %% [Figure_1.ipynb cell 24]

indir = _resolve_alisim_results_dir()
pattern = "*.fa/*.fasta/*.fas/*.faa, excluding names containing 'unaligned'"
eig_max = int(os.environ.get("PAPER_ALISIM_EIG_MAX", "32"))
max_reps = int(os.environ.get("PAPER_ALISIM_MAX_REPS", "100"))
out_pkl = WORK_DIR / "tmap_by_rep_and_eig.pkl"
out_json = WORK_DIR / "tmap_by_rep_and_eig.json"

def _jsonable(obj):
    import numpy as _np
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, (_np.integer,)): return int(obj)
    if isinstance(obj, (_np.ndarray,)): return obj.tolist()
    if isinstance(obj, dict): return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_jsonable(x) for x in obj]
    return obj

files = _discover_aligned_alisim_fastas(indir)
if not files:
    observed = sorted(p.name for p in indir.iterdir())[:20] if indir.is_dir() else []
    raise RuntimeError(
        f"No aligned files found in resolved AliSim directory {indir} matching {pattern}. "
        f"First observed entries: {observed}"
    )

print(f"[paper-exp] Using {len(files)} aligned AliSim FASTA files from {indir}", flush=True)

count = 0
results = {}

for i, fasta_path in tqdm(enumerate(files)):
    if count == max_reps:
        break
    rep_id = _replicate_id_from_path(fasta_path)

    sequences = fasta_to_prot20_sequences(fasta_path)
    knn_k = max(int(np.sqrt(len(sequences))), 2)

    landscape = FitnessLandscape.build(
        sequences,
        graph="knn",
        k=knn_k,
        backend="auto",
    )

    # Check if there is more than a single connected component and skip if so
    G = landscape.graph  # nx.Graph
    if not nx.is_connected(G):
        continue

    # Compute eigenpairs once (0..eig_max)
    eigvals, eigvecs = eigenmode_decomposition(
        landscape,
        matrix="laplacian",
        k=eig_max + 1,
    )

    rep_out = {
        "file": str(fasta_path),
        "n_sequences": len(sequences),
        "knn_k": knn_k,
        "eig_max": eig_max,
        "by_eig_k": {},
    }

    for eig_k in range(1, eig_max + 1):  # 1..64 (skip 0)
        layer_name = f"laplacian_eigvec_{eig_k}"
        landscape.attach(name=layer_name, values=eigvecs[:, eig_k], dtype="numeric")
        landscape.view(layer_name)

        tmap_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(landscape, t_min=1e-10, t_max=1e2, prior="uniform")

        rep_out["by_eig_k"][eig_k] = {
            "eig_k": eig_k,
            "eigval": eigvals[eig_k],
            "tmap": tmap_res,
        }

    results[rep_id] = rep_out
    count += 1

# %% [Figure_1.ipynb cell 26]
# pkl_path = Path("../alisim_results/tmap_by_rep_and_eig.pkl")

# with open(pkl_path, "rb") as f:
#     results = pickle.load(f)


def is_reliable_tmap_fit(tmap_result, min_lower=1e-10, max_ci_orders=2):
    t_map = float(tmap_result["t_map"])
    t_lo = float(tmap_result["t_lower_confidence_interval"])
    t_hi = float(tmap_result["t_upper_confidence_interval"])

    if not np.isfinite([t_map, t_lo, t_hi]).all():
        return False
    if t_lo <= min_lower or t_hi <= t_lo:
        return False
    if not (t_lo <= t_map <= t_hi):
        return False
    if np.log10(t_hi) - np.log10(t_lo) > max_ci_orders:
        return False

    return True

# %% [Figure_1.ipynb cell 27]
# Collect reliable t_map values per eigenmode index and replicate
max_eig = 24
cmap = plt.get_cmap("viridis")
replicate_color = cmap(0.0)

fig, ax = plt.subplots(figsize=(3, 2.75))

for rep_data in results.values():
    points = []

    for eig_k, d in sorted(rep_data["by_eig_k"].items()):
        if eig_k > max_eig:
            continue

        tmap_result = d["tmap"]
        if not is_reliable_tmap_fit(tmap_result):
            continue

        points.append((eig_k, float(tmap_result["t_map"])))

    if not points:
        continue

    eig_indices = [eig_k for eig_k, _ in points]
    tmap_values = [t_map for _, t_map in points]

    ax.plot(
        tmap_values,
        eig_indices,
        color='black',#replicate_color,
        alpha=0.25,
        marker="o",
        markersize=0,
        linestyle="-",
        linewidth=1,
    )

ax.set_xlabel(r"$t_{MAP}$")
ax.set_ylabel("Laplacian eigenvector index")
ax.invert_yaxis()

ax.set_yticks(range(0, max_eig + 1, 10))
ax.set_yticklabels([str(k) for k in range(0, max_eig + 1, 10)])
ax.grid(True, which="both", ls="--", c="0.7")
ax.set_xscale("log")
plt.tight_layout()
plt.savefig("../figures/figure_1/tmap_vs_laplacian_eigenvector_index.pdf")
plt.show()

# %% [Figure_1.ipynb cell 31]
rows = []

for rep_id, rep_out in results.items():
    n = int(rep_out["n_sequences"])
    eig_max = int(rep_out.get("eig_max", 0))
    knn_k = int(rep_out["knn_k"])

    by_eig = rep_out["by_eig_k"]

    for eig_k, entry in by_eig.items():
        tmap = entry["tmap"]

        t_map = float(tmap["t_map"])
        t_lo  = float(tmap["t_lower_confidence_interval"])
        t_hi  = float(tmap["t_upper_confidence_interval"])

        eig_index_norm = eig_k / eig_max if eig_max > 0 else np.nan

        rows.append({
            "rep": rep_id,
            "file": rep_out["file"],
            "N": n,                 # NOTE: here N = n_sequences (rename if you want)
            "knn_k": knn_k,
            "eig_max": eig_max,
            "eig_k": int(eig_k),
            "eig_index_norm": float(eig_index_norm),
            "eigval": float(entry["eigval"]),
            "t_map": t_map,
            "t_lo": t_lo,
            "t_hi": t_hi,
            "logpost_map": float(tmap["t_logposterior_map"]),
            "var_approx": float(tmap["variance_approximate"]),
        })

df = pd.DataFrame(rows)

# basic cleaning for safety
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["t_map", "eig_index_norm", "N"])
df = df[df["t_map"] > 0].copy()

df["log_t_map"] = np.log(df["t_map"])

# %% [Figure_1.ipynb cell 32]
rho, p_value = spearmanr(df["eig_k"], df["t_map"])
print(f"Spearman correlation between Laplacian eigenvector index and t_map: rho={rho:.4f}, p-value={p_value:.4g}")

# %% [Figure_1.ipynb cell 33]
df

# %% [Figure_1.ipynb cell 34]
model_j = smf.ols(
    "t_map ~ eig_k",
    data=df
).fit(cov_type="HC3")

print(model_j.summary())

# %% [Figure_1.ipynb cell 35]
# Compute p-value from z-score
from scipy.stats import norm

z = -81.838

log10_p_two_sided = np.log10(2) + norm.logsf(abs(z)) / np.log(10)
log10_p_two_sided

# --- End copied code from Figure_1.ipynb cells 0,24,26-27,31-35 ---

run_postprocess(SCRIPT_DIR / "postprocess.py", globals())
