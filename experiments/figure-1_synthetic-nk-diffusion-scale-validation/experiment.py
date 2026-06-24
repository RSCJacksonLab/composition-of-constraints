#!/usr/bin/env python3
"""Publication experiment runner.

Experiment code below is copied from Figure_1.ipynb cells 0,6-7,9-14,16-17,22 in
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

def display(*args, **kwargs):
    return None

os.chdir(NOTEBOOK_DIR)
print("[paper-exp] Running copied notebook-derived experiment code", flush=True)

# --- Begin copied code from Figure_1.ipynb cells 0,6-7,9-14,16-17,22 ---

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

# %% [Figure_1.ipynb cell 6]
# Init dict to store replicates
replicate_dict = {}

# Number of replicates
num_replicates = 10
for replicate in range(num_replicates):

    # Init dict to store results
    tmap_dict = {}

    # Range of N variables to consider
    n_range = list(range(4, 11))

    # Define K up to N-1 for each N
    for n_param in n_range:
        k_range = list(range(0, n_param))

        # Construct NK landscapes and compute diffusion map for each (N, K)
        for k_param in k_range:
            nk = fl.models.nk.create_nk_binary_landscape(N=n_param, K=k_param, seed=replicate)
            tmap_dict[(n_param, k_param)] = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(nk, t_min=1e-10, t_max=1e2, prior='uniform')

    # Store replicate results
    replicate_dict[replicate] = tmap_dict

# %% [Figure_1.ipynb cell 7]

# replicate_dict: {rep: {(N, K): {'t_map': ... , ...}, ...}, ...}
# Restructure to: restructured[rep][N] -> list of (K, t_map)

restructured = defaultdict(lambda: defaultdict(list))
all_n_values = set()

for rep, tmap_dict in replicate_dict.items():
    for (n, k), res in tmap_dict.items():
        t = res["t_map"]  # <- new field name
        restructured[rep][n].append((k, t))
        all_n_values.add(n)

sorted_n = sorted(all_n_values)

fig, ax = plt.subplots(figsize=(3, 2.75))
cmap = plt.get_cmap("viridis")
colors = {n: cmap(i / (len(sorted_n) - 1 if len(sorted_n) > 1 else 1))
          for i, n in enumerate(sorted_n)}

for rep_data in restructured.values():
    for n_value, points in rep_data.items():
        points.sort(key=lambda x: x[0])  # sort by K
        k_vals = [k for k, _ in points]
        tmap_vals = [t for _, t in points]

        ax.plot(
            tmap_vals, k_vals,
            color=colors[n_value],
            alpha=0.6,
            marker="o",
            markersize=4,
            linestyle="-"
        )

ax.set_xlabel(r"$t_{MAP}$")
ax.set_ylabel("K")
ax.set_xscale("log")
ax.invert_yaxis()
ax.grid(True, which="both", ls="--", c="0.7")

plt.tight_layout()
plt.savefig("../figures/figure_1/tmap_vs_k.pdf")
plt.show()

# %% [Figure_1.ipynb cell 9]
# Convert to to long form dataframe
rows = []
for rep, rep_data in restructured.items():
    for N, points in rep_data.items():
        for K, t in points:
            rows.append({
                "replicate": rep,
                "N": N,
                "K": K,
                "t_map": t,
                "log_t_map": np.log(t)
            })

df = pd.DataFrame(rows)
df_nk = df.copy()

# %% [Figure_1.ipynb cell 10]
rho, p_value = spearmanr(df["K"], df["t_map"])
print(f"Spearman correlation between K and t_map: rho={rho:.4f}, p-value={p_value:.4g}")

# %% [Figure_1.ipynb cell 11]

# OLS on parameter K with constant terms for N.
model_K = smf.ols(
    "log_t_map ~ K + C(N)",
    data=df
).fit(cov_type="HC3")  # robust SEs

print(model_K.summary())

# %% [Figure_1.ipynb cell 12]
# Comnpute p-value from z-score
z = -14.412
p = 2 * norm.sf(abs(z))
print(p)

# %% [Figure_1.ipynb cell 13]
# OLS on parameter N with constant terms for K.
model_N = smf.ols(
    "log_t_map ~ N + C(K)",
    data=df
).fit(cov_type="HC3")  # robust SEs

print(model_N.summary())

# %% [Figure_1.ipynb cell 14]
# Compute p-value from z-score
z = 11.197
p = 2 * norm.sf(abs(z))
print(p)

# %% [Figure_1.ipynb cell 16]
# Init list to store results
results = []

# Range of N variables to consider
n_range = list(range(4, 11))

for _,n_param in tqdm(enumerate(n_range)):

    # Construct NK landscape with dummy K
    nk = fl.models.nk.create_nk_binary_landscape(N=n_param, K=0, seed=0)

    # Perform eigenmode decomposition
    eigvals, eigvecs = eigenmode_decomposition(
        nk,
        matrix="laplacian",
    )

    # Total number of Laplacian eigenvectors (== number of nodes)
    n_eig = eigvecs.shape[1]  # should equal len(nk)

    # Only test first 25% of eigenmodes (skip k=0, )
    max_k = int(len(nk) * 0.50)

    for k in range(1, max_k + 1):
        # Normalized eigenvector index in (0, 1]
        k_norm = k / (n_eig - 1)

        # Attach the eigenvector as the "fitness" layer/signal
        layer_name = f"laplacian_eigvec_{k}"
        nk.attach(
            name=layer_name,
            values=eigvecs[:, k],
            dtype="numeric",
        )
        nk.view(layer_name)

        # Compute diffusion-scale ruggedness (optionally set t_min/t_max)
        tmap_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(
            nk,
            t_min=1e-10,
            t_max=1e2,
        )

        # Store everything you need for analysis
        results.append({
            "N": n_param,
            "num_nodes": len(nk),
            "eig_index": k,
            "eig_index_norm": k_norm,
            "eigval": float(eigvals[k]),
            "t_map": float(tmap_res["t_map"]),
            "t_lo": float(tmap_res["t_lower_confidence_interval"]),
            "t_hi": float(tmap_res["t_upper_confidence_interval"]),
            "logpost_map": float(tmap_res["t_logposterior_map"]),
            "var_approx": float(tmap_res["variance_approximate"]),
            "layer": layer_name,
        })

# Convert to DataFrame for convenience
df_eig = pd.DataFrame(results)

# %% [Figure_1.ipynb cell 17]
df = df_eig.copy()

# Collect N values and color-map them
sorted_n = sorted(df["N"].unique())
fig, ax = plt.subplots(figsize=(3, 2.75))
cmap = plt.get_cmap("viridis")
colors = {n: cmap(i / (len(sorted_n) - 1 if len(sorted_n) > 1 else 1))
          for i, n in enumerate(sorted_n)}

# Plot one line per N
for n in sorted_n:
    sub = df[df["N"] == n].sort_values("eig_index_norm")
    ax.plot(
        sub["t_map"].values,
        sub["eig_index_norm"].values,
        color=colors[n],
        alpha=1.0,
        markersize=0,
        linestyle="-",
    )

ax.set_xlabel(r"$t_{MAP}$")
ax.set_ylabel("Normalized Laplacian Index")
ax.set_xscale("log")
ax.invert_yaxis()
ax.grid(True, which="both", ls="--", c="0.7")

plt.tight_layout()
plt.savefig("../figures/figure_1/tmap_vs_norm_laplacian_idx.pdf")
plt.show()

# %% [Figure_1.ipynb cell 22]
# OLS on parameter N with constant terms for K.
df['log_t_map'] = np.log(df['t_map'])
model_j = smf.ols(
    "log_t_map ~ N + eig_index_norm",
    data=df
).fit(cov_type="HC3")  # robust SEs

print(model_j.summary())

# --- End copied code from Figure_1.ipynb cells 0,6-7,9-14,16-17,22 ---

run_postprocess(SCRIPT_DIR / "postprocess.py", globals())
