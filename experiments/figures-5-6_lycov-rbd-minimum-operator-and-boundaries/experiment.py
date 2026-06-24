#!/usr/bin/env python3
"""Publication experiment runner.

Experiment code below is copied from Figure_4_and_SI_LYCoV.ipynb cells 0-3,5,8-10 in
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

# --- Begin copied code from Figure_4_and_SI_LYCoV.ipynb cells 0-3,5,8-10 ---

# %% [Figure_4_and_SI_LYCoV.ipynb cell 0]
import fitness_landscape as fl
import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import networkx as nx
from tqdm import tqdm
from scipy.stats import linregress, zscore
from Bio import SeqIO
import matplotlib.pyplot as plt
import Bio.PDB
import py3Dmol
import re
from scipy.stats import pearsonr, spearmanr

# %% [Figure_4_and_SI_LYCoV.ipynb cell 1]
df = pd.read_csv('../data_files/lycov_combination_antibodies/RBD-LYCOV_SARS2_STARR_2021.csv')

df = df.drop_duplicates(subset="aa_seq", keep="first")
df.dropna(inplace=True, subset=["aa_seq", "fitness_LY-CoV-16", "fitness_LY-CoV-555", "fitness_LY-CoV-16-555"])

sequences = [fl.BaseNumpySequence(sequence) for sequence in df['aa_seq']]

# Construct fitness landscape
landscape = fl.FitnessLandscape.build(
    sequences,
    graph="hamming",
    _compute_hamming_edges=False,
)

# %% [Figure_4_and_SI_LYCoV.ipynb cell 2]
# Phenotype 1 folds, phenotype 2 folds + binds

cov_16_map = dict(zip(df["aa_seq"], df["fitness_LY-CoV-16"]))
landscape.attach(
    name="fitness_LY-CoV-16",
    values=cov_16_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

cov_555_map = dict(zip(df["aa_seq"], df["fitness_LY-CoV-555"]))
landscape.attach(
    name="fitness_LY-CoV-555",
    values=cov_555_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

cov_16_555_map = dict(zip(df["aa_seq"], df["fitness_LY-CoV-16-555"]))
landscape.attach(
    name="fitness_LY-CoV-16-555",
    values=cov_16_555_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

F = zscore(landscape.fitness_layers['fitness_LY-CoV-16'].to_scalar())
B = zscore(landscape.fitness_layers['fitness_LY-CoV-555'].to_scalar())
s = F - B  # <0 fold-limited, >0 bind-limited
landscape.attach(
    name="limiter_index",
    values=s,
    dtype="numeric")

# %% [Figure_4_and_SI_LYCoV.ipynb cell 3]
from pathlib import Path

landscape.export_lsbundle(
    Path('../data_files/lycov_combination_antibodies/lycov_combination_antibodies.lsbundle'),
    backend='portable',
    overwrite=True
)

# %% [Figure_4_and_SI_LYCoV.ipynb cell 5]
# SEARCH_TAG: LYCOV_MIN_OPERATOR_APPROX_TEST
required_layers = [
    "fitness_LY-CoV-16",
    "fitness_LY-CoV-555",
    "fitness_LY-CoV-16-555",
]
missing_layers = [name for name in required_layers if name not in landscape.fitness_layers]
if missing_layers:
    raise RuntimeError(f"Missing fitness layers: {missing_layers}")

SOFTMIN_BETA = 20.0
PRODUCT_EPS = 1e-6

f_16 = np.asarray(landscape.fitness_layers["fitness_LY-CoV-16"].to_scalar(), dtype=float).ravel()
f_555 = np.asarray(landscape.fitness_layers["fitness_LY-CoV-555"].to_scalar(), dtype=float).ravel()
f_combo = np.asarray(landscape.fitness_layers["fitness_LY-CoV-16-555"].to_scalar(), dtype=float).ravel()
stacked = np.stack([f_16, f_555], axis=0)

def _to_unit_interval(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)

def _softmin_weighted_average(stacked: np.ndarray, beta: float = SOFTMIN_BETA):
    shifted = stacked - np.min(stacked, axis=0, keepdims=True)
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights, axis=0, keepdims=True)
    composite_signal = np.sum(weights * stacked, axis=0)
    boundary_idx = np.argmax(weights, axis=0)
    return composite_signal, boundary_idx

def _normalized_product_signal(stacked: np.ndarray, eps: float = PRODUCT_EPS):
    mins = np.min(stacked, axis=1, keepdims=True)
    maxs = np.max(stacked, axis=1, keepdims=True)
    spans = np.where(np.isclose(maxs, mins), 1.0, maxs - mins)
    scaled = (stacked - mins) / spans
    scaled = np.clip(eps + (1.0 - eps) * scaled, eps, 1.0)

    log_product = np.mean(np.log(scaled), axis=0)
    composite_signal = _to_unit_interval(np.exp(log_product))
    boundary_idx = np.argmin(scaled, axis=0)
    return composite_signal, boundary_idx

f_piecewise_min = np.minimum(f_16, f_555)
f_piecewise_min_z = np.minimum(zscore(f_16), zscore(f_555))
softmin_signal, softmin_boundary_idx = _softmin_weighted_average(stacked, beta=SOFTMIN_BETA)
product_signal, product_boundary_idx = _normalized_product_signal(stacked, eps=PRODUCT_EPS)

min_regime = np.where(f_16 <= f_555, "LY-CoV-16 min", "LY-CoV-555 min")
softmin_regime = np.where(softmin_boundary_idx == 0, "LY-CoV-16 soft-min", "LY-CoV-555 soft-min")
product_regime = np.where(product_boundary_idx == 0, "LY-CoV-16 gate", "LY-CoV-555 gate")

min_colors = np.where(f_16 <= f_555, "#1f77b4", "#d62728")
softmin_colors = np.where(softmin_boundary_idx == 0, "#1f77b4", "#d62728")
product_colors = np.where(product_boundary_idx == 0, "#1f77b4", "#d62728")

def summarize_proxy(name, proxy, target):
    proxy = np.asarray(proxy, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()

    pear_r, pear_p = pearsonr(proxy, target)
    spear_rho, spear_p = spearmanr(proxy, target)

    X = np.column_stack([np.ones(len(proxy), dtype=float), proxy])
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    fitted = X @ beta
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    ss_res = float(np.sum((target - fitted) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((target - fitted) ** 2)))

    return {
        "proxy": name,
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_rho": float(spear_rho),
        "spearman_p": float(spear_p),
        "affine_intercept": float(beta[0]),
        "affine_slope": float(beta[1]),
        "affine_r2": float(r2),
        "affine_rmse": rmse,
        "fitted": fitted,
    }

proxy_summaries = [
    summarize_proxy("LY-CoV-16 only", f_16, f_combo),
    summarize_proxy("LY-CoV-555 only", f_555, f_combo),
    summarize_proxy("piecewise min", f_piecewise_min, f_combo),
    summarize_proxy("piecewise min (z-score)", f_piecewise_min_z, f_combo),
    summarize_proxy("soft-min", softmin_signal, f_combo),
    summarize_proxy("multiplicative gate", product_signal, f_combo),
]
summary_by_name = {row["proxy"]: row for row in proxy_summaries}

min_summary_df = pd.DataFrame(
    [
        {k: v for k, v in row.items() if k != "fitted"}
        for row in proxy_summaries
    ]
).sort_values("affine_r2", ascending=False).reset_index(drop=True)

display(min_summary_df)
print("SOFTMIN_BETA:", SOFTMIN_BETA)
print("PRODUCT_EPS:", PRODUCT_EPS)
print("Min-regime counts:", pd.Series(min_regime).value_counts().to_dict())
print("Soft-min active-factor counts:", pd.Series(softmin_regime).value_counts().to_dict())
print("Multiplicative-gate active-factor counts:", pd.Series(product_regime).value_counts().to_dict())

def plot_proxy(ax, proxy, target, colors, row, xlabel):
    proxy = np.asarray(proxy, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    ax.scatter(proxy, target, c=colors, s=9, alpha=0.35, linewidths=0)
    x_line = np.linspace(float(np.min(proxy)), float(np.max(proxy)), 200)
    ax.plot(
        x_line,
        row["affine_intercept"] + row["affine_slope"] * x_line,
        color="black",
        linewidth=1.2,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Observed fitness_LY-CoV-16-555")
    ax.set_title(f"r={row['pearson_r']:.2f}, R^2={row['affine_r2']:.2f}")
    ax.grid(alpha=0.25, linestyle="--")

fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))

plot_proxy(
    axes[0, 0],
    f_piecewise_min,
    f_combo,
    min_colors,
    summary_by_name["piecewise min"],
    "Raw piecewise min proxy",
)
plot_proxy(
    axes[0, 1],
    f_piecewise_min_z,
    f_combo,
    min_colors,
    summary_by_name["piecewise min (z-score)"],
    "Z-scored piecewise min proxy",
)
plot_proxy(
    axes[1, 0],
    softmin_signal,
    f_combo,
    softmin_colors,
    summary_by_name["soft-min"],
    f"Soft-min proxy (beta={SOFTMIN_BETA:g})",
)
plot_proxy(
    axes[1, 1],
    product_signal,
    f_combo,
    product_colors,
    summary_by_name["multiplicative gate"],
    f"Multiplicative gate (eps={PRODUCT_EPS:g})",
)

plt.tight_layout()
plt.show()

# %% [Figure_4_and_SI_LYCoV.ipynb cell 8]
# Compute fitness gradients / energy
f = landscape.fitness_layers["fitness_LY-CoV-16-555"].to_scalar()

node_to_f = dict(zip(landscape.graph.nodes(), f))

edge_energy = []
edge_nodes = []

for u, v in landscape.graph.edges():
    e = (node_to_f[u] - node_to_f[v])**2
    edge_energy.append(e)
    edge_nodes.append((u, v))

edge_energy = np.array(edge_energy)

# Collect limiter index
s = landscape.fitness_layers["limiter_index"].to_scalar()
node_to_s = dict(zip(landscape.graph.nodes(), s))

within_mask = []
between_mask = []

for (u, v) in edge_nodes:
    if node_to_s[u] * node_to_s[v] > 0:
        within_mask.append(True)
        between_mask.append(False)
    elif node_to_s[u] * node_to_s[v] < 0:
        within_mask.append(False)
        between_mask.append(True)
    else:
        within_mask.append(False)
        between_mask.append(False)

within_mask = np.array(within_mask)
between_mask = np.array(between_mask)

# %% [Figure_4_and_SI_LYCoV.ipynb cell 9]
# Use graph node order so `zip(landscape.graph.nodes(), shuffled_s)` stays aligned
graph_nodes = list(landscape.graph.nodes())
idx_by_node = {n: i for i, n in enumerate(landscape._node_order)}

f_all = np.asarray(landscape.fitness_layers["fitness_LY-CoV-16-555"].to_scalar(), dtype=float).ravel()
s_all = np.asarray(landscape.fitness_layers["limiter_index"].to_scalar(), dtype=float).ravel()

# Aligned vectors (same order as graph_nodes)
f = np.array([f_all[idx_by_node[n]] for n in graph_nodes], dtype=float)
s = np.array([s_all[idx_by_node[n]] for n in graph_nodes], dtype=float)

node_to_f = dict(zip(graph_nodes, f))
node_to_s = dict(zip(graph_nodes, s))

edge_nodes = []
edge_energy = []
E_within = []
E_between = []

for u, v in landscape.graph.edges():
    e = (node_to_f[u] - node_to_f[v]) ** 2
    edge_nodes.append((u, v))
    edge_energy.append(e)

    prod = node_to_s[u] * node_to_s[v]
    if prod < 0:
        E_between.append(e)
    elif prod > 0:
        E_within.append(e)
    # prod == 0: excluded

edge_energy = np.asarray(edge_energy, dtype=float)
E_within = np.asarray(E_within, dtype=float)
E_between = np.asarray(E_between, dtype=float)

obs_diff = E_between.mean() - E_within.mean()

print("within edges:", len(E_within), "between edges:", len(E_between))

# 90th Percentile and average energy between boundaries
n_perm = 1000

def q_stat(x, q=0.9):
    return np.quantile(x, q)

obs_stat = q_stat(E_between, 0.9) - q_stat(E_within, 0.9)

null_stats = []

for _ in range(n_perm):
    shuffled_s = np.random.permutation(s)
    node_to_s_shuf = dict(zip(landscape.graph.nodes(), shuffled_s))

    shuf_between = []
    shuf_within = []

    for (u, v), e in zip(edge_nodes, edge_energy):
        if node_to_s_shuf[u] * node_to_s_shuf[v] < 0:
            shuf_between.append(e)
        elif node_to_s_shuf[u] * node_to_s_shuf[v] > 0:
            shuf_within.append(e)

    if shuf_between and shuf_within:
        null_stats.append(
            q_stat(shuf_between, 0.9) - q_stat(shuf_within, 0.9)
        )

null_stats = np.array(null_stats)
p_val = (np.sum(null_stats >= obs_stat) + 1) / (len(null_stats) + 1)

print("Observed 90th-quantile diff:", obs_stat)
print("Permutation p-value:", p_val)
print("Observed diff:", obs_diff)
print("Permutation p-value:", p_val)

# %% [Figure_4_and_SI_LYCoV.ipynb cell 10]
#SEARCH TERM: interface energy share
import numpy as np

required = ["edge_nodes", "edge_energy", "graph_nodes", "node_to_s", "s"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(f"Missing prerequisite variables: {missing}. Run the edge-energy enrichment cell above first.")

def safe_share(num, den):
    return float(num / (den + 1e-12))

# Rebuild masks so every edge stays aligned to the full edge_energy array.
prod_edge_energy = np.array([node_to_s[u] * node_to_s[v] for u, v in edge_nodes], dtype=float)
valid_energy_mask = prod_edge_energy != 0.0
boundary_energy_mask = prod_edge_energy[valid_energy_mask] < 0.0
edge_energy_defined = np.asarray(edge_energy[valid_energy_mask], dtype=float)

n_edges_all = len(edge_energy)
n_edges_defined = int(valid_energy_mask.sum())
n_boundary = int(boundary_energy_mask.sum())

obs_boundary_edge_share = boundary_energy_mask.mean()
obs_boundary_energy_share = safe_share(edge_energy_defined[boundary_energy_mask].sum(), edge_energy_defined.sum())
obs_energy_share_excess = obs_boundary_energy_share - obs_boundary_edge_share
obs_energy_share_enrichment = safe_share(obs_boundary_energy_share, obs_boundary_edge_share)

n_perm_energy_share = 1000
rng = np.random.default_rng(0)

# Control 1: same-size random edge subsets, so edge count is fixed to the observed boundary fraction.
null_energy_share_fixed = np.empty(n_perm_energy_share, dtype=float)
for k in range(n_perm_energy_share):
    pick = np.zeros(n_edges_defined, dtype=bool)
    pick[rng.choice(n_edges_defined, size=n_boundary, replace=False)] = True
    null_energy_share_fixed[k] = safe_share(edge_energy_defined[pick].sum(), edge_energy_defined.sum())

null_energy_excess_fixed = null_energy_share_fixed - obs_boundary_edge_share
null_energy_enrichment_fixed = null_energy_share_fixed / (obs_boundary_edge_share + 1e-12)

p_energy_share_fixed = (np.sum(null_energy_share_fixed >= obs_boundary_energy_share) + 1) / (n_perm_energy_share + 1)
p_excess_fixed = (np.sum(null_energy_excess_fixed >= obs_energy_share_excess) + 1) / (n_perm_energy_share + 1)
p_enrichment_fixed = (np.sum(null_energy_enrichment_fixed >= obs_energy_share_enrichment) + 1) / (n_perm_energy_share + 1)

# Control 2: shuffle limiter labels, letting the interface location and size vary under the null.
null_edge_share_label = np.empty(n_perm_energy_share, dtype=float)
null_energy_share_label = np.empty(n_perm_energy_share, dtype=float)
null_energy_excess_label = np.empty(n_perm_energy_share, dtype=float)
null_energy_enrichment_label = np.empty(n_perm_energy_share, dtype=float)

for k in range(n_perm_energy_share):
    shuffled_s = rng.permutation(s)
    node_to_s_shuf = dict(zip(graph_nodes, shuffled_s))
    prod_shuf = np.array([node_to_s_shuf[u] * node_to_s_shuf[v] for u, v in edge_nodes], dtype=float)
    valid_shuf = prod_shuf != 0.0
    boundary_shuf = prod_shuf[valid_shuf] < 0.0
    energy_shuf = np.asarray(edge_energy[valid_shuf], dtype=float)

    null_edge_share_label[k] = boundary_shuf.mean()
    null_energy_share_label[k] = safe_share(energy_shuf[boundary_shuf].sum(), energy_shuf.sum())
    null_energy_excess_label[k] = null_energy_share_label[k] - null_edge_share_label[k]
    null_energy_enrichment_label[k] = safe_share(null_energy_share_label[k], null_edge_share_label[k])

p_energy_share_label = (np.sum(null_energy_share_label >= obs_boundary_energy_share) + 1) / (n_perm_energy_share + 1)
p_excess_label = (np.sum(null_energy_excess_label >= obs_energy_share_excess) + 1) / (n_perm_energy_share + 1)
p_enrichment_label = (np.sum(null_energy_enrichment_label >= obs_energy_share_enrichment) + 1) / (n_perm_energy_share + 1)

print("[Boundary share of total edge energy]")
print(f"Edges with defined regime label: {n_edges_defined} / {n_edges_all} (excluded={n_edges_all - n_edges_defined})")
print(f"Boundary edge share: {obs_boundary_edge_share:.4f}")
print(f"Boundary energy share: {obs_boundary_energy_share:.4f}")
print(f"Edge-share adjusted excess: {obs_energy_share_excess:.4f}")
print(f"Energy-share enrichment: {obs_energy_share_enrichment:.3f}x")

print("\n[Control: same-size random edge subsets]")
print(f"Null mean energy share: {null_energy_share_fixed.mean():.4f} | perm p={p_energy_share_fixed:.4g}")
print(f"Null mean excess: {null_energy_excess_fixed.mean():.4f} | perm p={p_excess_fixed:.4g}")
print(f"Null mean enrichment: {null_energy_enrichment_fixed.mean():.3f}x | perm p={p_enrichment_fixed:.4g}")

print("\n[Control: shuffled limiter labels]")
print(f"Null mean boundary edge share: {null_edge_share_label.mean():.4f}")
print(f"Null mean energy share: {null_energy_share_label.mean():.4f} | perm p={p_energy_share_label:.4g}")
print(f"Null mean excess: {null_energy_excess_label.mean():.4f} | perm p={p_excess_label:.4g}")
print(f"Null mean enrichment: {null_energy_enrichment_label.mean():.3f}x | perm p={p_enrichment_label:.4g}")

# Boundary enrichment export
from pathlib import Path
import json

system_name = 'LYCoV'
system_slug = 'lycov'
source_notebook = 'Figure_4_and_SI_LYCoV.ipynb'

def resolve_boundary_enrichment_output_dir():
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "figure_notebooks_rev").is_dir():
            return candidate / "figure_notebooks_rev" / "boundary_enrichment_results"
    return cwd / "boundary_enrichment_results"

result_dir = resolve_boundary_enrichment_output_dir()
result_dir.mkdir(parents=True, exist_ok=True)
result_path = result_dir / f"{system_slug}_boundary_enrichment.json"

result_payload = {
    "system_name": system_name,
    "system_slug": system_slug,
    "source_notebook": source_notebook,
    "summary_null_model": "shuffled_limiter_labels",
    "observed": {
        "n_edges_all": int(n_edges_all),
        "n_edges_defined": int(n_edges_defined),
        "n_boundary_edges": int(n_boundary),
        "boundary_edge_share": float(obs_boundary_edge_share),
        "boundary_energy_share": float(obs_boundary_energy_share),
        "energy_share_excess": float(obs_energy_share_excess),
        "energy_share_enrichment": float(obs_energy_share_enrichment),
    },
    "null_same_size_random_edge_subsets": {
        "n_perm": int(n_perm_energy_share),
        "p_energy_share": float(p_energy_share_fixed),
        "p_excess": float(p_excess_fixed),
        "p_enrichment": float(p_enrichment_fixed),
        "energy_share": null_energy_share_fixed.tolist(),
        "energy_excess": null_energy_excess_fixed.tolist(),
        "energy_enrichment": null_energy_enrichment_fixed.tolist(),
    },
    "null_shuffled_limiter_labels": {
        "n_perm": int(n_perm_energy_share),
        "p_energy_share": float(p_energy_share_label),
        "p_excess": float(p_excess_label),
        "p_enrichment": float(p_enrichment_label),
        "boundary_edge_share": null_edge_share_label.tolist(),
        "energy_share": null_energy_share_label.tolist(),
        "energy_excess": null_energy_excess_label.tolist(),
        "energy_enrichment": null_energy_enrichment_label.tolist(),
    },
}

result_path.write_text(json.dumps(result_payload, indent=2) + "\n")
print(f"\nWrote boundary enrichment summary to {result_path}")

# --- End copied code from Figure_4_and_SI_LYCoV.ipynb cells 0-3,5,8-10 ---

run_postprocess(SCRIPT_DIR / "postprocess.py", globals())
