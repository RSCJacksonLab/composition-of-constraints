#!/usr/bin/env python3
"""Publication experiment runner.

Experiment code below is copied from Figure_3.ipynb cells 0,2-3,17,20-21,24,33-36 in
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

# --- Begin copied code from Figure_3.ipynb cells 0,2-3,17,20-21,24,33-36 ---

# %% [Figure_3.ipynb cell 0]
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
# import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from scipy.stats import norm
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import matplotlib as mpl
from scipy.stats import mannwhitneyu
import seaborn as sns
import scipy.stats as stats

# %% [Figure_3.ipynb cell 2]
def _pad_var_by_order(var_by_order: dict, max_order: int) -> np.ndarray:
    """Return variance explained array v[0..max_order]. Missing orders -> 0."""
    v = np.zeros(max_order + 1, dtype=float)
    for k, val in var_by_order.items():
        try:
            kk = int(k)
        except Exception:
            continue
        if 0 <= kk <= max_order:
            v[kk] = float(val)
    return v

def run_min_comp_experiment(
    fl,
    N: int,
    K: int,
    m: int,
    seed: int,
    max_order: int = None,
    alpha_align: float = 0.0,
):
    """
    Creates m NK landscapes (N,K), optionally correlated via alpha_align, min-composes them,
    attaches the composite to a base landscape, and returns metrics + variance-by-order vector.

    alpha_align:
      0.0 -> independent constraints
      1.0 -> identical constraints (all equal to shared)
    """
    assert 0 <= K < N
    if max_order is None:
        # For binary sequences, Walsh orders go 0..N
        max_order = N

    rng = np.random.default_rng(seed)

    # Shared component (for alignment control)
    shared = fl.models.nk.create_nk_binary_landscape(N=N, K=K, seed=int(rng.integers(1e9)))
    shared_signal = shared.fitness_layers[f"nk_k={K}"].to_scalar()

    signals = []
    for _ in range(m):
        ind = fl.models.nk.create_nk_binary_landscape(N=N, K=K, seed=int(rng.integers(1e9)))
        ind_signal = ind.fitness_layers[f"nk_k={K}"].to_scalar()

        if alpha_align > 0:
            sig = alpha_align * shared_signal + (1 - alpha_align) * ind_signal
        else:
            sig = ind_signal

        signals.append(sig)

    stacked = np.stack(signals, axis=0)  # shape (m, 2^N)
    min_comp = np.min(stacked, axis=0)
    # Collect indices of min position to find boundaries
    min_idx = np.argmin(stacked, axis=0)

    # Alignment diagnostic: mean pairwise Pearson corr across constituent signals
    # (Compute on raw vectors; if constant, corr can be nan -> handle.)
    corr = np.corrcoef(stacked)
    # upper triangle mean excluding diagonal
    if m > 1:
        ut = corr[np.triu_indices(m, k=1)]
        mean_pairwise_corr = np.nanmean(ut)
    else:
        mean_pairwise_corr = np.nan

    # Attach composite to a base landscape (K=0 is fine; you just need the genotype graph)
    base = fl.models.nk.create_nk_binary_landscape(N=N, K=0, seed=int(rng.integers(1e9)))

    # Attach categorical boundary
    layer_name = "min_boundary"
    base.attach(name=layer_name, values=min_idx, dtype="categorical")

    # Attached numeric composite
    layer_name = "composite_min"
    base.attach(name=layer_name, values=min_comp, dtype="numeric")
    base.view(layer_name)


    res = fl.analysis.epistasis.calculate_epistasis_walsh(base, order=max_order)
    var_by_order = res.get("variance_explained", {})
    v = _pad_var_by_order(var_by_order, max_order=max_order)

    high_order_frac = float(v[2:].sum()) if len(v) > 2 else 0.0
    # Ignore order 0 in centroid (optional). Here I use orders 1..max_order.
    denom = float(v[1:].sum()) if v[1:].sum() > 0 else 1.0
    centroid = float((np.arange(1, max_order + 1) * v[1:]).sum() / denom)

    # "effective max order" = highest order above tiny epsilon
    eps = 1e-6
    nz = np.where(v > eps)[0]
    eff_max_order = int(nz.max()) if nz.size else 0
    tmap_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(base, t_min=1e-10, t_max=1e2)
    tmap = float(tmap_res.get("t_map", np.nan))
    tmap_upper_ci = float(tmap_res.get("t_upper_confidence_interval", np.nan))
    tmap_lower_ci = float(tmap_res.get("t_lower_confidence_interval", np.nan))

    out = {
        "N": N,
        "K": K,
        "m": m,
        "seed": seed,
        "alpha_align": alpha_align,
        "mean_pairwise_corr": mean_pairwise_corr,
        "high_order_frac_ge2": high_order_frac,
        "epistasis_centroid": centroid,
        "eff_max_order": eff_max_order,
        "tmap": tmap,
        "tmap_upper_ci": tmap_upper_ci,
        "tmap_lower_ci": tmap_lower_ci,
        # Store NK landscape in dict
        "nk_landscape": base,
    }

    # store variance-by-order as separate columns for easy plotting/aggregation
    for o in range(max_order + 1):
        out[f"v{o}"] = float(v[o])

    return out

def sweep_all(
    fl,
    Ns=(5,6,7,8),
    Ks=(0,1,2,3),
    ms=(1,2,4,8,16),
    alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
    seeds=range(25),
):
    rows = []
    for _,N in tqdm(enumerate(Ns)):
        for K in Ks:
            if K >= N:
                continue
            for m in ms:
                for a in alphas:
                    for seed in seeds:
                        rows.append(run_min_comp_experiment(
                            fl=fl,
                            N=N, K=K, m=m,
                            seed=seed,
                            max_order=N,          # Walsh orders 0..N
                            alpha_align=a,
                        ))
    return rows

# %% [Figure_3.ipynb cell 3]
nk_results = sweep_all(fl, Ns=[4,5,6], Ks=[0], ms=[2,4,8,10,20,50,100], alphas=[0.0,0.25,0.5,0.75, 0.9,1.0], seeds=range(10))
df = pd.DataFrame(nk_results)

# %% [Figure_3.ipynb cell 17]
# Edgewise energy

def boundary_edge_dirichlet_energy(landscape, *,
                                   boundary_layer="min_boundary",
                                   signal_layer="composite_min",
                                   weighted=False,
                                   aggregate_func=np.mean):
    G = landscape.graph

    boundary = {n: G.nodes[n][f"fitness_{boundary_layer}"] for n in G.nodes()}

    def _to_scalar(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return float(aggregate_func(x))
        return float(x)

    signal = {n: _to_scalar(G.nodes[n][f"fitness_{signal_layer}"]) for n in G.nodes()}

    boundary_total = 0.0
    internal_total = 0.0
    boundary_count = 0
    internal_count = 0
    total = 0.0

    for u, v, data in G.edges(data=True):

        w = data.get("weight", 1.0)
        de = 0.5 * (signal[u] - signal[v])**2
        if weighted:
            de *= w
        # Total edge-wise Dirichlet energy
        total += de

        # Cross boundary
        if boundary[u] != boundary[v]:
            boundary_total += de
            boundary_count += 1

        else:
            internal_total += de
            internal_count += 1

    return boundary_total / boundary_count if boundary_count > 0 else 0, internal_total / internal_count if internal_count > 0 else 0, total


def get_high_pass_signal_by_index(landscape, signal_layer="composite_min", N=None, A=2):
    """
    Filters out additive components by zeroing out the first N*(A-1)
    eigenvectors (the 'additive band').
    """
    # Perform Graph Fourier Transform
    eigvecs, eigvals, coeffs = fl.transforms.graph_fourier_transform(landscape)

    #  Calculate the exact additive cutoff
    # N = sequence length, A = alphabet size (e.g., 2 for binary)
    if N is None:
        # Fallback: estimate N if not provided, assuming binary Hamming graph
        N = int(np.log2(len(eigvals)))

    # Index 0 is the constant (mean) mode. Indices 1 to M are additive.
    # We want to keep everything AFTER index M.
    additive_cutoff = N * (A - 1)

    # Create the filter kernel
    # Add 1 to include the index 0 (DC component) in the mask
    filter_kernel = np.zeros_like(eigvals)
    filter_kernel[additive_cutoff + 1:] = 1.0

    # 4. Reconstruct filtered signal
    f_high = eigvecs @ (coeffs * filter_kernel)
    return f_high


def get_switching_edges(landscape, boundary_layer="min_boundary"):
    """
    Returns a list of edge tuples where the identity of the
    limiting constraint changes.
    """
    G = landscape.graph
    switching_edges = []

    # Map node to its limiting constraint ID
    boundary_map = {n: G.nodes[n][f"fitness_{boundary_layer}"] for n in G.nodes()}

    for u, v in G.edges():
        if boundary_map[u] != boundary_map[v]:
            switching_edges.append((u, v))

    return switching_edges

# %% [Figure_3.ipynb cell 20]

def get_high_pass_signal(landscape, N, A=2):
    """
    Filters out the additive components (lowest N*(A-1) modes) to isolate
    the ruggedness residuals created by switching boundaries.
    """
    # Assuming fl is your fitness landscape library
    eigvecs, eigvals, coeffs = fl.transforms.graph_fourier_transform(landscape)

    # Calculate index-based cutoff for additive components
    # Index 0 is constant, 1 to N*(A-1) are additive
    additive_cutoff = N * (A - 1)

    filter_kernel = np.zeros_like(eigvals)
    filter_kernel[additive_cutoff + 1:] = 1.0

    # Reconstruct the signal
    f_high = eigvecs @ (coeffs * filter_kernel)
    return f_high

def collect_edge_energies(landscape, f_high, boundary_layer="min_boundary"):
    """
    Computes individual Dirichlet energies for every edge and labels them
    as 'Switching' or 'Internal'.
    """
    G = landscape.graph
    boundary = {n: G.nodes[n][f"fitness_{boundary_layer}"] for n in G.nodes()}

    # Create a mapping of signal values for quick lookup
    # Mapping f_high array back to node order in the graph
    node_to_val = {node: val for node, val in zip(G.nodes(), f_high)}

    switching_energies = []
    internal_energies = []

    for u, v in G.edges():
        # Edge Dirichlet Energy: 0.5 * (f_u - f_v)^2
        de = 0.5 * (node_to_val[u] - node_to_val[v])**2

        if boundary[u] != boundary[v]:
            switching_energies.append(de)
        else:
            internal_energies.append(de)

    return switching_energies, internal_energies

# Main loop

all_switching = []
all_internal = []

for idx, row in df.iterrows():
    nk = row['nk_landscape']
    N = row['N']

    f_high = get_high_pass_signal(nk, N=N, A=2)

    s_eng, i_eng = collect_edge_energies(nk, f_high)

    all_switching.extend(s_eng)
    all_internal.extend(i_eng)


data = [np.array(all_switching), np.array(all_internal)]
labels = ["Switching", "Internal"]

plt.figure(figsize=(1.65, 2.75))

bp = plt.boxplot(
    data,
    tick_labels=labels,
    showfliers=False,
    widths=0.6,
    patch_artist=True,
    boxprops=dict(facecolor="lightgray", alpha=1),
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="black", alpha=1),
    capprops=dict(color="black", alpha=1),
)

plt.yscale("log")
# plt.ylabel(r"Edgewise Dirichlet Energy (High-Pass)")
# plt.xlabel("Edge type")

# Optional: tidy x labels for narrow panel
plt.xticks(rotation=90)

# # Put stats text in a consistent spot (axes coords; works well with log scale)
# plt.text(
#     0.5, 0.95,
#     rf"$p = {p_value:.1e}$" + "\n" + rf"$\mu_S/\mu_I = {mean_ratio:.2f}$",
#     transform=plt.gca().transAxes,
#     ha="center", va="top",
#     fontsize=8,
# )

plt.tight_layout()
plt.savefig("../figures/figure_3/edge_energy_switching_vs_internal_boxplot.pdf")
plt.show()

# %% [Figure_3.ipynb cell 21]
def _safe_divide(numerator, denominator):
    return np.nan if denominator == 0 else numerator / denominator


def _to_scalar(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(np.mean(value))
    return float(value)


pooled = {
    "total_edges": 0,
    "boundary_edges": 0,
    "full_total_energy": 0.0,
    "full_boundary_energy": 0.0,
    "high_total_energy": 0.0,
    "high_boundary_energy": 0.0,
}

for _, row in df.iterrows():
    nk = row["nk_landscape"]
    G = nk.graph
    node_order = list(G.nodes())
    boundary = {n: G.nodes[n]["fitness_min_boundary"] for n in node_order}
    full_signal = {n: _to_scalar(G.nodes[n]["fitness_composite_min"]) for n in node_order}
    nk.view("composite_min")
    f_high = get_high_pass_signal(nk, N=int(row["N"]), A=2)
    high_signal = {n: float(val) for n, val in zip(node_order, f_high)}

    pooled["total_edges"] += G.number_of_edges()

    for u, v in G.edges():
        is_boundary = boundary[u] != boundary[v]
        if is_boundary:
            pooled["boundary_edges"] += 1

        de_full = 0.5 * (full_signal[u] - full_signal[v])**2
        pooled["full_total_energy"] += de_full
        if is_boundary:
            pooled["full_boundary_energy"] += de_full

        de_high = 0.5 * (high_signal[u] - high_signal[v])**2
        pooled["high_total_energy"] += de_high
        if is_boundary:
            pooled["high_boundary_energy"] += de_high

boundary_edge_share = _safe_divide(pooled["boundary_edges"], pooled["total_edges"])

boundary_energy_share_summary = pd.DataFrame(
    [
        {
            "signal": "Full",
            "boundary_energy_share": _safe_divide(
                pooled["full_boundary_energy"], pooled["full_total_energy"]
            ),
            "boundary_edge_share": boundary_edge_share,
        },
        {
            "signal": "High-pass",
            "boundary_energy_share": _safe_divide(
                pooled["high_boundary_energy"], pooled["high_total_energy"]
            ),
            "boundary_edge_share": boundary_edge_share,
        },
    ]
)
boundary_energy_share_summary["relative_to_edge_share"] = (
    boundary_energy_share_summary["boundary_energy_share"]
    / boundary_energy_share_summary["boundary_edge_share"]
)

display(
    boundary_energy_share_summary.style.format(
        {
            "boundary_energy_share": "{:.2%}",
            "boundary_edge_share": "{:.2%}",
            "relative_to_edge_share": "{:.2f}x",
        }
    )
)

print(
    f"Boundary edges account for {boundary_edge_share:.2%} of all edges "
    f"({pooled['boundary_edges']} / {pooled['total_edges']})."
)
for _, row in boundary_energy_share_summary.iterrows():
    print(
        f"{row['signal']}: boundary edges contain {row['boundary_energy_share']:.2%} of total edge energy "
        f"({row['relative_to_edge_share']:.2f}x their edge share)."
    )

# %% [Figure_3.ipynb cell 24]
n_params = range(4,11)
seeds = range(10)
rows = []

for _, seed in tqdm(enumerate(seeds)):
    for n_value in n_params:

        # Iterate through K parameters
        for k_value in range(0, n_value):

            nk = fl.models.create_nk_binary_landscape(N=n_value, K=k_value, seed=seed)

            layer = nk.view(f"nk_k={k_value}")  # active by default, but explicit is safe
            fitness = layer.to_scalar()

            # Parititon f_min as the mean fitness
            mean_fitness = float(fitness.mean())
            solution_mask = fitness > mean_fitness

            # Attach categorical layer for the solution set
            labels = np.where(solution_mask, "solution", "non_solution").tolist()
            nk.attach(
                name="solution_set",
                values=labels,
                dtype="categorical",
                categories=["non_solution", "solution"],
            )

            # Conductance of the solution set
            G = nk.graph
            node_order = list(G.nodes())
            solution_nodes = [node_order[i] for i, is_sol in enumerate(solution_mask) if is_sol]
            phi = nx.algorithms.cuts.conductance(G, solution_nodes, weight="weight")

            # tmap on landscape
            layer = nk.view(f"nk_k={k_value}")
            tmap_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(nk, t_min=1e-10, t_max=1e2)
            tmap = float(tmap_res.get("t_map", np.nan))
            tmap_upper_ci = float(tmap_res.get("t_upper_confidence_interval", np.nan))
            tmap_lower_ci = float(tmap_res.get("t_lower_confidence_interval", np.nan))

            res = {
                "seed": seed,
                "N" : n_value,
                "K" : k_value,
                "conductance" : phi,
                "tmap" : tmap,
                "tmap_lower_ci" : tmap_lower_ci,
                "tmap_upper_ci" : tmap_upper_ci,
            }

            rows.append(res)

df = pd.DataFrame(rows)

# %% [Figure_3.ipynb cell 33]
def _build_aligned_nk_stack(
    fl,
    N: int,
    K: int,
    m: int,
    seed: int,
    alpha_align: float = 0.0,
):
    rng = np.random.default_rng(seed)

    shared = fl.models.nk.create_nk_binary_landscape(
        N=N, K=K, seed=int(rng.integers(1e9))
    )
    shared_signal = shared.fitness_layers[f"nk_k={K}"].to_scalar()

    signals = []
    for _ in range(m):
        ind = fl.models.nk.create_nk_binary_landscape(
            N=N, K=K, seed=int(rng.integers(1e9))
        )
        ind_signal = ind.fitness_layers[f"nk_k={K}"].to_scalar()

        if alpha_align > 0:
            sig = alpha_align * shared_signal + (1 - alpha_align) * ind_signal
        else:
            sig = ind_signal

        signals.append(sig)

    return np.stack(signals, axis=0), rng


def _mean_pairwise_corr(stacked: np.ndarray) -> float:
    corr = np.corrcoef(stacked)
    if stacked.shape[0] > 1:
        ut = corr[np.triu_indices(stacked.shape[0], k=1)]
        return float(np.nanmean(ut))
    return float("nan")


def _to_unit_interval(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _finalize_coupled_landscape(
    fl,
    *,
    N: int,
    K: int,
    m: int,
    seed: int,
    alpha_align: float,
    rng,
    max_order: int,
    stacked: np.ndarray,
    composite_signal: np.ndarray,
    boundary_idx: np.ndarray,
    signal_layer: str,
    boundary_layer: str,
    coupling_meta=None,
):
    mean_pairwise_corr = _mean_pairwise_corr(stacked)

    base = fl.models.nk.create_nk_binary_landscape(
        N=N, K=0, seed=int(rng.integers(1e9))
    )
    base.attach(name=boundary_layer, values=boundary_idx, dtype="categorical")
    base.attach(
        name=signal_layer,
        values=np.asarray(composite_signal, dtype=float),
        dtype="numeric",
    )
    base.view(signal_layer)

    res = fl.analysis.epistasis.calculate_epistasis_walsh(base, order=max_order)
    var_by_order = res.get("variance_explained", {})
    v = _pad_var_by_order(var_by_order, max_order=max_order)

    high_order_frac = float(v[2:].sum()) if len(v) > 2 else 0.0
    denom = float(v[1:].sum()) if v[1:].sum() > 0 else 1.0
    centroid = float((np.arange(1, max_order + 1) * v[1:]).sum() / denom)

    eps = 1e-6
    nz = np.where(v > eps)[0]
    eff_max_order = int(nz.max()) if nz.size else 0

    tmap_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(
        base, t_min=1e-10, t_max=1e2
    )
    tmap = float(tmap_res.get("t_map", np.nan))
    tmap_upper_ci = float(tmap_res.get("t_upper_confidence_interval", np.nan))
    tmap_lower_ci = float(tmap_res.get("t_lower_confidence_interval", np.nan))

    out = {
        "N": N,
        "K": K,
        "m": m,
        "seed": seed,
        "alpha_align": alpha_align,
        "mean_pairwise_corr": mean_pairwise_corr,
        "high_order_frac_ge2": high_order_frac,
        "epistasis_centroid": centroid,
        "eff_max_order": eff_max_order,
        "tmap": tmap,
        "tmap_upper_ci": tmap_upper_ci,
        "tmap_lower_ci": tmap_lower_ci,
        "signal_layer": signal_layer,
        "boundary_layer": boundary_layer,
        "nk_landscape": base,
    }

    if coupling_meta is not None:
        out.update(coupling_meta)

    for o in range(max_order + 1):
        out[f"v{o}"] = float(v[o])

    return out


def sweep_coupled_experiments(
    run_experiment,
    fl,
    Ns=(5, 6, 7, 8),
    Ks=(0, 1, 2, 3),
    ms=(1, 2, 4, 8, 16),
    alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
    seeds=range(25),
    **kwargs,
):
    rows = []
    for _, N in tqdm(enumerate(Ns)):
        for K in Ks:
            if K >= N:
                continue
            for m in ms:
                for a in alphas:
                    for seed in seeds:
                        rows.append(
                            run_experiment(
                                fl=fl,
                                N=N,
                                K=K,
                                m=m,
                                seed=seed,
                                max_order=N,
                                alpha_align=a,
                                **kwargs,
                            )
                        )
    return rows


SOFTMIN_BETA = 20.0


def _softmin_weighted_average(stacked: np.ndarray, beta: float = SOFTMIN_BETA):
    shifted = stacked - np.min(stacked, axis=0, keepdims=True)
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights, axis=0, keepdims=True)
    composite_signal = np.sum(weights * stacked, axis=0)
    boundary_idx = np.argmax(weights, axis=0)
    return composite_signal, boundary_idx


def run_softmin_comp_experiment(
    fl,
    N: int,
    K: int,
    m: int,
    seed: int,
    max_order: int = None,
    alpha_align: float = 0.0,
    softmin_beta: float = SOFTMIN_BETA,
):
    assert 0 <= K < N
    if max_order is None:
        max_order = N

    stacked, rng = _build_aligned_nk_stack(
        fl=fl,
        N=N,
        K=K,
        m=m,
        seed=seed,
        alpha_align=alpha_align,
    )
    composite_signal, boundary_idx = _softmin_weighted_average(
        stacked, beta=softmin_beta
    )

    return _finalize_coupled_landscape(
        fl,
        N=N,
        K=K,
        m=m,
        seed=seed,
        alpha_align=alpha_align,
        rng=rng,
        max_order=max_order,
        stacked=stacked,
        composite_signal=composite_signal,
        boundary_idx=boundary_idx,
        signal_layer="composite_softmin",
        boundary_layer="softmin_boundary",
        coupling_meta={"softmin_beta": softmin_beta},
    )


softmin_results = sweep_coupled_experiments(
    run_softmin_comp_experiment,
    fl,
    Ns=[4, 5, 6],
    Ks=[0],
    ms=[2, 4, 8, 10, 20, 50, 100],
    alphas=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
    seeds=range(10),
    softmin_beta=SOFTMIN_BETA,
)
softmin_df = pd.DataFrame(softmin_results)

# %% [Figure_3.ipynb cell 34]
def plot_epistasis_spectrum_vs_alpha(results_df: pd.DataFrame, *, figure_path=None):
    vcols = sorted(
        [c for c in results_df.columns if c.startswith("v") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    vcols = [c for c in vcols if int(c[1:]) != 0]
    bar_vcols = [c for c in vcols if 1 <= int(c[1:]) <= 6]

    if len(bar_vcols) != 6:
        raise ValueError(
            "Expected variance columns for epistasis orders 1 through 6."
        )

    plot_df = results_df[bar_vcols + ["alpha_align"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    stacked = (
        plot_df.groupby("alpha_align", as_index=True)[bar_vcols]
        .mean()
        .sort_index()
    )

    x = np.arange(len(stacked), dtype=float)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(bar_vcols)))
    bottom = np.zeros(len(stacked), dtype=float)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    for color, col in zip(colors, bar_vcols):
        values = stacked[col].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.75,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=f"{int(col[1:])}",
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{alpha:g}" for alpha in stacked.index.to_numpy(dtype=float)]
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Variance explained")
    ax.set_ylim(0, max(1.0, bottom.max() * 1.02))
    plt.xticks(rotation=45)

    fig.tight_layout()
    if figure_path is not None:
        fig.savefig(figure_path)
    plt.show()


def collect_switching_vs_internal_energies(
    results_df: pd.DataFrame,
    *,
    signal_layer: str,
    boundary_layer: str,
):
    all_switching = []
    all_internal = []

    for _, row in results_df.iterrows():
        nk = row["nk_landscape"]
        nk.view(signal_layer)
        f_high = get_high_pass_signal(nk, N=int(row["N"]), A=2)
        s_eng, i_eng = collect_edge_energies(
            nk,
            f_high,
            boundary_layer=boundary_layer,
        )
        all_switching.extend(s_eng)
        all_internal.extend(i_eng)

    switching = np.clip(np.asarray(all_switching, dtype=float), 1e-12, None)
    internal = np.clip(np.asarray(all_internal, dtype=float), 1e-12, None)
    return switching, internal


def plot_switching_vs_internal_boxplot(
    results_df: pd.DataFrame,
    *,
    signal_layer: str,
    boundary_layer: str,
    figure_path=None,
):
    switching, internal = collect_switching_vs_internal_energies(
        results_df,
        signal_layer=signal_layer,
        boundary_layer=boundary_layer,
    )

    plt.figure(figsize=(1.65, 2.75))
    plt.boxplot(
        [switching, internal],
        tick_labels=["Switching", "Internal"],
        showfliers=False,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray", alpha=1),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", alpha=1),
        capprops=dict(color="black", alpha=1),
    )
    plt.yscale("log")
    plt.xticks(rotation=90)
    plt.tight_layout()
    if figure_path is not None:
        plt.savefig(figure_path)
    plt.show()


plot_epistasis_spectrum_vs_alpha(
    softmin_df,
    figure_path="../figures/figure_3/var_by_order_stacked_alpha_softmin.pdf",
)
plot_switching_vs_internal_boxplot(
    softmin_df,
    signal_layer="composite_softmin",
    boundary_layer="softmin_boundary",
    figure_path="../figures/figure_3/edge_energy_switching_vs_internal_boxplot_softmin.pdf",
)

# %% [Figure_3.ipynb cell 35]
PRODUCT_EPS = 1e-6


def _normalized_product_signal(stacked: np.ndarray, eps: float = PRODUCT_EPS):
    mins = np.min(stacked, axis=1, keepdims=True)
    maxs = np.max(stacked, axis=1, keepdims=True)
    spans = np.where(np.isclose(maxs, mins), 1.0, maxs - mins)
    scaled = (stacked - mins) / spans
    scaled = np.clip(eps + (1.0 - eps) * scaled, eps, 1.0)

    # Accumulate the product in log-space and normalize by m to keep the
    # coupled signal numerically stable across different numbers of factors.
    log_product = np.mean(np.log(scaled), axis=0)
    composite_signal = _to_unit_interval(np.exp(log_product))

    # Use the locally smallest factor as the active contributor for the
    # switching-boundary analysis.
    boundary_idx = np.argmin(scaled, axis=0)
    return composite_signal, boundary_idx


def run_product_comp_experiment(
    fl,
    N: int,
    K: int,
    m: int,
    seed: int,
    max_order: int = None,
    alpha_align: float = 0.0,
    product_eps: float = PRODUCT_EPS,
):
    assert 0 <= K < N
    if max_order is None:
        max_order = N

    stacked, rng = _build_aligned_nk_stack(
        fl=fl,
        N=N,
        K=K,
        m=m,
        seed=seed,
        alpha_align=alpha_align,
    )
    composite_signal, boundary_idx = _normalized_product_signal(
        stacked, eps=product_eps
    )

    return _finalize_coupled_landscape(
        fl,
        N=N,
        K=K,
        m=m,
        seed=seed,
        alpha_align=alpha_align,
        rng=rng,
        max_order=max_order,
        stacked=stacked,
        composite_signal=composite_signal,
        boundary_idx=boundary_idx,
        signal_layer="composite_product",
        boundary_layer="product_boundary",
        coupling_meta={"product_eps": product_eps},
    )


product_results = sweep_coupled_experiments(
    run_product_comp_experiment,
    fl,
    Ns=[4, 5, 6],
    Ks=[0],
    ms=[2, 4, 8, 10, 20, 50, 100],
    alphas=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
    seeds=range(10),
    product_eps=PRODUCT_EPS,
)
product_df = pd.DataFrame(product_results)

# %% [Figure_3.ipynb cell 36]
plot_epistasis_spectrum_vs_alpha(
    product_df,
    figure_path="../figures/figure_3/var_by_order_stacked_alpha_product.pdf",
)
plot_switching_vs_internal_boxplot(
    product_df,
    signal_layer="composite_product",
    boundary_layer="product_boundary",
    figure_path="../figures/figure_3/edge_energy_switching_vs_internal_boxplot_product.pdf",
)

# --- End copied code from Figure_3.ipynb cells 0,2-3,17,20-21,24,33-36 ---

run_postprocess(SCRIPT_DIR / "postprocess.py", globals())
