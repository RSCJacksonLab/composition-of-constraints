#!/usr/bin/env python3
"""Publication experiment runner.

Experiment code below is copied from Figure_4_and_SI_SH3_comb_core.ipynb cells 0,2-3,5-8,11,17-19,22,31 in
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

# --- Begin copied code from Figure_4_and_SI_SH3_comb_core.ipynb cells 0,2-3,5-8,11,17-19,22,31 ---

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 0]
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

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 2]
df = pd.read_csv('../data_files/combinatorial_core/predicted_phenotypes_all.csv')
df = df.drop_duplicates(subset="core", keep="first")

sequences = [fl.BaseNumpySequence(sequence) for sequence in df['core']]

# Construct fitness landscape
landscape = fl.FitnessLandscape.build(
    sequences,
    graph="hamming",
    _compute_hamming_edges=False,
)

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 3]
# Phenotype 1 folds, phenotype 2 folds + binds

phenotype_map = dict(zip(df["core"], df["phenotype"]))
landscape.attach(
    name="phenotype_category",
    values=phenotype_map,
    dtype="categorical",
    map_by="sequence",
    on_duplicates="first"
    )


# Binding and folding categories

binding_map = dict(zip(df["core"], df["Binding"]))
landscape.attach(
    name="binding_category",
    values=binding_map,
    dtype="categorical",
    map_by="sequence",
    on_duplicates="first"
)


folding_map = dict(zip(df["core"], df["Abundance"]))
landscape.attach(
    name="folding_category",
    values=folding_map,
    dtype="categorical",
    map_by="sequence",
    on_duplicates="first"
)

# Numeric composite fitness and S.D
fitness_comp_map = dict(zip(df["core"], df["fitness"]))
landscape.attach(
    name="composite_fitness",
    values=fitness_comp_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="aggregate"
)

# Model individual latent fitness values
# construct mappings for fitness layer
trait0_cols = [
    c for c in df.columns
    if re.match(r"fold_\d+_additive_trait0", c)
]

# build dictionary: core sequence -> array of trait0 values
core_to_latent_fold = {
    row["core"]: row[trait0_cols].to_numpy(dtype=float)
    for _, row in df.iterrows()
}

landscape.attach(
    name="latent_fold",
    values=core_to_latent_fold,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="aggregate",
    allow_missing=False,
)

trait1_cols = [
    c for c in df.columns
    if re.match(r"fold_\d+_additive_trait1", c)
]

# build dictionary: core sequence -> array of trait1 values
core_to_latent_bind = {
    row["core"]: row[trait1_cols].to_numpy(dtype=float)
    for _, row in df.iterrows()
}

landscape.attach(
    name="latent_bind",
    values=core_to_latent_bind,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="aggregate",
    allow_missing=False,
)

F = zscore(landscape.fitness_layers['latent_fold'].to_scalar())
B = zscore(landscape.fitness_layers['latent_bind'].to_scalar())
s = F - B  # <0 fold-limited, >0 bind-limited
landscape.attach(
    name="limiter_index",
    values=s,
    dtype="numeric")

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 5]
# Compute fitness gradients / energy
f = landscape.fitness_layers["composite_fitness"].to_scalar()

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

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 6]
# Use graph node order so `zip(landscape.graph.nodes(), shuffled_s)` stays aligned
graph_nodes = list(landscape.graph.nodes())
idx_by_node = {n: i for i, n in enumerate(landscape._node_order)}

f_all = np.asarray(landscape.fitness_layers["composite_fitness"].to_scalar(), dtype=float).ravel()
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

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 7]
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

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 8]

G = landscape.graph

# Choose which fitness signal to compute energy on
fitness_key = "composite_fitness"  # or "latent_fold", "latent_bind"
f = landscape.fitness_layers[fitness_key].to_scalar()

# Node-wise energy
nodes = list(G.nodes())
node_to_f = dict(zip(nodes, f))

node_energy = {}
for u in nodes:
    s = 0.0
    fu = node_to_f[u]
    for v in G.neighbors(u):
        dv = fu - node_to_f[v]
        s += dv * dv
    node_energy[u] = s

E = np.array([node_energy[u] for u in nodes], dtype=float)


#    W_ij = 1/deg(i) if j is neighbor of i else 0
# Centered signal
z = E - E.mean()
n = len(z)

# Precompute neighbor lists and degrees in node order
nbrs = [list(G.neighbors(u)) for u in nodes]
deg = np.array([len(ns) for ns in nbrs], dtype=float)

# Handle isolated nodes (shouldn't happen in hamming graph, but safe)
valid = deg > 0
if not np.all(valid):
    # restrict to nodes with at least one neighbor
    nodes_v = [u for u, ok in zip(nodes, valid) if ok]
    z = z[valid]
    nbrs = [list(G.neighbors(u)) for u in nodes_v]
    deg = deg[valid]
    nodes = nodes_v
    n = len(z)

# For fast indexing neighbors -> indices
node_to_i = {u:i for i,u in enumerate(nodes)}
nbr_idx = [[node_to_i[v] for v in ns if v in node_to_i] for ns in nbrs]

# Moran numerator: sum_i sum_{j in N(i)} w_ij z_i z_j
# with w_ij = 1/deg(i)
num = 0.0
for i in range(n):
    zi = z[i]
    if deg[i] == 0:
        continue
    w = 1.0 / deg[i]
    for j in nbr_idx[i]:
        num += w * zi * z[j]

# Denominator: sum_i z_i^2
den = np.dot(z, z)

# Sum of weights S0 for row-standardized W is n (each row sums to 1)
S0 = float(n)

moran_I = (n / S0) * (num / (den + 1e-12))

# Permutation p-value (one-sided: clustered => I large/positive)

def moran_I_from_z(z):
    n = len(z)
    den = np.dot(z, z)
    num = 0.0
    for i in range(n):
        zi = z[i]
        if deg[i] == 0:
            continue
        w = 1.0 / deg[i]
        for j in nbr_idx[i]:
            num += w * zi * z[j]
    return (n / S0) * (num / (den + 1e-12))

n_perm = 2000  # bump to 10_000 for final publication number
rng = np.random.default_rng(0)

null_I = np.empty(n_perm, dtype=float)
for k in range(n_perm):
    z_shuf = rng.permutation(z)
    null_I[k] = moran_I_from_z(z_shuf)

# "+1" correction gives a conservative estimate and avoids p=0
p_val = (np.sum(null_I >= moran_I) + 1) / (n_perm + 1)

print(f"Fitness layer: {fitness_key}")
print(f"Nodes used: {n}")
print(f"Global Moran's I (local energy): {moran_I:.6f}")
print(f"Permutation p-value (one-sided, clustered): p = {p_val:.6g}")
print(f"Null I mean±sd: {null_I.mean():.6f} ± {null_I.std():.6f}")


#    I_i = z_i * sum_j w_ij z_j   (using same row-standardized W)

local_I = np.zeros(n, dtype=float)
for i in range(n):
    if deg[i] == 0:
        continue
    w = 1.0 / deg[i]
    local_I[i] = z[i] * (w * np.sum(z[nbr_idx[i]]))

# Useful summaries:
print("\nLocal Moran's I summary:")
print("  fraction positive local I:", (local_I > 0).mean())
topk = 20
top_idx = np.argsort(local_I)[-topk:][::-1]
print(f"  top {topk} nodes by local_I (for plotting / highlighting):")
for ii in top_idx[:10]:
    print(f"    node={nodes[ii]}  local_I={local_I[ii]:.6f}  energy={E[ii]:.6f}")

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 11]
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ============================================================
# FAST Moran-on-edges using BFS-induced subgraph (tmap/plot style)
# ============================================================

# ----------------------------
# Config (match your plotting/tmap settings)
# ----------------------------
BFS_FRAC = 0.10
BFS_SEED = 2
KEEP_LCC = True  # plotting cells do this; set False if you want raw BFS set only

fitness_key = "composite_fitness"
limiter_key = "limiter_index"

n_perm_I = 1000          # lower for speed; raise for final
n_perm_boundary = 1000   # lower for speed; raise for final
seed = 0

# ----------------------------
# 1) BFS-induced subgraph (same style as your plotting/tmap code)
# ----------------------------
def bfs_induced_subgraph(landscape_obj, frac=0.1, seed=0, keep_lcc=True):
    G = landscape_obj.graph
    rng = np.random.default_rng(seed)
    target_n = max(int(frac * G.number_of_nodes()), 2)

    G_bfs = G.to_undirected(as_view=True) if G.is_directed() else G
    start = rng.choice(list(G_bfs.nodes()))
    seen = {start}
    frontier = [start]

    while frontier and len(seen) < target_n:
        u = frontier.pop(0)
        for v in G_bfs.neighbors(u):
            if v not in seen:
                seen.add(v)
                frontier.append(v)
            if len(seen) >= target_n:
                break

    # preserve landscape order for layer alignment
    ordered_nodes = [n for n in landscape_obj._node_order if n in seen]
    G_sub = G.subgraph(ordered_nodes).copy()

    if keep_lcc and G_sub.number_of_nodes() > 0:
        if G_sub.is_directed():
            comps = list(nx.weakly_connected_components(G_sub))
        else:
            comps = list(nx.connected_components(G_sub))
        keep = max(comps, key=len)
        G_sub = G_sub.subgraph(keep).copy()
        ordered_nodes = [n for n in landscape_obj._node_order if n in G_sub]

    return G_sub, ordered_nodes

G_sub, node_list_sub = bfs_induced_subgraph(
    landscape, frac=BFS_FRAC, seed=BFS_SEED, keep_lcc=KEEP_LCC
)

print(
    f"BFS subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges "
    f"(frac={BFS_FRAC}, seed={BFS_SEED}, keep_lcc={KEEP_LCC})"
)

# ----------------------------
# 2) Build edge energy + boundary labels on G_sub
# ----------------------------
idx_by_node = {n: i for i, n in enumerate(landscape._node_order)}
f_all = np.asarray(landscape.fitness_layers[fitness_key].to_scalar(), dtype=float).ravel()
s_all = np.asarray(landscape.fitness_layers[limiter_key].to_scalar(), dtype=float).ravel()

f = np.array([f_all[idx_by_node[n]] for n in node_list_sub], dtype=float)
s = np.array([s_all[idx_by_node[n]] for n in node_list_sub], dtype=float)

node_to_f = dict(zip(node_list_sub, f))
node_to_s = dict(zip(node_list_sub, s))

edges = list(G_sub.edges())
edge_energy = np.array([(node_to_f[u] - node_to_f[v]) ** 2 for u, v in edges], dtype=float)

prod = np.array([node_to_s[u] * node_to_s[v] for u, v in edges], dtype=float)
valid = prod != 0.0
edges = [e for e, ok in zip(edges, valid) if ok]
edge_energy = edge_energy[valid]
boundary_mask = prod[valid] < 0
within_mask = prod[valid] > 0

print("edges used:", len(edges), "| boundary:", int(boundary_mask.sum()), "| within:", int(within_mask.sum()))

# ----------------------------
# 3) Edge-neighborhood (line-graph style)
# ----------------------------
incident = {n: [] for n in node_list_sub}
for ei, (u, v) in enumerate(edges):
    incident[u].append(ei)
    incident[v].append(ei)

nbr_idx = []
for ei, (u, v) in enumerate(edges):
    nb = set(incident[u])
    nb.update(incident[v])
    nb.discard(ei)
    nbr_idx.append(np.fromiter(nb, dtype=int))

# ----------------------------
# 4) Moran helpers
# ----------------------------
def moran_I_and_contrib(x, nbr_idx):
    z = x - x.mean()
    lag = np.array([z[nb].mean() if len(nb) else 0.0 for nb in nbr_idx], dtype=float)
    den = np.dot(z, z) + 1e-12
    I = float(np.dot(z, lag) / den)
    contrib = z * lag
    return I, z, lag, contrib

def perm_test_I(x, nbr_idx, n_perm=300, seed=0):
    rng = np.random.default_rng(seed)
    I_obs, z, _, _ = moran_I_and_contrib(x, nbr_idx)
    den = np.dot(z, z) + 1e-12
    null = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        zp = rng.permutation(z)
        lagp = np.array([zp[nb].mean() if len(nb) else 0.0 for nb in nbr_idx], dtype=float)
        null[k] = float(np.dot(zp, lagp) / den)
    p = (np.sum(null >= I_obs) + 1) / (n_perm + 1)
    return I_obs, p, null

def moran_subset(x, nbr_full, mask):
    idx = np.flatnonzero(mask)
    if len(idx) < 3:
        return np.nan, 0

    keep = np.zeros(len(mask), dtype=bool)
    keep[idx] = True
    remap = -np.ones(len(mask), dtype=int)
    remap[idx] = np.arange(len(idx))

    x_sub = x[idx]
    nbr_sub = []
    for i_old in idx:
        nb = [remap[j] for j in nbr_full[i_old] if keep[j]]
        nbr_sub.append(np.array(nb, dtype=int))

    noniso = np.array([len(nb) > 0 for nb in nbr_sub], dtype=bool)
    if noniso.sum() < 3:
        return np.nan, int(noniso.sum())

    remap2 = -np.ones(len(noniso), dtype=int)
    remap2[noniso] = np.arange(noniso.sum())

    nbr2 = []
    for i in np.where(noniso)[0]:
        nb2 = [remap2[j] for j in nbr_sub[i] if noniso[j]]
        nbr2.append(np.array(nb2, dtype=int))

    I_sub, _, _, _ = moran_I_and_contrib(x_sub[noniso], nbr2)
    return I_sub, int(noniso.sum())

# ----------------------------
# 5) Tests
# ----------------------------
I_all, p_all, null_all = perm_test_I(edge_energy, nbr_idx, n_perm=n_perm_I, seed=seed)
I_boundary, n_boundary_noniso = moran_subset(edge_energy, nbr_idx, boundary_mask)
I_within, n_within_noniso = moran_subset(edge_energy, nbr_idx, within_mask)

I_tmp, z, lag, contrib = moran_I_and_contrib(edge_energy, nbr_idx)
obs_delta_contrib = contrib[boundary_mask].mean() - contrib[within_mask].mean()

rng = np.random.default_rng(seed + 1)
null_delta_contrib = np.empty(n_perm_boundary, dtype=float)
n_b = int(boundary_mask.sum())
m = len(boundary_mask)
for k in range(n_perm_boundary):
    pmask = np.zeros(m, dtype=bool)
    pmask[rng.choice(m, size=n_b, replace=False)] = True
    null_delta_contrib[k] = contrib[pmask].mean() - contrib[~pmask].mean()

p_delta_contrib = (np.sum(null_delta_contrib >= obs_delta_contrib) + 1) / (n_perm_boundary + 1)

X = np.column_stack([np.ones(m), boundary_mask.astype(float)])
beta = np.linalg.lstsq(X, edge_energy, rcond=None)[0]
resid = edge_energy - X @ beta
I_resid, p_resid, null_resid = perm_test_I(resid, nbr_idx, n_perm=n_perm_I, seed=seed + 2)

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 17]
import numpy as np
import networkx as nx
import fitness_landscape as fl

# -------------------------------------------------------------------
# Order-preserving induced subgraph (critical for layer/node alignment)
# -------------------------------------------------------------------
def _induced_subgraph_in_order(parent_graph: nx.Graph, ordered_nodes):
    H = parent_graph.__class__()
    H.graph.update(parent_graph.graph)

    for n in ordered_nodes:
        H.add_node(n, **parent_graph.nodes[n])

    SG = parent_graph.subgraph(ordered_nodes)
    if SG.is_multigraph():
        for u, v, k, d in SG.edges(keys=True, data=True):
            H.add_edge(u, v, key=k, **dict(d))
    else:
        for u, v, d in SG.edges(data=True):
            H.add_edge(u, v, **dict(d))
    return H


# -------------------------------------------------------------------
# Subset landscape by BFS (tractable) with guaranteed order consistency
# -------------------------------------------------------------------
def bfs_sub_landscape(landscape: fl.FitnessLandscape, frac=0.1, seed=0):
    G = landscape.graph
    rng = np.random.default_rng(seed)
    target_n = max(int(frac * G.number_of_nodes()), 2)

    G_bfs = G.to_undirected(as_view=True) if G.is_directed() else G
    start = rng.choice(list(G_bfs.nodes()))
    seen = {start}
    frontier = [start]

    while frontier and len(seen) < target_n:
        u = frontier.pop(0)
        for v in G_bfs.neighbors(u):
            if v not in seen:
                seen.add(v)
                frontier.append(v)
            if len(seen) >= target_n:
                break

    # canonical order from parent landscape
    ordered_nodes = [n for n in landscape._node_order if n in seen]
    node_index_map = {n: i for i, n in enumerate(landscape._node_order)}
    indices = [node_index_map[n] for n in ordered_nodes]

    ordered_graph = _induced_subgraph_in_order(G, ordered_nodes)

    sub_sequences = [landscape.sequences[i] for i in indices]
    sub_fitness = landscape._subset_fitness_layers(indices)
    sub_annotations = landscape._subset_annotation_layers(indices)
    sub_embeddings = (
        {d: emb[indices].copy() for d, emb in landscape.embeddings.items()}
        if landscape.embeddings else None
    )

    L_sub = fl.FitnessLandscape(
        sequences=sub_sequences,
        graph=ordered_graph,
        fitness_layers=sub_fitness,
        annotation_layers=sub_annotations,
        embeddings=sub_embeddings,
        emb_arr_key=landscape._emb_arr_key,
        active_embedding_domain=landscape._active_embedding_domain,
        embedding_metadata=landscape.embedding_metadata,
    )

    if landscape._active_view_name is not None:
        L_sub.view(landscape._active_view_name)

    return L_sub


# -------------------------------------------------------------------
# Helper: build sub-landscape from any subgraph, preserving node order
# -------------------------------------------------------------------
def _sub_landscape_from_graph(landscape: fl.FitnessLandscape, sub_graph: nx.Graph):
    if sub_graph.number_of_nodes() == 0:
        return None

    ordered_nodes = [n for n in landscape._node_order if n in sub_graph]
    if not ordered_nodes:
        return None

    node_index_map = {n: i for i, n in enumerate(landscape._node_order)}
    indices = [node_index_map[n] for n in ordered_nodes]

    sub_sequences = [landscape.sequences[i] for i in indices]
    sub_fitness = landscape._subset_fitness_layers(indices)
    sub_annotations = landscape._subset_annotation_layers(indices)
    sub_embeddings = (
        {d: emb[indices].copy() for d, emb in landscape.embeddings.items()}
        if landscape.embeddings else None
    )

    ordered_graph = _induced_subgraph_in_order(sub_graph, ordered_nodes)

    sub = fl.FitnessLandscape(
        sequences=sub_sequences,
        graph=ordered_graph,
        fitness_layers=sub_fitness,
        annotation_layers=sub_annotations,
        embeddings=sub_embeddings,
        emb_arr_key=landscape._emb_arr_key,
        active_embedding_domain=landscape._active_embedding_domain,
        embedding_metadata=landscape.embedding_metadata,
    )
    if landscape._active_view_name is not None:
        sub.view(landscape._active_view_name)
    return sub


# -------------------------------------------------------------------
# Partition into fold / bind / boundary (tmap definition), sorted components
# -------------------------------------------------------------------
def partition_landscape_into_regime_components(
    landscape: fl.FitnessLandscape,
    limiter_layer: str = "limiter_index",
    eps: float = 0.0,
    *,
    return_landscapes: bool = True,
):
    G = landscape.graph
    if G is None:
        raise ValueError("Landscape has no graph.")
    if limiter_layer not in landscape.fitness_layers:
        raise KeyError(f"Missing layer: {limiter_layer!r}")

    s = np.asarray(landscape.fitness_layers[limiter_layer].to_scalar(), dtype=float).ravel()
    if len(s) != len(landscape._node_order):
        raise ValueError("Limiter layer length does not match node order.")

    node_to_s = {n: s[i] for i, n in enumerate(landscape._node_order)}
    nodes = list(landscape._node_order)

    fold_nodes = [n for n in nodes if node_to_s[n] < -eps]
    bind_nodes = [n for n in nodes if node_to_s[n] > +eps]

    boundary_edges = []
    boundary_nodes = set()
    for u, v in G.edges():
        su, sv = node_to_s[u], node_to_s[v]
        if (su < -eps and sv > +eps) or (su > +eps and sv < -eps):
            boundary_edges.append((u, v))
            boundary_nodes.add(u)
            boundary_nodes.add(v)

    G_fold = G.subgraph(fold_nodes)
    G_bind = G.subgraph(bind_nodes)
    G_boundary = G.edge_subgraph(boundary_edges)

    def components(graph):
        if graph.number_of_nodes() == 0:
            return []
        Gu = graph.to_undirected(as_view=True)
        return [list(c) for c in sorted(nx.connected_components(Gu), key=len, reverse=True)]

    fold_components = components(G_fold)
    bind_components = components(G_bind)
    boundary_components = components(G_boundary)

    out = {
        "fold_components": fold_components,
        "bind_components": bind_components,
        "boundary_components": boundary_components,
        "fold_nodes": fold_nodes,
        "bind_nodes": bind_nodes,
        "boundary_nodes": list(boundary_nodes),
        "boundary_edges": boundary_edges,
        "edge_counts": {
            "total": G.number_of_edges(),
            "fold": G_fold.number_of_edges(),
            "bind": G_bind.number_of_edges(),
            "boundary": G_boundary.number_of_edges(),
        },
    }

    if return_landscapes:
        fold_landscapes = [
            _sub_landscape_from_graph(landscape, G_fold.subgraph(c))
            for c in fold_components
        ]
        bind_landscapes = [
            _sub_landscape_from_graph(landscape, G_bind.subgraph(c))
            for c in bind_components
        ]
        boundary_landscapes = [
            _sub_landscape_from_graph(landscape, G_boundary.subgraph(c))
            for c in boundary_components
        ]

        out["fold_landscapes"] = fold_landscapes
        out["bind_landscapes"] = bind_landscapes
        out["boundary_landscapes"] = boundary_landscapes

        out["fold_largest_landscape"] = next((x for x in fold_landscapes if x is not None), None)
        out["bind_largest_landscape"] = next((x for x in bind_landscapes if x is not None), None)
        out["boundary_largest_landscape"] = next((x for x in boundary_landscapes if x is not None), None)

    return out


# -------------------------------------------------------------------
# Example usage for between-regime tmap
# -------------------------------------------------------------------
# L_sub = bfs_sub_landscape(landscape, frac=0.05, seed=2)
# components = partition_landscape_into_regime_components(L_sub, limiter_layer="limiter_index", eps=1.0)
# L_sub_fold = components["fold_largest_landscape"]
# L_sub_bind = components["bind_largest_landscape"]
# L_sub_bound = components["boundary_largest_landscape"]

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 18]
# Sample connected component for tractibility
L_sub = bfs_sub_landscape(landscape, frac=0.1, seed=0)

# Isolate landscape objects
components = partition_landscape_into_regime_components(L_sub)
L_sub_fold = components['fold_largest_landscape']
L_sub_bind = components['bind_largest_landscape']
L_sub_bound = components['boundary_largest_landscape']

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 19]
# Compute global ruggedness
L_sub_fold.view('composite_fitness')
tmap_fold_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(L_sub_fold, t_max=1e2, t_min=1e-20, prior='uniform')
print(tmap_fold_res)

L_sub_bind.view('composite_fitness')
tmap_bind_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(L_sub_bind, t_max=1e2, t_min=1e-20, prior='uniform')
print(tmap_bind_res)

L_sub_bound.view('composite_fitness')
tmap_bound_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(L_sub_bound, t_max=1e2, t_min=1e-20, prior='uniform')
print(tmap_bound_res)

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 22]
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
BFS_FRAC = 0.1
BFS_SEED = 2
EPS_GRID = np.linspace(0.0, 0.6, 10)  # adjust density if needed
VIEW_LAYER = "composite_fitness"

T_MIN = 1e-20
T_MAX = 1e2
PRIOR = "uniform"

GREY = "#d3d3d3"

# -----------------------------
# Build the base sub-landscape once (same as your tmap pipeline)
# -----------------------------
L_sub = bfs_sub_landscape(landscape, frac=BFS_FRAC, seed=BFS_SEED)
L_sub.view(VIEW_LAYER)

def _pick_boundary_landscape(parts):
    if "boundary_largest_landscape" in parts and parts["boundary_largest_landscape"] is not None:
        return parts["boundary_largest_landscape"]

    cands = [
        x for x in parts.get("boundary_landscapes", [])
        if x is not None and x.graph.number_of_nodes() > 0 and x.graph.number_of_edges() > 0
    ]
    if not cands:
        return None
    return max(cands, key=lambda L: L.graph.number_of_nodes())

# -----------------------------
# Sweep eps and compute interface t_map + CI
# -----------------------------
rows = []
for eps in EPS_GRID:
    parts = partition_landscape_into_regime_components(
        L_sub,
        limiter_layer="limiter_index",
        eps=float(eps),
        return_landscapes=True,
    )

    L_bnd = _pick_boundary_landscape(parts)

    row = {
        "eps": float(eps),
        "fold_nodes": len(parts.get("fold_nodes", [])),
        "bind_nodes": len(parts.get("bind_nodes", [])),
        "boundary_nodes_all": len(parts.get("boundary_nodes", [])),
        "boundary_edges_all": parts.get("edge_counts", {}).get("boundary", 0),
        "boundary_comp_nodes": np.nan,
        "boundary_comp_edges": np.nan,
        "t_map": np.nan,
        "t_lo": np.nan,
        "t_hi": np.nan,
    }

    if L_bnd is not None and L_bnd.graph.number_of_nodes() > 1 and L_bnd.graph.number_of_edges() > 0:
        row["boundary_comp_nodes"] = L_bnd.graph.number_of_nodes()
        row["boundary_comp_edges"] = L_bnd.graph.number_of_edges()

        L_bnd.view(VIEW_LAYER)
        res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(
            L_bnd,
            t_max=T_MAX,
            t_min=T_MIN,
            prior=PRIOR,
        )
        row["t_map"] = res.get("t_map", np.nan)
        row["t_lo"] = res.get("t_lower_confidence_interval", np.nan)
        row["t_hi"] = res.get("t_upper_confidence_interval", np.nan)

    rows.append(row)

# -----------------------------
# Arrays for plotting
# -----------------------------
eps_arr = np.array([r["eps"] for r in rows], dtype=float)
t_map = np.array([r["t_map"] for r in rows], dtype=float)
t_lo = np.array([r["t_lo"] for r in rows], dtype=float)
t_hi = np.array([r["t_hi"] for r in rows], dtype=float)

bnd_nodes_all = np.array([r["boundary_nodes_all"] for r in rows], dtype=float)
bnd_edges_all = np.array([r["boundary_edges_all"] for r in rows], dtype=float)
bnd_nodes_comp = np.array([r["boundary_comp_nodes"] for r in rows], dtype=float)

valid = np.isfinite(t_map) & np.isfinite(t_lo) & np.isfinite(t_hi)

# --- sanitize CI so it always brackets t_map (prevents negative widths / odd grid artifacts) ---
t_lo_s = np.minimum(t_lo, t_hi)
t_hi_s = np.maximum(t_lo, t_hi)
t_lo_s = np.minimum(t_lo_s, t_map)
t_hi_s = np.maximum(t_hi_s, t_map)

# -----------------------------
# Plot 1: partition stats vs eps
# -----------------------------
fig1, ax = plt.subplots(figsize=(4, 2))

ax.plot(eps_arr, bnd_nodes_all, marker="o", lw=1.5, ms=4, label="boundary nodes (all)")
ax.plot(eps_arr, bnd_edges_all, marker="o", lw=1.5, ms=4, label="boundary edges (all)")
ax.plot(eps_arr, bnd_nodes_comp, marker="o", lw=1.5, ms=4, label="boundary nodes (largest comp)")

ax.set_xlabel("eps")
ax.set_ylabel("Count")
ax.legend(frameon=False)

ax.grid(True, linestyle="--")

plt.tight_layout()
plt.ylim(0,7500)
plt.savefig('../figures/si_figures/si_figure_eps_vs_comp_size/eps_vs_comp_size.pdf')
plt.show()

# -----------------------------
# Plot 2: interface t_map vs eps with CI shading (grey to match previous)
# -----------------------------
fig2, ax = plt.subplots(figsize=(4, 2))

if valid.any():
    ax.plot(
        eps_arr[valid], t_map[valid],
        marker="o", lw=2, ms=5,
        color="black",
        label=r"$t_{\mathrm{MAP}}$ (interface)",
    )
    ax.fill_between(
        eps_arr[valid],
        t_lo_s[valid], t_hi_s[valid],
        color=GREY, alpha=0.8,
        label="95% CI",
        edgecolor="none",
    )

# Mark eps values where we couldn't compute a valid interface t
if (~valid).any():
    y0 = np.nanmin(t_map[valid]) if valid.any() else 0.0
    ax.scatter(
        eps_arr[~valid],
        np.full((~valid).sum(), y0),
        marker="x",
        color="black",
        label="no valid interface",
    )

ax.set_xlabel("eps")
ax.set_ylabel(r"$t_{\mathrm{MAP}}$")
ax.legend(frameon=False)

ax.grid(True, linestyle="--")

plt.tight_layout()
plt.savefig('../figures/figure_4/epsilon_vs_tmap_interface.pdf')
plt.show()

# %% [Figure_4_and_SI_SH3_comb_core.ipynb cell 31]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fitness_landscape as fl

# -----------------------------
# Latent resampling + regime t_map summary
# Keep the BFS subgraph fixed so variation comes from latent resampling.
# Match the earlier t_map workflow by using the largest component in each regime.
# -----------------------------
N_RESAMPLES = 10
BFS_FRAC = 0.1
BFS_SEED = 2
EPS = 0.0
VIEW_LAYER = "composite_fitness"
T_MIN = 1e-20
T_MAX = 1e2
PRIOR = "uniform"

def _safe_zscore(values):
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values)
    sigma = np.nanstd(values)
    if not np.isfinite(sigma) or sigma == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mu) / sigma

def _largest_component_tmap(component_landscape):
    if component_landscape is None:
        return np.nan
    if component_landscape.graph.number_of_nodes() <= 1 or component_landscape.graph.number_of_edges() == 0:
        return np.nan

    component_landscape.view(VIEW_LAYER)
    res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(
        component_landscape,
        t_max=T_MAX,
        t_min=T_MIN,
        prior=PRIOR,
    )
    return float(res.get("t_map", np.nan))

def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan

resampler_cls = fl.core.fitness.ResampleFitnessModifier
rows = []

for rep in range(N_RESAMPLES):
    L_rep = bfs_sub_landscape(landscape, frac=BFS_FRAC, seed=BFS_SEED)
    L_rep.view(VIEW_LAYER)

    fold_layer = L_rep.apply_fitness_modifier(
        resampler_cls(reps=1, seed=1000 + 2 * rep),
        source_layer="latent_fold",
        output_name="latent_fold_resampled",
        attach=False,
    )
    bind_layer = L_rep.apply_fitness_modifier(
        resampler_cls(reps=1, seed=1001 + 2 * rep),
        source_layer="latent_bind",
        output_name="latent_bind_resampled",
        attach=False,
    )

    L_rep.attach(layer=fold_layer)
    L_rep.attach(layer=bind_layer)

    limiter_resampled = _safe_zscore(fold_layer.to_scalar()) - _safe_zscore(bind_layer.to_scalar())
    L_rep.attach(
        name="limiter_index_resampled",
        values=limiter_resampled,
        dtype="numeric",
    )

    parts = partition_landscape_into_regime_components(
        L_rep,
        limiter_layer="limiter_index_resampled",
        eps=EPS,
        return_landscapes=True,
    )

    rows.append(
        {
            "replicate": rep + 1,
            "binding_limited": _largest_component_tmap(parts.get("bind_largest_landscape")),
            "boundary": _largest_component_tmap(parts.get("boundary_largest_landscape")),
            "folding_limited": _largest_component_tmap(parts.get("fold_largest_landscape")),
            "binding_nodes": len(parts.get("bind_nodes", [])),
            "boundary_nodes": len(parts.get("boundary_nodes", [])),
            "folding_nodes": len(parts.get("fold_nodes", [])),
        }
    )

latent_resample_tmap_df = pd.DataFrame(rows)
display(latent_resample_tmap_df)

plot_cols = ["binding_limited", "boundary", "folding_limited"]
plot_labels = ["Binding-limited", "Boundary", "Folding-limited"]
plot_data = []

for col in plot_cols:
    vals = latent_resample_tmap_df[col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    plot_data.append(vals if vals.size else np.array([np.nan]))

plot_means = [_nanmean_or_nan(latent_resample_tmap_df[col].to_numpy(dtype=float)) for col in plot_cols]

fig, ax = plt.subplots(figsize=(3.2, 2.6))
ax.boxplot(
    plot_data,
    tick_labels=plot_labels,
    patch_artist=True,
    widths=0.6,
    medianprops={"color": "black", "linewidth": 1.2},
    boxprops={"facecolor": "#d3d3d3", "edgecolor": "black", "linewidth": 1.0},
    whiskerprops={"color": "black", "linewidth": 1.0},
    capprops={"color": "black", "linewidth": 1.0},
)
ax.scatter(
    np.arange(1, len(plot_cols) + 1),
    plot_means,
    color="black",
    s=18,
    zorder=3,
    label="mean across resamples",
)
ax.set_ylabel(r"$t_{\mathrm{MAP}}$")
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.legend(frameon=False, loc="best")

plt.tight_layout()
plt.show()

latent_resample_tmap_summary = latent_resample_tmap_df[plot_cols].agg(["mean", "std", "count"]).T
display(latent_resample_tmap_summary)

# --- End copied code from Figure_4_and_SI_SH3_comb_core.ipynb cells 0,2-3,5,7-8,11,17-19,22,31 ---

run_postprocess(SCRIPT_DIR / "postprocess.py", globals())
