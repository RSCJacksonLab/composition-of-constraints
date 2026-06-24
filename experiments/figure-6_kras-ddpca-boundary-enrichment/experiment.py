#!/usr/bin/env python3
"""Publication experiment runner.

Experiment code below is copied from Figure_4_and_SI_KRAS_DARPin_K27.ipynb cells 0-2,4-6; Figure_4_and_SI_KRAS_DARPin_K55.ipynb cells 0-2,4-6; Figure_4_and_SI_KRAS_PIK3CG.ipynb cells 0-2,4-6; Figure_4_and_SI_KRAS_RAF1.ipynb cells 0-2,4-6; Figure_4_and_SI_KRAS_RALGDS.ipynb cells 0-2,4-6; Figure_4_and_SI_KRAS_SOS1.ipynb cells 0-2,4-6 in
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

# --- Begin copied code from Figure_4_and_SI_KRAS_DARPin_K27.ipynb cells 0-2,4-6 ---

# %% [Figure_4_and_SI_KRAS_DARPin_K27.ipynb cell 0]
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

# %% [Figure_4_and_SI_KRAS_DARPin_K27.ipynb cell 1]
df = pd.read_csv('../data_files/kras_genetic_arch/RESULTS/DARPin_K27_binding_fitness_decomposition_per_variant.tsv', sep='\t')

df = df.drop_duplicates(subset="aa_seq", keep="first")
df.dropna(inplace=True, subset=["aa_seq", "observed_binding_fitness", "latent_ddGf_kcal_mol", "latent_ddGb_kcal_mol"])

sequences = [fl.BaseNumpySequence(sequence) for sequence in df['aa_seq']]

# Construct fitness landscape
landscape = fl.FitnessLandscape.build(
    sequences,
    graph="hamming",
    _compute_hamming_edges=False,
)

# %% [Figure_4_and_SI_KRAS_DARPin_K27.ipynb cell 2]
# Phenotype 1 folds, phenotype 2 folds + binds

folding_map = dict(zip(df["aa_seq"], df["latent_ddGf_kcal_mol"]))
landscape.attach(
    name="latent_folding",
    values=folding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

binding_map = dict(zip(df["aa_seq"], df["latent_ddGb_kcal_mol"]))
landscape.attach(
    name="latent_binding",
    values=binding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

observed_fitness_map = dict(zip(df["aa_seq"], df["observed_binding_fitness"]))
landscape.attach(
    name="fitness_observed_binding",
    values=observed_fitness_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

F = zscore(landscape.fitness_layers['latent_folding'].to_scalar())
B = zscore(landscape.fitness_layers['latent_binding'].to_scalar())
s = F - B  # <0 fold-limited, >0 bind-limited
landscape.attach(
    name="limiter_index",
    values=s,
    dtype="numeric")

# %% [Figure_4_and_SI_KRAS_DARPin_K27.ipynb cell 4]
# Compute fitness gradients / energy
f = landscape.fitness_layers["fitness_observed_binding"].to_scalar()

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

# %% [Figure_4_and_SI_KRAS_DARPin_K27.ipynb cell 5]
# Use graph node order so `zip(landscape.graph.nodes(), shuffled_s)` stays aligned
graph_nodes = list(landscape.graph.nodes())
idx_by_node = {n: i for i, n in enumerate(landscape._node_order)}

f_all = np.asarray(landscape.fitness_layers["fitness_observed_binding"].to_scalar(), dtype=float).ravel()
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

# %% [Figure_4_and_SI_KRAS_DARPin_K27.ipynb cell 6]
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

system_name = 'KRAS DARPin K27'
system_slug = 'kras_darpin_k27'
source_notebook = 'Figure_4_and_SI_KRAS_DARPin_K27.ipynb'

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
    "summary_null_model": "same_size_random_edge_subsets",
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

# --- End copied code from Figure_4_and_SI_KRAS_DARPin_K27.ipynb cells 0-2,4-6 ---

# --- Begin copied code from Figure_4_and_SI_KRAS_DARPin_K55.ipynb cells 0-2,4-6 ---

# %% [Figure_4_and_SI_KRAS_DARPin_K55.ipynb cell 0]
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

# %% [Figure_4_and_SI_KRAS_DARPin_K55.ipynb cell 1]
df = pd.read_csv('../data_files/kras_genetic_arch/RESULTS/DARPin_K55_binding_fitness_decomposition_per_variant.tsv', sep='\t')

df = df.drop_duplicates(subset="aa_seq", keep="first")
df.dropna(inplace=True, subset=["aa_seq", "observed_binding_fitness", "latent_ddGf_kcal_mol", "latent_ddGb_kcal_mol"])

sequences = [fl.BaseNumpySequence(sequence) for sequence in df['aa_seq']]

# Construct fitness landscape
landscape = fl.FitnessLandscape.build(
    sequences,
    graph="hamming",
    _compute_hamming_edges=False,
)

# %% [Figure_4_and_SI_KRAS_DARPin_K55.ipynb cell 2]
# Phenotype 1 folds, phenotype 2 folds + binds

folding_map = dict(zip(df["aa_seq"], df["latent_ddGf_kcal_mol"]))
landscape.attach(
    name="latent_folding",
    values=folding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

binding_map = dict(zip(df["aa_seq"], df["latent_ddGb_kcal_mol"]))
landscape.attach(
    name="latent_binding",
    values=binding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

observed_fitness_map = dict(zip(df["aa_seq"], df["observed_binding_fitness"]))
landscape.attach(
    name="fitness_observed_binding",
    values=observed_fitness_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

F = zscore(landscape.fitness_layers['latent_folding'].to_scalar())
B = zscore(landscape.fitness_layers['latent_binding'].to_scalar())
s = F - B  # <0 fold-limited, >0 bind-limited
landscape.attach(
    name="limiter_index",
    values=s,
    dtype="numeric")

# %% [Figure_4_and_SI_KRAS_DARPin_K55.ipynb cell 4]
# Compute fitness gradients / energy
f = landscape.fitness_layers["fitness_observed_binding"].to_scalar()

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

# %% [Figure_4_and_SI_KRAS_DARPin_K55.ipynb cell 5]
# Use graph node order so `zip(landscape.graph.nodes(), shuffled_s)` stays aligned
graph_nodes = list(landscape.graph.nodes())
idx_by_node = {n: i for i, n in enumerate(landscape._node_order)}

f_all = np.asarray(landscape.fitness_layers["fitness_observed_binding"].to_scalar(), dtype=float).ravel()
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

# %% [Figure_4_and_SI_KRAS_DARPin_K55.ipynb cell 6]
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

system_name = 'KRAS DARPin K55'
system_slug = 'kras_darpin_k55'
source_notebook = 'Figure_4_and_SI_KRAS_DARPin_K55.ipynb'

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
    "summary_null_model": "same_size_random_edge_subsets",
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

# --- End copied code from Figure_4_and_SI_KRAS_DARPin_K55.ipynb cells 0-2,4-6 ---

# --- Begin copied code from Figure_4_and_SI_KRAS_PIK3CG.ipynb cells 0-2,4-6 ---

# %% [Figure_4_and_SI_KRAS_PIK3CG.ipynb cell 0]
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

# %% [Figure_4_and_SI_KRAS_PIK3CG.ipynb cell 1]
df = pd.read_csv('../data_files/kras_genetic_arch/RESULTS/PIK3CG_binding_fitness_decomposition_per_variant.tsv', sep='\t')

df = df.drop_duplicates(subset="aa_seq", keep="first")
df.dropna(inplace=True, subset=["aa_seq", "observed_binding_fitness", "latent_ddGf_kcal_mol", "latent_ddGb_kcal_mol"])

sequences = [fl.BaseNumpySequence(sequence) for sequence in df['aa_seq']]

# Construct fitness landscape
landscape = fl.FitnessLandscape.build(
    sequences,
    graph="hamming",
    _compute_hamming_edges=False,
)

# %% [Figure_4_and_SI_KRAS_PIK3CG.ipynb cell 2]
# Phenotype 1 folds, phenotype 2 folds + binds

folding_map = dict(zip(df["aa_seq"], df["latent_ddGf_kcal_mol"]))
landscape.attach(
    name="latent_folding",
    values=folding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

binding_map = dict(zip(df["aa_seq"], df["latent_ddGb_kcal_mol"]))
landscape.attach(
    name="latent_binding",
    values=binding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

observed_fitness_map = dict(zip(df["aa_seq"], df["observed_binding_fitness"]))
landscape.attach(
    name="fitness_observed_binding",
    values=observed_fitness_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

F = zscore(landscape.fitness_layers['latent_folding'].to_scalar())
B = zscore(landscape.fitness_layers['latent_binding'].to_scalar())
s = F - B  # <0 fold-limited, >0 bind-limited
landscape.attach(
    name="limiter_index",
    values=s,
    dtype="numeric")

# %% [Figure_4_and_SI_KRAS_PIK3CG.ipynb cell 4]
# Compute fitness gradients / energy
f = landscape.fitness_layers["fitness_observed_binding"].to_scalar()

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

# %% [Figure_4_and_SI_KRAS_PIK3CG.ipynb cell 5]
# Use graph node order so `zip(landscape.graph.nodes(), shuffled_s)` stays aligned
graph_nodes = list(landscape.graph.nodes())
idx_by_node = {n: i for i, n in enumerate(landscape._node_order)}

f_all = np.asarray(landscape.fitness_layers["fitness_observed_binding"].to_scalar(), dtype=float).ravel()
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

# %% [Figure_4_and_SI_KRAS_PIK3CG.ipynb cell 6]
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

system_name = 'KRAS PIK3CG'
system_slug = 'kras_pik3cg'
source_notebook = 'Figure_4_and_SI_KRAS_PIK3CG.ipynb'

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
    "summary_null_model": "same_size_random_edge_subsets",
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

# --- End copied code from Figure_4_and_SI_KRAS_PIK3CG.ipynb cells 0-2,4-6 ---

# --- Begin copied code from Figure_4_and_SI_KRAS_RAF1.ipynb cells 0-2,4-6 ---

# %% [Figure_4_and_SI_KRAS_RAF1.ipynb cell 0]
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

# %% [Figure_4_and_SI_KRAS_RAF1.ipynb cell 1]
df = pd.read_csv('../data_files/kras_genetic_arch/RESULTS/RAF1_binding_fitness_decomposition_per_variant.tsv', sep='\t')

df = df.drop_duplicates(subset="aa_seq", keep="first")
df.dropna(inplace=True, subset=["aa_seq", "observed_binding_fitness", "latent_ddGf_kcal_mol", "latent_ddGb_kcal_mol"])

sequences = [fl.BaseNumpySequence(sequence) for sequence in df['aa_seq']]

# Construct fitness landscape
landscape = fl.FitnessLandscape.build(
    sequences,
    graph="hamming",
    _compute_hamming_edges=False,
)

# %% [Figure_4_and_SI_KRAS_RAF1.ipynb cell 2]
# Phenotype 1 folds, phenotype 2 folds + binds

folding_map = dict(zip(df["aa_seq"], df["latent_ddGf_kcal_mol"]))
landscape.attach(
    name="latent_folding",
    values=folding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

binding_map = dict(zip(df["aa_seq"], df["latent_ddGb_kcal_mol"]))
landscape.attach(
    name="latent_binding",
    values=binding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

observed_fitness_map = dict(zip(df["aa_seq"], df["observed_binding_fitness"]))
landscape.attach(
    name="fitness_observed_binding",
    values=observed_fitness_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

F = zscore(landscape.fitness_layers['latent_folding'].to_scalar())
B = zscore(landscape.fitness_layers['latent_binding'].to_scalar())
s = F - B  # <0 fold-limited, >0 bind-limited
landscape.attach(
    name="limiter_index",
    values=s,
    dtype="numeric")

# %% [Figure_4_and_SI_KRAS_RAF1.ipynb cell 4]
# Compute fitness gradients / energy
f = landscape.fitness_layers["fitness_observed_binding"].to_scalar()

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

# %% [Figure_4_and_SI_KRAS_RAF1.ipynb cell 5]
# Use graph node order so `zip(landscape.graph.nodes(), shuffled_s)` stays aligned
graph_nodes = list(landscape.graph.nodes())
idx_by_node = {n: i for i, n in enumerate(landscape._node_order)}

f_all = np.asarray(landscape.fitness_layers["fitness_observed_binding"].to_scalar(), dtype=float).ravel()
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

# %% [Figure_4_and_SI_KRAS_RAF1.ipynb cell 6]
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

system_name = 'KRAS RAF1'
system_slug = 'kras_raf1'
source_notebook = 'Figure_4_and_SI_KRAS_RAF1.ipynb'

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
    "summary_null_model": "same_size_random_edge_subsets",
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

# --- End copied code from Figure_4_and_SI_KRAS_RAF1.ipynb cells 0-2,4-6 ---

# --- Begin copied code from Figure_4_and_SI_KRAS_RALGDS.ipynb cells 0-2,4-6 ---

# %% [Figure_4_and_SI_KRAS_RALGDS.ipynb cell 0]
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

# %% [Figure_4_and_SI_KRAS_RALGDS.ipynb cell 1]
df = pd.read_csv('../data_files/kras_genetic_arch/RESULTS/RALGDS_binding_fitness_decomposition_per_variant.tsv', sep='\t')

df = df.drop_duplicates(subset="aa_seq", keep="first")
df.dropna(inplace=True, subset=["aa_seq", "observed_binding_fitness", "latent_ddGf_kcal_mol", "latent_ddGb_kcal_mol"])

sequences = [fl.BaseNumpySequence(sequence) for sequence in df['aa_seq']]

# Construct fitness landscape
landscape = fl.FitnessLandscape.build(
    sequences,
    graph="hamming",
    _compute_hamming_edges=False,
)

# %% [Figure_4_and_SI_KRAS_RALGDS.ipynb cell 2]
# Phenotype 1 folds, phenotype 2 folds + binds

folding_map = dict(zip(df["aa_seq"], df["latent_ddGf_kcal_mol"]))
landscape.attach(
    name="latent_folding",
    values=folding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

binding_map = dict(zip(df["aa_seq"], df["latent_ddGb_kcal_mol"]))
landscape.attach(
    name="latent_binding",
    values=binding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

observed_fitness_map = dict(zip(df["aa_seq"], df["observed_binding_fitness"]))
landscape.attach(
    name="fitness_observed_binding",
    values=observed_fitness_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

F = zscore(landscape.fitness_layers['latent_folding'].to_scalar())
B = zscore(landscape.fitness_layers['latent_binding'].to_scalar())
s = F - B  # <0 fold-limited, >0 bind-limited
landscape.attach(
    name="limiter_index",
    values=s,
    dtype="numeric")

# %% [Figure_4_and_SI_KRAS_RALGDS.ipynb cell 4]
# Compute fitness gradients / energy
f = landscape.fitness_layers["fitness_observed_binding"].to_scalar()

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

# %% [Figure_4_and_SI_KRAS_RALGDS.ipynb cell 5]
# Use graph node order so `zip(landscape.graph.nodes(), shuffled_s)` stays aligned
graph_nodes = list(landscape.graph.nodes())
idx_by_node = {n: i for i, n in enumerate(landscape._node_order)}

f_all = np.asarray(landscape.fitness_layers["fitness_observed_binding"].to_scalar(), dtype=float).ravel()
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

# %% [Figure_4_and_SI_KRAS_RALGDS.ipynb cell 6]
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

system_name = 'KRAS RALGDS'
system_slug = 'kras_ralgds'
source_notebook = 'Figure_4_and_SI_KRAS_RALGDS.ipynb'

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
    "summary_null_model": "same_size_random_edge_subsets",
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

# --- End copied code from Figure_4_and_SI_KRAS_RALGDS.ipynb cells 0-2,4-6 ---

# --- Begin copied code from Figure_4_and_SI_KRAS_SOS1.ipynb cells 0-2,4-6 ---

# %% [Figure_4_and_SI_KRAS_SOS1.ipynb cell 0]
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

# %% [Figure_4_and_SI_KRAS_SOS1.ipynb cell 1]
df = pd.read_csv('../data_files/kras_genetic_arch/RESULTS/SOS1_binding_fitness_decomposition_per_variant.tsv', sep='\t')

df = df.drop_duplicates(subset="aa_seq", keep="first")
df.dropna(inplace=True, subset=["aa_seq", "observed_binding_fitness", "latent_ddGf_kcal_mol", "latent_ddGb_kcal_mol"])

sequences = [fl.BaseNumpySequence(sequence) for sequence in df['aa_seq']]

# Construct fitness landscape
landscape = fl.FitnessLandscape.build(
    sequences,
    graph="hamming",
    _compute_hamming_edges=False,
)

# %% [Figure_4_and_SI_KRAS_SOS1.ipynb cell 2]
# Phenotype 1 folds, phenotype 2 folds + binds

folding_map = dict(zip(df["aa_seq"], df["latent_ddGf_kcal_mol"]))
landscape.attach(
    name="latent_folding",
    values=folding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

binding_map = dict(zip(df["aa_seq"], df["latent_ddGb_kcal_mol"]))
landscape.attach(
    name="latent_binding",
    values=binding_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

observed_fitness_map = dict(zip(df["aa_seq"], df["observed_binding_fitness"]))
landscape.attach(
    name="fitness_observed_binding",
    values=observed_fitness_map,
    dtype="numeric",
    map_by="sequence",
    on_duplicates="first"
    )

F = zscore(landscape.fitness_layers['latent_folding'].to_scalar())
B = zscore(landscape.fitness_layers['latent_binding'].to_scalar())
s = F - B  # <0 fold-limited, >0 bind-limited
landscape.attach(
    name="limiter_index",
    values=s,
    dtype="numeric")

# %% [Figure_4_and_SI_KRAS_SOS1.ipynb cell 4]
# Compute fitness gradients / energy
f = landscape.fitness_layers["fitness_observed_binding"].to_scalar()

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

# %% [Figure_4_and_SI_KRAS_SOS1.ipynb cell 5]
# Use graph node order so `zip(landscape.graph.nodes(), shuffled_s)` stays aligned
graph_nodes = list(landscape.graph.nodes())
idx_by_node = {n: i for i, n in enumerate(landscape._node_order)}

f_all = np.asarray(landscape.fitness_layers["fitness_observed_binding"].to_scalar(), dtype=float).ravel()
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

# %% [Figure_4_and_SI_KRAS_SOS1.ipynb cell 6]
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

system_name = 'KRAS SOS1'
system_slug = 'kras_sos1'
source_notebook = 'Figure_4_and_SI_KRAS_SOS1.ipynb'

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
    "summary_null_model": "same_size_random_edge_subsets",
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

# --- End copied code from Figure_4_and_SI_KRAS_SOS1.ipynb cells 0-2,4-6 ---

run_postprocess(SCRIPT_DIR / "postprocess.py", globals())
