# Exported from analysis.ipynb

# %%
from pathlib import Path
import os
import sys
import types

analysis_dir = Path.cwd().resolve().parent
scripts_dir = analysis_dir.parents[1] / "scripts"
sys.path.insert(0, str(scripts_dir))
from paper_runtime import find_project_root, prepare_source_data_compat_dir, resolve_publication_data_dirs, _patch_matplotlib_savefig
project_root = find_project_root(analysis_dir)
data_dirs = resolve_publication_data_dirs(project_root)
compat_data = analysis_dir / "data_files"
compat_alisim = analysis_dir / "alisim_results"
prepare_source_data_compat_dir(data_dirs["data_files"], compat_data)
if compat_alisim.is_symlink() and not compat_alisim.exists():
    compat_alisim.unlink()
if not compat_alisim.exists():
    compat_alisim.symlink_to(data_dirs["alisim_results"], target_is_directory=True)
sys.path.insert(0, str(Path.cwd().resolve()))
sys.modules.setdefault("py3Dmol", types.ModuleType("py3Dmol"))
os.environ.setdefault("MPLBACKEND", "Agg")
_patch_matplotlib_savefig()
processed_dir = project_root / "data" / "processed"
figure_dir = analysis_dir / "figures"
table_dir = analysis_dir / "tables"
figure_dir.mkdir(exist_ok=True)
table_dir.mkdir(exist_ok=True)

def display(*args, **kwargs):
    return None

# %% [markdown]
# ## SH3 folding-binding boundary figures
#
# Copied and focused from `the copied source/figure_notebooks_rev snapshot/Figure_4_and_SI_SH3_comb_core.ipynb`.

# %%
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

# %%
# Load promoted outputs from the Figure 6 boundary experiments for summary analyses.
import json

_exp_boundary_dir = processed_dir / "experimental_boundaries"
processed_boundary_outputs = {}
for _path in _exp_boundary_dir.glob("*.json"):
    try:
        processed_boundary_outputs[_path.stem] = json.loads(_path.read_text())
    except Exception:
        pass
for _path in _exp_boundary_dir.glob("*.csv"):
    try:
        processed_boundary_outputs[_path.stem] = pd.read_csv(_path)
    except Exception:
        pass
_kras_dir = _exp_boundary_dir / "kras_boundary_enrichment_results"
kras_boundary_payloads = []
for _path in sorted(_kras_dir.glob("*_boundary_enrichment.json")):
    try:
        kras_boundary_payloads.append(json.loads(_path.read_text()))
    except Exception:
        pass

# %%
df = pd.read_csv('../data_files/combinatorial_core/predicted_phenotypes_all.csv')
df = df.drop_duplicates(subset="core", keep="first")

sequences = [fl.BaseNumpySequence(sequence) for sequence in df['core']]

# Construct fitness landscape
landscape = fl.FitnessLandscape.build(
    sequences,
    graph="hamming",
    _compute_hamming_edges=False,
)

# %%
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

# %%
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

# %%
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

# %%
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

# %%

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

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Wz as before
Wz = np.zeros(n, dtype=float)
for i in range(n):
    Wz[i] = np.mean(z[nbr_idx[i]]) if deg[i] > 0 else np.nan

mask = np.isfinite(Wz)
x = z[mask]
y = Wz[mask]

b, a = np.polyfit(x, y, 1)

# ---- fixed display window ----
xlim = (-20, 50)
ylim = (-15, 20)

# KDE evaluated only in that window
xy = np.vstack([x, y])
kde = gaussian_kde(xy)

nxg, nyg = 260, 260
xg = np.linspace(xlim[0], xlim[1], nxg)
yg = np.linspace(ylim[0], ylim[1], nyg)
Xg, Yg = np.meshgrid(xg, yg)
Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

# ---- mask very low density so only contoured area is drawn ----
# keep only top density mass (e.g., above 85th percentile of grid density)
z_thr = np.quantile(Zg, 0.85)
Zmask = np.ma.masked_less(Zg, z_thr)

fig, ax = plt.subplots(figsize=(2.5, 2.25))

# filled contours only where density is high enough
levels = np.linspace(Zmask.min(), Zmask.max(), 10)
cf = ax.contourf(Xg, Yg, Zmask, levels=levels, cmap="Blues", alpha=0.95)
ax.contour(Xg, Yg, Zmask, levels=levels[::2], colors="white", linewidths=0.7, alpha=0.9)

# regression line and axes guides
xx = np.linspace(xlim[0], xlim[1], 200)
ax.plot(xx, a + b * xx, lw=2)
ax.axvline(0, lw=1)
ax.axhline(0, lw=1)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xlabel("Centered local energy")
ax.set_ylabel("Spatial lag")
# ax.set_title(f"Moran density contour (I ≈ slope)\nobserved I = {moran_I:.3f}, fitted slope = {b:.3f}")

# cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label("Point density")

plt.tight_layout()
plt.savefig("../figures/si_figures/si_figure_spatial_autocorrelation/density_plot.pdf")
plt.show()

# %%
# Quadrant masks
HH = (x > 0) & (y > 0)
LL = (x < 0) & (y < 0)
HL = (x > 0) & (y < 0)
LH = (x < 0) & (y > 0)

# "Numerator contribution" proxy per point: z_i * (Wz)_i
c = x * y
tot_pos = c[HH].sum() + c[LL].sum()
tot_neg = c[HL].sum() + c[LH].sum()   # this will be negative

print("Counts (fraction):")
print("  HH:", HH.mean(), "LL:", LL.mean(), "HL:", HL.mean(), "LH:", LH.mean())

print("\nContribution to numerator sum_i z_i Wz_i:")
print("  HH:", c[HH].sum())
print("  LL:", c[LL].sum())
print("  HL:", c[HL].sum(), "(negative)")
print("  LH:", c[LH].sum(), "(negative)")
print("  Net:", c.sum())

print("\nShare of positive contribution (HH vs LL):")
print("  HH share:", c[HH].sum() / (tot_pos + 1e-12))
print("  LL share:", c[LL].sum() / (tot_pos + 1e-12))

# %%
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

# %%
# ----------------------------
# 6) Report + quick plots
# ----------------------------
print(f"\nGlobal edge Moran I: {I_all:.5f} (perm p={p_all:.4g})")
print(f"Boundary-only Moran I: {I_boundary:.5f} (non-isolated edges={n_boundary_noniso})")
print(f"Within-only Moran I:   {I_within:.5f} (non-isolated edges={n_within_noniso})")
print(f"Delta Moran (boundary - within): {I_boundary - I_within:.5f}")
print(f"\nBoundary contribution delta (mean c_boundary - mean c_within): {obs_delta_contrib:.5e}")
print(f"Permutation p (label shuffle): {p_delta_contrib:.4g}")
print(f"\nResidual Moran I after regressing out boundary label: {I_resid:.5f} (perm p={p_resid:.4g})")

fig, axes = plt.subplots(1, 2, figsize=(5, 2), constrained_layout=True)
axes[0].hist(null_all, bins=40, alpha=0.8)
axes[0].axvline(I_all, lw=2)
axes[0].set_title(f"Global I null\nobs={I_all:.4f}, p={p_all:.3g}")
axes[0].set_xlabel("Moran I (perm)")
axes[0].set_ylabel("count")

axes[1].hist(null_delta_contrib, bins=40, alpha=0.8)
axes[1].axvline(obs_delta_contrib, lw=2)
axes[1].set_title(f"Boundary contribution delta null\nobs={obs_delta_contrib:.3e}, p={p_delta_contrib:.3g}")
axes[1].set_xlabel("mean(c_boundary)-mean(c_within)")
axes[1].set_ylabel("count")
plt.savefig('../figures/si_figures/si_figure_spatial_autocorrelation/global_and_boundary_morans_I.pdf')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# ============================================================
# High-energy autocorrelation + boundary contribution to HH
# (edge-level, line-graph neighbors)
# ============================================================

# ---- knobs ----
q_hi = 0.90                 # "high-energy" threshold quantile
n_perm_high = 1000          # perm test for high-energy Moran I
n_perm_boundary = 1000      # perm test for boundary contribution to HH
rng = np.random.default_rng(0)

# ---- safety checks ----
required = ["edge_energy", "boundary_mask", "within_mask", "nbr_idx"]
missing = [v for v in required if v not in globals()]
if missing:
    raise RuntimeError(f"Missing variables: {missing}. Run the BFS-Moran cell immediately above first.")

edge_energy = np.asarray(edge_energy, dtype=float)
boundary_mask = np.asarray(boundary_mask, dtype=bool)
within_mask = np.asarray(within_mask, dtype=bool)

m = len(edge_energy)
if m == 0:
    raise RuntimeError("No edges available.")
if not (len(boundary_mask) == m and len(within_mask) == m and len(nbr_idx) == m):
    raise RuntimeError("Lengths of edge arrays/masks/neighborhoods do not match.")

# ---- build row-standardized sparse W over edge-neighborhoods ----
row_parts, col_parts, dat_parts = [], [], []
for i, nb in enumerate(nbr_idx):
    nb = np.asarray(nb, dtype=int)
    if nb.size == 0:
        continue
    row_parts.append(np.full(nb.size, i, dtype=int))
    col_parts.append(nb)
    dat_parts.append(np.full(nb.size, 1.0 / nb.size, dtype=float))

if row_parts:
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.concatenate(dat_parts)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(m, m))
else:
    W = sparse.csr_matrix((m, m))

def moran_I(x):
    z = x - x.mean()
    lag = W @ z
    den = np.dot(z, z) + 1e-12
    I = float(np.dot(z, lag) / den)
    return I, z, lag

# ============================================================
# 1) Are HIGH energies autocorrelated?
# ============================================================
thr = np.quantile(edge_energy, q_hi)
H = (edge_energy >= thr).astype(float)         # binary high-edge indicator
p_high = H.mean()

I_high_bin, zH, lagH = moran_I(H)

null_I_high_bin = np.empty(n_perm_high, dtype=float)
for k in range(n_perm_high):
    Hp = rng.permutation(H)
    null_I_high_bin[k] = moran_I(Hp)[0]
p_I_high_bin = (np.sum(null_I_high_bin >= I_high_bin) + 1) / (n_perm_high + 1)

# Also test magnitude of high tail only (excess above threshold)
X_tail = np.maximum(edge_energy - thr, 0.0)
I_high_tail, _, _ = moran_I(X_tail)

null_I_high_tail = np.empty(n_perm_high, dtype=float)
for k in range(n_perm_high):
    Xp = rng.permutation(X_tail)
    null_I_high_tail[k] = moran_I(Xp)[0]
p_I_high_tail = (np.sum(null_I_high_tail >= I_high_tail) + 1) / (n_perm_high + 1)

# ============================================================
# 2) How much does boundary contribute to HH autocorrelation?
# ============================================================
# Local HH score: edge is high * fraction of high neighbors
# (captures high-high clustering intensity around each edge)
lagH_raw = W @ H
hh_local = H * lagH_raw

# Observed boundary-vs-within HH contrast
obs_delta_hh = hh_local[boundary_mask].mean() - hh_local[within_mask].mean()

# Fraction of total HH mass carried by boundary edges
obs_hh_mass_boundary = hh_local[boundary_mask].sum() / (hh_local.sum() + 1e-12)
obs_boundary_edge_frac = boundary_mask.mean()
obs_hh_mass_enrichment = obs_hh_mass_boundary / (obs_boundary_edge_frac + 1e-12)

# Permute boundary labels (keep number of boundary edges fixed), HH field fixed
n_b = int(boundary_mask.sum())
null_delta_hh = np.empty(n_perm_boundary, dtype=float)
null_hh_mass_boundary = np.empty(n_perm_boundary, dtype=float)

for k in range(n_perm_boundary):
    bperm = np.zeros(m, dtype=bool)
    bperm[rng.choice(m, size=n_b, replace=False)] = True
    wperm = ~bperm

    null_delta_hh[k] = hh_local[bperm].mean() - hh_local[wperm].mean()
    null_hh_mass_boundary[k] = hh_local[bperm].sum() / (hh_local.sum() + 1e-12)

p_delta_hh = (np.sum(null_delta_hh >= obs_delta_hh) + 1) / (n_perm_boundary + 1)
p_hh_mass = (np.sum(null_hh_mass_boundary >= obs_hh_mass_boundary) + 1) / (n_perm_boundary + 1)

# ============================================================
# 3) Optional: quantile scan (no extra permutations)
# ============================================================
q_grid = np.linspace(0.80, 0.98, 10)
Iq = []
Dq = []
for q in q_grid:
    t = np.quantile(edge_energy, q)
    Hq = (edge_energy >= t).astype(float)
    Iq.append(moran_I(Hq)[0])
    hhq = Hq * (W @ Hq)
    Dq.append(hhq[boundary_mask].mean() - hhq[within_mask].mean())

Iq = np.array(Iq, dtype=float)
Dq = np.array(Dq, dtype=float)

# ============================================================
# 4) Report
# ============================================================
print(f"Edges: {m} | boundary: {boundary_mask.sum()} | within: {within_mask.sum()}")
print(f"High threshold q={q_hi:.2f}: energy >= {thr:.6g}  (high-edge fraction={p_high:.3f})")

print("\n[High-energy autocorrelation]")
print(f"Binary-high Moran I: {I_high_bin:.5f} (perm p={p_I_high_bin:.4g})")
print(f"High-tail Moran I:   {I_high_tail:.5f} (perm p={p_I_high_tail:.4g})")

print("\n[Boundary contribution to HH autocorrelation]")
print(f"Delta HH-local (boundary - within): {obs_delta_hh:.5e} (perm p={p_delta_hh:.4g})")
print(f"Boundary HH mass fraction: {obs_hh_mass_boundary:.4f} "
      f"(boundary edge fraction={obs_boundary_edge_frac:.4f}, enrichment={obs_hh_mass_enrichment:.3f}x, "
      f"perm p={p_hh_mass:.4g})")

# ============================================================
# 5) Plots
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.2), constrained_layout=True)

axes[0].hist(null_I_high_bin, bins=40, alpha=0.85)
axes[0].axvline(I_high_bin, lw=2)
axes[0].set_title(f"High-edge Moran I null\nobs={I_high_bin:.4f}, p={p_I_high_bin:.3g}")
axes[0].set_xlabel("I (binary high-edge)")
axes[0].set_ylabel("count")

axes[1].hist(null_delta_hh, bins=40, alpha=0.85)
axes[1].axvline(obs_delta_hh, lw=2)
axes[1].set_title(f"Boundary HH contribution null\nobs={obs_delta_hh:.3e}, p={p_delta_hh:.3g}")
axes[1].set_xlabel("mean(hh_local|boundary)-mean(hh_local|within)")
axes[1].set_ylabel("count")

ax2 = axes[2]
ax2.plot(q_grid, Iq, marker="o", label="Moran I of high-edge indicator")
ax2b = ax2.twinx()
ax2b.plot(q_grid, Dq, marker="s", linestyle="--", label="Boundary-minus-within HH local")
ax2.set_xlabel("High-energy quantile threshold q")
ax2.set_ylabel("I_high(q)")
ax2b.set_ylabel("Δ HH-local(q)")
ax2.set_title("Threshold sensitivity (no perm)")
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="best", frameon=False)

plt.savefig('../figures/si_figures/si_figure_spatial_autocorrelation/HH_filter_moran_I.pdf')
plt.show()

# %%
# Correlation between limiter and fitness giving spurious / autocorrelated results.

s = landscape.fitness_layers["limiter_index"].to_scalar()
f = landscape.fitness_layers["composite_fitness"].to_scalar()

pearson_r, pearson_p = pearsonr(s, f)
spearman_r, spearman_p = spearmanr(s, f)

print("Pearson r:", pearson_r, "p:", pearson_p)
print("Spearman r:", spearman_r, "p:", spearman_p)

plt.figure(figsize=(3, 3))
plt.scatter(s, f, s=5, alpha=0.3)
plt.xlabel("Limiter index (fold − bind)")
plt.ylabel("Composite fitness")
plt.axvline(0, color="k", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('../figures/si_figures/si_figure_limiter_index_vs_composite_fitness/limiter_index_vs_composite_fitness.pdf')
plt.show()

# %%
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

# %%
# Sample connected component for tractibility
L_sub = bfs_sub_landscape(landscape, frac=0.1, seed=0)

# Isolate landscape objects
components = partition_landscape_into_regime_components(L_sub)
L_sub_fold = components['fold_largest_landscape']
L_sub_bind = components['bind_largest_landscape']
L_sub_bound = components['boundary_largest_landscape']

# %%
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

# %%
import numpy as np
import matplotlib.pyplot as plt

labels = ["Binding-limited", "Boundary", "Folding-limited"]
results = [tmap_bind_res, tmap_bound_res, tmap_fold_res]

t  = np.array([r["t_map"] for r in results], dtype=float)
lo = np.array([r["t_lower_confidence_interval"] for r in results], dtype=float)
hi = np.array([r["t_upper_confidence_interval"] for r in results], dtype=float)

# --- sanitize to avoid negative yerr if CI doesn't bracket t_map ---
lo_fixed = np.minimum(lo, hi)
hi_fixed = np.maximum(lo, hi)
lo_plot = np.minimum(lo_fixed, t)
hi_plot = np.maximum(hi_fixed, t)
yerr = np.vstack([t - lo_plot, hi_plot - t])

x = np.arange(len(t))

fig, ax = plt.subplots(figsize=(1.5, 2))

# Bars: light gray fill, black outline
ax.bar(
    x, t,
    color="#d3d3d3",
    edgecolor="black",
    linewidth=1,
)

# Error bars: black
ax.errorbar(
    x, t,
    yerr=yerr,
    fmt="none",
    ecolor="black",
    elinewidth=1,
    capsize=6,
    capthick=1,
)

ax.set_xticks(x)
# ax.set_xticklabels(labels, rotation=0)
ax.set_ylabel(r"$t_{\mathrm{MAP}}$")

plt.tight_layout()
plt.savefig('../figures/figure_4/tmap_vs_subgraph_regime.pdf')
plt.show()

# %%
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

# %%
# ==== Reproduce one-off limiter plot + 3-panel component plot (same positions, fixed indexing) ====
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import fitness_landscape as fl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ----------------------------
# Config
# ----------------------------
BFS_FRAC = 0.1
BFS_SEED = 2
LAYOUT_SEED = 0
EPS = 0.5

NODE_SIZE = 10
NODE_LINE_WIDTH = 0.25
NODE_LINE_COLOR = "black"

# ----------------------------
# 1) Build G_sub
# ----------------------------
G = landscape.graph
rng = np.random.default_rng(BFS_SEED)
target_n = max(int(BFS_FRAC * G.number_of_nodes()), 2)

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

G_sub = G.subgraph(seen).copy()
if G_sub.is_directed():
    comps = list(nx.weakly_connected_components(G_sub))
else:
    comps = list(nx.connected_components(G_sub))
G_sub = G_sub.subgraph(max(comps, key=len)).copy()

# Canonical node order aligned to layers
node_list_sub = [n for n in landscape._node_order if n in G_sub]
print("Full graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")
print("BFS subgraph:", len(node_list_sub), "nodes,", G_sub.number_of_edges(), "edges")

# ----------------------------
# 2) Correct layer accessor (aligned to landscape._node_order)
# ----------------------------
idx_full = {n: i for i, n in enumerate(landscape._node_order)}

def layer_on_subgraph(layer_key):
    vals = np.asarray(landscape.fitness_layers[layer_key].to_scalar(), dtype=float)
    return np.array([vals[idx_full[n]] for n in node_list_sub], dtype=float)

# ----------------------------
# 3) Layout (reused in both plots)
# ----------------------------
def compute_layout(Gx, seed=0):
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(Gx, prog="sfdp")
    except Exception:
        return nx.spring_layout(Gx, seed=seed)

pos = compute_layout(G_sub, seed=LAYOUT_SEED)

# ----------------------------
# 4) One-off limiter plot
# ----------------------------
def plot_limiter_coolwarm(
    G_sub,
    node_list_sub,
    pos,
    layer_on_subgraph,
    *,
    node_size=8,
    node_line_width=0.25,
    node_line_color="black",
    edge_width=0.2,
    edge_alpha=0.0,
    fraction_nodes=1.0,
    seed=0,
    figsize=(4, 3),
    cmap="coolwarm_r",
):
    s = np.asarray(layer_on_subgraph("limiter_index"), dtype=float)
    scale = np.percentile(np.abs(s), 95) + 1e-12
    s_norm = np.clip(s / scale, -1, 1)

    n = len(node_list_sub)
    k = max(1, int(np.ceil(np.clip(fraction_nodes, 0.0, 1.0) * n)))
    rng = np.random.default_rng(seed)
    keep_idx = np.arange(n) if k == n else rng.choice(n, size=k, replace=False)
    keep_nodes = [node_list_sub[i] for i in keep_idx]
    keep_set = set(keep_nodes)

    xs = np.array([pos[n][0] for n in keep_nodes], dtype=float)
    ys = np.array([pos[n][1] for n in keep_nodes], dtype=float)
    cs = s_norm[keep_idx]

    fig, ax = plt.subplots(figsize=figsize)

    if edge_alpha > 0:
        for u, v in G_sub.edges():
            if u in keep_set and v in keep_set:
                x0, y0 = pos[u]; x1, y1 = pos[v]
                ax.plot([x0, x1], [y0, y1], color="black", lw=edge_width, alpha=edge_alpha, zorder=1)

    sc = ax.scatter(
        xs, ys, c=cs, cmap=cmap, vmin=-1, vmax=1,
        s=node_size, edgecolors=node_line_color, linewidths=node_line_width, zorder=2
    )


    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/figure_4/SH3_BFS_0.2_limiter_index.pdf')
    plt.show()

plot_limiter_coolwarm(
    G_sub, node_list_sub, pos, layer_on_subgraph,
    node_size=NODE_SIZE,
    node_line_width=NODE_LINE_WIDTH,
    node_line_color=NODE_LINE_COLOR,
    edge_width=0.25,
    edge_alpha=0.0,
    fraction_nodes=1.0,
    figsize=(4, 4),
    seed=0
)

# ----------------------------
# 5) Ordered induced-subgraph helper + sub-landscape
# ----------------------------
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

def sub_landscape_from_graph(base_landscape, sub_graph):
    ordered_nodes = [n for n in base_landscape._node_order if n in sub_graph]
    if not ordered_nodes:
        return None

    node_index_map = {n: i for i, n in enumerate(base_landscape._node_order)}
    indices = [node_index_map[n] for n in ordered_nodes]

    sub_sequences = [base_landscape.sequences[i] for i in indices]
    sub_fitness = base_landscape._subset_fitness_layers(indices)
    sub_annotations = base_landscape._subset_annotation_layers(indices)
    sub_embeddings = (
        {d: emb[indices].copy() for d, emb in base_landscape.embeddings.items()}
        if base_landscape.embeddings else None
    )

    ordered_graph = _induced_subgraph_in_order(sub_graph, ordered_nodes)

    sub = fl.FitnessLandscape(
        sequences=sub_sequences,
        graph=ordered_graph,
        fitness_layers=sub_fitness,
        annotation_layers=sub_annotations,
        embeddings=sub_embeddings,
        emb_arr_key=base_landscape._emb_arr_key,
        active_embedding_domain=base_landscape._active_embedding_domain,
        embedding_metadata=base_landscape.embedding_metadata,
    )
    if base_landscape._active_view_name is not None:
        sub.view(base_landscape._active_view_name)
    return sub

def partition_exact(landscape_sub, limiter_layer="limiter_index", eps=0.0):
    Gx = landscape_sub.graph
    s = np.asarray(landscape_sub.fitness_layers[limiter_layer].to_scalar(), dtype=float).ravel()
    node_to_s = {n: s[i] for i, n in enumerate(landscape_sub._node_order)}
    nodes = list(landscape_sub._node_order)

    fold_nodes = [n for n in nodes if node_to_s[n] < -eps]
    bind_nodes = [n for n in nodes if node_to_s[n] > +eps]

    boundary_edges = []
    for u, v in Gx.edges():
        su, sv = node_to_s[u], node_to_s[v]
        if (su < -eps and sv > +eps) or (su > +eps and sv < -eps):
            boundary_edges.append((u, v))

    G_fold = Gx.subgraph(fold_nodes)
    G_bind = Gx.subgraph(bind_nodes)
    G_boundary = Gx.edge_subgraph(boundary_edges)

    def comps(graph):
        if graph.number_of_nodes() == 0:
            return []
        Gu = graph.to_undirected(as_view=True)
        return [list(c) for c in nx.connected_components(Gu)]

    fold_ls = [sub_landscape_from_graph(landscape_sub, G_fold.subgraph(c)) for c in comps(G_fold)]
    bind_ls = [sub_landscape_from_graph(landscape_sub, G_bind.subgraph(c)) for c in comps(G_bind)]
    boundary_ls = [sub_landscape_from_graph(landscape_sub, G_boundary.subgraph(c)) for c in comps(G_boundary)]

    def largest(ls):
        ls = [x for x in ls if x is not None and x.graph.number_of_nodes() > 0]
        return max(ls, key=lambda x: x.graph.number_of_nodes()) if ls else None

    return {
        "L_sub_fold": largest(fold_ls),
        "L_sub_bind": largest(bind_ls),
        "L_sub_bound": largest(boundary_ls),
    }

# Build L_sub from THIS exact G_sub
L_sub = sub_landscape_from_graph(landscape, G_sub)
parts = partition_exact(L_sub, limiter_layer="limiter_index", eps=EPS)

L_sub_bind = parts["L_sub_bind"]
L_sub_fold = parts["L_sub_fold"]
L_sub_bound = parts["L_sub_bound"]

if any(x is None for x in [L_sub_bind, L_sub_fold, L_sub_bound]):
    raise RuntimeError("Could not build one of bind/fold/bound components.")

print(
    "Component sizes:",
    "bind=", L_sub_bind.graph.number_of_nodes(),
    "fold=", L_sub_fold.graph.number_of_nodes(),
    "boundary=", L_sub_bound.graph.number_of_nodes()
)

# ----------------------------
# 6) 3-panel plot reusing EXACT SAME pos + exact same limiter map
# ----------------------------
lim_map = {n: v for n, v in zip(node_list_sub, layer_on_subgraph("limiter_index"))}
scale = np.percentile(np.abs(np.array(list(lim_map.values()), dtype=float)), 95) + 1e-12
norm = Normalize(vmin=-1, vmax=1)

x_all = np.array([pos[n][0] for n in node_list_sub], dtype=float)
y_all = np.array([pos[n][1] for n in node_list_sub], dtype=float)
xmin, xmax = x_all.min(), x_all.max()
ymin, ymax = y_all.min(), y_all.max()

def draw_component(ax, comp, title):
    comp_nodes = set(comp.graph.nodes())
    ordered = [n for n in node_list_sub if n in comp_nodes]

    x = np.array([pos[n][0] for n in ordered], dtype=float)
    y = np.array([pos[n][1] for n in ordered], dtype=float)
    c = np.clip(np.array([lim_map[n] for n in ordered], dtype=float) / scale, -1, 1)

    ax.scatter(
        x, y, c=c, cmap="coolwarm_r", norm=norm,
        s=NODE_SIZE, edgecolors=NODE_LINE_COLOR, linewidths=NODE_LINE_WIDTH, zorder=2
    )
    # ax.set_title(f"{title}\n(n={len(ordered)})", fontsize=11)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

fig, axes = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)
draw_component(axes[0], L_sub_bind, "Binding-limited component")
draw_component(axes[1], L_sub_fold, "Folding-limited component")
draw_component(axes[2], L_sub_bound, "Boundary component")

sm = ScalarMappable(norm=norm, cmap="coolwarm_r")
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(["folding limited", "", "binding limited"])

plt.savefig('../figures/figure_4/SH3_boundaries_decomposed.pdf')
plt.show()

# Debug check: this should be ~0 now
lim_panel_map = {n: L_sub.fitness_layers["limiter_index"].to_scalar()[i] for i, n in enumerate(L_sub._node_order)}
common = set(lim_map) & set(lim_panel_map)
max_diff = max(abs(lim_map[n] - lim_panel_map[n]) for n in common)
print("max |oneoff - panel| over common nodes:", max_diff)

# %%
# ==== Single standalone plot: composite fitness normalized to [0,1], with z-order by fitness ====
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_composite_fitness_01_z(
    G_sub,
    node_list_sub,
    pos,
    layer_on_subgraph,
    *,
    node_size=8,
    node_line_width=0.25,
    node_line_color="black",
    edge_width=0.2,
    edge_alpha=0.0,
    fraction_nodes=1.0,
    seed=0,
    figsize=(3, 2),
    cmap="viridis",
    zorder_base=2.0,
    zorder_span=20.0,
):
    f = np.asarray(layer_on_subgraph("composite_fitness"), dtype=float)

    fmin, fmax = float(f.min()), float(f.max())
    f01 = (f - fmin) / (fmax - fmin + 1e-12)
    norm = Normalize(vmin=0.0, vmax=1.0)

    n = len(node_list_sub)
    k = max(1, int(np.ceil(np.clip(fraction_nodes, 0.0, 1.0) * n)))
    rng = np.random.default_rng(seed)
    keep_idx = np.arange(n) if k == n else rng.choice(n, size=k, replace=False)

    keep_nodes = [node_list_sub[i] for i in keep_idx]
    keep_set = set(keep_nodes)

    xs = np.array([pos[n][0] for n in keep_nodes], dtype=float)
    ys = np.array([pos[n][1] for n in keep_nodes], dtype=float)
    cs = f01[keep_idx]

    fig, ax = plt.subplots(figsize=figsize)

    if edge_alpha > 0:
        for u, v in G_sub.edges():
            if u in keep_set and v in keep_set:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                ax.plot([x0, x1], [y0, y1], color="black", lw=edge_width, alpha=edge_alpha, zorder=1)

    # Draw low fitness first, high fitness last
    order = np.argsort(cs)
    for i in order:
        z = zorder_base + zorder_span * cs[i]
        ax.scatter(
            [xs[i]], [ys[i]],
            c=[cs[i]], cmap=cmap, norm=norm,
            s=node_size,
            edgecolors=node_line_color,
            linewidths=node_line_width,
            zorder=z
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fitness")

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/figure_4/SH3_0.35_BFS_0.1_composite_fitness.pdf')
    plt.show()

plot_composite_fitness_01_z(
    G_sub, node_list_sub, pos, layer_on_subgraph,
    node_size=NODE_SIZE,
    node_line_width=NODE_LINE_WIDTH,
    node_line_color=NODE_LINE_COLOR,
    edge_width=0.25,
    edge_alpha=0.0,
    fraction_nodes=1,
    seed=0,
    figsize=(3.5, 3),
    cmap="viridis",
    zorder_base=2.0,
    zorder_span=20.0,
)

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ----------------------------
# Config
# ----------------------------
fold_layer = "latent_fold"
bind_layer = "latent_bind"
obs_bind_candidates = ["binding_fitness", "binding_observed", "binding", "Binding", "composite_fitness"]

DROP_BIND_ZERO = True
BIND_ZERO_EPS = 1e-10
MAX_POINTS = 15000
rng = np.random.default_rng(0)

# ----------------------------
# Layer getter
# ----------------------------
if "layer_on_subgraph" in globals():
    get_layer = layer_on_subgraph
    layer_keys = set(landscape.fitness_layers.keys())
elif "L_sub" in globals():
    idx_map = {n: i for i, n in enumerate(L_sub._node_order)}
    def get_layer(k):
        vals = np.asarray(L_sub.fitness_layers[k].to_scalar(), dtype=float)
        return np.array([vals[idx_map[n]] for n in L_sub._node_order], dtype=float)
    layer_keys = set(L_sub.fitness_layers.keys())
else:
    idx_map = {n: i for i, n in enumerate(landscape._node_order)}
    def get_layer(k):
        vals = np.asarray(landscape.fitness_layers[k].to_scalar(), dtype=float)
        return np.array([vals[idx_map[n]] for n in landscape._node_order], dtype=float)
    layer_keys = set(landscape.fitness_layers.keys())

def first_existing(cands):
    for c in cands:
        if c in layer_keys:
            return c
    return None

obs_bind_layer = first_existing(obs_bind_candidates)
if obs_bind_layer is None:
    raise KeyError(f"No observed binding layer found in {obs_bind_candidates}")

# ----------------------------
# Pull arrays
# ----------------------------
fold_raw = np.asarray(get_layer(fold_layer), dtype=float).ravel()
bind_raw = np.asarray(get_layer(bind_layer), dtype=float).ravel()
fit_raw  = np.asarray(get_layer(obs_bind_layer), dtype=float).ravel()

m = np.isfinite(fold_raw) & np.isfinite(bind_raw) & np.isfinite(fit_raw)
fold_raw, bind_raw, fit_raw = fold_raw[m], bind_raw[m], fit_raw[m]

# ----------------------------
# Diagnose / optional drop bind==0 line
# ----------------------------
is_bind_zero = np.isclose(bind_raw, 0.0, atol=BIND_ZERO_EPS)
print(f"latent_bind ~ 0 count: {is_bind_zero.sum()} / {len(bind_raw)} ({is_bind_zero.mean():.2%})")

keep = ~is_bind_zero if DROP_BIND_ZERO else np.ones_like(is_bind_zero, dtype=bool)
fold_raw, bind_raw, fit_raw = fold_raw[keep], bind_raw[keep], fit_raw[keep]
print(f"Points plotted: {len(fold_raw)} (DROP_BIND_ZERO={DROP_BIND_ZERO})")

# Downsample for speed
if len(fold_raw) > MAX_POINTS:
    idx = rng.choice(len(fold_raw), size=MAX_POINTS, replace=False)
    fold_raw, bind_raw, fit_raw = fold_raw[idx], bind_raw[idx], fit_raw[idx]

# ----------------------------
# Normalize:
# latent axes to [0,1] with 1 = most negative (free energy)
# fitness to [0,1]
# ----------------------------
def norm_free_energy(v):
    vmin, vmax = float(v.min()), float(v.max())
    return (vmax - v) / (vmax - vmin + 1e-12)  # most negative -> 1

fold_n = norm_free_energy(fold_raw)
bind_n = norm_free_energy(bind_raw)
fit_n  = (fit_raw - fit_raw.min()) / (fit_raw.max() - fit_raw.min() + 1e-12)

# Plot axes: X=Latent binding, Y=Latent folding, Z=Norm. Fitness
x_plot = bind_n
y_plot = fold_n
z_plot = fit_n

# ----------------------------
# Plot
# ----------------------------
fig = plt.figure(figsize=(3.0, 2.6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    x_plot, y_plot, z_plot,
    c=z_plot, cmap="viridis", norm=colors.Normalize(0, 1),
    s=6, alpha=0.95, linewidths=0
)

ax.view_init(elev=26, azim=160)

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_facecolor((0.92, 0.92, 0.92, 1.0))
    axis.pane.set_edgecolor((0.65, 0.65, 0.65, 1.0))
ax.grid(False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

ax.set_xlabel("Latent binding")
ax.set_ylabel("Latent folding")
ax.set_zlabel("Norm. Fitness")


plt.tight_layout()
plt.savefig('../figures/figure_4/latent_contribuions_fitness.pdf')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx

# -----------------------------
# 0) Rebuild CONSISTENT objects from current G_sub
# -----------------------------
L_sub = sub_landscape_from_graph(landscape, G_sub)
parts = partition_exact(L_sub, limiter_layer="limiter_index", eps=EPS)

L_sub_bind = parts["L_sub_bind"]
L_sub_fold = parts["L_sub_fold"]
L_sub_bound = parts["L_sub_bound"]

if any(x is None for x in [L_sub_bind, L_sub_fold, L_sub_bound]):
    raise RuntimeError("One of bind/fold/bound is empty for this EPS/BFS selection.")

node_order = list(L_sub._node_order)
node_set = set(node_order)

# strict sanity check
for title, comp in [("bind", L_sub_bind), ("fold", L_sub_fold), ("boundary", L_sub_bound)]:
    extra = set(comp.graph.nodes()) - node_set
    if extra:
        raise RuntimeError(f"{title} component still mismatched ({len(extra)} nodes not in L_sub). Restart kernel and rerun source cell.")

# -----------------------------
# 1) Ensure pos matches L_sub nodes
# -----------------------------
if not set(node_order).issubset(set(pos.keys())):
    # try string-key remap first
    if set(map(str, node_order)).issubset(set(pos.keys())):
        pos = {n: pos[str(n)] for n in node_order}
    else:
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            pos = graphviz_layout(L_sub.graph, prog="sfdp")
        except Exception:
            pos = nx.spring_layout(L_sub.graph, seed=LAYOUT_SEED)

# shared bounds
x_all = np.array([pos[n][0] for n in node_order], dtype=float)
y_all = np.array([pos[n][1] for n in node_order], dtype=float)
xmin, xmax = x_all.min(), x_all.max()
ymin, ymax = y_all.min(), y_all.max()

# shared node fitness map (from L_sub, not globals)
f = np.asarray(L_sub.fitness_layers["composite_fitness"].to_scalar(), dtype=float)
node_to_f = {n: f[i] for i, n in enumerate(node_order)}

panel_defs = [
    ("Binding-limited component", L_sub_bind),
    ("Folding-limited component", L_sub_fold),
    ("Boundary component", L_sub_bound),
]

print("Component sizes:",
      L_sub_bind.graph.number_of_nodes(),
      L_sub_fold.graph.number_of_nodes(),
      L_sub_bound.graph.number_of_nodes())

# -----------------------------
# 2) Figure A: nodes by composite fitness
# -----------------------------
f_lo, f_hi = np.percentile(f, [2, 98])
if f_hi <= f_lo:
    f_lo, f_hi = float(np.min(f)), float(np.max(f) + 1e-12)
norm_f = Normalize(vmin=f_lo, vmax=f_hi)

figA, axesA = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)

for ax, (title, comp) in zip(axesA, panel_defs):
    comp_nodes = [n for n in node_order if n in comp.graph]
    x = np.array([pos[n][0] for n in comp_nodes], dtype=float)
    y = np.array([pos[n][1] for n in comp_nodes], dtype=float)
    c = np.array([node_to_f[n] for n in comp_nodes], dtype=float)

    ax.scatter(
        x, y, c=c, cmap="viridis", norm=norm_f,
        s=NODE_SIZE, edgecolors=NODE_LINE_COLOR, linewidths=NODE_LINE_WIDTH, zorder=2
    )
    ax.set_title(f"{title}\n(n={len(comp_nodes)})", fontsize=11)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

smA = ScalarMappable(norm=norm_f, cmap="viridis")
smA.set_array([])
cbarA = figA.colorbar(smA, ax=axesA, fraction=0.025, pad=0.02)
cbarA.set_label("Composite fitness")
plt.savefig('../figures/si_figures/si_figure_fitness_partitioning/fitness_partitioning_node_norm_fitness.pdf')
plt.show()

# -----------------------------
# 3) Figure B: edges only by squared gradient in composite fitness
# -----------------------------
all_e = []
panel_edges = []
for _, comp in panel_defs:
    e_list = []
    for u, v in comp.graph.edges():
        e = (node_to_f[u] - node_to_f[v])**2
        e_list.append((u, v, e))
        all_e.append(e)
    panel_edges.append(e_list)

all_e = np.asarray(all_e, dtype=float)
e_lo, e_hi = np.percentile(all_e, [2, 98])
if e_hi <= e_lo:
    e_lo, e_hi = float(np.min(all_e)), float(np.max(all_e) + 1e-12)
norm_e = Normalize(vmin=e_lo, vmax=e_hi)
cmap_e = plt.get_cmap("viridis")

figB, axesB = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)

for ax, (title, _), e_list in zip(axesB, panel_defs, panel_edges):
    for u, v, e in sorted(e_list, key=lambda t: t[2]):  # low->high
        ee = np.clip(e, e_lo, e_hi)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1],
                color=cmap_e(norm_e(ee)),
                lw=0.45, alpha=0.95, zorder=1 + 10 * norm_e(ee),
                solid_capstyle="round")
    ax.set_title(f"{title}\n(edges={len(e_list)})", fontsize=11)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

smB = ScalarMappable(norm=norm_e, cmap="viridis")
smB.set_array([])
cbarB = figB.colorbar(smB, ax=axesB, fraction=0.025, pad=0.02)
cbarB.set_label(r"$(\Delta\,\mathrm{composite\ fitness})^2$")
# figB.suptitle("Edge energy on regime-partitioned subsets (edges only)", y=1.02)
plt.savefig('../figures/si_figures/si_figure_fitness_partitioning/fitness_partitioning_edge_energy.pdf')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ============================================================
# Quiver-equivalent scatter in trait space
# Uses SAME objects + SAME edge sets + SAME orientation rule:
#   orient each edge low limiter -> high limiter
# Then plot (Δfold, Δbind) per edge.
# ============================================================

C_BLUE  = "#2B5EA7"
C_RED   = "#D64545"
C_GOLD  = "#E8A838"
C_MUTED = "#888888"

# ---- knobs ----
WITHIN_USE_ABS = False   # False = match quiver directional info
POINT_SIZE = 8
L_Q = 0.995              # axis clipping quantile
MAG_Q = 0.5            # alpha scaling quantile

# ---- required objects ----
needed = ["L_sub", "L_sub_bind", "L_sub_fold", "L_sub_bound"]
missing = [k for k in needed if k not in globals() or globals()[k] is None]
if missing:
    raise RuntimeError(f"Missing required objects: {missing}")

node_order = list(L_sub._node_order)
node_set = set(node_order)

# Rebuild stale components if needed
if (
    not set(L_sub_bind.graph.nodes()).issubset(node_set)
    or not set(L_sub_fold.graph.nodes()).issubset(node_set)
    or not set(L_sub_bound.graph.nodes()).issubset(node_set)
):
    if "partition_exact" not in globals():
        raise RuntimeError("Components are stale and partition_exact is unavailable.")
    eps_val = float(globals().get("EPS", 0.0))
    parts = partition_exact(L_sub, limiter_layer="limiter_index", eps=eps_val)
    L_sub_bind = parts["L_sub_bind"]
    L_sub_fold = parts["L_sub_fold"]
    L_sub_bound = parts["L_sub_bound"]

# ---- layers on canonical L_sub order ----
node_to_i = {n: i for i, n in enumerate(node_order)}
F = np.asarray(L_sub.fitness_layers["latent_fold"].to_scalar(), dtype=float).ravel()
B = np.asarray(L_sub.fitness_layers["latent_bind"].to_scalar(), dtype=float).ravel()
S = np.asarray(L_sub.fitness_layers["limiter_index"].to_scalar(), dtype=float).ravel()
node_to_s = {n: float(S[node_to_i[n]]) for n in node_order}

def _clean_edges(edge_iter):
    out = []
    for e in edge_iter:
        u, v = e[:2]
        if (u in node_set) and (v in node_set):
            out.append((u, v))
    return out

def build_records(edges, use_abs=False):
    rec = []
    for u0, v0 in edges:
        u, v = u0, v0
        # SAME orientation as quiver: low limiter -> high limiter
        if node_to_s[u] > node_to_s[v]:
            u, v = v, u

        iu, iv = node_to_i[u], node_to_i[v]
        df = float(F[iv] - F[iu])
        db = float(B[iv] - B[iu])

        if use_abs:
            df = abs(df)
            db = abs(db)

        mag = (df * df + db * db) ** 0.5
        fs = abs(df) / (abs(df) + abs(db) + 1e-12)
        rec.append({"d_fold": df, "d_bind": db, "mag": mag, "fold_share": fs})
    return rec

bind_edges  = _clean_edges(L_sub_bind.graph.edges())
fold_edges  = _clean_edges(L_sub_fold.graph.edges())
bound_edges = _clean_edges(L_sub_bound.graph.edges())

bind_list  = build_records(bind_edges, use_abs=WITHIN_USE_ABS)
fold_list  = build_records(fold_edges, use_abs=WITHIN_USE_ABS)
bound_list = build_records(bound_edges, use_abs=False)  # boundary always signed

if any(len(x) == 0 for x in [bind_list, fold_list, bound_list]):
    print("Warning: at least one panel has zero edges.")

# shared axis / alpha scales
all_df = np.concatenate([np.array([r["d_fold"] for r in lst], float) for lst in [bind_list, fold_list, bound_list] if len(lst)])
all_db = np.concatenate([np.array([r["d_bind"] for r in lst], float) for lst in [bind_list, fold_list, bound_list] if len(lst)])
all_mag = np.concatenate([np.array([r["mag"] for r in lst], float) for lst in [bind_list, fold_list, bound_list] if len(lst)])

if all_df.size == 0:
    raise RuntimeError("No edges to plot.")

L = np.quantile(np.maximum(np.abs(all_df), np.abs(all_db)), L_Q) * 1.05
L = max(float(L), 1e-6)
mag_clip = max(float(np.quantile(all_mag, MAG_Q)), 1e-12)

def panel(ax, edge_list, title, color, show_ylabel=True):
    df = np.array([r["d_fold"] for r in edge_list], dtype=float)
    db = np.array([r["d_bind"] for r in edge_list], dtype=float)
    fs = np.array([r["fold_share"] for r in edge_list], dtype=float)
    mg = np.array([r["mag"] for r in edge_list], dtype=float)

    # alpha scaled by magnitude so strong edges stand out (more quiver-like)
    a = 0.05 + 0.85 * np.clip(mg / mag_clip, 0.0, 1.0)

    ax.scatter(df, db, c=fs, cmap="coolwarm", vmin=0, vmax=1,
               s=POINT_SIZE, alpha=a, edgecolors="none", zorder=3)

    ax.axhline(0, color=C_MUTED, lw=0.5, alpha=0.35)
    ax.axvline(0, color=C_MUTED, lw=0.5, alpha=0.35)
    ax.plot([-L, L], [-L, L], color=C_MUTED, lw=0.5, ls=":", alpha=0.30)
    ax.plot([-L, L], [L, -L], color=C_MUTED, lw=0.5, ls=":", alpha=0.30)

    mf = df.mean() if len(df) else 0.0
    mb = db.mean() if len(db) else 0.0
    ax.plot(mf, mb, marker="+", markersize=12, markeredgewidth=2, color=color, zorder=5)
    ax.arrow(0, 0, mf, mb, color=color, lw=1.2, head_width=0.03*L, length_includes_head=True, zorder=4)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δ folding" if not WITHIN_USE_ABS else "|Δ folding|")
    if show_ylabel:
        ax.set_ylabel("Δ binding" if not WITHIN_USE_ABS else "|Δ binding|")
    ax.set_title(title, fontsize=10, pad=6, color=color, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pct_both = 100.0 * np.mean((np.abs(df) > 1e-12) & (np.abs(db) > 1e-12))
    ax.text(0.97, 0.03, f"n={len(edge_list)} | both≠0: {pct_both:.1f}%",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom", color=C_MUTED)

fig, axes = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)

panel(
    axes[0], bind_list,
    f"Binding-limited edges",
    "black", show_ylabel=True
)
panel(
    axes[1], fold_list,
    f"Folding-limited edges",
    "black", show_ylabel=False
)
panel(
    axes[2], bound_list,
    "Boundary edges",
    "black", show_ylabel=False
)

sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="coolwarm")
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.024, pad=0.02)
cbar.set_label("|Δfold| / (|Δfold| + |Δbind|)")

# fig.suptitle("Latent edge contributions by regime (scatter; quiver-equivalent encoding)", y=1.02)
plt.savefig('../figures/si_figures/si_figure_delta_latent_vector_edges/edge_vector_scatterplot.pdf')
plt.show()

print("Edges used:",
      "bind=", len(bind_list),
      "fold=", len(fold_list),
      "boundary=", len(bound_list))


### As quiver plot on graph
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ============================================================
# Quiver version of your scatter analysis (same objects, same quantities)
# - Binding/Folding panels: ABS deltas
# - Boundary panel: SIGNED deltas, oriented fold-limited -> bind-limited
# - One arrow per EDGE (not node-averaged)
# ============================================================

# ---- style knobs ----
arrow_scale = 0.06      # global arrow-size multiplier (relative to layout extent)
arrow_width = 0.0028
arrow_alpha = 0.90
draw_background = True
background_alpha = 0.05
background_lw = 0.20

C_BLUE  = "#2B5EA7"
C_RED   = "#D64545"
C_GOLD  = "#E8A838"

# ---- required objects ----
needed = ["L_sub", "L_sub_bind", "L_sub_fold", "L_sub_bound"]
missing = [k for k in needed if k not in globals() or globals()[k] is None]
if missing:
    raise RuntimeError(f"Missing required objects: {missing}")

node_order = list(L_sub._node_order)
node_set = set(node_order)

# Rebuild stale components if needed
if (
    not set(L_sub_bind.graph.nodes()).issubset(node_set)
    or not set(L_sub_fold.graph.nodes()).issubset(node_set)
    or not set(L_sub_bound.graph.nodes()).issubset(node_set)
):
    if "partition_exact" not in globals():
        raise RuntimeError("Components are stale and partition_exact is unavailable.")
    eps_val = float(globals().get("EPS", 0.0))
    parts = partition_exact(L_sub, limiter_layer="limiter_index", eps=eps_val)
    L_sub_bind = parts["L_sub_bind"]
    L_sub_fold = parts["L_sub_fold"]
    L_sub_bound = parts["L_sub_bound"]

# ---- layout: reuse existing pos if complete; otherwise compute on L_sub.graph ----
if ("pos" not in globals()) or any(n not in pos for n in node_order):
    layout_seed = int(globals().get("LAYOUT_SEED", 0))
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(L_sub.graph, prog="sfdp")
    except Exception:
        pos = nx.spring_layout(L_sub.graph, seed=layout_seed)

# ---- pull layers on L_sub node order ----
node_to_i = {n: i for i, n in enumerate(node_order)}
F = np.asarray(L_sub.fitness_layers["latent_fold"].to_scalar(), dtype=float).ravel()
B = np.asarray(L_sub.fitness_layers["latent_bind"].to_scalar(), dtype=float).ravel()
S = np.asarray(L_sub.fitness_layers["limiter_index"].to_scalar(), dtype=float).ravel()
node_to_s = {n: float(S[node_to_i[n]]) for n in node_order}

def _clean_edges(edge_iter):
    out = []
    for e in edge_iter:
        u, v = e[:2]
        if (u in node_set) and (v in node_set):
            out.append((u, v))
    return out

def build_within_abs_records(edges):
    rec = []
    for u, v in edges:
        iu, iv = node_to_i[u], node_to_i[v]
        rec.append({
            "u": u, "v": v,
            "d_fold": abs(float(F[iv] - F[iu])),
            "d_bind": abs(float(B[iv] - B[iu])),
        })
    return rec

def build_boundary_signed_records(edges):
    rec = []
    for u0, v0 in edges:
        u, v = u0, v0
        # Orient fold-limited -> bind-limited (low limiter -> high limiter)
        if node_to_s[u] > node_to_s[v]:
            u, v = v, u
        iu, iv = node_to_i[u], node_to_i[v]
        rec.append({
            "u": u, "v": v,
            "d_fold": float(F[iv] - F[iu]),
            "d_bind": float(B[iv] - B[iu]),
        })
    return rec

bind_edges  = _clean_edges(L_sub_bind.graph.edges())
fold_edges  = _clean_edges(L_sub_fold.graph.edges())
bound_edges = _clean_edges(L_sub_bound.graph.edges())

bind_list  = build_within_abs_records(bind_edges)
fold_list  = build_within_abs_records(fold_edges)
bound_list = build_boundary_signed_records(bound_edges)

if any(len(x) == 0 for x in [bind_list, fold_list, bound_list]):
    print("Warning: at least one panel has zero edges.")

def fold_share(df, db):
    return np.abs(df) / (np.abs(df) + np.abs(db) + 1e-12)

# shared magnitude scaling for arrow lengths
mag_all = []
for lst in [bind_list, fold_list, bound_list]:
    for r in lst:
        mag_all.append((r["d_fold"]**2 + r["d_bind"]**2) ** 0.5)
if len(mag_all) == 0:
    raise RuntimeError("No edges available for plotting.")
mag_all = np.asarray(mag_all, dtype=float)
mag_clip = max(float(np.quantile(mag_all, 0.995)), 1e-12)

x_all = np.array([pos[n][0] for n in node_order], dtype=float)
y_all = np.array([pos[n][1] for n in node_order], dtype=float)
extent = max(x_all.max() - x_all.min(), y_all.max() - y_all.min())

norm = Normalize(vmin=0.0, vmax=1.0)

def records_to_quiver_arrays(records):
    X, Y, U, V, C = [], [], [], [], []
    for r in records:
        u, v = r["u"], r["v"]
        df, db = r["d_fold"], r["d_bind"]
        mag = (df*df + db*db) ** 0.5
        if mag < 1e-12:
            continue

        # anchor at edge midpoint in graph layout
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        xm, ym = 0.5*(x0+x1), 0.5*(y0+y1)

        # arrow direction from (d_fold, d_bind), same as scatter axes
        dirx, diry = df / mag, db / mag
        L = arrow_scale * extent * min(mag / mag_clip, 1.0)

        X.append(xm); Y.append(ym)
        U.append(dirx * L); V.append(diry * L)
        C.append(fold_share(df, db))

    return (
        np.asarray(X, float), np.asarray(Y, float),
        np.asarray(U, float), np.asarray(V, float),
        np.asarray(C, float)
    )

def panel_quiver(ax, records, edges_subset, title, title_color):
    if draw_background:
        for u, v in edges_subset:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color="black", alpha=background_alpha, lw=background_lw, zorder=1)

    X, Y, U, V, C = records_to_quiver_arrays(records)
    q = None
    if len(X) > 0:
        q = ax.quiver(
            X, Y, U, V, C,
            cmap="coolwarm", norm=norm,
            angles="xy", scale_units="xy", scale=1.0,
            width=arrow_width, alpha=arrow_alpha, zorder=3
        )
        ax.scatter(X, Y, s=3, c="black", alpha=0.10, zorder=2)
    else:
        ax.text(0.5, 0.5, "No edges", transform=ax.transAxes, ha="center", va="center")

    ax.set_title(f"{title}", color=title_color, fontsize=11, pad=6)
    ax.set_axis_off()
    return q

fig, axes = plt.subplots(1, 3, figsize=(8, 3))

qA = panel_quiver(axes[0], bind_list, bind_edges,  "", C_BLUE)
qB = panel_quiver(axes[1], fold_list, fold_edges,  "", C_RED)
qC = panel_quiver(axes[2], bound_list, bound_edges, "", C_GOLD)

# q_for_cbar = qC if qC is not None else (qB if qB is not None else qA)
# if q_for_cbar is not None:
#     cbar = fig.colorbar(q_for_cbar, ax=axes, fraction=0.046, pad=0.04, orientation="horizontal")
#     cbar.set_label("Fold share = |Δfold| / (|Δfold| + |Δbind|)")

fig.tight_layout()
plt.show()

print("Edges used:",
      "bind=", len(bind_list),
      "fold=", len(fold_list),
      "boundary=", len(bound_list))

# %%
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

# SEARCH_TAG: LATENT_RESAMPLING_ARTIFACT_EXPORT
latent_resample_outdir = Path('../figures/si_figures/si_figure_latent_resampling')
latent_resample_outdir.mkdir(parents=True, exist_ok=True)
latent_resample_replicates_csv = latent_resample_outdir / 'latent_resampling_tmap_replicates.csv'
latent_resample_summary_csv = latent_resample_outdir / 'latent_resampling_tmap_summary.csv'
latent_resample_pdf = latent_resample_outdir / 'latent_resampling_tmap_robustness.pdf'
latent_resample_png = latent_resample_outdir / 'latent_resampling_tmap_robustness.png'
latent_resample_tmap_df.to_csv(latent_resample_replicates_csv, index=False)
plt.tight_layout()
fig.savefig(latent_resample_pdf, bbox_inches='tight')
fig.savefig(latent_resample_png, dpi=300, bbox_inches='tight')
plt.show()

latent_resample_tmap_summary = latent_resample_tmap_df[plot_cols].agg(["mean", "std", "count"]).T
latent_resample_tmap_summary.to_csv(latent_resample_summary_csv)
display(latent_resample_tmap_summary)
print(f'Latent resampling replicate table written to: {latent_resample_replicates_csv}')
print(f'Latent resampling summary table written to: {latent_resample_summary_csv}')
print(f'Latent resampling figure written to: {latent_resample_pdf}')
print(f'Latent resampling figure written to: {latent_resample_png}')

# %% [markdown]
# ## KRAS/RBD boundary enrichment summary
#
# Copied and focused from `the copied source/figure_notebooks_rev snapshot/Figure_4_boundary_enrichment_summary.ipynb`.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULT_SPECS =[
    {
        "slug": "kras_darpin_k27",
        "system_name": "KRAS DARPin K27",
        "plot_label": "KRAS\nDARPin K27"
    },
    {
        "slug": "kras_darpin_k55",
        "system_name": "KRAS DARPin K55",
        "plot_label": "KRAS\nDARPin K55"
    },
    {
        "slug": "kras_pik3cg",
        "system_name": "KRAS PIK3CG",
        "plot_label": "KRAS\nPIK3CG"
    },
    {
        "slug": "kras_raf1",
        "system_name": "KRAS RAF1",
        "plot_label": "KRAS\nRAF1"
    },
    {
        "slug": "kras_ralgds",
        "system_name": "KRAS RALGDS",
        "plot_label": "KRAS\nRALGDS"
    },
    {
        "slug": "kras_sos1",
        "system_name": "KRAS SOS1",
        "plot_label": "KRAS\nSOS1"
    },
    {
        "slug": "lycov",
        "system_name": "LYCoV",
        "plot_label": "LYCoV"
    }
]

def resolve_boundary_enrichment_output_dir():
    outdir = Path("../figures/figure_4/boundary_enrichment_results")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

result_dir = resolve_boundary_enrichment_output_dir()

def boundary_enrichment_payload_path(slug):
    candidates = [
        processed_dir
        / "experimental_boundaries"
        / "kras_boundary_enrichment_results"
        / f"{slug}_boundary_enrichment.json",
        processed_dir
        / "experimental_boundaries"
        / f"{slug}_boundary_enrichment.json",
        result_dir / f"{slug}_boundary_enrichment.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

missing = [
    boundary_enrichment_payload_path(spec["slug"])
    for spec in RESULT_SPECS
    if not boundary_enrichment_payload_path(spec["slug"]).exists()
]
if missing:
    missing_list = "\n".join(str(path) for path in missing)
    raise FileNotFoundError(
        "Missing processed boundary enrichment result files. Re-run the Figure 6 KRAS and LyCoV experiments first:\n"
        + missing_list
    )

records = []
for order, spec in enumerate(RESULT_SPECS):
    payload = json.loads(boundary_enrichment_payload_path(spec["slug"]).read_text())
    records.append(
        {
            "sort_order": order,
            "system_slug": spec["slug"],
            "system_name": spec["system_name"],
            "plot_label": spec["plot_label"],
            "observed_enrichment": float(payload["observed"]["energy_share_enrichment"]),
            "null_p_enrichment": float(payload["null_same_size_random_edge_subsets"]["p_enrichment"]),
            "n_perm": int(payload["null_same_size_random_edge_subsets"]["n_perm"]),
        }
    )

summary_df = pd.DataFrame.from_records(records).sort_values("sort_order").reset_index(drop=True)
summary_df

# %%
GREY = "#d3d3d3"

def format_p_value(p):
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"

x = np.arange(len(summary_df))
y_max = float(summary_df["observed_enrichment"].max()) * 1.18

fig, ax = plt.subplots(figsize=(4, 2.5))

ax.bar(
    x,
    summary_df["observed_enrichment"],
    width=0.65,
    color="#d3d3d3",
    edgecolor="black",
    linewidth=0.8,
)

for xi, y, p in zip(x, summary_df["observed_enrichment"], summary_df["null_p_enrichment"]):
    ax.text(
        xi,
        y + 0.03 * y_max,
        format_p_value(float(p)),
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
ax.set_xticks(x, summary_df["plot_label"])
ax.set_ylabel("Energy enrichment (fold)")
ax.set_xlabel("")
ax.set_title("")
ax.set_ylim(0, y_max)
ax.set_axisbelow(True)
ax.grid(True, linestyle="--", color=GREY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()
plt.xticks(rotation=45, ha="right")

png_path = result_dir / "figure_4_boundary_enrichment_summary.png"
pdf_path = result_dir / "figure_4_boundary_enrichment_summary.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print(f"Saved summary plot to {png_path}")
print(f"Saved summary plot to {pdf_path}")
plt.show()
