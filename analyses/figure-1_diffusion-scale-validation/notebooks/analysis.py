# Exported from analysis.ipynb

# %%
from pathlib import Path
import os
import sys
import types

analysis_dir = Path.cwd().resolve().parent
scripts_dir = analysis_dir.parents[1] / "scripts"
sys.path.insert(0, str(scripts_dir))
from paper_runtime import find_project_root, _patch_matplotlib_savefig
project_root = find_project_root(analysis_dir)
compat_data = analysis_dir / "data_files"
compat_alisim = analysis_dir / "alisim_results"
for compat_path, target_path in [
    (compat_data, project_root / "data" / "source_datasets"),
    (compat_alisim, project_root / "data" / "alisim_results"),
]:
    if compat_path.is_symlink() and not compat_path.exists():
        compat_path.unlink()
    if not compat_path.exists():
        compat_path.symlink_to(target_path, target_is_directory=True)
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
# ## Main Figure 1 synthetic and sparse validation panels
#
# Copied and focused from `the copied source/figure_notebooks_rev snapshot/Figure_1.ipynb`.

# %%
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

# %%
# Load promoted outputs from the Figure 1 validation experiments.
import pickle
from collections import defaultdict

_diffusion_dir = processed_dir / "diffusion_scale_validation"
_nk_tmap_csv = _diffusion_dir / "nk_tmap_results.csv"
if _nk_tmap_csv.exists():
    nk_tmap_df = pd.read_csv(_nk_tmap_csv)
    replicate_dict = defaultdict(dict)
    for row in nk_tmap_df.itertuples(index=False):
        replicate_dict[int(row.replicate)][(int(row.N), int(row.K))] = {"t_map": float(row.t_map)}

_nk_eig_csv = _diffusion_dir / "nk_eigenmode_tmap.csv"
if _nk_eig_csv.exists():
    df_eig = pd.read_csv(_nk_eig_csv)

_alisim_pkl = _diffusion_dir / "alisim_tmap_by_rep_and_eig.pkl"
if _alisim_pkl.exists():
    with _alisim_pkl.open("rb") as handle:
        results = pickle.load(handle)

_alisim_csv = _diffusion_dir / "alisim_tmap_table.csv"
if _alisim_csv.exists():
    alisim_tmap_df = pd.read_csv(_alisim_csv)

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
    return (np.log10(t_hi) - np.log10(t_lo)) <= max_ci_orders

# %%
# replicate_dict: {rep: {(N, K): {'t_map': ... , ...}, ...}, ...}
# Restructure to: restructured[rep][N] -> list of (K, t_map)

restructured = defaultdict(lambda: defaultdict(list))
all_n_values = set()
all_k_values = set()

for rep, tmap_dict in replicate_dict.items():
    for (n, k), res in tmap_dict.items():
        t = res["t_map"]
        restructured[rep][n].append((k, t))
        all_n_values.add(n)
        all_k_values.add(k)

sorted_n = sorted(all_n_values)
sorted_k = sorted(all_k_values)

fig, ax = plt.subplots(figsize=(3, 2.75))
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=min(sorted_n), vmax=max(sorted_n))
colors = {n: cmap(norm(n)) for n in sorted_n}

for rep_data in restructured.values():
    for n_value, points in rep_data.items():
        points.sort(key=lambda x: x[0])  # sort by K
        k_vals = [k for k, _ in points]
        tmap_vals = [t for _, t in points]

        ax.plot(
            k_vals,
            tmap_vals,
            color=colors[n_value],
            alpha=0.6,
            marker="o",
            markersize=4,
            linestyle="-",
        )

ax.set_xlabel("K")
ax.set_ylabel(r"$t_{MAP}$")
ax.set_yscale("log")
ax.set_xticks(sorted_k[::2])
ax.grid(True, which="both", ls="--", c="0.7")

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("N")
cbar.set_ticks(sorted_n)

plt.tight_layout()
plt.savefig("../figures/figure_1/tmap_vs_k.pdf")
plt.show()

# %%
# Alternative view of the same NK experiment: N is the independent variable,
# and curves are grouped and coloured by K.
restructured_by_k = defaultdict(lambda: defaultdict(list))
all_n_values = set()
all_k_values = set()

for rep, tmap_dict in replicate_dict.items():
    for (n, k), res in tmap_dict.items():
        t = res["t_map"]
        restructured_by_k[rep][k].append((n, t))
        all_n_values.add(n)
        all_k_values.add(k)

sorted_n = sorted(all_n_values)
sorted_k = sorted(all_k_values)

fig, ax = plt.subplots(figsize=(3, 2.75))
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=min(sorted_k), vmax=max(sorted_k))
colors = {k: cmap(norm(k)) for k in sorted_k}

for rep_data in restructured_by_k.values():
    for k_value, points in rep_data.items():
        points.sort(key=lambda x: x[0])  # sort by N
        n_vals = [n for n, _ in points]
        tmap_vals = [t for _, t in points]

        ax.plot(
            n_vals,
            tmap_vals,
            color=colors[k_value],
            alpha=0.6,
            marker="o",
            markersize=4,
            linestyle="-",
        )

ax.set_xlabel("N")
ax.set_ylabel(r"$t_{MAP}$")
ax.set_yscale("log")
ax.set_xticks(sorted_n)
ax.grid(True, which="both", ls="--", c="0.7")

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("K")
cbar.set_ticks(sorted_k)

plt.tight_layout()
plt.savefig("../figures/figure_1/tmap_vs_n_colored_by_k.pdf")
plt.show()

# %%
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

# %%
# Init list to store results
gft_results = {}

# Range of N variables to consider
n_range = list(range(4, 11))

for _,n_param in tqdm(enumerate(n_range)):

    # Construct NK landscape with dummy K
    nk = fl.models.nk.create_nk_binary_landscape(N=n_param, K=0, seed=0)

    # Perform Graph Fourier Transform on K signal
    _, _, coefficients = fl.transforms.graph_fourier.graph_fourier_transform(nk)

    # Store everything you need for analysis
    gft_results[n_param] = coefficients

# %%
xmax = 0.25

fig, ax = plt.subplots(figsize=(2.35, 2.75))

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(n_range)))  # evenly spaced colours

for color, n_param in zip(colors, n_range):
    c = np.abs(gft_results[n_param]).astype(float)
    E = c / c.sum()
    x = np.arange(len(E)) / (len(E) - 1)
    cdf = np.cumsum(E)

    ax.plot(x, cdf, lw=1.5, color=color)

ax.set_ylim(0.6, 1)
ax.set_xlim(0, xmax)
ax.grid(True, ls="--", c="0.85")
ax.set_ylabel("Cumulative energy")
ax.set_xlabel("Norm. eigenvector index")


plt.tight_layout()
plt.savefig("../figures/figure_1/K0_GFT_cdf.pdf")
plt.show()

# %%
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
        eig_indices,
        tmap_values,
        color="black",
        alpha=0.25,
        marker="o",
        markersize=0,
        linestyle="-",
        linewidth=1,
    )

ax.set_xlabel("Laplacian eigenvector index")
ax.set_ylabel(r"$t_{MAP}$")
ax.set_yscale("log")
ax.set_xticks(range(0, max_eig + 1, 10))
ax.set_xticklabels([str(k) for k in range(0, max_eig + 1, 10)])
ax.grid(True, which="both", ls="--", c="0.7")

plt.tight_layout()
plt.savefig("../figures/figure_1/tmap_vs_laplacian_eigenvector_index.pdf")
plt.show()

# %%
if "landscape" not in globals():
    print("Skipping AliSim example graph panel; the sparse phylogenetic validation experiment promotes tMAP tables, not the transient FitnessLandscape object.")
else:
    layers = [item for item in landscape.fitness_layers.keys()]
    indices = [0,-1]
    for index in indices:
        f = landscape.fitness_layers[layers[index]].to_scalar()
        G = landscape.graph  # nx.Graph
        pos = nx.spring_layout(G, seed=0)
        plt.figure(figsize=(2, 2))
        nx.draw(G, pos=pos, with_labels=False, node_size=20, width=0.5, node_color=f, cmap="coolwarm")
        plt.tight_layout()
        plt.savefig("../figures/figure_1/alisim_rep1_eigvec_{}.pdf".format(layers[index].split("_")[-1]))
        plt.show()

# %% [markdown]
# ## Supplementary diffusion/correlogram controls
#
# Copied and focused from `the copied source/figure_notebooks_rev snapshot/Figure_1_SI.ipynb`.

# %%
import fitness_landscape as fl
from fitness_landscape.utils import fasta_to_prot20_sequences
from fitness_landscape.core.sequence import BinarySequence
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.transforms import eigenmode_decomposition
from fitness_landscape.analysis import (
    calculate_ruggedness_autocorrelation_analytical,
    calculate_ruggedness_autocorrelation_stochastic,
    compute_ruggedness_diffusion_scale,
)
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.ticker as mticker
from scipy.stats import pearsonr, spearmanr

# %%
nk = fl.models.nk.create_nk_binary_landscape(N=5, K=3, seed=42)

# Compute eigenmodes of nk model
eigvals, eigvecs = eigenmode_decomposition(
    nk,
    matrix="laplacian",
)

# Range of t values to demonstrate heat kernel
t_range = np.linspace(1, 50, 10)
colors = cm.cividis(np.linspace(0, 1, len(t_range)))

epsilon = 1e-8

fig, ax = plt.subplots(figsize=(3, 2.5))

for t, color in zip(t_range, colors):

    # Compute heat kernel values
    lambda_adjusted = eigvals + epsilon
    h_i = np.exp(-t * lambda_adjusted)

    ax.plot(eigvals, h_i, color=color, lw=1)

ax.set_xlabel("Laplacian eigenvalue")
ax.set_ylabel("Heat kernel value")
ax.set_yscale("log")

norm = mcolors.Normalize(vmin=t_range.min(), vmax=t_range.max())
sm = cm.ScalarMappable(cmap="cividis", norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Diffusion scale $t$")

plt.tight_layout()
plt.savefig("../figures/si_figures/si_figure_1/raw_heat_kernel_value.pdf")
plt.show()

# %%

t_range = np.linspace(1, 10, 4)
colors = cm.cividis(np.linspace(0, 1, 5))

total_plots = len(t_range) + 1
fig, axes = plt.subplots(total_plots, 1, figsize=(5, 3), sharex=True)

# Perform graph Fourier transform
eigenvectors, eigenvalues, f_hat = _, _, coefficients = fl.transforms.graph_fourier.graph_fourier_transform(nk)

# Compute the spectral energies of `G`
f_hat_energies = f_hat

# Plot original spectral energies
axes[0].plot(f_hat_energies, color='black')
axes[0].grid(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].tick_params(axis='y', which='both', left=False, labelleft=False)
axes[0].spines['left'].set_visible(False)

# Iterate through t values and plot modified spectral energies
for idx, (t, color) in enumerate(zip(t_range, colors)):

    if idx == 1:
        axes[idx + 1].set_ylabel('Fourier coefficient')

    # Compute the heat kernel values at timestep t
    lambda_adjusted = eigenvalues + epsilon
    h_i = np.exp(-t * lambda_adjusted)

    # Scale the energies of `G` according to the heat kernel eigenvalues
    f_hat_modified = h_i * f_hat

    # Compute the spectral energies of `G` with heat diffusion
    f_hat_modified_energies = np.abs(f_hat_modified)

    axes[idx + 1].plot(f_hat_modified_energies, color=color)
    axes[idx + 1].grid(False)
    axes[idx + 1].spines['right'].set_visible(False)
    axes[idx + 1].spines['top'].set_visible(False)
    axes[idx + 1].spines['left'].set_visible(False)
    axes[idx + 1].tick_params(axis='y', which='both', left=False, labelleft=False)

axes[-1].set_xlabel('Laplacian Eigenvalue Index')

plt.tight_layout()
plt.savefig("../figures/si_figures/si_figure_1/heat_kernel_spectral_energies.pdf")
plt.show()

# %%
tmap_dict = {}
auto_cor_dict = {}
dirichlet_dict = {}

# Range of N variables to consider
n_range = list(range(4, 11))

# Define K up to N-1 for each N
for n_param in n_range:
    k_range = list(range(0, n_param))
    
    # Construct NK landscapes and compute diffusion map for each (N, K)

    for k_param in k_range:
        nk = fl.models.nk.create_nk_binary_landscape(N=n_param, K=k_param, seed=42)
        tmap_dict[(n_param, k_param)] = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(nk, t_min=1e-10, t_max=1e2)
        auto_cor_dict[(n_param, k_param)] = fl.analysis.random_walk.calculate_ruggedness_autocorrelation_analytical(nk)
        dirichlet_dict[(n_param, k_param)] = fl.analysis.dirichlet_energy.calculate_ruggedness_dirichlet_energy(nk)

# Indicator vs. N and K parameters in NK model
rows = []
for (N, K), tmap_res in tmap_dict.items():
    rows.append({
        "N": int(N),
        "K": int(K),
        "t_map": float(tmap_res["t_map"]),
        "corr_len": float(auto_cor_dict[(N, K)]["correlation_length"]),
        "dirichlet": float(dirichlet_dict[(N, K)]["total_dirichlet_energy"]),
    })
df = pd.DataFrame(rows).sort_values(["N", "K"]).reset_index(drop=True)

# Comparison of different ruggedness metrics under ideal conditions
df_plot = df.copy()

# log-transform
eps = 1e-12
df_plot["log_tmap"] = np.log10(df_plot["t_map"] + eps)
df_plot["log_corr"] = np.log10(df_plot["corr_len"] + eps)
df_plot["log_dir"]  = np.log10(df_plot["dirichlet"] + eps)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from scipy.stats import pearsonr, spearmanr

cols = ["log_tmap", "log_corr", "log_dir"]
labels = [r"$\log_{10}(t_{\mathrm{MAP}})$",
          r"$\log_{10}(\mathrm{corr.\ length})$",
          r"$\log_{10}(\mathrm{Dirichlet})$"]

Ns = sorted(df_plot["N"].unique())
cmap = cm.viridis
colors = {N: cmap(i/(len(Ns)-1 if len(Ns) > 1 else 1)) for i, N in enumerate(Ns)}

def p_bound(p):
    if (p == 0) or (p < 1e-50):
        return "< 1e-50"
    return f"= {p:.1e}"

fig, axes = plt.subplots(3, 3, figsize=(6.6, 6.6), constrained_layout=False)
fig.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.98, wspace=0.12, hspace=0.12)

for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        x = df_plot[cols[j]].to_numpy()
        y = df_plot[cols[i]].to_numpy()

        if i == j:
            ax.hist(x, bins=16, color="0.75", edgecolor="0.25", linewidth=0.5)
        elif i > j:
            for N in Ns:
                sub = df_plot[df_plot["N"] == N]
                ax.scatter(sub[cols[j]], sub[cols[i]],
                           s=20, alpha=1, color=colors[N], linewidths=1)
        else:
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)

            txt = (
                f"Pearson r = {r_p:.2f}\n"
                f"Spearman ρ = {r_s:.2f}\n\n"
                f"p(Pearson) {p_bound(p_p)}\n"
                f"p(Spearman) {p_bound(p_s)}"
            )
            ax.text(0.5, 0.5, txt, ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, linespacing=1.25)

            ax.set_xticks([])
            ax.set_yticks([])

        if i < 2:
            ax.set_xticklabels([])
        if j > 0:
            ax.set_yticklabels([])

        ax.tick_params(length=2, width=0.8)

for i in range(3):
    axes[i, 0].set_ylabel(labels[i], fontsize=10)
for j in range(3):
    axes[2, j].set_xlabel(labels[j], fontsize=10)

for ax in axes[2, :]:
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
for ax in axes[:, 0]:
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

cax = fig.add_axes([0.88, 0.25, 0.02, 0.50])  # [left, bottom, width, height]
norm = mcolors.Normalize(vmin=min(Ns), vmax=max(Ns))
sm = cm.ScalarMappable(norm=norm, cmap="viridis")
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("N", rotation=90)

plt.savefig("../figures/si_figures/si_figure_2/laplacian_indicators_correlogram_clean.pdf")
plt.show()

# %%
# Spectral imbalance by mixing two eigenmodes

nk = fl.models.nk.create_nk_binary_landscape(N=5, K=0, seed=42)

eigvals, eig_vecs = eigenmode_decomposition(
    nk,
    matrix="laplacian",
)

low_index  = 1
high_index = -2

# cache components once
low_comp  = eig_vecs[:, low_index].astype(float)
high_comp = eig_vecs[:, high_index].astype(float)

# alpha sweep
alpha_grid = np.linspace(0, 1, 21)

rows = []

for a_idx, alpha in enumerate(alpha_grid):
    alpha = float(alpha)

    # Mixture in an orthonormal eigenbasis: sqrt-weights keep ||f|| stable
    f = np.sqrt(1.0 - alpha) * low_comp + np.sqrt(alpha) * high_comp

    # safer layer name
    layer_name = f"combined_eigvecs_low{low_index}_high{high_index}_a{a_idx:02d}"
    nk.attach(name=layer_name, values=f, dtype="numeric")
    nk.view(layer_name)

    # tMAP
    tmap_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(
        nk, t_min=1e-10, t_max=1e2
    )

    # autocorrelation
    ac_res = fl.analysis.random_walk.calculate_ruggedness_autocorrelation_analytical(
        nk, lag_max=None
    )
    rho = np.asarray(ac_res["autocorrelation"], dtype=float)

    # GFT coefficients 
    # returns (eigvals, eigvecs, coeffs) 
    _, _, coeffs = fl.transforms.graph_fourier.graph_fourier_transform(nk)
    coeffs = np.asarray(coeffs, dtype=float)

    E = np.abs(coeffs)
    E = E / (E.sum() if E.sum() > 0 else 1.0)

    rows.append({
        "alpha": alpha,
        "layer": layer_name,

        "t_map": float(tmap_res["t_map"]),
        "t_lo": float(tmap_res["t_lower_confidence_interval"]),
        "t_hi": float(tmap_res["t_upper_confidence_interval"]),
        "t_logpost_map": float(tmap_res["t_logposterior_map"]),
        "t_var_approx": float(tmap_res["variance_approximate"]),

        "corr_len": float(ac_res["correlation_length"]),

        "rho": rho,
        "rho_len": int(len(rho)),
        "rho_lag1": float(rho[1]) if len(rho) > 1 else np.nan,
        "rho_min": float(np.min(rho)) if len(rho) else np.nan,

        # store spectral energy vector for plotting
        "E": E,
    })

df = pd.DataFrame(rows)
df["log_tmap"] = np.log(df["t_map"].clip(lower=1e-300))

# guard correlation length
df["log_corr_len"] = np.log(df["corr_len"].clip(lower=1e-300))

fig, ax = plt.subplots(figsize=(3.2, 2.4))

for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    i = int(np.argmin(np.abs(df["alpha"].to_numpy() - alpha)))
    rho = df.loc[i, "rho"]
    ax.plot(np.arange(len(rho)), rho, lw=1, marker="o", ms=2, label=f"α={df.loc[i,'alpha']:.2f}")

ax.set_xlabel("lag τ")
ax.set_ylabel("autocorrelation ρ(τ)")
ax.grid(True, ls="--", c="0.85")
ax.legend(frameon=False, fontsize=7)
plt.tight_layout()
plt.savefig("../figures/si_figures/si_figure_3/autocorrelation_mixtures.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(3.2, 2.4))
ax.plot(df["alpha"], df["corr_len"], marker="o", ms=3, lw=1)

ax.set_xlabel("mixture weight α")
ax.set_ylabel("correlation length")
ax.grid(True, ls="--", c="0.85")
plt.tight_layout()
plt.savefig("../figures/si_figures/si_figure_3/autocorrelation_vs_alpha.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(3.2, 2.4))
ax.plot(df["alpha"], df["t_map"], marker="o", ms=3, lw=1)

# ax.set_yscale("log")
ax.set_xlabel("mixture weight α")
ax.set_ylabel(r"$t_{\mathrm{MAP}}$")
ax.grid(True, which="both", ls="--", c="0.85")
plt.tight_layout()
plt.savefig("../figures/si_figures/si_figure_3/tmap_vs_alpha.pdf")
plt.show()


# Plot GFT components
eps = 1e-300

# Use "power" spectrum; more standard than |c|
def gft_energy(coeffs):
    c = np.asarray(coeffs, float)
    P = np.abs(c)**2
    return P / (P.sum() + eps)

# Extract energy in the two modes you mixed
E_low  = []
E_high = []
for i in range(len(df)):
    E = df.loc[i, "E"]            # if you already stored E as power, keep it
    # otherwise recompute from stored coeffs (recommended to store coeffs too)
    E_low.append(E[low_index])
    E_high.append(E[high_index])

E_low = np.array(E_low)
E_high = np.array(E_high)

fig, ax = plt.subplots(figsize=(3.2, 2.2))
ax.plot(df["alpha"], E_low,  marker="o", ms=3, lw=1, label=f"mode 1 (low)")
ax.plot(df["alpha"], E_high, marker="o", ms=3, lw=1, label=f"mode 30 (high)")
ax.set_xlabel("mixture weight α")
ax.set_ylabel("spectral energy")
ax.set_ylim(-0.02, 1.02)
ax.grid(True, ls="--", c="0.85")
ax.legend(frameon=False, fontsize=7)
plt.tight_layout()
plt.savefig("../figures/si_figures/si_figure_3/spectral_contribution_vs_alpha.pdf")
plt.show()

# %%
def graph_metrics(G):
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    return {
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "density": float(nx.density(G)),
        "mean_degree": float(degrees.mean()) if len(degrees) else np.nan,
        "max_degree": float(degrees.max()) if len(degrees) else np.nan,
        "avg_clustering": float(nx.average_clustering(G)),
    }


def landscape_from_graph(G):
    graph = G.copy()
    for node in graph.nodes():
        graph.nodes[node]["sequence"] = fl.BaseNumpySequence([node], sequence_id=str(node))
    return FitnessLandscape.from_graph(graph)


def make_signal_from_graph(G, alpha, rng):
    nodelist = list(G.nodes())
    laplacian = nx.laplacian_matrix(G, nodelist=nodelist).toarray().astype(float)
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    low_index = 1 if eigvecs.shape[1] > 1 else 0
    high_index = eigvecs.shape[1] - 2 if eigvecs.shape[1] > 2 else eigvecs.shape[1] - 1
    low_comp = eigvecs[:, low_index]
    high_comp = eigvecs[:, high_index]
    signs = rng.choice([-1.0, 1.0], size=2)
    signal = np.sqrt(1.0 - alpha) * signs[0] * low_comp + np.sqrt(alpha) * signs[1] * high_comp
    signal = (signal - signal.mean()) / (signal.std() + 1e-12)
    return signal, {
        "low_index": int(low_index),
        "high_index": int(high_index),
        "low_eigval": float(eigvals[low_index]),
        "high_eigval": float(eigvals[high_index]),
    }

# %%

# Construct pathogenic graphs.
alpha_grid = np.linspace(0, 1, 21)

graphs = {
    "barbell": nx.barbell_graph(16, 2),# extreme bottleneck
    "lollipop": nx.lollipop_graph(20, 12),# clique + path
    "sbm_strong": nx.stochastic_block_model([16,16], [[0.5,0.01],[0.01,0.5]], seed=1),
    "sbm_weak": nx.stochastic_block_model([16,16], [[0.5,0.1],[0.1,0.5]], seed=2),
    "ba": nx.barabasi_albert_graph(32, 3, seed=3),# huby
    "ws": nx.watts_strogatz_graph(32, 4, 0.2, seed=4),# small-world-ish
}
n_reps = 30
seed = 0
lag_max = None

graphs_to_run = {
    "barbell": graphs["barbell"],
    "lollipop": graphs["lollipop"],
    "sbm_strong": graphs["sbm_strong"],
    "sbm_weak": graphs["sbm_weak"],
    "ba": graphs["ba"],
    "ws": graphs["ws"],
}

rng_master = np.random.default_rng(seed)
rows = []

for graph_name, G in graphs_to_run.items():
    topo = graph_metrics(G)
    L = landscape_from_graph(G)

    for ai, alpha in enumerate(alpha_grid):
        for rep in range(n_reps):

            sub_seed = int(rng_master.integers(0, 2**32 - 1))
            rng = np.random.default_rng(sub_seed)

            f, meta = make_signal_from_graph(G, alpha=float(alpha), rng=rng)

            layer_name = f"{graph_name}_a{int(round(alpha*100)):03d}_r{rep:03d}"
            L.add(name=layer_name, values=f, dtype="numeric")
            L.view(layer_name)

            ac = calculate_ruggedness_autocorrelation_analytical(L, lag_max=lag_max)
            tm = compute_ruggedness_diffusion_scale(L, t_min=1e-10, t_max=1e2)

            rho = np.asarray(ac["autocorrelation"], dtype=float)

            rows.append({
                "graph": graph_name,
                "alpha": float(alpha),
                "rep": int(rep),
                **topo,
                **meta,
                "corr_len": float(ac["correlation_length"]),
                "rho_lag1": float(rho[1]) if len(rho) > 1 else np.nan,
                "rho_min": float(np.min(rho)) if len(rho) else np.nan,
                "rho_sign_flips": int(np.sum(np.sign(rho[1:])[:-1] * np.sign(rho[1:])[1:] < 0)) if len(rho) > 2 else 0,
                "t_map": float(tm["t_map"]),
                "t_lo": float(tm["t_lower_confidence_interval"]),
                "t_hi": float(tm["t_upper_confidence_interval"]),
            })

df_all = pd.DataFrame(rows)
df_all["log_tmap"] = np.log10(df_all["t_map"].clip(lower=1e-300))
df_all["t_ci_width"] = df_all["t_hi"] - df_all["t_lo"]
df_all["log_t_ci_width"] = np.log10(df_all["t_ci_width"].clip(lower=1e-300))

fig, ax = plt.subplots(figsize=(4.2, 2.6))

for graph_name in graphs_to_run.keys():
    sub = df_all[df_all["graph"] == graph_name]
    g = sub.groupby("alpha")
    x = g["alpha"].first().to_numpy()
    med = g["t_map"].median().to_numpy()
    q1 = g["t_map"].quantile(0.25).to_numpy()
    q3 = g["t_map"].quantile(0.75).to_numpy()

    ax.plot(x, med, lw=1, marker="o", ms=2, label=graph_name)
    ax.fill_between(x, q1, q3, alpha=0.15)

ax.set_xlabel("mixture weight α (high-frequency)")
ax.set_ylabel(r"$t_{\mathrm{MAP}}$")
ax.grid(True, which="both", ls="--", c="0.85")
ax.legend(frameon=False, fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig("../figures/si_figures/si_figure_3/tmap_vs_alpha_all_graphs.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(4.2, 2.6))

for graph_name in graphs_to_run.keys():
    sub = df_all[df_all["graph"] == graph_name]
    g = sub.groupby("alpha")
    x = g["alpha"].first().to_numpy()
    med = g["corr_len"].median().to_numpy()
    q1 = g["corr_len"].quantile(0.25).to_numpy()
    q3 = g["corr_len"].quantile(0.75).to_numpy()

    ax.plot(x, med, lw=1, marker="o", ms=2, label=graph_name)
    ax.fill_between(x, q1, q3, alpha=0.15)

ax.set_xlabel("mixture weight α (high-frequency)")
ax.set_ylabel("Correlation length")
ax.grid(True, ls="--", c="0.85")
ax.legend(frameon=False, fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig("../figures/si_figures/si_figure_3/corr_len_vs_alpha_all_graphs.pdf")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4), sharex=True)

ax = axes[0]
for graph_name in graphs_to_run.keys():
    sub = df_all[df_all["graph"] == graph_name]
    g = sub.groupby("alpha")
    x = g["alpha"].first().to_numpy()
    med = g["rho_sign_flips"].median().to_numpy()
    ax.plot(x, med, lw=1, marker="o", ms=2, label=graph_name)
ax.set_xlabel("mixture weight α (high-frequency)")
ax.set_ylabel("# sign flips in ρ(τ)")
ax.grid(True, ls="--", c="0.85")

ax = axes[1]
for graph_name in graphs_to_run.keys():
    sub = df_all[df_all["graph"] == graph_name]
    g = sub.groupby("alpha")
    x = g["alpha"].first().to_numpy()
    med = g["rho_min"].median().to_numpy()
    ax.plot(x, med, lw=1, marker="o", ms=2, label=graph_name)
ax.set_xlabel("mixture weight α (high-frequency)")
ax.set_ylabel("min ρ(τ)")
ax.grid(True, ls="--", c="0.85")

axes[0].legend(frameon=False, fontsize=7, ncol=2)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(4.2, 2.6))

for graph_name in graphs_to_run.keys():
    sub = df_all[df_all["graph"] == graph_name]
    g = sub.groupby("alpha")
    x = g["alpha"].first().to_numpy()
    med = g["t_ci_width"].median().to_numpy()
    q1 = g["t_ci_width"].quantile(0.25).to_numpy()
    q3 = g["t_ci_width"].quantile(0.75).to_numpy()

    ax.plot(x, med, lw=1, marker="o", ms=2, label=graph_name)
    ax.fill_between(x, q1, q3, alpha=0.15)

ax.set_yscale("log")
ax.set_xlabel("mixture weight α (high-frequency)")
ax.set_ylabel(r"$t$ CI width (hi-lo)")
ax.grid(True, which="both", ls="--", c="0.85")
ax.legend(frameon=False, fontsize=7, ncol=2)
plt.tight_layout()
plt.show()

alphas_show = (0.0, 0.25, 0.5, 0.75, 1.0)

for graph_name, G in graphs_to_run.items():
    L = landscape_from_graph(G)
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    for i, a in enumerate(alphas_show):
        f, _ = make_signal_from_graph(G, alpha=float(a), rng=rng)
        layer_name = f"{graph_name}_example_a{int(round(a*100)):03d}_{i:02d}"
        L.add(name=layer_name, values=f, dtype="numeric")
        L.view(layer_name)

        ac = calculate_ruggedness_autocorrelation_analytical(L, lag_max=lag_max)
        rho = np.asarray(ac["autocorrelation"], dtype=float)

        ax.plot(np.arange(len(rho)), rho, marker="o", ms=2, lw=1, label=f"α={a:.2f}")

    ax.set_title(graph_name)
    ax.set_xlabel("lag τ")
    ax.set_ylabel("autocorrelation ρ(τ)")
    ax.grid(True, ls="--", c="0.85")
    ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    plt.show()

# %%
# Ruggedness indicators vs. N and K in NK model

tmap_dict = {}
auto_cor_dict = {}
dirichlet_dict = {}

# Range of N variables to consider
n_range = list(range(4, 11))

# Define K up to N-1 for each N
for n_param in n_range:
    k_range = list(range(0, n_param))
    
    # Construct NK landscapes and compute diffusion map for each (N, K)

    for k_param in k_range:
        nk = fl.models.nk.create_nk_binary_landscape(N=n_param, K=k_param, seed=42)
        tmap_dict[(n_param, k_param)] = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(nk, t_min=1e-10, t_max=1e2)
        auto_cor_dict[(n_param, k_param)] = fl.analysis.random_walk.calculate_ruggedness_autocorrelation_analytical(nk)
        dirichlet_dict[(n_param, k_param)] = fl.analysis.dirichlet_energy.calculate_ruggedness_dirichlet_energy(nk)

# Indicator vs. N and K parameters in NK model
rows = []
for (N, K), tmap_res in tmap_dict.items():
    rows.append({
        "N": int(N),
        "K": int(K),
        "t_map": float(tmap_res["t_map"]),
        "corr_len": float(auto_cor_dict[(N, K)]["correlation_length"]),
        "dirichlet": float(dirichlet_dict[(N, K)]["total_dirichlet_energy"]),
    })
df = pd.DataFrame(rows).sort_values(["N", "K"]).reset_index(drop=True)

Ns = sorted(df["N"].unique())
cmap = cm.viridis
colors = {N: cmap(i/(len(Ns)-1 if len(Ns) > 1 else 1)) for i, N in enumerate(Ns)}

fig, axes = plt.subplots(1, 3, figsize=(6, 2.4), sharey=True)
fig.subplots_adjust(right=0.88, wspace=0.35)  

x_cols   = ["t_map", "corr_len", "dirichlet"]
x_labels = [r"$t_{\mathrm{MAP}}$", "Correlation length", "Dirichlet energy"]

for ax, xcol, xlabel in zip(axes, x_cols, x_labels):
    for N in Ns:
        sub = df[df["N"] == N].sort_values("K")
        ax.plot(
            sub[xcol].to_numpy(),
            sub["K"].to_numpy(),
            color=colors[N],
            lw=1.25,
            marker="o",
            markersize=4,
            alpha=0.95
        )
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(True, ls="--", c="0.85")
    ax.invert_yaxis()

axes[0].set_ylabel("K")
fig.subplots_adjust(wspace=0.35)

ax_dir = axes[2]

# Fewer ticks on log axis
ax_dir.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4))
ax_dir.xaxis.set_minor_locator(mticker.NullLocator())  # remove minor ticks entirely

# Compact scientific formatting (10^x style)
ax_dir.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))

plt.tight_layout()
plt.savefig("../figures/si_figures/si_figure_4/nk_k_n_vs_ruggedness_indicators.pdf")
plt.show()

# %%
# Epistasis quantification at different N and K in NK model

rows = []

n_range = list(range(4, 11))

for n_param in n_range:
    for k_param in range(0, n_param):
        nk = fl.models.nk.create_nk_binary_landscape(N=n_param, K=k_param, seed=42)

        max_order = min(n_param, k_param + 1)  # NK interactions are bounded like this
        res = fl.analysis.epistasis.calculate_epistasis_walsh(nk, order=max_order)

        var_by_order = res.get("variance_explained", {})  # {order:int -> fraction}

        # Ensure we include missing orders as zeros for consistent plotting later
        for o in range(1, n_param + 1):
            rows.append({
                "N": n_param,
                "K": k_param,
                "order": o,
                "var_explained": float(var_by_order.get(o, 0.0))
            })

df_epi = pd.DataFrame(rows)

Ks = sorted(df_epi["K"].unique())
Ns = sorted(df_epi["N"].unique())

cmap = cm.viridis
colors = {N: cmap(i/(len(Ns)-1 if len(Ns) > 1 else 1)) for i, N in enumerate(Ns)}

ncols = 4
nrows = int(np.ceil(len(Ks) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(8, 6), sharex=False, sharey=True)
axes = np.array(axes).reshape(-1)

for ax, K in zip(axes, Ks):
    subK = df_epi[df_epi["K"] == K]

    for N in Ns:
        sub = subK[subK["N"] == N].sort_values("order")
        sub = sub[sub["order"] <= N]

        ax.plot(
            sub["order"].to_numpy(),
            sub["var_explained"].to_numpy(),
            color=colors[N],
            lw=1.25,
            alpha=0.9
        )

    ax.set_title(f"K={K}", fontsize=10)
    ax.grid(True, ls="--", c="0.85")

for ax in axes[len(Ks):]:
    ax.axis("off")

for ax in axes[:len(Ks)]:  # only the used axes
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

fig.supxlabel("Interaction order", y=0.04)
fig.supylabel("Fraction variance explained", x=0.04)

cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
norm = mcolors.Normalize(vmin=min(Ns), vmax=max(Ns))
sm = cm.ScalarMappable(norm=norm, cmap="viridis")
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("N")

plt.tight_layout(rect=[0.05, 0.05, 0.90, 1.0])
plt.savefig("../figures/si_figures/si_figure_4/si_nk_epistasis_variance_by_order_byK.pdf")
plt.show()
