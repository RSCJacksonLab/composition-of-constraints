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
# ## Smooth-constraint composition simulations and schematics
#
# Copied and focused from `the copied source/figure_notebooks_rev snapshot/Figure_3.ipynb`.

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

# %%
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

# %%
_composition_dir = processed_dir / "constraint_composition"
_nk_results_pkl = _composition_dir / "nk_min_composition_results.pkl"
if _nk_results_pkl.exists():
    with _nk_results_pkl.open("rb") as handle:
        nk_results = pickle.load(handle)
else:
    nk_results = sweep_all(fl, Ns=[4,5,6], Ks=[0], ms=[2,4,8,10,20,50,100], alphas=[0.0,0.25,0.5,0.75, 0.9,1.0], seeds=range(10))
df = pd.DataFrame(nk_results)

# %%
color_by = "alpha_align"

# find variance columns v0, v1, ...
vcols = sorted(
    [c for c in df.columns if c.startswith("v") and c[1:].isdigit()],
    key=lambda x: int(x[1:])
)

# drop v0 if you don't want order-0 (your code was doing this)
vcols = [c for c in vcols if int(c[1:]) != 0]

if len(vcols) == 0:
    raise ValueError("No variance columns found (expected v0, v1, ...).")

orders = np.array([int(c[1:]) for c in vcols], dtype=float)

Y_all = df[vcols].to_numpy(dtype=float)
C_all = df[color_by].to_numpy(dtype=float)

good = np.isfinite(Y_all).all(axis=1) & np.isfinite(C_all)
Y = Y_all[good]
C = C_all[good]

lines = [np.column_stack([orders, y]) for y in Y]

norm = Normalize(vmin=np.nanmin(C), vmax=np.nanmax(C))

lc = LineCollection(lines, cmap=plt.cm.viridis, norm=norm, linewidths=0.5, alpha=0.85)
lc.set_array(C)

# Plot (create fig explicitly)
fig, ax = plt.subplots(figsize=(3.75, 2.5))
ax.add_collection(lc)

ax.set_xlim(orders.min(), orders.max())
ymin = np.nanmin(Y)
ymax = np.nanmax(Y)
pad = 0.02 * (ymax - ymin) if ymax > ymin else 0.05
ax.set_ylim(ymin - pad, ymax + pad)

ax.set_xlabel("Epistasis order")
ax.set_ylabel("Variance explained")

cbar = fig.colorbar(lc, ax=ax)
cbar.set_label(color_by)

ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
fig.tight_layout()
fig.savefig("../figures/figure_3/var_by_order_min_compose.pdf")
plt.show()

# %%
bar_vcols = [c for c in vcols if 1 <= int(c[1:]) <= 6]

if len(bar_vcols) != 6:
    raise ValueError("Expected variance columns for epistasis orders 1 through 6.")

plot_df = pd.DataFrame(Y, columns=vcols)[bar_vcols].copy()
plot_df["alpha"] = C
stacked = plot_df.groupby("alpha", as_index=True)[bar_vcols].mean().sort_index()

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
ax.set_xticklabels([f"{alpha:g}" for alpha in stacked.index.to_numpy(dtype=float)])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Variance explained")
ax.set_ylim(0, max(1.0, bottom.max() * 1.02))
plt.xticks(rotation=45)
# ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
# ax.legend(title="Epistasis order", frameon=False, ncol=2)

fig.tight_layout()
fig.savefig("../figures/figure_3/var_by_order_stacked_alpha.pdf")
plt.show()

# %%
order_vcols = [c for c in vcols if int(c[1:]) != 0]

if len(order_vcols) == 0:
    raise ValueError("No nonzero epistasis-order variance columns found.")

plot_df = pd.DataFrame(Y, columns=vcols)[order_vcols].copy()
plot_df["alpha"] = C

stacked_by_order = (
    plot_df.groupby("alpha", as_index=True)[order_vcols]
    .mean()
    .sort_index()
    .T
)

x = np.arange(len(stacked_by_order), dtype=float)
alphas = stacked_by_order.columns.to_numpy(dtype=float)
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))
bottom = np.zeros(len(stacked_by_order), dtype=float)

fig, ax = plt.subplots(figsize=(3.0, 2.5))

for color, alpha in zip(colors, alphas):
    values = stacked_by_order[alpha].to_numpy(dtype=float)
    ax.bar(
        x,
        values,
        bottom=bottom,
        width=0.75,
        color=color,
        edgecolor="white",
        linewidth=0.4,
        label=f"{alpha:g}",
    )
    bottom += values

ax.set_xticks(x)
ax.set_xticklabels([str(int(col[1:])) for col in stacked_by_order.index])
ax.set_xlabel("Epistasis order")
ax.set_ylabel("log Variance explained")
# ax.set_ylim(0, max(1.0, bottom.max() * 1.02))
# ax.legend(title="$\\alpha$", frameon=False, ncol=2)

ax.set_yscale("symlog", linthresh=1e-4)

fig.tight_layout()
fig.savefig("../figures/figure_3/var_by_order_stacked_order_alpha.pdf")
plt.show()

# %%
plt.figure(figsize=(3.5, 2.5))

df['log2_m'] = np.log2(df['m'])

sc = plt.scatter(
    df['alpha_align'],
    df['tmap'],
    c=df['log2_m'],
    cmap='cividis',
    s=10
)

plt.ylabel(r"$t_{\mathrm{MAP}}$")
plt.xlabel(r"$\alpha_{\mathrm{}}$")

cbar = plt.colorbar(sc)
cbar.set_label(r"$\log_2(m)$")

plt.ylim(-1, 17)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig('../figures/figure_3/alpha_vs_tmap.pdf')
plt.show()

# %%
plt.figure(figsize=(3.5, 2.5))

sc = plt.scatter(
    df['m'],
    df['tmap'],
    c=df['alpha_align'],
    cmap='cividis',
    s=10
)
plt.ylim(-1, 17)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.ylabel(r"$t_{\mathrm{MAP}}$")
plt.xlabel(r"$m {\mathrm{}}$")

# %%
import numpy as np
import matplotlib.pyplot as plt

m_values = sorted(df['m'].unique())
m_values = [m for m in m_values if m != 8]  # drop m=8

alpha_values = sorted(df['alpha_align'].unique())
n_facets = len(m_values)
positions = np.arange(len(alpha_values))

fig, axes = plt.subplots(n_facets, 1,
                         figsize=(2, 0.75 * n_facets),
                         sharex=True, sharey=True,
                         gridspec_kw={'hspace': 0.0})

for idx, m_val in enumerate(m_values):
    ax = axes[idx]
    sub = df[df['m'] == m_val]

    data_per_alpha = []
    for a in alpha_values:
        vals = sub.loc[sub['alpha_align'] == a, 'tmap'].values
        data_per_alpha.append(vals)

    bp = ax.boxplot(
        data_per_alpha,
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray", alpha=1, linewidth=0.6),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", alpha=1, linewidth=0.6),
        capprops=dict(color="black", alpha=1, linewidth=0.6),
    )

    # Gridlines
    ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # # m label inside panel
    # ax.text(0.97, 0.85, f"m = {m_val}",
    #         transform=ax.transAxes, fontsize=7, fontweight='bold',
    #         ha='right', va='top')

    # Minimal y-axis
    ax.set_yticks([0, 5, 10])
    ax.set_yticklabels(['0', '5', '10'])
    ax.tick_params(axis='y', length=2, pad=2)

    # Separator line at top of each panel
    ax.axhline(y=ax.get_ylim()[1], color='black', linewidth=0.8,
               clip_on=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color('#888888')
    ax.tick_params(axis='x', length=0)

# Bottom axis labels only on last
axes[-1].set_xticks(positions)
axes[-1].set_xticklabels([f"{a:.2g}" for a in alpha_values])
axes[-1].set_xlabel(r"$\alpha$")
axes[-1].tick_params(axis='x', length=3)

# Shared y-label
fig.text(0.02, 0.5, r"$t_{\mathrm{MAP}}$", va='center',
         rotation='vertical')
plt.xticks(rotation=45)
fig.subplots_adjust(left=0.14, right=0.95, top=0.97, bottom=0.1)
plt.tight_layout()
fig.savefig('../figures/figure_3/alpha_vs_tmap_compressed.pdf')

plt.show()

# %%
heatmap_m_values = [m for m in sorted(df['m'].unique()) if m != 8]  # match compressed plot above
heatmap_alpha_values = sorted(df['alpha_align'].unique())

median_tmap = (
    df[df['m'].isin(heatmap_m_values)]
    .pivot_table(index='m', columns='alpha_align', values='tmap', aggfunc='mean')
    .reindex(index=heatmap_m_values, columns=heatmap_alpha_values)
)

fig, ax = plt.subplots(figsize=(2.9, 2.6))
sns.heatmap(
    median_tmap,
    ax=ax,
    cmap='coolwarm',
    annot=False,
    linewidths=0.4,
    linecolor='black',
    cbar_kws={'label': r'Median $t_{\mathrm{MAP}}$'},
)

ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$m$')
ax.set_xticklabels([f'{a:g}' for a in heatmap_alpha_values], rotation=45, ha='right')
ax.set_yticklabels([str(m) for m in heatmap_m_values], rotation=0)
ax.tick_params(axis='both')

cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=7)
cbar.set_label(r'Mean $t_{\mathrm{MAP}}$')

fig.tight_layout()
fig.savefig('../figures/figure_3/alpha_vs_tmap_median_heatmap.pdf')
plt.show()

# %%
df_sub = df[df["alpha_align"].isin([1.0, 0.9])]

# Collect data in order
data = [
    df_sub[df_sub["alpha_align"] == 1.0]["tmap"].values,
    df_sub[df_sub["alpha_align"] == 0.9]["tmap"].values,
]

plt.figure(figsize=(1.45, 2.5))

plt.boxplot(
    data,
    labels=["1.0", "0.9"],
    showfliers=False,
    widths=0.6,
    patch_artist=True,
    boxprops=dict(facecolor="lightgray", alpha=1),
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="black", alpha=1),
    capprops=dict(color="black", alpha=1),
)

plt.xlabel(r"$\alpha_{\mathrm{align}}$")
plt.ylabel(r"$t_{\mathrm{MAP}}$")
# plt.xticks(rotation=90)
plt.ylim(-1, 17)
plt.tight_layout()

plt.savefig("../figures/figure_3/tmap_vs_alpha_align_boxplot.pdf")
plt.show()

# %%
# Test for significance

x = df[df["alpha_align"] == 1.0]["tmap"].values
y = df[df["alpha_align"] == 0.9]["tmap"].values

u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")

print("Mann–Whitney U:", u_stat)
print("p-value:", p_value)

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.optimize import brentq


C_RED    = "#D64545"   # constraint 1 (e.g., folding)
C_BLUE   = "#2B5EA7"   # constraint 2 (e.g., binding)
C_GOLD   = "#E8A838"   # switching boundary
C_BLACK  = "#1a1a1a"
C_MUTED  = "#888888"
C_VIABLE = "#ededed"
C_BG     = "#F5F5F0"

# RGB base arrays for the two constraints
RGB_RED  = np.array([214, 69, 69]) / 255.0
RGB_BLUE = np.array([43, 94, 167]) / 255.0


# ── Shared 2D fitness functions (ground truth for both panels) ──
# Each constraint has a smooth elliptical solution set, but fitness rises
# exponentially toward each center. Because the centers are offset and the
# gradients point in different directions, the min creates a sharp ridge
# along the switching boundary.

def f1_2d(x, y):
    """Constraint 1 (folding): steep exponential peak centered left-up."""
    r2 = (x - 0.38)**2 * 1.2 + (y - 0.58)**2 * 0.85
    return np.exp(-5.0 * r2)

def f2_2d(x, y):
    """Constraint 2 (binding): steep exponential peak centered right-down."""
    r2 = (x - 0.68)**2 * 0.85 + (y - 0.38)**2 * 1.2
    return np.exp(-5.0 * r2)

THRESHOLD = 0.25

# ── 1D functions derived as a diagonal slice through 2D landscape ──
SLICE_START = np.array([-0.05, 0.85])
SLICE_END   = np.array([1.10, 0.10])

def _slice_point(t):
    """Map parameter t in [0,1] to a point on the diagonal slice."""
    return SLICE_START + t * (SLICE_END - SLICE_START)

def f1_1d(t):
    """Constraint 1 evaluated along the diagonal slice."""
    p = _slice_point(t)
    return f1_2d(p[0], p[1])

def f2_1d(t):
    """Constraint 2 evaluated along the diagonal slice."""
    p = _slice_point(t)
    return f2_2d(p[0], p[1])

# Vectorized versions
def f1_1d_vec(t):
    p = SLICE_START[:, None] + t[None, :] * (SLICE_END - SLICE_START)[:, None]
    return f1_2d(p[0], p[1])

def f2_1d_vec(t):
    p = SLICE_START[:, None] + t[None, :] * (SLICE_END - SLICE_START)[:, None]
    return f2_2d(p[0], p[1])


# ── View range for the 2D panel ──
VIEW_PAD = 0.3
VIEW_MIN = -VIEW_PAD
VIEW_MAX = 1.0 + VIEW_PAD


def make_panel_D(ax=None, save_path="panel_D.pdf"):
    """
    2D sequence space heatmap.
    Viable region colored by limiting constraint (red vs blue).
    Switching boundary in gold. Individual constraint boundaries dashed.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.2))
        standalone = True
    else:
        fig = ax.figure
        standalone = False

    res = 500
    x = np.linspace(VIEW_MIN, VIEW_MAX, res)
    y = np.linspace(VIEW_MIN, VIEW_MAX, res)
    X, Y = np.meshgrid(x, y)

    F1 = f1_2d(X, Y)
    F2 = f2_2d(X, Y)
    F_min = np.minimum(F1, F2)

    viable = F_min >= THRESHOLD
    limiter = np.argmin(np.stack([F1, F2], axis=-1), axis=-1)  # 0=red-limited, 1=blue-limited

    # Build RGBA image
    img = np.ones((res, res, 4))
    img[..., :3] = np.array([245, 245, 240]) / 255.0  # non-viable background

    for idx, rgb_base in enumerate([RGB_RED, RGB_BLUE]):
        mask = viable & (limiter == idx)
        brightness = np.clip(0.6 + 0.4 * (F_min - THRESHOLD) / 0.5, 0, 1)
        for c in range(3):
            img[..., c] = np.where(
                mask,
                rgb_base[c] * brightness + (1 - brightness) * 1.0,
                img[..., c],
            )
    img[viable, 3] = 0.75
    img[~viable, 3] = 1.0

    ax.imshow(img, origin="lower", extent=[VIEW_MIN, VIEW_MAX, VIEW_MIN, VIEW_MAX],
              aspect="equal", interpolation="bilinear")

    # ── Individual constraint boundaries (dashed contours) ──
    ax.contour(X, Y, F1, levels=[THRESHOLD], colors=[C_RED],
               linewidths=1.2, linestyles="dashed")
    ax.contour(X, Y, F2, levels=[THRESHOLD], colors=[C_BLUE],
               linewidths=1.2, linestyles="dashed")

    # ── Composite viable boundary (solid dark) ──
    ax.contour(X, Y, F_min, levels=[THRESHOLD], colors=[C_BLACK],
               linewidths=1.4, linestyles="solid")

    # ── Switching boundary (where F1 ≈ F2 inside viable region) ──
    diff = F1 - F2
    diff_masked = np.where(viable, diff, np.nan)
    # Draw switching boundary only within viable region
    cs = ax.contour(X, Y, diff_masked, levels=[0], colors=[C_GOLD],
                    linewidths=2.0, linestyles="solid")

    # ── Draw the 1D slice line (dashed, thin) to connect to panel C ──
    ax.plot([SLICE_START[0], SLICE_END[0]], [SLICE_START[1], SLICE_END[1]],
            color=C_BLACK, lw=0.8, ls=(0, (3, 3)), alpha=0.5, zorder=6)
    # Small label for the slice
    mid = _slice_point(0.5)
    angle = np.degrees(np.arctan2(SLICE_END[1] - SLICE_START[1],
                                   SLICE_END[0] - SLICE_START[0]))
    # ax.text(mid[0] + 0.06, mid[1] + 0.06, "1D slice", fontsize=5.5,
    #         color=C_MUTED, ha="center", va="bottom", rotation=angle,
    #         fontstyle="italic")

    # ── Labels ──
    ax.set_xlabel("Sequence Space dim 1")
    ax.set_ylabel("Sequence Space dim 2")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(VIEW_MIN, VIEW_MAX)
    ax.set_ylim(VIEW_MIN, VIEW_MAX)

    if standalone:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.show()

    return ax


def make_panel_C(ax=None, save_path="panel_C.pdf"):
    """
    1D cross-section through sequence space — a diagonal slice through the 2D landscape.
    Two smooth fitness functions, their piecewise minimum,
    viable region shaded, switching points marked.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.2))
        standalone = True
    else:
        fig = ax.figure
        standalone = False

    t = np.linspace(0, 1, 2000)
    y1 = f1_1d_vec(t)
    y2 = f2_1d_vec(t)
    y_min = np.minimum(y1, y2)
    threshold = THRESHOLD

    # ── Shade viable region (under composite, above threshold) ──
    viable_mask = y_min >= threshold
    ax.fill_between(t, threshold, y_min, where=viable_mask,
                    color=C_VIABLE, zorder=1)

    # ── Individual constraint curves (semi-transparent) ──
    ax.plot(t, y1, color=C_RED, lw=1.3, alpha=0.5, label="$f_1$ (folding)", zorder=2)
    ax.plot(t, y2, color=C_BLUE, lw=1.3, alpha=0.5, label="$f_2$ (binding)", zorder=2)

    # ── Composite min curve (bold) ──
    limiter = np.argmin(np.stack([y1, y2]), axis=0)
    segment_colors = [C_RED, C_BLUE]

    changes = np.where(np.diff(limiter) != 0)[0]
    boundaries = np.concatenate([[0], changes + 1, [len(t)]])
    for i in range(len(boundaries) - 1):
        sl = slice(boundaries[i], boundaries[i + 1])
        ax.plot(t[sl], y_min[sl], color=segment_colors[limiter[boundaries[i]]],
                lw=2.5, zorder=4, solid_capstyle="round")

    # ── Threshold line ──
    ax.axhline(threshold, color=C_MUTED, lw=0.8, ls=(0, (4, 4)), zorder=1)
    # ax.text(-0.03, threshold, "$f_{\\mathrm{min}}$", fontsize=8, color=C_MUTED,
    #         ha="right", va="center")

    # ── Switching points ──
    diff = y1 - y2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    switch_ts = []
    for idx in sign_changes:
        t0, t1_ = t[idx], t[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        t_cross = t0 - d0 * (t1_ - t0) / (d1 - d0)
        switch_ts.append(t_cross)

    for st in switch_ts:
        sy = min(f1_1d(st), f2_1d(st))
        ax.plot([st, st], [sy - 0.04, sy + 0.04], color=C_GOLD, lw=2.0, zorder=5)
        ax.plot(st, sy + 0.07, marker="v", markersize=6, color=C_GOLD,
                zorder=5, markeredgewidth=0)

    # ── Axes ──
    ax.set_xlabel("Sequence space (1D slice)")
    ax.set_ylabel("Fitness")
    ax.set_xlim(0, 1)
    # ax.set_ylim(-0.6, 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.show()
    return ax

# Generate both panels
make_panel_C(save_path="../figures/figure_3/intersection_1D_slice.pdf")
make_panel_D(save_path="../figures/figure_3/intersection_2D_solution_space.pdf")

plt.show()

# %%
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

# %%
idx = 7

nk = nk_results[idx]['nk_landscape']
if "high_pass_signal" in nk.fitness_layers:
    nk.detach("high_pass_signal")
nk.view("composite_min")
layer_name = "high_pass_signal"
f_high = get_high_pass_signal_by_index(nk, signal_layer="composite_min", N=nk_results[idx]['N'])
nk.attach(name=layer_name, values=f_high, dtype="numeric")
nk.view('high_pass_signal')

s_edges = get_switching_edges(nk)


G = nk.graph
pos = nx.spring_layout(G, seed=2)
node_values = [G.nodes[n][f'fitness_high_pass_signal'] for n in G.nodes()]

plt.figure(figsize=(2.75, 2))

nx.draw_networkx_edges(G, pos, alpha=0.35, edge_color='gray')
nx.draw_networkx_edges(G, pos, edgelist=s_edges, width=1, edge_color='black')

nodes = nx.draw_networkx_nodes(
    G, pos,
    node_size=150,
    node_color=node_values,
    cmap='coolwarm',
    edgecolors='black',
    linewidths=1
)

cbar = plt.colorbar(nodes, fraction=0.046, pad=0.04)
cbar.set_label("High-pass fitness signal", rotation=270, labelpad=12)

plt.axis("off")
plt.tight_layout()
plt.savefig('../figures/figure_3/ruggedness_localization_example.pdf')
plt.show()

# %%

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
    labels=labels,
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

# %%
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

# %%
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

# %%
plt.figure(figsize=(3.5, 2.5))

plt.scatter(df['conductance'], df['tmap'], c=df['N'], s=10, alpha=1)
plt.ylabel(r"$t_{\mathrm{MAP}}$")
plt.xlabel(r"$φ_{\mathrm{S}}$")
plt.tight_layout()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.savefig('../figures/figure_3/conductance_S_vs_tmap.pdf')
plt.show()

# %%
n_value = 3
k_value = 0
seed = 4

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
nk.view("solution_set")


G = nk.graph
node_order = list(G.nodes())

# Build node -> solution label map
solution_nodes = set(node_order[i] for i, is_sol in enumerate(solution_mask) if is_sol)

node_class = {
    node: ("solution" if node in solution_nodes else "non_solution")
    for node in node_order
}

# Split edges into boundary vs interior
boundary_edges = []
interior_edges = []

for u, v in G.edges():
    if node_class[u] != node_class[v]:
        boundary_edges.append((u, v))
    else:
        interior_edges.append((u, v))

# Layout (spring is fine for small NK graphs)
pos = nx.spring_layout(G, seed=1)

# Node colors
node_colors = [
    "#d84b43" if node in solution_nodes else "#cccccc"
    for node in G.nodes()
]

plt.figure(figsize=(1.5, 1.5))

# Draw nodes
nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=100,
    edgecolors="black"
)

# Draw interior edges (thin)
nx.draw_networkx_edges(
    G, pos,
    edgelist=interior_edges,
    width=1.,
    alpha=0.5,
    edge_color="black"
)

# Draw boundary edges (bold)
nx.draw_networkx_edges(
    G, pos,
    edgelist=boundary_edges,
    width=2.0,
    alpha=1.0,
    edge_color="black"
)

plt.axis("off")
plt.tight_layout()
plt.savefig('../figures/figure_3/nk_k=0_epsilonS.pdf')
plt.show()

# %%
n_value = 3
k_value = 2
seed = 4

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
nk.view("solution_set")


G = nk.graph
node_order = list(G.nodes())

# Build node -> solution label map
solution_nodes = set(node_order[i] for i, is_sol in enumerate(solution_mask) if is_sol)

node_class = {
    node: ("solution" if node in solution_nodes else "non_solution")
    for node in node_order
}

# Split edges into boundary vs interior
boundary_edges = []
interior_edges = []

for u, v in G.edges():
    if node_class[u] != node_class[v]:
        boundary_edges.append((u, v))
    else:
        interior_edges.append((u, v))

# Layout (spring is fine for small NK graphs)
pos = nx.spring_layout(G, seed=1)

# Node colors
node_colors = [
    "#d84b43" if node in solution_nodes else "#cccccc"
    for node in G.nodes()
]

plt.figure(figsize=(1.5, 1.5))

# Draw nodes
nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=100,
    edgecolors="black"
)

# Draw interior edges (thin)
nx.draw_networkx_edges(
    G, pos,
    edgelist=interior_edges,
    width=1.,
    alpha=0.5,
    edge_color="black"
)

# Draw boundary edges (bold)
nx.draw_networkx_edges(
    G, pos,
    edgelist=boundary_edges,
    width=2.0,
    alpha=1.0,
    edge_color="#E8A838"
)

plt.axis("off")
plt.tight_layout()
plt.savefig('../figures/figure_3/nk_k=2_epsilonS.pdf')
plt.show()

# %%
N = 4
K = 0
seeds = [1, 9]

landscapes = [
    fl.models.nk.create_nk_binary_landscape(N=N, K=K, seed=seed)
    for seed in seeds
]
signals = [landscape.fitness_layers[f"nk_k={K}"].to_scalar() for landscape in landscapes]
stacked = np.stack(signals, axis=0)
min_comp = np.minimum(signals[0], signals[1])
winner_idx = np.argmin(stacked, axis=0)

G = landscapes[0].graph
node_order = list(G.nodes())
pos = nx.spring_layout(G, seed=2)

all_values = np.concatenate(signals + [min_comp])
norm = mpl.colors.Normalize(vmin=all_values.min(), vmax=all_values.max())
cmap = plt.cm.coolwarm
panel_shapes = ['s', 'o']

fig, axes = plt.subplots(1, 3, figsize=(4.5, 1.7))

for ax, values, shape in zip(axes[:2], signals, panel_shapes):
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color='gray', width=1.0)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=150,
        node_color=values,
        cmap=cmap,
        vmin=norm.vmin,
        vmax=norm.vmax,
        edgecolors='black',
        linewidths=0.9,
        node_shape=shape,
    )
    ax.axis('off')

nx.draw_networkx_edges(G, pos, ax=axes[2], alpha=0.4, edge_color='gray', width=1.0)
for win_idx, shape in enumerate(panel_shapes):
    group_idx = [i for i in range(len(node_order)) if winner_idx[i] == win_idx]
    group_nodes = [node_order[i] for i in group_idx]
    group_colors = min_comp[group_idx]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=group_nodes,
        ax=axes[2],
        node_size=150,
        node_color=group_colors,
        cmap=cmap,
        vmin=norm.vmin,
        vmax=norm.vmax,
        edgecolors='black',
        linewidths=0.9,
        node_shape=shape,
    )
axes[2].axis('off')
fig.tight_layout(pad=0.2, w_pad=0.6)
fig.savefig('../figures/figure_3/nk_min_composition_schematic.pdf', bbox_inches='tight')
plt.show()

# %%
idx = 7

nk = nk_results[idx]['nk_landscape']
if 'high_pass_signal' in nk.fitness_layers:
    nk.detach('high_pass_signal')
nk.view('composite_min')

eigvecs, eigvals, coeffs = fl.transforms.graph_fourier_transform(nk)
additive_cutoff = nk_results[idx]['N']
filter_kernel = np.zeros_like(eigvals)
filter_kernel[additive_cutoff + 1:] = 1.0
coeffs_filtered = coeffs * filter_kernel

layer_name = 'high_pass_signal'
f_high = get_high_pass_signal_by_index(
    nk,
    signal_layer='composite_min',
    N=nk_results[idx]['N'],
)
nk.attach(name=layer_name, values=f_high, dtype='numeric')

G = nk.graph
pos = nx.spring_layout(G, seed=2)
node_values = np.asarray(
    [G.nodes[n]['fitness_composite_min'] for n in G.nodes()],
    dtype=float,
)
signal_norm = mpl.colors.Normalize(vmin=node_values.min(), vmax=node_values.max())

fig_graph, ax_graph = plt.subplots(figsize=(1.75, 1.25))
nx.draw_networkx_edges(G, pos, ax=ax_graph, alpha=0.35, edge_color='gray')
nodes = nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax_graph,
    node_size=75,
    node_color=node_values,
    cmap='coolwarm',
    vmin=signal_norm.vmin,
    vmax=signal_norm.vmax,
    edgecolors='black',
    linewidths=1,
)
cbar_graph = fig_graph.colorbar(nodes, fraction=0.046, pad=0.04)
cbar_graph.ax.tick_params()
ax_graph.axis('off')
fig_graph.tight_layout()
fig_graph.savefig('../figures/figure_3/composite_min_signal_example.pdf', bbox_inches='tight')
plt.show()

mode_idx = np.arange(len(coeffs))
coeff_mag = np.abs(coeffs)
coeff_filtered_mag = np.abs(coeffs_filtered)
ymax = max(coeff_mag.max(), coeff_filtered_mag.max())
ymax = 1.05 * ymax if ymax > 0 else 1.0

fig_spec, (ax_full, ax_filtered) = plt.subplots(
    2,
    1,
    figsize=(3.1, 2),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for ax in (ax_full, ax_filtered):
    ax.axvspan(-0.5, additive_cutoff + 0.5, color='lightgray', alpha=0.6, zorder=0)
    ax.axvline(additive_cutoff + 0.5, color='black', linestyle='--', linewidth=0.9)
    ax.set_ylim(0, ymax)
    ax.tick_params(axis='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax_full.bar(mode_idx, coeff_mag, color='0.6', edgecolor='none', width=0.85)

ax_filtered.bar(mode_idx, coeff_filtered_mag, color='black', edgecolor='none', width=0.85)


fig_spec.savefig('../figures/figure_3/gft_high_pass_filter_example.pdf', bbox_inches='tight')
plt.show()

# %%
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

# %%
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
        labels=["Switching", "Internal"],
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

# %%
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

# %%
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

# %% [markdown]
# ## RBD minimum-operator reconstruction
#
# Copied and focused from `the copied source/figure_notebooks_rev snapshot/Figure_4_and_SI_LYCoV.ipynb`.

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

# %%
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

# %%
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

# SEARCH_TAG: RBD_MIN_OPERATOR_ARTIFACT_EXPORT
rbd_min_outdir = Path('../figures/si_figures/si_figure_rbd_min_operator')
rbd_min_outdir.mkdir(parents=True, exist_ok=True)
rbd_min_summary_csv = rbd_min_outdir / 'rbd_min_operator_proxy_summary.csv'
rbd_min_pdf = rbd_min_outdir / 'rbd_min_operator_proxy_reconstruction.pdf'
rbd_min_png = rbd_min_outdir / 'rbd_min_operator_proxy_reconstruction.png'
min_summary_df.to_csv(rbd_min_summary_csv, index=False)
plt.tight_layout()
fig.savefig(rbd_min_pdf, bbox_inches='tight')
fig.savefig(rbd_min_png, dpi=300, bbox_inches='tight')
plt.show()
print(f'RBD min-operator proxy summary written to: {rbd_min_summary_csv}')
print(f'RBD min-operator proxy figure written to: {rbd_min_pdf}')
print(f'RBD min-operator proxy figure written to: {rbd_min_png}')

# %% [markdown]
# ## Soft-min and multiplicative Figure 5 operator controls
#
# Recreates the Figure 5 panel layout using the already collected soft-min
# and multiplicative-control experiment outputs from the Figures 4-5 composition experiment.

# %%
exec(Path("render_operator_control_figure5.py").read_text(encoding="utf-8"))
