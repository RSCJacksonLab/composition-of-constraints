# Exported from analysis.ipynb

# %%
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
# ## Stability DMS tMAP and spectral cutset panels
#
# Copied and focused from `the copied source/figure_notebooks_rev snapshot/Figure_2_reviewer_checks.ipynb`.

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
import matplotlib.pyplot as plt

# %%
# Reload results
out_pkl = processed_dir / "stability_dms" / "megascale_folding_tmap.pkl"
if not out_pkl.exists():
    out_pkl = Path("../data_files/protein_gym/DMS_assays_substitutions/megascale_folding_tmap.pkl")

with open(out_pkl, "rb") as f:
    results = pickle.load(f)

# %%
from pathlib import Path
import re
import requests
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, percentileofscore, spearmanr, pearsonr, linregress, f as f_dist, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DMS_DIR = Path('../data_files/megascale_folding')
PDB_CACHE_DIR = Path('../data_files/pdb_cache')
PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_dms_tmap_table(results, apply_poor_fit_filter=True, clamp_ci_to_t=True, verbose=True):
    rows = []
    for r in results:
        file_name = r['file']
        dataset = '-'.join(file_name.replace('.csv', '').split('_')[0:2])
        rows.append({
            'dataset': dataset,
            'file': file_name,
            'filepath': str(DMS_DIR / file_name),
            'n_sequences': int(r.get('n_sequences', np.nan)),
            't': float(r['tmap']['t_map']),
            't_lo': float(r['tmap']['t_lower_confidence_interval']),
            't_hi': float(r['tmap']['t_upper_confidence_interval']),
        })

    table = pd.DataFrame(rows)
    n_raw = len(table)

    for c in ['t', 't_lo', 't_hi']:
        table[c] = pd.to_numeric(table[c], errors='coerce')

    table = table.replace([np.inf, -np.inf], np.nan).dropna(subset=['t', 't_lo', 't_hi']).reset_index(drop=True)
    n_after_finite = len(table)

    lo = np.minimum(table['t_lo'].to_numpy(), table['t_hi'].to_numpy())
    hi = np.maximum(table['t_lo'].to_numpy(), table['t_hi'].to_numpy())
    table['t_lo'] = lo
    table['t_hi'] = hi

    eps = 1e-300
    poor_fit = ((table['t_hi'] / np.maximum(table['t_lo'], eps)) >= 10.0) & (table['t_hi'] > 1.0)
    table['poor_fit'] = poor_fit

    if apply_poor_fit_filter:
        table = table.loc[~poor_fit].reset_index(drop=True)

    bad_low = table['t'] < table['t_lo']
    bad_high = table['t'] > table['t_hi']
    bad = bad_low | bad_high
    table['ci_was_adjusted'] = bad

    if clamp_ci_to_t:
        table.loc[bad_low, 't_lo'] = table.loc[bad_low, 't']
        table.loc[bad_high, 't_hi'] = table.loc[bad_high, 't']

    if verbose:
        n_poor = int(poor_fit.sum())
        n_ci = int(bad.sum())
        print(
            f'Table build counts: raw={n_raw}, after_finite={n_after_finite}, '
            f'poor_fit_removed={n_poor if apply_poor_fit_filter else 0}, '
            f'ci_adjusted={n_ci}, final={len(table)}'
        )
        if n_ci > 0:
            print('CI-adjusted examples (first 10):')
            print(table.loc[bad, ['dataset', 't', 't_lo', 't_hi']].head(10))

    table = table.sort_values('t', ascending=False).reset_index(drop=True)
    return table


def regression_global_f_pvalue(y_true, y_pred, n_features):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    p = int(n_features)

    if n <= p + 1 or n < 3:
        return np.nan, np.nan

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_reg = ss_tot - ss_res

    if not np.isfinite(ss_tot) or ss_tot <= 0:
        return np.nan, np.nan

    df_model = p
    df_resid = n - p - 1

    if df_model <= 0 or df_resid <= 0:
        return np.nan, np.nan

    ms_reg = ss_reg / df_model
    ms_res = ss_res / df_resid

    if ms_res <= 0:
        if ms_reg > 0:
            return np.inf, 0.0
        return np.nan, np.nan

    f_stat = ms_reg / ms_res
    p_value = float(f_dist.sf(f_stat, df_model, df_resid))
    return float(f_stat), p_value


dms_tmap_df = build_dms_tmap_table(results, apply_poor_fit_filter=False, clamp_ci_to_t=True, verbose=True)
print(f'Retained domains after Figure 2 filtering: {len(dms_tmap_df)}')
dms_tmap_df.head()

# %%
# Load promoted outputs from the Figure 2 stability-DMS experiment.
import ast

_stability_dir = processed_dir / "stability_dms"
_perm_path = _stability_dir / "folding_permutation_short.df"
if _perm_path.exists():
    perm_results_df = pd.read_csv(_perm_path)
    if "perm_t_maps" in perm_results_df.columns and "null_samples" not in perm_results_df.columns:
        perm_results_df["null_samples"] = perm_results_df["perm_t_maps"].apply(ast.literal_eval)

_megascale_table = _stability_dir / "megascale_folding_tmap_table.csv"
if _megascale_table.exists():
    dms_tmap_df = pd.read_csv(_megascale_table)

processed_tables = {}
for _path in _stability_dir.glob("*.csv"):
    try:
        processed_tables[_path.stem] = pd.read_csv(_path)
    except Exception:
        pass

# %%
from scipy.stats import gaussian_kde

fig, axes = plt.subplots(
    len(perm_results_df), 1,
    figsize=(4, 1.75 * len(perm_results_df)),
    squeeze=False
)

for ax, row in zip(axes.ravel(), perm_results_df.itertuples(index=False)):
    samples = np.asarray(row.null_samples, dtype=float)
    samples = samples[np.isfinite(samples)]

    if len(samples) >= 2 and np.std(samples) > 0:
        x_grid = np.linspace(samples.min(), samples.max(), 300)
        kde = gaussian_kde(samples)
        y_kde = kde(x_grid)

        ax.plot(x_grid, y_kde, color='black', lw=1.5, label='null KDE')
        ax.fill_between(x_grid, 0, y_kde, color='lightgrey', alpha=0.7)
    else:
        # Fallback if too few/degenerate samples
        ax.axvline(samples[0] if len(samples) else row.null_mean_t_map, color='black', lw=1.5, label='null KDE (degenerate)')
    ax.grid(True, linestyle="--")
    ax.axvline(row.observed_t_map, color='crimson', lw=2, label='observed $t_{MAP}$')
    ax.set_xlabel(r'Null $t_{MAP}$')
    ax.set_ylabel('density')
    ax.legend(frameon=False)
    

plt.tight_layout()
plt.savefig('../figures/figure_2/permutation_test_kde.pdf')
plt.show()

# %%
# SEARCH_TAG: TMAP_BAR_WITH_PERM_SIGNIFICANCE
# Remake t_MAP barplot without permutation-significance stars.
if 'dms_tmap_df' not in globals():
    raise RuntimeError('Run the t_MAP table cell first (requires dms_tmap_df).')

# Build plotting frame (same core structure as the original bar plot).
df = dms_tmap_df.copy().sort_values('t', ascending=False).reset_index(drop=True)

t = df['t'].to_numpy(dtype=float)
yerr = np.vstack([
    np.clip(t - df['t_lo'].to_numpy(dtype=float), a_min=0.0, a_max=None),
    np.clip(df['t_hi'].to_numpy(dtype=float) - t, a_min=0.0, a_max=None),
])

# -----------------------------
# Plot (single bar call; x aligned with errorbar)
# -----------------------------
x = np.arange(len(df))

fig, ax = plt.subplots(figsize=(8, 2.75))

ax.bar(
    x,
    t,
    color='lightgrey',
    edgecolor='black',
    linewidth=0.75,
    zorder=2
)

ax.errorbar(
    x=x,
    y=t,
    yerr=yerr,
    fmt='none',
    ecolor='black',
    elinewidth=1,
    capsize=2,
    zorder=3
)

step = 1  # change to 3 if you truly want every 3rd label
ax.set_xticks(x[::step])
ax.set_xticklabels(df['dataset'].iloc[::step], rotation=90, fontsize=8)

ax.set_ylabel(r'$t_{\mathrm{MAP}}$')
# ax.set_yscale('log')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', ls='--', c='0.85', zorder=1)

plt.tight_layout()
plt.savefig('../figures/figure_2/tmap_vs_dms.pdf')
plt.show()

# %%
import re
MUT_TOKEN_RE = re.compile(r'([A-Z])(\d+)([A-Z\*])')


def load_domain_dataframe(file_name):
    domain_df = pd.read_csv(DMS_DIR / file_name)
    domain_df = domain_df.replace([np.inf, -np.inf], np.nan)
    domain_df = domain_df.dropna(subset=['mutated_sequence', 'DMS_score']).reset_index(drop=True)
    return domain_df


def build_hamming_landscape_from_df(domain_df):
    sequences = [fl.BaseNumpySequence(seq) for seq in domain_df['mutated_sequence']]
    fitness = domain_df['DMS_score'].to_numpy(dtype=float)

    landscape = fl.FitnessLandscape.build(
        sequences,
        graph='hamming',
    )

    if not nx.is_connected(landscape.graph):
        raise ValueError('Sampled landscape is disconnected.')

    return landscape, fitness


def compute_tmap_on_landscape_values(
    landscape,
    values,
    eigenvalues=None,
    eigenvectors=None,
    layer_name='tmp_layer',
    detach=True,
    t_min=1e-10,
    t_max=1e2,
    prior='uniform',
):
    values = np.asarray(values, dtype=float)
    landscape.attach(name=layer_name, values=values, dtype='numeric')
    landscape.view(layer_name)

    tmap_kwargs = {
        't_min': t_min,
        't_max': t_max,
        'prior': prior,
    }

    if eigenvalues is not None and eigenvectors is not None:
        tmap_kwargs['_eigenvalues'] = np.asarray(eigenvalues, dtype=float)
        tmap_kwargs['_eigenvectors'] = np.asarray(eigenvectors, dtype=float)

    try:
        tmap_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(
            landscape,
            **tmap_kwargs,
        )
    finally:
        if detach:
            landscape.detach(layer_name)

    return float(tmap_res['t_map'])

# %%
# SEARCH_TAG: GRAPH_SIZE_DOWNSAMPLE_TMAP_ROBUSTNESS
# Re-run the legacy graph-size robustness check: downsample every DMS graph to
# the smallest domain size and recompute t_MAP on connected sampled subgraphs.
SUBSAMPLE_REPS = 3
MAX_ATTEMPTS_FACTOR = 20
subsample_rng = np.random.default_rng(2026)

downsample_outdir = Path('../figures/SI_figures/SI_figure_DMS')
downsample_outdir.mkdir(parents=True, exist_ok=True)
downsample_fig_path = downsample_outdir / 'graph_size_sampling_vs_tmap.pdf'
downsample_domain_csv = downsample_outdir / 'graph_size_sampling_vs_tmap_domain_summary.csv'
downsample_stats_csv = downsample_outdir / 'graph_size_sampling_vs_tmap_stats.csv'

target_n = int(pd.to_numeric(dms_tmap_df['n_sequences'], errors='coerce').min())
print(f'Graph-size downsample target n: {target_n}')

subsample_rows = []
for row in tqdm(dms_tmap_df.itertuples(index=False), total=len(dms_tmap_df), desc='Downsample DMS graphs'):
    file_name = str(row.file)
    dataset = str(row.dataset)
    observed_t = float(row.t)
    domain_df = load_domain_dataframe(file_name)

    reps_done = 0
    attempts = 0
    max_attempts = SUBSAMPLE_REPS * MAX_ATTEMPTS_FACTOR
    last_error = None

    while reps_done < SUBSAMPLE_REPS and attempts < max_attempts:
        attempts += 1
        sample_idx = subsample_rng.choice(len(domain_df), size=target_n, replace=False)
        sampled_df = domain_df.iloc[sample_idx].reset_index(drop=True)
        try:
            sampled_landscape, sampled_fitness = build_hamming_landscape_from_df(sampled_df)
            sampled_t = compute_tmap_on_landscape_values(
                sampled_landscape,
                sampled_fitness,
                layer_name=f'downsample_{dataset}_{reps_done + 1}',
                detach=True,
            )
        except Exception as exc:
            last_error = str(exc)
            continue

        reps_done += 1
        subsample_rows.append({
            'dataset': dataset,
            'file': file_name,
            'replicate': int(reps_done),
            'target_n_sequences': int(target_n),
            'source_n_sequences': int(len(domain_df)),
            'observed_t_map': observed_t,
            'downsampled_t_map': float(sampled_t),
            'attempts_used': int(attempts),
            'status': 'ok',
            'error': '',
        })

    if reps_done < SUBSAMPLE_REPS:
        subsample_rows.append({
            'dataset': dataset,
            'file': file_name,
            'replicate': np.nan,
            'target_n_sequences': int(target_n),
            'source_n_sequences': int(len(domain_df)),
            'observed_t_map': observed_t,
            'downsampled_t_map': np.nan,
            'attempts_used': int(attempts),
            'status': 'failed_to_sample_connected_subgraph',
            'error': last_error or 'no connected sample found',
        })

subsample_tmap_df = pd.DataFrame(subsample_rows)
ok_subsample_df = subsample_tmap_df.loc[subsample_tmap_df['status'] == 'ok'].copy()
if len(ok_subsample_df) == 0:
    raise RuntimeError('Graph-size downsample check produced no connected sampled landscapes.')

downsample_domain_summary = (
    ok_subsample_df
    .groupby(['dataset', 'file', 'target_n_sequences', 'source_n_sequences', 'observed_t_map'], as_index=False)
    .agg(
        n_successful_replicates=('downsampled_t_map', 'count'),
        downsampled_t_map_mean=('downsampled_t_map', 'mean'),
        downsampled_t_map_std=('downsampled_t_map', 'std'),
        downsampled_t_map_min=('downsampled_t_map', 'min'),
        downsampled_t_map_max=('downsampled_t_map', 'max'),
    )
)
downsample_domain_summary['delta_downsampled_minus_observed'] = (
    downsample_domain_summary['downsampled_t_map_mean'] - downsample_domain_summary['observed_t_map']
)
failed_domains = subsample_tmap_df.loc[subsample_tmap_df['status'] != 'ok', ['dataset', 'file', 'status', 'error']]
if len(failed_domains) > 0:
    print('Downsample domains with incomplete connected samples:')
    display(failed_domains)

valid_rank_df = downsample_domain_summary.dropna(subset=['observed_t_map', 'downsampled_t_map_mean']).copy()
if len(valid_rank_df) >= 2:
    spear_rho, spear_p = spearmanr(valid_rank_df['observed_t_map'], valid_rank_df['downsampled_t_map_mean'])
    pear_r, pear_p = pearsonr(valid_rank_df['observed_t_map'], valid_rank_df['downsampled_t_map_mean'])
else:
    spear_rho = spear_p = pear_r = pear_p = np.nan

try:
    if len(valid_rank_df) > 0 and not np.allclose(
        valid_rank_df['observed_t_map'].to_numpy(dtype=float),
        valid_rank_df['downsampled_t_map_mean'].to_numpy(dtype=float),
    ):
        wilcoxon_stat, wilcoxon_p = wilcoxon(
            valid_rank_df['downsampled_t_map_mean'],
            valid_rank_df['observed_t_map'],
            zero_method='wilcox',
        )
    else:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan
except ValueError:
    wilcoxon_stat, wilcoxon_p = np.nan, np.nan

downsample_stats_df = pd.DataFrame([
    {
        'target_n_sequences': int(target_n),
        'subsample_reps': int(SUBSAMPLE_REPS),
        'n_domains_with_successful_samples': int(len(valid_rank_df)),
        'spearman_rho': float(spear_rho) if np.isfinite(spear_rho) else np.nan,
        'spearman_p': float(spear_p) if np.isfinite(spear_p) else np.nan,
        'pearson_r': float(pear_r) if np.isfinite(pear_r) else np.nan,
        'pearson_p': float(pear_p) if np.isfinite(pear_p) else np.nan,
        'wilcoxon_stat': float(wilcoxon_stat) if np.isfinite(wilcoxon_stat) else np.nan,
        'wilcoxon_p': float(wilcoxon_p) if np.isfinite(wilcoxon_p) else np.nan,
        'mean_delta_downsampled_minus_observed': float(valid_rank_df['delta_downsampled_minus_observed'].mean()),
        'median_delta_downsampled_minus_observed': float(valid_rank_df['delta_downsampled_minus_observed'].median()),
    }
])

downsample_domain_summary.to_csv(downsample_domain_csv, index=False)
downsample_stats_df.to_csv(downsample_stats_csv, index=False)

display(downsample_domain_summary)
display(downsample_stats_df)

fig, ax = plt.subplots(figsize=(4.2, 3.3))
plot_df = valid_rank_df.sort_values('observed_t_map').copy()
yerr = plot_df['downsampled_t_map_std'].fillna(0.0).to_numpy(dtype=float)
ax.errorbar(
    plot_df['observed_t_map'],
    plot_df['downsampled_t_map_mean'],
    yerr=yerr,
    fmt='o',
    color='black',
    ecolor='0.65',
    elinewidth=0.8,
    capsize=2,
    markersize=4,
)
lims = [
    float(np.nanmin([plot_df['observed_t_map'].min(), plot_df['downsampled_t_map_mean'].min()])),
    float(np.nanmax([plot_df['observed_t_map'].max(), plot_df['downsampled_t_map_mean'].max()])),
]
pad = 0.05 * (lims[1] - lims[0] if lims[1] > lims[0] else 1.0)
lims = [lims[0] - pad, lims[1] + pad]
ax.plot(lims, lims, color='0.4', linestyle='--', linewidth=1.0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel(r'Observed $t_{MAP}$')
ax.set_ylabel(r'Downsampled mean $t_{MAP}$')
ax.set_title(f'n={target_n} sequences per domain')
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
fig.savefig(downsample_fig_path)
plt.show()

print(f'Graph-size downsample figure written to: {downsample_fig_path}')
print(f'Graph-size downsample domain summary written to: {downsample_domain_csv}')
print(f'Graph-size downsample stats written to: {downsample_stats_csv}')

# %%
# SEARCH_TAG: SPECTRAL_PARTITION_BOUNDARY_TMAP
# Experiment 9a: spectral partitioning on the fitness-weighted Hamming graph.
from pathlib import Path
import importlib
import spectral_boundary_experiments as sbe

importlib.reload(sbe)

spectral_partition_results = sbe.run_spectral_partition_boundary_tmap(
    dms_tmap_df=dms_tmap_df,
    load_domain_dataframe=load_domain_dataframe,
    build_hamming_landscape_from_df=build_hamming_landscape_from_df,
    compute_tmap_on_landscape_values=compute_tmap_on_landscape_values,
    outdir=Path('../figures/SI_figures/SI_figure_DMS'),
    load_streamed_frames=False,
)

spectral_partition_domain_df = spectral_partition_results['domain_df']
spectral_partition_assoc_df = spectral_partition_results['assoc_df']
spectral_partition_domain_csv = spectral_partition_results['domain_csv']
spectral_partition_edge_csv = spectral_partition_results['edge_csv']
spectral_partition_component_csv = spectral_partition_results['component_csv']
spectral_partition_edge_df = None
spectral_partition_component_df = None

print('Domains processed:', len(spectral_partition_domain_df))
print('Successful spectral partitions:', int((spectral_partition_domain_df['status'] == 'ok').sum()))
display(
    spectral_partition_domain_df[
        [
            'dataset', 'observed_t_map', 'status', 'partition_size_left', 'partition_size_right',
            'n_cut_edges', 'cut_energy_enrichment', 'post_partition_t_map',
            'post_partition_t_map_gain', 'post_partition_t_map_ratio',
            'partition_tmap_node_coverage', 'error',
        ]
    ]
)
display(spectral_partition_assoc_df)
print(f"Spectral partition domain summary written to: {spectral_partition_domain_csv}")
print(f"Spectral partition edge details streamed to: {spectral_partition_edge_csv}")
print(f"Spectral partition component summary streamed to: {spectral_partition_component_csv}")
print(f"Spectral partition association summary written to: {spectral_partition_results['assoc_csv']}")
print(f"Spectral partition figure written to: {spectral_partition_results['fig_path']}")

# %%
# Figure plot for spectral partitioning results: Single most rugged and single smoothest example, network graphs showing the partitioning and cut edges, and t_MAP values before and after partitioning.

from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import spectral_boundary_experiments as sbe

if 'spectral_partition_domain_df' in globals():
    _spectral_examples_df = spectral_partition_domain_df.copy()
elif 'spectral_partition_domain_csv' in globals():
    _spectral_examples_df = pd.read_csv(spectral_partition_domain_csv)
else:
    raise RuntimeError('Run Experiment 9a first (requires spectral_partition_domain_df or spectral_partition_domain_csv).')

if '_reviewer_get_graph_layout' not in globals():
    def _reviewer_get_graph_layout(graph, cache_key, seed=0):
        cache = globals().setdefault('_reviewer_graph_layout_cache', {})
        if cache_key in cache:
            return cache[cache_key]

        pos = None
        for module_name in ('networkx.drawing.nx_agraph', 'networkx.drawing.nx_pydot'):
            try:
                graphviz_layout = __import__(module_name, fromlist=['graphviz_layout']).graphviz_layout
                pos = graphviz_layout(graph, prog='sfdp')
                break
            except Exception:
                continue

        if pos is None:
            try:
                pos = nx.spectral_layout(graph, dim=2)
                pos = nx.spring_layout(graph, seed=seed, pos=pos, iterations=40)
            except Exception:
                pos = nx.spring_layout(graph, seed=seed, iterations=80)

        pos = {int(node): np.asarray(coords, dtype=float)[:2] for node, coords in pos.items()}
        cache[cache_key] = pos
        return pos

work_df = _spectral_examples_df.loc[
    _spectral_examples_df['status'].eq('ok')
    & np.isfinite(_spectral_examples_df['observed_t_map'].to_numpy(dtype=float)),
].copy()
if len(work_df) < 2:
    raise RuntimeError('Need at least two successful spectral partitions for example plots.')

work_df = work_df.sort_values(['observed_t_map', 'dataset', 'file']).reset_index(drop=True)
example_rows = [
    ('Smoothest', work_df.iloc[0]),
    ('Most rugged', work_df.iloc[-1]),
]
side_palette = {0: '#4C78A8', 1: '#E45756'}
example_data = []

for label, row in example_rows:
    domain_df = load_domain_dataframe(row.file).reset_index(drop=True)
    landscape, fitness = build_hamming_landscape_from_df(domain_df)
    labels, edge_records, _ = sbe._spectral_build_cut(landscape.graph, fitness)
    cut_edges = [
        (int(rec['u']), int(rec['v']))
        for rec in edge_records
        if bool(rec['is_cut'])
    ]
    pos = _reviewer_get_graph_layout(
        landscape.graph,
        cache_key=(str(row.file), 'spectral_partition_boundary'),
        seed=1,
    )

    example_data.append(
        {
            'label': label,
            'dataset': str(row.dataset),
            'observed_t_map': float(row.observed_t_map),
            'post_partition_t_map': float(row.post_partition_t_map),
            'post_partition_t_map_ratio': float(row.post_partition_t_map_ratio),
            'side_0_t_map': float(row.side_0_t_map),
            'side_1_t_map': float(row.side_1_t_map),
            'graph': landscape.graph,
            'labels': labels,
            'cut_edges': cut_edges,
            'pos': pos,
        }
    )

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10.8, 8.4),
    gridspec_kw={'width_ratios': [1.5, 1.0]},
)

for row_idx, ex in enumerate(example_data):
    ax_graph = axes[row_idx, 0]
    ax_bar = axes[row_idx, 1]
    graph = ex['graph']
    node_order = [int(node) for node in sorted(graph.nodes())]
    node_colors = [side_palette[int(ex['labels'][node])] for node in node_order]
    node_size = float(np.clip(250.0 / np.sqrt(max(len(node_order), 1)), 3.0, 10.0))

    if ex['cut_edges']:
        nx.draw_networkx_edges(
            graph,
            ex['pos'],
            ax=ax_graph,
            edgelist=ex['cut_edges'],
            edge_color='black',
            width=0.7,
            alpha=0.35,
        )
    nx.draw_networkx_nodes(
        graph,
        ex['pos'],
        ax=ax_graph,
        nodelist=node_order,
        node_color=node_colors,
        node_size=node_size,
        linewidths=0.0,
        alpha=0.95,
    )
    ax_graph.set_title(
        f"{ex['label']}: {ex['dataset']}\ncut edges={len(ex['cut_edges'])}, observed $t_{{MAP}}$={ex['observed_t_map']:.2f}",
        fontsize=10,
    )
    ax_graph.set_axis_off()

    bar_labels = ['whole', 'side 0', 'side 1', 'weighted']
    bar_values = [
        ex['observed_t_map'],
        ex['side_0_t_map'],
        ex['side_1_t_map'],
        ex['post_partition_t_map'],
    ]
    bar_colors = ['#6B6B6B', side_palette[0], side_palette[1], '#111111']
    bar_x = np.arange(len(bar_labels))

    ax_bar.bar(bar_x, bar_values, color=bar_colors, edgecolor='black', linewidth=0.7)
    ax_bar.axhline(ex['observed_t_map'], color='0.45', linestyle='--', linewidth=1.0)
    ax_bar.set_xticks(bar_x)
    ax_bar.set_xticklabels(bar_labels, rotation=25, ha='right')
    ax_bar.set_ylabel(r'$t_{\mathrm{MAP}}$')
    ax_bar.set_title(
        f"gain={ex['post_partition_t_map'] - ex['observed_t_map']:+.2f}, ratio={ex['post_partition_t_map_ratio']:.2f}",
        fontsize=10,
    )
    ax_bar.grid(axis='y', linestyle='--', alpha=0.3)

legend_handles = [
    Patch(facecolor=side_palette[0], edgecolor='none', label='Partition side 0'),
    Patch(facecolor=side_palette[1], edgecolor='none', label='Partition side 1'),
    Line2D([0], [0], color='black', linewidth=1.0, label='Cut edge'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_dir = Path('../figures/figure_2')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'spectral_partition_examples_most_vs_smoothest.pdf'
plt.savefig(out_path)
plt.show()
print(f'Figure written to: {out_path}')

# %% [markdown]
# ## Node-label permutation null collection for spectral bipartitioning
#
# Collect the Figure 2 stability-DMS experiment null chunks in which node fitness labels were permuted before recomputing the fitness-weighted spectral bipartition.

# %%
# Collect node-label permutation null chunks for the biased spectral bipartitioning experiment.
null_dir = processed_dir / "stability_dms" / "node_label_permutation_nulls"
spectral_null_paths = sorted(null_dir.glob("chunk_*_node_label_spectral_bipartition_null.csv"))

if spectral_null_paths:
    spectral_bipartition_null_df = pd.concat(
        [pd.read_csv(path) for path in spectral_null_paths],
        ignore_index=True,
    )
    spectral_bipartition_null_path = table_dir / "spectral_bipartition_node_label_null.csv"
    spectral_bipartition_null_df.to_csv(spectral_bipartition_null_path, index=False)

    if "spectral_partition_domain_df" in globals():
        observed_spectral_df = spectral_partition_domain_df.copy()
    else:
        observed_spectral_df = pd.read_csv(processed_dir / "stability_dms" / "spectral_partition_boundary_tmap_domain_summary.csv")

    observed_spectral_df = observed_spectral_df.loc[observed_spectral_df["status"].eq("ok")].copy()
    ok_spectral_null = spectral_bipartition_null_df.loc[spectral_bipartition_null_df["status"].eq("ok")].copy()

    empirical_rows = []
    for row in observed_spectral_df.itertuples(index=False):
        sub = ok_spectral_null.loc[ok_spectral_null["file"].eq(row.file)].copy()
        n_null = int(len(sub))
        if n_null == 0:
            continue
        empirical_rows.append(
            {
                "dataset": row.dataset,
                "file": row.file,
                "n_null": n_null,
                "observed_cut_energy_enrichment": float(row.cut_energy_enrichment),
                "null_cut_energy_enrichment_median": float(sub["cut_energy_enrichment"].median()),
                "empirical_p_cut_energy_enrichment_ge_observed": float((np.sum(sub["cut_energy_enrichment"] >= row.cut_energy_enrichment) + 1) / (n_null + 1)),
                "observed_cut_energy_fraction": float(row.cut_energy_fraction),
                "null_cut_energy_fraction_median": float(sub["cut_energy_fraction"].median()),
                "empirical_p_cut_energy_fraction_ge_observed": float((np.sum(sub["cut_energy_fraction"] >= row.cut_energy_fraction) + 1) / (n_null + 1)),
                "observed_partition_balance": float(row.partition_balance),
                "null_partition_balance_median": float(sub["partition_balance"].median()),
            }
        )

    spectral_bipartition_empirical_df = pd.DataFrame(empirical_rows)
    spectral_bipartition_empirical_path = table_dir / "spectral_bipartition_node_label_null_empirical_p.csv"
    spectral_bipartition_empirical_df.to_csv(spectral_bipartition_empirical_path, index=False)

    null_by_perm = ok_spectral_null.groupby("perm_idx", as_index=False).agg(
        median_cut_energy_enrichment=("cut_energy_enrichment", "median"),
        median_cut_energy_fraction=("cut_energy_fraction", "median"),
        median_cut_edge_fraction=("cut_edge_fraction", "median"),
        median_partition_balance=("partition_balance", "median"),
    )
    observed_median_enrichment = float(observed_spectral_df["cut_energy_enrichment"].median())
    observed_median_cut_fraction = float(observed_spectral_df["cut_energy_fraction"].median())

    spectral_bipartition_null_summary = {
        "null_type": "node-label permutation before recomputing fitness-weighted spectral bipartitions",
        "n_null_rows": int(len(ok_spectral_null)),
        "n_null_permutations": int(ok_spectral_null["perm_idx"].nunique()),
        "n_domains": int(observed_spectral_df["file"].nunique()),
        "observed_median_cut_energy_enrichment": observed_median_enrichment,
        "null_median_of_permutation_median_cut_energy_enrichment": float(null_by_perm["median_cut_energy_enrichment"].median()),
        "empirical_p_median_cut_energy_enrichment_ge_observed": float((np.sum(null_by_perm["median_cut_energy_enrichment"] >= observed_median_enrichment) + 1) / (len(null_by_perm) + 1)),
        "observed_median_cut_energy_fraction": observed_median_cut_fraction,
        "null_median_of_permutation_median_cut_energy_fraction": float(null_by_perm["median_cut_energy_fraction"].median()),
        "empirical_p_median_cut_energy_fraction_ge_observed": float((np.sum(null_by_perm["median_cut_energy_fraction"] >= observed_median_cut_fraction) + 1) / (len(null_by_perm) + 1)),
        "source_files": [str(path) for path in spectral_null_paths],
    }
    spectral_bipartition_null_summary_path = table_dir / "spectral_bipartition_node_label_null_summary.json"
    with spectral_bipartition_null_summary_path.open("w") as handle:
        json.dump(spectral_bipartition_null_summary, handle, indent=2, sort_keys=True)

    print(f"Collected {len(spectral_null_paths)} spectral bipartition null chunks from {null_dir}")
else:
    spectral_bipartition_null_df = pd.DataFrame()
    print(f"No node-label spectral bipartition null chunks found in {null_dir}; skipping null collection.")

# %%
# SEARCH_TAG: SPECTRAL_CUT_DISTANCE_PROFILE
# Experiment 9b: edge energy as a function of graph distance from the spectral cut.
from pathlib import Path
import importlib
import spectral_boundary_experiments as sbe

importlib.reload(sbe)

spectral_profile_results = sbe.run_spectral_cut_distance_profile(
    spectral_partition_domain_df=spectral_partition_domain_df,
    spectral_partition_edge_csv=spectral_partition_edge_csv,
    outdir=Path('../figures/SI_figures/SI_figure_DMS'),
)

spectral_shell_profile_df = spectral_profile_results['profile_df']
spectral_shell_domain_df = spectral_profile_results['domain_df']
spectral_shell_assoc_df = spectral_profile_results['assoc_df']

print('Domains with shell profiles:', len(spectral_shell_domain_df))
if len(spectral_shell_assoc_df) > 0:
    display(spectral_shell_assoc_df)
display(spectral_shell_domain_df.sort_values('boundary_peak_enrichment', ascending=False))
print(f"Shell-profile summary written to: {spectral_profile_results['profile_csv']}")
print(f"Shell-domain summary written to: {spectral_profile_results['domain_csv']}")
print(f"Shell-profile figure written to: {spectral_profile_results['fig_path']}")

# %%
# Figure 2 plot for spectral cut distance: heatmap-only version sized for the
# compact bottom-row slot between permutation nulls and the graph example.

from pathlib import Path
import json

if 'spectral_shell_profile_df' in globals():
    _spectral_heatmap_profile_df = spectral_shell_profile_df.copy()
else:
    _spectral_heatmap_profile_df = pd.read_csv(
        Path('../figures/SI_figures/SI_figure_DMS/spectral_cut_distance_shell_profiles.csv')
    )

if 'spectral_partition_domain_df' in globals():
    _spectral_heatmap_domain_df = spectral_partition_domain_df.copy()
elif 'spectral_profile_results' in globals():
    _spectral_heatmap_domain_df = spectral_profile_results['domain_df'].copy()
else:
    _spectral_heatmap_domain_df = pd.read_csv(
        Path('../figures/SI_figures/SI_figure_DMS/spectral_cut_distance_domain_metrics.csv')
    )

shell_levels = sorted(int(shell) for shell in _spectral_heatmap_profile_df['shell_plot'].dropna().unique())
heatmap_df = (
    _spectral_heatmap_profile_df.pivot_table(
        index='file',
        columns='shell_plot',
        values='mean_edge_energy_norm',
        aggfunc='mean',
    )
    .reindex(columns=shell_levels)
    .merge(
        _spectral_heatmap_domain_df[['file', 'dataset', 'observed_t_map']].drop_duplicates(),
        on='file',
        how='left',
    )
    .sort_values('observed_t_map', ascending=False)
    .reset_index(drop=True)
)
heatmap_values = heatmap_df[shell_levels].to_numpy(dtype=float)

heatmap_cmap = plt.cm.viridis.copy()
heatmap_cmap.set_bad('white')
heatmap_norm = plt.Normalize(
    vmin=0.0,
    vmax=float(np.nanmax(heatmap_values)),
)

HEATMAP_WIDTH_IN = 3.3
HEATMAP_HEIGHT_IN = 2.19

fig = plt.figure(figsize=(HEATMAP_WIDTH_IN, HEATMAP_HEIGHT_IN))
ax = fig.add_axes([0.16, 0.23, 0.66, 0.65])
cax = fig.add_axes([0.86, 0.23, 0.045, 0.65])
im = ax.imshow(
    heatmap_values,
    aspect='auto',
    cmap=heatmap_cmap,
    norm=heatmap_norm,
    interpolation='nearest',
)
ax.set_xlabel('Graph distance from spectral cut', fontsize=7)
ax.set_ylabel('Domains sorted by $t_{MAP}$', fontsize=6.5)
ax.set_xticks(np.arange(len(shell_levels)))
ax.set_xticklabels(
    [
        str(shell) if shell < int(getattr(sbe, 'SPECTRAL_MAX_SHELL_PLOT', 6)) else f"{int(getattr(sbe, 'SPECTRAL_MAX_SHELL_PLOT', 6))}+"
        for shell in shell_levels
    ],
    fontsize=6.5,
)
ax.set_yticks([])
ax.tick_params(axis='x', width=0.6, length=2.5, pad=1)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Normalized mean edge energy', fontsize=6.5, labelpad=3)
cbar.ax.tick_params(labelsize=6.5, width=0.6, length=2.5, pad=1)

out_dir = Path('../figures/figure_2')
out_dir.mkdir(parents=True, exist_ok=True)
heatmap_pdf = out_dir / 'spectral_cut_distance_heatmap_all_domains.pdf'
heatmap_png = out_dir / 'spectral_cut_distance_heatmap_all_domains.png'
fig.savefig(heatmap_pdf)
fig.savefig(heatmap_png, dpi=300)
plt.close(fig)

heatmap_spec = {
    'pdf': str(heatmap_pdf),
    'png': str(heatmap_png),
    'width_inches': HEATMAP_WIDTH_IN,
    'height_inches': HEATMAP_HEIGHT_IN,
    'width_mm': HEATMAP_WIDTH_IN * 25.4,
    'height_mm': HEATMAP_HEIGHT_IN * 25.4,
    'shell_levels': shell_levels,
    'n_domains': int(heatmap_values.shape[0]),
    'row_order': 'descending observed_t_map',
    'value': 'mean edge energy in each graph-distance shell divided by the domain-wide mean edge energy',
    'colormap': 'viridis',
}
heatmap_spec_path = Path('../tables/spectral_cut_distance_heatmap_all_domains_spec.json')
with heatmap_spec_path.open('w') as handle:
    json.dump(heatmap_spec, handle, indent=2, sort_keys=True)

print(f'Figure 2 heatmap-only spectral cut distance plot written to: {heatmap_pdf}')
print(f'Heatmap panel size: {heatmap_spec["width_mm"]:.1f} x {heatmap_spec["height_mm"]:.1f} mm')

# %%
# Figure plot for spectral cut distance: Most rugged example showing the edge energy profile as a function of distance from the spectral cut, with the network graph and edges coloured by energy.

from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import spectral_boundary_experiments as sbe

if 'spectral_shell_domain_df' in globals():
    _spectral_shell_examples_df = spectral_shell_domain_df.copy()
elif 'spectral_profile_results' in globals():
    _spectral_shell_examples_df = spectral_profile_results['domain_df'].copy()
else:
    raise RuntimeError('Run Experiment 9b first (requires spectral_shell_domain_df or spectral_profile_results).')

if 'spectral_shell_profile_df' in globals():
    _spectral_shell_profile_plot_df = spectral_shell_profile_df.copy()
elif 'spectral_profile_results' in globals():
    _spectral_shell_profile_plot_df = spectral_profile_results['profile_df'].copy()
else:
    raise RuntimeError('Run Experiment 9b first (requires spectral_shell_profile_df or spectral_profile_results).')

if '_reviewer_get_graph_layout' not in globals():
    def _reviewer_get_graph_layout(graph, cache_key, seed=0):
        cache = globals().setdefault('_reviewer_graph_layout_cache', {})
        if cache_key in cache:
            return cache[cache_key]

        pos = None
        for module_name in ('networkx.drawing.nx_agraph', 'networkx.drawing.nx_pydot'):
            try:
                graphviz_layout = __import__(module_name, fromlist=['graphviz_layout']).graphviz_layout
                pos = graphviz_layout(graph, prog='sfdp')
                break
            except Exception:
                continue

        if pos is None:
            try:
                pos = nx.spectral_layout(graph, dim=2)
                pos = nx.spring_layout(graph, seed=seed, pos=pos, iterations=40)
            except Exception:
                pos = nx.spring_layout(graph, seed=seed, iterations=80)

        pos = {int(node): np.asarray(coords, dtype=float)[:2] for node, coords in pos.items()}
        cache[cache_key] = pos
        return pos

work_df = _spectral_shell_examples_df.loc[
    np.isfinite(_spectral_shell_examples_df['observed_t_map'].to_numpy(dtype=float)),
].copy()
if len(work_df) == 0:
    raise RuntimeError('No shell-profile domains available for plotting.')

work_df = work_df.sort_values(['observed_t_map', 'dataset', 'file']).reset_index(drop=True)
example_row = work_df.iloc[-1]

profile_df = _spectral_shell_profile_plot_df.loc[
    _spectral_shell_profile_plot_df['file'].astype(str).eq(str(example_row.file))
].copy()
profile_df = profile_df.sort_values('shell_plot').reset_index(drop=True)
if len(profile_df) == 0:
    raise RuntimeError('No shell profile found for the selected rugged example.')

domain_df = load_domain_dataframe(example_row.file).reset_index(drop=True)
landscape, fitness = build_hamming_landscape_from_df(domain_df)
labels, edge_records, _ = sbe._spectral_build_cut(landscape.graph, fitness)
edge_df = pd.DataFrame(edge_records)
pos = _reviewer_get_graph_layout(
    landscape.graph,
    cache_key=(str(example_row.file), 'spectral_cut_distance_energy'),
    seed=2,
)

side_palette = {0: '#4C78A8', 1: '#E45756'}
node_order = [int(node) for node in sorted(landscape.graph.nodes())]
node_colors = [side_palette[int(labels[node])] for node in node_order]
node_size = float(np.clip(250.0 / np.sqrt(max(len(node_order), 1)), 3.0, 10.0))

edges = list(zip(edge_df['u'].astype(int), edge_df['v'].astype(int)))
energy_raw = np.maximum(edge_df['edge_energy'].to_numpy(dtype=float), 1e-12)
energy_plot = np.log10(energy_raw)
cut_mask = edge_df['is_cut'].to_numpy(dtype=bool)
noncut_mask = ~cut_mask
norm = plt.Normalize(
    vmin=float(np.nanpercentile(energy_plot, 5)),
    vmax=float(np.nanpercentile(energy_plot, 99)),
)
edge_rgba = plt.cm.magma(norm(energy_plot))

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.8, 4.6),
    gridspec_kw={'width_ratios': [1.35, 1.0]},
)
ax_graph, ax_profile = axes

if np.any(noncut_mask):
    nx.draw_networkx_edges(
        landscape.graph,
        pos,
        ax=ax_graph,
        edgelist=[edges[i] for i in np.flatnonzero(noncut_mask)],
        edge_color=[edge_rgba[i] for i in np.flatnonzero(noncut_mask)],
        width=0.2,
        alpha=0.35,
    )
if np.any(cut_mask):
    nx.draw_networkx_edges(
        landscape.graph,
        pos,
        ax=ax_graph,
        edgelist=[edges[i] for i in np.flatnonzero(cut_mask)],
        edge_color=[edge_rgba[i] for i in np.flatnonzero(cut_mask)],
        width=0.85,
        alpha=0.95,
    )
nx.draw_networkx_nodes(
    landscape.graph,
    pos,
    ax=ax_graph,
    nodelist=node_order,
    node_color=node_colors,
    node_size=node_size,
    linewidths=0.0,
    alpha=0.9,
)
ax_graph.set_title(
    f"Most rugged: {example_row.dataset}\nobserved $t_{{MAP}}$={float(example_row.observed_t_map):.2f}",
    fontsize=10,
)
ax_graph.set_axis_off()

shell_x = profile_df['shell_plot'].to_numpy(dtype=int)
shell_y = profile_df['mean_edge_energy_norm'].to_numpy(dtype=float)
ax_profile.plot(shell_x, shell_y, color='black', linewidth=1.5, marker='o')
ax_profile.axhline(1.0, color='0.5', linestyle='--', linewidth=1.0)
ax_profile.set_xlabel('Graph distance from spectral cut')
ax_profile.set_ylabel('Mean edge energy / domain mean')
ax_profile.set_xticks(shell_x)
ax_profile.set_xticklabels([
    str(x) if x < int(getattr(sbe, 'SPECTRAL_MAX_SHELL_PLOT', 6)) else f"{int(getattr(sbe, 'SPECTRAL_MAX_SHELL_PLOT', 6))}+"
    for x in shell_x
])
ax_profile.set_title(
    f"peak enrichment={float(example_row.boundary_peak_enrichment):.2f}",
    fontsize=10,
)
ax_profile.grid(axis='y', linestyle='--', alpha=0.3)

sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.magma)
sm.set_array([])
fig.colorbar(sm, ax=ax_graph, fraction=0.046, pad=0.02, label=r'$\log_{10}$(edge energy)')

legend_handles = [
    Patch(facecolor=side_palette[0], edgecolor='none', label='Partition side 0'),
    Patch(facecolor=side_palette[1], edgecolor='none', label='Partition side 1'),
    Line2D([0], [0], color='black', linewidth=1.2, label='Shell profile'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_dir = Path('../figures/figure_2')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'spectral_cut_distance_most_rugged_example.pdf'
plt.savefig(out_path)
plt.show()
print(f'Figure written to: {out_path}')

# %%
# Figure plot for spectral bipartitioning: paired dot plot of mean edge energy on vs off the cut.

from pathlib import Path

if 'spectral_partition_domain_df' in globals():
    _spectral_pair_df = spectral_partition_domain_df.copy()
elif 'spectral_partition_results' in globals():
    _spectral_pair_df = spectral_partition_results['domain_df'].copy()
else:
    raise RuntimeError('Run Experiment 9a first (requires spectral_partition_domain_df or spectral_partition_results).')

work_df = _spectral_pair_df.loc[
    _spectral_pair_df['status'].eq('ok')
    & np.isfinite(_spectral_pair_df['mean_edge_energy_all'].to_numpy(dtype=float))
    & np.isfinite(_spectral_pair_df['mean_edge_energy_cut'].to_numpy(dtype=float))
    & np.isfinite(_spectral_pair_df['mean_edge_energy_noncut'].to_numpy(dtype=float))
    & (_spectral_pair_df['mean_edge_energy_all'].to_numpy(dtype=float) > 0),
    [
        'dataset',
        'file',
        'observed_t_map',
        'mean_edge_energy_all',
        'mean_edge_energy_cut',
        'mean_edge_energy_noncut',
    ],
].copy()
if len(work_df) == 0:
    raise RuntimeError('No valid spectral bipartition edge-energy summaries available for plotting.')

work_df = work_df.sort_values(['observed_t_map', 'dataset', 'file']).reset_index(drop=True)
work_df['cut_mean_norm'] = work_df['mean_edge_energy_cut'] / work_df['mean_edge_energy_all']
work_df['noncut_mean_norm'] = work_df['mean_edge_energy_noncut'] / work_df['mean_edge_energy_all']

x = np.array([0.0, 1.0], dtype=float)

fig, ax = plt.subplots(figsize=(3.3, 4.0))
for row in work_df.itertuples(index=False):
    y = np.asarray([row.cut_mean_norm, row.noncut_mean_norm], dtype=float)
    ax.plot(x, y, color='0.85', linewidth=1.0, alpha=0.9, zorder=1)

ax.scatter(
    np.full(len(work_df), x[0]),
    work_df['cut_mean_norm'].to_numpy(dtype=float),
    s=52,
    color='#4C78A8',
    edgecolor='none',
    zorder=3,
)
ax.scatter(
    np.full(len(work_df), x[1]),
    work_df['noncut_mean_norm'].to_numpy(dtype=float),
    s=52,
    color='#E45756',
    edgecolor='none',
    zorder=3,
)

ax.axhline(1.0, color='0.9', linestyle='--', linewidth=1.0, zorder=0)
ax.set_xlim(-0.15, 1.15)
ax.set_xticks(x)
ax.set_xticklabels(['Bipartition\nedges', 'Off\nbipartition'])
ax.set_ylabel('Mean edge energy / domain mean')
ax.grid(axis='y', linestyle='--', alpha=0.25)

plt.tight_layout()
out_dir = Path('../figures/figure_2')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'spectral_bipartition_edge_energy_paired.pdf'
plt.savefig(out_path)
plt.show()
print(f'Figure written to: {out_path}')

# %%
# Figure plot for at, near, far profiles energies for all landscapes as lineplot, with each landscape as a separate line.

from pathlib import Path

if 'spectral_shell_domain_df' in globals():
    _spectral_profile_line_df = spectral_shell_domain_df.copy()
elif 'spectral_profile_results' in globals():
    _spectral_profile_line_df = spectral_profile_results['domain_df'].copy()
else:
    raise RuntimeError('Run Experiment 9b first (requires spectral_shell_domain_df or spectral_profile_results).')

work_df = _spectral_profile_line_df.loc[
    np.isfinite(_spectral_profile_line_df['observed_t_map'].to_numpy(dtype=float))
    & np.isfinite(_spectral_profile_line_df['cut_shell_norm_energy'].to_numpy(dtype=float))
    & np.isfinite(_spectral_profile_line_df['near_shell_norm_energy'].to_numpy(dtype=float))
    & np.isfinite(_spectral_profile_line_df['far_shell_norm_energy'].to_numpy(dtype=float)),
    [
        'dataset',
        'file',
        'observed_t_map',
        'cut_shell_norm_energy',
        'near_shell_norm_energy',
        'far_shell_norm_energy',
    ],
].copy()
if len(work_df) == 0:
    raise RuntimeError('No valid at/near/far shell profiles available for plotting.')

work_df = work_df.sort_values(['observed_t_map', 'dataset', 'file']).reset_index(drop=True)
x = np.arange(3)
xticklabels = ['At cut', 'Near', 'Far']

fig, ax = plt.subplots(figsize=(5.4, 3.6))
for row in work_df.itertuples(index=False):
    y = np.asarray([
        row.cut_shell_norm_energy,
        row.near_shell_norm_energy,
        row.far_shell_norm_energy,
    ], dtype=float)
    ax.plot(x, y, color='0.45', linewidth=0.9, alpha=0.45)

ax.axhline(1.0, color='0.2', linestyle='--', linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Distance bin from spectral cut')
ax.set_ylabel('Mean edge energy / domain mean')
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
out_dir = Path('../figures/figure_2')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'spectral_cut_at_near_far_profiles_all_domains.pdf'
plt.savefig(out_path)
plt.show()
print(f'Figure written to: {out_path}')

# %%
# SEARCH_TAG: SPECTRAL_CHEEGER_ENRICHMENT
# Experiment 9c: Cheeger-style enrichment of Dirichlet energy on the spectral cut.
from pathlib import Path
import importlib
import spectral_boundary_experiments as sbe

importlib.reload(sbe)

spectral_cheeger_results = sbe.run_spectral_cheeger_enrichment(
    spectral_partition_domain_df=spectral_partition_domain_df,
    outdir=Path('../figures/SI_figures/SI_figure_DMS'),
)

spectral_cheeger_df = spectral_cheeger_results['summary_df']
spectral_cheeger_stats_df = spectral_cheeger_results['stats_df']

print('Median cut-energy enrichment:', float(spectral_cheeger_df['cut_energy_enrichment'].median()))
print('Domains with enrichment > 1:', int((spectral_cheeger_df['cut_energy_enrichment'] > 1).sum()), '/', len(spectral_cheeger_df))
display(spectral_cheeger_df.sort_values('cut_energy_enrichment', ascending=False))
display(spectral_cheeger_stats_df)
print(f"Cheeger-style summary written to: {spectral_cheeger_results['summary_csv']}")
print(f"Cheeger-style stats written to: {spectral_cheeger_results['stats_csv']}")
print(f"Cheeger-style figure written to: {spectral_cheeger_results['fig_path']}")
