#!/usr/bin/env python3
"""Publication experiment runner.

Experiment code below is copied from Figure_2_reviewer_checks.ipynb cells 0,2,7-8,36,38,42 in
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

# --- Begin copied code from Figure_2_reviewer_checks.ipynb cells 0,2,7-8,36,38,42 ---

# %% [Figure_2_reviewer_checks.ipynb cell 0]
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

# %% [Figure_2_reviewer_checks.ipynb cell 2]
# Prepare file list
file_list = os.listdir('../data_files/megascale_folding/')
file_list = [file for file in file_list if file.endswith('.csv') and not file.startswith('megascale_folding_tmap')]
file_list.sort()

# Initialise empty results list
results = []

for _, file in tqdm(enumerate(file_list)):
    df = pd.read_csv(os.path.join('../data_files/megascale_folding', file))

    # Extract sequence objects
    sequences = [fl.BaseNumpySequence(sequence) for sequence in df['mutated_sequence']]
    fitness = np.array(df['DMS_score'])

    rep_out = {
        "file": str(file),
        "n_sequences": len(sequences)}

    # Construct fitness landscape with Hamming graph for DMS data
    landscape = fl.FitnessLandscape.build(
        sequences,
        graph="hamming",
    )

    # Attach fitness values
    layer_name = f"dms_score"
    landscape.attach(name=layer_name, values=fitness, dtype="numeric")
    landscape.view(layer_name)

    # Check if there is more than a single connected component and skip if so
    G = landscape.graph  # nx.Graph
    if not nx.is_connected(G):
        continue
        # landscape = landscape.get_components()[0]


    # Compute tmap results
    tmap_res = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(landscape, t_min=1e-10, t_max=1e2, prior='uniform')

    # Collect results
    rep_out["tmap"] =  tmap_res
    results.append(rep_out)

# %% [Figure_2_reviewer_checks.ipynb cell 7]
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

# %% [Figure_2_reviewer_checks.ipynb cell 8]
import re
from unittest.mock import patch
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

    # Reuse precomputed graph-spectrum basis when provided.
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


class _TMapPermutationStatistic:
    def __init__(
        self,
        landscape,
        *,
        eigenvalues=None,
        eigenvectors=None,
        layer_prefix='perm',
        t_min=1e-10,
        t_max=1e2,
        prior='uniform',
    ):
        self.landscape = landscape
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.layer_prefix = layer_prefix
        self.t_min = t_min
        self.t_max = t_max
        self.prior = prior
        self.calls = 0
        self.null_samples = []

    def __call__(self, left, right):
        values = np.concatenate([
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
        ])
        t_val = compute_tmap_on_landscape_values(
            self.landscape,
            values,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            layer_name=f'{self.layer_prefix}_{self.calls}',
            detach=True,
            t_min=self.t_min,
            t_max=self.t_max,
            prior=self.prior,
        )

        if self.calls > 0:
            self.null_samples.append(float(t_val))

        self.calls += 1
        return float(t_val)


def run_tmap_permutation_test(
    landscape,
    fitness,
    *,
    n_permutations,
    seed=None,
    rng=None,
    eigenvalues=None,
    eigenvectors=None,
    layer_prefix='perm',
    t_min=1e-10,
    t_max=1e2,
    prior='uniform',
):
    fitness = np.asarray(fitness, dtype=float)

    if fitness.ndim != 1:
        raise ValueError('fitness must be one-dimensional.')
    if len(fitness) == 0:
        raise ValueError('fitness must contain at least one value.')
    if seed is not None and rng is not None:
        raise ValueError('Provide either seed or rng, not both.')
    if int(n_permutations) < 0:
        raise ValueError('n_permutations must be non-negative.')

    n_permutations = int(n_permutations)

    if len(fitness) == 1:
        observed_t = compute_tmap_on_landscape_values(
            landscape,
            fitness,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            layer_name=f'{layer_prefix}_observed',
            detach=True,
            t_min=t_min,
            t_max=t_max,
            prior=prior,
        )
        return {
            'observed_t_map': float(observed_t),
            'null_samples': [float(observed_t)] * n_permutations,
            'landscapy_result': None,
        }

    statistic = _TMapPermutationStatistic(
        landscape,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        layer_prefix=layer_prefix,
        t_min=t_min,
        t_max=t_max,
        prior=prior,
    )

    groups = {
        'left': fitness[:1].copy(),
        'right': fitness[1:].copy(),
    }

    default_rng_factory = np.random.default_rng

    if rng is not None:
        with patch.object(
            np.random,
            'default_rng',
            side_effect=lambda *args, **kwargs: (
                rng if len(args) == 0 and len(kwargs) == 0
                else default_rng_factory(*args, **kwargs)
            ),
        ):
            result = fl.analysis.permutation_test(
                groups=groups,
                statistic_func=statistic,
                n_permutations=n_permutations,
                alternative='two-sided',
            )
    elif seed is not None:
        with patch.object(
            np.random,
            'default_rng',
            side_effect=lambda *args, **kwargs: (
                default_rng_factory(seed) if len(args) == 0 and len(kwargs) == 0
                else default_rng_factory(*args, **kwargs)
            ),
        ):
            result = fl.analysis.permutation_test(
                groups=groups,
                statistic_func=statistic,
                n_permutations=n_permutations,
                alternative='two-sided',
            )
    else:
        result = fl.analysis.permutation_test(
            groups=groups,
            statistic_func=statistic,
            n_permutations=n_permutations,
            alternative='two-sided',
        )

    pair_result = next(iter(result.values()))
    return {
        'observed_t_map': float(pair_result['observed']),
        'null_samples': [float(x) for x in statistic.null_samples],
        'landscapy_result': pair_result,
    }


def infer_wildtype_sequence(domain_df):
    wt_seq = list(domain_df['mutated_sequence'].iloc[0])

    if 'mutant' not in domain_df.columns:
        return ''.join(wt_seq)

    for mutant in domain_df['mutant'].astype(str):
        for token in mutant.split(':'):
            match = MUT_TOKEN_RE.fullmatch(token.strip())
            if not match:
                continue

            wt_aa, pos_str, _ = match.groups()
            idx = int(pos_str) - 1

            if 0 <= idx < len(wt_seq):
                wt_seq[idx] = wt_aa

    return ''.join(wt_seq)


def estimate_wt_fitness(domain_df):
    if 'mutant' in domain_df.columns:
        wt_mask = domain_df['mutant'].astype(str).str.upper().eq('WT')
        if wt_mask.any():
            return float(domain_df.loc[wt_mask, 'DMS_score'].iloc[0]), 'explicit_wt_row'

    wt_seq = infer_wildtype_sequence(domain_df)
    seq_mask = domain_df['mutated_sequence'].astype(str).eq(wt_seq)
    if seq_mask.any():
        return float(domain_df.loc[seq_mask, 'DMS_score'].iloc[0]), 'inferred_wt_sequence'

    # ProteinGym substitution sets are often centered around WT ~= 0.
    return 0.0, 'assumed_zero'


def domain_fitness_summary(domain_df):
    values = domain_df['DMS_score'].to_numpy(dtype=float)
    wt_score, wt_source = estimate_wt_fitness(domain_df)

    mean_val = float(values.mean())
    std_val = float(values.std(ddof=0))

    return {
        'fitness_mean': mean_val,
        'fitness_variance': float(values.var(ddof=0)),
        'fitness_skewness': float(skew(values, bias=False, nan_policy='omit')) if len(values) >= 3 else np.nan,
        'wt_score': wt_score,
        'wt_score_source': wt_source,
        'wt_minus_mean': wt_score - mean_val,
        'wt_zscore': (wt_score - mean_val) / std_val if std_val > 0 else np.nan,
        'wt_percentile': float(percentileofscore(values, wt_score, kind='mean')),
        'n_variants': int(len(values)),
    }

# %% [Figure_2_reviewer_checks.ipynb cell 36]
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

# %% [Figure_2_reviewer_checks.ipynb cell 38]
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

# %% [Figure_2_reviewer_checks.ipynb cell 42]
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

# --- End copied code from Figure_2_reviewer_checks.ipynb cells 0,2,7-8,36,38,42 ---

run_postprocess(SCRIPT_DIR / "postprocess.py", globals())
