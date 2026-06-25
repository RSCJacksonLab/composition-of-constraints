# Exported from analysis.ipynb

# %%
# Exported from analysis.ipynb

# %%
# Exported from analysis.ipynb

# %%
# Exported from analysis.ipynb

# %%
# Exported from analysis.ipynb

# %% [markdown]
# # Figure 2 node-label permutation nulls for spectral cutsets and autocorrelation
#
# Collect public node-label permutation chunks for the Figure 2 stability-DMS experiment and quantify null distributions
# for the fitness-weighted spectral bipartition cutset and line-graph edge-energy
# autocorrelation statistics.

# %%
from __future__ import annotations

import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.stats import gaussian_kde

analysis_dir = Path(__file__).resolve().parents[1]
scripts_dir = analysis_dir.parents[1] / "scripts"
sys.path.insert(0, str(scripts_dir))
from paper_runtime import find_project_root, resolve_publication_data_dirs

project_root = find_project_root(analysis_dir)
data_dirs = resolve_publication_data_dirs(project_root)
table_dir = analysis_dir / "tables"
figure_dir = analysis_dir / "figures" / "figure_2"
si_figure_dir = analysis_dir / "figures" / "SI_figures" / "SI_figure_DMS"
for path in [table_dir, figure_dir, si_figure_dir]:
    path.mkdir(parents=True, exist_ok=True)

null_dir = project_root / "data" / "processed" / "stability_dms" / "node_label_permutation_nulls"
spectral_observed_path = project_root / "data" / "processed" / "stability_dms" / "spectral_partition_boundary_tmap_domain_summary.csv"
autocorr_observed_path = (
    project_root
    / "analyses"
    / "figure-2_edge-energy-autocorrelation-localization"
    / "tables"
    / "edge_energy_autocorrelation_domain_summary.csv"
)
quantile_observed_path = (
    project_root
    / "analyses"
    / "figure-2_edge-energy-autocorrelation-localization"
    / "tables"
    / "edge_energy_quantile_adjacency_summary.csv"
)

# %%
def _read_chunks(pattern: str) -> pd.DataFrame:
    paths = sorted(null_dir.glob(pattern))
    if not paths:
        raise RuntimeError(f"Expected at least one chunk for {pattern}, found none.")
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return df


spectral_null = _read_chunks("chunk_*_node_label_spectral_bipartition_null.csv")
autocorr_null = _read_chunks("chunk_*_node_label_autocorrelation_domain_null.csv")
quantile_null = _read_chunks("chunk_*_node_label_autocorrelation_quantile_null.csv")

spectral_observed = pd.read_csv(spectral_observed_path).loc[lambda d: d["status"].eq("ok")].copy()
autocorr_observed = pd.read_csv(autocorr_observed_path).copy()
quantile_observed = pd.read_csv(quantile_observed_path).copy()

for label, df in [
    ("spectral_null", spectral_null),
    ("autocorr_null", autocorr_null),
    ("quantile_null", quantile_null),
]:
    if "chunk_id" not in df.columns:
        raise RuntimeError(f"{label} is missing the public chunk_id column.")

ok_spectral_null = spectral_null.loc[spectral_null["status"].eq("ok")].copy()
ok_autocorr_null = autocorr_null.loc[autocorr_null["status"].eq("ok")].copy()
ok_quantile_null = quantile_null.loc[quantile_null["status"].eq("ok")].copy()

if ok_spectral_null["perm_idx"].nunique() != 100:
    raise RuntimeError("Spectral null does not contain 100 permutation indices.")
if ok_autocorr_null["perm_idx"].nunique() != 100:
    raise RuntimeError("Autocorrelation null does not contain 100 permutation indices.")

spectral_null.to_csv(table_dir / "node_label_spectral_bipartition_null.csv", index=False)
autocorr_null.to_csv(table_dir / "node_label_autocorrelation_domain_null.csv", index=False)
quantile_null.to_csv(table_dir / "node_label_autocorrelation_quantile_null.csv", index=False)

print(f"Collected spectral null rows: {len(ok_spectral_null)}")
print(f"Collected autocorrelation null rows: {len(ok_autocorr_null)}")
print(f"Collected quantile null rows: {len(ok_quantile_null)}")

# %%
def empirical_p_greater(null_values: pd.Series | np.ndarray, observed: float) -> float:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    return float((np.sum(values >= observed) + 1) / (len(values) + 1))


def empirical_p_less(null_values: pd.Series | np.ndarray, observed: float) -> float:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    return float((np.sum(values <= observed) + 1) / (len(values) + 1))


spectral_by_perm = ok_spectral_null.groupby("perm_idx", as_index=False).agg(
    spectral_cut_energy_enrichment_median=("cut_energy_enrichment", "median"),
    spectral_cut_energy_fraction_median=("cut_energy_fraction", "median"),
    spectral_cut_edge_fraction_median=("cut_edge_fraction", "median"),
    spectral_partition_balance_median=("partition_balance", "median"),
)
autocorr_by_perm = ok_autocorr_null.groupby("perm_idx", as_index=False).agg(
    moran_i_median=("moran_i", "median"),
    geary_c_median=("geary_c", "median"),
    one_minus_geary_c_median=("one_minus_geary_c", "median"),
)
quantile_by_perm = ok_quantile_null.groupby(["perm_idx", "energy_quantile"], as_index=False).agg(
    neighbor_high_enrichment_median=("neighbor_high_enrichment", "median"),
    binary_moran_i_median=("binary_moran_i", "median"),
    binary_geary_c_median=("binary_geary_c", "median"),
)

median_nulls = spectral_by_perm.merge(autocorr_by_perm, on="perm_idx", how="inner")
median_nulls.to_csv(table_dir / "node_label_permutation_median_nulls.csv", index=False)
quantile_by_perm.to_csv(table_dir / "node_label_quantile_permutation_median_nulls.csv", index=False)
quantile_by_perm.to_csv(
    table_dir / "node_label_quantile_neighbor_enrichment_null_distribution.csv",
    index=False,
)

observed = {
    "spectral_cut_energy_enrichment_median": float(spectral_observed["cut_energy_enrichment"].median()),
    "spectral_cut_energy_fraction_median": float(spectral_observed["cut_energy_fraction"].median()),
    "spectral_cut_edge_fraction_median": float(spectral_observed["cut_edge_fraction"].median()),
    "moran_i_median": float(autocorr_observed["moran_i"].median()),
    "geary_c_median": float(autocorr_observed["geary_c"].median()),
    "one_minus_geary_c_median": float(autocorr_observed["one_minus_geary_c"].median()),
}

population_tests = {
    "spectral_cut_energy_enrichment": {
        "observed_median": observed["spectral_cut_energy_enrichment_median"],
        "null_median_of_medians": float(median_nulls["spectral_cut_energy_enrichment_median"].median()),
        "null_q025": float(median_nulls["spectral_cut_energy_enrichment_median"].quantile(0.025)),
        "null_q975": float(median_nulls["spectral_cut_energy_enrichment_median"].quantile(0.975)),
        "empirical_p_ge_observed": empirical_p_greater(
            median_nulls["spectral_cut_energy_enrichment_median"],
            observed["spectral_cut_energy_enrichment_median"],
        ),
    },
    "moran_i": {
        "observed_median": observed["moran_i_median"],
        "null_median_of_medians": float(median_nulls["moran_i_median"].median()),
        "null_q025": float(median_nulls["moran_i_median"].quantile(0.025)),
        "null_q975": float(median_nulls["moran_i_median"].quantile(0.975)),
        "empirical_p_ge_observed": empirical_p_greater(
            median_nulls["moran_i_median"],
            observed["moran_i_median"],
        ),
    },
    "geary_c": {
        "observed_median": observed["geary_c_median"],
        "null_median_of_medians": float(median_nulls["geary_c_median"].median()),
        "null_q025": float(median_nulls["geary_c_median"].quantile(0.025)),
        "null_q975": float(median_nulls["geary_c_median"].quantile(0.975)),
        "empirical_p_le_observed": empirical_p_less(
            median_nulls["geary_c_median"],
            observed["geary_c_median"],
        ),
    },
    "one_minus_geary_c": {
        "observed_median": observed["one_minus_geary_c_median"],
        "null_median_of_medians": float(median_nulls["one_minus_geary_c_median"].median()),
        "null_q025": float(median_nulls["one_minus_geary_c_median"].quantile(0.025)),
        "null_q975": float(median_nulls["one_minus_geary_c_median"].quantile(0.975)),
        "empirical_p_ge_observed": empirical_p_greater(
            median_nulls["one_minus_geary_c_median"],
            observed["one_minus_geary_c_median"],
        ),
    },
}

population_p_values = pd.DataFrame(
    [
        {
            "test": "spectral_cut_energy_enrichment",
            "statistic": "median cut-energy enrichment across domains",
            "alternative": "observed >= null",
            "observed": population_tests["spectral_cut_energy_enrichment"]["observed_median"],
            "null_median": population_tests["spectral_cut_energy_enrichment"]["null_median_of_medians"],
            "null_q025": population_tests["spectral_cut_energy_enrichment"]["null_q025"],
            "null_q975": population_tests["spectral_cut_energy_enrichment"]["null_q975"],
            "empirical_p": population_tests["spectral_cut_energy_enrichment"]["empirical_p_ge_observed"],
            "n_permutations": int(median_nulls["perm_idx"].nunique()),
            "p_value_method": "plus-one empirical p-value: (count_extreme + 1) / (n_permutations + 1)",
        },
        {
            "test": "moran_i",
            "statistic": "median Moran's I across domains",
            "alternative": "observed >= null",
            "observed": population_tests["moran_i"]["observed_median"],
            "null_median": population_tests["moran_i"]["null_median_of_medians"],
            "null_q025": population_tests["moran_i"]["null_q025"],
            "null_q975": population_tests["moran_i"]["null_q975"],
            "empirical_p": population_tests["moran_i"]["empirical_p_ge_observed"],
            "n_permutations": int(median_nulls["perm_idx"].nunique()),
            "p_value_method": "plus-one empirical p-value: (count_extreme + 1) / (n_permutations + 1)",
        },
        {
            "test": "geary_c",
            "statistic": "median Geary's C across domains",
            "alternative": "observed <= null",
            "observed": population_tests["geary_c"]["observed_median"],
            "null_median": population_tests["geary_c"]["null_median_of_medians"],
            "null_q025": population_tests["geary_c"]["null_q025"],
            "null_q975": population_tests["geary_c"]["null_q975"],
            "empirical_p": population_tests["geary_c"]["empirical_p_le_observed"],
            "n_permutations": int(median_nulls["perm_idx"].nunique()),
            "p_value_method": "plus-one empirical p-value: (count_extreme + 1) / (n_permutations + 1)",
        },
        {
            "test": "one_minus_geary_c",
            "statistic": "median 1 - Geary's C across domains",
            "alternative": "observed >= null",
            "observed": population_tests["one_minus_geary_c"]["observed_median"],
            "null_median": population_tests["one_minus_geary_c"]["null_median_of_medians"],
            "null_q025": population_tests["one_minus_geary_c"]["null_q025"],
            "null_q975": population_tests["one_minus_geary_c"]["null_q975"],
            "empirical_p": population_tests["one_minus_geary_c"]["empirical_p_ge_observed"],
            "n_permutations": int(median_nulls["perm_idx"].nunique()),
            "p_value_method": "plus-one empirical p-value: (count_extreme + 1) / (n_permutations + 1)",
        },
    ]
)
population_p_values.to_csv(table_dir / "node_label_permutation_population_p_values.csv", index=False)

quantile_population_rows: list[dict[str, object]] = []
quantile_population_tests: dict[str, dict[str, float | int | str]] = {}
for energy_quantile in sorted(quantile_by_perm["energy_quantile"].dropna().unique()):
    q_key = str(float(energy_quantile))
    null_sub = quantile_by_perm.loc[
        quantile_by_perm["energy_quantile"].eq(energy_quantile)
    ].copy()
    observed_sub = quantile_observed.loc[
        quantile_observed["energy_quantile"].eq(energy_quantile)
    ].copy()
    observed_neighbor_enrichment = float(observed_sub["neighbor_high_enrichment"].median())
    observed_binary_moran_i = float(observed_sub["binary_moran_i"].median())
    observed_binary_geary_c = float(observed_sub["binary_geary_c"].median())
    null_neighbor = null_sub["neighbor_high_enrichment_median"]
    null_binary_moran = null_sub["binary_moran_i_median"]
    null_binary_geary = null_sub["binary_geary_c_median"]
    high_energy_fraction = 1.0 - float(energy_quantile)
    row = {
        "energy_quantile": float(energy_quantile),
        "high_energy_set": f"top {100.0 * high_energy_fraction:g}% edges",
        "statistic": "median high-energy neighbour enrichment across domains",
        "alternative": "observed >= null",
        "observed_neighbor_high_enrichment_median": observed_neighbor_enrichment,
        "null_neighbor_high_enrichment_median": float(null_neighbor.median()),
        "null_neighbor_high_enrichment_q025": float(null_neighbor.quantile(0.025)),
        "null_neighbor_high_enrichment_q975": float(null_neighbor.quantile(0.975)),
        "empirical_p_neighbor_high_enrichment_ge_observed": empirical_p_greater(
            null_neighbor,
            observed_neighbor_enrichment,
        ),
        "observed_binary_moran_i_median": observed_binary_moran_i,
        "null_binary_moran_i_median": float(null_binary_moran.median()),
        "null_binary_moran_i_q025": float(null_binary_moran.quantile(0.025)),
        "null_binary_moran_i_q975": float(null_binary_moran.quantile(0.975)),
        "empirical_p_binary_moran_i_ge_observed": empirical_p_greater(
            null_binary_moran,
            observed_binary_moran_i,
        ),
        "observed_binary_geary_c_median": observed_binary_geary_c,
        "null_binary_geary_c_median": float(null_binary_geary.median()),
        "null_binary_geary_c_q025": float(null_binary_geary.quantile(0.025)),
        "null_binary_geary_c_q975": float(null_binary_geary.quantile(0.975)),
        "empirical_p_binary_geary_c_le_observed": empirical_p_less(
            null_binary_geary,
            observed_binary_geary_c,
        ),
        "n_permutations": int(null_sub["perm_idx"].nunique()),
        "p_value_method": "plus-one empirical p-value: (count_extreme + 1) / (n_permutations + 1)",
    }
    quantile_population_rows.append(row)
    quantile_population_tests[q_key] = row

quantile_population_p_values = pd.DataFrame(quantile_population_rows)
quantile_population_p_values.to_csv(
    table_dir / "node_label_quantile_population_p_values.csv",
    index=False,
)

print(json.dumps(population_tests, indent=2))

# %%
domain_rows: list[dict[str, object]] = []
quantile_domain_rows: list[dict[str, object]] = []
spectral_obs_by_file = spectral_observed.set_index("file")
autocorr_obs_by_file = autocorr_observed.set_index("file")
all_files = sorted(set(spectral_obs_by_file.index) & set(autocorr_obs_by_file.index))

for file_name in all_files:
    spec_obs = spectral_obs_by_file.loc[file_name]
    auto_obs = autocorr_obs_by_file.loc[file_name]
    spec_null = ok_spectral_null.loc[ok_spectral_null["file"].eq(file_name)]
    auto_null = ok_autocorr_null.loc[ok_autocorr_null["file"].eq(file_name)]
    domain_rows.append(
        {
            "file": file_name,
            "dataset": str(spec_obs["dataset"]),
            "n_spectral_null": int(len(spec_null)),
            "n_autocorr_null": int(len(auto_null)),
            "observed_cut_energy_enrichment": float(spec_obs["cut_energy_enrichment"]),
            "null_cut_energy_enrichment_median": float(spec_null["cut_energy_enrichment"].median()),
            "empirical_p_cut_energy_enrichment_ge_observed": empirical_p_greater(
                spec_null["cut_energy_enrichment"],
                float(spec_obs["cut_energy_enrichment"]),
            ),
            "observed_moran_i": float(auto_obs["moran_i"]),
            "null_moran_i_median": float(auto_null["moran_i"].median()),
            "empirical_p_moran_i_ge_observed": empirical_p_greater(
                auto_null["moran_i"],
                float(auto_obs["moran_i"]),
            ),
            "observed_geary_c": float(auto_obs["geary_c"]),
            "null_geary_c_median": float(auto_null["geary_c"].median()),
            "empirical_p_geary_c_le_observed": empirical_p_less(
                auto_null["geary_c"],
                float(auto_obs["geary_c"]),
            ),
            "observed_one_minus_geary_c": float(auto_obs["one_minus_geary_c"]),
            "null_one_minus_geary_c_median": float(auto_null["one_minus_geary_c"].median()),
            "empirical_p_one_minus_geary_c_ge_observed": empirical_p_greater(
                auto_null["one_minus_geary_c"],
                float(auto_obs["one_minus_geary_c"]),
            ),
        }
    )
    observed_quantile_file = quantile_observed.loc[quantile_observed["file"].eq(file_name)].copy()
    null_quantile_file = ok_quantile_null.loc[ok_quantile_null["file"].eq(file_name)].copy()
    for qrow in observed_quantile_file.itertuples(index=False):
        q_null = null_quantile_file.loc[
            null_quantile_file["energy_quantile"].eq(float(qrow.energy_quantile))
        ]
        quantile_domain_rows.append(
            {
                "file": file_name,
                "dataset": str(spec_obs["dataset"]),
                "energy_quantile": float(qrow.energy_quantile),
                "high_energy_set": f"top {100.0 * (1.0 - float(qrow.energy_quantile)):g}% edges",
                "n_null": int(len(q_null)),
                "observed_neighbor_high_enrichment": float(qrow.neighbor_high_enrichment),
                "null_neighbor_high_enrichment_median": float(q_null["neighbor_high_enrichment"].median()),
                "empirical_p_neighbor_high_enrichment_ge_observed": empirical_p_greater(
                    q_null["neighbor_high_enrichment"],
                    float(qrow.neighbor_high_enrichment),
                ),
                "observed_binary_moran_i": float(qrow.binary_moran_i),
                "null_binary_moran_i_median": float(q_null["binary_moran_i"].median()),
                "empirical_p_binary_moran_i_ge_observed": empirical_p_greater(
                    q_null["binary_moran_i"],
                    float(qrow.binary_moran_i),
                ),
                "observed_binary_geary_c": float(qrow.binary_geary_c),
                "null_binary_geary_c_median": float(q_null["binary_geary_c"].median()),
                "empirical_p_binary_geary_c_le_observed": empirical_p_less(
                    q_null["binary_geary_c"],
                    float(qrow.binary_geary_c),
                ),
            }
        )

domain_empirical = pd.DataFrame(domain_rows).sort_values("dataset").reset_index(drop=True)
domain_empirical.to_csv(table_dir / "node_label_permutation_domain_empirical_p.csv", index=False)
quantile_domain_empirical = (
    pd.DataFrame(quantile_domain_rows)
    .sort_values(["energy_quantile", "dataset"])
    .reset_index(drop=True)
)
quantile_domain_empirical.to_csv(
    table_dir / "node_label_quantile_domain_empirical_p.csv",
    index=False,
)

summary = {
    "analysis_id": "figure-2_node-label-permutation-nulls",
    "null_type": "node-label permutation within each DMS landscape before recomputing edge energies",
    "runs": RUN_IDS,
    "n_runs": len(RUN_IDS),
    "n_permutations": int(median_nulls["perm_idx"].nunique()),
    "n_domains": int(len(all_files)),
    "p_value_correction": "plus-one empirical p-value: (count_extreme + 1) / (n_permutations + 1)",
    "observed_population_medians": observed,
    "population_tests": population_tests,
    "quantile_population_tests": quantile_population_tests,
    "domain_counts": {
        "spectral_cut_energy_enrichment_ge_observed_at_min_p": int(
            (domain_empirical["empirical_p_cut_energy_enrichment_ge_observed"] <= 1 / 101).sum()
        ),
        "moran_i_ge_observed_at_min_p": int(
            (domain_empirical["empirical_p_moran_i_ge_observed"] <= 1 / 101).sum()
        ),
        "geary_c_le_observed_at_min_p": int(
            (domain_empirical["empirical_p_geary_c_le_observed"] <= 1 / 101).sum()
        ),
        "quantile_neighbor_enrichment_ge_observed_at_min_p": {
            str(float(energy_quantile)): int(
                (
                    quantile_domain_empirical.loc[
                        quantile_domain_empirical["energy_quantile"].eq(energy_quantile),
                        "empirical_p_neighbor_high_enrichment_ge_observed",
                    ]
                    <= 1 / 101
                ).sum()
            )
            for energy_quantile in sorted(quantile_domain_empirical["energy_quantile"].dropna().unique())
        },
    },
    "source_tables": {
        "spectral_observed": str(spectral_observed_path.relative_to(project_root)),
        "autocorrelation_observed": str(autocorr_observed_path.relative_to(project_root)),
        "quantile_observed": str(quantile_observed_path.relative_to(project_root)),
        "null_chunks": str(null_dir.relative_to(project_root)),
    },
}
with (table_dir / "node_label_permutation_summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)

domain_empirical.head()

# %%
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _plot_null_kde(ax, values, observed_value, color, xlabel, p_label):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed_value = float(observed_value)
    xlo = float(min(np.min(values), observed_value))
    xhi = float(max(np.max(values), observed_value))
    pad = 0.08 * (xhi - xlo if xhi > xlo else 1.0)
    x_grid = np.linspace(xlo - pad, xhi + pad, 500)
    if len(values) >= 2 and float(np.std(values)) > 0:
        density = gaussian_kde(values)(x_grid)
        ax.fill_between(x_grid, 0, density, color=color, alpha=0.42, linewidth=0)
        ax.plot(x_grid, density, color=color, linewidth=1.4)
        ymax = float(np.max(density))
    else:
        ax.axvline(float(values[0]) if len(values) else observed_value, color=color, linewidth=1.4)
        ymax = 1.0
    observed_line = ax.axvline(observed_value, color="black", linewidth=1.5)
    null_line = ax.axvline(float(np.median(values)), color=color, linewidth=1.2, linestyle="--")
    ax.text(
        0.98,
        0.94,
        p_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    ax.set_ylim(0, ymax * 1.08)
    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return observed_line, null_line


fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25))
obs_line, null_line = _plot_null_kde(
    axes[0],
    median_nulls["spectral_cut_energy_enrichment_median"],
    observed["spectral_cut_energy_enrichment_median"],
    "#4C78A8",
    "Median cut-energy enrichment",
    r"$P_{emp}=1/101$",
)
_plot_null_kde(
    axes[1],
    median_nulls["moran_i_median"],
    observed["moran_i_median"],
    "#59A14F",
    "Median Moran's I",
    r"$P_{emp}=1/101$",
)
_plot_null_kde(
    axes[2],
    median_nulls["geary_c_median"],
    observed["geary_c_median"],
    "#B07AA1",
    "Median Geary's C",
    r"$P_{emp}=1/101$",
)
axes[0].set_title("Spectral cutset")
axes[1].set_title("Autocorrelation")
axes[2].set_title("Autocorrelation")
axes[0].legend(
    [obs_line, null_line],
    ["Observed median", "Null median"],
    loc="upper center",
    bbox_to_anchor=(0.52, 0.78),
    frameon=False,
    fontsize=7,
    handlelength=1.8,
)
for label, ax in zip(["a", "b", "c"], axes):
    ax.text(-0.22, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold")
fig.tight_layout(w_pad=1.6)
main_pdf = figure_dir / "node_label_permutation_nulls_main_panel.pdf"
main_png = figure_dir / "node_label_permutation_nulls_main_panel.png"
fig.savefig(main_pdf, bbox_inches="tight")
fig.savefig(main_png, dpi=300, bbox_inches="tight")
plt.close(fig)

# %%
quantile_colors = {
    0.75: "#4C78A8",
    0.90: "#59A14F",
    0.95: "#E3BA22",
    0.99: "#B07AA1",
}
quantile_titles = {
    0.75: "Top 25% edges",
    0.90: "Top 10% edges",
    0.95: "Top 5% edges",
    0.99: "Top 1% edges",
}

fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.0))
for label, ax, energy_quantile in zip(["a", "b", "c", "d"], axes.ravel(), [0.75, 0.90, 0.95, 0.99]):
    null_values = quantile_by_perm.loc[
        quantile_by_perm["energy_quantile"].eq(energy_quantile),
        "neighbor_high_enrichment_median",
    ]
    row = quantile_population_p_values.loc[
        quantile_population_p_values["energy_quantile"].eq(energy_quantile)
    ].iloc[0]
    p_label = r"$P_{emp}=1/101$" if np.isclose(float(row["empirical_p_neighbor_high_enrichment_ge_observed"]), 1 / 101) else (
        rf"$P_{{emp}}={float(row['empirical_p_neighbor_high_enrichment_ge_observed']):.3g}$"
    )
    _plot_null_kde(
        ax,
        null_values,
        float(row["observed_neighbor_high_enrichment_median"]),
        quantile_colors[energy_quantile],
        "Median neighbour enrichment",
        p_label,
    )
    ax.set_title(quantile_titles[energy_quantile])
    ax.text(-0.22, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold")

fig.tight_layout(w_pad=1.4, h_pad=1.5)
quantile_pdf = si_figure_dir / "node_label_quantile_neighbor_enrichment_nulls.pdf"
quantile_png = si_figure_dir / "node_label_quantile_neighbor_enrichment_nulls.png"
fig.savefig(quantile_pdf, bbox_inches="tight")
fig.savefig(quantile_png, dpi=300, bbox_inches="tight")
plt.close(fig)

# %%
quantile_order = [0.75, 0.90, 0.95, 0.99]
x_quantile = {q: idx for idx, q in enumerate(quantile_order)}
plot_quantile = quantile_observed.loc[
    quantile_observed["energy_quantile"].isin(quantile_order)
].copy()
plot_quantile["x"] = plot_quantile["energy_quantile"].map(x_quantile)

observed_median_by_quantile = (
    plot_quantile.groupby("energy_quantile", sort=True)["neighbor_high_enrichment"]
    .median()
    .reindex(quantile_order)
)
observed_q25_by_quantile = (
    plot_quantile.groupby("energy_quantile", sort=True)["neighbor_high_enrichment"]
    .quantile(0.25)
    .reindex(quantile_order)
)
observed_q75_by_quantile = (
    plot_quantile.groupby("energy_quantile", sort=True)["neighbor_high_enrichment"]
    .quantile(0.75)
    .reindex(quantile_order)
)
null_median_by_quantile = (
    quantile_by_perm.groupby("energy_quantile", sort=True)["neighbor_high_enrichment_median"]
    .median()
    .reindex(quantile_order)
)

all_enrichment_values = pd.concat(
    [
        plot_quantile["neighbor_high_enrichment"],
        ok_quantile_null["neighbor_high_enrichment"],
        quantile_by_perm["neighbor_high_enrichment_median"],
    ],
    ignore_index=True,
).replace([np.inf, -np.inf], np.nan).dropna()
ymin = 0.8
ymax = max(32.0, float(all_enrichment_values.max()) * 1.12)
log_y_grid = np.linspace(np.log2(ymin), np.log2(ymax), 320)
y_grid = 2**log_y_grid

fig, ax = plt.subplots(figsize=(3.45, 3.6), constrained_layout=True)
null_color = "#59A14F"
observed_color = "#4C78A8"

for energy_quantile in quantile_order:
    x0 = x_quantile[energy_quantile]
    density_values = ok_quantile_null.loc[
        ok_quantile_null["energy_quantile"].eq(energy_quantile),
        "neighbor_high_enrichment",
    ].to_numpy(dtype=float)
    density_values = density_values[np.isfinite(density_values) & (density_values > 0)]
    null_values = quantile_by_perm.loc[
        quantile_by_perm["energy_quantile"].eq(energy_quantile),
        "neighbor_high_enrichment_median",
    ].to_numpy(dtype=float)
    null_values = null_values[np.isfinite(null_values) & (null_values > 0)]
    if len(density_values) >= 3 and np.nanstd(density_values) > 0:
        density = gaussian_kde(np.log2(density_values))(log_y_grid)
        density = density / density.max() * 0.28
        ax.fill_betweenx(
            y_grid,
            x0 - density,
            x0 + density,
            color=null_color,
            alpha=0.26,
            linewidth=0,
            zorder=1,
        )
    null_median = float(np.median(null_values))
    ax.plot([x0 - 0.20, x0 + 0.20], [null_median, null_median], color=null_color, lw=1.1, zorder=3)

for _, group in plot_quantile.groupby("dataset", sort=False):
    group = group.sort_values("energy_quantile")
    ax.plot(
        group["x"],
        group["neighbor_high_enrichment"],
        color="0.70",
        linewidth=0.65,
        alpha=0.42,
        zorder=2,
    )

xs = np.arange(len(quantile_order))
ax.fill_between(
    xs,
    observed_q25_by_quantile,
    observed_q75_by_quantile,
    color=observed_color,
    alpha=0.16,
    linewidth=0,
    zorder=4,
)
observed_line, = ax.plot(
    xs,
    observed_median_by_quantile,
    color=observed_color,
    marker="o",
    markersize=4.6,
    linewidth=1.9,
    zorder=6,
)
null_line, = ax.plot(
    xs,
    null_median_by_quantile,
    color=null_color,
    marker="o",
    markersize=3.8,
    linewidth=1.3,
    linestyle="--",
    zorder=5,
)
null_patch = mpl.patches.Patch(facecolor=null_color, alpha=0.26, edgecolor="none")

ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
ax.set_yscale("log", base=2)
ax.set_ylim(ymin, ymax)
ax.set_yticks([1, 2, 4, 8, 16, 32])
ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
ax.set_xticks(xs)
ax.set_xticklabels(["Top\n25%", "Top\n10%", "Top\n5%", "Top\n1%"])
ax.set_xlabel("Edge-energy quantile")
ax.set_ylabel("High-energy neighbor enrichment")
ax.set_title("High-energy adjacency")
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(
    [observed_line, null_line, null_patch],
    ["Observed median", "Null median", "Null KDE"],
    loc="upper left",
    frameon=False,
    fontsize=7,
    handlelength=1.8,
)

quantile_overlay_pdf = figure_dir / "node_label_quantile_neighbor_enrichment_null_overlay.pdf"
quantile_overlay_png = figure_dir / "node_label_quantile_neighbor_enrichment_null_overlay.png"
fig.savefig(quantile_overlay_pdf, bbox_inches="tight")
fig.savefig(quantile_overlay_png, dpi=300, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Example graph visualizations
# A single reproducible DMS example is used to show edge-energy localization on
# the original genotype graph and as a line-graph adjacency matrix. The selected
# example is the most rugged domain with fewer than 15,000 graph edges, which
# keeps the line-graph matrix readable while still showing a rugged landscape.

# %%
EXAMPLE_EDGE_LIMIT = 15_000
EXAMPLE_QUANTILE = 0.95
EXAMPLE_PERM_IDX = 0
EXAMPLE_BASE_SEED = 2026052701
example_data_dir = data_dirs["data_files"] / "megascale_folding"
edge_detail_path = project_root / "data" / "processed" / "stability_dms" / "spectral_partition_boundary_tmap_edge_details.csv"


def _seed_for(base_seed: int, perm_idx: int, file_name: str) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{perm_idx}:{file_name}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % (2**32)


def _read_edge_details_for_file(file_name: str) -> pd.DataFrame:
    chunks = []
    usecols = ["file", "dataset", "u", "v", "edge_energy"]
    for chunk in pd.read_csv(edge_detail_path, usecols=usecols, chunksize=500_000):
        sub = chunk.loc[chunk["file"].eq(file_name)].copy()
        if len(sub) > 0:
            chunks.append(sub)
    if not chunks:
        raise RuntimeError(f"No edge details found for {file_name}.")
    edge_table = pd.concat(chunks, ignore_index=True)
    edge_table["u"] = edge_table["u"].astype(np.int64)
    edge_table["v"] = edge_table["v"].astype(np.int64)
    edge_table["edge_energy"] = edge_table["edge_energy"].astype(float)
    return edge_table


def _load_domain_fitness(file_name: str) -> np.ndarray:
    domain_df = pd.read_csv(example_data_dir / file_name, usecols=["mutated_sequence", "DMS_score"])
    domain_df = domain_df.replace([np.inf, -np.inf], np.nan)
    domain_df = domain_df.dropna(subset=["mutated_sequence", "DMS_score"]).reset_index(drop=True)
    return domain_df["DMS_score"].to_numpy(dtype=float)


def _top_k_mask(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    k = int(max(1, min(k, len(values))))
    selected = np.argpartition(values, len(values) - k)[len(values) - k :]
    mask = np.zeros(len(values), dtype=bool)
    mask[selected] = True
    return mask


def _node_spectral_layout(u: np.ndarray, v: np.ndarray, n_nodes: int) -> np.ndarray:
    data = np.ones(2 * len(u), dtype=float)
    adjacency = sparse.coo_matrix(
        (data, (np.r_[u, v], np.r_[v, u])),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    laplacian = csgraph.laplacian(adjacency, normed=True)
    _, vectors = eigsh(laplacian, k=3, which="SM", tol=1e-3, maxiter=50_000)
    coords = vectors[:, 1:3].copy()
    coords -= np.nanmedian(coords, axis=0)
    scale = np.nanpercentile(np.abs(coords), 98, axis=0)
    scale[scale == 0] = 1.0
    coords /= scale
    return coords


def _incidence_lists(u: np.ndarray, v: np.ndarray, n_nodes: int) -> list[list[int]]:
    incidence: list[list[int]] = [[] for _ in range(n_nodes)]
    for edge_idx, (node_u, node_v) in enumerate(zip(u, v)):
        incidence[int(node_u)].append(edge_idx)
        incidence[int(node_v)].append(edge_idx)
    return incidence


def _line_graph_adjacency(incidence: list[list[int]], n_edges: int) -> sparse.csr_matrix:
    rows = []
    cols = []
    for incident_edges in incidence:
        degree = len(incident_edges)
        if degree < 2:
            continue
        idx = np.asarray(incident_edges, dtype=np.int64)
        rr, cc = np.triu_indices(degree, k=1)
        rows.append(idx[rr])
        cols.append(idx[cc])
    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        row = np.r_[row, col]
        col = np.r_[col, row[: len(col)]]
        data = np.ones(len(row), dtype=np.uint8)
    else:
        row = np.array([], dtype=np.int64)
        col = np.array([], dtype=np.int64)
        data = np.array([], dtype=np.uint8)
    return sparse.coo_matrix((data, (row, col)), shape=(n_edges, n_edges)).tocsr()


def _node_adjacency(u: np.ndarray, v: np.ndarray, n_nodes: int) -> sparse.csr_matrix:
    data = np.ones(2 * len(u), dtype=np.uint8)
    return sparse.coo_matrix(
        (data, (np.r_[u, v], np.r_[v, u])),
        shape=(n_nodes, n_nodes),
    ).tocsr()


def _largest_high_edge_cluster(line_adj: sparse.csr_matrix, high_mask: np.ndarray) -> np.ndarray:
    high_indices = np.flatnonzero(high_mask)
    if len(high_indices) == 0:
        return high_indices
    high_adj = line_adj[high_indices][:, high_indices]
    n_components, labels = csgraph.connected_components(high_adj, directed=False)
    if n_components == 0:
        return high_indices[:0]
    counts = np.bincount(labels)
    largest_label = int(np.argmax(counts))
    return high_indices[labels == largest_label]


def _multi_source_bfs(line_adj: sparse.csr_matrix, sources: np.ndarray) -> np.ndarray:
    distances = np.full(line_adj.shape[0], np.inf)
    queue: deque[int] = deque()
    for source in sources:
        source = int(source)
        distances[source] = 0.0
        queue.append(source)
    indptr = line_adj.indptr
    indices = line_adj.indices
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1.0
        for neighbor in indices[indptr[current] : indptr[current + 1]]:
            if np.isinf(distances[neighbor]):
                distances[neighbor] = next_distance
                queue.append(int(neighbor))
    return distances


def _multi_source_node_bfs(node_adj: sparse.csr_matrix, sources: np.ndarray) -> np.ndarray:
    distances = np.full(node_adj.shape[0], np.inf)
    queue: deque[int] = deque()
    for source in sources:
        source = int(source)
        distances[source] = 0.0
        queue.append(source)
    indptr = node_adj.indptr
    indices = node_adj.indices
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1.0
        for neighbor in indices[indptr[current] : indptr[current + 1]]:
            if np.isinf(distances[neighbor]):
                distances[neighbor] = next_distance
                queue.append(int(neighbor))
    return distances


example_candidates = (
    spectral_observed.loc[spectral_observed["n_edges"].lt(EXAMPLE_EDGE_LIMIT)]
    .sort_values(["observed_t_map", "cut_energy_enrichment"], ascending=[False, False])
    .reset_index(drop=True)
)
if len(example_candidates) == 0:
    raise RuntimeError("No suitable example graph found for localization visualizations.")
example_row = example_candidates.iloc[0]
example_file = str(example_row["file"])
example_dataset = str(example_row["dataset"])
example_edges = _read_edge_details_for_file(example_file)
example_u = example_edges["u"].to_numpy(dtype=np.int64)
example_v = example_edges["v"].to_numpy(dtype=np.int64)
example_observed_energy = example_edges["edge_energy"].to_numpy(dtype=float)
example_n_nodes = int(max(example_u.max(), example_v.max()) + 1)
example_n_edges = int(len(example_edges))

example_fitness = _load_domain_fitness(example_file)
if len(example_fitness) < example_n_nodes:
    raise RuntimeError(f"Domain fitness table for {example_file} does not cover all graph nodes.")
validation_energy = (example_fitness[example_u] - example_fitness[example_v]) ** 2
if not np.allclose(validation_energy, example_observed_energy, rtol=1e-7, atol=1e-10):
    raise RuntimeError(f"Observed edge energies for {example_file} do not match the source DMS table.")

example_perm_seed = _seed_for(EXAMPLE_BASE_SEED, EXAMPLE_PERM_IDX, example_file)
example_permuted_fitness = np.random.default_rng(example_perm_seed).permutation(example_fitness)
example_null_energy = (example_permuted_fitness[example_u] - example_permuted_fitness[example_v]) ** 2
example_n_high = int(np.ceil((1.0 - EXAMPLE_QUANTILE) * example_n_edges))
example_high_observed = _top_k_mask(example_observed_energy, example_n_high)
example_high_null = _top_k_mask(example_null_energy, example_n_high)
example_coords = _node_spectral_layout(example_u, example_v, example_n_nodes)
example_segments = np.stack([example_coords[example_u], example_coords[example_v]], axis=1)

energy_norm = mpl.colors.Normalize(
    vmin=float(np.log1p(np.r_[example_observed_energy[example_high_observed], example_null_energy[example_high_null]]).min()),
    vmax=float(np.log1p(np.r_[example_observed_energy[example_high_observed], example_null_energy[example_high_null]]).max()),
)
energy_cmap = mpl.colormaps["magma"]

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.65), constrained_layout=True)
for label, ax, title, energy, high_mask in [
    ("a", axes[0], "Observed", example_observed_energy, example_high_observed),
    ("b", axes[1], "Node-label null", example_null_energy, example_high_null),
]:
    background = LineCollection(
        example_segments,
        colors="0.82",
        linewidths=0.12,
        alpha=0.20,
        rasterized=True,
        zorder=1,
    )
    ax.add_collection(background)
    ax.scatter(
        example_coords[:, 0],
        example_coords[:, 1],
        s=1.1,
        color="0.72",
        alpha=0.42,
        linewidth=0,
        rasterized=True,
        zorder=2,
    )
    high_segments = LineCollection(
        example_segments[high_mask],
        cmap=energy_cmap,
        norm=energy_norm,
        linewidths=0.68,
        alpha=0.92,
        rasterized=True,
        zorder=4,
    )
    high_segments.set_array(np.log1p(energy[high_mask]))
    ax.add_collection(high_segments)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-0.05, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold")

cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=energy_norm, cmap=energy_cmap),
    ax=axes,
    fraction=0.035,
    pad=0.015,
)
cbar.set_label(r"$\log(1 + edge energy)$")
graph_pdf = si_figure_dir / "example_edge_energy_graph_observed_vs_node_label_null.pdf"
graph_png = si_figure_dir / "example_edge_energy_graph_observed_vs_node_label_null.png"
fig.savefig(graph_pdf, bbox_inches="tight")
fig.savefig(graph_png, dpi=300, bbox_inches="tight")
plt.close(fig)

example_incidence = _incidence_lists(example_u, example_v, example_n_nodes)
example_line_adj = _line_graph_adjacency(example_incidence, example_n_edges)
largest_cluster = _largest_high_edge_cluster(example_line_adj, example_high_observed)
line_distances = _multi_source_bfs(example_line_adj, largest_cluster)
finite_distances = np.where(np.isfinite(line_distances), line_distances, np.nanmax(line_distances[np.isfinite(line_distances)]) + 1)
edge_order = np.lexsort((-example_observed_energy, ~example_high_observed, finite_distances))
edge_position = np.empty(example_n_edges, dtype=np.int64)
edge_position[edge_order] = np.arange(example_n_edges)
line_coo = example_line_adj.tocoo()
off_diagonal = line_coo.row != line_coo.col
matrix_x = edge_position[line_coo.col[off_diagonal]]
matrix_y = edge_position[line_coo.row[off_diagonal]]
high_sorted = example_high_observed[edge_order].astype(float)

fig = plt.figure(figsize=(3.6, 3.75), constrained_layout=True)
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[0.12, 1.0],
    height_ratios=[0.12, 1.0],
    wspace=0.02,
    hspace=0.02,
)
ax_top = fig.add_subplot(gs[0, 1])
ax_left = fig.add_subplot(gs[1, 0])
ax = fig.add_subplot(gs[1, 1])
high_cmap = mpl.colors.ListedColormap(["white", "#C23B22"])

ax.scatter(matrix_x, matrix_y, s=0.015, color="0.18", alpha=0.07, linewidth=0, rasterized=True)
ax.set_xlim(-0.5, example_n_edges - 0.5)
ax.set_ylim(example_n_edges - 0.5, -0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Edges sorted by distance from largest high-energy cluster")
ax.set_ylabel("Same edge ordering")
ax.set_title("Line-graph adjacency; top 5% marked", pad=5)

ax_top.imshow(high_sorted[np.newaxis, :], aspect="auto", cmap=high_cmap, vmin=0, vmax=1, interpolation="nearest")
ax_left.imshow(high_sorted[:, np.newaxis], aspect="auto", cmap=high_cmap, vmin=0, vmax=1, interpolation="nearest")
ax_top.set_axis_off()
ax_left.set_axis_off()
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

line_graph_pdf = si_figure_dir / "example_line_graph_adjacency_matrix_high_energy_cluster_order.pdf"
line_graph_png = si_figure_dir / "example_line_graph_adjacency_matrix_high_energy_cluster_order.png"
fig.savefig(line_graph_pdf, bbox_inches="tight")
fig.savefig(line_graph_png, dpi=300, bbox_inches="tight")
plt.close(fig)

example_spec = {
    "dataset": example_dataset,
    "file": example_file,
    "selection_rule": f"highest observed_t_map with n_edges < {EXAMPLE_EDGE_LIMIT}",
    "observed_t_map": float(example_row["observed_t_map"]),
    "n_nodes": example_n_nodes,
    "n_edges": example_n_edges,
    "high_energy_quantile": EXAMPLE_QUANTILE,
    "n_high_edges": example_n_high,
    "node_label_null_perm_idx": EXAMPLE_PERM_IDX,
    "node_label_null_seed": int(example_perm_seed),
    "largest_high_energy_line_graph_cluster_edges": int(len(largest_cluster)),
    "line_graph_max_distance_from_largest_high_energy_cluster": float(np.nanmax(finite_distances)),
    "graph_pdf": str(graph_pdf.relative_to(analysis_dir)),
    "graph_png": str(graph_png.relative_to(analysis_dir)),
    "line_graph_pdf": str(line_graph_pdf.relative_to(analysis_dir)),
    "line_graph_png": str(line_graph_png.relative_to(analysis_dir)),
}
example_spec_path = table_dir / "example_edge_energy_localization_visualization_spec.json"
with example_spec_path.open("w") as handle:
    json.dump(example_spec, handle, indent=2, sort_keys=True)

# %%
heatmap_row = autocorr_observed.sort_values(["moran_i", "one_minus_geary_c"], ascending=False).iloc[0]
heatmap_file = str(heatmap_row["file"])
heatmap_dataset = str(heatmap_row["dataset"])
heatmap_edges = _read_edge_details_for_file(heatmap_file)
heatmap_u = heatmap_edges["u"].to_numpy(dtype=np.int64)
heatmap_v = heatmap_edges["v"].to_numpy(dtype=np.int64)
heatmap_observed_energy = heatmap_edges["edge_energy"].to_numpy(dtype=float)
heatmap_n_nodes = int(max(heatmap_u.max(), heatmap_v.max()) + 1)
heatmap_n_edges = int(len(heatmap_edges))
heatmap_fitness = _load_domain_fitness(heatmap_file)
if len(heatmap_fitness) < heatmap_n_nodes:
    raise RuntimeError(f"Domain fitness table for {heatmap_file} does not cover all graph nodes.")
heatmap_validation_energy = (heatmap_fitness[heatmap_u] - heatmap_fitness[heatmap_v]) ** 2
if not np.allclose(heatmap_validation_energy, heatmap_observed_energy, rtol=1e-7, atol=1e-10):
    raise RuntimeError(f"Observed edge energies for {heatmap_file} do not match the source DMS table.")

heatmap_perm_seed = _seed_for(EXAMPLE_BASE_SEED, EXAMPLE_PERM_IDX, heatmap_file)
heatmap_null_fitness = np.random.default_rng(heatmap_perm_seed).permutation(heatmap_fitness)
heatmap_null_energy = (heatmap_null_fitness[heatmap_u] - heatmap_null_fitness[heatmap_v]) ** 2
heatmap_n_high = int(np.ceil((1.0 - EXAMPLE_QUANTILE) * heatmap_n_edges))
heatmap_high_observed = _top_k_mask(heatmap_observed_energy, heatmap_n_high)

heatmap_incidence = _incidence_lists(heatmap_u, heatmap_v, heatmap_n_nodes)
heatmap_line_adj = _line_graph_adjacency(heatmap_incidence, heatmap_n_edges)
heatmap_largest_cluster = _largest_high_edge_cluster(heatmap_line_adj, heatmap_high_observed)
heatmap_source_nodes = np.unique(np.r_[heatmap_u[heatmap_largest_cluster], heatmap_v[heatmap_largest_cluster]])
heatmap_node_adj = _node_adjacency(heatmap_u, heatmap_v, heatmap_n_nodes)
heatmap_node_distances = _multi_source_node_bfs(heatmap_node_adj, heatmap_source_nodes)
heatmap_coords = _node_spectral_layout(heatmap_u, heatmap_v, heatmap_n_nodes)
heatmap_finite_distances = np.where(
    np.isfinite(heatmap_node_distances),
    heatmap_node_distances,
    np.nanmax(heatmap_node_distances[np.isfinite(heatmap_node_distances)]) + 1,
)
heatmap_node_order = np.lexsort((heatmap_coords[:, 0], heatmap_finite_distances))
heatmap_node_position = np.empty(heatmap_n_nodes, dtype=np.int64)
heatmap_node_position[heatmap_node_order] = np.arange(heatmap_n_nodes)

heatmap_rows = np.r_[heatmap_node_position[heatmap_u], heatmap_node_position[heatmap_v]]
heatmap_cols = np.r_[heatmap_node_position[heatmap_v], heatmap_node_position[heatmap_u]]
heatmap_observed_values = np.log1p(np.r_[heatmap_observed_energy, heatmap_observed_energy])
heatmap_null_values = np.log1p(np.r_[heatmap_null_energy, heatmap_null_energy])
heatmap_vmin = 0.0
heatmap_vmax = float(np.quantile(np.r_[heatmap_observed_values, heatmap_null_values], 0.995))
heatmap_cmap = mpl.colormaps["viridis"].copy()
heatmap_cmap.set_under("white")

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.65), constrained_layout=True)
for label, ax, title, values in [
    ("a", axes[0], "Observed", heatmap_observed_values),
    ("b", axes[1], "Node-label null", heatmap_null_values),
]:
    matrix = sparse.coo_matrix(
        (values, (heatmap_rows, heatmap_cols)),
        shape=(heatmap_n_nodes, heatmap_n_nodes),
    ).toarray()
    im = ax.imshow(
        matrix,
        cmap=heatmap_cmap,
        vmin=max(heatmap_vmin, 1e-12),
        vmax=heatmap_vmax,
        interpolation="nearest",
        aspect="equal",
        rasterized=True,
    )
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Genotypes ordered from high-energy cluster")
    if ax is axes[0]:
        ax.set_ylabel("Same genotype ordering")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.text(-0.07, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold")

cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.015)
cbar.set_label(r"$\log(1 + edge energy)$")
heatmap_pdf = si_figure_dir / "example_node_adjacency_edge_energy_heatmap_observed_vs_null.pdf"
heatmap_png = si_figure_dir / "example_node_adjacency_edge_energy_heatmap_observed_vs_null.png"
fig.savefig(heatmap_pdf, bbox_inches="tight")
fig.savefig(heatmap_png, dpi=300, bbox_inches="tight")
plt.close(fig)

heatmap_spec = {
    "dataset": heatmap_dataset,
    "file": heatmap_file,
    "selection_rule": "largest continuous edge-energy autocorrelation by Moran's I in the edge-energy autocorrelation analysis",
    "moran_i": float(heatmap_row["moran_i"]),
    "geary_c": float(heatmap_row["geary_c"]),
    "one_minus_geary_c": float(heatmap_row["one_minus_geary_c"]),
    "n_nodes": heatmap_n_nodes,
    "n_edges": heatmap_n_edges,
    "node_ordering": "graph distance from endpoints of largest observed top-5% high-energy line-graph cluster, then first topology-only spectral coordinate",
    "high_energy_quantile_for_ordering": EXAMPLE_QUANTILE,
    "largest_high_energy_line_graph_cluster_edges": int(len(heatmap_largest_cluster)),
    "largest_high_energy_cluster_nodes": int(len(heatmap_source_nodes)),
    "node_label_null_perm_idx": EXAMPLE_PERM_IDX,
    "node_label_null_seed": int(heatmap_perm_seed),
    "color_scale": "shared observed/null viridis scale on log1p(edge_energy), clipped at the pooled 99.5th percentile",
    "pdf": str(heatmap_pdf.relative_to(analysis_dir)),
    "png": str(heatmap_png.relative_to(analysis_dir)),
}
heatmap_spec_path = table_dir / "example_node_adjacency_edge_energy_heatmap_spec.json"
with heatmap_spec_path.open("w") as handle:
    json.dump(heatmap_spec, handle, indent=2, sort_keys=True)

# %%
domain_plot = domain_empirical.copy()
fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25))
plot_specs = [
    (
        "null_cut_energy_enrichment_median",
        "observed_cut_energy_enrichment",
        "#4C78A8",
        "Cut-energy enrichment",
    ),
    ("null_moran_i_median", "observed_moran_i", "#59A14F", "Moran's I"),
    ("null_geary_c_median", "observed_geary_c", "#B07AA1", "Geary's C"),
]
for ax, (xcol, ycol, color, title) in zip(axes, plot_specs):
    x = domain_plot[xcol].to_numpy(dtype=float)
    y = domain_plot[ycol].to_numpy(dtype=float)
    lo = min(float(np.nanmin(x)), float(np.nanmin(y)))
    hi = max(float(np.nanmax(x)), float(np.nanmax(y)))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    ax.scatter(x, y, s=18, facecolor=color, edgecolor="black", linewidth=0.4, alpha=0.8)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title)
    ax.set_xlabel("Null median")
    ax.set_ylabel("Observed")
    ax.grid(axis="both", linestyle="--", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
for label, ax in zip(["a", "b", "c"], axes):
    ax.text(-0.22, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold")
fig.tight_layout(w_pad=1.5)
domain_pdf = si_figure_dir / "node_label_permutation_domain_observed_vs_null.pdf"
domain_png = si_figure_dir / "node_label_permutation_domain_observed_vs_null.png"
fig.savefig(domain_pdf, bbox_inches="tight")
fig.savefig(domain_png, dpi=300, bbox_inches="tight")
plt.close(fig)

# %%
print(f"Wrote main permutation-null panel: {main_pdf.relative_to(project_root)}")
print(f"Wrote SI quantile null panel: {quantile_pdf.relative_to(project_root)}")
print(f"Wrote quantile null overlay panel: {quantile_overlay_pdf.relative_to(project_root)}")
print(f"Wrote example graph localization panel: {graph_pdf.relative_to(project_root)}")
print(f"Wrote example line-graph adjacency matrix: {line_graph_pdf.relative_to(project_root)}")
print(f"Wrote example node adjacency edge-energy heatmap: {heatmap_pdf.relative_to(project_root)}")
print(f"Wrote SI domain observed-vs-null panel: {domain_pdf.relative_to(project_root)}")
print(f"Wrote summary table: {(table_dir / 'node_label_permutation_summary.json').relative_to(project_root)}")
