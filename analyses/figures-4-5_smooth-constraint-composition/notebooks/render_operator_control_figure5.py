from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.stats import mannwhitneyu


PROJECT_ROOT = next(
    path for path in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    if (path / "README.md").exists() and (path / "experiments").is_dir() and (path / "analyses").is_dir()
)
ANALYSIS_DIR = PROJECT_ROOT / "analyses/figures-4-5_smooth-constraint-composition"
PROCESSED_DIR = PROJECT_ROOT / "data/processed/constraint_composition"
OUT_DIR = ANALYSIS_DIR / "figures/figure_5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOFTMIN_BETA = 20.0
PRODUCT_EPS = 1e-6
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
M_VALUES = [2, 4, 10, 20, 50, 100]


def generate_nk_states(N: int, K: int, seed: int):
    rng = np.random.default_rng(seed)
    sequences = np.asarray(list(product([0, 1], repeat=N)), dtype=int)
    fitness_values = np.zeros(len(sequences), dtype=float)

    neighbor_sets = []
    for site in range(N):
        choices = [i for i in range(N) if i != site]
        neighbors = rng.choice(choices, size=K, replace=False).tolist() if K else []
        neighbor_sets.append([site] + sorted(neighbors))

    tables = []
    for idxs in neighbor_sets:
        table = rng.random(2 ** len(idxs))
        table -= table.mean()
        tables.append((idxs, table))

    for row_idx, seq in enumerate(sequences):
        total = 0.0
        for idxs, table in tables:
            table_idx = 0
            for site in idxs:
                table_idx = table_idx * 2 + int(seq[site])
            total += table[table_idx]
        fitness_values[row_idx] = total / float(N)

    return sequences, fitness_values


def build_aligned_stack(N: int, K: int, m: int, seed: int, alpha_align: float):
    rng = np.random.default_rng(seed)
    sequences, shared_signal = generate_nk_states(N, K, int(rng.integers(1e9)))
    signals = []
    for _ in range(m):
        _, independent_signal = generate_nk_states(N, K, int(rng.integers(1e9)))
        if alpha_align > 0:
            signal = alpha_align * shared_signal + (1.0 - alpha_align) * independent_signal
        else:
            signal = independent_signal
        signals.append(signal)
    return sequences, np.stack(signals, axis=0)


def to_unit_interval(values):
    values = np.asarray(values, dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def softmin_operator(stacked, beta=SOFTMIN_BETA):
    shifted = stacked - np.min(stacked, axis=0, keepdims=True)
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights, axis=0, keepdims=True)
    composite = np.sum(weights * stacked, axis=0)
    boundary_idx = np.argmax(weights, axis=0)
    return composite, boundary_idx


def product_operator(stacked, eps=PRODUCT_EPS):
    mins = np.min(stacked, axis=1, keepdims=True)
    maxs = np.max(stacked, axis=1, keepdims=True)
    spans = np.where(np.isclose(maxs, mins), 1.0, maxs - mins)
    scaled = (stacked - mins) / spans
    scaled = np.clip(eps + (1.0 - eps) * scaled, eps, 1.0)
    log_product = np.mean(np.log(scaled), axis=0)
    composite = to_unit_interval(np.exp(log_product))
    boundary_idx = np.argmin(scaled, axis=0)
    return composite, boundary_idx


def high_pass_walsh(sequences, signal):
    bits = np.asarray(sequences, dtype=int)
    masks = bits.copy()
    parity = (bits @ masks.T) % 2
    H = np.where(parity == 0, 1.0, -1.0) / np.sqrt(len(bits))
    coeffs = H.T @ np.asarray(signal, dtype=float)
    orders = masks.sum(axis=1)
    coeffs[orders <= 1] = 0.0
    return H @ coeffs


def hypercube_edges(sequences):
    seq_to_idx = {tuple(seq): idx for idx, seq in enumerate(sequences)}
    edges = []
    for idx, seq in enumerate(sequences):
        for site in range(len(seq)):
            if seq[site] == 0:
                neighbor = seq.copy()
                neighbor[site] = 1
                edges.append((idx, seq_to_idx[tuple(neighbor)]))
    return edges


def collect_operator_energies(df, operator_name):
    operator = softmin_operator if operator_name == "softmin" else product_operator
    rows = []
    energy_rows = []
    for _, row in df.iterrows():
        N = int(row["N"])
        K = int(row["K"])
        sequences, stacked = build_aligned_stack(
            N=N,
            K=K,
            m=int(row["m"]),
            seed=int(row["seed"]),
            alpha_align=float(row["alpha_align"]),
        )
        composite, boundary_idx = operator(stacked)
        high_pass = high_pass_walsh(sequences, composite)
        for u, v in hypercube_edges(sequences):
            edge_type = "Boundary" if boundary_idx[u] != boundary_idx[v] else "Internal"
            energy = 0.5 * float((high_pass[u] - high_pass[v]) ** 2)
            energy_rows.append(
                {
                    "operator": operator_name,
                    "edge_type": edge_type,
                    "energy": max(energy, 1e-12),
                    "N": N,
                    "m": int(row["m"]),
                    "alpha_align": float(row["alpha_align"]),
                    "seed": int(row["seed"]),
                }
            )
        rows.append(len(energy_rows))
    return pd.DataFrame(energy_rows)


def example_operator_graph(operator_name):
    operator = softmin_operator if operator_name == "softmin" else product_operator
    sequences, stacked = build_aligned_stack(N=4, K=0, m=2, seed=3, alpha_align=0.0)
    composite, boundary_idx = operator(stacked)
    high_pass = high_pass_walsh(sequences, composite)
    edges = hypercube_edges(sequences)
    return sequences, edges, boundary_idx, high_pass


def add_panel_label(ax, label):
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
    )


def plot_heatmap(ax, df, vmax):
    matrix = (
        df[df["m"].isin(M_VALUES)]
        .pivot_table(index="m", columns="alpha_align", values="tmap", aggfunc="mean")
        .reindex(index=M_VALUES, columns=ALPHAS)
    )
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="coolwarm", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(ALPHAS)))
    ax.set_yticks(np.arange(len(M_VALUES)))
    ax.set_xticks(np.arange(-0.5, len(ALPHAS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(M_VALUES), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Mean $t_{\mathrm{MAP}}$")
    ax.set_xlabel("")
    ax.set_ylabel(r"$m$")
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS], rotation=45, ha="right")
    ax.set_yticklabels([str(m) for m in M_VALUES], rotation=0)


def plot_alpha_boxplot(ax, df):
    data = [
        df.loc[df["alpha_align"].eq(1.0), "tmap"].to_numpy(dtype=float),
        df.loc[df["alpha_align"].eq(0.9), "tmap"].to_numpy(dtype=float),
    ]
    ax.boxplot(
        data,
        tick_labels=["1.0", "0.9"],
        showfliers=False,
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": "lightgray", "alpha": 1},
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "black", "alpha": 1},
        capprops={"color": "black", "alpha": 1},
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$t_{\mathrm{MAP}}$")
    ax.set_ylim(-1, 17)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_spectra(ax, df):
    vcols = [f"v{i}" for i in range(1, 7) if f"v{i}" in df.columns]
    orders = np.array([int(col[1:]) for col in vcols], dtype=float)
    plot_df = df.dropna(subset=vcols + ["alpha_align"]).copy()
    y = plot_df[vcols].to_numpy(dtype=float)
    c = plot_df["alpha_align"].to_numpy(dtype=float)
    lines = [np.column_stack([orders, values]) for values in y]
    norm = Normalize(vmin=0, vmax=1)
    lc = LineCollection(lines, cmap=plt.cm.viridis, norm=norm, linewidths=0.55, alpha=0.72)
    lc.set_array(c)
    ax.add_collection(lc)
    ax.set_xlim(1, 6)
    ax.set_ylim(0, max(1.0, np.nanmax(y) * 1.03))
    ax.set_xlabel("Epistasis order")
    ax.set_ylabel("Variance explained")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    for alpha in [1.0, 0.9, 0.5, 0.25, 0.0]:
        ax.plot([], [], color=plt.cm.viridis(norm(alpha)), lw=4, label=f"{alpha:g}")
    ax.legend(title=r"$\alpha$", frameon=False, fontsize=7, title_fontsize=8, loc="upper right")


def plot_example_graph(ax, operator_name):
    sequences, edges, boundary_idx, high_pass = example_operator_graph(operator_name)
    G = nx.Graph()
    tuple_nodes = [tuple(seq) for seq in sequences]
    G.add_nodes_from(tuple_nodes)
    idx_to_node = dict(enumerate(tuple_nodes))
    boundary_edges = [(idx_to_node[u], idx_to_node[v]) for u, v in edges if boundary_idx[u] != boundary_idx[v]]
    internal_edges = [(idx_to_node[u], idx_to_node[v]) for u, v in edges if boundary_idx[u] == boundary_idx[v]]
    G.add_edges_from(boundary_edges + internal_edges)
    pos = nx.spring_layout(G, seed=2)
    node_values = {idx_to_node[i]: value for i, value in enumerate(high_pass)}
    norm = mpl.colors.Normalize(vmin=float(np.min(high_pass)), vmax=float(np.max(high_pass)))

    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, ax=ax, alpha=0.32, edge_color="#bdbdbd", width=1.0)
    nx.draw_networkx_edges(G, pos, edgelist=boundary_edges, ax=ax, alpha=0.95, edge_color="#E8A838", width=1.5)
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=78,
        node_color=[node_values[node] for node in G.nodes()],
        cmap="coolwarm",
        vmin=norm.vmin,
        vmax=norm.vmax,
        edgecolors="black",
        linewidths=0.7,
    )
    ax.text(0.73, 0.74, "Boundary", transform=ax.transAxes, fontsize=7, ha="left", va="center")
    ax.axis("off")
    return nodes


def plot_energy_boxplot(ax, energy_df):
    data = [
        energy_df.loc[energy_df["edge_type"].eq("Boundary"), "energy"].to_numpy(dtype=float),
        energy_df.loc[energy_df["edge_type"].eq("Internal"), "energy"].to_numpy(dtype=float),
    ]
    ax.boxplot(
        data,
        tick_labels=["Boundary", "Internal"],
        showfliers=False,
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": "lightgray", "alpha": 1},
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "black", "alpha": 1},
        capprops={"color": "black", "alpha": 1},
    )
    ax.set_yscale("log")
    ax.set_ylabel("High-pass energy")
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_operator_figure(operator_name, label, df, energy_df, vmax):
    fig = plt.figure(figsize=(8.8, 5.2))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[2.35, 1.0, 0.35, 3.35],
        height_ratios=[1.0, 1.0],
        wspace=0.95,
        hspace=0.78,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 3])
    ax_d = fig.add_subplot(gs[1, 1:3])
    ax_e = fig.add_subplot(gs[1, 3])

    plot_heatmap(ax_a, df, vmax)
    plot_alpha_boxplot(ax_b, df)
    plot_spectra(ax_c, df)
    graph_nodes = plot_example_graph(ax_d, operator_name)
    plot_energy_boxplot(ax_e, energy_df)

    cbar = fig.colorbar(graph_nodes, ax=ax_d, location="left", fraction=0.05, pad=0.03)
    cbar.set_label("High-pass fitness", rotation=270, labelpad=11)

    for ax, panel in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"), (ax_e, "e")]:
        add_panel_label(ax, panel)

    fig.suptitle(label, x=0.52, y=0.99, fontsize=11)
    out_pdf = OUT_DIR / f"figure_5_{operator_name}_operator_control.pdf"
    out_png = OUT_DIR / f"figure_5_{operator_name}_operator_control.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


def main():
    softmin_df = pd.read_csv(PROCESSED_DIR / "softmin_df.csv")
    product_df = pd.read_csv(PROCESSED_DIR / "product_df.csv")
    vmax = float(
        np.nanmax(
            [
                softmin_df.loc[softmin_df["m"].isin(M_VALUES), "tmap"].mean(),
                product_df.loc[product_df["m"].isin(M_VALUES), "tmap"].mean(),
                softmin_df["tmap"].quantile(0.98),
                product_df["tmap"].quantile(0.98),
            ]
        )
    )
    vmax = max(9.0, min(17.0, vmax))

    energy_frames = {
        "softmin": collect_operator_energies(softmin_df, "softmin"),
        "multiplicative": collect_operator_energies(product_df, "multiplicative"),
    }

    summary_rows = []
    for operator_name, energy_df in energy_frames.items():
        boundary = energy_df.loc[energy_df["edge_type"].eq("Boundary"), "energy"].to_numpy(dtype=float)
        internal = energy_df.loc[energy_df["edge_type"].eq("Internal"), "energy"].to_numpy(dtype=float)
        stat, p_value = mannwhitneyu(boundary, internal, alternative="two-sided")
        summary_rows.append(
            {
                "operator": operator_name,
                "n_boundary_edges": int(len(boundary)),
                "n_internal_edges": int(len(internal)),
                "median_boundary_energy": float(np.median(boundary)),
                "median_internal_energy": float(np.median(internal)),
                "median_boundary_over_internal": float(np.median(boundary) / np.median(internal)),
                "mannwhitney_u": float(stat),
                "mannwhitney_p": float(p_value),
            }
        )

    all_energy = pd.concat(energy_frames.values(), ignore_index=True)
    all_energy.to_csv(OUT_DIR / "figure_5_operator_control_edge_energies.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "figure_5_operator_control_energy_summary.csv", index=False)

    render_operator_figure("softmin", "Soft-min operator", softmin_df, energy_frames["softmin"], vmax)
    render_operator_figure(
        "multiplicative",
        "Multiplicative operator",
        product_df,
        energy_frames["multiplicative"],
        vmax,
    )


if __name__ == "__main__":
    main()
