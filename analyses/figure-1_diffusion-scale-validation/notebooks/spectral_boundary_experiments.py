import gc
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, pearsonr, spearmanr, wilcoxon
from tqdm.auto import tqdm


SPECTRAL_MIN_PARTITION_SIZE = 2
SPECTRAL_EDGE_WEIGHT_QUANTILE_FLOOR = 0.05
SPECTRAL_MAX_SHELL_PLOT = 6
SPECTRAL_EDGE_COLUMNS = [
    "file",
    "dataset",
    "u",
    "v",
    "edge_energy",
    "edge_weight",
    "is_cut",
    "shell_distance",
    "same_side",
]
SPECTRAL_COMPONENT_COLUMNS = [
    "file",
    "dataset",
    "side",
    "component_size",
    "component_t_map",
]
SVD_SPECTRAL_NULL_COLUMNS = [
    "dataset",
    "file",
    "perm_idx",
    "jaccard_overlap",
    "log_odds_ratio_corrected",
    "frac_cut_edges_svd_boundary",
]


def _append_frame_to_csv(frame, csv_path, *, write_header):
    if len(frame) == 0:
        return write_header
    frame.to_csv(
        csv_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )
    return False


def _load_csv_or_empty(csv_path, *, columns=None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def _coerce_dataframe_or_csv(dataframe, csv_path, *, columns=None, label="table"):
    if dataframe is not None:
        return dataframe.copy()
    if csv_path is None:
        raise ValueError(f"Need either `{label}_df` or `{label}_csv`.")
    return _load_csv_or_empty(csv_path, columns=columns)


def _spectral_edge_energy(values, u, v):
    delta = float(values[int(u)] - values[int(v)])
    return float(delta * delta)


def _spectral_build_cut(graph, fitness):
    nodes = [int(n) for n in sorted(graph.nodes())]
    if len(nodes) < 4:
        raise ValueError("Need at least four nodes for a stable spectral bipartition.")

    edge_rows = []
    positive_energies = []
    for u, v in graph.edges():
        u_i = int(u)
        v_i = int(v)
        energy = _spectral_edge_energy(fitness, u_i, v_i)
        edge_rows.append((u_i, v_i, energy))
        if energy > 0:
            positive_energies.append(energy)

    if len(edge_rows) == 0:
        raise ValueError("Graph has no edges.")

    if len(positive_energies) > 0:
        energy_floor = float(
            np.quantile(
                np.asarray(positive_energies, dtype=float),
                SPECTRAL_EDGE_WEIGHT_QUANTILE_FLOOR,
            )
        )
        energy_floor = max(energy_floor, 1e-12)
    else:
        energy_floor = 1e-12

    weighted_graph = nx.Graph()
    weighted_graph.add_nodes_from(nodes)
    for u_i, v_i, energy in edge_rows:
        weight = 1.0 / max(float(energy), energy_floor)
        weighted_graph.add_edge(
            u_i,
            v_i,
            weight=float(weight),
            edge_energy=float(energy),
        )

    try:
        fiedler = np.asarray(
            nx.fiedler_vector(
                weighted_graph,
                weight="weight",
                normalized=False,
                method="tracemin_pcg",
            ),
            dtype=float,
        )
    except Exception:
        lap = nx.laplacian_matrix(
            weighted_graph,
            nodelist=nodes,
            weight="weight",
        ).astype(float)
        eigvals, eigvecs = np.linalg.eigh(np.asarray(lap.todense(), dtype=float))
        order = np.argsort(eigvals)
        if len(order) < 2:
            raise ValueError("Could not compute a nontrivial Fiedler vector.")
        fiedler = np.asarray(eigvecs[:, order[1]], dtype=float)

    threshold = float(np.median(fiedler))
    left_nodes = [nodes[i] for i, val in enumerate(fiedler) if val <= threshold]
    right_nodes = [nodes[i] for i, val in enumerate(fiedler) if val > threshold]

    if (
        len(left_nodes) < SPECTRAL_MIN_PARTITION_SIZE
        or len(right_nodes) < SPECTRAL_MIN_PARTITION_SIZE
    ):
        order = np.argsort(fiedler)
        split_idx = max(SPECTRAL_MIN_PARTITION_SIZE, len(nodes) // 2)
        split_idx = min(split_idx, len(nodes) - SPECTRAL_MIN_PARTITION_SIZE)
        if split_idx <= 0 or split_idx >= len(nodes):
            raise ValueError("Could not form a nontrivial spectral bipartition.")
        left_nodes = [nodes[i] for i in order[:split_idx]]
        right_nodes = [nodes[i] for i in order[split_idx:]]

    labels = {int(n): 0 for n in left_nodes}
    labels.update({int(n): 1 for n in right_nodes})

    cut_edges = [
        (u_i, v_i, energy)
        for u_i, v_i, energy in edge_rows
        if labels[u_i] != labels[v_i]
    ]
    if len(cut_edges) == 0:
        raise ValueError("Spectral partition produced no cut edges.")

    boundary_nodes = sorted(
        {u_i for u_i, _, _ in cut_edges} | {v_i for _, v_i, _ in cut_edges}
    )
    internal_graph = graph.copy()
    internal_graph.remove_edges_from([(u_i, v_i) for u_i, v_i, _ in cut_edges])

    node_shell = {int(n): np.nan for n in nodes}
    shell_lengths = nx.multi_source_dijkstra_path_length(
        internal_graph,
        boundary_nodes,
        weight=None,
    )
    for node_i, dist in shell_lengths.items():
        node_shell[int(node_i)] = int(dist)

    edge_records = []
    for u_i, v_i, energy in edge_rows:
        is_cut = labels[u_i] != labels[v_i]
        if is_cut:
            shell_distance = 0
        else:
            d_u = node_shell.get(u_i, np.nan)
            d_v = node_shell.get(v_i, np.nan)
            if np.isfinite(d_u) and np.isfinite(d_v):
                shell_distance = 1 + int(min(d_u, d_v))
            else:
                shell_distance = np.nan

        edge_records.append(
            {
                "u": int(u_i),
                "v": int(v_i),
                "edge_energy": float(energy),
                "edge_weight": float(weighted_graph[u_i][v_i]["weight"]),
                "is_cut": bool(is_cut),
                "shell_distance": shell_distance,
                "same_side": bool(not is_cut),
            }
        )

    return labels, edge_records, boundary_nodes


def _spectral_component_tmap(
    domain_df_subset,
    build_hamming_landscape_from_df,
    compute_tmap_on_landscape_values,
):
    if len(domain_df_subset) < 2:
        return np.nan
    sub_landscape, sub_fitness = build_hamming_landscape_from_df(
        domain_df_subset.reset_index(drop=True)
    )
    return float(
        compute_tmap_on_landscape_values(
            sub_landscape,
            sub_fitness,
            layer_name="spectral_partition_component",
            detach=True,
        )
    )


def _spectral_partition_tmap(
    domain_df,
    graph,
    labels,
    build_hamming_landscape_from_df,
    compute_tmap_on_landscape_values,
):
    side_rows = []
    component_frames = []
    used_nodes_total = 0

    for side_label in [0, 1]:
        side_nodes = sorted(
            int(node_i) for node_i, lab in labels.items() if int(lab) == side_label
        )
        side_graph = graph.subgraph(side_nodes)
        side_components = [
            sorted(int(n) for n in comp)
            for comp in nx.connected_components(side_graph)
        ]

        comp_rows = []
        comp_tmaps = []
        comp_sizes = []
        for comp_nodes in side_components:
            comp_size = int(len(comp_nodes))
            comp_t = np.nan
            if comp_size >= 2:
                comp_df = domain_df.iloc[comp_nodes].reset_index(drop=True)
                comp_t = _spectral_component_tmap(
                    comp_df,
                    build_hamming_landscape_from_df,
                    compute_tmap_on_landscape_values,
                )
                if np.isfinite(comp_t):
                    comp_tmaps.append(float(comp_t))
                    comp_sizes.append(comp_size)
                    used_nodes_total += comp_size
            comp_rows.append(
                {
                    "side": int(side_label),
                    "component_size": comp_size,
                    "component_t_map": (
                        float(comp_t) if np.isfinite(comp_t) else np.nan
                    ),
                }
            )

        component_frames.append(pd.DataFrame(comp_rows))
        if len(comp_tmaps) > 0:
            side_t = float(
                np.average(
                    np.asarray(comp_tmaps, dtype=float),
                    weights=np.asarray(comp_sizes, dtype=float),
                )
            )
        else:
            side_t = np.nan

        side_rows.append(
            {
                "side": int(side_label),
                "n_nodes": int(len(side_nodes)),
                "n_components": int(len(side_components)),
                "largest_component_size": int(
                    max((len(comp) for comp in side_components), default=0)
                ),
                "n_nodes_with_tmap": int(sum(comp_sizes)),
                "t_map": float(side_t) if np.isfinite(side_t) else np.nan,
            }
        )

    side_df = pd.DataFrame(side_rows)
    component_df = (
        pd.concat(component_frames, ignore_index=True)
        if len(component_frames)
        else pd.DataFrame()
    )
    valid_side = side_df.loc[
        np.isfinite(side_df["t_map"].to_numpy(dtype=float))
        & side_df["n_nodes_with_tmap"].gt(0)
    ].copy()
    if len(valid_side) > 0:
        post_partition_t = float(
            np.average(
                valid_side["t_map"].to_numpy(dtype=float),
                weights=valid_side["n_nodes_with_tmap"].to_numpy(dtype=float),
            )
        )
    else:
        post_partition_t = np.nan

    return {
        "side_df": side_df,
        "component_df": component_df,
        "post_partition_t_map": post_partition_t,
        "node_coverage_fraction": (
            float(used_nodes_total / len(domain_df)) if len(domain_df) else np.nan
        ),
    }


def _summarize_spectral_partition_domain(
    row,
    load_domain_dataframe,
    build_hamming_landscape_from_df,
    compute_tmap_on_landscape_values,
):
    try:
        t_obs = float(getattr(row, "t", getattr(row, "t_map", np.nan)))
    except Exception:
        t_obs = np.nan

    out = {
        "dataset": getattr(row, "dataset", Path(str(getattr(row, "file"))).stem),
        "file": str(getattr(row, "file")),
        "observed_t_map": float(t_obs) if np.isfinite(t_obs) else np.nan,
        "status": "ok",
        "error": "",
        "n_nodes": np.nan,
        "n_edges": np.nan,
        "partition_size_left": np.nan,
        "partition_size_right": np.nan,
        "partition_balance": np.nan,
        "n_boundary_nodes": np.nan,
        "n_cut_edges": np.nan,
        "cut_edge_fraction": np.nan,
        "total_edge_energy": np.nan,
        "mean_edge_energy_all": np.nan,
        "mean_edge_energy_cut": np.nan,
        "mean_edge_energy_noncut": np.nan,
        "cut_energy_fraction": np.nan,
        "cut_energy_enrichment": np.nan,
        "side_0_t_map": np.nan,
        "side_1_t_map": np.nan,
        "post_partition_t_map": np.nan,
        "post_partition_t_map_gain": np.nan,
        "post_partition_t_map_ratio": np.nan,
        "partition_tmap_node_coverage": np.nan,
        "side_0_components": np.nan,
        "side_1_components": np.nan,
        "side_0_largest_component": np.nan,
        "side_1_largest_component": np.nan,
    }

    edge_df = pd.DataFrame()
    component_df = pd.DataFrame()

    try:
        domain_df = load_domain_dataframe(out["file"])
        landscape, fitness = build_hamming_landscape_from_df(domain_df)
        graph = landscape.graph
        out["n_nodes"] = int(graph.number_of_nodes())
        out["n_edges"] = int(graph.number_of_edges())

        labels, edge_records, boundary_nodes = _spectral_build_cut(graph, fitness)
        edge_df = pd.DataFrame(edge_records)
        edge_df.insert(0, "file", out["file"])
        edge_df.insert(1, "dataset", out["dataset"])

        part_res = _spectral_partition_tmap(
            domain_df,
            graph,
            labels,
            build_hamming_landscape_from_df,
            compute_tmap_on_landscape_values,
        )
        side_df = part_res["side_df"]
        component_df = part_res["component_df"].copy()
        if len(component_df) > 0:
            component_df.insert(0, "file", out["file"])
            component_df.insert(1, "dataset", out["dataset"])

        side_sizes = side_df.set_index("side")["n_nodes"].to_dict()
        out["partition_size_left"] = int(side_sizes.get(0, 0))
        out["partition_size_right"] = int(side_sizes.get(1, 0))
        left = float(out["partition_size_left"])
        right = float(out["partition_size_right"])
        if max(left, right) > 0:
            out["partition_balance"] = float(min(left, right) / max(left, right))

        out["n_boundary_nodes"] = int(len(boundary_nodes))
        out["n_cut_edges"] = int(edge_df["is_cut"].sum())
        out["cut_edge_fraction"] = (
            float(out["n_cut_edges"] / out["n_edges"]) if out["n_edges"] else np.nan
        )
        out["total_edge_energy"] = float(edge_df["edge_energy"].sum())
        out["mean_edge_energy_all"] = float(edge_df["edge_energy"].mean())

        cut_mask = edge_df["is_cut"].to_numpy(dtype=bool)
        if cut_mask.any():
            cut_energy = float(edge_df.loc[cut_mask, "edge_energy"].sum())
            out["mean_edge_energy_cut"] = float(
                edge_df.loc[cut_mask, "edge_energy"].mean()
            )
            out["cut_energy_fraction"] = (
                float(cut_energy / out["total_edge_energy"])
                if out["total_edge_energy"] > 0
                else np.nan
            )
        noncut_mask = ~cut_mask
        if noncut_mask.any():
            out["mean_edge_energy_noncut"] = float(
                edge_df.loc[noncut_mask, "edge_energy"].mean()
            )

        if (
            np.isfinite(out["cut_energy_fraction"])
            and np.isfinite(out["cut_edge_fraction"])
            and out["cut_edge_fraction"] > 0
        ):
            out["cut_energy_enrichment"] = float(
                out["cut_energy_fraction"] / out["cut_edge_fraction"]
            )

        out["partition_tmap_node_coverage"] = float(part_res["node_coverage_fraction"])
        out["post_partition_t_map"] = (
            float(part_res["post_partition_t_map"])
            if np.isfinite(part_res["post_partition_t_map"])
            else np.nan
        )
        if np.isfinite(out["observed_t_map"]) and np.isfinite(
            out["post_partition_t_map"]
        ):
            out["post_partition_t_map_gain"] = float(
                out["post_partition_t_map"] - out["observed_t_map"]
            )
            if out["observed_t_map"] > 0:
                out["post_partition_t_map_ratio"] = float(
                    out["post_partition_t_map"] / out["observed_t_map"]
                )

        for _, side_row in side_df.iterrows():
            side = int(side_row["side"])
            out[f"side_{side}_t_map"] = (
                float(side_row["t_map"])
                if np.isfinite(side_row["t_map"])
                else np.nan
            )
            out[f"side_{side}_components"] = int(side_row["n_components"])
            out[f"side_{side}_largest_component"] = int(
                side_row["largest_component_size"]
            )

    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"

    return out, edge_df, component_df


def run_spectral_partition_boundary_tmap(
    *,
    dms_tmap_df,
    load_domain_dataframe,
    build_hamming_landscape_from_df,
    compute_tmap_on_landscape_values,
    outdir,
    max_domains=None,
    load_streamed_frames=False,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    domain_csv = outdir / "spectral_partition_boundary_tmap_domain_summary.csv"
    assoc_csv = outdir / "spectral_partition_boundary_tmap_associations.csv"
    edge_csv = outdir / "spectral_partition_boundary_tmap_edge_details.csv"
    component_csv = outdir / "spectral_partition_boundary_tmap_component_summary.csv"
    fig_path = outdir / "spectral_partition_boundary_tmap.pdf"
    for output_path in [domain_csv, assoc_csv, edge_csv, component_csv, fig_path]:
        if output_path.exists():
            output_path.unlink()

    work_df = dms_tmap_df.copy()
    if max_domains is not None:
        work_df = work_df.head(int(max_domains)).copy()

    domain_rows = []
    write_domain_header = True
    write_edge_header = True
    write_component_header = True
    iterator = work_df.itertuples(index=False)
    for row in tqdm(iterator, total=len(work_df), desc="Spectral partitioning"):
        domain_row, edge_df, component_df = _summarize_spectral_partition_domain(
            row,
            load_domain_dataframe,
            build_hamming_landscape_from_df,
            compute_tmap_on_landscape_values,
        )
        domain_rows.append(domain_row)
        write_domain_header = _append_frame_to_csv(
            pd.DataFrame([domain_row]),
            domain_csv,
            write_header=write_domain_header,
        )
        write_edge_header = _append_frame_to_csv(
            edge_df,
            edge_csv,
            write_header=write_edge_header,
        )
        write_component_header = _append_frame_to_csv(
            component_df,
            component_csv,
            write_header=write_component_header,
        )
        del edge_df
        del component_df
        gc.collect()

    domain_df = pd.DataFrame(domain_rows)
    if write_edge_header:
        pd.DataFrame(columns=SPECTRAL_EDGE_COLUMNS).to_csv(edge_csv, index=False)
    if write_component_header:
        pd.DataFrame(columns=SPECTRAL_COMPONENT_COLUMNS).to_csv(
            component_csv,
            index=False,
        )

    assoc_rows = []
    for feat in [
        "post_partition_t_map_ratio",
        "post_partition_t_map_gain",
        "cut_energy_enrichment",
        "partition_balance",
    ]:
        tmp = domain_df.loc[
            domain_df["status"].eq("ok")
            & np.isfinite(domain_df["observed_t_map"].to_numpy(dtype=float))
            & np.isfinite(domain_df[feat].to_numpy(dtype=float)),
            ["observed_t_map", feat],
        ].copy()
        if len(tmp) < 5:
            continue
        x = tmp[feat].to_numpy(dtype=float)
        if np.unique(x).size < 2:
            continue
        y = tmp["observed_t_map"].to_numpy(dtype=float)
        pear_r, pear_p = pearsonr(x, y)
        spear_rho, spear_p = spearmanr(x, y)
        assoc_rows.append(
            {
                "feature": feat,
                "pearson_r": float(pear_r),
                "pearson_p": float(pear_p),
                "spearman_rho": float(spear_rho),
                "spearman_p": float(spear_p),
                "n_domains": int(len(tmp)),
            }
        )

    assoc_df = pd.DataFrame(assoc_rows)
    if len(assoc_df) > 0:
        assoc_df = assoc_df.sort_values(
            "spearman_rho",
            key=np.abs,
            ascending=False,
        ).reset_index(drop=True)

    domain_df.to_csv(domain_csv, index=False)
    assoc_df.to_csv(assoc_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    plot_specs = [
        ("post_partition_t_map_ratio", "Post-partition / whole $t_{MAP}$"),
        ("post_partition_t_map_gain", "Post-partition $t_{MAP}$ gain"),
    ]
    for ax, (feat, xlabel) in zip(axes, plot_specs):
        tmp = domain_df.loc[
            domain_df["status"].eq("ok")
            & np.isfinite(domain_df["observed_t_map"].to_numpy(dtype=float))
            & np.isfinite(domain_df[feat].to_numpy(dtype=float)),
            ["observed_t_map", feat],
        ].copy()
        if len(tmp) < 3 or np.unique(tmp[feat].to_numpy(dtype=float)).size < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = tmp[feat].to_numpy(dtype=float)
        y = tmp["observed_t_map"].to_numpy(dtype=float)
        rho, rho_p = spearmanr(x, y)
        ax.scatter(
            x,
            y,
            s=28,
            facecolor="lightgrey",
            edgecolor="black",
            linewidth=0.75,
            zorder=3,
        )
        coef = np.polyfit(x, y, deg=1)
        x_line = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(
            x_line,
            coef[0] * x_line + coef[1],
            color="black",
            linewidth=1.0,
            zorder=4,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Observed $t_{\mathrm{MAP}}$")
        ax.set_title(f"rho={rho:.2f}, p={rho_p:.3g}, n={len(tmp)}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    return {
        "domain_df": domain_df,
        "edge_df": (
            _load_csv_or_empty(edge_csv, columns=SPECTRAL_EDGE_COLUMNS)
            if load_streamed_frames
            else None
        ),
        "component_df": (
            _load_csv_or_empty(component_csv, columns=SPECTRAL_COMPONENT_COLUMNS)
            if load_streamed_frames
            else None
        ),
        "assoc_df": assoc_df,
        "domain_csv": domain_csv,
        "assoc_csv": assoc_csv,
        "edge_csv": edge_csv,
        "component_csv": component_csv,
        "fig_path": fig_path,
    }


def run_spectral_cut_distance_profile(
    *,
    spectral_partition_domain_df=None,
    spectral_partition_edge_df=None,
    spectral_partition_domain_csv=None,
    spectral_partition_edge_csv=None,
    outdir,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    profile_csv = outdir / "spectral_cut_distance_shell_profiles.csv"
    domain_csv = outdir / "spectral_cut_distance_domain_metrics.csv"
    fig_path = outdir / "spectral_cut_distance_profile.pdf"

    spectral_partition_domain_df = _coerce_dataframe_or_csv(
        spectral_partition_domain_df,
        spectral_partition_domain_csv,
        label="spectral_partition_domain",
    )
    spectral_partition_edge_df = _coerce_dataframe_or_csv(
        spectral_partition_edge_df,
        spectral_partition_edge_csv,
        columns=SPECTRAL_EDGE_COLUMNS,
        label="spectral_partition_edge",
    )

    valid_domains = spectral_partition_domain_df.loc[
        spectral_partition_domain_df["status"].eq("ok")
        & np.isfinite(
            spectral_partition_domain_df["mean_edge_energy_all"].to_numpy(dtype=float)
        ),
        ["file", "dataset", "observed_t_map", "mean_edge_energy_all"],
    ].copy()

    valid_edges = spectral_partition_edge_df.merge(
        valid_domains,
        on=["file", "dataset"],
        how="inner",
    )
    valid_edges = valid_edges.loc[
        np.isfinite(valid_edges["edge_energy"].to_numpy(dtype=float))
        & np.isfinite(valid_edges["shell_distance"].to_numpy(dtype=float))
    ].copy()
    if len(valid_edges) == 0:
        raise RuntimeError("No valid spectral edge-shell records found.")

    valid_edges["shell_distance"] = valid_edges["shell_distance"].astype(int)
    valid_edges["shell_plot"] = np.minimum(
        valid_edges["shell_distance"],
        SPECTRAL_MAX_SHELL_PLOT,
    )
    valid_edges["edge_energy_norm"] = (
        valid_edges["edge_energy"] / valid_edges["mean_edge_energy_all"]
    )

    profile_df = (
        valid_edges.groupby(
            ["file", "dataset", "observed_t_map", "shell_plot"],
            as_index=False,
        ).agg(
            mean_edge_energy=("edge_energy", "mean"),
            median_edge_energy=("edge_energy", "median"),
            mean_edge_energy_norm=("edge_energy_norm", "mean"),
            n_edges=("edge_energy", "size"),
        )
    )

    shell_metric_rows = []
    for file_name, grp in profile_df.groupby("file", sort=False):
        grp = grp.sort_values("shell_plot").reset_index(drop=True)
        dataset_name = str(grp["dataset"].iloc[0])
        t_map_value = float(grp["observed_t_map"].iloc[0])
        shell_to_mean = dict(
            zip(grp["shell_plot"].astype(int), grp["mean_edge_energy_norm"].astype(float))
        )
        shell_to_weight = dict(zip(grp["shell_plot"].astype(int), grp["n_edges"].astype(float)))

        cut_norm = float(shell_to_mean.get(0, np.nan))
        near_norm = float(shell_to_mean.get(1, np.nan))
        far_shells = [shell for shell in shell_to_mean if shell >= 2]
        if len(far_shells) > 0:
            far_vals = np.asarray([shell_to_mean[shell] for shell in far_shells], dtype=float)
            far_w = np.asarray([shell_to_weight[shell] for shell in far_shells], dtype=float)
            far_norm = float(np.average(far_vals, weights=far_w))
        else:
            far_norm = np.nan

        if np.isfinite(cut_norm) and np.isfinite(far_norm) and far_norm > 0:
            peak_enrichment = float(cut_norm / far_norm)
        else:
            peak_enrichment = np.nan

        shell_metric_rows.append(
            {
                "file": file_name,
                "dataset": dataset_name,
                "observed_t_map": t_map_value,
                "cut_shell_norm_energy": cut_norm,
                "near_shell_norm_energy": near_norm,
                "far_shell_norm_energy": far_norm,
                "boundary_peak_enrichment": peak_enrichment,
                "max_shell_observed": int(grp["shell_plot"].max()),
            }
        )

    domain_df = pd.DataFrame(shell_metric_rows)
    profile_df.to_csv(profile_csv, index=False)
    domain_df.to_csv(domain_csv, index=False)

    assoc_df = pd.DataFrame()
    valid_peak = domain_df.loc[
        np.isfinite(domain_df["observed_t_map"].to_numpy(dtype=float))
        & np.isfinite(domain_df["boundary_peak_enrichment"].to_numpy(dtype=float)),
        ["observed_t_map", "boundary_peak_enrichment"],
    ].copy()
    if len(valid_peak) >= 5 and np.unique(
        valid_peak["boundary_peak_enrichment"].to_numpy(dtype=float)
    ).size >= 2:
        x = valid_peak["boundary_peak_enrichment"].to_numpy(dtype=float)
        y = valid_peak["observed_t_map"].to_numpy(dtype=float)
        pear_r, pear_p = pearsonr(x, y)
        spear_rho, spear_p = spearmanr(x, y)
        assoc_df = pd.DataFrame(
            [
                {
                    "feature": "boundary_peak_enrichment",
                    "pearson_r": float(pear_r),
                    "pearson_p": float(pear_p),
                    "spearman_rho": float(spear_rho),
                    "spearman_p": float(spear_p),
                    "n_domains": int(len(valid_peak)),
                }
            ]
        )

    shell_levels = list(range(0, SPECTRAL_MAX_SHELL_PLOT + 1))
    agg_rows = []
    for shell in shell_levels:
        vals = profile_df.loc[
            profile_df["shell_plot"].eq(shell),
            "mean_edge_energy_norm",
        ].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            agg_rows.append(
                {
                    "shell_plot": shell,
                    "mean_norm": np.nan,
                    "sem_norm": np.nan,
                    "n_domains": 0,
                }
            )
            continue
        sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        agg_rows.append(
            {
                "shell_plot": shell,
                "mean_norm": float(np.mean(vals)),
                "sem_norm": sem,
                "n_domains": int(len(vals)),
            }
        )
    agg_profile_df = pd.DataFrame(agg_rows)

    heatmap_df = (
        profile_df.pivot_table(
            index="file",
            columns="shell_plot",
            values="mean_edge_energy_norm",
            aggfunc="mean",
        ).reindex(columns=shell_levels)
    )
    heatmap_df = heatmap_df.merge(
        valid_domains[["file", "dataset", "observed_t_map"]].drop_duplicates(),
        on="file",
        how="left",
    )
    heatmap_df = heatmap_df.sort_values("observed_t_map", ascending=False).reset_index(
        drop=True
    )
    heatmap_values = heatmap_df[shell_levels].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.2),
        gridspec_kw={"width_ratios": [1.0, 1.3]},
    )
    ax = axes[0]
    ax.errorbar(
        agg_profile_df["shell_plot"],
        agg_profile_df["mean_norm"],
        yerr=agg_profile_df["sem_norm"],
        fmt="o-",
        color="black",
        ecolor="grey",
        elinewidth=1,
        capsize=2,
    )
    ax.set_xlabel("Graph distance from spectral cut")
    ax.set_ylabel("Mean edge energy / domain mean")
    ax.set_xticks(shell_levels)
    ax.set_xticklabels(
        [
            str(s) if s < SPECTRAL_MAX_SHELL_PLOT else f"{SPECTRAL_MAX_SHELL_PLOT}+"
            for s in shell_levels
        ]
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax = axes[1]
    im = ax.imshow(heatmap_values, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Graph distance from spectral cut")
    ax.set_xticks(range(len(shell_levels)))
    ax.set_xticklabels(
        [
            str(s) if s < SPECTRAL_MAX_SHELL_PLOT else f"{SPECTRAL_MAX_SHELL_PLOT}+"
            for s in shell_levels
        ]
    )
    ax.set_ylabel("Domains (sorted by observed $t_{MAP}$)")
    ax.set_yticks([])
    ax.set_title("Per-domain normalized edge-energy profiles")
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Normalized mean edge energy",
    )

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    return {
        "profile_df": profile_df,
        "domain_df": domain_df,
        "assoc_df": assoc_df,
        "profile_csv": profile_csv,
        "domain_csv": domain_csv,
        "fig_path": fig_path,
    }


def run_spectral_cheeger_enrichment(
    *,
    spectral_partition_domain_df=None,
    spectral_partition_domain_csv=None,
    outdir,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "spectral_cut_cheeger_enrichment_summary.csv"
    stats_csv = outdir / "spectral_cut_cheeger_enrichment_stats.csv"
    fig_path = outdir / "spectral_cut_cheeger_enrichment.pdf"

    spectral_partition_domain_df = _coerce_dataframe_or_csv(
        spectral_partition_domain_df,
        spectral_partition_domain_csv,
        label="spectral_partition_domain",
    )

    summary_df = spectral_partition_domain_df.loc[
        spectral_partition_domain_df["status"].eq("ok"),
        [
            "dataset",
            "file",
            "observed_t_map",
            "cut_edge_fraction",
            "cut_energy_fraction",
            "cut_energy_enrichment",
            "n_cut_edges",
            "n_edges",
            "partition_balance",
        ],
    ].copy()

    summary_df = summary_df.loc[
        np.isfinite(summary_df["cut_energy_enrichment"].to_numpy(dtype=float))
        & np.isfinite(summary_df["observed_t_map"].to_numpy(dtype=float))
    ].reset_index(drop=True)
    if len(summary_df) == 0:
        raise RuntimeError("No valid spectral-cut enrichment values found.")

    summary_df["log2_cut_energy_enrichment"] = np.log2(
        summary_df["cut_energy_enrichment"]
    )
    summary_df.to_csv(summary_csv, index=False)

    stats_rows = []
    ratios = summary_df["cut_energy_enrichment"].to_numpy(dtype=float)
    if len(ratios) >= 5:
        try:
            w_stat, w_p = wilcoxon(
                ratios - 1.0,
                alternative="greater",
                zero_method="wilcox",
            )
            stats_rows.append(
                {
                    "test": "wilcoxon_greater_than_1",
                    "statistic": float(w_stat),
                    "p_value": float(w_p),
                    "n_domains": int(len(ratios)),
                }
            )
        except ValueError as exc:
            stats_rows.append(
                {
                    "test": "wilcoxon_greater_than_1",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "n_domains": int(len(ratios)),
                    "note": str(exc),
                }
            )

    if len(ratios) >= 5 and np.unique(ratios).size >= 2:
        x = summary_df["cut_energy_enrichment"].to_numpy(dtype=float)
        y = summary_df["observed_t_map"].to_numpy(dtype=float)
        pear_r, pear_p = pearsonr(x, y)
        spear_rho, spear_p = spearmanr(x, y)
        stats_rows.append(
            {
                "test": "correlation_with_t_map",
                "pearson_r": float(pear_r),
                "pearson_p": float(pear_p),
                "spearman_rho": float(spear_rho),
                "spearman_p": float(spear_p),
                "n_domains": int(len(ratios)),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(stats_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    ax = axes[0]
    ax.hist(
        summary_df["log2_cut_energy_enrichment"].to_numpy(dtype=float),
        bins=16,
        color="lightgrey",
        edgecolor="black",
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\log_2$(cut energy enrichment)")
    ax.set_ylabel("Domains")
    ax.set_title(f"median={summary_df['cut_energy_enrichment'].median():.2f}")

    ax = axes[1]
    ax.scatter(
        summary_df["cut_energy_enrichment"],
        summary_df["observed_t_map"],
        s=28,
        facecolor="lightgrey",
        edgecolor="black",
        linewidth=0.75,
        zorder=3,
    )
    if len(summary_df) >= 3 and np.unique(
        summary_df["cut_energy_enrichment"].to_numpy(dtype=float)
    ).size >= 2:
        x = summary_df["cut_energy_enrichment"].to_numpy(dtype=float)
        y = summary_df["observed_t_map"].to_numpy(dtype=float)
        rho, rho_p = spearmanr(x, y)
        coef = np.polyfit(x, y, deg=1)
        x_line = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(x_line, coef[0] * x_line + coef[1], color="black", linewidth=1.0)
        ax.set_title(f"rho={rho:.2f}, p={rho_p:.3g}, n={len(x)}")
    ax.set_xlabel("Cut energy enrichment")
    ax.set_ylabel(r"Observed $t_{\mathrm{MAP}}$")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    return {
        "summary_df": summary_df,
        "stats_df": stats_df,
        "summary_csv": summary_csv,
        "stats_csv": stats_csv,
        "fig_path": fig_path,
    }


def _compute_boundary_overlap_stats(spectral_cut_mask, comparison_boundary_mask):
    spectral_cut_mask = np.asarray(spectral_cut_mask, dtype=bool)
    comparison_boundary_mask = np.asarray(comparison_boundary_mask, dtype=bool)
    if spectral_cut_mask.shape != comparison_boundary_mask.shape:
        raise ValueError("Boundary masks must have the same shape.")

    a = int(np.sum(spectral_cut_mask & comparison_boundary_mask))
    b = int(np.sum(spectral_cut_mask & ~comparison_boundary_mask))
    c = int(np.sum(~spectral_cut_mask & comparison_boundary_mask))
    d = int(np.sum(~spectral_cut_mask & ~comparison_boundary_mask))

    union = a + b + c
    jaccard = float(a / union) if union > 0 else np.nan
    frac_cut_edges_svd_boundary = float(a / (a + b)) if (a + b) > 0 else np.nan
    frac_svd_boundary_edges_on_cut = float(a / (a + c)) if (a + c) > 0 else np.nan

    if (b * c) == 0:
        if (a * d) > 0:
            odds_ratio_raw = np.inf
        elif (a + b + c + d) > 0:
            odds_ratio_raw = 0.0
        else:
            odds_ratio_raw = np.nan
    else:
        odds_ratio_raw = float((a * d) / (b * c))

    odds_ratio_corrected = float(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
    log_odds_ratio_corrected = float(np.log(odds_ratio_corrected))

    _, fisher_p_greater = fisher_exact([[a, b], [c, d]], alternative="greater")
    _, fisher_p_two_sided = fisher_exact([[a, b], [c, d]], alternative="two-sided")

    return {
        "a_cut_and_svd_boundary": a,
        "b_cut_only": b,
        "c_svd_boundary_only": c,
        "d_neither": d,
        "jaccard_overlap": jaccard,
        "odds_ratio_raw": odds_ratio_raw,
        "odds_ratio_corrected": odds_ratio_corrected,
        "log_odds_ratio_corrected": log_odds_ratio_corrected,
        "fisher_p_greater": float(fisher_p_greater),
        "fisher_p_two_sided": float(fisher_p_two_sided),
        "frac_cut_edges_svd_boundary": frac_cut_edges_svd_boundary,
        "frac_svd_boundary_edges_on_cut": frac_svd_boundary_edges_on_cut,
    }


def _empirical_greater_pvalue(observed_value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0 or not np.isfinite(observed_value):
        return np.nan
    return float((np.sum(null_values >= observed_value) + 1) / (len(null_values) + 1))


def _null_zscore(observed_value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0 or not np.isfinite(observed_value):
        return np.nan
    null_std = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0
    if not np.isfinite(null_std) or null_std <= 0:
        return np.nan
    return float((observed_value - float(np.mean(null_values))) / null_std)


def _plot_overlap_feature_vs_tmap(ax, domain_df, feature, ylabel):
    tmp = domain_df.loc[
        domain_df["status"].eq("ok")
        & np.isfinite(domain_df["observed_t_map"].to_numpy(dtype=float))
        & np.isfinite(domain_df[feature].to_numpy(dtype=float)),
        ["observed_t_map", feature],
    ].copy()
    if len(tmp) < 3 or np.unique(tmp[feature].to_numpy(dtype=float)).size < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()
        return

    x = tmp["observed_t_map"].to_numpy(dtype=float)
    y = tmp[feature].to_numpy(dtype=float)
    rho, rho_p = spearmanr(x, y)
    ax.scatter(
        x,
        y,
        s=28,
        facecolor="lightgrey",
        edgecolor="black",
        linewidth=0.75,
        zorder=3,
    )
    coef = np.polyfit(x, y, deg=1)
    x_line = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(x_line, coef[0] * x_line + coef[1], color="black", linewidth=1.0, zorder=4)
    ax.set_xlabel(r"Observed $t_{\mathrm{MAP}}$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"rho={rho:.2f}, p={rho_p:.3g}, n={len(tmp)}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _summarize_svd_spectral_overlap_domain(
    row,
    spectral_edge_group,
    load_domain_dataframe,
    build_limiter_svd_model,
    assign_limiter_to_sequence,
    *,
    n_permutations,
    seed,
):
    try:
        observed_t = float(getattr(row, "t", getattr(row, "t_map", np.nan)))
    except Exception:
        observed_t = np.nan

    out = {
        "dataset": getattr(row, "dataset", Path(str(getattr(row, "file"))).stem),
        "file": str(getattr(row, "file")),
        "observed_t_map": float(observed_t) if np.isfinite(observed_t) else np.nan,
        "status": "ok",
        "error": "",
        "n_genotypes_total": np.nan,
        "n_genotypes_unique_limiter": np.nan,
        "n_genotypes_incomplete_limiter": np.nan,
        "n_genotypes_ambiguous_limiter": np.nan,
        "frac_genotypes_unique_limiter": np.nan,
        "n_limiters_observed": np.nan,
        "n_spectral_edges_total": np.nan,
        "n_edges_with_both_labels": np.nan,
        "frac_edges_with_both_labels": np.nan,
        "n_spectral_cut_edges_tested": np.nan,
        "n_svd_boundary_edges_tested": np.nan,
        "a_cut_and_svd_boundary": np.nan,
        "b_cut_only": np.nan,
        "c_svd_boundary_only": np.nan,
        "d_neither": np.nan,
        "jaccard_overlap": np.nan,
        "odds_ratio_raw": np.nan,
        "odds_ratio_corrected": np.nan,
        "log_odds_ratio_corrected": np.nan,
        "fisher_p_greater": np.nan,
        "fisher_p_two_sided": np.nan,
        "frac_cut_edges_svd_boundary": np.nan,
        "frac_svd_boundary_edges_on_cut": np.nan,
        "n_permutations": int(n_permutations),
        "null_mean_jaccard_overlap": np.nan,
        "null_mean_log_odds_ratio_corrected": np.nan,
        "null_mean_frac_cut_edges_svd_boundary": np.nan,
        "null_std_jaccard_overlap": np.nan,
        "null_std_log_odds_ratio_corrected": np.nan,
        "null_std_frac_cut_edges_svd_boundary": np.nan,
        "perm_p_jaccard_overlap_greater": np.nan,
        "perm_p_log_odds_ratio_corrected_greater": np.nan,
        "perm_p_frac_cut_edges_svd_boundary_greater": np.nan,
        "perm_z_jaccard_overlap": np.nan,
        "perm_z_log_odds_ratio_corrected": np.nan,
        "perm_z_frac_cut_edges_svd_boundary": np.nan,
    }

    null_df = pd.DataFrame(columns=SVD_SPECTRAL_NULL_COLUMNS)

    try:
        if spectral_edge_group is None:
            raise ValueError("No spectral edge labels found for this domain.")

        domain_df = load_domain_dataframe(out["file"]).reset_index(drop=True)
        model = build_limiter_svd_model(domain_df)

        n_genotypes = int(len(domain_df))
        node_labels = np.full(n_genotypes, np.nan, dtype=float)
        assignment_statuses = []
        for idx, geno_row in enumerate(domain_df.itertuples(index=False)):
            limiter_out = assign_limiter_to_sequence(
                getattr(geno_row, "mutated_sequence"),
                model,
            )
            assignment_statuses.append(str(limiter_out.get("assignment_status", "")))
            limiter_component = limiter_out.get("limiter_component", np.nan)
            if (
                limiter_out.get("assignment_status") == "ok"
                and np.isfinite(limiter_component)
            ):
                node_labels[int(idx)] = int(limiter_component)

        status_series = pd.Series(assignment_statuses, dtype=str)
        unique_mask = np.isfinite(node_labels)
        out["n_genotypes_total"] = n_genotypes
        out["n_genotypes_unique_limiter"] = int(np.sum(unique_mask))
        out["n_genotypes_incomplete_limiter"] = int(status_series.eq("incomplete").sum())
        out["n_genotypes_ambiguous_limiter"] = int(
            status_series.isin(["ambiguous", "wt_or_no_mutations"]).sum()
        )
        if n_genotypes > 0:
            out["frac_genotypes_unique_limiter"] = float(np.mean(unique_mask))

        if np.any(unique_mask):
            out["n_limiters_observed"] = int(
                np.unique(node_labels[unique_mask].astype(int)).size
            )

        u = np.asarray(spectral_edge_group["u"], dtype=np.int32)
        v = np.asarray(spectral_edge_group["v"], dtype=np.int32)
        spectral_cut = np.asarray(spectral_edge_group["is_cut"], dtype=bool)
        out["n_spectral_edges_total"] = int(len(u))

        valid_edge_mask = np.isfinite(node_labels[u]) & np.isfinite(node_labels[v])
        if not np.any(valid_edge_mask):
            raise ValueError("No edges have both spectral and SVD limiter labels.")

        u_valid = u[valid_edge_mask]
        v_valid = v[valid_edge_mask]
        spectral_cut_valid = spectral_cut[valid_edge_mask]

        defined_nodes = np.flatnonzero(np.isfinite(node_labels)).astype(np.int32)
        node_to_defined_idx = np.full(n_genotypes, -1, dtype=np.int32)
        node_to_defined_idx[defined_nodes] = np.arange(len(defined_nodes), dtype=np.int32)
        u_local = node_to_defined_idx[u_valid]
        v_local = node_to_defined_idx[v_valid]
        defined_labels = node_labels[defined_nodes].astype(np.int16, copy=False)
        svd_boundary_valid = defined_labels[u_local] != defined_labels[v_local]

        out["n_edges_with_both_labels"] = int(len(u_valid))
        out["frac_edges_with_both_labels"] = (
            float(len(u_valid) / len(u)) if len(u) > 0 else np.nan
        )
        out["n_spectral_cut_edges_tested"] = int(np.sum(spectral_cut_valid))
        out["n_svd_boundary_edges_tested"] = int(np.sum(svd_boundary_valid))
        out.update(
            _compute_boundary_overlap_stats(
                spectral_cut_valid,
                svd_boundary_valid,
            )
        )

        if int(n_permutations) > 0 and len(defined_labels) > 1:
            rng = np.random.default_rng(seed)
            null_rows = []
            null_jaccard = []
            null_log_or = []
            null_frac_cut = []
            for perm_idx in range(int(n_permutations)):
                permuted_labels = rng.permutation(defined_labels)
                perm_boundary = permuted_labels[u_local] != permuted_labels[v_local]
                perm_stats = _compute_boundary_overlap_stats(
                    spectral_cut_valid,
                    perm_boundary,
                )
                null_rows.append(
                    {
                        "dataset": out["dataset"],
                        "file": out["file"],
                        "perm_idx": int(perm_idx),
                        "jaccard_overlap": perm_stats["jaccard_overlap"],
                        "log_odds_ratio_corrected": perm_stats["log_odds_ratio_corrected"],
                        "frac_cut_edges_svd_boundary": perm_stats[
                            "frac_cut_edges_svd_boundary"
                        ],
                    }
                )
                null_jaccard.append(perm_stats["jaccard_overlap"])
                null_log_or.append(perm_stats["log_odds_ratio_corrected"])
                null_frac_cut.append(perm_stats["frac_cut_edges_svd_boundary"])

            null_df = pd.DataFrame(null_rows, columns=SVD_SPECTRAL_NULL_COLUMNS)
            for feat in [
                "jaccard_overlap",
                "log_odds_ratio_corrected",
                "frac_cut_edges_svd_boundary",
            ]:
                null_values = null_df[feat].to_numpy(dtype=float)
                out[f"null_mean_{feat}"] = (
                    float(np.nanmean(null_values))
                    if np.isfinite(null_values).any()
                    else np.nan
                )
                out[f"null_std_{feat}"] = (
                    float(np.nanstd(null_values, ddof=1))
                    if np.sum(np.isfinite(null_values)) > 1
                    else 0.0
                )
                out[f"perm_p_{feat}_greater"] = _empirical_greater_pvalue(
                    out[feat],
                    null_values,
                )
                out[f"perm_z_{feat}"] = _null_zscore(out[feat], null_values)

    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"

    return out, null_df


def run_svd_spectral_boundary_alignment(
    *,
    dms_tmap_df,
    load_domain_dataframe,
    build_limiter_svd_model,
    assign_limiter_to_sequence,
    spectral_partition_edge_df=None,
    spectral_partition_edge_csv=None,
    outdir,
    n_permutations=500,
    seed=2026,
    max_domains=None,
    load_null_samples=False,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    domain_csv = outdir / "svd_spectral_boundary_overlap_domain_summary.csv"
    assoc_csv = outdir / "svd_spectral_boundary_overlap_associations.csv"
    stats_csv = outdir / "svd_spectral_boundary_overlap_stats.csv"
    null_csv = outdir / "svd_spectral_boundary_overlap_null_samples.csv"
    fig_path = outdir / "svd_spectral_boundary_overlap_summary.pdf"
    for output_path in [domain_csv, assoc_csv, stats_csv, null_csv, fig_path]:
        if output_path.exists():
            output_path.unlink()

    spectral_partition_edge_df = _coerce_dataframe_or_csv(
        spectral_partition_edge_df,
        spectral_partition_edge_csv,
        columns=SPECTRAL_EDGE_COLUMNS,
        label="spectral_partition_edge",
    )
    spectral_partition_edge_df = spectral_partition_edge_df[
        ["file", "u", "v", "is_cut"]
    ].copy()
    spectral_partition_edge_df["file"] = spectral_partition_edge_df["file"].astype(str)
    spectral_partition_edge_df["u"] = spectral_partition_edge_df["u"].astype(np.int32)
    spectral_partition_edge_df["v"] = spectral_partition_edge_df["v"].astype(np.int32)
    spectral_partition_edge_df["is_cut"] = spectral_partition_edge_df["is_cut"].astype(bool)

    spectral_edges_by_file = {}
    for file_name, grp in spectral_partition_edge_df.groupby("file", sort=False):
        spectral_edges_by_file[str(file_name)] = {
            "u": grp["u"].to_numpy(dtype=np.int32, copy=True),
            "v": grp["v"].to_numpy(dtype=np.int32, copy=True),
            "is_cut": grp["is_cut"].to_numpy(dtype=bool, copy=True),
        }
    del spectral_partition_edge_df
    gc.collect()

    work_df = dms_tmap_df.copy()
    if max_domains is not None:
        work_df = work_df.head(int(max_domains)).copy()

    domain_rows = []
    write_null_header = True
    master_rng = np.random.default_rng(seed)
    iterator = work_df.itertuples(index=False)
    for row in tqdm(iterator, total=len(work_df), desc="SVD/spectral overlap"):
        file_name = str(getattr(row, "file"))
        domain_seed = (
            int(master_rng.integers(0, np.iinfo(np.int32).max))
            if int(n_permutations) > 0
            else None
        )
        domain_row, null_df = _summarize_svd_spectral_overlap_domain(
            row,
            spectral_edges_by_file.get(file_name),
            load_domain_dataframe,
            build_limiter_svd_model,
            assign_limiter_to_sequence,
            n_permutations=int(n_permutations),
            seed=domain_seed,
        )
        domain_rows.append(domain_row)
        write_null_header = _append_frame_to_csv(
            null_df,
            null_csv,
            write_header=write_null_header,
        )
        del null_df
        gc.collect()

    domain_df = pd.DataFrame(domain_rows)
    if write_null_header:
        pd.DataFrame(columns=SVD_SPECTRAL_NULL_COLUMNS).to_csv(null_csv, index=False)

    assoc_rows = []
    for feat in [
        "log_odds_ratio_corrected",
        "jaccard_overlap",
        "frac_cut_edges_svd_boundary",
    ]:
        tmp = domain_df.loc[
            domain_df["status"].eq("ok")
            & np.isfinite(domain_df["observed_t_map"].to_numpy(dtype=float))
            & np.isfinite(domain_df[feat].to_numpy(dtype=float)),
            ["observed_t_map", feat],
        ].copy()
        if len(tmp) < 5 or np.unique(tmp[feat].to_numpy(dtype=float)).size < 2:
            continue
        x = tmp[feat].to_numpy(dtype=float)
        y = tmp["observed_t_map"].to_numpy(dtype=float)
        pear_r, pear_p = pearsonr(x, y)
        spear_rho, spear_p = spearmanr(x, y)
        assoc_rows.append(
            {
                "feature": feat,
                "pearson_r": float(pear_r),
                "pearson_p": float(pear_p),
                "spearman_rho": float(spear_rho),
                "spearman_p": float(spear_p),
                "n_domains": int(len(tmp)),
            }
        )

    assoc_df = pd.DataFrame(assoc_rows)
    if len(assoc_df) > 0:
        assoc_df = assoc_df.sort_values(
            "spearman_rho",
            key=np.abs,
            ascending=False,
        ).reset_index(drop=True)

    stats_rows = []
    valid_log_or = domain_df.loc[
        domain_df["status"].eq("ok")
        & np.isfinite(domain_df["log_odds_ratio_corrected"].to_numpy(dtype=float)),
        "log_odds_ratio_corrected",
    ].to_numpy(dtype=float)
    if len(valid_log_or) >= 5:
        try:
            w_stat, w_p = wilcoxon(
                valid_log_or,
                alternative="greater",
                zero_method="wilcox",
            )
            stats_rows.append(
                {
                    "test": "wilcoxon_log_odds_ratio_corrected_greater_than_0",
                    "statistic": float(w_stat),
                    "p_value": float(w_p),
                    "n_domains": int(len(valid_log_or)),
                    "median_value": float(np.median(valid_log_or)),
                }
            )
        except ValueError as exc:
            stats_rows.append(
                {
                    "test": "wilcoxon_log_odds_ratio_corrected_greater_than_0",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "n_domains": int(len(valid_log_or)),
                    "note": str(exc),
                }
            )

    valid_fisher = domain_df.loc[
        domain_df["status"].eq("ok")
        & np.isfinite(domain_df["fisher_p_greater"].to_numpy(dtype=float)),
        "fisher_p_greater",
    ].to_numpy(dtype=float)
    if len(valid_fisher) > 0:
        n_sig = int(np.sum(valid_fisher < 0.05))
        binom_res = binomtest(n_sig, len(valid_fisher), p=0.05, alternative="greater")
        stats_rows.append(
            {
                "test": "count_fisher_p_greater_lt_0.05",
                "n_significant": n_sig,
                "n_domains": int(len(valid_fisher)),
                "expected_under_null": float(0.05 * len(valid_fisher)),
                "p_value": float(binom_res.pvalue),
            }
        )

    for feat in [
        "log_odds_ratio_corrected",
        "jaccard_overlap",
        "frac_cut_edges_svd_boundary",
    ]:
        vals = domain_df.loc[
            domain_df["status"].eq("ok")
            & np.isfinite(domain_df[feat].to_numpy(dtype=float)),
            feat,
        ].to_numpy(dtype=float)
        if len(vals) > 0:
            stats_rows.append(
                {
                    "test": f"median_{feat}",
                    "median_value": float(np.median(vals)),
                    "mean_value": float(np.mean(vals)),
                    "n_domains": int(len(vals)),
                }
            )

    stats_df = pd.DataFrame(stats_rows)
    domain_df.to_csv(domain_csv, index=False)
    assoc_df.to_csv(assoc_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    ax = axes[0, 0]
    hist_vals = domain_df.loc[
        domain_df["status"].eq("ok")
        & np.isfinite(domain_df["log_odds_ratio_corrected"].to_numpy(dtype=float)),
        "log_odds_ratio_corrected",
    ].to_numpy(dtype=float)
    if len(hist_vals) > 0:
        ax.hist(hist_vals, bins=16, color="lightgrey", edgecolor="black")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Corrected log odds ratio")
        ax.set_ylabel("Domains")
        ax.set_title(f"median={np.median(hist_vals):.2f}")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()

    _plot_overlap_feature_vs_tmap(
        axes[0, 1],
        domain_df,
        "frac_cut_edges_svd_boundary",
        "Fraction of cut edges\nthat are SVD boundaries",
    )
    _plot_overlap_feature_vs_tmap(
        axes[1, 0],
        domain_df,
        "jaccard_overlap",
        "Jaccard overlap",
    )
    _plot_overlap_feature_vs_tmap(
        axes[1, 1],
        domain_df,
        "log_odds_ratio_corrected",
        "Corrected log odds ratio",
    )

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    return {
        "domain_df": domain_df,
        "assoc_df": assoc_df,
        "stats_df": stats_df,
        "null_samples_df": (
            _load_csv_or_empty(null_csv, columns=SVD_SPECTRAL_NULL_COLUMNS)
            if load_null_samples
            else None
        ),
        "domain_csv": domain_csv,
        "assoc_csv": assoc_csv,
        "stats_csv": stats_csv,
        "null_csv": null_csv,
        "fig_path": fig_path,
    }
