#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", module=r"requests(\..*)?")

try:
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:
    pass

import fitness_landscape as fl
import matplotlib
import networkx as nx
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_BFS_SEEDS = list(range(10))
DEFAULT_EPS_GRID = np.linspace(0.0, 0.6, 10)
DEFAULT_BFS_FRAC = 0.10
DEFAULT_T_MIN = 1e-20
DEFAULT_T_MAX = 1e2
DEFAULT_PRIOR = "uniform"
DEFAULT_PLOT_EPS = 0.5
GREY = "#d3d3d3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the Figure 4 / SI SH3-style analysis from a saved "
            "portable .lsbundle landscape, using ddG folding/binding layers."
        )
    )
    parser.add_argument(
        "input_lsbundle",
        type=Path,
        help="Path to the input portable .lsbundle archive or extracted bundle directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("RESULTS"),
        help="Directory to write JSON outputs and PDF plots.",
    )
    parser.add_argument(
        "--fitness-layer",
        default="binding_fitness",
        help="Landscape layer used as the observable fitness signal.",
    )
    parser.add_argument(
        "--fold-layer",
        default="latent_folding_ddG",
        help="Landscape layer used as the folding ddG signal.",
    )
    parser.add_argument(
        "--bind-layer",
        default="latent_binding_ddG",
        help="Landscape layer used as the binding ddG signal.",
    )
    parser.add_argument(
        "--limiter-layer",
        default="limiter_index",
        help="Name of the attached limiter-index layer to create.",
    )
    parser.add_argument(
        "--bfs-frac",
        type=float,
        default=DEFAULT_BFS_FRAC,
        help="Fraction of nodes to keep in each BFS subsample.",
    )
    parser.add_argument(
        "--bfs-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_BFS_SEEDS,
        help="Seeds for the 10 BFS subsampled graphs.",
    )
    parser.add_argument(
        "--plot-eps",
        type=float,
        default=DEFAULT_PLOT_EPS,
        help="Epsilon used for the boundary-decomposition plot panels.",
    )
    parser.add_argument(
        "--eps-grid-max",
        type=float,
        default=float(DEFAULT_EPS_GRID.max()),
        help="Maximum epsilon used in the interface t_map scan.",
    )
    parser.add_argument(
        "--eps-grid-count",
        type=int,
        default=len(DEFAULT_EPS_GRID),
        help="Number of epsilon values used in the interface t_map scan.",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=DEFAULT_T_MIN,
        help="Lower diffusion-scale bound passed to compute_ruggedness_diffusion_scale.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=DEFAULT_T_MAX,
        help="Upper diffusion-scale bound passed to compute_ruggedness_diffusion_scale.",
    )
    parser.add_argument(
        "--prior",
        default=DEFAULT_PRIOR,
        help="Prior passed to compute_ruggedness_diffusion_scale.",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=1000,
        help="Number of permutations for Dirichlet edge-energy enrichment tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the results directory if it already exists.",
    )
    return parser.parse_args()


def ensure_results_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(
                f"Results directory already exists: {path}\n"
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)


def safe_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    out = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return out
    mu = float(np.nanmean(values[mask]))
    sigma = float(np.nanstd(values[mask]))
    if not np.isfinite(sigma) or sigma == 0.0:
        out[mask] = 0.0
        return out
    out[mask] = (values[mask] - mu) / sigma
    return out


def search_bundle_dir(root: Path) -> Path:
    candidates = [root, root / "bundle"]
    candidates.extend(child for child in root.iterdir() if child.is_dir())
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate manifest.json in extracted bundle under {root}")


def load_landscape_from_bundle(path: Path) -> fl.FitnessLandscape:
    if not path.exists():
        raise FileNotFoundError(f"Input artifact does not exist: {path}")
    if path.is_dir():
        bundle_dir = search_bundle_dir(path)
        return fl.FitnessLandscape.load_bundle_dir(bundle_dir)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}.extract-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(tmp_dir)
        bundle_dir = search_bundle_dir(tmp_dir)
        return fl.FitnessLandscape.load_bundle_dir(bundle_dir)


def require_layer(landscape: fl.FitnessLandscape, layer_name: str) -> None:
    if layer_name not in landscape.fitness_layers:
        raise KeyError(
            f"Required layer '{layer_name}' not found. "
            f"Available layers: {sorted(landscape.fitness_layers.keys())}"
        )


def attach_limiter_index(
    landscape: fl.FitnessLandscape,
    *,
    fold_layer: str,
    bind_layer: str,
    limiter_layer: str,
) -> dict[str, Any]:
    require_layer(landscape, fold_layer)
    require_layer(landscape, bind_layer)

    fold = np.asarray(landscape.fitness_layers[fold_layer].to_scalar(), dtype=float).ravel()
    bind = np.asarray(landscape.fitness_layers[bind_layer].to_scalar(), dtype=float).ravel()
    limiter = safe_zscore(fold) - safe_zscore(bind)

    if limiter_layer in landscape.fitness_layers:
        landscape.detach(limiter_layer)
    landscape.attach(name=limiter_layer, values=limiter, dtype="numeric")

    finite = np.isfinite(limiter)
    negative = finite & (limiter < 0.0)
    positive = finite & (limiter > 0.0)
    zero = finite & np.isclose(limiter, 0.0)

    return {
        "layer_name": limiter_layer,
        "n_total": int(limiter.size),
        "n_finite": int(finite.sum()),
        "n_fold_limited": int(negative.sum()),
        "n_bind_limited": int(positive.sum()),
        "n_zero": int(zero.sum()),
        "min": float(np.nanmin(limiter)) if finite.any() else np.nan,
        "max": float(np.nanmax(limiter)) if finite.any() else np.nan,
        "mean": float(np.nanmean(limiter)) if finite.any() else np.nan,
        "std": float(np.nanstd(limiter)) if finite.any() else np.nan,
    }


def _induced_subgraph_in_order(parent_graph: nx.Graph, ordered_nodes: list[Any]) -> nx.Graph:
    graph_out = parent_graph.__class__()
    graph_out.graph.update(parent_graph.graph)
    for node in ordered_nodes:
        graph_out.add_node(node, **parent_graph.nodes[node])

    subgraph = parent_graph.subgraph(ordered_nodes)
    if subgraph.is_multigraph():
        for u, v, key, data in subgraph.edges(keys=True, data=True):
            graph_out.add_edge(u, v, key=key, **dict(data))
    else:
        for u, v, data in subgraph.edges(data=True):
            graph_out.add_edge(u, v, **dict(data))
    return graph_out


def sub_landscape_from_graph(
    landscape: fl.FitnessLandscape,
    sub_graph: nx.Graph,
) -> fl.FitnessLandscape | None:
    if sub_graph.number_of_nodes() == 0:
        return None

    ordered_nodes = [node for node in landscape._node_order if node in sub_graph]
    if not ordered_nodes:
        return None

    node_index_map = {node: idx for idx, node in enumerate(landscape._node_order)}
    indices = [node_index_map[node] for node in ordered_nodes]

    sub_sequences = [landscape.sequences[idx] for idx in indices]
    sub_fitness = landscape._subset_fitness_layers(indices)
    sub_annotations = landscape._subset_annotation_layers(indices)
    sub_embeddings = (
        {domain: emb[indices].copy() for domain, emb in landscape.embeddings.items()}
        if landscape.embeddings
        else None
    )

    ordered_graph = _induced_subgraph_in_order(sub_graph, ordered_nodes)
    sub_landscape = fl.FitnessLandscape(
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
        sub_landscape.view(landscape._active_view_name)
    return sub_landscape


def bfs_sub_landscape(
    landscape: fl.FitnessLandscape,
    *,
    frac: float,
    seed: int,
) -> fl.FitnessLandscape:
    graph = landscape.graph
    rng = np.random.default_rng(seed)
    target_n = max(int(frac * graph.number_of_nodes()), 2)

    graph_bfs = graph.to_undirected(as_view=True) if graph.is_directed() else graph
    start = rng.choice(list(graph_bfs.nodes()))
    seen = {start}
    frontier = [start]

    while frontier and len(seen) < target_n:
        node = frontier.pop(0)
        for neighbor in graph_bfs.neighbors(node):
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
            if len(seen) >= target_n:
                break

    ordered_nodes = [node for node in landscape._node_order if node in seen]
    ordered_graph = _induced_subgraph_in_order(graph, ordered_nodes)
    return sub_landscape_from_graph(landscape, ordered_graph)


def bfs_induced_subgraph(
    landscape: fl.FitnessLandscape,
    *,
    frac: float,
    seed: int,
    keep_lcc: bool,
) -> tuple[nx.Graph, list[Any]]:
    graph = landscape.graph
    rng = np.random.default_rng(seed)
    target_n = max(int(frac * graph.number_of_nodes()), 2)

    graph_bfs = graph.to_undirected(as_view=True) if graph.is_directed() else graph
    start = rng.choice(list(graph_bfs.nodes()))
    seen = {start}
    frontier = [start]

    while frontier and len(seen) < target_n:
        node = frontier.pop(0)
        for neighbor in graph_bfs.neighbors(node):
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
            if len(seen) >= target_n:
                break

    sub_graph = graph.subgraph(seen).copy()
    if keep_lcc and sub_graph.number_of_nodes() > 0:
        if sub_graph.is_directed():
            components = list(nx.weakly_connected_components(sub_graph))
        else:
            components = list(nx.connected_components(sub_graph))
        largest = max(components, key=len)
        sub_graph = sub_graph.subgraph(largest).copy()

    ordered_nodes = [node for node in landscape._node_order if node in sub_graph]
    sub_graph = _induced_subgraph_in_order(graph, ordered_nodes)
    return sub_graph, ordered_nodes


def partition_landscape_into_regime_components(
    landscape: fl.FitnessLandscape,
    *,
    limiter_layer: str,
    eps: float,
    return_landscapes: bool = True,
) -> dict[str, Any]:
    graph = landscape.graph
    require_layer(landscape, limiter_layer)

    limiter = np.asarray(landscape.fitness_layers[limiter_layer].to_scalar(), dtype=float).ravel()
    node_to_s = {node: limiter[idx] for idx, node in enumerate(landscape._node_order)}
    nodes = list(landscape._node_order)

    fold_nodes = [node for node in nodes if np.isfinite(node_to_s[node]) and node_to_s[node] < -eps]
    bind_nodes = [node for node in nodes if np.isfinite(node_to_s[node]) and node_to_s[node] > +eps]

    boundary_edges: list[tuple[Any, Any]] = []
    boundary_nodes: set[Any] = set()
    for u, v in graph.edges():
        su = node_to_s[u]
        sv = node_to_s[v]
        if not (np.isfinite(su) and np.isfinite(sv)):
            continue
        if (su < -eps and sv > +eps) or (su > +eps and sv < -eps):
            boundary_edges.append((u, v))
            boundary_nodes.add(u)
            boundary_nodes.add(v)

    graph_fold = graph.subgraph(fold_nodes)
    graph_bind = graph.subgraph(bind_nodes)
    graph_boundary = graph.edge_subgraph(boundary_edges)

    def sorted_components(component_graph: nx.Graph) -> list[list[Any]]:
        if component_graph.number_of_nodes() == 0:
            return []
        graph_u = component_graph.to_undirected(as_view=True)
        return [
            list(component)
            for component in sorted(nx.connected_components(graph_u), key=len, reverse=True)
        ]

    fold_components = sorted_components(graph_fold)
    bind_components = sorted_components(graph_bind)
    boundary_components = sorted_components(graph_boundary)

    result: dict[str, Any] = {
        "fold_components": fold_components,
        "bind_components": bind_components,
        "boundary_components": boundary_components,
        "fold_nodes": fold_nodes,
        "bind_nodes": bind_nodes,
        "boundary_nodes": list(boundary_nodes),
        "boundary_edges": boundary_edges,
        "edge_counts": {
            "total": int(graph.number_of_edges()),
            "fold": int(graph_fold.number_of_edges()),
            "bind": int(graph_bind.number_of_edges()),
            "boundary": int(graph_boundary.number_of_edges()),
        },
    }

    if return_landscapes:
        fold_landscapes = [
            sub_landscape_from_graph(landscape, graph_fold.subgraph(component))
            for component in fold_components
        ]
        bind_landscapes = [
            sub_landscape_from_graph(landscape, graph_bind.subgraph(component))
            for component in bind_components
        ]
        boundary_landscapes = [
            sub_landscape_from_graph(landscape, graph_boundary.subgraph(component))
            for component in boundary_components
        ]

        result["fold_landscapes"] = fold_landscapes
        result["bind_landscapes"] = bind_landscapes
        result["boundary_landscapes"] = boundary_landscapes
        result["fold_largest_landscape"] = next((x for x in fold_landscapes if x is not None), None)
        result["bind_largest_landscape"] = next((x for x in bind_landscapes if x is not None), None)
        result["boundary_largest_landscape"] = next((x for x in boundary_landscapes if x is not None), None)

    return result


def partition_exact(
    landscape: fl.FitnessLandscape,
    *,
    limiter_layer: str,
    eps: float,
) -> dict[str, fl.FitnessLandscape | None]:
    parts = partition_landscape_into_regime_components(
        landscape,
        limiter_layer=limiter_layer,
        eps=eps,
        return_landscapes=True,
    )
    return {
        "L_sub_fold": parts.get("fold_largest_landscape"),
        "L_sub_bind": parts.get("bind_largest_landscape"),
        "L_sub_bound": parts.get("boundary_largest_landscape"),
    }


def compute_layout(graph: nx.Graph, *, seed: int) -> dict[Any, Any]:
    try:
        from networkx.drawing.nx_agraph import graphviz_layout

        return graphviz_layout(graph, prog="sfdp")
    except Exception:
        return nx.spring_layout(graph, seed=seed)


def sanitize_ci_arrays(t: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo_fixed = np.minimum(lo, hi)
    hi_fixed = np.maximum(lo, hi)
    lo_plot = np.minimum(lo_fixed, t)
    hi_plot = np.maximum(hi_fixed, t)
    return lo_plot, hi_plot


def diffusion_scale_result(
    landscape: fl.FitnessLandscape | None,
    *,
    fitness_layer: str,
    t_min: float,
    t_max: float,
    prior: str,
) -> dict[str, Any]:
    result = {
        "valid": False,
        "n_nodes": 0,
        "n_edges": 0,
        "t_map": np.nan,
        "t_lower_confidence_interval": np.nan,
        "t_upper_confidence_interval": np.nan,
        "raw": None,
        "reason": None,
    }
    if landscape is None:
        result["reason"] = "missing_landscape"
        return result

    result["n_nodes"] = int(landscape.graph.number_of_nodes())
    result["n_edges"] = int(landscape.graph.number_of_edges())
    if landscape.graph.number_of_nodes() <= 1 or landscape.graph.number_of_edges() == 0:
        result["reason"] = "insufficient_graph"
        return result

    require_layer(landscape, fitness_layer)
    values = np.asarray(landscape.fitness_layers[fitness_layer].to_scalar(), dtype=float).ravel()
    if not np.all(np.isfinite(values)):
        result["reason"] = "non_finite_fitness"
        return result

    landscape.view(fitness_layer)
    raw = fl.analysis.diffusion_scale.compute_ruggedness_diffusion_scale(
        landscape,
        t_max=t_max,
        t_min=t_min,
        prior=prior,
    )
    result["valid"] = True
    result["raw"] = raw
    result["t_map"] = raw.get("t_map", np.nan)
    result["t_lower_confidence_interval"] = raw.get("t_lower_confidence_interval", np.nan)
    result["t_upper_confidence_interval"] = raw.get("t_upper_confidence_interval", np.nan)
    return result


def plot_message_pdf(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tmap_bar(
    bind_res: dict[str, Any],
    boundary_res: dict[str, Any],
    fold_res: dict[str, Any],
    *,
    output_pdf: Path,
) -> None:
    t = np.array(
        [
            bind_res.get("t_map", np.nan),
            boundary_res.get("t_map", np.nan),
            fold_res.get("t_map", np.nan),
        ],
        dtype=float,
    )
    lo = np.array(
        [
            bind_res.get("t_lower_confidence_interval", np.nan),
            boundary_res.get("t_lower_confidence_interval", np.nan),
            fold_res.get("t_lower_confidence_interval", np.nan),
        ],
        dtype=float,
    )
    hi = np.array(
        [
            bind_res.get("t_upper_confidence_interval", np.nan),
            boundary_res.get("t_upper_confidence_interval", np.nan),
            fold_res.get("t_upper_confidence_interval", np.nan),
        ],
        dtype=float,
    )
    x = np.arange(len(t))
    valid = np.isfinite(t) & np.isfinite(lo) & np.isfinite(hi)

    fig, ax = plt.subplots(figsize=(1.5, 2))
    if valid.any():
        lo_plot, hi_plot = sanitize_ci_arrays(t, lo, hi)
        yerr = np.vstack([t - lo_plot, hi_plot - t])
        ax.bar(
            x[valid],
            t[valid],
            color=GREY,
            edgecolor="black",
            linewidth=1,
        )
        ax.errorbar(
            x[valid],
            t[valid],
            yerr=yerr[:, valid],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=6,
            capthick=1,
        )
    if (~valid).any():
        ax.scatter(x[~valid], np.zeros((~valid).sum(), dtype=float), marker="x", color="black", zorder=3)
    ax.set_xticks(x)
    ax.set_ylabel(r"$t_{\mathrm{MAP}}$")
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)


def _pick_boundary_landscape(parts: dict[str, Any]) -> fl.FitnessLandscape | None:
    boundary_landscape = parts.get("boundary_largest_landscape")
    if boundary_landscape is not None:
        return boundary_landscape
    candidates = [
        landscape
        for landscape in parts.get("boundary_landscapes", [])
        if landscape is not None
        and landscape.graph.number_of_nodes() > 0
        and landscape.graph.number_of_edges() > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda landscape: landscape.graph.number_of_nodes())


def run_epsilon_scan(
    landscape: fl.FitnessLandscape,
    *,
    limiter_layer: str,
    fitness_layer: str,
    eps_grid: np.ndarray,
    t_min: float,
    t_max: float,
    prior: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    landscape.view(fitness_layer)

    for eps in eps_grid:
        parts = partition_landscape_into_regime_components(
            landscape,
            limiter_layer=limiter_layer,
            eps=float(eps),
            return_landscapes=True,
        )
        boundary_landscape = _pick_boundary_landscape(parts)

        row: dict[str, Any] = {
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
            "tmap_result": None,
        }
        if (
            boundary_landscape is not None
            and boundary_landscape.graph.number_of_nodes() > 1
            and boundary_landscape.graph.number_of_edges() > 0
        ):
            row["boundary_comp_nodes"] = int(boundary_landscape.graph.number_of_nodes())
            row["boundary_comp_edges"] = int(boundary_landscape.graph.number_of_edges())
            result = diffusion_scale_result(
                boundary_landscape,
                fitness_layer=fitness_layer,
                t_min=t_min,
                t_max=t_max,
                prior=prior,
            )
            row["tmap_result"] = result
            row["t_map"] = result.get("t_map", np.nan)
            row["t_lo"] = result.get("t_lower_confidence_interval", np.nan)
            row["t_hi"] = result.get("t_upper_confidence_interval", np.nan)
        rows.append(row)
    return rows


def plot_epsilon_scan(
    rows: list[dict[str, Any]],
    *,
    size_pdf: Path,
    tmap_pdf: Path,
) -> None:
    eps_arr = np.array([row["eps"] for row in rows], dtype=float)
    t_map = np.array([row["t_map"] for row in rows], dtype=float)
    t_lo = np.array([row["t_lo"] for row in rows], dtype=float)
    t_hi = np.array([row["t_hi"] for row in rows], dtype=float)
    boundary_nodes_all = np.array([row["boundary_nodes_all"] for row in rows], dtype=float)
    boundary_edges_all = np.array([row["boundary_edges_all"] for row in rows], dtype=float)
    boundary_nodes_comp = np.array([row["boundary_comp_nodes"] for row in rows], dtype=float)
    valid = np.isfinite(t_map) & np.isfinite(t_lo) & np.isfinite(t_hi)

    fig1, ax1 = plt.subplots(figsize=(4, 2))
    ax1.plot(eps_arr, boundary_nodes_all, marker="o", lw=1.5, ms=4, label="boundary nodes (all)")
    ax1.plot(eps_arr, boundary_edges_all, marker="o", lw=1.5, ms=4, label="boundary edges (all)")
    ax1.plot(eps_arr, boundary_nodes_comp, marker="o", lw=1.5, ms=4, label="boundary nodes (largest comp)")
    ax1.set_xlabel("eps")
    ax1.set_ylabel("Count")
    ax1.legend(frameon=False)
    ax1.grid(True, linestyle="--")
    fig1.tight_layout()
    ax1.set_ylim(0, 7500)
    fig1.savefig(size_pdf)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(4, 2))
    if valid.any():
        t_lo_s, t_hi_s = sanitize_ci_arrays(t_map, t_lo, t_hi)
        ax2.plot(
            eps_arr[valid],
            t_map[valid],
            marker="o",
            lw=2,
            ms=5,
            color="black",
            label=r"$t_{\mathrm{MAP}}$ (interface)",
        )
        ax2.fill_between(
            eps_arr[valid],
            t_lo_s[valid],
            t_hi_s[valid],
            color=GREY,
            alpha=0.8,
            label="95% CI",
            edgecolor="none",
        )
    if (~valid).any():
        y0 = float(np.nanmin(t_map[valid])) if valid.any() else 0.0
        ax2.scatter(
            eps_arr[~valid],
            np.full((~valid).sum(), y0, dtype=float),
            marker="x",
            color="black",
            label="no valid interface",
        )
    ax2.set_xlabel("eps")
    ax2.set_ylabel(r"$t_{\mathrm{MAP}}$")
    ax2.legend(frameon=False)
    ax2.grid(True, linestyle="--")
    fig2.tight_layout()
    fig2.savefig(tmap_pdf)
    plt.close(fig2)


def plot_limiter_coolwarm(
    graph_sub: nx.Graph,
    node_list_sub: list[Any],
    pos: dict[Any, Any],
    layer_on_subgraph,
    *,
    output_pdf: Path,
    node_size: float = 10.0,
    node_line_width: float = 0.25,
    node_line_color: str = "black",
    edge_width: float = 0.25,
    edge_alpha: float = 0.0,
    fraction_nodes: float = 1.0,
    seed: int = 0,
    figsize: tuple[float, float] = (4.0, 4.0),
    cmap: str = "coolwarm_r",
) -> None:
    if not node_list_sub:
        plot_message_pdf(output_pdf, "Limiter Index", "No nodes in BFS subgraph.")
        return

    limiter = np.asarray(layer_on_subgraph("limiter_index"), dtype=float)
    finite = np.isfinite(limiter)
    if not finite.any():
        plot_message_pdf(output_pdf, "Limiter Index", "Limiter values are all non-finite.")
        return

    scale = np.percentile(np.abs(limiter[finite]), 95) + 1e-12
    limiter_norm = np.clip(limiter / scale, -1, 1)

    n = len(node_list_sub)
    k = max(1, int(np.ceil(np.clip(fraction_nodes, 0.0, 1.0) * n)))
    rng = np.random.default_rng(seed)
    keep_idx = np.arange(n) if k == n else rng.choice(n, size=k, replace=False)
    keep_nodes = [node_list_sub[idx] for idx in keep_idx]
    keep_set = set(keep_nodes)

    xs = np.array([pos[node][0] for node in keep_nodes], dtype=float)
    ys = np.array([pos[node][1] for node in keep_nodes], dtype=float)
    cs = limiter_norm[keep_idx]

    fig, ax = plt.subplots(figsize=figsize)
    if edge_alpha > 0:
        for u, v in graph_sub.edges():
            if u in keep_set and v in keep_set:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                ax.plot([x0, x1], [y0, y1], color="black", lw=edge_width, alpha=edge_alpha, zorder=1)

    ax.scatter(
        xs,
        ys,
        c=cs,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        s=node_size,
        edgecolors=node_line_color,
        linewidths=node_line_width,
        zorder=2,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_boundary_components(
    node_list_sub: list[Any],
    pos: dict[Any, Any],
    lim_map: dict[Any, float],
    bind_landscape: fl.FitnessLandscape | None,
    fold_landscape: fl.FitnessLandscape | None,
    boundary_landscape: fl.FitnessLandscape | None,
    *,
    output_pdf: Path,
    node_size: float = 10.0,
    node_line_width: float = 0.25,
    node_line_color: str = "black",
) -> None:
    if not node_list_sub:
        plot_message_pdf(output_pdf, "Boundary Components", "No nodes in BFS subgraph.")
        return

    lim_values = np.array(list(lim_map.values()), dtype=float)
    finite = np.isfinite(lim_values)
    if not finite.any():
        plot_message_pdf(output_pdf, "Boundary Components", "Limiter values are all non-finite.")
        return

    scale = np.percentile(np.abs(lim_values[finite]), 95) + 1e-12
    norm = Normalize(vmin=-1, vmax=1)

    x_all = np.array([pos[node][0] for node in node_list_sub], dtype=float)
    y_all = np.array([pos[node][1] for node in node_list_sub], dtype=float)
    xmin, xmax = float(x_all.min()), float(x_all.max())
    ymin, ymax = float(y_all.min()), float(y_all.max())

    def draw_component(ax, landscape_component: fl.FitnessLandscape | None) -> None:
        if landscape_component is None or landscape_component.graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No component", ha="center", va="center", transform=ax.transAxes)
        else:
            component_nodes = set(landscape_component.graph.nodes())
            ordered = [node for node in node_list_sub if node in component_nodes]
            x = np.array([pos[node][0] for node in ordered], dtype=float)
            y = np.array([pos[node][1] for node in ordered], dtype=float)
            c = np.clip(np.array([lim_map[node] for node in ordered], dtype=float) / scale, -1, 1)
            ax.scatter(
                x,
                y,
                c=c,
                cmap="coolwarm_r",
                norm=norm,
                s=node_size,
                edgecolors=node_line_color,
                linewidths=node_line_width,
                zorder=2,
            )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)
        ax.set_aspect("equal", adjustable="box")

    fig, axes = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)
    draw_component(axes[0], bind_landscape)
    draw_component(axes[1], fold_landscape)
    draw_component(axes[2], boundary_landscape)

    sm = ScalarMappable(norm=norm, cmap="coolwarm_r")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["folding limited", "", "binding limited"])
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_composite_fitness_01_z(
    graph_sub: nx.Graph,
    node_list_sub: list[Any],
    pos: dict[Any, Any],
    layer_on_subgraph,
    *,
    output_pdf: Path,
    node_size: float = 10.0,
    node_line_width: float = 0.25,
    node_line_color: str = "black",
    edge_width: float = 0.25,
    edge_alpha: float = 0.0,
    fraction_nodes: float = 1.0,
    seed: int = 0,
    figsize: tuple[float, float] = (3.5, 3.0),
    cmap: str = "viridis",
    zorder_base: float = 2.0,
    zorder_span: float = 20.0,
) -> None:
    if not node_list_sub:
        plot_message_pdf(output_pdf, "Fitness", "No nodes in BFS subgraph.")
        return

    fitness = np.asarray(layer_on_subgraph("composite_fitness"), dtype=float)
    finite = np.isfinite(fitness)
    if not finite.any():
        plot_message_pdf(output_pdf, "Fitness", "Fitness values are all non-finite.")
        return

    fmin = float(np.nanmin(fitness))
    fmax = float(np.nanmax(fitness))
    fitness_01 = (fitness - fmin) / (fmax - fmin + 1e-12)
    norm = Normalize(vmin=0.0, vmax=1.0)

    n = len(node_list_sub)
    k = max(1, int(np.ceil(np.clip(fraction_nodes, 0.0, 1.0) * n)))
    rng = np.random.default_rng(seed)
    keep_idx = np.arange(n) if k == n else rng.choice(n, size=k, replace=False)
    keep_nodes = [node_list_sub[idx] for idx in keep_idx]
    keep_set = set(keep_nodes)

    xs = np.array([pos[node][0] for node in keep_nodes], dtype=float)
    ys = np.array([pos[node][1] for node in keep_nodes], dtype=float)
    cs = fitness_01[keep_idx]

    fig, ax = plt.subplots(figsize=figsize)
    if edge_alpha > 0:
        for u, v in graph_sub.edges():
            if u in keep_set and v in keep_set:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                ax.plot([x0, x1], [y0, y1], color="black", lw=edge_width, alpha=edge_alpha, zorder=1)

    order = np.argsort(cs)
    for idx in order:
        zorder = zorder_base + zorder_span * cs[idx]
        ax.scatter(
            [xs[idx]],
            [ys[idx]],
            c=[cs[idx]],
            cmap=cmap,
            norm=norm,
            s=node_size,
            edgecolors=node_line_color,
            linewidths=node_line_width,
            zorder=zorder,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fitness")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)


def norm_free_energy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (float(np.nanmax(values)) - values) / (float(np.nanmax(values) - np.nanmin(values)) + 1e-12)


def plot_latent_contributions_fitness(
    get_layer,
    *,
    output_pdf: Path,
    fold_layer: str,
    bind_layer: str,
    fitness_layer: str,
    drop_bind_zero: bool = True,
    bind_zero_eps: float = 1e-10,
    max_points: int = 15000,
    seed: int = 0,
) -> dict[str, Any]:
    fold_raw = np.asarray(get_layer(fold_layer), dtype=float).ravel()
    bind_raw = np.asarray(get_layer(bind_layer), dtype=float).ravel()
    fit_raw = np.asarray(get_layer(fitness_layer), dtype=float).ravel()

    mask = np.isfinite(fold_raw) & np.isfinite(bind_raw) & np.isfinite(fit_raw)
    fold_raw = fold_raw[mask]
    bind_raw = bind_raw[mask]
    fit_raw = fit_raw[mask]

    bind_is_zero = np.isclose(bind_raw, 0.0, atol=bind_zero_eps)
    keep = ~bind_is_zero if drop_bind_zero else np.ones_like(bind_is_zero, dtype=bool)
    fold_raw = fold_raw[keep]
    bind_raw = bind_raw[keep]
    fit_raw = fit_raw[keep]

    metadata = {
        "n_input_points": int(mask.sum()),
        "n_bind_zero_removed": int(bind_is_zero.sum()) if drop_bind_zero else 0,
        "n_points_after_filtering": int(len(fold_raw)),
    }

    if len(fold_raw) == 0:
        plot_message_pdf(output_pdf, "Latent Contributions", "No finite points remain after filtering.")
        return metadata

    rng = np.random.default_rng(seed)
    if len(fold_raw) > max_points:
        idx = rng.choice(len(fold_raw), size=max_points, replace=False)
        fold_raw = fold_raw[idx]
        bind_raw = bind_raw[idx]
        fit_raw = fit_raw[idx]
    metadata["n_points_plotted"] = int(len(fold_raw))

    fold_n = norm_free_energy(fold_raw)
    bind_n = norm_free_energy(bind_raw)
    fit_n = (fit_raw - float(np.nanmin(fit_raw))) / (float(np.nanmax(fit_raw) - np.nanmin(fit_raw)) + 1e-12)

    fig = plt.figure(figsize=(3.0, 2.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        bind_n,
        fold_n,
        fit_n,
        c=fit_n,
        cmap="viridis",
        norm=colors.Normalize(0, 1),
        s=6,
        alpha=0.95,
        linewidths=0,
    )
    ax.view_init(elev=26, azim=160)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.92, 0.92, 0.92, 1.0))
        axis.pane.set_edgecolor((0.65, 0.65, 0.65, 1.0))
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("Latent binding")
    ax.set_ylabel("Latent folding")
    ax.set_zlabel("Norm. Fitness")
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)
    return metadata


def build_layer_on_subgraph(
    landscape: fl.FitnessLandscape,
    node_list_sub: list[Any],
    *,
    aliases: dict[str, str],
):
    idx_full = {node: idx for idx, node in enumerate(landscape._node_order)}

    def getter(layer_key: str) -> np.ndarray:
        actual_key = aliases.get(layer_key, layer_key)
        require_layer(landscape, actual_key)
        values = np.asarray(landscape.fitness_layers[actual_key].to_scalar(), dtype=float)
        return np.array([values[idx_full[node]] for node in node_list_sub], dtype=float)

    return getter


def edge_energy_enrichment(
    landscape: fl.FitnessLandscape,
    *,
    fitness_layer: str,
    limiter_layer: str,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    require_layer(landscape, fitness_layer)
    require_layer(landscape, limiter_layer)

    node_order = list(landscape._node_order)
    fitness_values = np.asarray(landscape.fitness_layers[fitness_layer].to_scalar(), dtype=float).ravel()
    limiter_values = np.asarray(landscape.fitness_layers[limiter_layer].to_scalar(), dtype=float).ravel()
    node_to_f = {node: fitness_values[idx] for idx, node in enumerate(node_order)}
    node_to_s = {node: limiter_values[idx] for idx, node in enumerate(node_order)}

    edge_nodes: list[tuple[Any, Any]] = []
    edge_energy: list[float] = []
    edge_regime_prod: list[float] = []
    energy_within: list[float] = []
    energy_between: list[float] = []

    for u, v in landscape.graph.edges():
        fu = node_to_f[u]
        fv = node_to_f[v]
        su = node_to_s[u]
        sv = node_to_s[v]
        if not (np.isfinite(fu) and np.isfinite(fv) and np.isfinite(su) and np.isfinite(sv)):
            continue

        energy = float((fu - fv) ** 2)
        prod = float(su * sv)
        edge_nodes.append((u, v))
        edge_energy.append(energy)
        edge_regime_prod.append(prod)

        if prod < 0.0:
            energy_between.append(energy)
        elif prod > 0.0:
            energy_within.append(energy)

    edge_energy_arr = np.asarray(edge_energy, dtype=float)
    edge_prod_arr = np.asarray(edge_regime_prod, dtype=float)
    energy_within_arr = np.asarray(energy_within, dtype=float)
    energy_between_arr = np.asarray(energy_between, dtype=float)

    result: dict[str, Any] = {
        "n_edges_used": int(edge_energy_arr.size),
        "n_within_edges": int(energy_within_arr.size),
        "n_between_edges": int(energy_between_arr.size),
        "mean_diff_between_minus_within": np.nan,
        "q90_diff_between_minus_within": np.nan,
        "q90_null": [],
        "q90_p_value": np.nan,
        "boundary_edge_share": np.nan,
        "boundary_energy_share": np.nan,
        "boundary_energy_share_excess": np.nan,
        "boundary_energy_share_enrichment": np.nan,
        "null_energy_share_fixed": [],
        "null_energy_share_label": [],
        "null_edge_share_label": [],
        "p_energy_share_fixed": np.nan,
        "p_excess_fixed": np.nan,
        "p_enrichment_fixed": np.nan,
        "p_energy_share_label": np.nan,
        "p_excess_label": np.nan,
        "p_enrichment_label": np.nan,
    }
    if edge_energy_arr.size == 0 or energy_within_arr.size == 0 or energy_between_arr.size == 0:
        return result

    def q_stat(values: np.ndarray, q: float = 0.9) -> float:
        return float(np.quantile(values, q))

    result["mean_diff_between_minus_within"] = float(energy_between_arr.mean() - energy_within_arr.mean())
    obs_q90 = q_stat(energy_between_arr, 0.9) - q_stat(energy_within_arr, 0.9)
    result["q90_diff_between_minus_within"] = float(obs_q90)

    rng = np.random.default_rng(seed)
    q90_null = []
    for _ in range(n_perm):
        shuffled_s = rng.permutation(limiter_values)
        node_to_s_shuf = dict(zip(node_order, shuffled_s))
        shuf_between = []
        shuf_within = []
        for (u, v), energy in zip(edge_nodes, edge_energy_arr):
            prod = node_to_s_shuf[u] * node_to_s_shuf[v]
            if prod < 0.0:
                shuf_between.append(energy)
            elif prod > 0.0:
                shuf_within.append(energy)
        if shuf_between and shuf_within:
            q90_null.append(q_stat(np.asarray(shuf_between, dtype=float), 0.9) - q_stat(np.asarray(shuf_within, dtype=float), 0.9))
    q90_null_arr = np.asarray(q90_null, dtype=float)
    result["q90_null"] = q90_null_arr
    if q90_null_arr.size:
        result["q90_p_value"] = float((np.sum(q90_null_arr >= obs_q90) + 1) / (q90_null_arr.size + 1))

    valid_mask = edge_prod_arr != 0.0
    boundary_mask = edge_prod_arr[valid_mask] < 0.0
    edge_energy_defined = edge_energy_arr[valid_mask]
    n_edges_defined = int(valid_mask.sum())
    n_boundary = int(boundary_mask.sum())
    result["n_edges_defined"] = n_edges_defined
    result["n_boundary_edges_defined"] = n_boundary
    if n_edges_defined == 0 or n_boundary == 0:
        return result

    def safe_share(num: float, den: float) -> float:
        return float(num / (den + 1e-12))

    obs_boundary_edge_share = float(boundary_mask.mean())
    obs_boundary_energy_share = safe_share(float(edge_energy_defined[boundary_mask].sum()), float(edge_energy_defined.sum()))
    obs_energy_share_excess = obs_boundary_energy_share - obs_boundary_edge_share
    obs_energy_share_enrichment = safe_share(obs_boundary_energy_share, obs_boundary_edge_share)

    result["boundary_edge_share"] = obs_boundary_edge_share
    result["boundary_energy_share"] = obs_boundary_energy_share
    result["boundary_energy_share_excess"] = obs_energy_share_excess
    result["boundary_energy_share_enrichment"] = obs_energy_share_enrichment

    null_energy_share_fixed = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        pick = np.zeros(n_edges_defined, dtype=bool)
        pick[rng.choice(n_edges_defined, size=n_boundary, replace=False)] = True
        null_energy_share_fixed[idx] = safe_share(float(edge_energy_defined[pick].sum()), float(edge_energy_defined.sum()))

    null_energy_excess_fixed = null_energy_share_fixed - obs_boundary_edge_share
    null_energy_enrichment_fixed = null_energy_share_fixed / (obs_boundary_edge_share + 1e-12)

    result["null_energy_share_fixed"] = null_energy_share_fixed
    result["p_energy_share_fixed"] = float((np.sum(null_energy_share_fixed >= obs_boundary_energy_share) + 1) / (n_perm + 1))
    result["p_excess_fixed"] = float((np.sum(null_energy_excess_fixed >= obs_energy_share_excess) + 1) / (n_perm + 1))
    result["p_enrichment_fixed"] = float((np.sum(null_energy_enrichment_fixed >= obs_energy_share_enrichment) + 1) / (n_perm + 1))

    null_edge_share_label = np.empty(n_perm, dtype=float)
    null_energy_share_label = np.empty(n_perm, dtype=float)
    null_energy_excess_label = np.empty(n_perm, dtype=float)
    null_energy_enrichment_label = np.empty(n_perm, dtype=float)

    for idx in range(n_perm):
        shuffled_s = rng.permutation(limiter_values)
        node_to_s_shuf = dict(zip(node_order, shuffled_s))
        prod_shuf = np.array([node_to_s_shuf[u] * node_to_s_shuf[v] for u, v in edge_nodes], dtype=float)
        valid_shuf = prod_shuf != 0.0
        boundary_shuf = prod_shuf[valid_shuf] < 0.0
        energy_shuf = edge_energy_arr[valid_shuf]
        null_edge_share_label[idx] = float(boundary_shuf.mean()) if boundary_shuf.size else np.nan
        null_energy_share_label[idx] = safe_share(float(energy_shuf[boundary_shuf].sum()), float(energy_shuf.sum()))
        null_energy_excess_label[idx] = null_energy_share_label[idx] - null_edge_share_label[idx]
        null_energy_enrichment_label[idx] = safe_share(null_energy_share_label[idx], null_edge_share_label[idx])

    result["null_edge_share_label"] = null_edge_share_label
    result["null_energy_share_label"] = null_energy_share_label
    result["p_energy_share_label"] = float((np.sum(null_energy_share_label >= obs_boundary_energy_share) + 1) / (n_perm + 1))
    result["p_excess_label"] = float((np.sum(null_energy_excess_label >= obs_energy_share_excess) + 1) / (n_perm + 1))
    result["p_enrichment_label"] = float((np.sum(null_energy_enrichment_label >= obs_energy_share_enrichment) + 1) / (n_perm + 1))
    return result


def run_replicate(
    landscape: fl.FitnessLandscape,
    *,
    replicate_index: int,
    bfs_seed: int,
    args: argparse.Namespace,
    eps_grid: np.ndarray,
    results_dir: Path,
) -> dict[str, Any]:
    replicate_dir = results_dir / f"replicate_{replicate_index:02d}"
    replicate_dir.mkdir(parents=True, exist_ok=True)

    tmap_sub_landscape = bfs_sub_landscape(landscape, frac=args.bfs_frac, seed=bfs_seed)
    if tmap_sub_landscape is None:
        raise RuntimeError(f"Failed to build BFS sub-landscape for seed={bfs_seed}")
    tmap_sub_landscape.view(args.fitness_layer)

    tmap_parts = partition_landscape_into_regime_components(
        tmap_sub_landscape,
        limiter_layer=args.limiter_layer,
        eps=0.0,
        return_landscapes=True,
    )
    tmap_bind_res = diffusion_scale_result(
        tmap_parts.get("bind_largest_landscape"),
        fitness_layer=args.fitness_layer,
        t_min=args.t_min,
        t_max=args.t_max,
        prior=args.prior,
    )
    tmap_boundary_res = diffusion_scale_result(
        tmap_parts.get("boundary_largest_landscape"),
        fitness_layer=args.fitness_layer,
        t_min=args.t_min,
        t_max=args.t_max,
        prior=args.prior,
    )
    tmap_fold_res = diffusion_scale_result(
        tmap_parts.get("fold_largest_landscape"),
        fitness_layer=args.fitness_layer,
        t_min=args.t_min,
        t_max=args.t_max,
        prior=args.prior,
    )
    plot_tmap_bar(
        tmap_bind_res,
        tmap_boundary_res,
        tmap_fold_res,
        output_pdf=replicate_dir / "tmap_vs_subgraph_regime.pdf",
    )

    epsilon_rows = run_epsilon_scan(
        tmap_sub_landscape,
        limiter_layer=args.limiter_layer,
        fitness_layer=args.fitness_layer,
        eps_grid=eps_grid,
        t_min=args.t_min,
        t_max=args.t_max,
        prior=args.prior,
    )
    plot_epsilon_scan(
        epsilon_rows,
        size_pdf=replicate_dir / "eps_vs_comp_size.pdf",
        tmap_pdf=replicate_dir / "epsilon_vs_tmap_interface.pdf",
    )

    plot_graph_sub, plot_node_order = bfs_induced_subgraph(
        landscape,
        frac=args.bfs_frac,
        seed=bfs_seed,
        keep_lcc=True,
    )
    plot_pos = compute_layout(plot_graph_sub, seed=0)

    aliases = {"composite_fitness": args.fitness_layer}
    layer_on_subgraph = build_layer_on_subgraph(landscape, plot_node_order, aliases=aliases)
    plot_limiter_coolwarm(
        plot_graph_sub,
        plot_node_order,
        plot_pos,
        layer_on_subgraph,
        output_pdf=replicate_dir / "limiter_index.pdf",
    )
    plot_composite_fitness_01_z(
        plot_graph_sub,
        plot_node_order,
        plot_pos,
        layer_on_subgraph,
        output_pdf=replicate_dir / "composite_fitness.pdf",
    )
    latent_plot_meta = plot_latent_contributions_fitness(
        layer_on_subgraph,
        output_pdf=replicate_dir / "latent_contribuions_fitness.pdf",
        fold_layer=args.fold_layer,
        bind_layer=args.bind_layer,
        fitness_layer=args.fitness_layer,
        seed=bfs_seed,
    )

    plot_sub_landscape = sub_landscape_from_graph(landscape, plot_graph_sub)
    parts_exact = partition_exact(
        plot_sub_landscape,
        limiter_layer=args.limiter_layer,
        eps=args.plot_eps,
    )
    bind_plot = parts_exact.get("L_sub_bind")
    fold_plot = parts_exact.get("L_sub_fold")
    boundary_plot = parts_exact.get("L_sub_bound")

    lim_values = layer_on_subgraph("limiter_index")
    lim_map = {node: value for node, value in zip(plot_node_order, lim_values)}
    plot_boundary_components(
        plot_node_order,
        plot_pos,
        lim_map,
        bind_plot,
        fold_plot,
        boundary_plot,
        output_pdf=replicate_dir / "boundaries_decomposed.pdf",
    )

    result = {
        "replicate": replicate_index,
        "bfs_seed": bfs_seed,
        "tmap_subgraph": {
            "n_nodes": int(tmap_sub_landscape.graph.number_of_nodes()),
            "n_edges": int(tmap_sub_landscape.graph.number_of_edges()),
        },
        "plot_subgraph": {
            "n_nodes": int(plot_graph_sub.number_of_nodes()),
            "n_edges": int(plot_graph_sub.number_of_edges()),
            "keep_lcc": True,
        },
        "tmap_components": {
            "bind_nodes": len(tmap_parts.get("bind_nodes", [])),
            "boundary_nodes": len(tmap_parts.get("boundary_nodes", [])),
            "fold_nodes": len(tmap_parts.get("fold_nodes", [])),
            "edge_counts": tmap_parts.get("edge_counts", {}),
        },
        "tmap_results": {
            "binding_limited": tmap_bind_res,
            "boundary": tmap_boundary_res,
            "folding_limited": tmap_fold_res,
        },
        "epsilon_scan": epsilon_rows,
        "plot_partition": {
            "eps": args.plot_eps,
            "binding_nodes": int(bind_plot.graph.number_of_nodes()) if bind_plot is not None else 0,
            "folding_nodes": int(fold_plot.graph.number_of_nodes()) if fold_plot is not None else 0,
            "boundary_nodes": int(boundary_plot.graph.number_of_nodes()) if boundary_plot is not None else 0,
        },
        "latent_plot": latent_plot_meta,
        "pdfs": {
            "tmap_vs_subgraph_regime": replicate_dir / "tmap_vs_subgraph_regime.pdf",
            "eps_vs_comp_size": replicate_dir / "eps_vs_comp_size.pdf",
            "epsilon_vs_tmap_interface": replicate_dir / "epsilon_vs_tmap_interface.pdf",
            "limiter_index": replicate_dir / "limiter_index.pdf",
            "boundaries_decomposed": replicate_dir / "boundaries_decomposed.pdf",
            "composite_fitness": replicate_dir / "composite_fitness.pdf",
            "latent_contribuions_fitness": replicate_dir / "latent_contribuions_fitness.pdf",
        },
    }
    write_json(result, replicate_dir / "results.json")
    return result


def main() -> int:
    args = parse_args()
    if len(args.bfs_seeds) == 0:
        raise SystemExit("Provide at least one BFS seed.")

    ensure_results_dir(args.results_dir, overwrite=args.overwrite)
    landscape = load_landscape_from_bundle(args.input_lsbundle.expanduser())

    require_layer(landscape, args.fitness_layer)
    require_layer(landscape, args.fold_layer)
    require_layer(landscape, args.bind_layer)

    limiter_summary = attach_limiter_index(
        landscape,
        fold_layer=args.fold_layer,
        bind_layer=args.bind_layer,
        limiter_layer=args.limiter_layer,
    )

    eps_grid = np.linspace(0.0, args.eps_grid_max, args.eps_grid_count)

    full_graph_energy = edge_energy_enrichment(
        landscape,
        fitness_layer=args.fitness_layer,
        limiter_layer=args.limiter_layer,
        n_perm=args.n_perm,
        seed=0,
    )

    replicate_results = []
    for replicate_index, bfs_seed in enumerate(args.bfs_seeds, start=1):
        replicate_results.append(
            run_replicate(
                landscape,
                replicate_index=replicate_index,
                bfs_seed=bfs_seed,
                args=args,
                eps_grid=eps_grid,
                results_dir=args.results_dir,
            )
        )

    summary = {
        "input_lsbundle": args.input_lsbundle.expanduser(),
        "results_dir": args.results_dir,
        "config": {
            "fitness_layer": args.fitness_layer,
            "fold_layer": args.fold_layer,
            "bind_layer": args.bind_layer,
            "limiter_layer": args.limiter_layer,
            "bfs_frac": args.bfs_frac,
            "bfs_seeds": args.bfs_seeds,
            "plot_eps": args.plot_eps,
            "eps_grid": eps_grid,
            "t_min": args.t_min,
            "t_max": args.t_max,
            "prior": args.prior,
            "n_perm": args.n_perm,
        },
        "landscape": {
            "n_nodes": int(landscape.graph.number_of_nodes()),
            "n_edges": int(landscape.graph.number_of_edges()),
            "available_layers": sorted(landscape.fitness_layers.keys()),
        },
        "limiter_summary": limiter_summary,
        "full_graph_dirichlet_energy_enrichment": full_graph_energy,
        "replicates": replicate_results,
    }
    write_json(summary, args.results_dir / "summary.json")
    print(f"Wrote analysis outputs to {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
