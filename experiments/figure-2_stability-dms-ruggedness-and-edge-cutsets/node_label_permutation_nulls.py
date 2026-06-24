#!/usr/bin/env python3
"""Node-label permutation nulls for the Figure 2 stability-DMS experiment.

Each invocation processes a half-open range of global permutation indices. For every
permutation and every retained stability-DMS domain, the node fitness labels are
permuted before recomputing edge energies, line-graph autocorrelation, and the
fitness-weighted spectral bipartition.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import fitness_landscape as fl
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from paper_runtime import prepare_native_experiment

ENV = prepare_native_experiment(__file__)
SCRIPT_DIR = ENV["script_dir"]
PROJECT_ROOT = ENV["project_root"]
OUTPUT_DIR = ENV["output_dir"]
WORK_DIR = ENV["work_dir"]
NOTEBOOK_DIR = ENV["notebook_dir"]
PROCESSED_DIR = ENV["processed_dir"] / "stability_dms"
DATA_FILES = ENV["data_files"]

os.chdir(NOTEBOOK_DIR)
import spectral_boundary_experiments as sbe

QUANTILES = (0.75, 0.90, 0.95, 0.99)
DEFAULT_BASE_SEED = 2026052701


def _load_run_null_config() -> dict[str, object]:
    config: dict[str, object] = {}
    for env_name, key in [
        ("PAPER_NULL_PERM_START", "perm_start"),
        ("PAPER_NULL_PERM_STOP", "perm_stop"),
        ("PAPER_NULL_BASE_SEED", "base_seed"),
    ]:
        if env_name in os.environ:
            config[key] = int(os.environ[env_name])

    config.setdefault("perm_start", 0)
    config.setdefault("perm_stop", 100)
    config.setdefault("base_seed", DEFAULT_BASE_SEED)
    config["perm_start"] = int(config["perm_start"])
    config["perm_stop"] = int(config["perm_stop"])
    config["base_seed"] = int(config["base_seed"])
    if config["perm_stop"] <= config["perm_start"]:
        raise ValueError("Permutation stop must be greater than permutation start.")
    return config


def _seed_for(base_seed: int, perm_idx: int, file_name: str) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{perm_idx}:{file_name}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % (2**32)


def _load_domain_table() -> pd.DataFrame:
    table_path = PROCESSED_DIR / "megascale_folding_tmap_table.csv"
    if table_path.exists():
        table = pd.read_csv(table_path)
        return table.loc[table["file"].notna(), ["dataset", "file"]].drop_duplicates().reset_index(drop=True)

    dms_dir = DATA_FILES / "megascale_folding"
    files = sorted(path.name for path in dms_dir.glob("*.csv") if not path.name.startswith("megascale_folding_tmap"))
    return pd.DataFrame(
        {
            "dataset": ["-".join(file_name.replace(".csv", "").split("_")[:2]) for file_name in files],
            "file": files,
        }
    )


def _load_domain_dataframe(file_name: str) -> pd.DataFrame:
    domain_df = pd.read_csv(DATA_FILES / "megascale_folding" / file_name)
    domain_df = domain_df.replace([np.inf, -np.inf], np.nan)
    return domain_df.dropna(subset=["mutated_sequence", "DMS_score"]).reset_index(drop=True)


def _build_hamming_graph(domain_df: pd.DataFrame) -> tuple[nx.Graph, np.ndarray]:
    sequences = [fl.BaseNumpySequence(seq) for seq in domain_df["mutated_sequence"]]
    fitness = domain_df["DMS_score"].to_numpy(dtype=float)
    landscape = fl.FitnessLandscape.build(sequences, graph="hamming")
    graph = landscape.graph
    if not nx.is_connected(graph):
        raise ValueError("Sampled landscape is disconnected.")
    return graph, fitness


def _edge_arrays(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    edges = [(int(u), int(v)) for u, v in graph.edges()]
    if not edges:
        raise ValueError("Graph has no edges.")
    u, v = zip(*edges)
    return np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)


def _edge_energy(values: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    delta = values[u] - values[v]
    return np.asarray(delta * delta, dtype=float)


def _incidence_lists(u: np.ndarray, v: np.ndarray) -> dict[int, list[int]]:
    incidence: dict[int, list[int]] = {}
    for edge_idx, (node_u, node_v) in enumerate(zip(u, v)):
        incidence.setdefault(int(node_u), []).append(edge_idx)
        incidence.setdefault(int(node_v), []).append(edge_idx)
    return incidence


def _row_standardized_line_graph_moments(values: np.ndarray, incidence: dict[int, list[int]]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    n_edges = int(x.size)
    centered = x - float(np.mean(x))
    denom = float(np.dot(centered, centered))

    line_degree = np.zeros(n_edges, dtype=np.int64)
    neighbor_centered_sum = np.zeros(n_edges, dtype=float)
    neighbor_sqdiff_sum = np.zeros(n_edges, dtype=float)
    undirected_line_pairs = 0

    for incident_edges in incidence.values():
        degree = len(incident_edges)
        if degree < 2:
            continue
        idx = np.asarray(incident_edges, dtype=np.int64)
        undirected_line_pairs += degree * (degree - 1) // 2
        line_degree[idx] += degree - 1

        x_i = x[idx]
        centered_i = centered[idx]
        sum_x = float(np.sum(x_i))
        sum_x2 = float(np.dot(x_i, x_i))
        sum_centered = float(np.sum(centered_i))

        neighbor_centered_sum[idx] += sum_centered - centered_i
        neighbor_sqdiff_sum[idx] += (
            (degree - 1) * x_i * x_i
            - 2.0 * x_i * (sum_x - x_i)
            + (sum_x2 - x_i * x_i)
        )

    has_neighbors = line_degree > 0
    if denom <= 0 or not np.any(has_neighbors):
        return {
            "n_edges": n_edges,
            "n_line_pairs": int(undirected_line_pairs),
            "moran_i": np.nan,
            "geary_c": np.nan,
            "line_degree_mean": float(np.mean(line_degree[has_neighbors])) if np.any(has_neighbors) else np.nan,
        }

    weighted_neighbor_mean = neighbor_centered_sum[has_neighbors] / line_degree[has_neighbors]
    moran_i = float(np.sum(centered[has_neighbors] * weighted_neighbor_mean) / denom)

    weighted_sqdiff_mean = neighbor_sqdiff_sum[has_neighbors] / line_degree[has_neighbors]
    geary_c = float(((n_edges - 1) / (2.0 * n_edges)) * np.sum(weighted_sqdiff_mean) / denom)

    return {
        "n_edges": n_edges,
        "n_line_pairs": int(undirected_line_pairs),
        "moran_i": moran_i,
        "geary_c": geary_c,
        "line_degree_mean": float(np.mean(line_degree[has_neighbors])),
    }


def _quantile_neighbor_stats(
    edge_energy: np.ndarray,
    incidence: dict[int, list[int]],
    quantile: float,
) -> dict[str, float]:
    energy = np.asarray(edge_energy, dtype=float)
    threshold = float(np.quantile(energy, quantile))
    high = energy >= threshold
    n_edges = int(energy.size)
    n_high = int(np.sum(high))

    line_degree = np.zeros(n_edges, dtype=np.int64)
    high_neighbor_count = np.zeros(n_edges, dtype=np.int64)

    for incident_edges in incidence.values():
        degree = len(incident_edges)
        if degree < 2:
            continue
        idx = np.asarray(incident_edges, dtype=np.int64)
        n_incident_high = int(np.sum(high[idx]))
        line_degree[idx] += degree - 1
        high_neighbor_count[idx] += n_incident_high - high[idx].astype(np.int64)

    high_with_neighbors = high & (line_degree > 0)
    if not np.any(high_with_neighbors) or n_edges <= 1 or n_high <= 1:
        observed_neighbor_high_probability = np.nan
        expected_neighbor_high_probability = np.nan
        neighbor_high_enrichment = np.nan
    else:
        observed_neighbor_high_probability = float(
            np.mean(high_neighbor_count[high_with_neighbors] / line_degree[high_with_neighbors])
        )
        expected_neighbor_high_probability = float((n_high - 1) / (n_edges - 1))
        neighbor_high_enrichment = observed_neighbor_high_probability / expected_neighbor_high_probability

    binary_stats = _row_standardized_line_graph_moments(high.astype(float), incidence)
    return {
        "energy_quantile": quantile,
        "energy_threshold": threshold,
        "n_high_edges": n_high,
        "high_edge_fraction": float(n_high / n_edges),
        "observed_neighbor_high_probability": observed_neighbor_high_probability,
        "expected_neighbor_high_probability": expected_neighbor_high_probability,
        "neighbor_high_enrichment": neighbor_high_enrichment,
        "binary_moran_i": binary_stats["moran_i"],
        "binary_geary_c": binary_stats["geary_c"],
    }


def _spectral_summary(graph: nx.Graph, permuted_fitness: np.ndarray) -> dict[str, object]:
    labels, edge_records, boundary_nodes = sbe._spectral_build_cut(graph, permuted_fitness)
    edge_df = pd.DataFrame(edge_records)
    n_edges = int(len(edge_df))
    cut_mask = edge_df["is_cut"].to_numpy(dtype=bool)
    total_edge_energy = float(edge_df["edge_energy"].sum())
    n_cut_edges = int(cut_mask.sum())
    cut_edge_fraction = float(n_cut_edges / n_edges) if n_edges else np.nan

    if n_cut_edges > 0:
        cut_energy = float(edge_df.loc[cut_mask, "edge_energy"].sum())
        mean_edge_energy_cut = float(edge_df.loc[cut_mask, "edge_energy"].mean())
        cut_energy_fraction = float(cut_energy / total_edge_energy) if total_edge_energy > 0 else np.nan
    else:
        mean_edge_energy_cut = np.nan
        cut_energy_fraction = np.nan

    noncut_mask = ~cut_mask
    mean_edge_energy_noncut = (
        float(edge_df.loc[noncut_mask, "edge_energy"].mean()) if noncut_mask.any() else np.nan
    )
    cut_energy_enrichment = (
        float(cut_energy_fraction / cut_edge_fraction)
        if np.isfinite(cut_energy_fraction) and np.isfinite(cut_edge_fraction) and cut_edge_fraction > 0
        else np.nan
    )

    side_counts = pd.Series(labels).value_counts().to_dict()
    left = int(side_counts.get(0, 0))
    right = int(side_counts.get(1, 0))
    partition_balance = float(min(left, right) / max(left, right)) if max(left, right) else np.nan

    return {
        "status": "ok",
        "error": "",
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": n_edges,
        "partition_size_left": left,
        "partition_size_right": right,
        "partition_balance": partition_balance,
        "n_boundary_nodes": int(len(boundary_nodes)),
        "n_cut_edges": n_cut_edges,
        "cut_edge_fraction": cut_edge_fraction,
        "total_edge_energy": total_edge_energy,
        "mean_edge_energy_all": float(edge_df["edge_energy"].mean()),
        "mean_edge_energy_cut": mean_edge_energy_cut,
        "mean_edge_energy_noncut": mean_edge_energy_noncut,
        "cut_energy_fraction": cut_energy_fraction,
        "cut_energy_enrichment": cut_energy_enrichment,
    }


def _write_table(frame: pd.DataFrame, path: Path, mirror_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    shutil.copy2(path, mirror_dir / path.name)
    print(f"Wrote {path}")


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        if math.isnan(value_f) or math.isinf(value_f):
            return None
        return value_f
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    run_config = _load_run_null_config()
    perm_start = int(run_config["perm_start"])
    perm_stop = int(run_config["perm_stop"])
    base_seed = int(run_config["base_seed"])
    chunk_id = f"chunk_{perm_start:04d}_{perm_stop:04d}"
    perm_indices = list(range(perm_start, perm_stop))

    domain_table = _load_domain_table()
    processed_null_dir = PROCESSED_DIR / "node_label_permutation_nulls"
    output_null_dir = OUTPUT_DIR / "node_label_permutation_nulls"
    processed_null_dir.mkdir(parents=True, exist_ok=True)
    output_null_dir.mkdir(parents=True, exist_ok=True)

    autocorr_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    spectral_rows: list[dict[str, object]] = []
    started = time.time()

    print(
        f"[{chunk_id}] node-label null permutations {perm_start}:{perm_stop} "
        f"({len(perm_indices)} permutations) across {len(domain_table)} domains.",
        flush=True,
    )

    iterator = domain_table.itertuples(index=False)
    for row in tqdm(iterator, total=len(domain_table), desc=f"{chunk_id} domains"):
        dataset = str(getattr(row, "dataset"))
        file_name = str(getattr(row, "file"))
        try:
            domain_df = _load_domain_dataframe(file_name)
            graph, observed_fitness = _build_hamming_graph(domain_df)
            u, v = _edge_arrays(graph)
            incidence = _incidence_lists(u, v)
        except Exception as exc:
            for perm_idx in perm_indices:
                base = {
                    "chunk_id": chunk_id,
                    "perm_idx": int(perm_idx),
                    "dataset": dataset,
                    "file": file_name,
                    "domain_perm_seed": _seed_for(base_seed, perm_idx, file_name),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                autocorr_rows.append(base.copy())
                spectral_rows.append(base.copy())
            continue

        for perm_idx in perm_indices:
            seed = _seed_for(base_seed, perm_idx, file_name)
            rng = np.random.default_rng(seed)
            permuted_fitness = rng.permutation(observed_fitness)
            edge_energy = _edge_energy(permuted_fitness, u, v)
            log_energy = np.log1p(edge_energy)
            moments = _row_standardized_line_graph_moments(log_energy, incidence)
            base = {
                "chunk_id": chunk_id,
                "perm_idx": int(perm_idx),
                "dataset": dataset,
                "file": file_name,
                "domain_perm_seed": int(seed),
                "status": "ok",
                "error": "",
            }

            autocorr = base.copy()
            autocorr.update(moments)
            autocorr["energy_transform"] = "node-label permutation; log1p(edge_energy)"
            autocorr["one_minus_geary_c"] = (
                float(1.0 - moments["geary_c"]) if np.isfinite(moments["geary_c"]) else np.nan
            )
            autocorr_rows.append(autocorr)

            for quantile in QUANTILES:
                qrow = base.copy()
                qrow.update(_quantile_neighbor_stats(edge_energy, incidence, quantile))
                quantile_rows.append(qrow)

            spectral = base.copy()
            try:
                spectral.update(_spectral_summary(graph, permuted_fitness))
            except Exception as exc:
                spectral.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            spectral_rows.append(spectral)

        del graph
        gc.collect()

    prefix = f"{chunk_id}_node_label"
    autocorr_df = pd.DataFrame(autocorr_rows)
    quantile_df = pd.DataFrame(quantile_rows)
    spectral_df = pd.DataFrame(spectral_rows)

    _write_table(
        autocorr_df,
        processed_null_dir / f"{prefix}_autocorrelation_domain_null.csv",
        output_null_dir,
    )
    _write_table(
        quantile_df,
        processed_null_dir / f"{prefix}_autocorrelation_quantile_null.csv",
        output_null_dir,
    )
    _write_table(
        spectral_df,
        processed_null_dir / f"{prefix}_spectral_bipartition_null.csv",
        output_null_dir,
    )

    manifest = {
        "chunk_id": chunk_id,
        "mode": "node_label_permutation_null",
        "node_label_permutation": True,
        "perm_start": perm_start,
        "perm_stop": perm_stop,
        "n_permutations": len(perm_indices),
        "base_seed": base_seed,
        "n_domains": int(len(domain_table)),
        "quantiles": list(QUANTILES),
        "runtime_seconds": time.time() - started,
        "outputs": {
            "autocorrelation_domain_null": str(processed_null_dir / f"{prefix}_autocorrelation_domain_null.csv"),
            "autocorrelation_quantile_null": str(processed_null_dir / f"{prefix}_autocorrelation_quantile_null.csv"),
            "spectral_bipartition_null": str(processed_null_dir / f"{prefix}_spectral_bipartition_null.csv"),
        },
    }
    manifest_path = processed_null_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    shutil.copy2(manifest_path, output_null_dir / manifest_path.name)
    print(json.dumps(_jsonable(manifest), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
