from __future__ import annotations

import binascii
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon


SV_TOP_K = 3
SV_TOP_LOADING_QUANTILE = 0.75
SV_NULL_REPS = 300
SV_MIN_ROWS_FOR_SVD = 8
SV_MIN_MEASURED_PER_ROW = 5
SV_RANDOM_SEED = 2026

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
MUT_TOKEN_RE = re.compile(r"([A-Z])([0-9]+)([A-Z])")
DISTANCE_BINS = np.array([0.0, 8.0, 12.0, 16.0, 20.0, 28.0, np.inf])

PDB_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "README.md").exists() and (path / "experiments").is_dir() and (path / "analyses").is_dir():
            return path
    raise FileNotFoundError("Could not locate repository root")


def build_dms_tmap_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        file_name = result["file"]
        rows.append(
            {
                "dataset": "-".join(file_name.replace(".csv", "").split("_")[0:2]),
                "file": file_name,
                "t_map": float(result["tmap"]["t_map"]),
            }
        )
    table = pd.DataFrame(rows)
    return table.replace([np.inf, -np.inf], np.nan).dropna(subset=["t_map"]).reset_index(drop=True)


def parse_single_mutant_token(mutant: str):
    tokens = [tok.strip() for tok in str(mutant).split(":") if tok.strip()]
    if len(tokens) != 1:
        return None
    match = MUT_TOKEN_RE.fullmatch(tokens[0])
    if match is None:
        return None
    wt_aa, pos_str, mut_aa = match.groups()
    return wt_aa, int(pos_str), mut_aa


def extract_single_mutation_table(domain_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in domain_df.itertuples(index=False):
        parsed = parse_single_mutant_token(getattr(row, "mutant", ""))
        if parsed is None:
            continue
        wt_aa, position, mut_aa = parsed
        if mut_aa not in AA20:
            continue
        score = float(getattr(row, "DMS_score"))
        if not np.isfinite(score):
            continue
        rows.append(
            {
                "position": int(position),
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "DMS_score": score,
            }
        )
    if len(rows) == 0:
        return pd.DataFrame(columns=["position", "wt_aa", "mut_aa", "DMS_score"])
    return (
        pd.DataFrame(rows)
        .groupby(["position", "wt_aa", "mut_aa"], as_index=False)["DMS_score"]
        .mean()
    )


def infer_wildtype_sequence(domain_df: pd.DataFrame) -> str:
    wt_seq = list(str(domain_df["mutated_sequence"].iloc[0]))
    for mutant in domain_df["mutant"].astype(str):
        for token in mutant.split(":"):
            match = MUT_TOKEN_RE.fullmatch(token.strip())
            if match is None:
                continue
            wt_aa, pos_str, _ = match.groups()
            idx = int(pos_str) - 1
            if 0 <= idx < len(wt_seq):
                wt_seq[idx] = wt_aa
    return "".join(wt_seq)


def extract_pdb_id_from_file(file_name: str) -> str:
    stem = Path(str(file_name)).stem
    token = stem.split("_")[-1].strip().upper()
    match = re.search(r"([0-9][A-Z0-9]{3})$", token)
    if match:
        return match.group(1)
    if len(token) >= 4:
        return token[-4:]
    raise ValueError(f"Cannot parse PDB id from file name: {file_name}")


def read_pdb_text(pdb_id: str, pdb_cache_dir: Path) -> str:
    pdb_id = str(pdb_id).upper()
    candidates = [
        pdb_cache_dir / f"{pdb_id}.pdb",
        pdb_cache_dir / f"{pdb_id.lower()}.pdb",
        pdb_cache_dir / f"pdb{pdb_id.lower()}.ent",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(f"PDB cache missing for {pdb_id}")


def extract_main_chain_ensemble(pdb_text: str):
    models = defaultdict(lambda: defaultdict(dict))
    model_id = 1

    for line in str(pdb_text).splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            model_raw = line[10:14].strip()
            try:
                model_id = int(model_raw)
            except Exception:
                model_id = 1
            continue
        if record != "ATOM":
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        altloc = line[16].strip()
        if altloc not in ("", "A", "1"):
            continue
        chain = line[21].strip() or "A"
        resname = line[17:20].strip().upper()
        try:
            resseq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except Exception:
            continue
        icode = line[26].strip()
        key = (resseq, icode, resname)
        if key not in models[model_id][chain]:
            models[model_id][chain][key] = (x, y, z)

    if len(models) == 0:
        raise ValueError("No CA atoms parsed from PDB text")

    model_ids = sorted(models.keys())
    first_model = model_ids[0]
    chain_sizes = {ch: len(res_map) for ch, res_map in models[first_model].items()}
    main_chain = sorted(chain_sizes.items(), key=lambda item: (-item[1], item[0]))[0][0]
    first_keys = list(models[first_model][main_chain].keys())
    common_keys = [
        key
        for key in first_keys
        if all(main_chain in models[mid] and key in models[mid][main_chain] for mid in model_ids)
    ]
    if len(common_keys) == 0:
        raise ValueError("No common CA residues across ensemble models")

    coords_models = []
    valid_model_ids = []
    for mid in model_ids:
        if main_chain not in models[mid]:
            continue
        if not all(key in models[mid][main_chain] for key in common_keys):
            continue
        coords_models.append(np.array([models[mid][main_chain][key] for key in common_keys], dtype=float))
        valid_model_ids.append(mid)

    seq_chars = [PDB_AA3_TO_1.get(resname, "X") for _, _, resname in common_keys]
    return coords_models, valid_model_ids, str(main_chain), "".join(seq_chars)


def align_dms_to_structure_positions(wt_seq: str, structure_seq: str):
    a = str(wt_seq)
    b = str(structure_seq)
    if len(a) == 0 or len(b) == 0:
        return {}, np.nan

    match_score = 2.0
    mismatch_score = -1.0
    gap_score = -2.0
    n, m = len(a), len(b)
    score = np.zeros((n + 1, m + 1), dtype=float)
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] + gap_score
        ptr[i, 0] = 2
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] + gap_score
        ptr[0, j] = 3

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (match_score if a[i - 1] == b[j - 1] else mismatch_score)
            up = score[i - 1, j] + gap_score
            left = score[i, j - 1] + gap_score
            best = diag
            move = 1
            if up > best:
                best = up
                move = 2
            if left > best:
                best = left
                move = 3
            score[i, j] = best
            ptr[i, j] = move

    i, j = n, m
    aln_a = []
    aln_b = []
    while i > 0 or j > 0:
        move = ptr[i, j] if i >= 0 and j >= 0 else 0
        if i > 0 and j > 0 and move == 1:
            aln_a.append(a[i - 1])
            aln_b.append(b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 2):
            aln_a.append(a[i - 1])
            aln_b.append("-")
            i -= 1
        else:
            aln_a.append("-")
            aln_b.append(b[j - 1])
            j -= 1

    mapping = {}
    i_dms = -1
    i_struct = -1
    n_match = 0
    n_pairs = 0
    for ca, cb in zip(aln_a[::-1], aln_b[::-1]):
        if ca != "-":
            i_dms += 1
        if cb != "-":
            i_struct += 1
        if ca != "-" and cb != "-":
            mapping[int(i_dms + 1)] = int(i_struct)
            n_pairs += 1
            if ca == cb:
                n_match += 1
    return mapping, float(n_match / n_pairs) if n_pairs > 0 else np.nan


def mean_pairwise_ca_distance(coords: np.ndarray, indices: list[int]) -> float:
    idx = np.asarray(sorted(set(indices)), dtype=int)
    if len(idx) < 2:
        return np.nan
    dist = squareform(pdist(coords[idx]))
    tri = np.triu_indices(len(idx), k=1)
    return float(np.mean(dist[tri]))


def build_svd_model(domain_df: pd.DataFrame):
    mut_df = extract_single_mutation_table(domain_df)
    if len(mut_df) == 0:
        raise ValueError("No usable single-mutation records")

    submat = (
        mut_df.pivot_table(index="position", columns="mut_aa", values="DMS_score", aggfunc="mean")
        .reindex(columns=AA20)
        .sort_index()
    ).copy(deep=True)
    wt_by_pos = mut_df.groupby("position")["wt_aa"].agg(lambda s: s.value_counts().idxmax())
    col_to_idx = {aa: i for i, aa in enumerate(submat.columns)}
    arr_all = np.array(submat.to_numpy(dtype=float, copy=True), dtype=float, copy=True)
    for r_i, pos in enumerate(submat.index.to_numpy(dtype=int)):
        wt_aa = wt_by_pos.get(int(pos), None)
        c_i = col_to_idx.get(wt_aa, None)
        if c_i is not None:
            arr_all[r_i, c_i] = np.nan

    measured_per_row = np.isfinite(arr_all).sum(axis=1)
    keep_mask = measured_per_row >= SV_MIN_MEASURED_PER_ROW
    positions = submat.index.to_numpy(dtype=int)[keep_mask]
    arr = np.array(arr_all[keep_mask], dtype=float, copy=True)
    if arr.shape[0] < SV_MIN_ROWS_FOR_SVD:
        raise ValueError(f"Insufficient rows after filtering: {arr.shape[0]}")

    row_means = np.nanmean(arr, axis=1)
    nan_mask = ~np.isfinite(arr)
    if nan_mask.any():
        arr[nan_mask] = row_means[np.where(nan_mask)[0]]
    arr = arr - np.mean(arr, axis=1, keepdims=True)
    if not np.isfinite(arr).all() or np.allclose(arr, 0.0):
        raise ValueError("Centered substitution matrix is degenerate")

    u, s, vt = np.linalg.svd(arr, full_matrices=False)
    return positions, u, s, vt


def distance_bin_label(left: float, right: float) -> str:
    if np.isinf(right):
        return f">{left:.0f}"
    return f"{left:.0f}-{right:.0f}"


def summarize_domain(row, dms_dir: Path, pdb_cache_dir: Path):
    file_name = row.file
    dataset = row.dataset
    domain_df = pd.read_csv(dms_dir / file_name)
    domain_df = domain_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mutated_sequence", "DMS_score"]
    )

    positions, u, singular_values, _ = build_svd_model(domain_df)
    pdb_id = extract_pdb_id_from_file(file_name)
    pdb_text = read_pdb_text(pdb_id, pdb_cache_dir)
    coords_models, model_ids, main_chain, structure_seq = extract_main_chain_ensemble(pdb_text)
    coords = np.asarray(coords_models[0], dtype=float)
    wt_seq = infer_wildtype_sequence(domain_df)
    dms_to_struct, seq_identity = align_dms_to_structure_positions(wt_seq, structure_seq)

    mapped_rows = []
    mapped_struct = []
    for arr_idx, pos in enumerate(positions.astype(int)):
        struct_idx = dms_to_struct.get(int(pos))
        if struct_idx is None:
            continue
        mapped_rows.append(int(arr_idx))
        mapped_struct.append(int(struct_idx))

    if len(mapped_rows) < SV_MIN_ROWS_FOR_SVD:
        raise ValueError("Too few mapped positions for structural clustering")

    mapped_rows_arr = np.asarray(mapped_rows, dtype=int)
    mapped_struct_arr = np.asarray(mapped_struct, dtype=int)
    mapped_pool = np.asarray(sorted(set(mapped_struct)), dtype=int)

    k_eff = int(min(SV_TOP_K, u.shape[1]))
    if k_eff < 1:
        raise ValueError("No usable singular vectors")

    seed = SV_RANDOM_SEED + int(binascii.crc32(str(file_name).encode("utf-8")) % 1_000_000)
    rng = np.random.default_rng(seed)

    axis_rows = []
    for axis_idx in range(k_eff):
        abs_load = np.abs(u[:, axis_idx])
        n_select = int(max(3, np.ceil((1.0 - SV_TOP_LOADING_QUANTILE) * len(abs_load))))
        top_idx = np.argsort(abs_load)[::-1][:n_select]
        top_struct = sorted(
            {
                int(dms_to_struct[int(positions[i])])
                for i in top_idx
                if int(positions[i]) in dms_to_struct
            }
        )
        axis_row = {
            "dataset": dataset,
            "file": file_name,
            "pdb_id": pdb_id,
            "axis": int(axis_idx + 1),
            "singular_value": float(singular_values[axis_idx]),
            "explained_power_fraction": float((singular_values[axis_idx] ** 2) / np.sum(singular_values ** 2)),
            "n_positions_selected": int(len(top_struct)),
            "observed_mean_pair_dist": np.nan,
            "null_mean_pair_dist": np.nan,
            "null_std_pair_dist": np.nan,
            "z_vs_null": np.nan,
            "p_cluster_one_sided": np.nan,
            "cluster_effect_ratio_obs_over_null": np.nan,
        }
        if len(top_struct) >= 3 and len(mapped_pool) >= len(top_struct):
            obs = mean_pairwise_ca_distance(coords, top_struct)
            null_vals = []
            for _ in range(SV_NULL_REPS):
                rand_idx = rng.choice(mapped_pool, size=len(top_struct), replace=False)
                null_vals.append(mean_pairwise_ca_distance(coords, rand_idx))
            null_arr = np.asarray([val for val in null_vals if np.isfinite(val)], dtype=float)
            if len(null_arr) > 0 and np.isfinite(obs):
                null_mean = float(np.mean(null_arr))
                null_std = float(np.std(null_arr, ddof=1)) if len(null_arr) > 1 else np.nan
                axis_row.update(
                    {
                        "observed_mean_pair_dist": float(obs),
                        "null_mean_pair_dist": null_mean,
                        "null_std_pair_dist": null_std,
                        "z_vs_null": float((obs - null_mean) / (null_std + 1e-12))
                        if np.isfinite(null_std)
                        else np.nan,
                        "p_cluster_one_sided": float((np.sum(null_arr <= obs) + 1) / (len(null_arr) + 1)),
                        "cluster_effect_ratio_obs_over_null": float(obs / null_mean)
                        if null_mean != 0
                        else np.nan,
                    }
                )
        axis_rows.append(axis_row)

    dominant_axis = np.argmax(np.abs(u[mapped_rows_arr, :k_eff]), axis=1) + 1
    pair_dist = squareform(pdist(coords[mapped_struct_arr]))
    tri = np.triu_indices(len(mapped_struct_arr), k=1)
    pair_dist_vals = pair_dist[tri]
    same_axis_vals = (dominant_axis[tri[0]] == dominant_axis[tri[1]]).astype(float)

    bin_rows = []
    null_bin_rows = []
    for bin_idx, (left, right) in enumerate(zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:])):
        if np.isinf(right):
            mask = pair_dist_vals >= left
        else:
            mask = (pair_dist_vals >= left) & (pair_dist_vals < right)
        n_pairs = int(np.sum(mask))
        obs_frac = float(np.mean(same_axis_vals[mask])) if n_pairs > 0 else np.nan
        bin_rows.append(
            {
                "dataset": dataset,
                "file": file_name,
                "pdb_id": pdb_id,
                "bin_index": int(bin_idx),
                "distance_bin": distance_bin_label(left, right),
                "distance_left_angstrom": float(left),
                "distance_right_angstrom": float(right),
                "n_pairs": n_pairs,
                "same_axis_fraction": obs_frac,
            }
        )
        if n_pairs > 0:
            for rep_idx in range(SV_NULL_REPS):
                shuffled = rng.permutation(dominant_axis)
                null_same = (shuffled[tri[0]] == shuffled[tri[1]]).astype(float)
                null_bin_rows.append(
                    {
                        "dataset": dataset,
                        "file": file_name,
                        "bin_index": int(bin_idx),
                        "distance_bin": distance_bin_label(left, right),
                        "null_rep": int(rep_idx),
                        "same_axis_fraction": float(np.mean(null_same[mask])),
                    }
                )

    valid_axes = pd.DataFrame(axis_rows).dropna(subset=["p_cluster_one_sided"])
    domain_row = {
        "dataset": dataset,
        "file": file_name,
        "t_map": float(row.t_map),
        "status": "ok",
        "error": "",
        "pdb_id": pdb_id,
        "sequence_alignment_identity": float(seq_identity),
        "n_positions_submat": int(len(positions)),
        "n_positions_mapped": int(len(mapped_rows_arr)),
        "n_axes_tested": int(len(valid_axes)),
        "n_axes_clustered_p05": int((valid_axes["p_cluster_one_sided"] < 0.05).sum())
        if len(valid_axes)
        else 0,
        "mean_axis_cluster_z": float(valid_axes["z_vs_null"].mean()) if len(valid_axes) else np.nan,
        "median_axis_cluster_p": float(valid_axes["p_cluster_one_sided"].median()) if len(valid_axes) else np.nan,
        "median_cluster_effect_ratio_obs_over_null": float(
            valid_axes["cluster_effect_ratio_obs_over_null"].median()
        )
        if len(valid_axes)
        else np.nan,
    }
    return domain_row, axis_rows, bin_rows, null_bin_rows


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)


def make_figure(axis_df: pd.DataFrame, bin_df: pd.DataFrame, null_bin_df: pd.DataFrame, out_pdf: Path, out_png: Path):
    plot_axis = axis_df.dropna(subset=["observed_mean_pair_dist", "null_mean_pair_dist"]).copy()
    plot_axis["distance_ratio"] = (
        plot_axis["observed_mean_pair_dist"].to_numpy(dtype=float)
        / plot_axis["null_mean_pair_dist"].to_numpy(dtype=float)
    )
    axis_colors = {1: "#4C78A8", 2: "#59A14F", 3: "#E15759"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    for axis_id, sub in plot_axis.groupby("axis"):
        ax.scatter(
            sub["null_mean_pair_dist"],
            sub["observed_mean_pair_dist"],
            s=18,
            alpha=0.78,
            color=axis_colors.get(int(axis_id), "black"),
            label=f"Axis {int(axis_id)}",
            linewidth=0,
        )
    lim_min = float(np.nanmin([plot_axis["observed_mean_pair_dist"].min(), plot_axis["null_mean_pair_dist"].min()]))
    lim_max = float(np.nanmax([plot_axis["observed_mean_pair_dist"].max(), plot_axis["null_mean_pair_dist"].max()]))
    pad = 0.05 * (lim_max - lim_min)
    ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], color="#555555", lw=0.9)
    ax.set_xlim(lim_min - pad, lim_max + pad)
    ax.set_ylim(lim_min - pad, lim_max + pad)
    ax.set_xlabel("Random-set mean C-alpha distance (A)", fontsize=9)
    ax.set_ylabel("SVD-axis mean C-alpha distance (A)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(0.02, 0.98, "a", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    style_axes(ax)

    ax = axes[1]
    ratio_groups = [
        plot_axis.loc[plot_axis["axis"].eq(axis_id), "distance_ratio"].dropna().to_numpy(dtype=float)
        for axis_id in [1, 2, 3]
    ]
    ax.boxplot(
        ratio_groups,
        positions=[1, 2, 3],
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "#E6E6E6", "edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    rng = np.random.default_rng(123)
    for axis_id in [1, 2, 3]:
        vals = plot_axis.loc[plot_axis["axis"].eq(axis_id), "distance_ratio"].dropna().to_numpy(dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), axis_id) + jitter,
            vals,
            s=14,
            alpha=0.55,
            color=axis_colors[axis_id],
            linewidth=0,
        )
    ax.axhline(1.0, color="#555555", lw=0.9, linestyle="--")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Axis 1", "Axis 2", "Axis 3"])
    ax.set_xlabel("SVD axis", fontsize=9)
    ax.set_ylabel("Observed / random mean distance", fontsize=9)
    y_max = float(np.nanmax(plot_axis["distance_ratio"].to_numpy(dtype=float)))
    ax.set_ylim(0.45, max(1.25, y_max * 1.08))
    ax.text(0.02, 0.98, "b", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    style_axes(ax)

    fig.tight_layout(w_pad=1.8)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    notebook_dir = Path(__file__).resolve().parent
    analysis_dir = notebook_dir.parent
    project_root = find_project_root(analysis_dir)
    dms_dir = project_root / "data" / "source_datasets" / "megascale_folding"
    pdb_cache_dir = project_root / "data" / "source_datasets" / "pdb_cache"
    processed_pickle = project_root / "data" / "processed" / "stability_dms" / "megascale_folding_tmap.pkl"

    with processed_pickle.open("rb") as handle:
        results = pickle.load(handle)
    tmap_df = build_dms_tmap_table(results)

    domain_rows = []
    axis_rows = []
    bin_rows = []
    null_bin_rows = []
    for row in tmap_df.itertuples(index=False):
        try:
            domain_row, domain_axis_rows, domain_bin_rows, domain_null_rows = summarize_domain(
                row, dms_dir, pdb_cache_dir
            )
        except Exception as exc:
            domain_row = {
                "dataset": row.dataset,
                "file": row.file,
                "t_map": float(row.t_map),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            domain_axis_rows = []
            domain_bin_rows = []
            domain_null_rows = []
        domain_rows.append(domain_row)
        axis_rows.extend(domain_axis_rows)
        bin_rows.extend(domain_bin_rows)
        null_bin_rows.extend(domain_null_rows)

    outdir = analysis_dir / "figures" / "SI_figures" / "SI_figure_DMS"
    outdir.mkdir(parents=True, exist_ok=True)

    domain_df = pd.DataFrame(domain_rows)
    axis_df = pd.DataFrame(axis_rows)
    bin_df = pd.DataFrame(bin_rows)
    null_bin_df = pd.DataFrame(null_bin_rows)

    domain_csv = outdir / "left_sv_structural_clustering_domain_summary.csv"
    axis_csv = outdir / "left_sv_structural_clustering_axis_summary.csv"
    bin_csv = outdir / "left_sv_axis_distance_binned_same_axis.csv"
    null_csv = outdir / "left_sv_axis_distance_binned_same_axis_null.csv"
    stats_csv = outdir / "left_sv_structural_clustering_summary_stats.csv"

    domain_df.to_csv(domain_csv, index=False)
    axis_df.to_csv(axis_csv, index=False)
    bin_df.to_csv(bin_csv, index=False)
    null_bin_df.to_csv(null_csv, index=False)

    valid_axis = axis_df.dropna(subset=["cluster_effect_ratio_obs_over_null"]).copy()
    valid_domain = domain_df.loc[domain_df["status"].eq("ok")].copy()
    ratio = valid_axis["cluster_effect_ratio_obs_over_null"].to_numpy(dtype=float)
    p_cluster = valid_axis["p_cluster_one_sided"].to_numpy(dtype=float)
    try:
        wilcox_stat, wilcox_p = wilcoxon(ratio - 1.0, alternative="less")
    except Exception:
        wilcox_stat, wilcox_p = np.nan, np.nan
    stats_df = pd.DataFrame(
        [
            {
                "n_domains_ok": int(len(valid_domain)),
                "n_axes_tested": int(len(valid_axis)),
                "median_obs_over_null_distance_ratio": float(np.nanmedian(ratio)),
                "n_axes_clustered_p05": int(np.sum(p_cluster < 0.05)),
                "fraction_axes_clustered_p05": float(np.mean(p_cluster < 0.05)),
                "wilcoxon_ratio_less_than_one_stat": float(wilcox_stat) if np.isfinite(wilcox_stat) else np.nan,
                "wilcoxon_ratio_less_than_one_p": float(wilcox_p) if np.isfinite(wilcox_p) else np.nan,
            }
        ]
    )
    stats_df.to_csv(stats_csv, index=False)

    out_pdf = outdir / "left_sv_structural_clustering_summary.pdf"
    out_png = outdir / "left_sv_structural_clustering_summary.png"
    make_figure(axis_df, bin_df, null_bin_df, out_pdf, out_png)

    print(f"Domains processed: {len(domain_df)}")
    print(f"Successful domains: {int(domain_df['status'].eq('ok').sum())}")
    print(stats_df.to_string(index=False))
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {axis_csv}")


if __name__ == "__main__":
    main()
