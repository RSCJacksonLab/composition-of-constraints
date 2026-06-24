from pathlib import Path
from itertools import combinations
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare, fisher_exact, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from Bio.PDB import PDBParser, ShrakeRupley


STRUCT_LIMITER_TOP_K = 3
STRUCT_LIMITER_TOP_LOADING_QUANTILE = 0.75
STRUCT_LIMITER_MIN_TOP_POSITIONS = 3
STRUCT_LIMITER_ALPHA = 0.05
STRUCT_LIMITER_BIN_COUNT = 10
STRUCT_LIMITER_HYDROPHOBIC_AA = set("AFILMVWY")
AA_HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
AA_SIDECHAIN_VOLUME = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}
AA_MAX_ASA = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 225.0,
    "E": 223.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}


def _safe_mannwhitneyu(x, y):
    x = pd.Series(x, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    y = pd.Series(y, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, int(len(x)), int(len(y))

    try:
        stat, pval = mannwhitneyu(
            x.to_numpy(dtype=float),
            y.to_numpy(dtype=float),
            alternative="two-sided",
        )
    except ValueError:
        return np.nan, np.nan, int(len(x)), int(len(y))

    return float(stat), float(pval), int(len(x)), int(len(y))


def _safe_spearmanr(x, y):
    paired = pd.DataFrame({"x": x, "y": y}, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < 3:
        return np.nan, np.nan, int(len(paired))

    rho, pval = spearmanr(
        paired["x"].to_numpy(dtype=float),
        paired["y"].to_numpy(dtype=float),
    )
    return float(rho), float(pval), int(len(paired))


def _safe_chisquare(obs_cross, obs_same, expected_cross_frac):
    n_total = int(obs_cross + obs_same)
    if n_total < 1 or not np.isfinite(expected_cross_frac):
        return np.nan, np.nan, np.nan, np.nan

    expected_cross = float(n_total * expected_cross_frac)
    expected_same = float(n_total - expected_cross)
    if expected_cross <= 0.0 or expected_same <= 0.0:
        return np.nan, np.nan, expected_cross, expected_same

    stat, pval = chisquare(
        f_obs=np.asarray([obs_cross, obs_same], dtype=float),
        f_exp=np.asarray([expected_cross, expected_same], dtype=float),
    )
    return float(stat), float(pval), expected_cross, expected_same


def _safe_qcut_codes(values, q):
    ser = pd.Series(values, dtype=float)
    out = pd.Series(np.nan, index=ser.index, dtype=float)
    valid = ser.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2:
        return out
    try:
        out.loc[valid.index] = pd.qcut(valid, q=min(int(q), len(valid)), labels=False, duplicates="drop")
    except ValueError:
        return out
    return out


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


def _load_csv_or_empty(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _single_hamming_difference(seq_a, seq_b):
    seq_a = str(seq_a)
    seq_b = str(seq_b)
    if len(seq_a) != len(seq_b):
        raise ValueError(f"Sequence length mismatch: {len(seq_a)} vs {len(seq_b)}.")

    diff = [(idx, aa_a, aa_b) for idx, (aa_a, aa_b) in enumerate(zip(seq_a, seq_b), start=1) if aa_a != aa_b]
    if len(diff) != 1:
        return np.nan, "", "", int(len(diff))

    pos, aa_a, aa_b = diff[0]
    return int(pos), str(aa_a), str(aa_b), 1


def _compute_residue_sasa_table(prepared):
    residue_df = prepared["residue_df"].copy().reset_index(drop=True)
    out = pd.DataFrame(
        {
            "struct_idx": np.arange(len(residue_df), dtype=int),
            "residue_sasa": np.nan,
            "relative_sasa": np.nan,
        }
    )
    status = "unavailable"

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", StringIO(str(prepared["pdb_text"])))
        model = next(structure.get_models())
        sr = ShrakeRupley()
        sr.compute(model, level="R")

        chain_id = str(prepared["main_chain"])
        chain = model[chain_id] if chain_id in model else next(model.get_chains())
        sasa_map = {}
        for residue in chain.get_residues():
            hetflag, resseq, icode = residue.id
            if str(hetflag).strip() not in ("", " "):
                continue
            resname = str(residue.resname).strip().upper()
            key = (int(resseq), str(icode).strip(), resname)
            sasa_map[key] = float(getattr(residue, "sasa", np.nan))

        sasa_vals = []
        rel_vals = []
        for row in residue_df.itertuples(index=False):
            key = (int(row.resseq), str(row.icode), str(row.resname))
            sasa = float(sasa_map.get(key, np.nan))
            max_asa = float(AA_MAX_ASA.get(str(getattr(row, "aa", "")), np.nan))
            rel_sasa = float(sasa / max_asa) if np.isfinite(sasa) and np.isfinite(max_asa) and max_asa > 0 else np.nan
            sasa_vals.append(sasa)
            rel_vals.append(rel_sasa)

        out["residue_sasa"] = np.asarray(sasa_vals, dtype=float)
        out["relative_sasa"] = np.asarray(rel_vals, dtype=float)
        status = "ok"
    except Exception as exc:
        status = f"{type(exc).__name__}: {exc}"

    return out, status


def _classify_core_surface(position_df):
    position_df = position_df.copy()
    mapped_mask = position_df["is_dms_mapped"].to_numpy(dtype=bool)

    exposure_ref = position_df.loc[mapped_mask, "exposure_knn_distance"].to_numpy(dtype=float)
    packing_ref = position_df.loc[mapped_mask, "packing_contact_density"].to_numpy(dtype=float)
    exposure_median = float(np.nanmedian(exposure_ref)) if np.isfinite(exposure_ref).any() else np.nan
    packing_median = float(np.nanmedian(packing_ref)) if np.isfinite(packing_ref).any() else np.nan

    position_df["exposure_median_domain"] = exposure_median
    position_df["packing_median_domain"] = packing_median
    position_df["abs_exposure_dev_from_median"] = np.abs(
        position_df["exposure_knn_distance"].to_numpy(dtype=float) - exposure_median
    )
    position_df["abs_packing_dev_from_median"] = np.abs(
        position_df["packing_contact_density"].to_numpy(dtype=float) - packing_median
    )

    exposure_label = np.full(len(position_df), "", dtype=object)
    packing_label = np.full(len(position_df), "", dtype=object)

    exposure_vals = position_df["exposure_knn_distance"].to_numpy(dtype=float)
    packing_vals = position_df["packing_contact_density"].to_numpy(dtype=float)
    exposure_ok = np.isfinite(exposure_vals) & np.isfinite(exposure_median)
    packing_ok = np.isfinite(packing_vals) & np.isfinite(packing_median)

    exposure_label[exposure_ok] = np.where(exposure_vals[exposure_ok] <= exposure_median, "core", "surface")
    packing_label[packing_ok] = np.where(packing_vals[packing_ok] >= packing_median, "core", "surface")

    consensus = np.full(len(position_df), "", dtype=object)
    both_ok = exposure_ok & packing_ok
    consensus[both_ok] = np.where(
        exposure_label[both_ok] == packing_label[both_ok],
        exposure_label[both_ok],
        "mixed",
    )

    position_df["core_surface_label_exposure"] = exposure_label
    position_df["core_surface_label_packing"] = packing_label
    position_df["core_surface_label_consensus"] = consensus

    if "relative_sasa" in position_df.columns:
        rel_ref = position_df.loc[mapped_mask, "relative_sasa"].to_numpy(dtype=float)
        rel_median = float(np.nanmedian(rel_ref)) if np.isfinite(rel_ref).any() else np.nan
        position_df["relative_sasa_median_domain"] = rel_median
        position_df["abs_relative_sasa_dev_from_median"] = np.abs(
            position_df["relative_sasa"].to_numpy(dtype=float) - rel_median
        )
        rel_label = np.full(len(position_df), "", dtype=object)
        rel_vals = position_df["relative_sasa"].to_numpy(dtype=float)
        rel_ok = np.isfinite(rel_vals) & np.isfinite(rel_median)
        rel_label[rel_ok] = np.where(rel_vals[rel_ok] <= rel_median, "core", "surface")
        position_df["core_surface_label_relative_sasa"] = rel_label
        thresh_label = np.full(len(position_df), "", dtype=object)
        thresh_ok = np.isfinite(rel_vals)
        thresh_label[thresh_ok] = np.where(rel_vals[thresh_ok] <= 0.20, "core", "surface")
        position_df["core_surface_label_relative_sasa_020"] = thresh_label
    else:
        position_df["relative_sasa_median_domain"] = np.nan
        position_df["abs_relative_sasa_dev_from_median"] = np.nan
        position_df["core_surface_label_relative_sasa"] = ""
        position_df["core_surface_label_relative_sasa_020"] = ""

    position_df["exposure_decile"] = _safe_qcut_codes(
        position_df["exposure_knn_distance"].where(position_df["is_dms_mapped"]),
        STRUCT_LIMITER_BIN_COUNT,
    )
    if "relative_sasa" in position_df.columns:
        position_df["relative_sasa_decile"] = _safe_qcut_codes(
            position_df["relative_sasa"].where(position_df["is_dms_mapped"]),
            STRUCT_LIMITER_BIN_COUNT,
        )
    else:
        position_df["relative_sasa_decile"] = np.nan

    return position_df, exposure_median, packing_median


def _select_top_loading_rows(frame, loading_col, *, quantile, min_positions):
    work = frame.loc[np.isfinite(frame[loading_col].to_numpy(dtype=float))].copy()
    if len(work) == 0:
        return work

    n_select = int(max(min_positions, np.ceil((1.0 - float(quantile)) * len(work))))
    n_select = int(min(len(work), n_select))
    order = np.argsort(work[loading_col].to_numpy(dtype=float))[::-1][:n_select]
    return work.iloc[order].copy()


def _build_position_loading_table(prepared, n_axes):
    mapped_df = pd.DataFrame(
        {
            "dms_position": np.asarray(prepared["mapped_dms_positions"], dtype=int),
            "struct_idx": np.asarray(prepared["mapped_struct_idx"], dtype=int),
        }
    )
    for ax_idx in range(int(n_axes)):
        signed = np.asarray(prepared["U_map"][:, ax_idx], dtype=float)
        mapped_df[f"axis{ax_idx + 1}_loading_signed"] = signed
        mapped_df[f"axis{ax_idx + 1}_loading_abs"] = np.abs(signed)

    agg = {
        "dms_position": lambda s: ";".join(str(int(x)) for x in sorted(set(int(v) for v in s))),
    }
    for ax_idx in range(int(n_axes)):
        agg[f"axis{ax_idx + 1}_loading_signed"] = "mean"
        agg[f"axis{ax_idx + 1}_loading_abs"] = "mean"

    grouped = mapped_df.groupby("struct_idx", as_index=False).agg(agg)
    grouped["n_mapped_dms_positions"] = grouped["dms_position"].apply(
        lambda s: 0 if not isinstance(s, str) or len(s) == 0 else len(s.split(";"))
    )

    position_df = prepared["structure_predictor_df"].copy()
    residue_meta = prepared["residue_df"].copy().reset_index(drop=True)
    residue_meta = residue_meta.assign(struct_idx=np.arange(len(residue_meta), dtype=int))
    keep_cols = ["struct_idx", "aa"] if "aa" in residue_meta.columns else ["struct_idx"]
    position_df = position_df.merge(residue_meta[keep_cols], on="struct_idx", how="left")
    position_df = position_df.merge(grouped, on="struct_idx", how="left")
    position_df["is_dms_mapped"] = position_df["n_mapped_dms_positions"].fillna(0).astype(int).gt(0)

    abs_cols = [f"axis{ax_idx + 1}_loading_abs" for ax_idx in range(int(n_axes))]
    abs_matrix = position_df[abs_cols].to_numpy(dtype=float)
    dominant_axis = np.full(len(position_df), np.nan, dtype=float)
    has_any = np.isfinite(abs_matrix).any(axis=1)
    if np.any(has_any):
        argmax = np.argmax(np.where(np.isfinite(abs_matrix), abs_matrix, -np.inf), axis=1)
        dominant_axis[has_any] = argmax[has_any] + 1
    position_df["dominant_loading_axis"] = dominant_axis

    return position_df


def _build_axis_profiles(
    *,
    file_name,
    dataset_name,
    pdb_id,
    position_df,
    exposure_median,
    packing_median,
    n_axes,
    top_k,
    top_loading_quantile,
):
    axis_rows = []
    axis_top_frames = {}

    for axis in range(1, int(n_axes) + 1):
        loading_col = f"axis{axis}_loading_abs"
        dominant_mask = position_df["dominant_loading_axis"].to_numpy(dtype=float) == float(axis)
        dominant_df = position_df.loc[position_df["is_dms_mapped"].to_numpy(dtype=bool) & dominant_mask].copy()
        top_df = _select_top_loading_rows(
            dominant_df,
            loading_col,
            quantile=top_loading_quantile,
            min_positions=STRUCT_LIMITER_MIN_TOP_POSITIONS,
        )
        axis_top_frames[int(axis)] = top_df.copy()

        mean_exposure = float(top_df["exposure_knn_distance"].mean()) if len(top_df) else np.nan
        mean_packing = float(top_df["packing_contact_density"].mean()) if len(top_df) else np.nan
        mean_relative_sasa = float(top_df["relative_sasa"].mean()) if "relative_sasa" in top_df.columns and len(top_df) else np.nan
        axis_type_exposure = ""
        axis_type_packing = ""
        axis_type_relative_sasa = ""
        if np.isfinite(mean_exposure) and np.isfinite(exposure_median):
            axis_type_exposure = "core" if mean_exposure <= exposure_median else "surface"
        if np.isfinite(mean_packing) and np.isfinite(packing_median):
            axis_type_packing = "core" if mean_packing >= packing_median else "surface"
        rel_median = (
            float(top_df["relative_sasa_median_domain"].dropna().iloc[0])
            if "relative_sasa_median_domain" in top_df.columns and top_df["relative_sasa_median_domain"].notna().any()
            else np.nan
        )
        if np.isfinite(mean_relative_sasa) and np.isfinite(rel_median):
            axis_type_relative_sasa = "core" if mean_relative_sasa <= rel_median else "surface"

        row = {
            "file": file_name,
            "dataset": dataset_name,
            "pdb_id": pdb_id,
            "axis": int(axis),
            "is_leading_axis": bool(axis <= int(top_k)),
            "n_positions_dominant": int(len(dominant_df)),
            "n_positions_top_loading": int(len(top_df)),
            "top_mean_exposure_knn_distance": mean_exposure,
            "top_median_exposure_knn_distance": (
                float(top_df["exposure_knn_distance"].median()) if len(top_df) else np.nan
            ),
            "top_mean_packing_contact_density": mean_packing,
            "top_median_packing_contact_density": (
                float(top_df["packing_contact_density"].median()) if len(top_df) else np.nan
            ),
            "top_mean_relative_sasa": mean_relative_sasa,
            "top_median_relative_sasa": (
                float(top_df["relative_sasa"].median()) if "relative_sasa" in top_df.columns and len(top_df) else np.nan
            ),
            "axis_core_surface_type_exposure": axis_type_exposure,
            "axis_core_surface_type_packing": axis_type_packing,
            "axis_core_surface_type_relative_sasa": axis_type_relative_sasa,
            "frac_top_positions_core_by_exposure": (
                float(np.mean(top_df["core_surface_label_exposure"].to_numpy(dtype=object) == "core"))
                if len(top_df)
                else np.nan
            ),
            "frac_top_positions_core_by_packing": (
                float(np.mean(top_df["core_surface_label_packing"].to_numpy(dtype=object) == "core"))
                if len(top_df)
                else np.nan
            ),
            "frac_top_positions_core_by_relative_sasa": (
                float(np.mean(top_df["core_surface_label_relative_sasa"].to_numpy(dtype=object) == "core"))
                if "core_surface_label_relative_sasa" in top_df.columns and len(top_df)
                else np.nan
            ),
        }
        axis_rows.append(row)

    return pd.DataFrame(axis_rows), axis_top_frames


def _build_axis_pair_tests(
    *,
    file_name,
    dataset_name,
    pdb_id,
    axis_top_frames,
    top_k,
):
    pair_rows = []
    leading_axes = [axis for axis in sorted(axis_top_frames) if axis <= int(top_k)]
    for axis_a, axis_b in combinations(leading_axes, 2):
        df_a = axis_top_frames[int(axis_a)]
        df_b = axis_top_frames[int(axis_b)]
        exp_stat, exp_p, n_a, n_b = _safe_mannwhitneyu(
            df_a["exposure_knn_distance"],
            df_b["exposure_knn_distance"],
        )
        pack_stat, pack_p, _, _ = _safe_mannwhitneyu(
            df_a["packing_contact_density"],
            df_b["packing_contact_density"],
        )
        sasa_stat, sasa_p, _, _ = _safe_mannwhitneyu(
            df_a["relative_sasa"] if "relative_sasa" in df_a.columns else [],
            df_b["relative_sasa"] if "relative_sasa" in df_b.columns else [],
        )
        pair_rows.append(
            {
                "file": file_name,
                "dataset": dataset_name,
                "pdb_id": pdb_id,
                "axis_a": int(axis_a),
                "axis_b": int(axis_b),
                "n_a": int(n_a),
                "n_b": int(n_b),
                "mean_exposure_axis_a": float(df_a["exposure_knn_distance"].mean()) if len(df_a) else np.nan,
                "mean_exposure_axis_b": float(df_b["exposure_knn_distance"].mean()) if len(df_b) else np.nan,
                "mean_packing_axis_a": float(df_a["packing_contact_density"].mean()) if len(df_a) else np.nan,
                "mean_packing_axis_b": float(df_b["packing_contact_density"].mean()) if len(df_b) else np.nan,
                "mean_relative_sasa_axis_a": float(df_a["relative_sasa"].mean()) if "relative_sasa" in df_a.columns and len(df_a) else np.nan,
                "mean_relative_sasa_axis_b": float(df_b["relative_sasa"].mean()) if "relative_sasa" in df_b.columns and len(df_b) else np.nan,
                "exposure_mwu_stat": exp_stat,
                "exposure_mwu_p": exp_p,
                "packing_mwu_stat": pack_stat,
                "packing_mwu_p": pack_p,
                "relative_sasa_mwu_stat": sasa_stat,
                "relative_sasa_mwu_p": sasa_p,
            }
        )
    return pd.DataFrame(pair_rows)


def _expected_cross_axis_type_fraction(component_counts, axis_type_map):
    usable = {
        int(comp): float(count)
        for comp, count in component_counts.items()
        if axis_type_map.get(int(comp), "") in {"core", "surface"}
    }
    total = float(sum(usable.values()))
    if total <= 0.0:
        return np.nan

    probs = {comp: count / total for comp, count in usable.items()}
    same_component_mass = float(sum(p * p for p in probs.values()))
    distinct_component_mass = float(1.0 - same_component_mass)
    if distinct_component_mass <= 0.0:
        return np.nan

    cross_mass = 0.0
    comps = sorted(probs)
    for comp_a in comps:
        for comp_b in comps:
            if comp_a == comp_b:
                continue
            if axis_type_map.get(comp_a, "") == axis_type_map.get(comp_b, ""):
                continue
            cross_mass += probs[comp_a] * probs[comp_b]

    return float(cross_mass / distinct_component_mass)


def _expected_axis_gap(component_counts, axis_value_map):
    usable = {
        int(comp): float(count)
        for comp, count in component_counts.items()
        if np.isfinite(axis_value_map.get(int(comp), np.nan))
    }
    total = float(sum(usable.values()))
    if total <= 0.0:
        return np.nan

    probs = {comp: count / total for comp, count in usable.items()}
    same_component_mass = float(sum(p * p for p in probs.values()))
    distinct_component_mass = float(1.0 - same_component_mass)
    if distinct_component_mass <= 0.0:
        return np.nan

    gap_mass = 0.0
    for comp_a, prob_a in probs.items():
        for comp_b, prob_b in probs.items():
            if comp_a == comp_b:
                continue
            gap_mass += prob_a * prob_b * abs(float(axis_value_map[comp_a]) - float(axis_value_map[comp_b]))
    return float(gap_mass / distinct_component_mass)


def _mutational_property_record(aa_a, aa_b):
    aa_a = str(aa_a or "").upper()
    aa_b = str(aa_b or "").upper()
    hydro_a = float(AA_HYDROPHOBICITY.get(aa_a, np.nan))
    hydro_b = float(AA_HYDROPHOBICITY.get(aa_b, np.nan))
    vol_a = float(AA_SIDECHAIN_VOLUME.get(aa_a, np.nan))
    vol_b = float(AA_SIDECHAIN_VOLUME.get(aa_b, np.nan))
    return {
        "mut_abs_hydrophobicity_change": abs(hydro_b - hydro_a) if np.isfinite(hydro_a) and np.isfinite(hydro_b) else np.nan,
        "mut_abs_volume_change": abs(vol_b - vol_a) if np.isfinite(vol_a) and np.isfinite(vol_b) else np.nan,
        "mut_hydrophobic_class_flip": bool((aa_a in STRUCT_LIMITER_HYDROPHOBIC_AA) != (aa_b in STRUCT_LIMITER_HYDROPHOBIC_AA)),
    }


def _build_exposure_bin_rows(edge_df):
    rows = []
    work = edge_df.loc[
        edge_df["edge_has_limiter_labels"].to_numpy(dtype=bool)
        & edge_df["edge_position_mapped_to_structure"].to_numpy(dtype=bool)
        & np.isfinite(edge_df["edge_position_exposure_decile"].to_numpy(dtype=float))
    ].copy()
    for (dataset, file_name), sub in work.groupby(["dataset", "file"], sort=False):
        grouped = (
            sub.groupby("edge_position_exposure_decile", as_index=False)
            .agg(
                n_edges=("is_boundary_edge", "size"),
                n_boundary_edges=("is_boundary_edge", "sum"),
            )
            .sort_values("edge_position_exposure_decile")
            .reset_index(drop=True)
        )
        grouped["frac_boundary_edges"] = grouped["n_boundary_edges"] / grouped["n_edges"]
        grouped.insert(0, "file", file_name)
        grouped.insert(0, "dataset", dataset)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if len(rows) else pd.DataFrame()


def _summarize_exposure_bin_trend(bin_df):
    rows = []
    if len(bin_df) == 0:
        return pd.DataFrame()

    for (dataset, file_name), sub in bin_df.groupby(["dataset", "file"], sort=False):
        x = sub["edge_position_exposure_decile"].to_numpy(dtype=float)
        y = sub["frac_boundary_edges"].to_numpy(dtype=float)
        w = np.sqrt(np.maximum(sub["n_edges"].to_numpy(dtype=float), 1.0))

        linear_slope = np.nan
        quad_coef = np.nan
        peak_location = np.nan
        interior_peak = False
        if len(sub) >= 2 and np.unique(x).size >= 2:
            linear_slope = float(np.polyfit(x, y, deg=1, w=w)[0])
        if len(sub) >= 3 and np.unique(x).size >= 3:
            coef2, coef1, _ = np.polyfit(x, y, deg=2, w=w)
            quad_coef = float(coef2)
            if coef2 < 0:
                peak_location = float(-coef1 / (2.0 * coef2))
                interior_peak = bool((peak_location > np.min(x)) and (peak_location < np.max(x)))

        rows.append(
            {
                "dataset": dataset,
                "file": file_name,
                "n_bins_tested": int(len(sub)),
                "boundary_frac_core_bin": float(sub["frac_boundary_edges"].iloc[0]),
                "boundary_frac_surface_bin": float(sub["frac_boundary_edges"].iloc[-1]),
                "boundary_frac_surface_minus_core": float(sub["frac_boundary_edges"].iloc[-1] - sub["frac_boundary_edges"].iloc[0]),
                "boundary_bin_linear_slope": linear_slope,
                "boundary_bin_quadratic_coef": quad_coef,
                "boundary_bin_peak_location": peak_location,
                "boundary_bin_interior_peak": interior_peak,
            }
        )
    return pd.DataFrame(rows)


def _run_mutation_stratification(edge_df):
    work = edge_df.loc[edge_df["edge_has_limiter_labels"].to_numpy(dtype=bool)].copy()
    if len(work) == 0:
        return pd.DataFrame()

    volume_q75 = float(work["mut_abs_volume_change"].quantile(0.75)) if np.isfinite(work["mut_abs_volume_change"]).any() else np.nan
    hydro_q75 = float(work["mut_abs_hydrophobicity_change"].quantile(0.75)) if np.isfinite(work["mut_abs_hydrophobicity_change"]).any() else np.nan

    category_specs = [
        ("Hydrophobic class flip", work["mut_hydrophobic_class_flip"].astype(bool)),
        (
            "Top quartile |delta hydrophobicity|",
            np.isfinite(work["mut_abs_hydrophobicity_change"].to_numpy(dtype=float))
            & (work["mut_abs_hydrophobicity_change"].to_numpy(dtype=float) >= hydro_q75),
        ),
        (
            "Top quartile |delta volume|",
            np.isfinite(work["mut_abs_volume_change"].to_numpy(dtype=float))
            & (work["mut_abs_volume_change"].to_numpy(dtype=float) >= volume_q75),
        ),
    ]

    rows = []
    y = work["is_boundary_edge"].to_numpy(dtype=bool)
    for label, mask in category_specs:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0 or (~mask).sum() == 0:
            fisher_or = np.nan
            fisher_p = np.nan
        else:
            table = np.array(
                [
                    [int(np.sum(y & mask)), int(np.sum((~y) & mask))],
                    [int(np.sum(y & (~mask))), int(np.sum((~y) & (~mask)))],
                ],
                dtype=int,
            )
            fisher_or, fisher_p = fisher_exact(table)

        rows.append(
            {
                "category": label,
                "n_edges_in_category": int(mask.sum()),
                "boundary_fraction_in_category": float(np.mean(y[mask])) if mask.sum() else np.nan,
                "n_edges_outside_category": int((~mask).sum()),
                "boundary_fraction_outside_category": float(np.mean(y[~mask])) if (~mask).sum() else np.nan,
                "boundary_fraction_diff": (
                    float(np.mean(y[mask]) - np.mean(y[~mask])) if mask.sum() and (~mask).sum() else np.nan
                ),
                "fisher_odds_ratio": float(fisher_or) if np.isfinite(fisher_or) else np.nan,
                "fisher_p": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _run_pooled_logistic_model(edge_df):
    work = edge_df.loc[
        edge_df["edge_has_limiter_labels"].to_numpy(dtype=bool)
        & edge_df["edge_position_mapped_to_structure"].to_numpy(dtype=bool)
    ].copy()
    if len(work) < 50 or work["is_boundary_edge"].nunique() < 2:
        return pd.DataFrame([{"status": "insufficient_data"}])

    candidate_numeric = [
        "edge_position_exposure_knn_distance",
        "edge_position_packing_contact_density",
        "edge_position_backbone_curvature_deg",
        "edge_position_relative_sasa",
        "mut_abs_hydrophobicity_change",
        "mut_abs_volume_change",
    ]
    numeric_cols = []
    for col in candidate_numeric:
        if col not in work.columns:
            continue
        vals = work[col].to_numpy(dtype=float)
        if np.isfinite(vals).sum() < 20:
            continue
        if np.nanstd(vals) <= 0:
            continue
        numeric_cols.append(col)

    if len(numeric_cols) == 0:
        return pd.DataFrame([{"status": "no_numeric_features"}])

    X_num = work[numeric_cols].copy()
    for col in numeric_cols:
        med = float(X_num[col].median())
        std = float(X_num[col].std(ddof=0))
        X_num[col] = X_num[col].fillna(med)
        if std > 0:
            X_num[col] = (X_num[col] - med) / std
        else:
            X_num[col] = 0.0

    X = pd.concat(
        [
            X_num.reset_index(drop=True),
            pd.get_dummies(work["dataset"], prefix="dataset", drop_first=True, dtype=float).reset_index(drop=True),
        ],
        axis=1,
    )
    y = work["is_boundary_edge"].astype(int).to_numpy(dtype=int)
    groups = work["dataset"].astype(str).to_numpy(dtype=object)

    model = LogisticRegression(max_iter=5000, solver="lbfgs")
    model.fit(X, y)
    prob = model.predict_proba(X)[:, 1]
    auc_train = float(roc_auc_score(y, prob))

    unique_groups = np.unique(groups)
    auc_scores = []
    if len(unique_groups) >= 3:
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        for train_idx, test_idx in splitter.split(X, y, groups):
            y_train = y[train_idx]
            y_test = y[test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue
            fold_model = LogisticRegression(max_iter=5000, solver="lbfgs")
            fold_model.fit(X.iloc[train_idx], y_train)
            fold_prob = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
            auc_scores.append(float(roc_auc_score(y_test, fold_prob)))

    coef_map = dict(zip(X.columns.tolist(), model.coef_[0].tolist()))
    rows = []
    for col in numeric_cols:
        coef = float(coef_map.get(col, np.nan))
        rows.append(
            {
                "feature": col,
                "standardized_logit_coef": coef,
                "odds_ratio_per_sd": float(np.exp(coef)) if np.isfinite(coef) else np.nan,
                "train_auc": auc_train,
                "group_cv_auc_mean": float(np.mean(auc_scores)) if len(auc_scores) else np.nan,
                "group_cv_auc_std": float(np.std(auc_scores, ddof=1)) if len(auc_scores) > 1 else np.nan,
                "n_edges": int(len(work)),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _build_edge_records(
    *,
    file_name,
    dataset_name,
    domain_df,
    landscape,
    fitness,
    genotype_df,
    axis_type_map,
    axis_exposure_map,
    axis_relative_sasa_map,
    dms_position_to_struct_idx,
    position_lookup,
):
    edge_rows = []

    seqs = domain_df["mutated_sequence"].astype(str).tolist()
    for u, v in landscape.graph.edges():
        u_i = int(u)
        v_i = int(v)
        mut_position, aa_u, aa_v, hamming_count = _single_hamming_difference(seqs[u_i], seqs[v_i])

        lim_u = genotype_df.at[u_i, "limiter_component"]
        lim_v = genotype_df.at[v_i, "limiter_component"]
        has_limiter_labels = bool(np.isfinite(lim_u) and np.isfinite(lim_v))
        is_boundary = bool(has_limiter_labels and int(lim_u) != int(lim_v))

        struct_idx = dms_position_to_struct_idx.get(int(mut_position), np.nan) if np.isfinite(mut_position) else np.nan
        position_info = position_lookup.get(int(struct_idx), {}) if np.isfinite(struct_idx) else {}
        axis_type_u = axis_type_map.get(int(lim_u), "") if has_limiter_labels else ""
        axis_type_v = axis_type_map.get(int(lim_v), "") if has_limiter_labels else ""
        axis_switch_tested = bool(is_boundary and axis_type_u in {"core", "surface"} and axis_type_v in {"core", "surface"})
        axis_exposure_u = axis_exposure_map.get(int(lim_u), np.nan) if has_limiter_labels else np.nan
        axis_exposure_v = axis_exposure_map.get(int(lim_v), np.nan) if has_limiter_labels else np.nan
        axis_relative_sasa_u = axis_relative_sasa_map.get(int(lim_u), np.nan) if has_limiter_labels else np.nan
        axis_relative_sasa_v = axis_relative_sasa_map.get(int(lim_v), np.nan) if has_limiter_labels else np.nan
        mutation_props = _mutational_property_record(aa_u, aa_v)
        margin_u = genotype_df.at[u_i, "limiter_margin"] if "limiter_margin" in genotype_df.columns else np.nan
        margin_v = genotype_df.at[v_i, "limiter_margin"] if "limiter_margin" in genotype_df.columns else np.nan
        finite_margins = [float(m) for m in [margin_u, margin_v] if np.isfinite(m)]
        mean_margin = float(np.mean(finite_margins)) if len(finite_margins) else np.nan
        min_margin = float(np.min(finite_margins)) if len(finite_margins) else np.nan

        edge_rows.append(
            {
                "file": file_name,
                "dataset": dataset_name,
                "u": u_i,
                "v": v_i,
                "edge_hamming_distance": int(hamming_count),
                "mut_position": int(mut_position) if np.isfinite(mut_position) else np.nan,
                "mut_from_aa": aa_u,
                "mut_to_aa": aa_v,
                "struct_idx": int(struct_idx) if np.isfinite(struct_idx) else np.nan,
                "edge_position_mapped_to_structure": bool(np.isfinite(struct_idx)),
                "edge_position_exposure_knn_distance": position_info.get("exposure_knn_distance", np.nan),
                "edge_position_packing_contact_density": position_info.get("packing_contact_density", np.nan),
                "edge_position_backbone_curvature_deg": position_info.get("backbone_curvature_deg", np.nan),
                "edge_position_relative_sasa": position_info.get("relative_sasa", np.nan),
                "edge_position_abs_exposure_dev_from_median": position_info.get("abs_exposure_dev_from_median", np.nan),
                "edge_position_abs_relative_sasa_dev_from_median": position_info.get("abs_relative_sasa_dev_from_median", np.nan),
                "edge_position_core_surface_label_exposure": position_info.get("core_surface_label_exposure", ""),
                "edge_position_core_surface_label_packing": position_info.get("core_surface_label_packing", ""),
                "edge_position_core_surface_label_consensus": position_info.get("core_surface_label_consensus", ""),
                "edge_position_core_surface_label_relative_sasa": position_info.get("core_surface_label_relative_sasa", ""),
                "edge_position_exposure_decile": position_info.get("exposure_decile", np.nan),
                "edge_position_relative_sasa_decile": position_info.get("relative_sasa_decile", np.nan),
                "limiter_component_u": int(lim_u) if has_limiter_labels else np.nan,
                "limiter_component_v": int(lim_v) if has_limiter_labels else np.nan,
                "limiter_margin_u": float(margin_u) if np.isfinite(margin_u) else np.nan,
                "limiter_margin_v": float(margin_v) if np.isfinite(margin_v) else np.nan,
                "edge_mean_limiter_margin": mean_margin,
                "edge_min_limiter_margin": min_margin,
                "edge_has_limiter_labels": has_limiter_labels,
                "is_boundary_edge": is_boundary,
                "axis_core_surface_type_u": axis_type_u,
                "axis_core_surface_type_v": axis_type_v,
                "axis_mean_exposure_u": axis_exposure_u,
                "axis_mean_exposure_v": axis_exposure_v,
                "axis_exposure_gap": (
                    float(abs(axis_exposure_u - axis_exposure_v))
                    if np.isfinite(axis_exposure_u) and np.isfinite(axis_exposure_v)
                    else np.nan
                ),
                "axis_mean_relative_sasa_u": axis_relative_sasa_u,
                "axis_mean_relative_sasa_v": axis_relative_sasa_v,
                "axis_relative_sasa_gap": (
                    float(abs(axis_relative_sasa_u - axis_relative_sasa_v))
                    if np.isfinite(axis_relative_sasa_u) and np.isfinite(axis_relative_sasa_v)
                    else np.nan
                ),
                "axis_structural_switch_tested": axis_switch_tested,
                "cross_core_surface_axis_switch": bool(axis_switch_tested and axis_type_u != axis_type_v),
                "abs_fitness_delta": float(abs(fitness[u_i] - fitness[v_i])),
                **mutation_props,
            }
        )

    return pd.DataFrame(edge_rows)


def _summarize_domain(
    row,
    *,
    load_domain_dataframe,
    build_hamming_landscape_from_df,
    prepare_left_sv_domain_for_pressure_control,
    build_limiter_svd_model,
    assign_limiter_to_sequence,
    outdir,
    top_k,
    top_loading_quantile,
):
    try:
        t_map = float(getattr(row, "t", getattr(row, "t_map", np.nan)))
    except Exception:
        t_map = np.nan

    file_name = str(getattr(row, "file"))
    dataset_name = str(getattr(row, "dataset", Path(file_name).stem))
    domain_out = {
        "dataset": dataset_name,
        "file": file_name,
        "t_map": float(t_map) if np.isfinite(t_map) else np.nan,
        "status": "ok",
        "error": "",
        "pdb_id": "",
        "sequence_alignment_identity": np.nan,
        "direct_sasa_status": "",
        "n_positions_mapped_to_structure": np.nan,
        "structure_exposure_median": np.nan,
        "structure_packing_median": np.nan,
        "structure_relative_sasa_median": np.nan,
        "n_axes_profiled": np.nan,
        "n_leading_axis_pairs_tested": 0,
        "n_leading_axis_pairs_exposure_sig_p05": 0,
        "frac_leading_axis_pairs_exposure_sig_p05": np.nan,
        "axis1_top_mean_exposure": np.nan,
        "axis2_top_mean_exposure": np.nan,
        "axis1_vs_axis2_mean_exposure_diff": np.nan,
        "axis1_vs_axis2_exposure_mwu_p": np.nan,
        "axis1_vs_axis2_packing_mwu_p": np.nan,
        "axis1_vs_axis2_relative_sasa_mwu_p": np.nan,
        "n_edges_total": np.nan,
        "n_edges_with_limiter_labels": np.nan,
        "n_edges_with_limiter_and_structure": np.nan,
        "n_boundary_edges_with_structure": np.nan,
        "n_non_boundary_edges_with_structure": np.nan,
        "boundary_edge_position_mean_exposure": np.nan,
        "non_boundary_edge_position_mean_exposure": np.nan,
        "boundary_vs_non_boundary_exposure_diff": np.nan,
        "boundary_vs_non_boundary_exposure_mwu_p": np.nan,
        "boundary_edge_position_mean_abs_exposure_dev": np.nan,
        "non_boundary_edge_position_mean_abs_exposure_dev": np.nan,
        "boundary_vs_non_boundary_abs_exposure_dev_diff": np.nan,
        "boundary_vs_non_boundary_abs_exposure_dev_mwu_p": np.nan,
        "boundary_edge_position_mean_relative_sasa": np.nan,
        "non_boundary_edge_position_mean_relative_sasa": np.nan,
        "boundary_vs_non_boundary_relative_sasa_diff": np.nan,
        "boundary_vs_non_boundary_relative_sasa_mwu_p": np.nan,
        "boundary_edge_core_fraction_exposure": np.nan,
        "non_boundary_edge_core_fraction_exposure": np.nan,
        "n_boundary_edges_axis_type_tested": np.nan,
        "n_boundary_edges_cross_core_surface_switch": np.nan,
        "boundary_cross_core_surface_switch_fraction": np.nan,
        "boundary_cross_core_surface_switch_expected_fraction": np.nan,
        "boundary_cross_core_surface_switch_enrichment_diff": np.nan,
        "boundary_cross_core_surface_switch_chisq_p": np.nan,
        "boundary_margin_vs_abs_exposure_dev_spearman_rho": np.nan,
        "boundary_margin_vs_abs_exposure_dev_spearman_p": np.nan,
        "boundary_axis_exposure_gap_mean_observed": np.nan,
        "boundary_axis_exposure_gap_mean_expected": np.nan,
        "boundary_axis_exposure_gap_enrichment_diff": np.nan,
        "boundary_axis_relative_sasa_gap_mean_observed": np.nan,
        "boundary_axis_relative_sasa_gap_mean_expected": np.nan,
        "boundary_axis_relative_sasa_gap_enrichment_diff": np.nan,
    }

    position_df = pd.DataFrame()
    axis_df = pd.DataFrame()
    axis_pair_df = pd.DataFrame()
    edge_df = pd.DataFrame()

    try:
        domain_df = load_domain_dataframe(file_name).reset_index(drop=True)
        prepared = prepare_left_sv_domain_for_pressure_control(file_name)
        model = build_limiter_svd_model(domain_df)
        n_axes = int(min(model["k_eff"], prepared["U_map"].shape[1]))
        if n_axes < 2:
            raise ValueError("Need at least two SVD axes for structural boundary analysis.")

        position_df = _build_position_loading_table(prepared, n_axes)
        sasa_df, sasa_status = _compute_residue_sasa_table(prepared)
        position_df = position_df.merge(sasa_df, on="struct_idx", how="left")
        position_df, exposure_median, packing_median = _classify_core_surface(position_df)
        position_df.insert(0, "dataset", dataset_name)
        position_df.insert(1, "file", file_name)
        position_df.insert(2, "pdb_id", prepared["pdb_id"])

        axis_df, axis_top_frames = _build_axis_profiles(
            file_name=file_name,
            dataset_name=dataset_name,
            pdb_id=prepared["pdb_id"],
            position_df=position_df,
            exposure_median=exposure_median,
            packing_median=packing_median,
            n_axes=n_axes,
            top_k=top_k,
            top_loading_quantile=top_loading_quantile,
        )
        axis_pair_df = _build_axis_pair_tests(
            file_name=file_name,
            dataset_name=dataset_name,
            pdb_id=prepared["pdb_id"],
            axis_top_frames=axis_top_frames,
            top_k=min(int(top_k), int(n_axes)),
        )

        landscape, fitness = build_hamming_landscape_from_df(domain_df)
        fitness = np.asarray(fitness, dtype=float)
        genotype_rows = []
        for idx, geno_row in enumerate(domain_df.itertuples(index=False)):
            limiter_out = assign_limiter_to_sequence(getattr(geno_row, "mutated_sequence"), model)
            genotype_rows.append(
                {
                    "node_idx": int(idx),
                    "mutated_sequence": getattr(geno_row, "mutated_sequence"),
                    **limiter_out,
                }
            )
        genotype_df = pd.DataFrame(genotype_rows)
        unique_mask = genotype_df["assignment_status"].eq("ok") & genotype_df["limiter_component"].notna()
        component_counts = (
            genotype_df.loc[unique_mask, "limiter_component"].astype(int).value_counts().sort_index().to_dict()
        )

        axis_type_map = (
            axis_df.set_index("axis")["axis_core_surface_type_exposure"].astype(str).to_dict()
            if len(axis_df)
            else {}
        )
        axis_exposure_map = (
            axis_df.set_index("axis")["top_mean_exposure_knn_distance"].to_dict()
            if len(axis_df)
            else {}
        )
        axis_relative_sasa_map = (
            axis_df.set_index("axis")["top_mean_relative_sasa"].to_dict()
            if len(axis_df) and "top_mean_relative_sasa" in axis_df.columns
            else {}
        )
        dms_position_to_struct_idx = {
            int(pos): int(struct_idx)
            for pos, struct_idx in zip(
                np.asarray(prepared["mapped_dms_positions"], dtype=int),
                np.asarray(prepared["mapped_struct_idx"], dtype=int),
            )
        }
        position_lookup = (
            position_df.set_index("struct_idx")[
                [
                    "exposure_knn_distance",
                    "packing_contact_density",
                    "backbone_curvature_deg",
                    "relative_sasa",
                    "abs_exposure_dev_from_median",
                    "abs_relative_sasa_dev_from_median",
                    "core_surface_label_exposure",
                    "core_surface_label_packing",
                    "core_surface_label_consensus",
                    "core_surface_label_relative_sasa",
                    "exposure_decile",
                    "relative_sasa_decile",
                ]
            ]
            .to_dict(orient="index")
        )

        edge_df = _build_edge_records(
            file_name=file_name,
            dataset_name=dataset_name,
            domain_df=domain_df,
            landscape=landscape,
            fitness=fitness,
            genotype_df=genotype_df,
            axis_type_map=axis_type_map,
            axis_exposure_map=axis_exposure_map,
            axis_relative_sasa_map=axis_relative_sasa_map,
            dms_position_to_struct_idx=dms_position_to_struct_idx,
            position_lookup=position_lookup,
        )

        leading_pair_df = axis_pair_df.loc[np.isfinite(axis_pair_df["exposure_mwu_p"].to_numpy(dtype=float))].copy()
        axis12_df = axis_pair_df.loc[
            axis_pair_df["axis_a"].eq(1) & axis_pair_df["axis_b"].eq(2)
        ].copy()
        boundary_struct_df = edge_df.loc[
            edge_df["edge_has_limiter_labels"].to_numpy(dtype=bool)
            & edge_df["edge_position_mapped_to_structure"].to_numpy(dtype=bool)
        ].copy()
        boundary_edge_df = boundary_struct_df.loc[boundary_struct_df["is_boundary_edge"].to_numpy(dtype=bool)].copy()
        non_boundary_edge_df = boundary_struct_df.loc[~boundary_struct_df["is_boundary_edge"].to_numpy(dtype=bool)].copy()

        exp_stat, exp_p, _, _ = _safe_mannwhitneyu(
            boundary_edge_df["edge_position_exposure_knn_distance"],
            non_boundary_edge_df["edge_position_exposure_knn_distance"],
        )
        abs_dev_stat, abs_dev_p, _, _ = _safe_mannwhitneyu(
            boundary_edge_df["edge_position_abs_exposure_dev_from_median"],
            non_boundary_edge_df["edge_position_abs_exposure_dev_from_median"],
        )
        sasa_stat, sasa_p, _, _ = _safe_mannwhitneyu(
            boundary_edge_df["edge_position_relative_sasa"],
            non_boundary_edge_df["edge_position_relative_sasa"],
        )

        axis_switch_df = edge_df.loc[
            edge_df["is_boundary_edge"].to_numpy(dtype=bool)
            & edge_df["axis_structural_switch_tested"].to_numpy(dtype=bool)
        ].copy()
        obs_cross = int(axis_switch_df["cross_core_surface_axis_switch"].sum()) if len(axis_switch_df) else 0
        obs_same = int(len(axis_switch_df) - obs_cross)
        expected_cross_frac = _expected_cross_axis_type_fraction(component_counts, axis_type_map)
        chi_stat, chi_p, _, _ = _safe_chisquare(obs_cross, obs_same, expected_cross_frac)
        margin_rho, margin_p, _ = _safe_spearmanr(
            boundary_edge_df["edge_mean_limiter_margin"],
            boundary_edge_df["edge_position_abs_exposure_dev_from_median"],
        )
        expected_axis_gap = _expected_axis_gap(component_counts, axis_exposure_map)
        observed_axis_gap = (
            float(axis_switch_df["axis_exposure_gap"].mean())
            if len(axis_switch_df) and np.isfinite(axis_switch_df["axis_exposure_gap"]).any()
            else np.nan
        )
        expected_axis_relative_sasa_gap = _expected_axis_gap(component_counts, axis_relative_sasa_map)
        observed_axis_relative_sasa_gap = (
            float(axis_switch_df["axis_relative_sasa_gap"].mean())
            if len(axis_switch_df) and np.isfinite(axis_switch_df["axis_relative_sasa_gap"]).any()
            else np.nan
        )

        domain_out.update(
            {
                "pdb_id": str(prepared["pdb_id"]),
                "sequence_alignment_identity": float(prepared["sequence_alignment_identity"]),
                "direct_sasa_status": sasa_status,
                "n_positions_mapped_to_structure": int(position_df["is_dms_mapped"].sum()),
                "structure_exposure_median": exposure_median,
                "structure_packing_median": packing_median,
                "structure_relative_sasa_median": (
                    float(position_df.loc[position_df["is_dms_mapped"], "relative_sasa"].median())
                    if "relative_sasa" in position_df.columns and np.isfinite(position_df.loc[position_df["is_dms_mapped"], "relative_sasa"]).any()
                    else np.nan
                ),
                "n_axes_profiled": int(n_axes),
                "n_leading_axis_pairs_tested": int(len(leading_pair_df)),
                "n_leading_axis_pairs_exposure_sig_p05": int((leading_pair_df["exposure_mwu_p"] < STRUCT_LIMITER_ALPHA).sum()),
                "frac_leading_axis_pairs_exposure_sig_p05": (
                    float((leading_pair_df["exposure_mwu_p"] < STRUCT_LIMITER_ALPHA).mean())
                    if len(leading_pair_df)
                    else np.nan
                ),
                "axis1_top_mean_exposure": (
                    float(axis_df.loc[axis_df["axis"].eq(1), "top_mean_exposure_knn_distance"].iloc[0])
                    if len(axis_df.loc[axis_df["axis"].eq(1)]) > 0
                    else np.nan
                ),
                "axis2_top_mean_exposure": (
                    float(axis_df.loc[axis_df["axis"].eq(2), "top_mean_exposure_knn_distance"].iloc[0])
                    if len(axis_df.loc[axis_df["axis"].eq(2)]) > 0
                    else np.nan
                ),
                "axis1_vs_axis2_mean_exposure_diff": (
                    float(axis12_df["mean_exposure_axis_b"].iloc[0] - axis12_df["mean_exposure_axis_a"].iloc[0])
                    if len(axis12_df)
                    else np.nan
                ),
                "axis1_vs_axis2_exposure_mwu_p": (
                    float(axis12_df["exposure_mwu_p"].iloc[0]) if len(axis12_df) else np.nan
                ),
                "axis1_vs_axis2_packing_mwu_p": (
                    float(axis12_df["packing_mwu_p"].iloc[0]) if len(axis12_df) else np.nan
                ),
                "axis1_vs_axis2_relative_sasa_mwu_p": (
                    float(axis12_df["relative_sasa_mwu_p"].iloc[0]) if len(axis12_df) else np.nan
                ),
                "n_edges_total": int(landscape.graph.number_of_edges()),
                "n_edges_with_limiter_labels": int(edge_df["edge_has_limiter_labels"].sum()),
                "n_edges_with_limiter_and_structure": int(len(boundary_struct_df)),
                "n_boundary_edges_with_structure": int(len(boundary_edge_df)),
                "n_non_boundary_edges_with_structure": int(len(non_boundary_edge_df)),
                "boundary_edge_position_mean_exposure": (
                    float(boundary_edge_df["edge_position_exposure_knn_distance"].mean()) if len(boundary_edge_df) else np.nan
                ),
                "non_boundary_edge_position_mean_exposure": (
                    float(non_boundary_edge_df["edge_position_exposure_knn_distance"].mean()) if len(non_boundary_edge_df) else np.nan
                ),
                "boundary_vs_non_boundary_exposure_diff": (
                    float(boundary_edge_df["edge_position_exposure_knn_distance"].mean() - non_boundary_edge_df["edge_position_exposure_knn_distance"].mean())
                    if len(boundary_edge_df) and len(non_boundary_edge_df)
                    else np.nan
                ),
                "boundary_vs_non_boundary_exposure_mwu_p": exp_p,
                "boundary_edge_position_mean_abs_exposure_dev": (
                    float(boundary_edge_df["edge_position_abs_exposure_dev_from_median"].mean()) if len(boundary_edge_df) else np.nan
                ),
                "non_boundary_edge_position_mean_abs_exposure_dev": (
                    float(non_boundary_edge_df["edge_position_abs_exposure_dev_from_median"].mean()) if len(non_boundary_edge_df) else np.nan
                ),
                "boundary_vs_non_boundary_abs_exposure_dev_diff": (
                    float(boundary_edge_df["edge_position_abs_exposure_dev_from_median"].mean() - non_boundary_edge_df["edge_position_abs_exposure_dev_from_median"].mean())
                    if len(boundary_edge_df) and len(non_boundary_edge_df)
                    else np.nan
                ),
                "boundary_vs_non_boundary_abs_exposure_dev_mwu_p": abs_dev_p,
                "boundary_edge_position_mean_relative_sasa": (
                    float(boundary_edge_df["edge_position_relative_sasa"].mean()) if len(boundary_edge_df) else np.nan
                ),
                "non_boundary_edge_position_mean_relative_sasa": (
                    float(non_boundary_edge_df["edge_position_relative_sasa"].mean()) if len(non_boundary_edge_df) else np.nan
                ),
                "boundary_vs_non_boundary_relative_sasa_diff": (
                    float(boundary_edge_df["edge_position_relative_sasa"].mean() - non_boundary_edge_df["edge_position_relative_sasa"].mean())
                    if len(boundary_edge_df) and len(non_boundary_edge_df)
                    else np.nan
                ),
                "boundary_vs_non_boundary_relative_sasa_mwu_p": sasa_p,
                "boundary_edge_core_fraction_exposure": (
                    float(np.mean(boundary_edge_df["edge_position_core_surface_label_exposure"].to_numpy(dtype=object) == "core"))
                    if len(boundary_edge_df)
                    else np.nan
                ),
                "non_boundary_edge_core_fraction_exposure": (
                    float(np.mean(non_boundary_edge_df["edge_position_core_surface_label_exposure"].to_numpy(dtype=object) == "core"))
                    if len(non_boundary_edge_df)
                    else np.nan
                ),
                "n_boundary_edges_axis_type_tested": int(len(axis_switch_df)),
                "n_boundary_edges_cross_core_surface_switch": int(obs_cross),
                "boundary_cross_core_surface_switch_fraction": (
                    float(obs_cross / len(axis_switch_df)) if len(axis_switch_df) else np.nan
                ),
                "boundary_cross_core_surface_switch_expected_fraction": expected_cross_frac,
                "boundary_cross_core_surface_switch_enrichment_diff": (
                    float((obs_cross / len(axis_switch_df)) - expected_cross_frac)
                    if len(axis_switch_df) and np.isfinite(expected_cross_frac)
                    else np.nan
                ),
                "boundary_cross_core_surface_switch_chisq_p": chi_p,
                "boundary_margin_vs_abs_exposure_dev_spearman_rho": margin_rho,
                "boundary_margin_vs_abs_exposure_dev_spearman_p": margin_p,
                "boundary_axis_exposure_gap_mean_observed": observed_axis_gap,
                "boundary_axis_exposure_gap_mean_expected": expected_axis_gap,
                "boundary_axis_exposure_gap_enrichment_diff": (
                    float(observed_axis_gap - expected_axis_gap)
                    if np.isfinite(observed_axis_gap) and np.isfinite(expected_axis_gap)
                    else np.nan
                ),
                "boundary_axis_relative_sasa_gap_mean_observed": observed_axis_relative_sasa_gap,
                "boundary_axis_relative_sasa_gap_mean_expected": expected_axis_relative_sasa_gap,
                "boundary_axis_relative_sasa_gap_enrichment_diff": (
                    float(observed_axis_relative_sasa_gap - expected_axis_relative_sasa_gap)
                    if np.isfinite(observed_axis_relative_sasa_gap) and np.isfinite(expected_axis_relative_sasa_gap)
                    else np.nan
                ),
            }
        )

    except Exception as exc:
        domain_out["status"] = "error"
        domain_out["error"] = f"{type(exc).__name__}: {exc}"

    return domain_out, position_df, axis_df, axis_pair_df, edge_df


def _build_summary_table(domain_df):
    summary_rows = []

    def add_metric(metric, mask_tested, mask_positive, effect_col=None, p_col=None):
        tested_df = domain_df.loc[mask_tested].copy()
        if len(tested_df) == 0:
            summary_rows.append(
                {
                    "metric": metric,
                    "n_tested": 0,
                    "n_positive": 0,
                    "fraction_positive": np.nan,
                    "median_effect": np.nan,
                    "median_p_value": np.nan,
                }
            )
            return

        positive = tested_df.loc[mask_positive.loc[tested_df.index]].copy()
        summary_rows.append(
            {
                "metric": metric,
                "n_tested": int(len(tested_df)),
                "n_positive": int(len(positive)),
                "fraction_positive": float(len(positive) / len(tested_df)),
                "median_effect": (
                    float(tested_df[effect_col].median())
                    if effect_col is not None and effect_col in tested_df.columns
                    else np.nan
                ),
                "median_p_value": (
                    float(tested_df[p_col].median())
                    if p_col is not None and p_col in tested_df.columns
                    else np.nan
                ),
            }
        )

    ok_mask = domain_df["status"].eq("ok")
    add_metric(
        "Axis 1 vs Axis 2 exposure separation (p<0.05)",
        ok_mask & np.isfinite(domain_df["axis1_vs_axis2_exposure_mwu_p"].to_numpy(dtype=float)),
        ok_mask
        & np.isfinite(domain_df["axis1_vs_axis2_exposure_mwu_p"].to_numpy(dtype=float))
        & (domain_df["axis1_vs_axis2_exposure_mwu_p"] < STRUCT_LIMITER_ALPHA),
        effect_col="axis1_vs_axis2_mean_exposure_diff",
        p_col="axis1_vs_axis2_exposure_mwu_p",
    )
    add_metric(
        "Any leading-axis exposure pair separation (fraction significant > 0)",
        ok_mask & np.isfinite(domain_df["frac_leading_axis_pairs_exposure_sig_p05"].to_numpy(dtype=float)),
        ok_mask
        & np.isfinite(domain_df["frac_leading_axis_pairs_exposure_sig_p05"].to_numpy(dtype=float))
        & (domain_df["frac_leading_axis_pairs_exposure_sig_p05"] > 0.0),
        effect_col="frac_leading_axis_pairs_exposure_sig_p05",
    )
    add_metric(
        "Boundary vs non-boundary edge exposure differs (p<0.05)",
        ok_mask & np.isfinite(domain_df["boundary_vs_non_boundary_exposure_mwu_p"].to_numpy(dtype=float)),
        ok_mask
        & np.isfinite(domain_df["boundary_vs_non_boundary_exposure_mwu_p"].to_numpy(dtype=float))
        & (domain_df["boundary_vs_non_boundary_exposure_mwu_p"] < STRUCT_LIMITER_ALPHA),
        effect_col="boundary_vs_non_boundary_exposure_diff",
        p_col="boundary_vs_non_boundary_exposure_mwu_p",
    )
    add_metric(
        "Boundary vs non-boundary relative SASA differs (p<0.05)",
        ok_mask & np.isfinite(domain_df["boundary_vs_non_boundary_relative_sasa_mwu_p"].to_numpy(dtype=float)),
        ok_mask
        & np.isfinite(domain_df["boundary_vs_non_boundary_relative_sasa_mwu_p"].to_numpy(dtype=float))
        & (domain_df["boundary_vs_non_boundary_relative_sasa_mwu_p"] < STRUCT_LIMITER_ALPHA),
        effect_col="boundary_vs_non_boundary_relative_sasa_diff",
        p_col="boundary_vs_non_boundary_relative_sasa_mwu_p",
    )
    add_metric(
        "Boundary edges sit closer to the exposure median (p<0.05, lower abs dev)",
        ok_mask & np.isfinite(domain_df["boundary_vs_non_boundary_abs_exposure_dev_mwu_p"].to_numpy(dtype=float)),
        ok_mask
        & np.isfinite(domain_df["boundary_vs_non_boundary_abs_exposure_dev_mwu_p"].to_numpy(dtype=float))
        & (domain_df["boundary_vs_non_boundary_abs_exposure_dev_mwu_p"] < STRUCT_LIMITER_ALPHA)
        & (domain_df["boundary_vs_non_boundary_abs_exposure_dev_diff"] < 0.0),
        effect_col="boundary_vs_non_boundary_abs_exposure_dev_diff",
        p_col="boundary_vs_non_boundary_abs_exposure_dev_mwu_p",
    )
    add_metric(
        "Boundary margins increase away from the exposure median (rho>0, p<0.05)",
        ok_mask & np.isfinite(domain_df["boundary_margin_vs_abs_exposure_dev_spearman_p"].to_numpy(dtype=float)),
        ok_mask
        & np.isfinite(domain_df["boundary_margin_vs_abs_exposure_dev_spearman_p"].to_numpy(dtype=float))
        & (domain_df["boundary_margin_vs_abs_exposure_dev_spearman_p"] < STRUCT_LIMITER_ALPHA)
        & (domain_df["boundary_margin_vs_abs_exposure_dev_spearman_rho"] > 0.0),
        effect_col="boundary_margin_vs_abs_exposure_dev_spearman_rho",
        p_col="boundary_margin_vs_abs_exposure_dev_spearman_p",
    )
    if "boundary_bin_interior_peak" in domain_df.columns:
        add_metric(
            "Boundary incidence peaks at intermediate exposure bins",
            ok_mask & domain_df["boundary_bin_interior_peak"].notna(),
            ok_mask & domain_df["boundary_bin_interior_peak"].fillna(False),
            effect_col="boundary_bin_quadratic_coef",
        )
    add_metric(
        "Boundary switches enriched for core/surface axis flips (p<0.05, observed>expected)",
        ok_mask & np.isfinite(domain_df["boundary_cross_core_surface_switch_chisq_p"].to_numpy(dtype=float)),
        ok_mask
        & np.isfinite(domain_df["boundary_cross_core_surface_switch_chisq_p"].to_numpy(dtype=float))
        & (domain_df["boundary_cross_core_surface_switch_chisq_p"] < STRUCT_LIMITER_ALPHA)
        & (domain_df["boundary_cross_core_surface_switch_enrichment_diff"] > 0.0),
        effect_col="boundary_cross_core_surface_switch_enrichment_diff",
        p_col="boundary_cross_core_surface_switch_chisq_p",
    )
    add_metric(
        "Boundary axis exposure gap exceeds the marginal null",
        ok_mask & np.isfinite(domain_df["boundary_axis_exposure_gap_enrichment_diff"].to_numpy(dtype=float)),
        ok_mask
        & np.isfinite(domain_df["boundary_axis_exposure_gap_enrichment_diff"].to_numpy(dtype=float))
        & (domain_df["boundary_axis_exposure_gap_enrichment_diff"] > 0.0),
        effect_col="boundary_axis_exposure_gap_enrichment_diff",
    )
    return pd.DataFrame(summary_rows)


def _plot_summary(domain_df, summary_df, fig_path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    ax_axis = axes[0, 0]
    ax_boundary = axes[0, 1]
    ax_cross = axes[1, 0]
    ax_bar = axes[1, 1]

    axis12_df = domain_df.loc[
        domain_df["status"].eq("ok")
        & np.isfinite(domain_df["axis1_top_mean_exposure"].to_numpy(dtype=float))
        & np.isfinite(domain_df["axis2_top_mean_exposure"].to_numpy(dtype=float))
    ].copy()
    if len(axis12_df) > 0:
        x = axis12_df["axis1_top_mean_exposure"].to_numpy(dtype=float)
        y = axis12_df["axis2_top_mean_exposure"].to_numpy(dtype=float)
        ax_axis.scatter(x, y, s=28, facecolor="white", edgecolor="black", linewidth=0.8)
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax_axis.plot([lo, hi], [lo, hi], linestyle="--", color="0.6", linewidth=1.0)
        sig_n = int(
            np.sum(axis12_df["axis1_vs_axis2_exposure_mwu_p"].to_numpy(dtype=float) < STRUCT_LIMITER_ALPHA)
        )
        ax_axis.set_xlabel("Axis 1 mean exposure")
        ax_axis.set_ylabel("Axis 2 mean exposure")
        ax_axis.set_title(f"A. Axis 1 vs 2 top-loading burial ({sig_n}/{len(axis12_df)} p<0.05)")
        ax_axis.grid(alpha=0.25, linestyle="--")
    else:
        ax_axis.text(0.5, 0.5, "No axis 1/2 structural comparisons", ha="center", va="center")
        ax_axis.set_axis_off()

    boundary_df = domain_df.loc[
        domain_df["status"].eq("ok")
        & np.isfinite(domain_df["boundary_edge_position_mean_abs_exposure_dev"].to_numpy(dtype=float))
        & np.isfinite(domain_df["non_boundary_edge_position_mean_abs_exposure_dev"].to_numpy(dtype=float))
    ].copy()
    if len(boundary_df) > 0:
        for row in boundary_df.itertuples(index=False):
            ax_boundary.plot(
                [0, 1],
                [
                    float(row.non_boundary_edge_position_mean_abs_exposure_dev),
                    float(row.boundary_edge_position_mean_abs_exposure_dev),
                ],
                color="0.7",
                linewidth=0.8,
                alpha=0.8,
            )
        ax_boundary.scatter(
            np.zeros(len(boundary_df)),
            boundary_df["non_boundary_edge_position_mean_abs_exposure_dev"].to_numpy(dtype=float),
            color="#4C78A8",
            s=24,
            zorder=3,
        )
        ax_boundary.scatter(
            np.ones(len(boundary_df)),
            boundary_df["boundary_edge_position_mean_abs_exposure_dev"].to_numpy(dtype=float),
            color="#E45756",
            s=24,
            zorder=3,
        )
        sig_n = int(
            np.sum(
                (boundary_df["boundary_vs_non_boundary_abs_exposure_dev_mwu_p"].to_numpy(dtype=float) < STRUCT_LIMITER_ALPHA)
                & (boundary_df["boundary_vs_non_boundary_abs_exposure_dev_diff"].to_numpy(dtype=float) < 0.0)
            )
        )
        ax_boundary.set_xticks([0, 1])
        ax_boundary.set_xticklabels(["Non-boundary", "Boundary"])
        ax_boundary.set_ylabel("|Exposure - domain median|")
        ax_boundary.set_title(f"B. Boundary positions nearer the median ({sig_n}/{len(boundary_df)})")
        ax_boundary.grid(axis="y", alpha=0.25, linestyle="--")
    else:
        ax_boundary.text(0.5, 0.5, "No boundary-structure comparisons", ha="center", va="center")
        ax_boundary.set_axis_off()

    cross_df = domain_df.loc[
        domain_df["status"].eq("ok")
        & np.isfinite(domain_df["boundary_cross_core_surface_switch_fraction"].to_numpy(dtype=float))
        & np.isfinite(domain_df["boundary_cross_core_surface_switch_expected_fraction"].to_numpy(dtype=float))
    ].copy()
    if len(cross_df) > 0:
        for row in cross_df.itertuples(index=False):
            ax_cross.plot(
                [0, 1],
                [
                    float(row.boundary_cross_core_surface_switch_expected_fraction),
                    float(row.boundary_cross_core_surface_switch_fraction),
                ],
                color="0.7",
                linewidth=0.8,
                alpha=0.8,
            )
        ax_cross.scatter(
            np.zeros(len(cross_df)),
            cross_df["boundary_cross_core_surface_switch_expected_fraction"].to_numpy(dtype=float),
            color="#4C78A8",
            s=24,
            zorder=3,
        )
        ax_cross.scatter(
            np.ones(len(cross_df)),
            cross_df["boundary_cross_core_surface_switch_fraction"].to_numpy(dtype=float),
            color="#E45756",
            s=24,
            zorder=3,
        )
        sig_n = int(
            np.sum(
                (cross_df["boundary_cross_core_surface_switch_chisq_p"].to_numpy(dtype=float) < STRUCT_LIMITER_ALPHA)
                & (cross_df["boundary_cross_core_surface_switch_enrichment_diff"].to_numpy(dtype=float) > 0.0)
            )
        )
        ax_cross.set_xticks([0, 1])
        ax_cross.set_xticklabels(["Expected", "Observed"])
        ax_cross.set_ylabel("Cross-type boundary fraction")
        ax_cross.set_title(f"C. Core/surface axis-flip enrichment ({sig_n}/{len(cross_df)})")
        ax_cross.grid(axis="y", alpha=0.25, linestyle="--")
    else:
        ax_cross.text(0.5, 0.5, "No axis-type switch tests", ha="center", va="center")
        ax_cross.set_axis_off()

    if len(summary_df) > 0:
        plot_df = summary_df.copy()
        ypos = np.arange(len(plot_df))
        ax_bar.barh(ypos, plot_df["fraction_positive"].to_numpy(dtype=float), color="0.2")
        ax_bar.set_yticks(ypos)
        ax_bar.set_yticklabels(plot_df["metric"].tolist(), fontsize=8)
        ax_bar.set_xlim(0.0, 1.0)
        ax_bar.set_xlabel("Fraction of tested domains")
        ax_bar.set_title("D. Across-domain summary")
        for y_idx, row in enumerate(plot_df.itertuples(index=False)):
            if np.isfinite(row.fraction_positive):
                ax_bar.text(
                    min(float(row.fraction_positive) + 0.02, 0.98),
                    y_idx,
                    f"{int(row.n_positive)}/{int(row.n_tested)}",
                    va="center",
                    fontsize=8,
                )
        ax_bar.grid(axis="x", alpha=0.25, linestyle="--")
    else:
        ax_bar.text(0.5, 0.5, "No summary metrics available", ha="center", va="center")
        ax_bar.set_axis_off()

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)


def run_structural_limiter_boundary_checks(
    *,
    dms_tmap_df,
    load_domain_dataframe,
    build_hamming_landscape_from_df,
    prepare_left_sv_domain_for_pressure_control,
    build_limiter_svd_model,
    assign_limiter_to_sequence,
    outdir,
    top_k=STRUCT_LIMITER_TOP_K,
    top_loading_quantile=STRUCT_LIMITER_TOP_LOADING_QUANTILE,
    max_domains=None,
    load_streamed_frames=False,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    domain_csv = outdir / "sv_limiter_structure_boundary_domain_summary.csv"
    axis_csv = outdir / "sv_limiter_structure_boundary_axis_summary.csv"
    axis_pair_csv = outdir / "sv_limiter_structure_boundary_axis_pair_tests.csv"
    position_csv = outdir / "sv_limiter_structure_boundary_position_summary.csv"
    edge_csv = outdir / "sv_limiter_structure_boundary_edge_details.csv"
    exposure_bin_csv = outdir / "sv_limiter_structure_boundary_exposure_bin_summary.csv"
    exposure_bin_domain_csv = outdir / "sv_limiter_structure_boundary_exposure_bin_domain_summary.csv"
    logistic_csv = outdir / "sv_limiter_structure_boundary_logistic_summary.csv"
    mutation_csv = outdir / "sv_limiter_structure_boundary_mutation_summary.csv"
    summary_csv = outdir / "sv_limiter_structure_boundary_summary.csv"
    fig_path = outdir / "sv_limiter_structure_boundary_summary.pdf"

    for output_path in [
        domain_csv,
        axis_csv,
        axis_pair_csv,
        position_csv,
        edge_csv,
        exposure_bin_csv,
        exposure_bin_domain_csv,
        logistic_csv,
        mutation_csv,
        summary_csv,
        fig_path,
    ]:
        if output_path.exists():
            output_path.unlink()

    work_df = dms_tmap_df.copy()
    if max_domains is not None:
        work_df = work_df.head(int(max_domains)).copy()

    domain_rows = []
    write_domain_header = True
    write_axis_header = True
    write_axis_pair_header = True
    write_position_header = True
    write_edge_header = True

    for row in tqdm(work_df.itertuples(index=False), total=len(work_df), desc="Structural limiter boundaries"):
        domain_row, position_df, axis_df, axis_pair_df, edge_df = _summarize_domain(
            row,
            load_domain_dataframe=load_domain_dataframe,
            build_hamming_landscape_from_df=build_hamming_landscape_from_df,
            prepare_left_sv_domain_for_pressure_control=prepare_left_sv_domain_for_pressure_control,
            build_limiter_svd_model=build_limiter_svd_model,
            assign_limiter_to_sequence=assign_limiter_to_sequence,
            outdir=outdir,
            top_k=top_k,
            top_loading_quantile=top_loading_quantile,
        )
        domain_rows.append(domain_row)
        write_domain_header = _append_frame_to_csv(pd.DataFrame([domain_row]), domain_csv, write_header=write_domain_header)
        write_axis_header = _append_frame_to_csv(axis_df, axis_csv, write_header=write_axis_header)
        write_axis_pair_header = _append_frame_to_csv(axis_pair_df, axis_pair_csv, write_header=write_axis_pair_header)
        write_position_header = _append_frame_to_csv(position_df, position_csv, write_header=write_position_header)
        write_edge_header = _append_frame_to_csv(edge_df, edge_csv, write_header=write_edge_header)

    domain_df = pd.DataFrame(domain_rows).sort_values("t_map", ascending=False).reset_index(drop=True)
    axis_df = _load_csv_or_empty(axis_csv)
    axis_pair_df = _load_csv_or_empty(axis_pair_csv)
    position_df = _load_csv_or_empty(position_csv)
    edge_df = _load_csv_or_empty(edge_csv)

    exposure_bin_df = _build_exposure_bin_rows(edge_df)
    exposure_bin_domain_df = _summarize_exposure_bin_trend(exposure_bin_df)
    if len(exposure_bin_df) > 0:
        exposure_bin_df.to_csv(exposure_bin_csv, index=False)
    if len(exposure_bin_domain_df) > 0:
        exposure_bin_domain_df.to_csv(exposure_bin_domain_csv, index=False)
        domain_df = domain_df.merge(exposure_bin_domain_df, on=["dataset", "file"], how="left")

    logistic_df = _run_pooled_logistic_model(edge_df)
    mutation_df = _run_mutation_stratification(edge_df)
    logistic_df.to_csv(logistic_csv, index=False)
    mutation_df.to_csv(mutation_csv, index=False)

    summary_df = _build_summary_table(domain_df)
    summary_df.to_csv(summary_csv, index=False)
    _plot_summary(domain_df, summary_df, fig_path)

    return {
        "domain_df": domain_df,
        "axis_df": axis_df,
        "axis_pair_df": axis_pair_df,
        "position_df": position_df,
        "edge_df": edge_df,
        "exposure_bin_df": exposure_bin_df,
        "exposure_bin_domain_df": exposure_bin_domain_df,
        "logistic_df": logistic_df,
        "mutation_df": mutation_df,
        "summary_df": summary_df,
        "domain_csv": domain_csv,
        "axis_csv": axis_csv,
        "axis_pair_csv": axis_pair_csv,
        "position_csv": position_csv,
        "edge_csv": edge_csv,
        "exposure_bin_csv": exposure_bin_csv,
        "exposure_bin_domain_csv": exposure_bin_domain_csv,
        "logistic_csv": logistic_csv,
        "mutation_csv": mutation_csv,
        "summary_csv": summary_csv,
        "fig_path": fig_path,
    }
