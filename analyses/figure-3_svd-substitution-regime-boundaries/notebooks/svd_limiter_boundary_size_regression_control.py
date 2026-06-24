from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as student_t


DEFAULT_LIMITER_FEATURES = [
    "frac_boundary_edges",
    "boundary_abs_fitness_delta_diff",
    "boundary_abs_fitness_delta_ratio",
    "limiter_effective_count",
    "frac_genotypes_unique_limiter",
    "mean_limiter_margin",
]

DEFAULT_PLOT_FEATURES = [
    "frac_boundary_edges",
    "boundary_abs_fitness_delta_diff",
    "limiter_effective_count",
    "frac_genotypes_unique_limiter",
]


def _zscore_series(values):
    arr = np.asarray(values, dtype=float)
    mu = np.nanmean(arr)
    sigma = np.nanstd(arr, ddof=0)
    if (not np.isfinite(sigma)) or sigma <= 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sigma


def _residualize_against_covariates(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X[:, None]

    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    resid = np.full(len(y), np.nan, dtype=float)
    if int(np.sum(valid)) < X.shape[1] + 3:
        return resid, valid

    y_valid = y[valid]
    X_valid = X[valid]
    X_design = np.column_stack([np.ones(len(y_valid)), X_valid])
    beta, *_ = np.linalg.lstsq(X_design, y_valid, rcond=None)
    resid[valid] = y_valid - X_design @ beta
    return resid, valid


def _safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(np.sum(valid))
    if n_valid < 3:
        return np.nan, np.nan, n_valid

    x = x[valid]
    y = y[valid]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, n_valid

    if method == "pearson":
        stat, pval = pearsonr(x, y)
    elif method == "spearman":
        stat, pval = spearmanr(x, y)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return float(stat), float(pval), n_valid


def _standardized_tmap_coef_with_p(y_feature, x_tmap, covariates):
    y_feature = np.asarray(y_feature, dtype=float)
    x_tmap = np.asarray(x_tmap, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    if covariates.ndim == 1:
        covariates = covariates[:, None]

    valid = (
        np.isfinite(y_feature)
        & np.isfinite(x_tmap)
        & np.all(np.isfinite(covariates), axis=1)
    )
    n_valid = int(np.sum(valid))
    p_cov = int(covariates.shape[1])
    if n_valid < p_cov + 4:
        return np.nan, np.nan, n_valid

    y = _zscore_series(y_feature[valid])
    x = _zscore_series(x_tmap[valid])
    C = np.column_stack(
        [_zscore_series(covariates[valid, j]) for j in range(p_cov)]
    )

    X = np.column_stack([np.ones(n_valid), x, C])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    df_resid = n_valid - X.shape[1]
    if df_resid <= 0:
        return np.nan, np.nan, n_valid

    sigma2 = float(np.sum(resid**2) / df_resid)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_tmap = float(np.sqrt(max(0.0, sigma2 * XtX_inv[1, 1])))
    if se_tmap <= 0 or not np.isfinite(se_tmap):
        return np.nan, np.nan, n_valid

    t_stat = float(beta[1] / se_tmap)
    p_value = float(2.0 * student_t.sf(np.abs(t_stat), df=df_resid))
    return float(beta[1]), p_value, n_valid


def _plot_adjusted_feature(ax, domain_df, feature, size_covariate):
    tmp = domain_df[["t_map", feature, size_covariate]].copy().dropna()
    if len(tmp) < 4:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()
        return

    cov_mat = tmp[[size_covariate]].to_numpy(dtype=float)
    x_resid, _ = _residualize_against_covariates(
        tmp["t_map"].to_numpy(dtype=float),
        cov_mat,
    )
    y_resid, _ = _residualize_against_covariates(
        tmp[feature].to_numpy(dtype=float),
        cov_mat,
    )
    valid = np.isfinite(x_resid) & np.isfinite(y_resid)
    if int(np.sum(valid)) < 3:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()
        return

    x_plot = x_resid[valid]
    y_plot = y_resid[valid]
    if np.unique(x_plot).size < 2 or np.unique(y_plot).size < 2:
        ax.text(0.5, 0.5, "Insufficient variation", ha="center", va="center")
        ax.set_axis_off()
        return

    rho, rho_p, n_plot = _safe_corr(x_plot, y_plot, method="spearman")
    ax.scatter(
        x_plot,
        y_plot,
        s=26,
        facecolor="lightgrey",
        edgecolor="black",
        linewidth=0.75,
        zorder=3,
    )
    coef = np.polyfit(x_plot, y_plot, deg=1)
    x_line = np.linspace(np.min(x_plot), np.max(x_plot), 200)
    ax.plot(x_line, coef[0] * x_line + coef[1], color="black", linewidth=1.0)
    ax.set_xlabel(r"$t_{\mathrm{MAP}}$ residual | $\log_{10}$(size)")
    ax.set_ylabel(f"{feature} residual | " + r"$\log_{10}$(size)")
    ax.set_title(f"rho={rho:.2f}, p={rho_p:.3g}, n={n_plot}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def run_svd_limiter_boundary_size_regression_control(
    *,
    sv_limiter_domain_df=None,
    limiter_domain_csv=None,
    outdir,
    size_column="n_genotypes_total",
    limiter_features=None,
    plot_features=None,
    show_plot=True,
):
    if sv_limiter_domain_df is None:
        if limiter_domain_csv is None:
            raise ValueError("Need `sv_limiter_domain_df` or `limiter_domain_csv`.")
        sv_limiter_domain_df = pd.read_csv(limiter_domain_csv)
    else:
        sv_limiter_domain_df = sv_limiter_domain_df.copy()

    if limiter_features is None:
        limiter_features = list(DEFAULT_LIMITER_FEATURES)
    if plot_features is None:
        plot_features = list(DEFAULT_PLOT_FEATURES)

    required_cols = {"dataset", "file", "t_map", "status", size_column}
    missing_cols = sorted(required_cols.difference(sv_limiter_domain_df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    domain_csv = outdir / "sv_limiter_boundary_size_control_domain_summary.csv"
    assoc_csv = outdir / "sv_limiter_boundary_size_control_associations.csv"
    fig_path = outdir / "tmap_vs_sv_limiter_boundary_metrics_adjusted_for_size.pdf"

    domain_df = sv_limiter_domain_df.loc[
        sv_limiter_domain_df["status"].eq("ok")
        & np.isfinite(sv_limiter_domain_df["t_map"].to_numpy(dtype=float))
        & np.isfinite(sv_limiter_domain_df[size_column].to_numpy(dtype=float))
        & sv_limiter_domain_df[size_column].gt(0),
    ].copy()
    if len(domain_df) == 0:
        raise RuntimeError("No successful SVD-limiter domains with valid size values.")

    domain_df["landscape_size"] = domain_df[size_column].to_numpy(dtype=float)
    domain_df["log10_landscape_size"] = np.log10(domain_df["landscape_size"])
    domain_df = domain_df.sort_values("t_map", ascending=False).reset_index(drop=True)

    assoc_rows = []
    for feat in limiter_features:
        if feat not in domain_df.columns:
            continue

        tmp = domain_df[["t_map", feat, "log10_landscape_size"]].copy().dropna()
        if len(tmp) < 5:
            continue
        if (
            np.unique(tmp["t_map"].to_numpy(dtype=float)).size < 2
            or np.unique(tmp[feat].to_numpy(dtype=float)).size < 2
        ):
            continue

        raw_r, raw_r_p, n_raw = _safe_corr(tmp["t_map"], tmp[feat], method="pearson")
        raw_rho, raw_rho_p, _ = _safe_corr(
            tmp["t_map"], tmp[feat], method="spearman"
        )

        cov_mat = tmp[["log10_landscape_size"]].to_numpy(dtype=float)
        resid_t, _ = _residualize_against_covariates(
            tmp["t_map"].to_numpy(dtype=float),
            cov_mat,
        )
        resid_feat, _ = _residualize_against_covariates(
            tmp[feat].to_numpy(dtype=float),
            cov_mat,
        )

        adj_r, adj_r_p, n_adj = _safe_corr(resid_t, resid_feat, method="pearson")
        adj_rho, adj_rho_p, _ = _safe_corr(
            resid_t,
            resid_feat,
            method="spearman",
        )
        beta_tmap, beta_tmap_p, n_model = _standardized_tmap_coef_with_p(
            tmp[feat].to_numpy(dtype=float),
            tmp["t_map"].to_numpy(dtype=float),
            cov_mat,
        )

        assoc_rows.append(
            {
                "feature": feat,
                "size_covariate": f"log10_{size_column}",
                "n_domains_raw": int(n_raw),
                "raw_pearson_r": raw_r,
                "raw_pearson_p": raw_r_p,
                "raw_spearman_rho": raw_rho,
                "raw_spearman_p": raw_rho_p,
                "n_domains_adjusted": int(n_adj),
                "adjusted_pearson_r": adj_r,
                "adjusted_pearson_p": adj_r_p,
                "adjusted_spearman_rho": adj_rho,
                "adjusted_spearman_p": adj_rho_p,
                "std_tmap_coef": beta_tmap,
                "std_tmap_coef_p": beta_tmap_p,
                "n_domains_model": int(n_model),
            }
        )

    assoc_df = pd.DataFrame(assoc_rows)
    if len(assoc_df) > 0:
        assoc_df = assoc_df.sort_values(
            "adjusted_spearman_rho",
            key=np.abs,
            ascending=False,
        ).reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), squeeze=False)
    for ax, feat in zip(axes.ravel(), plot_features):
        _plot_adjusted_feature(
            ax,
            domain_df,
            feat,
            size_covariate="log10_landscape_size",
        )

    plt.tight_layout()
    plt.savefig(fig_path)
    if show_plot:
        plt.show()
    plt.close(fig)

    domain_df.to_csv(domain_csv, index=False)
    assoc_df.to_csv(assoc_csv, index=False)

    return {
        "domain_df": domain_df,
        "assoc_df": assoc_df,
        "domain_csv": domain_csv,
        "assoc_csv": assoc_csv,
        "fig_path": fig_path,
    }
