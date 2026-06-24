from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


SEED = 42
N_POS = 60
N_AA = 20
N_COMPONENTS_TRUE = 5

C1 = "#2166ac"
C2 = "#b2182b"
C3 = "#fddc8a"
C3_TEXT = "#d8a11d"
HILITE = "#f28e00"
EDGE_LIGHT = "#d0d0d0"
EDGE_BOUNDARY = "#3a3a3a"

SELECTED_POSITIONS = [5, 14, 23]
SELECTED_AA_INDICES = [13, 7, 11]
DISPLAY_CONTRIB_TARGET = np.array([-1.79, -1.17, -0.89], dtype=float)


def _build_svd_example():
    rng = np.random.default_rng(SEED)
    matrix = np.zeros((N_POS, N_AA), dtype=float)
    for _ in range(N_COMPONENTS_TRUE):
        matrix += np.outer(rng.normal(0, 1, N_POS), rng.normal(0, 1, N_AA))

    matrix_centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(matrix_centered, full_matrices=False)
    pos_axes = (u[:, :3] * s[:3]).T
    pos_axes /= np.max(np.abs(pos_axes))
    components = [((u[:, i : i + 1] * s[i]) @ vt[i : i + 1, :]).T for i in range(3)]

    raw_contrib = np.array(
        [
            sum(
                components[k][aa_idx, pos_idx]
                for aa_idx, pos_idx in zip(SELECTED_AA_INDICES, SELECTED_POSITIONS)
            )
            for k in range(3)
        ],
        dtype=float,
    )
    scale = float(
        (raw_contrib @ DISPLAY_CONTRIB_TARGET) / (raw_contrib @ raw_contrib)
    )
    scaled_contrib = raw_contrib * scale
    display_contrib = DISPLAY_CONTRIB_TARGET.copy()
    assigned_axis = int(np.argmin(scaled_contrib))

    return {
        "pos_axes": pos_axes,
        "display_contrib": display_contrib,
        "assigned_axis": assigned_axis,
    }


def render_svd_diagram_second_half(outpath, show_plot=True):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    example = _build_svd_example()
    pos_axes = example["pos_axes"]
    display_contrib = example["display_contrib"]
    assigned_axis = example["assigned_axis"]

    fig = plt.figure(figsize=(7.2, 3.15))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[3.3, 0.18, 2.25],
        height_ratios=[1.0, 1.55],
        wspace=0.03,
        hspace=0.28,
    )

    strip_grid = gs[0, 0].subgridspec(3, 1, hspace=0.44)
    strip_axes = [fig.add_subplot(strip_grid[i, 0]) for i in range(3)]
    labels = [("Axis 1", C1), ("Axis 2", C2), ("Axis 3", C3_TEXT)]

    for idx, ax in enumerate(strip_axes):
        ax.imshow(
            pos_axes[idx][None, :],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        for pos_idx in SELECTED_POSITIONS:
            ax.add_patch(
                Rectangle(
                    (pos_idx - 0.5, -0.5),
                    1.0,
                    1.0,
                    fill=False,
                    lw=1.2,
                    ec=HILITE,
                )
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#999999")
        ax.text(
            0.0,
            1.14,
            labels[idx][0],
            color=labels[idx][1],
            fontsize=7,
            fontweight="bold",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )

    for pos_idx in SELECTED_POSITIONS:
        strip_axes[-1].text(
            pos_idx,
            -0.95,
            f"pos {pos_idx + 1}",
            color=HILITE,
            fontsize=5.5,
            ha="center",
            va="top",
            transform=strip_axes[-1].transData,
        )

    ax_bar = fig.add_subplot(gs[1, 0])
    bar_colors = [C1, C2, C3]
    x = np.arange(3)
    ax_bar.bar(
        x,
        display_contrib,
        width=0.92,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    ax_bar.axhline(0, color="#888888", linewidth=0.6)
    ax_bar.set_ylim(-2.25, 0.3)
    ax_bar.set_xlim(-0.55, 2.55)
    ax_bar.set_yticks([-2.2, -1.4, -0.6, 0.2])
    ax_bar.set_ylabel("Contribution", fontsize=6)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["Axis 1", "Axis 2", "Axis 3"], fontsize=6)
    ax_bar.tick_params(axis="y", labelsize=6, pad=1)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

    value_labels = [f"{val:.2f}" for val in display_contrib]
    for idx, (x_pos, y_val, label) in enumerate(zip(x, display_contrib, value_labels)):
        ax_bar.text(
            x_pos,
            y_val * 0.52,
            label,
            ha="center",
            va="center",
            fontsize=6,
            color="white" if idx < 2 else "#7a5b00",
            fontweight="bold",
        )

    ax_bar.add_patch(
        Rectangle(
            (assigned_axis - 0.52, -2.05),
            1.04,
            1.98,
            fill=False,
            lw=1.4,
            ec="#d2b24c",
        )
    )
    ax_bar.annotate(
        "sum per-axis\ncontributions",
        xy=(1.0, 0.98),
        xycoords="axes fraction",
        xytext=(0.72, 1.14),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=6,
        color="#b3b3b3",
        fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color="#b3b3b3", lw=0.8),
    )
    ax_bar.text(
        0.0,
        -0.26,
        f"Assigned: Axis {assigned_axis + 1}",
        color=labels[assigned_axis][1],
        fontsize=8,
        fontweight="bold",
        transform=ax_bar.transAxes,
    )
    ax_bar.text(
        0.0,
        -0.40,
        "(most negative contribution)",
        color="#b3b3b3",
        fontsize=6,
        transform=ax_bar.transAxes,
    )

    for grid_slot in [gs[0, 1], gs[1, 1]]:
        ax_arrow = fig.add_subplot(grid_slot)
        ax_arrow.axis("off")
        ax_arrow.text(
            0.45,
            0.54,
            "→",
            fontsize=16,
            color="#9b9b9b",
            ha="center",
            va="center",
        )

    ax_graph = fig.add_subplot(gs[:, 2])
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")

    coords = {
        "b0": (-1.75, 0.35),
        "b1": (-1.45, 0.72),
        "b2": (-1.15, 0.45),
        "b3": (-0.75, 0.95),
        "b4": (-0.45, 0.62),
        "b5": (0.10, 0.20),
        "b6": (0.10, -0.20),
        "r0": (0.60, 0.18),
        "r1": (1.05, 0.48),
        "r2": (1.40, 0.82),
        "r3": (1.85, 1.00),
        "r4": (2.08, 0.70),
        "r5": (1.02, -0.12),
        "r6": (1.55, 0.05),
        "y0": (-0.38, -0.72),
        "y1": (0.22, -0.95),
        "y2": (0.78, -0.64),
        "y3": (0.02, -1.34),
    }
    thin_edges = [
        ("b0", "b1"),
        ("b0", "b2"),
        ("b1", "b2"),
        ("b1", "b4"),
        ("b2", "b4"),
        ("b4", "b5"),
        ("b5", "b6"),
        ("r0", "r1"),
        ("r1", "r2"),
        ("r2", "r3"),
        ("r2", "r4"),
        ("r3", "r4"),
        ("r0", "r5"),
        ("r0", "r6"),
        ("r5", "r6"),
        ("y0", "y1"),
        ("y1", "y2"),
        ("y1", "y3"),
        ("y2", "y3"),
        ("y0", "b6"),
        ("y2", "r6"),
    ]
    boundary_edges = [("b5", "r0"), ("b6", "r5")]

    for node_a, node_b in thin_edges:
        xa, ya = coords[node_a]
        xb, yb = coords[node_b]
        ax_graph.plot([xa, xb], [ya, yb], color=EDGE_LIGHT, lw=0.75, zorder=1)
    for node_a, node_b in boundary_edges:
        xa, ya = coords[node_a]
        xb, yb = coords[node_b]
        ax_graph.plot(
            [xa, xb],
            [ya, yb],
            color=EDGE_BOUNDARY,
            lw=2.2,
            zorder=2,
        )

    node_groups = {
        "Axis 1": (["b0", "b1", "b2", "b3", "b4", "b5", "b6"], C1),
        "Axis 2": (["r0", "r1", "r2", "r3", "r4", "r5", "r6"], C2),
        "Axis 3": (["y0", "y1", "y2", "y3"], C3),
    }
    for nodes, color in node_groups.values():
        xs = [coords[node][0] for node in nodes]
        ys = [coords[node][1] for node in nodes]
        ax_graph.scatter(
            xs,
            ys,
            s=180,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    ax_graph.set_xlim(-2.05, 3.25)
    ax_graph.set_ylim(-1.95, 1.25)
    ax_graph.annotate(
        "Boundary edge:\nregime shifts 1 \u2192 2",
        xy=(0.35, 0.19),
        xycoords="data",
        xytext=(1.45, 0.02),
        textcoords="data",
        fontsize=6,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bdbdbd", lw=0.6),
        arrowprops=dict(arrowstyle="-", color="#bdbdbd", lw=0.7),
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=C1,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            label="Axis 1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=C2,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            label="Axis 2",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=C3,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            label="Axis 3",
        ),
        Line2D([0, 1], [0, 0], color=EDGE_BOUNDARY, lw=2.0, label="Boundary"),
        Line2D([0, 1], [0, 0], color=EDGE_LIGHT, lw=0.75, label="Intra-regime"),
    ]
    ax_graph.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.34, -0.13),
        frameon=False,
        fontsize=6,
        ncol=1,
        handlelength=1.3,
        handletextpad=0.6,
    )

    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.20)
    fig.savefig(outpath)
    if show_plot:
        plt.show()
    plt.close(fig)

    return {
        "outpath": outpath,
        "selected_positions": [pos + 1 for pos in SELECTED_POSITIONS],
        "display_contrib": display_contrib,
        "assigned_axis": assigned_axis + 1,
    }
