from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


SEED = 42
N_POS = 60
N_AA = 20
N_COMPONENTS_TRUE = 5

AXIS_PALETTE = {
    1: "#4C78A8",
    2: "#F58518",
    3: "#54A24B",
}
INTRA_EDGE_COLOR = "#D0D0D0"
BOUNDARY_EDGE_COLOR = "#111111"
SELECTED_OUTLINE_COLOR = "#111111"

SELECTED_AA_INDEX = 3
SELECTED_POSITION_INDEX = 33

GRAPH_COORDS = {
    "b0": (-1.50, 0.25),
    "b1": (-1.20, 0.55),
    "b2": (-0.95, 0.35),
    "b3": (-0.55, 0.55),
    "b4": (-0.35, 0.18),
    "b5": (-0.40, -0.10),
    "o0": (0.00, 0.15),
    "o1": (0.35, 0.38),
    "o2": (0.70, 0.65),
    "o3": (1.05, 0.82),
    "o4": (1.28, 0.58),
    "o5": (0.42, -0.02),
    "g0": (-0.80, -0.55),
    "g1": (-0.35, -0.72),
    "g2": (0.05, -0.52),
}
GRAPH_INTRA_EDGES = [
    ("b0", "b1"),
    ("b1", "b2"),
    ("b2", "b3"),
    ("b3", "b4"),
    ("b4", "b5"),
    ("o0", "o1"),
    ("o1", "o2"),
    ("o2", "o3"),
    ("o3", "o4"),
    ("o0", "o5"),
    ("g0", "g1"),
    ("g1", "g2"),
    ("g2", "o5"),
    ("g0", "b5"),
]
GRAPH_BOUNDARY_EDGES = [
    ("b4", "o0"),
    ("b5", "o5"),
]
GRAPH_NODE_GROUPS = {
    1: ["b0", "b1", "b2", "b3", "b4", "b5"],
    2: ["o0", "o1", "o2", "o3", "o4", "o5"],
    3: ["g0", "g1", "g2"],
}


def _build_example_data():
    rng = np.random.default_rng(SEED)
    matrix = np.zeros((N_POS, N_AA), dtype=float)
    for _ in range(N_COMPONENTS_TRUE):
        matrix += np.outer(rng.normal(0, 1, N_POS), rng.normal(0, 1, N_AA))

    matrix_centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(matrix_centered, full_matrices=False)
    components = [
        ((u[:, i : i + 1] * s[i]) @ vt[i : i + 1, :]).T
        for i in range(3)
    ]
    component_stack = np.stack(components, axis=0)
    component_vmax = float(np.max(np.abs(component_stack)) * 0.9)
    component_norm = TwoSlopeNorm(
        vmin=-component_vmax,
        vcenter=0.0,
        vmax=component_vmax,
    )

    selected_values = np.array(
        [
            component[SELECTED_AA_INDEX, SELECTED_POSITION_INDEX]
            for component in components
        ],
        dtype=float,
    )
    selected_row_series = [
        component[SELECTED_AA_INDEX, :].copy()
        for component in components
    ]

    return {
        "matrix_t": matrix_centered.T,
        "components": components,
        "component_norm": component_norm,
        "selected_values": selected_values,
        "selected_row_series": selected_row_series,
        "selected_amino_acid_index": SELECTED_AA_INDEX,
        "selected_position_index": SELECTED_POSITION_INDEX,
    }


def _style_matrix_axis(ax):
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def _add_selected_outline(ax, x_idx, y_idx, linewidth=2.0):
    ax.add_patch(
        Rectangle(
            (x_idx - 0.5, y_idx - 0.5),
            1.0,
            1.0,
            fill=False,
            linewidth=linewidth,
            edgecolor=SELECTED_OUTLINE_COLOR,
        )
    )


def render_svd_diagram_first_half(outpath, show_plot=True):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    example = _build_example_data()
    matrix_t = example["matrix_t"]
    components = example["components"]
    selected_aa_index = example["selected_amino_acid_index"]
    selected_position_index = example["selected_position_index"]
    component_norm = example["component_norm"]

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[2.55, 0.24, 2.55],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.18,
        hspace=0.20,
    )

    ax_matrix = fig.add_subplot(gs[0, 0])
    ax_matrix.imshow(
        matrix_t,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax_matrix.set_title("M", fontsize=10, fontweight="bold", pad=4)
    _style_matrix_axis(ax_matrix)
    _add_selected_outline(
        ax_matrix,
        selected_position_index,
        selected_aa_index,
        linewidth=2.2,
    )

    ax_equal = fig.add_subplot(gs[:, 1])
    ax_equal.axis("off")
    ax_equal.text(
        0.5,
        0.5,
        "=",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )

    for axis_idx, component in enumerate(components, start=1):
        ax = fig.add_subplot(gs[axis_idx - 1, 2])
        ax.imshow(
            component,
            aspect="auto",
            cmap="RdBu_r",
            norm=component_norm,
            interpolation="nearest",
        )
        ax.set_title(
            f"Axis {axis_idx}",
            fontsize=9,
            fontweight="bold",
            color=AXIS_PALETTE[axis_idx],
            pad=4,
        )
        _style_matrix_axis(ax)
        _add_selected_outline(
            ax,
            selected_position_index,
            selected_aa_index,
            linewidth=2.2,
        )

    fig.subplots_adjust(left=0.05, right=0.985, top=0.92, bottom=0.08)
    fig.savefig(outpath)
    if show_plot:
        plt.show()
    plt.close(fig)

    return {
        "outpath": outpath,
        "selected_position": selected_position_index + 1,
        "selected_amino_acid_index": selected_aa_index + 1,
        "selected_values": example["selected_values"].copy(),
    }


def render_svd_diagram_second_half(outpath, show_plot=True):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    example = _build_example_data()
    selected_row_series = example["selected_row_series"]
    component_norm = example["component_norm"]
    selected_values = example["selected_values"]
    selected_position_index = example["selected_position_index"]

    fig = plt.figure(figsize=(7.2, 2.8))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[3.05, 0.18, 1.95],
        height_ratios=[0.72, 1.30],
        wspace=0.05,
        hspace=0.25,
    )

    strip_grid = gs[0, 0].subgridspec(3, 1, hspace=0.18)
    for axis_idx, row_series in enumerate(selected_row_series, start=1):
        ax = fig.add_subplot(strip_grid[axis_idx - 1, 0])
        ax.imshow(
            row_series[None, :],
            aspect="auto",
            cmap="RdBu_r",
            norm=component_norm,
            interpolation="nearest",
        )
        _add_selected_outline(ax, selected_position_index, 0, linewidth=1.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#999999")

    ax_bar = fig.add_subplot(gs[1, 0])
    x = np.arange(3)
    ax_bar.bar(
        x,
        selected_values,
        width=0.92,
        color=[AXIS_PALETTE[1], AXIS_PALETTE[2], AXIS_PALETTE[3]],
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    ax_bar.axhline(0.0, color="#888888", linewidth=0.6, zorder=2)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax_bar.set_xlim(-0.55, 2.55)
    ax_bar.set_xticks([])
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_ylim(float(np.min(selected_values) * 1.10), 0.15)

    for idx, (x_pos, y_val) in enumerate(zip(x, selected_values), start=1):
        ax_bar.text(
            x_pos,
            y_val * 0.52,
            f"{y_val:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white" if idx < 3 else "black",
            zorder=4,
        )

    for slot in [gs[0, 1], gs[1, 1]]:
        ax_arrow = fig.add_subplot(slot)
        ax_arrow.axis("off")
        ax_arrow.text(
            0.5,
            0.5,
            "→",
            ha="center",
            va="center",
            fontsize=18,
            color="#999999",
        )

    ax_graph = fig.add_subplot(gs[:, 2])
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")

    for node_a, node_b in GRAPH_INTRA_EDGES:
        xa, ya = GRAPH_COORDS[node_a]
        xb, yb = GRAPH_COORDS[node_b]
        ax_graph.plot(
            [xa, xb],
            [ya, yb],
            color=INTRA_EDGE_COLOR,
            linewidth=0.8,
            zorder=1,
        )

    for node_a, node_b in GRAPH_BOUNDARY_EDGES:
        xa, ya = GRAPH_COORDS[node_a]
        xb, yb = GRAPH_COORDS[node_b]
        ax_graph.plot(
            [xa, xb],
            [ya, yb],
            color=BOUNDARY_EDGE_COLOR,
            linewidth=2.2,
            zorder=2,
        )

    for axis_idx, nodes in GRAPH_NODE_GROUPS.items():
        ax_graph.scatter(
            [GRAPH_COORDS[node][0] for node in nodes],
            [GRAPH_COORDS[node][1] for node in nodes],
            s=180,
            c=AXIS_PALETTE[axis_idx],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    ax_graph.set_xlim(-1.72, 1.48)
    ax_graph.set_ylim(-0.95, 1.00)

    fig.subplots_adjust(left=0.05, right=0.985, top=0.95, bottom=0.12)
    fig.savefig(outpath)
    if show_plot:
        plt.show()
    plt.close(fig)

    return {
        "outpath": outpath,
        "selected_position": selected_position_index + 1,
        "selected_amino_acid_index": example["selected_amino_acid_index"] + 1,
        "selected_values": selected_values.copy(),
    }
