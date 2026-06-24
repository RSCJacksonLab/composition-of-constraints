from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


C_RED = "#D64545"
C_BLUE = "#2B5EA7"
C_GOLD = "#E8A838"
C_BLACK = "#1a1a1a"
C_BG = "#F5F5F0"
C_GREY = "#A8A8A0"
C_LIGHT = "#F2F2EC"

RGB_RED = np.array([214, 69, 69]) / 255.0
RGB_BLUE = np.array([43, 94, 167]) / 255.0


def f1_2d(x, y):
    r2 = (x - 0.38) ** 2 * 1.2 + (y - 0.58) ** 2 * 0.85
    return np.exp(-5.0 * r2)


def f2_2d(x, y):
    r2 = (x - 0.68) ** 2 * 0.85 + (y - 0.38) ** 2 * 1.2
    return np.exp(-5.0 * r2)


def grid_graph(n=7):
    nodes = [(i, j) for i in range(n) for j in range(n)]
    positions = {(i, j): (i / (n - 1), j / (n - 1)) for i, j in nodes}
    edges = []
    for i, j in nodes:
        if i + 1 < n:
            edges.append(((i, j), (i + 1, j)))
        if j + 1 < n:
            edges.append(((i, j), (i, j + 1)))
    return nodes, edges, positions


def classify_nodes(nodes, positions, cutoff):
    values = {
        node: min(f1_2d(*positions[node]), f2_2d(*positions[node]))
        for node in nodes
    }
    return {node for node, value in values.items() if value >= cutoff}, values


def edge_classes(edges, s_nodes):
    s_nodes = set(s_nodes)
    internal = []
    boundary = []
    outside = []
    for u, v in edges:
        in_u = u in s_nodes
        in_v = v in s_nodes
        if in_u and in_v:
            internal.append((u, v))
        elif in_u != in_v:
            boundary.append((u, v))
        else:
            outside.append((u, v))
    volume = 2 * len(internal) + len(boundary)
    ratio = len(boundary) / volume if volume else np.nan
    return internal, boundary, outside, volume, ratio


def draw_fitness_background(ax, cutoff):
    res = 300
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y)
    F1 = f1_2d(X, Y)
    F2 = f2_2d(X, Y)
    F_min = np.minimum(F1, F2)
    viable = F_min >= cutoff
    limiter = np.argmin(np.stack([F1, F2], axis=-1), axis=-1)

    img = np.ones((res, res, 4))
    img[..., :3] = np.array([245, 245, 240]) / 255.0
    img[..., 3] = 1.0

    for idx, rgb_base in enumerate([RGB_RED, RGB_BLUE]):
        mask = viable & (limiter == idx)
        brightness = np.clip(0.58 + 0.42 * (F_min - cutoff) / max(1e-9, 1 - cutoff), 0, 1)
        for channel in range(3):
            img[..., channel] = np.where(
                mask,
                rgb_base[channel] * brightness + (1 - brightness) * 1.0,
                img[..., channel],
            )
    img[viable, 3] = 0.28

    ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], interpolation="bilinear", zorder=0)
    ax.contour(X, Y, F_min, levels=[cutoff], colors=[C_BLACK], linewidths=0.85, zorder=1)


def draw_partition(ax, nodes, edges, positions, cutoff, title, subtitle):
    s_nodes, _ = classify_nodes(nodes, positions, cutoff)
    internal, boundary, outside, volume, ratio = edge_classes(edges, s_nodes)

    ax.set_aspect("equal")
    ax.set_facecolor(C_BG)
    draw_fitness_background(ax, cutoff)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor(C_BLACK)

    for edge_group, color, lw, alpha, zorder in [
        (outside, C_GREY, 0.65, 0.42, 2),
        (internal, C_BLUE, 1.9, 0.92, 3),
        (boundary, C_GOLD, 2.2, 0.96, 4),
    ]:
        for u, v in edge_group:
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha, zorder=zorder)

    for node in nodes:
        x, y = positions[node]
        if node in s_nodes:
            ax.scatter(x, y, s=30, color=C_BLUE, edgecolor="white", linewidth=0.55, zorder=6)
        else:
            ax.scatter(x, y, s=21, color=C_LIGHT, edgecolor="#9D9D96", linewidth=0.5, zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_title(title, fontsize=8.7, pad=4)
    ax.text(
        0.5,
        -0.13,
        subtitle + f"\n$|\\partial S|/\\mathrm{{vol}}(S)={len(boundary)}/{volume}={ratio:.2f}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        linespacing=1.12,
    )


def main():
    nodes, edges, positions = grid_graph(n=7)
    panels = [
        (0.25, r"$f_{\min}=0.25$", "large $S$; low boundary/volume"),
        (0.65, r"$f_{\min}=0.65$", "small $S$; boundary-dominated"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(3.85, 1.82))
    for ax, (cutoff, title, subtitle) in zip(axes, panels):
        draw_partition(ax, nodes, edges, positions, cutoff, title, subtitle)

    legend_handles = [
        Line2D([0], [0], color=C_BLUE, lw=2.0, label="$S \\leftrightarrow S$"),
        Line2D([0], [0], color=C_GOLD, lw=2.2, label="$S \\leftrightarrow \\hat{S}$"),
        Line2D([0], [0], color=C_BLACK, lw=0.9, label="$f=f_{\\min}$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.0,
        bbox_to_anchor=(0.55, -0.035),
        handlelength=1.5,
        columnspacing=1.05,
    )
    fig.text(0.018, 0.93, "c", ha="left", va="top", fontsize=20, fontweight="bold", color=C_BLACK)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.34, wspace=0.22)
    out_dir = Path("../figures/figure_3")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "boundary_volume_schematic_panel_c.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "boundary_volume_schematic_panel_c.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
