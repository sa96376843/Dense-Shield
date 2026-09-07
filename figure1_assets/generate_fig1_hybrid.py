#!/usr/bin/env python3
"""Deterministically rebuild DenseShield Fig. 1 from the GPT Image 2 layout.

The overall two-swimlane composition follows the GPT Image 2 concept retained
in this directory. All small components, labels, mathematical notation, and
connectors are generated here as editable vector primitives.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath


OUT = Path(__file__).resolve().parent

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#8E5AA9"
INK = "#202124"
MID = "#5F6368"
PALE = "#FAFBFC"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def rounded(ax, x, y, w, h, edge, radius=0.8, fill=PALE, lw=1.15):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.16,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        joinstyle="round",
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, dashed=False, color=INK, lw=1.15, rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "solid",
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=5,
    )
    ax.add_patch(patch)
    return patch


def stage_box(ax, x, y, w, h, number, color, title, details):
    rounded(ax, x, y, w, h, color)
    ax.add_patch(Circle((x + 1.65, y + h - 1.65), 1.05, facecolor=color, edgecolor="none", zorder=6))
    ax.text(x + 1.65, y + h - 1.65, str(number), color="white", weight="bold", ha="center", va="center", fontsize=8.2, zorder=7)
    ax.text(x + w / 2 + 0.45, y + h - 0.95, f"STAGE {number}", ha="center", va="top", weight="bold", fontsize=7.3)
    ax.text(x + w / 2, y + h - 4.55, title, ha="center", va="center", weight="bold", fontsize=7.2, linespacing=1.05)
    line_y = y + 3.25
    ax.plot([x + 1.1, x + w - 1.1], [line_y, line_y], color=color, linewidth=0.85)
    ax.text(x + w / 2, y + 2.15, details[0], ha="center", va="center", fontsize=6.8)
    if len(details) > 1:
        ax.text(x + w / 2, y + 0.92, details[1], ha="center", va="center", fontsize=6.8)


def database(ax, cx, cy, color=BLUE, scale=1.0):
    w, h = 5.2 * scale, 5.8 * scale
    ax.add_patch(Ellipse((cx, cy + h / 2), w, 1.45 * scale, facecolor="white", edgecolor=color, linewidth=1.2))
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2, cy + h / 2], color=color, linewidth=1.2)
    ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2, cy + h / 2], color=color, linewidth=1.2)
    ax.add_patch(Ellipse((cx, cy - h / 2), w, 1.45 * scale, facecolor="white", edgecolor=color, linewidth=1.2))
    for frac in (-0.05, 0.27):
        yy = cy + h * frac
        ax.add_patch(Ellipse((cx, yy), w, 1.35 * scale, facecolor="none", edgecolor=color, linewidth=0.9))


def file_icon(ax, cx, cy, color=INK, scale=1.0):
    w, h = 4.4 * scale, 5.8 * scale
    x0, y0 = cx - w / 2, cy - h / 2
    fold = 1.35 * scale
    pts = [(x0, y0), (x0, y0 + h), (x0 + w - fold, y0 + h), (x0 + w, y0 + h - fold), (x0 + w, y0), (x0, y0)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=color, linewidth=1.15))
    ax.plot([x0 + w - fold, x0 + w - fold, x0 + w], [y0 + h, y0 + h - fold, y0 + h - fold], color=color, linewidth=1.0)
    for yy in (y0 + 1.4, y0 + 2.6, y0 + 3.8):
        ax.plot([x0 + 0.9, x0 + w - 0.8], [yy, yy], color=color, linewidth=0.9)


def classifier(ax, cx, cy, color=VERMILION, scale=1.0):
    # Shield is a deterministic path; inner nodes depict the MLP backbone.
    verts = [
        (cx, cy + 3.5 * scale),
        (cx - 1.6 * scale, cy + 2.1 * scale),
        (cx - 3.0 * scale, cy + 1.7 * scale),
        (cx - 2.7 * scale, cy - 1.8 * scale),
        (cx, cy - 3.6 * scale),
        (cx + 2.7 * scale, cy - 1.8 * scale),
        (cx + 3.0 * scale, cy + 1.7 * scale),
        (cx + 1.6 * scale, cy + 2.1 * scale),
        (cx, cy + 3.5 * scale),
    ]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * 7 + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="white", edgecolor=color, linewidth=1.35, joinstyle="round"))
    nodes = [(cx - 1.35, cy + 0.85), (cx - 1.35, cy - 1.0), (cx, cy + 1.45), (cx, cy), (cx, cy - 1.6), (cx + 1.35, cy + 0.85), (cx + 1.35, cy - 1.0)]
    edges = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 5), (3, 5), (3, 6), (4, 6)]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=INK, linewidth=0.72)
    for px, py in nodes:
        ax.add_patch(Circle((px, py), 0.34 * scale, facecolor="white", edgecolor=INK, linewidth=0.85, zorder=4))


def main():
    fig, ax = plt.subplots(figsize=(7.16, 3.41))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 47.62)
    ax.axis("off")

    # GPT Image 2 established this two-swimlane composition.
    rounded(ax, 1.0, 25.0, 98.0, 21.3, BLUE, radius=1.0, fill="white", lw=1.25)
    rounded(ax, 1.0, 1.1, 98.0, 22.0, GREEN, radius=1.0, fill="white", lw=1.25)
    ax.text(50, 44.8, "OFFLINE TRAINING & PURIFICATION", color=BLUE, ha="center", va="center", fontsize=9.4, weight="bold")
    ax.text(50, 21.7, "ONLINE INFERENCE & ADAPTATION", color=GREEN, ha="center", va="center", fontsize=9.4, weight="bold")

    # Offline path.
    database(ax, 6.4, 36.1, BLUE, 0.9)
    ax.text(6.4, 31.8, "Training data", ha="center", va="top", fontsize=7.4, weight="bold")
    ax.text(6.4, 30.5, r"$\mathcal{D}_{tr}$", ha="center", va="top", fontsize=7.5)
    stage_box(ax, 14.0, 30.5, 19.0, 12.0, 1, BLUE, "Sparse Feature\nCompression\n(SFC)", (r"rare features $\rightarrow B$ hash bins", r"$\tilde{\mathbf{x}}=\phi(\mathbf{x})$"))
    stage_box(ax, 37.0, 30.5, 20.0, 12.0, 2, GREEN, "Density-Boosted\nAdversarial Training\n(DBAT)", (r"$k$-NN density weights", "weighted min--max training"))
    stage_box(ax, 61.0, 30.5, 19.5, 12.0, 3, PURPLE, "D-PUR Backdoor\nPurification", (r"saliency $\times$ rarity", "counter-example unlearning"))
    classifier(ax, 90.5, 37.1, VERMILION, 0.9)
    ax.text(90.5, 32.0, "Robust classifier", ha="center", va="top", fontsize=7.3, weight="bold")
    ax.text(90.5, 30.7, r"$f_{\theta^{\star}}$", ha="center", va="top", fontsize=7.5)
    arrow(ax, 9.2, 36.8, 13.5, 36.8)
    arrow(ax, 33.4, 36.8, 36.5, 36.8)
    arrow(ax, 57.4, 36.8, 60.5, 36.8)
    arrow(ax, 80.9, 36.8, 86.8, 36.8)

    database(ax, 70.8, 27.7, PURPLE, 0.42)
    ax.text(75.0, 27.8, "1% clean\nhold-out", ha="left", va="center", fontsize=6.9, weight="bold", linespacing=1.0)
    arrow(ax, 70.8, 29.4, 70.8, 30.25, dashed=True, color=PURPLE, lw=1.0)

    # Shared deterministic SFC mapping.
    ax.plot([23.5, 23.5], [31.8, 22.9], color=BLUE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(23.5, 24.4, "shared density-centric representation", color=BLUE, ha="center", va="center", fontsize=7.1,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=BLUE, linewidth=0.8))

    # Online main path and code-native icons.
    file_icon(ax, 6.3, 15.5, INK, 0.9)
    ax.text(6.3, 11.9, "Incoming", ha="center", va="top", fontsize=7.1, weight="bold")
    ax.text(6.3, 10.65, r"sample  $\mathbf{x}$", ha="center", va="top", fontsize=7.1, weight="bold")
    rounded(ax, 14.0, 12.2, 16.0, 7.6, BLUE, fill=PALE)
    ax.text(22.0, 17.3, "SFC mapping", ha="center", va="center", fontsize=8.1, weight="bold")
    ax.text(22.0, 14.5, r"$\tilde{\mathbf{x}}=\phi(\mathbf{x})$", ha="center", va="center", fontsize=8.3)
    classifier(ax, 39.0, 16.2, VERMILION, 0.82)
    ax.text(39.0, 12.25, "Robust classifier", ha="center", va="top", fontsize=6.8, weight="bold")
    ax.text(39.0, 10.75, r"$f_{\theta^{\star}}$", ha="center", va="top", fontsize=7.2)
    rounded(ax, 46.3, 13.5, 10.0, 6.3, BLUE, radius=0.65, fill="white", lw=1.05)
    ax.text(51.3, 18.0, "Prediction", ha="center", va="center", fontsize=7.6, weight="bold")
    ax.text(51.3, 15.6, "benign /\nmalware", ha="center", va="center", fontsize=7.0, linespacing=1.05)
    arrow(ax, 9.0, 16.1, 13.5, 16.1)
    arrow(ax, 30.4, 16.1, 35.8, 16.1)
    arrow(ax, 42.0, 16.7, 45.8, 16.7)

    # Drift branch and adaptation loop.
    stage_box(ax, 50.0, 3.2, 21.0, 10.7, 4, ORANGE, "Uncertainty-Conformal\nDrift Detection", (r"uncertainty $u(\mathbf{x})$", r"batch $p$-value $p_t<\alpha$"))
    stage_box(ax, 77.0, 3.2, 21.0, 10.7, 5, GREEN, "Drift-adaptive\nMinimal-Label Adaptation\n(DMLA)", (r"top-$q$ labels + pseudo-labels", "replay regularization"))
    # Solid classifier-output branch to uncertainty-based detection.
    ax.plot([42.0, 45.0, 45.0], [15.0, 15.0, 8.5], color=INK, linewidth=1.15)
    arrow(ax, 45.0, 8.5, 49.5, 8.5)
    arrow(ax, 71.5, 8.5, 76.5, 8.5)
    ax.text(74.0, 10.0, "drift\nalarm", ha="center", va="center", fontsize=6.1, weight="bold", linespacing=0.95)
    # Dashed feedback is explicitly supervisory/model-update only.
    ax.plot([87.5, 87.5, 32.0, 32.0], [3.0, 2.25, 2.25, 13.8], color=INK, linewidth=1.0, linestyle=(0, (4, 3)))
    arrow(ax, 32.0, 13.8, 36.2, 14.5, dashed=True, lw=1.0)
    ax.text(64.0, 2.25, "model update", ha="center", va="center", fontsize=6.2, weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.3))

    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.012, top=0.988)
    fig.savefig(OUT / "fig1_framework_hybrid.pdf", facecolor="white")
    fig.savefig(OUT / "fig1_framework_hybrid.png", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
