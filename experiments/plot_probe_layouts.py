#!/usr/bin/env python3
"""Generate a publication-quality multi-panel figure comparing probe electrode layouts
across the four benchmark datasets in the STAR-Mem project:
1. Hybrid Janelia (16-ch Staggered Silicon Probe)
2. MEArec (32-ch Tri-Column Poly3 Shank)
3. Yger 2D MEA (252-ch 16x16 Planar Grid)
4. Neuropixels 1.0 (384-ch Ultra-Dense Linear Shank: Macro & Micro Views)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Set publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
})

def main():
    # 1. Hybrid Janelia (16 channels)
    geom_hj = np.array([
        [43., 80.], [11., 80.], [57., 100.], [27., 100.],
        [43., 120.], [11., 120.], [57., 140.], [27., 140.],
        [43., 160.], [11., 160.], [57., 180.], [27., 180.],
        [43., 200.], [11., 200.], [57., 220.], [27., 220.]
    ])
    geom_hj_disp = geom_hj.copy()
    geom_hj_disp[:, 0] -= geom_hj[:, 0].min()
    geom_hj_disp[:, 1] -= geom_hj[:, 1].min()

    # 2. MEArec (32 channels)
    geom_mea = np.array([
        [-18., -117.1875], [-18., -92.1875], [-18., -67.1875], [-18., -42.1875],
        [-18., -17.1875], [-18., 7.8125], [-18., 32.8125], [-18., 57.8125],
        [-18., 82.8125], [-18., 107.8125],
        [0., -129.6875], [0., -104.6875], [0., -79.6875], [0., -54.6875],
        [0., -29.6875], [0., -4.6875], [0., 20.3125], [0., 45.3125],
        [0., 70.3125], [0., 95.3125], [0., 120.3125], [0., 145.3125],
        [18., -117.1875], [18., -92.1875], [18., -67.1875], [18., -42.1875],
        [18., -17.1875], [18., 7.8125], [18., 32.8125], [18., 57.8125],
        [18., 82.8125], [18., 107.8125]
    ])
    geom_mea_disp = geom_mea.copy()
    geom_mea_disp[:, 1] -= geom_mea[:, 1].min()

    # 3. Yger 2D MEA (252 channels)
    x_grid, y_grid = np.meshgrid(np.arange(16) * 30.0, np.arange(16) * 30.0)
    all_yger = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    corner_indices = [0, 15, 240, 255]
    mask_yger = np.ones(256, dtype=bool)
    mask_yger[corner_indices] = False
    geom_yger = all_yger[mask_yger]
    geom_yger_corners = all_yger[corner_indices]

    # 4. Neuropixels 1.0 (384 channels)
    np_cols = np.tile([0.0, 32.0], 192)
    np_rows = np.repeat(np.arange(192) * 20.0, 2)
    geom_np = np.column_stack([np_cols, np_rows])

    # Create figure
    fig = plt.figure(figsize=(16.8, 8.4), dpi=300)
    gs = GridSpec(1, 5, width_ratios=[1.15, 1.35, 2.5, 0.65, 1.35], wspace=0.36,
                 left=0.045, right=0.975, top=0.88, bottom=0.25)

    # Palette
    c_shank = "#ECEFF1"
    c_shank_border = "#90A4AE"
    c_hj = "#1F77B4"
    c_mea = "#2CA02C"
    c_yger = "#7D3C98"
    c_np = "#D35400"
    c_pad_edge = "#212F3D"

    # ==========================================
    # Panel (a): Hybrid Janelia (16-ch)
    # ==========================================
    ax_hj = fig.add_subplot(gs[0])
    
    # Shank outline
    shank_hj = patches.Polygon([
        [-10, -22], [56, -22], [56, 155], [23, 185], [-10, 155]
    ], closed=True, facecolor=c_shank, edgecolor=c_shank_border, linewidth=1.2, zorder=1)
    ax_hj.add_patch(shank_hj)

    # Local neighborhood highlight around channel 6 (y=60)
    k7_indices_hj = [2, 3, 4, 5, 6, 7, 8]
    k7_hull = patches.Polygon([
        [-6, 12], [52, 12], [52, 92], [-6, 92]
    ], closed=True, facecolor="#FADBD8", edgecolor="#E74C3C", linewidth=1.1, linestyle="--", alpha=0.45, zorder=2)
    ax_hj.add_patch(k7_hull)

    # Electrodes
    for i, (x, y) in enumerate(geom_hj_disp):
        color = "#C0392B" if i == 6 else (c_hj if i in k7_indices_hj else "#5DADE2")
        circ = patches.Circle((x, y), radius=5.2, facecolor=color, edgecolor=c_pad_edge, linewidth=0.8, zorder=3)
        ax_hj.add_patch(circ)
        ax_hj.text(x, y, str(i), fontsize=5.8, color="white", ha="center", va="center", weight="bold", zorder=4)

    # Dimension annotations
    ax_hj.annotate("", xy=(58, 20), xytext=(58, 40),
                   arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.1))
    ax_hj.text(62, 30, r"$\Delta y = 20\,\mu\mathrm{m}$", fontsize=8, color="#C0392B", va="center")

    ax_hj.annotate("", xy=(0, -10), xytext=(46, -10),
                   arrowprops=dict(arrowstyle="<->", color="#2C3E50", lw=1.1))
    ax_hj.text(23, -17, r"$W = 46\,\mu\mathrm{m}$", fontsize=8, color="#2C3E50", ha="center")

    ax_hj.text(23, 98, "K=7 Neighborhood", fontsize=7.2, color="#C0392B", weight="bold", ha="center", zorder=5)

    ax_hj.set_xlim(-18, 102)
    ax_hj.set_ylim(-30, 195)
    ax_hj.set_aspect("equal")
    ax_hj.set_title("(a) Hybrid Janelia\n16-Ch Staggered Shank", fontsize=10.5, weight="bold", pad=8)
    ax_hj.set_xlabel(r"$X\;(\mu\mathrm{m})$")
    ax_hj.set_ylabel(r"$Y\;(\mu\mathrm{m})$")
    ax_hj.grid(True, linestyle="--", alpha=0.35)

    # ==========================================
    # Panel (b): MEArec Poly3 (32-ch)
    # ==========================================
    ax_mea = fig.add_subplot(gs[1])

    # Shank outline
    shank_mea = patches.Polygon([
        [-30, -20], [30, -20], [30, 290], [0, 325], [-30, 290]
    ], closed=True, facecolor=c_shank, edgecolor=c_shank_border, linewidth=1.2, zorder=1)
    ax_mea.add_patch(shank_mea)

    # Local spatial neighborhood highlight
    k_mea_hull = patches.Polygon([
        [-24, 110], [24, 110], [24, 215], [-24, 215]
    ], closed=True, facecolor="#D5F5E3", edgecolor="#27AE60", linewidth=1.1, linestyle="--", alpha=0.45, zorder=2)
    ax_mea.add_patch(k_mea_hull)

    # Electrodes (poly3 pads ~11x11 um)
    for i, (x, y) in enumerate(geom_mea_disp):
        rect = patches.Rectangle((x - 5.5, y - 5.5), 11, 11, facecolor=c_mea, edgecolor=c_pad_edge, linewidth=0.8, zorder=3)
        ax_mea.add_patch(rect)
        if i in [0, 10, 16, 21, 22, 31] or i % 3 == 0:
            ax_mea.text(x, y, str(i), fontsize=5.2, color="white", ha="center", va="center", weight="bold", zorder=4)

    # Pitch annotations
    ax_mea.annotate("", xy=(25, 42.5), xytext=(25, 67.5),
                    arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.1))
    ax_mea.text(28, 55, r"$\Delta y = 25\,\mu\mathrm{m}$", fontsize=8, color="#C0392B", va="center")

    ax_mea.annotate("", xy=(-18, -10), xytext=(18, -10),
                    arrowprops=dict(arrowstyle="<->", color="#2C3E50", lw=1.1))
    ax_mea.text(0, -17, r"$\Delta x = 18\,\mu\mathrm{m}$", fontsize=8, color="#2C3E50", ha="center")

    ax_mea.set_xlim(-38, 75)
    ax_mea.set_ylim(-30, 335)
    ax_mea.set_aspect("equal")
    ax_mea.set_title("(b) MEArec Poly3\n32-Ch Tri-Column", fontsize=10.5, weight="bold", pad=8)
    ax_mea.set_xlabel(r"$X\;(\mu\mathrm{m})$")
    ax_mea.set_ylabel(r"$Y\;(\mu\mathrm{m})$")
    ax_mea.grid(True, linestyle="--", alpha=0.35)

    # ==========================================
    # Panel (c): Yger 2D MEA (252-ch)
    # ==========================================
    ax_yger = fig.add_subplot(gs[2])

    # Substrate outline
    sub_yger = patches.Rectangle((-18, -18), 486, 486, facecolor="#FBF8FC", edgecolor="#AF7AC5", linewidth=1.2, zorder=1)
    ax_yger.add_patch(sub_yger)

    # Active electrode grid
    for x, y in geom_yger:
        circ = patches.Circle((x, y), radius=6.2, facecolor=c_yger, edgecolor=c_pad_edge, linewidth=0.6, zorder=3)
        ax_yger.add_patch(circ)

    # Omitted corner electrodes
    for x, y in geom_yger_corners:
        circ_c = patches.Circle((x, y), radius=6.2, facecolor="white", edgecolor="#BDC3C7", linestyle=":", linewidth=1.0, zorder=2)
        ax_yger.add_patch(circ_c)
        ax_yger.text(x, y, "✕", fontsize=6.5, color="#95A5A6", ha="center", va="center", zorder=3)

    # Search ball illustration (R=70um candidate ball around center)
    center_pt = (210.0, 210.0)
    search_ball = patches.Circle(center_pt, radius=70.0, facecolor="#F39C12", alpha=0.22, edgecolor="#D35400", linewidth=1.5, linestyle="--", zorder=4)
    ax_yger.add_patch(search_ball)
    ax_yger.plot(center_pt[0], center_pt[1], "r*", markersize=9, zorder=5)
    ax_yger.text(center_pt[0]+12, center_pt[1]+12, r"$\mathbf{COM}\;(R=70\,\mu\mathrm{m})$", fontsize=8, color="#B9770E", weight="bold", zorder=5)

    # Pitch annotations
    ax_yger.annotate("", xy=(30, 465), xytext=(60, 465),
                     arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.1))
    ax_yger.text(45, 475, r"$\Delta x = 30\,\mu\mathrm{m}$", fontsize=8, color="#C0392B", ha="center")

    ax_yger.annotate("", xy=(465, 30), xytext=(465, 60),
                     arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.1))
    ax_yger.text(476, 45, r"$\Delta y = 30\,\mu\mathrm{m}$", fontsize=8, color="#C0392B", va="center", rotation=270)

    ax_yger.set_xlim(-30, 520)
    ax_yger.set_ylim(-30, 505)
    ax_yger.set_aspect("equal")
    ax_yger.set_title("(c) Yger In-Vitro Retina MEA\n252-Ch 2D Planar Grid (16 × 16)", fontsize=10.5, weight="bold", pad=8)
    ax_yger.set_xlabel(r"$X\;(\mu\mathrm{m})$")
    ax_yger.set_ylabel(r"$Y\;(\mu\mathrm{m})$")
    ax_yger.grid(True, linestyle="--", alpha=0.35)

    # ==========================================
    # Panel (d1): Neuropixels 1.0 Full Shank (384-ch)
    # ==========================================
    ax_np_full = fig.add_subplot(gs[3])

    # Draw full shank
    shank_np_full = patches.Polygon([
        [-6, -40], [38, -40], [38, 3850], [16, 3920], [-6, 3850]
    ], closed=True, facecolor=c_shank, edgecolor=c_shank_border, linewidth=0.9, zorder=1)
    ax_np_full.add_patch(shank_np_full)

    # Dense points
    ax_np_full.scatter(geom_np[:, 0], geom_np[:, 1], s=0.8, color=c_np, alpha=0.7, zorder=2)

    # Zoom window highlight
    zoom_rect = patches.Rectangle((-12, 100), 56, 200, facecolor="#FAD7A0", edgecolor="#E67E22", linewidth=1.2, linestyle="--", zorder=3)
    ax_np_full.add_patch(zoom_rect)
    ax_np_full.text(16, 330, "Zoom", fontsize=7.5, color="#D35400", weight="bold", ha="center")

    ax_np_full.set_xlim(-25, 55)
    ax_np_full.set_ylim(-100, 4000)
    ax_np_full.set_title("(d1) NP 1.0\nFull Shank", fontsize=9.5, weight="bold", pad=8)
    ax_np_full.set_xlabel(r"$X\,(\mu\mathrm{m})$", fontsize=8)
    ax_np_full.set_ylabel(r"$Y\,(\mu\mathrm{m})$", fontsize=8)
    ax_np_full.set_yticks([0, 1000, 2000, 3000, 3800])
    ax_np_full.set_yticklabels(["0", "1 mm", "2 mm", "3 mm", "3.8 mm"], fontsize=7.5)

    # ==========================================
    # Panel (d2): Neuropixels 1.0 Zoom-in (Checkered Layout)
    # ==========================================
    ax_np_zoom = fig.add_subplot(gs[4])

    # Shank segment
    shank_np_zoom = patches.Rectangle((-12, 90), 56, 220, facecolor=c_shank, edgecolor=c_shank_border, linewidth=1.2, zorder=1)
    ax_np_zoom.add_patch(shank_np_zoom)

    # Plot electrodes in the zoom range [100, 300]
    mask_np_zoom = (geom_np[:, 1] >= 100) & (geom_np[:, 1] <= 300)
    for i, (x, y) in enumerate(geom_np[mask_np_zoom]):
        rect = patches.Rectangle((x - 6, y - 6), 12, 12, facecolor=c_np, edgecolor=c_pad_edge, linewidth=0.8, zorder=3)
        ax_np_zoom.add_patch(rect)
        ax_np_zoom.text(x, y, f"c{int(x//32)}", fontsize=5.8, color="white", ha="center", va="center", weight="bold", zorder=4)

    # Pitch annotations
    ax_np_zoom.annotate("", xy=(32, 160), xytext=(32, 180),
                        arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.1))
    ax_np_zoom.text(35, 170, r"$\Delta y = 20\,\mu\mathrm{m}$", fontsize=8, color="#C0392B", va="center")

    ax_np_zoom.annotate("", xy=(0, 115), xytext=(32, 115),
                        arrowprops=dict(arrowstyle="<->", color="#2C3E50", lw=1.1))
    ax_np_zoom.text(16, 107, r"$\Delta x = 32\,\mu\mathrm{m}$", fontsize=8, color="#2C3E50", ha="center")

    # Pad size annotation
    ax_np_zoom.text(0, 240, "$12\\times 12\,\mu\mathrm{m}^2$\nPad", fontsize=7.2, color="#7B241C", ha="center", va="center",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="#FDEDEC", edgecolor="#E74C3C", lw=0.6))

    ax_np_zoom.set_xlim(-18, 70)
    ax_np_zoom.set_ylim(90, 310)
    ax_np_zoom.set_aspect("equal")
    ax_np_zoom.set_title("(d2) NP 1.0 Detail\nCheckered Stagger", fontsize=10.5, weight="bold", pad=8)
    ax_np_zoom.set_xlabel(r"$X\;(\mu\mathrm{m})$")
    ax_np_zoom.set_ylabel(r"$Y\;(\mu\mathrm{m})$")
    ax_np_zoom.grid(True, linestyle="--", alpha=0.35)

    # Force drawing to resolve positions
    fig.canvas.draw()

    # Align all bottom info cards perfectly along figure coordinates
    axes_list = [ax_hj, ax_mea, ax_yger, ax_np_full, ax_np_zoom]
    info_texts = [
        ("• Channels: 16\n• 4 Staggered Cols\n• Pitch: $\Delta y=20\,\mu\mathrm{m}$\n• Span: $46\\times 140\,\mu\mathrm{m}$\n• Linear SiProbe", "#EBF5FB", c_hj, 8),
        ("• Channels: 32\n• 3-Col Hexagonal\n• Pitch: $25\\times 18\,\mu\mathrm{m}$\n• Span: $36\\times 275\,\mu\mathrm{m}$\n• Neuronexus Poly3", "#EAFAF1", c_mea, 8),
        ("• Channels: 252\n• Uniform 2D Grid\n• Pitch: $\Delta x=\Delta y=30\,\mu\mathrm{m}$\n• Span: $450\\times 450\,\mu\mathrm{m}$\n• Planar MEA Array", "#F5EEF8", c_yger, 8),
        ("• 384 AP Channels\n• Span: $3.82\,\mathrm{mm}$\n• Pitch: $20\,\mu\mathrm{m}$", "#FBEEE6", c_np, 7.5),
        ("• Channels: 384\n• Checkered 2-Col\n• Pitch: $20\\times 32\,\mu\mathrm{m}$\n• Pad: $12\\times 12\,\mu\mathrm{m}^2$\n• Penetrating Shank", "#FBEEE6", c_np, 8),
    ]

    for ax, (text, bg_col, edge_col, fsize) in zip(axes_list, info_texts):
        bbox = ax.get_position()
        cx = bbox.x0 + bbox.width / 2.0
        fig.text(cx, 0.17, text, transform=fig.transFigure, fontsize=fsize, linespacing=1.35,
                 ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.45", facecolor=bg_col, edgecolor=edge_col, alpha=0.95))

    # Overall title
    fig.suptitle("High-Density Neural Probe Architectures and Electrode Layouts Across Benchmark Datasets",
                 fontsize=13, weight="bold", y=0.96)

    # Save outputs
    out_paths = [
        Path("/home/xinyuan/SNN_SpikeSorting/DAC2027/figures/dataset_probe_layouts.pdf"),
        Path("/home/xinyuan/SNN_SpikeSorting/DAC2027/figures/dataset_probe_layouts.png"),
        Path("/home/xinyuan/SNN_SpikeSorting/Spatial/docs/figures/dataset_probe_layouts.pdf"),
        Path("/home/xinyuan/SNN_SpikeSorting/Spatial/docs/figures/dataset_probe_layouts.png"),
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, format=p.suffix[1:], dpi=300)
        print(f"Saved: {p}")

    plt.close()

if __name__ == "__main__":
    main()
