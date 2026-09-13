import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

HEX_R = 1.3     # radius of the background hexagon (Polanyi's field)
INNER_R = 0.55  # radius of the inner n-shape (the task's centres)

def hexagon_positions():
    """The six points of Polanyi's Fig. 1/2, vertex at top (nail) and bottom (weight)."""
    angles = np.radians(90 + np.arange(6) * 60)
    return HEX_R * np.column_stack([np.cos(angles), np.sin(angles)])

def inner_positions(n):
    """Vertices of the n centres, at the same centre as the hexagon."""
    if n == 1:
        return np.array([[0.0, 0.0]])
    angle_start = {2: 0, 3: 90, 4: 45}[n]
    angles = np.radians(angle_start + np.arange(n) * 360 / n)
    return INNER_R * np.column_stack([np.cos(angles), np.sin(angles)])

def inner_edges(n):
    """Edges for the inner n-shape, split into primary (perimeter) and
    background (diagonals). n<=3 has no separate diagonal -- the triangle's
    perimeter already is its complete graph."""
    if n <= 3:
        return list(combinations(range(n), 2)), []
    primary = [(i, (i + 1) % n) for i in range(n)]
    background = [(i, j) for i, j in combinations(range(n), 2) if (i, j) not in primary and (j, i) not in primary]
    return primary, background

hex_pts = hexagon_positions()

fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))

for ax, n in zip(axes, [1, 2, 3, 4]):

    # background: Polanyi's hexagon, all diagonals -- same as Fig. 1/2 in the original
    for i, j in combinations(range(6), 2):
        ax.plot([hex_pts[i, 0], hex_pts[j, 0]], [hex_pts[i, 1], hex_pts[j, 1]],
                color='0.55', linewidth=1.0, zorder=1)
    # mark the hexagon's own vertices (small, so the inner n-shape still reads as the focus)
    ax.scatter(hex_pts[:, 0], hex_pts[:, 1], s=45, facecolor='white',
               edgecolor='black', linewidth=1.2, zorder=2)

    # the n centres, plotted at the centre of the hexagon
    pts = inner_positions(n)
    primary, background = inner_edges(n)
    # background diagonals: same style as the hexagon's own lines
    for i, j in background:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color='0.55', linewidth=1.0, zorder=2)
    # primary perimeter: bold, the active shape
    for i, j in primary:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color='black', linewidth=1.4, zorder=3)

    # highlight the vertices/nodes of the n-centre system: plain filled dots, no outline
    ax.scatter(pts[:, 0], pts[:, 1], s=140, facecolor='black', edgecolor='none', zorder=5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.85, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    # labels placed manually underneath, since axis('off') would hide a normal xlabel
    label = {1: "Monocentric", 2: "Dyadic", 3: "Polycentric", 4: "Polycentric"}[n]
    ax.text(0, -1.62, f"n = {n}", ha='center', va='top', fontsize=13, fontweight='bold')
    ax.text(0, -1.82, label, ha='center', va='top', fontsize=11, style='italic', color='0.25')

import os

plt.tight_layout()

# Save next to this script, regardless of where it's run from -- no hardcoded
# sandbox path, so this works when run locally.
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polycentricity_n_series.png')
plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches='tight')
print(f"saved to {OUTPUT_PATH}")

# Preview in a window when run locally (no-ops safely if no display is available).
plt.show()
