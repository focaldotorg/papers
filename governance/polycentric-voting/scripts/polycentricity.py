import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import combinations

def hexagon_positions():
    angles = np.radians(90 + np.arange(6) * 60)
    return 1.3 * np.column_stack([np.cos(angles), np.sin(angles)])

def inner_positions(n):
    if n == 1:
        return np.array([[0.0, 0.0]])
    angle_start = {2: 0, 3: 90, 4: 45}[n]
    angles = np.radians(angle_start + np.arange(n) * 360 / n)
    return 0.55 * np.column_stack([np.cos(angles), np.sin(angles)])

def inner_edges(n):
    if n <= 3:
        return list(combinations(range(n), 2)), []
    primary = [(i, (i + 1) % n) for i in range(n)]
    background = [(i, j) for i, j in combinations(range(n), 2) if (i, j) not in primary and (j, i) not in primary]
    return primary, background

hex_pts = hexagon_positions()

fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))

for ax, n in zip(axes, [1, 2, 3, 4]):
    for i, j in combinations(range(6), 2):
        ax.plot([hex_pts[i, 0], hex_pts[j, 0]], [hex_pts[i, 1], hex_pts[j, 1]],
                color='0.55', linewidth=1.0, zorder=1)
    
    ax.scatter(hex_pts[:, 0], hex_pts[:, 1], s=45, facecolor='0.55',
               edgecolor='none', zorder=2)

    pts = inner_positions(n)
    primary, background = inner_edges(n)

    for i, j in background:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color='1', linewidth=1.0, zorder=2)
    for i, j in primary:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color='black', linewidth=1.4, zorder=3)

    ax.scatter(pts[:, 0], pts[:, 1], s=140, facecolor='black', edgecolor='none', zorder=5)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2.1, 1.8) 
    ax.set_aspect('equal')
    ax.axis('off')

    label = {1: "Monocentric", 2: "Duocentric", 3: "Polycentric", 4: "Polycentric"}[n]
    sub = {1: "(n = 1)", 2: "(n = 2)", 3: "(n = 3)", 4: "(n = 4)"}[n]

    ax.text(0, -1.62, label, ha='center', va='top', fontsize=15)
    ax.text(0, -1.90, sub, ha='center', va='top', fontsize=15)


plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.savefig("./polycentricity_n_series.png", dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white', edgecolor='black')
