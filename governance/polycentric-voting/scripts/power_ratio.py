import numpy as np
import matplotlib.pyplot as plt

T_MAX = 720
X_MAX = 800
POINT_EVERY = 200

CONFIGS = [
    dict(r_min=0.2, r_max=1.0, k=0.5),
    dict(r_min=0.2, r_max=1.0, k=1.0),
    dict(r_min=0.2, r_max=1.0, k=2),
]

def power_ratio(t_eff, r_min, r_max, k):
    return r_min + (r_max - r_min) * (t_eff / T_MAX) ** k

def effective_time(t):
    return np.minimum(t, T_MAX)

def time_grid():
    near_zero = np.geomspace(1e-3, 1, 300)
    rest = np.linspace(1, X_MAX, 2000)
    return np.concatenate([[0.0], near_zero, rest])

def main():
    t = time_grid()
    point_days = np.arange(POINT_EVERY, X_MAX + 1, POINT_EVERY)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, cfg in zip(axes, CONFIGS):
        r_min, r_max, k = cfg["r_min"], cfg["r_max"], cfg["k"]

        ax.plot(t, power_ratio(effective_time(t), r_min, r_max, k),
                color="#252525", linewidth=1.5)

        for i, d in enumerate(point_days):
            y = power_ratio(effective_time(d), r_min, r_max, k)
            ax.plot(d, y, "o", color="#252525", markersize=7)

            last = i == len(point_days) - 1
            ax.annotate(f"{y:.2f}", (d, y), textcoords="offset points",
                        xytext=(-8, -15) if last else (6, -15),
                        ha="right" if last else "left",
                        fontsize=9, color="#252525", fontweight="bold")

        ax.text(0.97, 0.06,
                rf"$r_{{min}}={r_min}$, $r_{{max}}={r_max}$, $k={k}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="#000000", alpha=0.5))

        ax.set_xlim(0, X_MAX)
        ax.set_ylim(0.2, 1.05)
        ax.set_xticks(np.arange(0, X_MAX + 1, 200))
        ax.set_xlabel("Time (days)", fontsize=11)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"Power Ratio ($\alpha$)", fontsize=11)
    plt.tight_layout()
    plt.savefig("./power-ratio.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
