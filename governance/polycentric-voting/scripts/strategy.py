import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AXIS_COLOR = '#000000'

plt.rcParams.update({
    'text.color':          AXIS_COLOR,
    'axes.labelcolor':     AXIS_COLOR,
    'xtick.color':         AXIS_COLOR,
    'ytick.color':         AXIS_COLOR,
    'axes.edgecolor':      AXIS_COLOR,
    'axes.facecolor':      'white',
    'figure.facecolor':    'white',
    'savefig.facecolor':   'white',
    'savefig.transparent': False,
    'grid.color':          AXIS_COLOR,
})

T_MAX = 720
T1    = 180
T2    = 365

COLORS = {
    'Singleton':  '#289128',
    'Sequential': 'b',
    'Splitting':  'r',
    'Dynamic':    'orange',
}

STRATEGIES = {
    'Singleton':  [(0, +10000, '+10,000')],
    'Sequential': [(0, +3333, '+3,333'), (90, +1667, '+1,667'),
                   (180, +2500, '+2,500'), (270, +2500, '+2,500')],
    'Splitting':  [(0, +100, '100 x 100')],
    'Dynamic':    [(0, +5000, '+5,000'), (180, -2500, '-2,500'),
                   (365, +5000, '+5,000'), (540, -2500, '-2,500')],
}

strategy_order = ['Singleton', 'Sequential', 'Splitting', 'Dynamic']

for name in strategy_order:
    color  = COLORS[name]
    events = STRATEGIES[name]

    fig, ax = plt.subplots(1, 1, figsize=(11, 2.2))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.75, bottom=0.25)

    ybot = 0.0
    ytop = 1.0
    mid  = (ybot + ytop) / 2

    # Centre line using axes-fraction y
    ax.hlines(
        y=mid,
        xmin=0,
        xmax=1,
        transform=ax.get_yaxis_transform(),
        color=color,
        linewidth=1.8,
        alpha=0.85
    )

    # T1 / T2 vertical dashed lines with labels on all plots
    for xv, lbl in [(T1, '$T_1$'), (T2, '$T_2$')]:
        ax.axvline(xv, color=AXIS_COLOR, linestyle=':', linewidth=0.9, alpha=0.5)
        ax.text(xv, 1.15, lbl, transform=ax.get_xaxis_transform(),
                color=AXIS_COLOR, fontsize=8, ha='center', va='top', alpha=0.7)

    # Arrows and labels
    arrow_up_tip   = mid + 0.35
    arrow_down_tip = mid - 0.35

    for day, delta, label in events:
        if delta > 0:
            ax.annotate('', xy=(day, arrow_up_tip), xytext=(day, mid),
                        xycoords=('data', 'axes fraction'),
                        textcoords=('data', 'axes fraction'),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
            ax.text(day, arrow_up_tip, label,
                    transform=ax.get_xaxis_transform(),
                    color=color, fontsize=9,
                    va='bottom', ha='center',
                    fontweight='bold')
        else:
            ax.annotate('', xy=(day, arrow_down_tip), xytext=(day, mid),
                        xycoords=('data', 'axes fraction'),
                        textcoords=('data', 'axes fraction'),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
            ax.text(day, arrow_down_tip, label,
                    transform=ax.get_xaxis_transform(),
                    color=color, fontsize=9,
                    va='top', ha='center',
                    fontweight='bold')

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(ybot, ytop)
    ax.set_yticks([])
    ax.set_xlabel('Time (days)', fontsize=10)
    ax.xaxis.set_tick_params(labelsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor(AXIS_COLOR)
        spine.set_linewidth(0.8)

    out = f'./pv_strategy_{name.lower()}_light.png'
    plt.savefig(out, format='png', dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)
