import numpy as np
import matplotlib.pyplot as plt

T_MAX = 720
T_PERIOD = 1000 

SINGLETON_EVENTS  = [(0, 10000)]
SEQUENTIAL_EVENTS = [(0, 3333), (90, 1667), (180, 2500), (270, 2500)]
DYNAMIC_EVENTS    = [(0, 5000), (180, -2500), (365, 5000), (540, -2500)]
TRANCHE_MARKERS = [(180, 'T$_1$ (180d)'), (365, 'T$_2$ (365d)'), (720, '$t_{max}$ (720d)') ]
STRATEGIES = {
    'Singleton':  (SINGLETON_EVENTS,  '#4477CC'),
    'Sequential': (SEQUENTIAL_EVENTS, '#2ca02c'),
    'Dynamic':    (DYNAMIC_EVENTS,    '#E07800'),
}

def calculate_effective_time(events, day):
    shares, t_eff = 0, 0
    for event_day, delta in sorted(events):
        if event_day > day:
            break
        if delta > 0:
            elapsed = day - event_day
            t_eff = (shares * t_eff + delta * elapsed) / (shares + delta)
            shares += delta
        elif delta < 0:
            shares = max(0, shares - min(abs(delta), shares))
    return min(t_eff, T_MAX)

def plot_strategy(ax, label, events, color):
    days = np.arange(0, T_PERIOD)
    t_effs = [calculate_effective_time(events, d) for d in days]
    ax.plot(days, t_effs, color=color, linewidth=2.0, label=label)

    for day, marker_label in TRANCHE_MARKERS:
        ax.axvline(x=day, color='grey', linestyle='--', alpha=0.4)
        ax.text(day, ax.get_ylim()[1], marker_label,
                color='black', fontsize=8, ha='center', va='bottom')

    if label == 'Dynamic' or label == 'Sequential':
        for event_day, delta in events[1:]:
            ax.axvline(x=event_day, color=color, linestyle=':', alpha=0.7)
            ax.text(event_day + 4, ax.get_ylim()[1] * 0.9,
                    f'{"+" if delta > 0 else ""}{delta:,}',
                    fontsize=7, color=color, fontweight=600, alpha=0.8)

    ax.set_xlim(0, T_PERIOD)
    ax.set_ylim(0, T_MAX + 40)
    ax.set_xlabel('Time (days)', fontsize=9)
    ax.set_ylabel('Effective Time ($t_{\\mathrm{effective}}$)', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

days_range = np.arange(0, T_PERIOD + 1, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, (label, (events, color)) in zip(axes, STRATEGIES.items()):
    plot_strategy(ax, label, events, color)

# Remove duplicate y labels
for ax in axes[1:]:
    ax.set_ylabel('')

plt.tight_layout()
plt.savefig('path_dependency.png', dpi=150, bbox_inches='tight')
plt.show()
