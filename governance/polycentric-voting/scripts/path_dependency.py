import numpy as np
import matplotlib.pyplot as plt

T_MAX = 720
T_PERIOD = 1240

SINGLETON_EVENTS  = [(0, 10000)]
SEQUENTIAL_EVENTS = [(0, 3333), (90, 1667), (180, 2500), (270, 2500)]
DYNAMIC_EVENTS    = [(0, 5000), (180, -2500), (365, 5000), (540, -2500)]

STRATEGIES = {
    'Singleton':  SINGLETON_EVENTS,
    'Sequential': SEQUENTIAL_EVENTS,
    'Dynamic':    DYNAMIC_EVENTS,
}

COLORS = {
    'Singleton':  'b',
    'Sequential': '#2ca02c',
    'Dynamic':    '#ff7f0e',
}

def calculate_effective_time(events, day):
    current_shares = 0
    current_t_eff  = 0

    for event_day, delta in sorted(events):
        if event_day > day:
            break
        elapsed = day - event_day
        if delta > 0:
            if current_shares + delta == 0:
                current_t_eff = 0
            else:
                numerator     = (current_shares * current_t_eff) + (delta * elapsed)
                denominator   = current_shares + delta
                current_t_eff = numerator / denominator
            current_shares += delta
        elif delta < 0:
            withdraw = min(abs(delta), current_shares)
            current_shares = max(0, current_shares - withdraw)
   
    if (current_t_eff > T_MAX):
        return T_MAX

    return current_t_eff

days_range = np.arange(0, T_PERIOD, 1)

t_eff_results = {}
for strategy, events in STRATEGIES.items():
    t_eff_results[strategy] = [calculate_effective_time(events, d) for d in days_range]

rho_results = {}
for strategy, t_effs in t_eff_results.items():
    rho_results[strategy] = [np.var(t_effs[:i+1]) for i in range(len(days_range))]

fig, ax = plt.subplots(figsize=(11, 6))

for strategy, rho in rho_results.items():
    ax.plot(days_range, rho, color=COLORS[strategy], linewidth=2.0, label=f'{strategy}')

ax.axvline(x=180, color='grey', linestyle='--', alpha=0.4)
ax.axvline(x=365, color='grey', linestyle='--', alpha=0.4)
ax.axvline(x=720, color='grey', linestyle='--', alpha=0.4)

ax.text(180, ax.get_ylim()[1], '$T_{1}$ (180d)', 
        color='black', alpha=1, fontsize=9,
        ha='center', va='bottom')
ax.text(365, ax.get_ylim()[1], '$T_{2}$ (365d)', 
        color='black', alpha=1, fontsize=9,
        ha='center', va='bottom')
ax.text(720, ax.get_ylim()[1], '$T_{3}$ (720d)', 
        color='black', alpha=1, fontsize=9,
        ha='center', va='bottom')

ax.set_xlabel('Time (days)')
ax.set_ylabel('Path-dependency (ρ)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('path_dependency_rho.png', dpi=150, bbox_inches='tight')
plt.show()

checkpoints = [90, 180, 365, 540, 720, T_PERIOD - 20 ]
print(f"\n{'Strategy':<12} " + " ".join(f"{'d='+str(d):<12}" for d in checkpoints))
print("-" * 75)
for strategy, rho in rho_results.items():
    tau_vals = []
    for d in checkpoints:
        # slope of rho up to day d via linear regression
        x = np.arange(d + 1, dtype=float)
        y = np.array(rho[:d + 1])
        slope = np.polyfit(x, y, 1)[0]
        tau_vals.append(f"{slope:.4f}")
    print(f"{strategy:<12} " + " ".join(f"{v:<12}" for v in tau_vals))
