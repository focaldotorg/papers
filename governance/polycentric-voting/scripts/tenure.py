import numpy as np
import matplotlib.pyplot as plt

T_MAX=720
T_PERIOD=2000

SMALL_SINGLETON_EVENTS = [(0, 100)]
MAJOR_DYNAMIC_EVENTS = [ (0,   2000), (180, -1000), (365,  2000), (540, -1000) ]
SEQUENTIAL_EVENTS = [(0, 50), (180, 10), (230, 10), (230, 10), (300, 10), (360, 10)]
COLORS = {'Singleton':  '#4477CC', 'Sequential': '#2ca02c', 'Dynamic': '#E07800' }
STRATEGIES = {
    'Singleton': SMALL_SINGLETON_EVENTS,
    'Sequential': SEQUENTIAL_EVENTS,
    'Dynamic':   MAJOR_DYNAMIC_EVENTS,
}

def calculate_effective_time(events, day):
    current_shares = 0
    current_t_eff  = 0
    for event_day, delta in sorted(events):
        if event_day > day:
            break
        elapsed = day - event_day
        if delta > 0: 
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

days_range = np.arange(0, T_PERIOD + 1, 1)

t_eff_results = {}
for strategy, events in STRATEGIES.items():
    t_eff_results[strategy] = [calculate_effective_time(events, d) for d in days_range]

rho_results = {}
for strategy, t_effs in t_eff_results.items():
    rho_results[strategy] = [np.var(t_effs[:i+1]) for i in range(len(days_range))]

tau_results = {}
for strategy, rho in rho_results.items():
    tau_series = []
    for i in range(len(days_range)):
        if i < 2:
            tau_series.append(0.0)
        else:
            x = np.arange(i + 1, dtype=float)
            y = np.array(rho[:i + 1])
            slope = np.polyfit(x, y, 1)[0]
            tau_series.append(slope)
    tau_results[strategy] = tau_series

fig, ax = plt.subplots(figsize=(11, 6))

for strategy, tau in tau_results.items():
    ax.plot(days_range, tau, color=COLORS[strategy], linewidth=2.0, label=strategy)

ax.axvline(x=180, color='grey', linestyle='--', alpha=0.4)
ax.axvline(x=365, color='grey', linestyle='--', alpha=0.4)
ax.axvline(x=720, color='grey', linestyle='--', alpha=0.4)

ax.text(180, ax.get_ylim()[1], '$T_1$ (180d)', 
        color='black', alpha=1, fontsize=9,
        ha='center', va='bottom')
ax.text(365, ax.get_ylim()[1], '$T_2$ (365d)', 
        color='black', alpha=1, fontsize=9,
        ha='center', va='bottom')
ax.text(720, ax.get_ylim()[1], '$t_{max}$ (720d)',
        color='black', alpha=1, fontsize=9,
        ha='center', va='bottom')

for event_day, delta in MAJOR_DYNAMIC_EVENTS[1:]:
    ax.axvline(x=event_day, color='#ff7f0e', linestyle=':', alpha=0.7)
    ax.text(event_day + 4, max(max(t) for t in tau_results.values()) * 0.95,
            f'{"+" if delta > 0 else ""}{delta:,}', fontsize=7, fontweight=600, color='#ff7f0e', alpha=0.7)

ax.set_xlabel('Time (days)')
ax.set_ylabel('Tenure (τ)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('tenure_benchmark.png', dpi=150, bbox_inches='tight')
plt.show()

checkpoints = [90, 180, 365, 540, 720]
print(f"\n{'Strategy':<22} " + " ".join(f"{'d='+str(d):<10}" for d in checkpoints))
print("-" * 75)
for strategy, tau in tau_results.items():
    vals = [f"{tau[d]:<10.4f}" for d in checkpoints]
    print(f"{strategy:<22} " + " ".join(vals))
