import numpy as np
import matplotlib.pyplot as plt


# ─── Effective time with dilution / withdrawal ────────────────────────────────

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
    return current_t_eff


# ─── Strategy definitions ─────────────────────────────────────────────────────

SINGLETON_EVENTS  = [(0, 10000)]
SEQUENTIAL_EVENTS = [(0, 3333), (90, 1667), (180, 2500), (270, 2500)]
DYNAMIC_EVENTS    = [(0, 5000), (180, -2500), (365, 5000), (540, -2500)]

STRATEGIES = {
    'Singleton':  SINGLETON_EVENTS,
    'Sequential': SEQUENTIAL_EVENTS,
    'Dynamic':    DYNAMIC_EVENTS,
}

COLORS = {
    'Singleton':  '#1f77b4',
    'Sequential': '#2ca02c',
    'Dynamic':    '#ff7f0e',
}


# ─── Simulation ───────────────────────────────────────────────────────────────

days_range = np.arange(0, 731, 1)

# t_effective per strategy per day
t_eff_results = {}
for strategy, events in STRATEGIES.items():
    t_eff_results[strategy] = [calculate_effective_time(events, d) for d in days_range]

# rho = variance of t_effective over time per strategy independently
rho_results = {}
for strategy, t_effs in t_eff_results.items():
    rho_results[strategy] = [np.var(t_effs[:i+1]) for i in range(len(days_range))]


# ─── Plot rho ─────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

for strategy, rho in rho_results.items():
    ax.plot(days_range, rho, color=COLORS[strategy], linewidth=2.0, label=f'ρ — {strategy}')

ax.axvline(x=180, color='grey', linestyle='--', alpha=0.4, label='T1 (180d)')
ax.axvline(x=365, color='grey', linestyle=':',  alpha=0.4, label='T2 (365d)')

# Dynamic event markers
for event_day, delta in DYNAMIC_EVENTS[1:]:
    ax.axvline(x=event_day, color='orange', linestyle=':', alpha=0.3)
    ax.text(event_day + 4, max(rho) * 0.95,
            f'{"+" if delta > 0 else ""}{delta:,}', fontsize=7, color='orange', alpha=0.8)

ax.set_title('Path Dependency: ρ (Variance of $t_{\\mathrm{effective}}$) Across Strategies (Agile Configuration)')
ax.set_xlabel('Days')
ax.set_ylabel('ρ (Variance)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('path_dependency_rho.png', dpi=150, bbox_inches='tight')
plt.show()


# ─── Tenure table ─────────────────────────────────────────────────────────────

checkpoints = [90, 180, 365, 540, 720]
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
