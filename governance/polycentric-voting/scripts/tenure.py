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

# Major holder — dynamic strategy (2000 tokens, withdrawals and deposits)
MAJOR_DYNAMIC_EVENTS = [
    (0,   2000),
    (180, -1000),
    (365,  2000),
    (540, -1000),
]

# Small holder — singleton strategy (100 tokens, held to maturity)
SMALL_SINGLETON_EVENTS = [(0, 100)]

STRATEGIES = {
    'Major — Dynamic':   MAJOR_DYNAMIC_EVENTS,
    'Small — Singleton': SMALL_SINGLETON_EVENTS,
}

COLORS = {
    'Major — Dynamic':   '#d62728',
    'Small — Singleton': '#1f77b4',
}


# ─── Simulation ───────────────────────────────────────────────────────────────

days_range = np.arange(0, 731, 1)

t_eff_results = {}
for strategy, events in STRATEGIES.items():
    t_eff_results[strategy] = [calculate_effective_time(events, d) for d in days_range]

# rho = variance of t_effective over time per strategy
rho_results = {}
for strategy, t_effs in t_eff_results.items():
    rho_results[strategy] = [np.var(t_effs[:i+1]) for i in range(len(days_range))]

# tau = slope of rho up to each day via linear regression
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


# ─── Plot tenure ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

for strategy, tau in tau_results.items():
    ax.plot(days_range, tau, color=COLORS[strategy], linewidth=2.0, label=strategy)

ax.axvline(x=180, color='grey', linestyle='--', alpha=0.4, label='T1 (180d)')
ax.axvline(x=365, color='grey', linestyle=':',  alpha=0.4, label='T2 (365d)')

# Dynamic event markers
for event_day, delta in MAJOR_DYNAMIC_EVENTS[1:]:
    ax.axvline(x=event_day, color='red', linestyle=':', alpha=0.2)
    ax.text(event_day + 4, max(max(t) for t in tau_results.values()) * 0.95,
            f'{"+" if delta > 0 else ""}{delta:,}', fontsize=7, color='red', alpha=0.7)

ax.set_title('Tenure (τ): Major Dynamic vs Small Singleton (Agile Configuration)')
ax.set_xlabel('Days')
ax.set_ylabel('τ (Slope of ρ)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('tenure_benchmark.png', dpi=150, bbox_inches='tight')
plt.show()


# ─── Tenure table ─────────────────────────────────────────────────────────────

checkpoints = [90, 180, 365, 540, 720]
print(f"\n{'Strategy':<22} " + " ".join(f"{'d='+str(d):<10}" for d in checkpoints))
print("-" * 75)
for strategy, tau in tau_results.items():
    vals = [f"{tau[d]:<10.4f}" for d in checkpoints]
    print(f"{strategy:<22} " + " ".join(vals))
