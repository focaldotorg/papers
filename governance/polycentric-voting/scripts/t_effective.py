import numpy as np
import matplotlib.pyplot as plt


# ─── Effective time with dilution / withdrawal ────────────────────────────────

def calculate_effective_time(events, day):
    """
    events: list of (day, delta_tokens) — positive = deposit, negative = withdrawal
    """
    current_shares = 0
    current_t_eff  = 0

    for event_day, delta in sorted(events):
        if event_day > day:
            break

        elapsed = day - event_day

        if delta > 0:
            # Deposit — dilutes t_effective
            if current_shares + delta == 0:
                current_t_eff = 0
            else:
                numerator     = (current_shares * current_t_eff) + (delta * elapsed)
                denominator   = current_shares + delta
                current_t_eff = numerator / denominator
            current_shares += delta

        elif delta < 0:
            # Withdrawal — preserves t_effective, reduces shares
            withdraw = min(abs(delta), current_shares)
            current_shares = max(0, current_shares - withdraw)
            # t_effective preserved on withdrawal per equation 1

    return current_t_eff


# ─── Strategy definitions ─────────────────────────────────────────────────────

# Singleton: 10000 tokens deposited at day 0, held to maturity
SINGLETON_EVENTS = [(0, 10000)]

# Sequential: staged deposits totalling 10000
SEQUENTIAL_EVENTS = [(0, 3333), (90, 1667), (180, 2500), (270, 2500)]

# Dynamic: deposits and withdrawals throughout lifecycle
DYNAMIC_EVENTS = [
    (0,   5000),   # initial deposit
    (180, -2500),  # partial withdrawal at T1
    (365,  5000),  # large deposit at T2
    (540, -2500),  # partial withdrawal
]

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

results = {}
for strategy, events in STRATEGIES.items():
    results[strategy] = [calculate_effective_time(events, d) for d in days_range]


# ─── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

for strategy, t_effs in results.items():
    ax.plot(days_range, t_effs, label=strategy, color=COLORS[strategy], linewidth=2.0)

# Raw days reference
ax.plot(days_range, days_range, color='grey', linestyle='--', linewidth=1.2,
        alpha=0.5, label='Raw Days (no dilution)')

# Tranche markers
ax.axvline(x=180, color='grey', linestyle='--', alpha=0.4, label='T1 (180d)')
ax.axvline(x=365, color='grey', linestyle=':',  alpha=0.4, label='T2 (365d)')

# Dynamic event markers
for event_day, delta in DYNAMIC_EVENTS[1:]:
    label = f'{"+" if delta > 0 else ""}{delta:,}'
    ax.axvline(x=event_day, color='orange', linestyle=':', alpha=0.3)
    ax.text(event_day + 4, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
            label, fontsize=7, color='orange', alpha=0.8)

ax.set_xlabel('Days')
ax.set_ylabel('$t_{\\mathrm{effective}}$ (days)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('path_dependency.png', dpi=150, bbox_inches='tight')
plt.show()


# ─── Tenure table ─────────────────────────────────────────────────────────────

checkpoints = [90, 180, 365, 540, 720]
print(f"\n{'Strategy':<12} " + " ".join(f"{'d='+str(d):<10}" for d in checkpoints))
print("-" * 70)
for strategy, t_effs in results.items():
    vals = [f"{t_effs[d]:<10.1f}" for d in checkpoints]
    print(f"{strategy:<12} " + " ".join(vals))
