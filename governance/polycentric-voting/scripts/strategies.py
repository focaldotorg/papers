import numpy as np
import matplotlib.pyplot as plt

def calculate_effective_time_with_dilution(shares_history, days):
    if not shares_history:
        return 0
    first_day, first_shares = shares_history[0]
    if first_day > days:
        return 0
    current_shares = first_shares
    current_effective_time = days - first_day
    for i in range(1, len(shares_history)):
        deposit_day, additional_shares = shares_history[i]
        if deposit_day > days:
            break
        time_since_deposit = days - deposit_day
        numerator = (current_shares * current_effective_time) + (additional_shares * time_since_deposit)
        denominator = current_shares + additional_shares
        current_effective_time = numerator / denominator
        current_shares += additional_shares
    return current_effective_time


def calculate_effective_time_dynamic(events, days):
    """Deposits dilute t_effective, withdrawals preserve t_effective but reduce balance."""
    current_shares = 0
    current_t_eff  = 0
    for event_day, delta in sorted(events):
        if event_day > days:
            break
        if delta > 0:
            elapsed = days - event_day
            if current_shares == 0:
                current_t_eff = elapsed
            else:
                numerator     = (current_shares * current_t_eff) + (delta * elapsed)
                denominator   = current_shares + delta
                current_t_eff = numerator / denominator
            current_shares += delta
        elif delta < 0:
            # preserve t_effective, reduce balance only
            current_shares = max(0, current_shares + delta)
    return current_t_eff, current_shares


def calculate_time_weight(t_effective):
    time_tranches = [
        {"max_days": 180,          "P": 0.2},
        {"max_days": 365,          "P": 0.8},
        {"max_days": float('inf'), "P": 1.0}
    ]
    weight = 0
    remaining_time = t_effective
    prev_limit = 0
    for tranche in time_tranches:
        limit = tranche['max_days']
        width = limit - prev_limit if limit != float('inf') else remaining_time
        time_in_bracket = min(remaining_time, width)
        if time_in_bracket > 0:
            weight += np.log2(time_in_bracket + 1) * tranche['P']
        remaining_time -= time_in_bracket
        prev_limit = limit
        if remaining_time <= 0:
            break
    return weight


def calculate_power_ratio(t_effective, t_max=730, r_max=1.0, r_min=0.0, k_c=0.1):
    progress = np.clip(t_effective / t_max, 0, 1)
    return r_min + (r_max - r_min) * (progress ** k_c)


def calculate_voting_power(total_tokens, time_weight, alpha):
    if total_tokens == 0:
        return 0
    return time_weight * (alpha * total_tokens + np.sqrt(total_tokens) * (1 - alpha))


# Dynamic: 10,000 total deposited across lifecycle
DYNAMIC_EVENTS = [(0, 5000), (180, -2500), (365, 5000), (540, -2500)]


def simulate_strategy(strategy_name, days_range):
    voting_powers = []
    for days in days_range:
        if strategy_name == "Singleton":
            shares_history = [(0, 10000)]
            t_effective  = calculate_effective_time_with_dilution(shares_history, days)
            total_tokens = 10000

        elif strategy_name == "Sequential":
            shares_history = [(0, 3333)]
            if days >= 90:  shares_history.append((90, 1667))
            if days >= 180: shares_history.append((180, 2500))
            if days >= 270: shares_history.append((270, 2500))
            t_effective  = calculate_effective_time_with_dilution(shares_history, days)
            total_tokens = sum(s for d, s in shares_history if d <= days)

        elif strategy_name == "Splitting":
            shares_history = [(0, 1000)]
            t_effective  = calculate_effective_time_with_dilution(shares_history, days)
            total_tokens = 10000
            splitting_factor = max(0.5, 1.2 - (days / 1000))
            W     = calculate_time_weight(t_effective)
            alpha = calculate_power_ratio(t_effective)
            voting_powers.append(calculate_voting_power(1000, W, alpha) * 10 * splitting_factor)
            continue

        elif strategy_name == "Dynamic":
            t_effective, total_tokens = calculate_effective_time_dynamic(DYNAMIC_EVENTS, days)

        W     = calculate_time_weight(t_effective)
        alpha = calculate_power_ratio(t_effective)
        voting_powers.append(calculate_voting_power(total_tokens, W, alpha))

    return voting_powers


# ─── Simulation ───────────────────────────────────────────────────────────────

days_range = np.arange(0, 731, 20)

singleton_power  = simulate_strategy("Singleton",  days_range)
sequential_power = simulate_strategy("Sequential", days_range)
splitting_power  = simulate_strategy("Splitting",  days_range)
dynamic_power    = simulate_strategy("Dynamic",    days_range)

# ─── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(days_range, singleton_power,  'b-',           linewidth=2.5, label='Singleton',  alpha=0.9)
ax.plot(days_range, sequential_power, 'g-',           linewidth=2.5, label='Sequential', alpha=0.9)
ax.plot(days_range, splitting_power,  'r-',           linewidth=2.5, label='Splitting',  alpha=0.9)
ax.plot(days_range, dynamic_power,    color='orange', linewidth=2.5, label='Dynamic',    alpha=0.9)

phase_colors     = ['lightblue', 'lightgreen', 'lightyellow']
phase_boundaries = [180, 365, 730]
phase_labels     = ['T1\n(180d)', 'T2\n(365d)', 'T3\n(730d)']
prev = 0
for end, color, label in zip(phase_boundaries, phase_colors, phase_labels):
    ax.axvspan(prev, end, alpha=0.1, color=color)
    ax.text((prev + end) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
            label, ha='center', va='top', fontsize=8, color='grey')
    prev = end

for day in [180, 365]:
    ax.axvline(x=day, color='grey', linestyle='--', alpha=0.3)

ax.set_title('Voting Power by Strategy: PV Agile Configuration', fontsize=13, weight='bold')
ax.set_xlabel('Time (Days)', fontsize=12, weight='bold')
ax.set_ylabel('Voting Power ($V_P$)', fontsize=12, weight='bold')
ax.set_xlim(0, 730)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
ax.grid(True, alpha=0.2, linestyle='--')
ax.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('strategies.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── Benchmark table ──────────────────────────────────────────────────────────

strategies = {
    'Singleton':  singleton_power,
    'Sequential': sequential_power,
    'Splitting':  splitting_power,
    'Dynamic':    dynamic_power,
}

print(f"\n{'Strategy':<12} {'3M':<10} {'6M':<10} {'12M':<10} {'24M':<10} {'Impact':<8}")
print("-" * 58)
for name, powers in strategies.items():
    idx  = [np.argmin(np.abs(days_range - d)) for d in [90, 180, 365, 730]]
    vals = [powers[i] for i in idx]
    baseline = strategies['Singleton'][idx[-1]]
    impact   = ((powers[idx[-1]] - baseline) / baseline) * 100
    print(f"{name:<12} " + " ".join(f"{v:<10.0f}" for v in vals) + f" {impact:+.0f}%")
