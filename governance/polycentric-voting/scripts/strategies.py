import numpy as np
import matplotlib.pyplot as plt

T_MAX=720
T_PERIOD=980

COLORS = {'Splitting': 'r', 'Singleton': '#289128', 'Sequential': 'b', 'Dynamic': 'orange'}
STRATEGIES = {
    'Splitting':  [(0, 100)],   # 100x accounts of 100
    'Singleton':  [(0, 10000)],
    'Sequential': [(0, 3333), (90, 1667), (180, 2500), (270, 2500)],
    'Dynamic':    [(0, 7000), (180, -2000), (365, +6000), (540, -1000)],
}

def calculate_effective_time(events, days):
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
            current_shares = max(0, current_shares + delta)

    if (current_t_eff > T_MAX):
        current_t_eff = T_MAX

    return current_t_eff, current_shares


def calculate_convergence_day(events, t_max=T_MAX, horizon_days=20000, day_step=1):
    deposit_days = [day for day, delta in events if delta > 0]
    if not deposit_days:
        return None  # never accumulates any time-weight

    last_deposit_day = max(deposit_days)
    t_eff_at_last_deposit, _ = calculate_effective_time(events, last_deposit_day)

    if t_eff_at_last_deposit < t_max:
        # Closed-form: t_eff grows at slope 1 with no more discontinuities
        # after the last deposit, so solve directly.
        return last_deposit_day + (t_max - t_eff_at_last_deposit)

    last_below = None
    day = 0
    while day <= horizon_days:
        t_eff, _ = calculate_effective_time(events, day)
        if t_eff < t_max - 1e-9:
            last_below = day
        day += day_step
    return None if last_below is None else last_below + day_step


def calculate_time_weight(t_effective):
    time_tranches = [
        {"max_days": 180,          "P": 0.020},
        {"max_days": 365,          "P": 0.035},
        {"max_days": float('inf'), "P": 0.070}
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


def calculate_power_ratio(t_effective, t_max=720, r_max=1.0, r_min=0.2, k_c=0.5):
    progress = np.clip(t_effective / t_max, r_min, 1)
    return r_min + (r_max - r_min) * (progress ** k_c)

def calculate_voting_power(total_tokens, time_weight, alpha):
    if total_tokens == 0:
        return 0
    return time_weight * (alpha * total_tokens + np.sqrt(total_tokens) * (1 - alpha))


def simulate_strategy(name, events, days_range):
    voting_powers = []
    for days in days_range:
        t_eff, tokens = calculate_effective_time(events, days)

        if name == 'Splitting':
            tokens = 100 
            W     = calculate_time_weight(t_eff)
            alpha = calculate_power_ratio(t_eff)
            voting_powers.append(calculate_voting_power(100, W, alpha) * 100)
            continue

        W     = calculate_time_weight(t_eff)
        alpha = calculate_power_ratio(t_eff)
        voting_powers.append(calculate_voting_power(tokens, W, alpha))
    return voting_powers


days_range = np.arange(0, T_PERIOD, 20)

results = {name: simulate_strategy(name, events, days_range)
           for name, events in STRATEGIES.items()}

fig, ax = plt.subplots(figsize=(12, 8))

for name, powers in results.items():
    ax.plot(days_range, powers, color=COLORS[name], linewidth=2.5, label=name, alpha=0.9)

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


ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Voting Power ($V_P$)', fontsize=12)
ax.set_xlim(0, T_PERIOD)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
ax.grid(True, alpha=0.2, linestyle='--')
ax.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('strategies.png', dpi=150, bbox_inches='tight')
plt.show()

checkpoints = [90, 180, 365, T_PERIOD - 1]

print(f"\n{'Strategy':<12} {'3M':<10} {'6M':<10} {'12M':<10} {'24M':<10} {'Convergence':<12}")
print("-" * 62)
for name, powers in results.items():
    idx  = [np.argmin(np.abs(days_range - d)) for d in checkpoints]
    vals = [powers[i] for i in idx]
    conv_day = calculate_convergence_day(STRATEGIES[name])
    conv_str = f"{conv_day:.0f}d" if conv_day is not None else "never"
    print(f"{name:<12} " + " ".join(f"{v:<10.0f}" for v in vals) + f" {conv_str:<12}")
