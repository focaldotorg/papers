import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

# ─── Banzhaf (Puentedura 2004) ────────────────────────────────────────────────

def banzhaf(weight, quota):
    max_order = sum(weight)
    polynomial = [1] + max_order * [0]
    current_order = 0
    aux_polynomial = polynomial[:]
    for i in range(len(weight)):
        current_order += weight[i]
        offset_polynomial = weight[i] * [0] + polynomial
        for j in range(current_order + 1):
            aux_polynomial[j] = polynomial[j] + offset_polynomial[j]
        polynomial = aux_polynomial[:]
    banzhaf_power = len(weight) * [0]
    swings = quota * [0]
    for i in range(len(weight)):
        for j in range(quota):
            if j < weight[i]:
                swings[j] = polynomial[j]
            else:
                swings[j] = polynomial[j] - swings[j - weight[i]]
        for k in range(weight[i]):
            banzhaf_power[i] += swings[quota - 1 - k]
    total_power = sum(banzhaf_power)
    return [float(p) / total_power if total_power > 0 else 0.0 for p in banzhaf_power]


# ─── PV Engine (agile configuration) ─────────────────────────────────────────

AGILE_TRANCHES = [
    {"max_days": 180,          "P": 0.2},
    {"max_days": 365,          "P": 0.8},
    {"max_days": float('inf'), "P": 1.0},
]
TMAX = 720
RMIN = 0.2
RMAX = 1.0


def calculate_time_weight(t_effective):
    weight = 0
    remaining = t_effective
    prev = 0
    for tranche in AGILE_TRANCHES:
        limit = tranche['max_days']
        width = limit - prev if limit != float('inf') else remaining
        in_bracket = min(remaining, width)
        if in_bracket > 0:
            weight += np.log2(in_bracket + 1) * tranche['P']
        remaining -= in_bracket
        prev = limit
        if remaining <= 0:
            break
    return weight

def calculate_power_ratio(t_effective):
    progress = np.clip(t_effective / TMAX, RMIN, RMAX)
    return RMIN + (RMAX - RMIN) * (progress ** 0.1)

def calculate_voting_power(tokens, t_effective):
    if t_effective <= 0:
        return 0.0
    W = calculate_time_weight(t_effective)
    alpha = calculate_power_ratio(t_effective)
    return W * (alpha * tokens + np.sqrt(tokens) * (1 - alpha))


# ─── Stakeholders (original distribution) ────────────────────────────────────

SCALE = 100  # divide all weights by this constant before passing to Banzhaf

CAT_TOKENS = {'Major': 2000, 'Large': 1000, 'Median': 500, 'Minor': 200, 'Small': 100}

SCHEDULE = {
    0:  {'Major': 1, 'Large': 2,  'Median': 3,  'Minor': 6,  'Small': 4},
    20: {'Major': 2, 'Large': 4,  'Median': 7,  'Minor': 14, 'Small': 14},
    40: {'Major': 3, 'Large': 7,  'Median': 12, 'Minor': 25, 'Small': 32},
    60: {'Major': 4, 'Large': 9,  'Median': 17, 'Minor': 35, 'Small': 53},
    80: {'Major': 5, 'Large': 10, 'Median': 20, 'Minor': 40, 'Small': 70},
}

def build_stakeholders():
    all_s = []
    prev = {cat: 0 for cat in CAT_TOKENS}
    for day in sorted(SCHEDULE.keys()):
        for cat, count in SCHEDULE[day].items():
            for _ in range(count - prev[cat]):
                all_s.append({'category': cat, 'join_day': day, 'tokens': CAT_TOKENS[cat]})
            prev[cat] = count
    return all_s


# ─── Simulation ───────────────────────────────────────────────────────────────

def run_banzhaf_benchmark(max_days=720, step=5):
    stakeholders = build_stakeholders()
    cats = list(CAT_TOKENS.keys())
    records = []

    for day in range(0, max_days + 1, step):
        active = [s for s in stakeholders if s['join_day'] <= day]
        if not active:
            continue

        # Individual voter weights scaled down by constant factor
        pv_weights    = [max(1, int(calculate_voting_power(s['tokens'], day - s['join_day']) / SCALE)) for s in active]
        pluto_weights = [max(1, int(s['tokens'] / SCALE)) for s in active]

        # 51% quota
        pv_quota    = ceil(sum(pv_weights)    * 0.66)
        pluto_quota = ceil(sum(pluto_weights) * 0.66)

        pv_idx    = banzhaf(pv_weights,    pv_quota)
        pluto_idx = banzhaf(pluto_weights, pluto_quota)

        # Aggregate by coalition
        cat_pv    = {c: 0.0 for c in cats}
        cat_pluto = {c: 0.0 for c in cats}
        for i, s in enumerate(active):
            cat_pv[s['category']]    += pv_idx[i]
            cat_pluto[s['category']] += pluto_idx[i]

        row = {'day': day}
        for cat in cats:
            row[f'pv_{cat}']    = cat_pv[cat]
            row[f'pluto_{cat}'] = cat_pluto[cat]
        records.append(row)

    return pd.DataFrame(records)


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_banzhaf(df):
    cats   = list(CAT_TOKENS.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, (label, prefix) in zip(axes, [
        ('PV — Polycentric Voting', 'pv_'),
        ('Plutocracy — Raw Tokens', 'pluto_'),
    ]):
        for cat, color in zip(cats, colors):
            ax.plot(df['day'], df[f'{prefix}{cat}'], label=cat, color=color, linewidth=1.8)
        ax.axvline(x=180, color='grey', linestyle='--', alpha=0.4, label='T1 (180d)')
        ax.axvline(x=365, color='grey', linestyle=':',  alpha=0.4, label='T2 (365d)')
        ax.set_title(label)
        ax.set_xlabel('Days')
        ax.set_ylabel('Banzhaf Power Index')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Banzhaf Power Index Time Series: PV vs Plutocracy (Agile Configuration)', fontsize=13)
    plt.tight_layout()
    plt.savefig('banzhaf_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    df = run_banzhaf_benchmark()
    plot_banzhaf(df)
