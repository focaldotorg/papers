import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import ceil

T_MAX = 720
RMIN  = 0.2
RMAX  = 1.0
SCALE = 100

AGILE_TRANCHES = [
    {"max_days": 180,          "P": 0.2},
    {"max_days": 365,          "P": 0.8},
    {"max_days": float('inf'), "P": 1.0},
]

CAT_TOKENS = {'Major': 2000, 'Large': 1000, 'Median': 500, 'Minor': 200, 'Small': 100}
CAT_COLORS = ['#EF5350', '#FF8A65', '#66BB6A', '#42A5F5', '#1565C0']
SCHEDULE = {
    0:  {'Major': 1, 'Large': 2,  'Median': 3,  'Minor': 6,  'Small': 8},
    20: {'Major': 2, 'Large': 4,  'Median': 7,  'Minor': 14, 'Small': 24},
    40: {'Major': 3, 'Large': 7,  'Median': 12, 'Minor': 25, 'Small': 36},
    60: {'Major': 4, 'Large': 10, 'Median': 17, 'Minor': 35, 'Small': 57},
    80: {'Major': 6, 'Large': 12, 'Median': 20, 'Minor': 40, 'Small': 80},
}

def banzhaf(weight, quota):
    max_order      = sum(weight)
    polynomial     = [1] + max_order * [0]
    current_order  = 0
    aux_polynomial = polynomial[:]
    for i in range(len(weight)):
        current_order     += weight[i]
        offset_polynomial  = weight[i] * [0] + polynomial
        for j in range(current_order + 1):
            aux_polynomial[j] = polynomial[j] + offset_polynomial[j]
        polynomial = aux_polynomial[:]
    banzhaf_power = len(weight) * [0]
    swings        = quota * [0]
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


def calculate_time_weight(t_effective):
    weight    = 0
    remaining = t_effective
    prev      = 0
    for tranche in AGILE_TRANCHES:
        limit      = tranche['max_days']
        width      = limit - prev if limit != float('inf') else remaining
        in_bracket = min(remaining, width)
        if in_bracket > 0:
            weight += np.log2(in_bracket + 1) * tranche['P']
        remaining -= in_bracket
        prev       = limit
        if remaining <= 0:
            break
    return min(weight, T_MAX)

def calculate_power_ratio(t_effective):
    progress = np.clip(t_effective / T_MAX, RMIN, 1)
    return RMIN + (RMAX - RMIN) * (progress ** 0.1)

def calculate_voting_power(tokens, t_effective):
    if t_effective <= 0:
        return 0.0
    W     = calculate_time_weight(t_effective)
    alpha = calculate_power_ratio(t_effective)
    return W * (alpha * tokens + np.sqrt(tokens) * (1 - alpha))

def build_stakeholders():
    all_s = []
    prev  = {cat: 0 for cat in CAT_TOKENS}
    for day in sorted(SCHEDULE.keys()):
        for cat, count in SCHEDULE[day].items():
            for _ in range(count - prev[cat]):
                all_s.append({'category': cat, 'join_day': day, 'tokens': CAT_TOKENS[cat]})
            prev[cat] = count
    return all_s


def run_banzhaf_benchmark(max_days=720, step=5):
    stakeholders = build_stakeholders()
    cats         = list(CAT_TOKENS.keys())
    records      = []

    for day in range(0, max_days + 1, step):
        active = [s for s in stakeholders if s['join_day'] <= day]
        if not active:
            continue

        pv_weights    = [max(1, int(calculate_voting_power(s['tokens'], day - s['join_day']) / SCALE)) for s in active]
        pluto_weights = [max(1, int(s['tokens'] / SCALE)) for s in active]

        pv_quota    = ceil(sum(pv_weights)    * 0.33)
        pluto_quota = ceil(sum(pluto_weights) * 0.33)

        total_pv    = sum(pv_weights)
        total_pluto = sum(pluto_weights)
        pv_norm     = [max(1, int(w * total_pluto / total_pv)) for w in pv_weights]
        cross_quota = ceil(total_pluto * 0.33)

        pv_idx    = banzhaf(pv_weights,    pv_quota)
        pluto_idx = banzhaf(pluto_weights, pluto_quota)
        cross_idx = banzhaf(pv_norm,       cross_quota)

        cat_pv    = {c: 0.0 for c in cats}
        cat_pluto = {c: 0.0 for c in cats}
        cat_cross = {c: 0.0 for c in cats}

        for i, s in enumerate(active):
            cat_pv[s['category']]    += pv_idx[i]
            cat_pluto[s['category']] += pluto_idx[i]
            cat_cross[s['category']] += cross_idx[i]

        row = {'day': day}
        for cat in cats:
            row[f'pv_{cat}']    = cat_pv[cat]
            row[f'pluto_{cat}'] = cat_pluto[cat]
            row[f'cross_{cat}'] = cat_cross[cat]
        records.append(row)

    return pd.DataFrame(records)


def plot_banzhaf(df):
    cats   = list(CAT_TOKENS.keys())
    colors = CAT_COLORS

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for cat, color in zip(cats, colors):
        ax1.plot(df['day'], df[f'pv_{cat}'], label=cat, color=color, linewidth=1.8)
    ax1.axvline(x=180, color='grey', linestyle='--', alpha=0.4)
    ax1.axvline(x=365, color='grey', linestyle='--', alpha=0.4)
    ax1.text(180, ax1.get_ylim()[1], '$T_{1}$ (180d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax1.text(365, ax1.get_ylim()[1], '$T_{2}$ (365d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Banzhaf Index')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('banzhaf_pv.png', dpi=150, bbox_inches='tight')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for cat, color in zip(cats, colors):
        ax2.plot(df['day'], df[f'pluto_{cat}'], label=cat, color=color, linewidth=1.8)
    ax2.axvline(x=180, color='grey', linestyle='--', alpha=0.4)
    ax2.axvline(x=365, color='grey', linestyle='--', alpha=0.4)
    ax2.text(180, ax2.get_ylim()[1], '$T_{1}$ (180d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax2.text(365, ax2.get_ylim()[1], '$T_{2}$ (365d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Banzhaf Index')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('banzhaf_plutocracy.png', dpi=150, bbox_inches='tight')
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for cat, color in zip(cats, colors):
        ax3.plot(df['day'], df[f'cross_{cat}'], label=cat, color=color, linewidth=1.8)
    ax3.axvline(x=180, color='grey', linestyle='--', alpha=0.4)
    ax3.axvline(x=365, color='grey', linestyle='--', alpha=0.4)
    ax3.text(180, ax3.get_ylim()[1], '$T_{1}$ (180d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax3.text(365, ax3.get_ylim()[1], '$T_{2}$ (365d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Banzhaf Index')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('banzhaf_cross.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_range_table(df):
    cats      = list(CAT_TOKENS.keys())
    snapshots = [90, 180, 270, 365, 450, 540, 630, 720]

    col_w = 14
    header = f"{'Day':<8} {'Pluto Range':>{col_w}} {'PV Range':>{col_w}} {'Cross Range':>{col_w}}"
    sep    = '-' * len(header)
    print(header)
    print(sep)

    for day in snapshots:
        row = df[df['day'] == day]
        if row.empty:
            row = df.iloc[(df['day'] - day).abs().argsort()[:1]]

        pv_vals    = [row[f'pv_{c}'].values[0]    for c in cats]
        pluto_vals = [row[f'pluto_{c}'].values[0] for c in cats]
        cross_vals = [row[f'cross_{c}'].values[0] for c in cats]

        pv_r    = max(pv_vals)    - min(pv_vals)
        pluto_r = max(pluto_vals) - min(pluto_vals)
        cross_r = max(cross_vals) - min(cross_vals)

        print(f"{int(row['day'].values[0]):<8} {pluto_r:>{col_w}.4f} {pv_r:>{col_w}.4f} {cross_r:>{col_w}.4f}")

    # Variance of raw Banzhaf indices across full time series per system
    pv_all    = df[[f'pv_{c}'    for c in cats]].values.flatten()
    pluto_all = df[[f'pluto_{c}' for c in cats]].values.flatten()
    cross_all = df[[f'cross_{c}' for c in cats]].values.flatten()

    print(sep)
    print(f"{'Variance':<8} {np.var(pv_all):>{col_w}.6f} {np.var(pluto_all):>{col_w}.6f} {np.var(cross_all):>{col_w}.6f}")


if __name__ == "__main__":
    df = run_banzhaf_benchmark()
    plot_banzhaf(df)
    print("\nPower Concentration Range & Variance Table\n")
    print_range_table(df)
