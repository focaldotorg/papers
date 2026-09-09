import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import ceil

T_MAX = 1825 
T_PERIOD = 2000
RMIN  = 0.2
RMAX  = 1.0
SCALE = 100

PROGRESSIVE_TRANCHES = [
    {"max_days": 260, "P": 0.010},
    {"max_days": 520, "P": 0.020},
    {"max_days": 780, "P": 0.025},
    {"max_days": 1040, "P": 0.030},
    {"max_days": 1300, "P": 0.045},
    {"max_days": 1560, "P": 0.050},
    {"max_days": float('inf'), "P": 0.070}
]

CAT_TOKENS = {'Major': 2000, 'Large': 1000, 'Median': 500, 'Minor': 200, 'Small': 100, 'Insiders': 10000 }
CAT_COLORS = ['#EF5350', '#FF8A65', '#66BB6A', '#42A5F5', '#1565C0', '#7E57C2']
SCHEDULE = {
    0:  {'Insiders': 1, 'Major': 1, 'Large': 2,  'Median': 3,  'Minor': 6,  'Small': 8},
    20: {'Insiders': 1, 'Major': 2, 'Large': 4,  'Median': 7,  'Minor': 14, 'Small': 24},
    40: {'Insiders': 1, 'Major': 3, 'Large': 7,  'Median': 12, 'Minor': 25, 'Small': 36},
    60: {'Insiders': 1, 'Major': 4, 'Large': 10, 'Median': 17, 'Minor': 35, 'Small': 57},
    80: {'Insiders': 1, 'Major': 6, 'Large': 12, 'Median': 20, 'Minor': 40, 'Small': 80},
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
    for tranche in PROGRESSIVE_TRANCHES:
        limit      = tranche['max_days']
        width      = limit - prev if limit != float('inf') else remaining
        in_bracket = min(remaining, width)
        if in_bracket > 0:
            weight += np.log2(in_bracket + 1) * tranche['P']
        remaining -= in_bracket
        prev       = limit
        if remaining <= 0:
            break
    return weight

def calculate_power_ratio(t_effective):
    progress = np.clip(t_effective / T_MAX, RMIN, RMAX)
    return RMIN + (RMAX - RMIN) * (progress ** 0.5)

def calculate_voting_power(tokens, t_effective, label):
    if t_effective <= 0:
        return 0.0
    if label == 'Insiders': 
        t_effective += 260 
        if t_effective > T_MAX: 
            t_effective = T_MAX
    W     = calculate_time_weight(t_effective)
    alpha = calculate_power_ratio(t_effective)
    # Include share weight
    return tokens + W * (alpha * tokens + np.sqrt(tokens) * (1 - alpha))

def build_stakeholders():
    all_s = []
    prev  = {cat: 0 for cat in CAT_TOKENS}
    for day in sorted(SCHEDULE.keys()):
        for cat, count in SCHEDULE[day].items():

            for _ in range(count - prev[cat]):
                all_s.append({'category': cat, 'join_day': day, 'tokens': CAT_TOKENS[cat]})
            prev[cat] = count
    return all_s

def weighted(value):
    return max(1, int(value / SCALE))

def run_banzhaf_benchmark(max_days=720, step=5):
    stakeholders = build_stakeholders()
    cats         = list(CAT_TOKENS.keys())
    records      = []

    for day in range(0, max_days + 1, step):
        active = [s for s in stakeholders if s['join_day'] <= day]
        if not active:
            continue

        pv_weights = [weighted(calculate_voting_power(s['tokens'], day - s['join_day'], s['category'])) for s in active]
        pluto_weights = [weighted(s['tokens']) for s in active] 
        pv_quota    = ceil(sum(pv_weights)    * 0.51)
        pluto_quota = ceil(sum(pluto_weights) * 0.51)

        total_pv    = sum(pv_weights)
        total_pluto = sum(pluto_weights)
        pv_norm     = [max(1, int(w * total_pluto / total_pv)) for w in pv_weights]
        cross_quota = ceil(total_pluto * 0.51)

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
    ax1.axvline(x=260,  color='grey', linestyle='--', alpha=0.4)
    ax1.axvline(x=520,  color='grey', linestyle='--', alpha=0.4)
    ax1.axvline(x=780,  color='grey', linestyle='--', alpha=0.4)
    ax1.axvline(x=1040, color='grey', linestyle='--', alpha=0.4)
    ax1.axvline(x=1300, color='grey', linestyle='--', alpha=0.4)
    ax1.axvline(x=1560, color='grey', linestyle='--', alpha=0.4)
    ax1.axvline(x=T_MAX, color='grey', linestyle='--', alpha=0.4)
    ax1.text(260,  ax1.get_ylim()[1], '$T_{1}$ (260d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax1.text(520,  ax1.get_ylim()[1], '$T_{2}$ (520d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax1.text(780,  ax1.get_ylim()[1], '$T_{3}$ (780d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax1.text(1040, ax1.get_ylim()[1], '$T_{4}$ (1040d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax1.text(1300, ax1.get_ylim()[1], '$T_{5}$ (1300d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax1.text(1560, ax1.get_ylim()[1], '$T_{6}$ (1560d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax1.text(T_MAX, ax1.get_ylim()[1], f'$t_{{max}}$ ({T_MAX}d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Banzhaf Index')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()

    plt.savefig('progressive_banzhaf_pv.png', dpi=150, bbox_inches='tight')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for cat, color in zip(cats, colors):
        ax2.plot(df['day'], df[f'pluto_{cat}'], label=cat, color=color, linewidth=1.8)
    ax2.axvline(x=260,  color='grey', linestyle='--', alpha=0.4)
    ax2.axvline(x=520,  color='grey', linestyle='--', alpha=0.4)
    ax2.axvline(x=780,  color='grey', linestyle='--', alpha=0.4)
    ax2.axvline(x=1040, color='grey', linestyle='--', alpha=0.4)
    ax2.axvline(x=1300, color='grey', linestyle='--', alpha=0.4)
    ax2.axvline(x=1560, color='grey', linestyle='--', alpha=0.4)
    ax2.axvline(x=T_MAX, color='grey', linestyle='--', alpha=0.4)
    ax2.text(260,  ax2.get_ylim()[1], '$T_{1}$ (260d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax2.text(520,  ax2.get_ylim()[1], '$T_{2}$ (520d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax2.text(780,  ax2.get_ylim()[1], '$T_{3}$ (780d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax2.text(1040, ax2.get_ylim()[1], '$T_{4}$ (1040d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax2.text(1300, ax2.get_ylim()[1], '$T_{5}$ (1300d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax2.text(1560, ax2.get_ylim()[1], '$T_{6}$ (1560d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax2.text(T_MAX, ax2.get_ylim()[1], f'$t_{{max}}$ ({T_MAX}d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Banzhaf Index')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('progressive_banzhaf_plutocracy.png', dpi=150, bbox_inches='tight')
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for cat, color in zip(cats, colors):
        ax3.plot(df['day'][2:], df[f'cross_{cat}'][2:], label=cat, color=color, linewidth=1.8)

    ax3.axvline(x=260,  color='grey', linestyle='--', alpha=0.4)
    ax3.axvline(x=520,  color='grey', linestyle='--', alpha=0.4)
    ax3.axvline(x=780,  color='grey', linestyle='--', alpha=0.4)
    ax3.axvline(x=1040, color='grey', linestyle='--', alpha=0.4)
    ax3.axvline(x=1300, color='grey', linestyle='--', alpha=0.4)
    ax3.axvline(x=1560, color='grey', linestyle='--', alpha=0.4)
    ax3.axvline(x=T_MAX, color='grey', linestyle='--', alpha=0.4)
    ax3.text(260,  ax3.get_ylim()[1], '$T_{1}$ (260d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax3.text(520,  ax3.get_ylim()[1], '$T_{2}$ (520d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax3.text(780,  ax3.get_ylim()[1], '$T_{3}$ (780d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax3.text(1040, ax3.get_ylim()[1], '$T_{4}$ (1040d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax3.text(1300, ax3.get_ylim()[1], '$T_{5}$ (1300d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax3.text(1560, ax3.get_ylim()[1], '$T_{6}$ (1560d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')
    ax3.text(T_MAX, ax3.get_ylim()[1], f'$t_{{max}}$ ({T_MAX}d)', color='black', alpha=1, fontsize=9, ha='center', va='bottom')

    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Banzhaf Index')
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('progressive_banzhaf_cross.png', dpi=150, bbox_inches='tight')
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
    print(f"{'Std Dev':<8} {np.std(pv_all):>{col_w}.6f} {np.std(pluto_all):>{col_w}.6f} {np.std(cross_all):>{col_w}.6f}")
    print(f"{'Variance':<8} {np.var(pv_all):>{col_w}.6f} {np.var(pluto_all):>{col_w}.6f} {np.var(cross_all):>{col_w}.6f}")


if __name__ == "__main__":
    df = run_banzhaf_benchmark(T_PERIOD)
    plot_banzhaf(df)
    print("\nPower Concentration Range & Variance Table\n")
    print_range_table(df)
