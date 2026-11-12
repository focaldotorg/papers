import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from math import sqrt, log2

T_MAX = 1825
BUFFER_DAYS = 260 
TEAM_TOKENS = 10000
T_PERIOD = 2000 

CAT_TOKENS = {'Major': 2000, 'Large': 1000, 'Median': 500, 'Minor': 200, 'Small': 100}
CAT_COLORS = {'Major': '#EF5350', 'Large': '#FF8A65', 'Median': '#66BB6A', 'Minor': '#42A5F5', 'Small': '#1565C0', 'Insiders': '#7E57C2' }
SCHEDULE = {
    0:  {'Major': 1, 'Large': 2,  'Median': 3,  'Minor': 6,  'Small': 8},
    20: {'Major': 2, 'Large': 4,  'Median': 7,  'Minor': 14, 'Small': 24},
    40: {'Major': 3, 'Large': 7,  'Median': 12, 'Minor': 25, 'Small': 36},
    60: {'Major': 4, 'Large': 10, 'Median': 17, 'Minor': 35, 'Small': 57},
    80: {'Major': 6, 'Large': 12, 'Median': 20, 'Minor': 40, 'Small': 80},
}

class PVAnalyzer:
    def __init__(self, rmin=0.2, rmax=1.0, tmax=720, time_tranches=None):
        self.rmin = rmin
        self.rmax = rmax
        self.tmax = tmax
        self.time_tranches = time_tranches or [
            {"max_days": 180, "P": 0.2},
            {"max_days": 365, "P": 0.8},
            {"max_days": float('inf'), "P": 1.0}
        ]

    def calculate_time_weight(self, t_effective):
        weight = 0
        remaining_time = t_effective
        prev_limit = 0
        for tranche in self.time_tranches:
            limit = tranche['max_days']
            if limit != float('inf'):
                width = limit - prev_limit
            else:
                width = remaining_time   
            time_in_bracket = min(remaining_time, width)
            if time_in_bracket > 0:
                weight += np.log2(time_in_bracket + 1) * tranche['P']
            remaining_time -= time_in_bracket
            prev_limit = limit
            if remaining_time <= 0:
                break
        return weight

    def calculate_power_ratio(self, t_effective):
        progress = np.clip(t_effective / self.tmax, self.rmin, self.rmax)
        k = 0.5
        return self.rmin + (self.rmax - self.rmin) * (progress ** k)

    def calculate_voting_power(self, tokens, t_effective):
        if t_effective <= 0:
            return 0
        W = self.calculate_time_weight(t_effective)
        alpha = self.calculate_power_ratio(t_effective)
        return W * (alpha * tokens + np.sqrt(tokens) * (1 - alpha))


def build_stakeholders():
    all_stakeholders = []
    prev_counts = {cat: 0 for cat in CAT_TOKENS.keys()}
    for day in sorted(SCHEDULE.keys()):
        for cat, count in SCHEDULE[day].items():
            new_joiners = count - prev_counts[cat]
            for _ in range(new_joiners):
                all_stakeholders.append({
                    'category': cat,
                    'join_day': day,
                    'tokens': CAT_TOKENS[cat],
                    'buffer': 0,
                })
            prev_counts[cat] = count
    # Insiders constituency  
    all_stakeholders.append({
        'category': 'Insiders',
        'join_day': 0,
        'tokens': TEAM_TOKENS,
        'buffer': BUFFER_DAYS,
    })
    return all_stakeholders


def run_simulation(analyzer, max_days):
    all_stakeholders = build_stakeholders()
    all_cats = list(CAT_TOKENS.keys()) + ['Insiders']
    days = range(0, max_days + 1, 5)
    results = []
    for day in days:
        day_results = {'day': day}
        cat_vp_totals = {cat: 0 for cat in all_cats}
        for s in all_stakeholders:
            if s['join_day'] <= day:
                t_eff = (day - s['join_day']) + s['buffer']
                vp = analyzer.calculate_voting_power(s['tokens'], t_eff)
                cat_vp_totals[s['category']] += vp
        for cat, total_vp in cat_vp_totals.items():
            day_results[f'{cat}_Vp'] = total_vp
        results.append(day_results)
    return pd.DataFrame(results)


def plot_simulation(df, tranche_markers, filename=None):
    fig, ax = plt.subplots(figsize=(12, 7))

    all_cats = list(CAT_TOKENS.keys()) + ['Insiders']
    cat_lines = {}
    for cat in all_cats:
        line, = ax.plot(
            df['day'], df[f'{cat}_Vp'],
            label=cat,
            color=CAT_COLORS[cat],
            linewidth=2.0,
            linestyle='-',
        )
        cat_lines[cat] = line

    for i, (day, label) in enumerate(tranche_markers):
        ax.axvline(x=day, color='grey', linestyle='--', alpha=0.4)
        ax.text(day, ax.get_ylim()[1], label,
                color='black', alpha=1, fontsize=9,
                ha='center', va='bottom')

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Voting Power ($V_P$)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fname = filename or 'plot.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    graduated_tranches = [
        {"max_days": 260,          "P": 0.010},
        {"max_days": 520,          "P": 0.020},
        {"max_days": 780,          "P": 0.025},
        {"max_days": 1040,         "P": 0.030},
        {"max_days": 1300,         "P": 0.045},
        {"max_days": 1560,         "P": 0.050},
        {"max_days": float('inf'), "P": 0.070}
    ]
    graduated_analyzer = PVAnalyzer(tmax=T_MAX, time_tranches=graduated_tranches)
    graduated_df = run_simulation(graduated_analyzer, max_days=T_PERIOD)
    plot_simulation(
        graduated_df,
        tranche_markers=[
            (260,  '$T_1$ (260d)'),
            (520,  '$T_2$ (520d)'),
            (780,  '$T_3$ (780d)'),
            (1040, '$T_4$ (1040d)'),
            (1300, '$T_5$ (1300d)'),
            (1560, '$T_6$ (1560d)'),
            (T_MAX, f'$t_{{max}}$ ({T_MAX}d)'),
        ],
        filename='power_progression_graduated.png'
    )
