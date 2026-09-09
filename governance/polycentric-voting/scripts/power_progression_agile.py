import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from math import sqrt, log2

T_MAX = 720
T_PERIOD = 1000 
CAT_TOKENS = {'Major': 2000, 'Large': 1000, 'Median': 500, 'Minor': 200, 'Small': 100}
CAT_COLORS = {'Major': '#EF5350', 'Large': '#FF8A65', 'Median': '#66BB6A', 'Minor': '#42A5F5', 'Small': '#1565C0'}
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
            width = limit - prev_limit if limit != float('inf') else remaining_time
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
                    'tokens': CAT_TOKENS[cat]
                })
            prev_counts[cat] = count
    return all_stakeholders


def run_simulation(analyzer, max_days):
    all_stakeholders = build_stakeholders()
    days = range(0, max_days + 1, 5)
    results = []
    for day in days:
        day_results = {'day': day}
        cat_vp_totals = {cat: 0 for cat in CAT_TOKENS.keys()}
        for s in all_stakeholders:
            if s['join_day'] <= day:
                t_eff = day - s['join_day']
                vp = analyzer.calculate_voting_power(s['tokens'], t_eff)
                cat_vp_totals[s['category']] += vp
        for cat, total_vp in cat_vp_totals.items():
            day_results[f'{cat}_Vp'] = total_vp
        results.append(day_results)
    return pd.DataFrame(results)


def plot_simulation(df, tranche_markers, zoom_window=None, inset_pos=None, filename=None):
    fig, ax = plt.subplots(figsize=(12, 7))

    cat_lines = {}
    for cat in CAT_TOKENS.keys():
        line, = ax.plot(
            df['day'], df[f'{cat}_Vp'],
            label=cat,
            color=CAT_COLORS[cat],
            linewidth=2.0
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

    if zoom_window:
        x0, x1 = zoom_window
        pos = inset_pos or [0.58, 0.05, 0.40, 0.50]
        axins = ax.inset_axes(pos)

        for cat in CAT_TOKENS.keys():
            mask = (df['day'] >= x0) & (df['day'] <= x1)
            axins.plot(
                df.loc[mask, 'day'], df.loc[mask, f'{cat}_Vp'],
                color=CAT_COLORS[cat],
                linewidth=1.5
            )

        y_vals = []
        for cat in CAT_TOKENS.keys():
            mask = (df['day'] >= x0) & (df['day'] <= x1)
            y_vals.extend(df.loc[mask, f'{cat}_Vp'].tolist())

        axins.set_xlim(x0, x1)
        axins.set_ylim(min(y_vals) * 0.95, max(y_vals) * 1.05)
        axins.set_title(f'Zoom: days {x0}–{x1}', fontsize=8)
        axins.grid(True, alpha=0.2)
        axins.tick_params(labelsize=7)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)

    plt.tight_layout()
    fname = filename or 'plot.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    agile_tranches = [
        {"max_days": 180,          "P": 0.020},
        {"max_days": 365,          "P": 0.035},
        {"max_days": float('inf'), "P": 0.070}
    ]
    agile_analyzer = PVAnalyzer(tmax=T_MAX, time_tranches=agile_tranches)
    agile_df = run_simulation(agile_analyzer, max_days=T_PERIOD)
    plot_simulation(
        agile_df,
        tranche_markers=[(180, '$T_1$ (180d)'), (365, '$T_2$ (365d)'), (720, '$t_{max}$ (720d)' )],
        zoom_window=(10, 90),
        filename='power_progression_agile.png'
    )
