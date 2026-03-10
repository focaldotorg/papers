import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, log2

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
        k = self.rmax / np.log2(self.tmax)
        alpha = max(self.rmin, min(self.rmax, self.rmin + (t_effective * k)))
        return alpha

    def calculate_voting_power(self, tokens, t_effective):
        if t_effective <= 0: return 0
        W = self.calculate_time_weight(t_effective)
        alpha = self.calculate_power_ratio(t_effective)
        linear_component = alpha * tokens
        quadratic_component = np.sqrt(tokens) * (1 - alpha)
        return W * (linear_component + quadratic_component)


# Share distribution
# Major:     5 * 2000  = 10,000 (10.0%)
# Large:    10 * 1000  = 10,000 (10.0%)
# Median:   20 * 500   = 10,000 (10.0%)
# Minor:    40 * 200   =  8,000  (8.0%)
# Small:   100 * 50    =  5,000  (5.0%)
# Subtotal stakeholders: 43,000 (43.0%)
# Operations:            40,000 (40.0%)
# Liquidity:             17,000 (17.0%)
# Total:                100,000 (100.0%)

cat_tokens = {'Major': 2000, 'Large': 1000, 'Median': 500, 'Minor': 200, 'Small': 50}

schedule = {
    0:  {'Major': 1, 'Large': 2,  'Median': 3,  'Minor': 6,  'Small': 5},
    20: {'Major': 2, 'Large': 4,  'Median': 7,  'Minor': 14, 'Small': 20},
    40: {'Major': 3, 'Large': 7,  'Median': 12, 'Minor': 25, 'Small': 45},
    60: {'Major': 4, 'Large': 9,  'Median': 17, 'Minor': 35, 'Small': 75},
    80: {'Major': 5, 'Large': 10, 'Median': 20, 'Minor': 40, 'Small': 100}
}


def build_stakeholders():
    all_stakeholders = []
    prev_counts = {cat: 0 for cat in cat_tokens.keys()}
    for day in sorted(schedule.keys()):
        for cat, count in schedule[day].items():
            new_joiners = count - prev_counts[cat]
            for _ in range(new_joiners):
                all_stakeholders.append({
                    'category': cat,
                    'join_day': day,
                    'tokens': cat_tokens[cat]
                })
            prev_counts[cat] = count
    return all_stakeholders


def run_simulation(analyzer, max_days):
    all_stakeholders = build_stakeholders()
    days = range(0, max_days + 1, 5)
    results = []

    for day in days:
        day_results = {'day': day}
        cat_vp_totals = {cat: 0 for cat in cat_tokens.keys()}
        for s in all_stakeholders:
            if s['join_day'] <= day:
                t_eff = day - s['join_day']
                vp = analyzer.calculate_voting_power(s['tokens'], t_eff)
                cat_vp_totals[s['category']] += vp
        for cat, total_vp in cat_vp_totals.items():
            day_results[f'{cat}_Vp'] = total_vp
        results.append(day_results)

    return pd.DataFrame(results)


def plot_simulation(df, title, tranche_markers):
    plt.figure(figsize=(12, 7))
    for cat in cat_tokens.keys():
        plt.plot(df['day'], df[f'{cat}_Vp'], label=f'{cat}')

    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink']
    for i, (day, label) in enumerate(tranche_markers):
        plt.axvline(x=day, color=colors[i % len(colors)], linestyle='--',
                    alpha=0.3, label=label)

    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Voting Power ($V_P$)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Agile configuration
    agile_tranches = [
        {"max_days": 180,          "P": 0.2},
        {"max_days": 365,          "P": 0.8},
        {"max_days": float('inf'), "P": 1.0}
    ]
    agile_analyzer = PVAnalyzer(tmax=720, time_tranches=agile_tranches)
    agile_df = run_simulation(agile_analyzer, max_days=720)
    plot_simulation(
        agile_df,
        title='Cumulative Coalition Voting Power: Agile Configuration',
        tranche_markers=[(180, 'T1 (180d)'), (365, 'T2 (365d)')]
    )

    # Graduated configuration
    graduated_tranches = [
        {"max_days": 260,          "P": 0.05},
        {"max_days": 520,          "P": 0.15},
        {"max_days": 780,          "P": 0.30},
        {"max_days": 1040,         "P": 0.50},
        {"max_days": 1300,         "P": 0.70},
        {"max_days": 1560,         "P": 0.85},
        {"max_days": float('inf'), "P": 1.0}
    ]
    graduated_analyzer = PVAnalyzer(tmax=1825, time_tranches=graduated_tranches)
    graduated_df = run_simulation(graduated_analyzer, max_days=1825)
    plot_simulation(
        graduated_df,
        title='Cumulative Coalition Voting Power: Graduated Configuration',
        tranche_markers=[
            (260,  'T1 (260d)'),
            (520,  'T2 (520d)'),
            (780,  'T3 (780d)'),
            (1040, 'T4 (1040d)'),
            (1300, 'T5 (1300d)'),
            (1560, 'T6 (1560d)')
        ]
    )
