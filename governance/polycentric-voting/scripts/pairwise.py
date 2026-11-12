import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from power_progression_agile import PVAnalyzer, run_simulation, CAT_TOKENS, SCHEDULE, T_MAX, T_PERIOD

agile_tranches = [
    {"max_days": 180,          "P": 0.020},
    {"max_days": 365,          "P": 0.035},
    {"max_days": float('inf'), "P": 0.070}
]
analyzer = PVAnalyzer(tmax=T_MAX, time_tranches=agile_tranches)

PAIRS = [
    ('Median', 'Minor'),
    ('Median', 'Large'),
    ('Large',  'Minor'),
    ('Major',  'Large'),
    ('Major',  'Minor'),
    ('Median', 'Major'),
    ('Small',  'Minor'),
    ('Median', 'Small'),
    ('Major',  'Small'),
]

N = len(PAIRS)
RIDGE_SPACING = 1.15
RIDGE_HEIGHT = 0.85
RIDGE_COLORS = [
    '#1B4F72', '#1F618D', '#148F77', '#0E6655', '#B7950B',
    '#B9770E', '#CA6F1E', '#BA4A00', '#A93226', '#78281F',
]

def lighten(hex_color, amount=0.7):
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = r + (255 - r) * amount
    g = g + (255 - g) * amount
    b = b + (255 - b) * amount
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

def build_static_stakeholders():
    final_counts = SCHEDULE[max(SCHEDULE.keys())]
    stakeholders = []
    for cat, count in final_counts.items():
        for _ in range(count):
            stakeholders.append({'category': cat, 'join_day': 0, 'tokens': CAT_TOKENS[cat]})
    return stakeholders

def run_static_simulation(analyzer, max_days):
    stakeholders = build_static_stakeholders()
    results = []
    for day in range(0, max_days + 1, 1):
        row = {'day': day}
        totals = {cat: 0 for cat in CAT_TOKENS.keys()}
        for s in stakeholders:
            t_eff = day - s['join_day']
            totals[s['category']] += analyzer.calculate_voting_power(s['tokens'], t_eff)
        row.update({f'{cat}_Vp': total for cat, total in totals.items()})
        results.append(row)
    return pd.DataFrame(results)

def plot_ridgeline(df, out_path):
    df = df[df['day'] > 0].reset_index(drop=True)
    days = df['day'].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 10))
    x0, x1 = 0, days[-1]
    CURVE_PAD_FRAC = 0.025

    for i, (num, den) in enumerate(PAIRS):
        label = f'{num} / {den}'
        ratio = (df[f'{num}_Vp'] / df[f'{den}_Vp']).to_numpy()
        rmin, rmax = ratio.min(), ratio.max()
        norm = (ratio - rmin) / (rmax - rmin) if rmax > rmin else np.zeros_like(ratio)

        baseline = (N - 1 - i) * RIDGE_SPACING
        inner_height = RIDGE_HEIGHT * (1 - 2 * CURVE_PAD_FRAC)
        y = baseline + RIDGE_HEIGHT * CURVE_PAD_FRAC + norm * inner_height

        days_plot = np.concatenate(([0], days))
        y_plot = np.concatenate(([y[0]], y))

        stroke_color = RIDGE_COLORS[i]
        fill_color = lighten(stroke_color, 0.7)

        ax.fill_between(days_plot, baseline, y_plot, color=fill_color, alpha=0.95,
                         linewidth=0, zorder=1, clip_on=False)
        ax.add_patch(Rectangle((x0, baseline), x1 - x0, RIDGE_HEIGHT,
                                facecolor='none', edgecolor='black', linewidth=1.0,
                                zorder=3, clip_on=False))
        ax.plot(days_plot, y_plot, color=stroke_color, linewidth=2.2, zorder=2,
                clip_on=False, solid_capstyle='butt')

        ax.text(x1 + 20, baseline + RIDGE_HEIGHT, f'{rmax:.2f}', fontsize=8,
                ha='left', va='center', color=stroke_color, zorder=5)
        ax.text(x1 + 20, baseline, f'{rmin:.2f}', fontsize=8,
                ha='left', va='center', color=stroke_color, zorder=5)
        ax.text(x1 + 20, baseline + RIDGE_HEIGHT / 2, f'{ratio[-1]:.2f}', fontsize=9,
                ha='left', va='center', color=stroke_color, fontweight='bold', zorder=5)

        ax.text(-25, baseline + RIDGE_HEIGHT * 0.35, label, fontsize=12,
                ha='right', va='center', color="black")

    ax.set_yticks([])
    ax.set_ylim(-0.05, (N - 1) * RIDGE_SPACING + RIDGE_HEIGHT + 0.15)
    ax.set_xlim(x0 - 55, x1 + 60)
    ax.set_xlabel('Time (days)')

    top = ax.get_ylim()[1]
    for day_mark, mlabel in [(180, '$T_{1}$ (180d)'), (365, '$T_{2}$ (365d)'), (720, '$t_{max}$ (720d)')]:
        ax.text(day_mark, top, mlabel, color='black', fontsize=9, ha='center', va='bottom')
        ax.axvline(x=day_mark, color='grey', linestyle='--', alpha=0.4)

    for spine in ['left', 'top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='x', pad=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close()

df_staggered = run_simulation(analyzer, max_days=T_PERIOD)
plot_ridgeline(df_staggered, './power_progression_ratios_ridgeline.png')

df_static = run_static_simulation(analyzer, max_days=T_PERIOD)
plot_ridgeline(df_static, './power_progression_static_schedule.png')
