import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, log2

class QLVAnalyzer:
    def __init__(self, rmin=0.2, rmax=1.0, tmax=720):
        self.rmin = rmin
        self.rmax = rmax
        self.tmax = tmax
        self.k_const = rmax / log2(tmax)

    def calculate_time_weight(self, t_effective):
        """Refined 3-tranche log-summation: (180: 0.2), (365: 0.8), (inf: 1.0)"""
        time_tranches = [
            {"max_days": 180, "P": 0.2},
            {"max_days": 365, "P": 0.8},
            {"max_days": float('inf'), "P": 1.0}
        ]
        weight, remaining_time, prev_limit = 0, t_effective, 0
        for tranche in time_tranches:
            limit = tranche['max_days']
            width = limit - prev_limit if limit != float('inf') else remaining_time
            time_in_bracket = min(remaining_time, width)
            if time_in_bracket > 0:
                weight += np.log2(time_in_bracket + 1) * tranche['P']
            remaining_time -= time_in_bracket
            prev_limit = limit
            if remaining_time <= 0: break
        return weight

    def calculate_voting_power(self, tokens, t_effective):
        if t_effective <= 0: return 0
        W = self.calculate_time_weight(t_effective)
        # alpha progression - equation (4)
        alpha = max(self.rmin, min(self.rmax, self.rmin + (t_effective * (self.k_const / self.tmax))))
        
        linear_comp = alpha * tokens
        quad_comp = sqrt(tokens) * (1 - alpha)
        return W * (linear_comp + quad_comp)

def benchmark_profiles(max_days=720):
    analyzer = QLVAnalyzer(tmax=max_days)
    
    # Representative stakeholder join-days based on Table 1 start dates
    # Profiles: One from each tier joining at the earliest possible milestone
    profiles = [
        {'cat': 'Major',  'day': 0,  'tokens': 2000},
        {'cat': 'Large',  'day': 0,  'tokens': 1000},
        {'cat': 'Median', 'day': 0,  'tokens': 500},
        {'cat': 'Minor',  'day': 0,  'tokens': 200},
        {'cat': 'Small',  'day': 0,  'tokens': 50}
    ]
    
    days = range(0, max_days + 1, 5)
    results = []
    
    for day in days:
        day_results = {'day': day}
        for p in profiles:
            t_eff = max(0, day - p['day'])
            vp = analyzer.calculate_voting_power(p['tokens'], t_eff)
            day_results[f"{p['cat']}_Vp"] = vp
        results.append(day_results)
    
    df = pd.DataFrame(results)

    plt.figure(figsize=(12, 7))
    for p in profiles:
        plt.plot(df['day'], df[f"{p['cat']}_Vp"], label=f"Single {p['cat']} ({p['tokens']} tokens)")
    
    # Marker for Alpha hitting its stride
    plt.axvline(x=365, color='gray', linestyle='--', alpha=0.5, label='Tranche 2 Boundary')
    
    plt.title('Individual $V_P$ Benchmarks: One Profile Per Demographic')
    plt.xlabel('Days Since Day 0')
    plt.ylabel('Voting Power Units')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return df

if __name__ == "__main__":
    df = benchmark_profiles(720)
