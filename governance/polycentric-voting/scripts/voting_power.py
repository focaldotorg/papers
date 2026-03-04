import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, log2

class QLVAnalyzer:
    def __init__(self, rmin=0.2, rmax=1.0, tmax=720):
        self.rmin = rmin
        self.rmax = rmax
        self.tmax = tmax
    
    def calculate_effective_time_weight(self, shares_history, current_day):
        """Calculate effective time weight with proper dilution handling"""
        if not shares_history:
            return 0, 0
    
        history = [h for h in shares_history if h[0] <= current_day]
        if not history:
            return 0, 0
        
        first_day, current_shares = history[0]
        current_effective_time = current_day - first_day
    
        for i in range(1, len(history)):
            deposit_day, additional_shares = history[i]
            time_since_deposit = current_day - deposit_day
        
            # Dilution formula: T_eff = (S1*T1 + S2*T2) / (S1 + S2)
            numerator = (current_shares * current_effective_time) + (additional_shares * time_since_deposit)
            denominator = current_shares + additional_shares
        
            current_effective_time = numerator / denominator
            current_shares += additional_shares
    
        return current_effective_time, current_shares

    def calculate_time_weight(self, t_effective):
        """Calculate time weight W using summation of P * t_effective across tranches"""
        time_tranches = [
            {"max_days": 180, "P": 0.2},
            {"max_days": 365, "P": 0.8},
            {"max_days": float('inf'), "P": 1}
        ]
    
        weight = 0
        remaining_time = t_effective

        for i, tranche in enumerate(time_tranches):
            tranche_time = min(remaining_time, 
                tranche['max_days'] if i == 0 else tranche['max_days'] - time_tranches[i-1]['max_days'])
        
            if tranche_time > 0:
                # Use logarithmic time weighting for more gradual progression
                weight += np.log2(tranche_time + 1) * tranche['P']
        
            remaining_time -= tranche_time
            if remaining_time <= 0:
                break
    
        return weight
    
    def calculate_power_ratio(self, t_effective):
        """Calculate power ratio α (alpha) - equation (4)"""
        k = self.rmax / np.log2(self.tmax)  
        alpha = max(self.rmin, min(self.rmax, self.rmin + (t_effective * k)))
    
        return alpha

    def calculate_voting_power(self, tokens, t_effective):
        """Standard QLV Power Formula: V_P = W * (αT + √T(1-α))"""
        W = self.calculate_time_weight(t_effective)
        alpha = self.calculate_power_ratio(t_effective)
        
        linear_component = alpha * tokens 
        quadratic_component = np.sqrt(tokens) * (1 - alpha)

        return W * (linear_component + quadratic_component)

def run_simulation(max_days=720):
    analyzer = QLVAnalyzer(tmax=max_days)
    
    # Category Token Values
    cat_tokens = {'Major': 2000, 'Large': 1000, 'Median': 500, 'Minor': 200, 'Small': 50}
    
    # Expanded Stakeholder Schedule (Table 1)
    schedule = {
        0:  {'Major': 1, 'Large': 2, 'Median': 3, 'Minor': 6, 'Small': 2},
        20: {'Major': 2, 'Large': 4, 'Median': 7, 'Minor': 14, 'Small': 8},
        40: {'Major': 3, 'Large': 7, 'Median': 12, 'Minor': 25, 'Small': 18},
        60: {'Major': 4, 'Large': 9, 'Median': 17, 'Minor': 35, 'Small': 30},
        80: {'Major': 5, 'Large': 10, 'Median': 20, 'Minor': 40, 'Small': 40}
    }

    # 1. Generate INDEPENDENT stakeholders
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

    # 2. Run Simulation Day-by-Day
    days = range(0, max_days + 1, 5)
    results = []
    
    for day in days:
        day_results = {'day': day}
        # Reset totals for each category for this specific day
        cat_vp_totals = {cat: 0 for cat in cat_tokens.keys()}
        
        for s in all_stakeholders:
            # Only process if they have already joined
            if s['join_day'] <= day:
                t_eff = day - s['join_day']
                vp = analyzer.calculate_voting_power(s['tokens'], t_eff)
                cat_vp_totals[s['category']] += vp
        
        # Merge totals into results
        for cat, total_vp in cat_vp_totals.items():
            day_results[f'{cat}_Vp'] = total_vp
            
        results.append(day_results)
    
    df = pd.DataFrame(results)
    
    # 3. Plotting
    plt.figure(figsize=(12, 7))
    for cat in cat_tokens.keys():
        plt.plot(df['day'], df[f'{cat}_Vp'], label=f'Total {cat} Demographic')
    
    plt.axvline(x=180, color='red', linestyle='--', alpha=0.3, label='Tranche 1 (180d)')
    plt.axvline(x=365, color='green', linestyle='--', alpha=0.3, label='Tranche 2 (365d)')
    
    plt.title('Demographic Influence: Independent Stakeholder Summation')
    plt.xlabel('Days')
    plt.ylabel('Aggregate Voting Power ($V_P$)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
    
    return df

if __name__ == "__main__":
    df = run_simulation(720)
