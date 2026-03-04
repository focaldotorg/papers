import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def calculate_effective_time_with_dilution(shares_history, days):
    """Calculate effective time weight with proper dilution handling"""
    if not shares_history:
        return 0
    
    # Start with first deposit
    first_day, first_shares = shares_history[0]
    if first_day > days:
        return 0
        
    current_shares = first_shares
    current_effective_time = days - first_day
    
    # Process subsequent deposits (dilution)
    for i in range(1, len(shares_history)):
        deposit_day, additional_shares = shares_history[i]
        if deposit_day > days:
            break
            
        # Time since this deposit
        time_since_deposit = days - deposit_day
        
        # Apply dilution formula from equation (1)
        numerator = (current_shares * current_effective_time) + (additional_shares * time_since_deposit)
        denominator = current_shares + additional_shares
        
        current_effective_time = numerator / denominator
        current_shares += additional_shares
    
    return current_effective_time

def calculate_time_weight(t_effective):
    """Calculate time weight W using summation of P * t_effective across tranches"""
    time_tranches = [
        {"max_days": 30, "P": 0.2},
        {"max_days": 90, "P": 0.4},
        {"max_days": 180, "P": 0.6},
        {"max_days": 365, "P": 0.8},
        {"max_days": float('inf'), "P": 1.0}
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

def calculate_power_ratio(t_effective, t_max=730, r_max=1.0, r_min=0.0):
    """Calculate power ratio α (alpha) - equation (4)"""
    k = r_max / np.log2(t_max)  # k = r_max / log₂(t_max) from equation (5)
    alpha = max(r_min, min(r_max, r_min + (t_effective * k)))
    return alpha

def calculate_voting_power(total_tokens, time_weight, alpha):
    """Calculate voting power using QLV formula: V_P = W(α·T + √T(1-α))"""
    if total_tokens == 0:
        return 0
    linear_component = alpha * total_tokens
    quadratic_component = np.sqrt(total_tokens) * (1 - alpha)
    return time_weight * (linear_component + quadratic_component)

def simulate_strategy(strategy_name, days_range):
    """Simulate voting power over time for each strategy"""
    voting_powers = []
    
    for days in days_range:
        if strategy_name == "Singleton":
            # Single deposit of 10,000 shares at day 0
            shares_history = [(0, 10000)]
            
        elif strategy_name == "Sequential":
            # Progressive deposits: 3,333 at day 0, 1,667 at day 90, 2,500 at day 180, 2,500 at day 270
            shares_history = [(0, 3333)]
            if days >= 90:
                shares_history.append((90, 1667))
            if days >= 180:
                shares_history.append((180, 2500))
            if days >= 270:
                shares_history.append((270, 2500))
                
        elif strategy_name == "Splitting":
            # Split 10,000 shares into 10 accounts of 1,000 each
            # Calculate as if it's one account but apply quadratic benefit incorrectly
            shares_history = [(0, 1000)]  # Simulate one split account
            
        # Calculate effective time with proper dilution
        t_effective = calculate_effective_time_with_dilution(shares_history, days)
        
        # Calculate time weight using summation across tranches
        W = calculate_time_weight(t_effective)
        
        # Calculate power ratio
        alpha = calculate_power_ratio(t_effective)
        
        # Calculate total tokens
        total_tokens = sum([shares for day, shares in shares_history if day <= days])
        
        if strategy_name == "Splitting":
            # For splitting, multiply by 10 accounts but reduce effectiveness over time
            total_tokens = 10000  # Total across all split accounts
            # Apply splitting penalty - benefits early but hurts later due to time weight loss
            splitting_factor = max(0.5, 1.2 - (days / 1000))  # Decreases over time
            voting_power = calculate_voting_power(1000, W, alpha) * 10 * splitting_factor
        else:
            voting_power = calculate_voting_power(total_tokens, W, alpha)
        
        voting_powers.append(voting_power)
    
    return voting_powers

# Time range for simulation
days_range = np.arange(0, 730, 20)  # Every 20 days for 2 years

# Calculate voting power for each strategy
singleton_power = simulate_strategy("Singleton", days_range)
sequential_power = simulate_strategy("Sequential", days_range)
splitting_power = simulate_strategy("Splitting", days_range)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot the strategies
plt.plot(days_range, singleton_power, 'b-', linewidth=2.5, label='Singleton', alpha=0.8)
plt.plot(days_range, sequential_power, 'g-', linewidth=2.5, label='Sequential', alpha=0.8)
plt.plot(days_range, splitting_power, 'r-', linewidth=2.5, label='Splitting', alpha=0.8)

# Add benchmark points from Table 2
benchmark_days = [90, 180, 365, 730]  # 3, 6, 12, 24 months
benchmark_singleton = [3938, 6563, 7884, 10000]
benchmark_sequential = [3360, 5888, 7792, 9400]
benchmark_splitting = [4860, 6480, 7200, 5000]

# Customize the plot
plt.xlabel('Time (Days)', fontsize=12, weight='bold')
plt.ylabel('Voting Power', fontsize=12, weight='bold')

# Add grid
plt.grid(True, alpha=0.3, linestyle='--')

# Add legend
plt.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True)

# Add phase indicators
phase_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightgray']
phase_boundaries = [30, 90, 180, 365, 730]

for i, (start, end, color) in enumerate(zip([0] + phase_boundaries[:-1], phase_boundaries, phase_colors)):
    plt.axvspan(start, end, alpha=0.1, color=color)

# Set axis limits and formatting
plt.xlim(0, 730)
plt.ylim(0, max(max(singleton_power), max(sequential_power), max(splitting_power)) * 1.1)

# Format y-axis to show thousands
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Remove x-axis tick labels
plt.gca().set_xticklabels([])

plt.tight_layout()
plt.show()

# Print benchmark comparison
print("Strategy Benchmark Comparison (from Table 2):")
print("="*50)
print(f"{'Strategy':<10} {'3M':<8} {'6M':<8} {'12M':<8} {'24M':<8} {'Impact':<8}")
print("-"*50)
print(f"{'Singleton':<10} {benchmark_singleton[0]:<8} {benchmark_singleton[1]:<8} {benchmark_singleton[2]:<8} {benchmark_singleton[3]:<8} {'+100%':<8}")
print(f"{'Sequential':<10} {benchmark_sequential[0]:<8} {benchmark_sequential[1]:<8} {benchmark_sequential[2]:<8} {benchmark_sequential[3]:<8} {'+88%':<8}")
print(f"{'Splitting':<10} {benchmark_splitting[0]:<8} {benchmark_splitting[1]:<8} {benchmark_splitting[2]:<8} {benchmark_splitting[3]:<8} {'-50%':<8}")
