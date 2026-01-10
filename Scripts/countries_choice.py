import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
wvs_data = pd.read_csv('WVS_Cross-National_Wave_7_csv_v6_0.csv')
synthetic_data = pd.read_csv('synthetic_responses.csv')

# Define countries and their full names for titles
countries = {
    'ARM': 'Armenia',
    'KOR': 'South Korea',
    'MEX': 'Mexico'
}

# Filter for the three countries of interest
wvs_filtered = wvs_data[wvs_data['B_COUNTRY_ALPHA'].isin(countries.keys())].copy()
synthetic_filtered = synthetic_data[synthetic_data['country'].isin(countries.keys())].copy()

# Q49 is the life satisfaction question in WVS (scale 1-10)
# Remove missing values (-1, -2, -4, etc. are typically missing value codes)
wvs_filtered = wvs_filtered[wvs_filtered['Q49'] > 0].copy()

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Life Satisfaction Score: Real WVS Data vs Synthetic LLM Responses',
             fontsize=16, fontweight='bold', y=1.02)

# Define bins for histogram (1-10 scale)
bins = np.arange(0.5, 11.5, 1)

# Plot for each country
for idx, (country_code, country_name) in enumerate(countries.items()):
    ax = axes[idx]

    # Get data for this country
    wvs_country = wvs_filtered[wvs_filtered['B_COUNTRY_ALPHA'] == country_code]
    synthetic_country = synthetic_filtered[synthetic_filtered['country'] == country_code]

    # Calculate weighted histogram for WVS data
    weights = wvs_country['W_WEIGHT'].values
    wvs_values = wvs_country['Q49'].values

    # Normalize weights to sum to 1 for relative frequency
    weights_normalized = weights / weights.sum()

    # Plot WVS data (weighted) - more transparent
    ax.hist(wvs_values, bins=bins, weights=weights_normalized,
            alpha=0.5, label='Real WVS (Weighted)', color='#2E86AB', edgecolor='white', linewidth=0.5)

    # Plot synthetic data with relative frequency
    synthetic_values = synthetic_country['response'].values
    synthetic_weights = np.ones(len(synthetic_values)) / len(synthetic_values)
    ax.hist(synthetic_values, bins=bins, weights=synthetic_weights,
            alpha=0.7, label='Synthetic LLM', color='#A23B72', edgecolor='white', linewidth=0.5)

    # Formatting
    ax.set_xlabel('Life Satisfaction Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'{country_name}', fontsize=14, fontweight='bold')
    # Place ticks in the middle of each score
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('life_satisfaction_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'life_satisfaction_comparison.png'")

# Print summary statistics
print("\n=== Summary Statistics ===\n")
for country_code, country_name in countries.items():
    wvs_country = wvs_filtered[wvs_filtered['B_COUNTRY_ALPHA'] == country_code]
    synthetic_country = synthetic_filtered[synthetic_filtered['country'] == country_code]

    # Weighted mean for WVS
    wvs_weighted_mean = np.average(wvs_country['Q49'], weights=wvs_country['W_WEIGHT'])
    synthetic_mean = synthetic_country['response'].mean()

    print(f"{country_name} ({country_code}):")
    print(f"  WVS - Weighted Mean: {wvs_weighted_mean:.2f}, n={len(wvs_country):,}")
    print(f"  Synthetic - Mean: {synthetic_mean:.2f}, n={len(synthetic_country):,}")
    print(f"  Difference: {synthetic_mean - wvs_weighted_mean:.2f}")
    print()
