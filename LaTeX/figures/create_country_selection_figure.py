"""
Creates country selection figure showing only real WVS distributions.
Removes synthetic LLM responses per advisor feedback.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load WVS data
wvs_path = '/Users/milana/Documents/UniMannheim/Thesis/Scripts/WVS_Cross-National_Wave_7_csv_v6_0.csv'

# Country codes for Armenia, South Korea, Mexico
country_codes = {
    51: 'Armenia',
    410: 'South Korea',
    484: 'Mexico'
}

# Read WVS data
print("Loading WVS data...")
wvs = pd.read_csv(wvs_path, low_memory=False)

# Q49 is life satisfaction (1-10 scale)
# B_COUNTRY is country code
# Filter for our countries and valid responses
wvs_filtered = wvs[wvs['B_COUNTRY'].isin(country_codes.keys())].copy()
wvs_filtered = wvs_filtered[wvs_filtered['Q49'].between(1, 10)]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for idx, (country_code, country_name) in enumerate(country_codes.items()):
    ax = axes[idx]

    # Get data for this country
    country_data = wvs_filtered[wvs_filtered['B_COUNTRY'] == country_code]['Q49']

    # Calculate relative frequencies
    counts = country_data.value_counts().sort_index()
    rel_freq = counts / counts.sum()

    # Plot histogram
    bars = ax.bar(rel_freq.index, rel_freq.values, color='#5DADE2', edgecolor='#2874A6',
                  alpha=0.8, width=0.8)

    ax.set_xlabel('Life Satisfaction Score', fontsize=10)
    ax.set_ylabel('Relative Frequency', fontsize=10)
    ax.set_title(country_name, fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 0.4)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

plt.suptitle('Life Satisfaction Distributions: Country Selection Criteria',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
output_path = '/Users/milana/Documents/UniMannheim/MasterThesis/figures/fig_country_selection_distributions.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Figure saved to {output_path}")
