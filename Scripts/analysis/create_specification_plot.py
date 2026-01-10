"""
Specification Curve Plot for Master's Thesis
Subgroup Analysis - Synthetic Survey Generation for Life Satisfaction Research

This script creates specification curve visualizations showing how different
combinations of factors (model, questionnaire, country, health, income)
affect approximation quality (Wasserstein distance).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.axislines as axislines
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Path Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # Thesis/
DATA_DIR = BASE_DIR / "Outputs" / "metrics_tables" / "subgroup_analysis"
FIGURES_DIR = BASE_DIR / "Outputs" / "plots" / "subgroup_analysis"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
df_all = pd.read_csv(DATA_DIR / 'all_segment_metrics.csv')
df = df_all[df_all['level'] == 'Full'].copy()
print(f"  Using {len(df)} segment-level (Full) rows")

# Define display names and colors
MODEL_NAMES = {
    'llama3.1_8b': 'LLaMA 3.1 8B',
    'llama3.3_70b': 'LLaMA 3.3 70B',
    'qwen2.5_72b': 'Qwen 2.5 72B'
}

QUESTIONNAIRE_NAMES = {
    'base': 'Original WVS',
    'cantril': 'Cantril Ladder',
    'reverse': 'Reverse Scale',
    'swls': 'SWLS',
    'ohq': 'OHQ'
}

COUNTRY_NAMES = {
    'USA': 'USA',
    'IDN': 'Indonesia',
    'NLD': 'Netherlands',
    'MEX': 'Mexico'
}

HEALTH_ORDER = ['Good', 'Fair', 'Poor']
INCOME_ORDER = ['Low', 'Medium', 'High']
MODEL_ORDER = ['LLaMA 3.1 8B', 'LLaMA 3.3 70B', 'Qwen 2.5 72B']
COUNTRY_ORDER = ['USA', 'Indonesia', 'Netherlands', 'Mexico']
QUESTIONNAIRE_ORDER = ['Original WVS', 'Reverse Scale', 'Cantril Ladder', 'SWLS', 'OHQ']

# Colors for questionnaires (main grouping variable) - consistent with other thesis plots
QUESTIONNAIRE_COLORS = {
    'Original WVS': '#58C747',
    'Reverse Scale': '#ff7e0e',
    'Cantril Ladder': '#66aada',
    'SWLS': '#9c27d6',
    'OHQ': '#ff5722'
}

# Prepare data for plotting
print("Preparing plot data...")
plot_df = df.copy()
plot_df['Model'] = plot_df['model'].replace(MODEL_NAMES)
plot_df['Questionnaire'] = plot_df['questionnaire'].replace(QUESTIONNAIRE_NAMES)
plot_df['Country'] = plot_df['country'].replace(COUNTRY_NAMES)
plot_df['Health Status'] = plot_df['health_level']
plot_df['Income Level'] = plot_df['income_level']

# Define grouping columns for specification plot
GROUP_COLS = ['Questionnaire', 'Model', 'Country', 'Health Status', 'Income Level']

# Sort by Wasserstein distance (descending = worst first, best at right)
plot_df = plot_df.sort_values('wasserstein', ascending=False).reset_index(drop=True)

print(f"  Data range: W = {plot_df['wasserstein'].min():.2f} to {plot_df['wasserstein'].max():.2f}")

# ============================================================================
# Create Specification Plot
# ============================================================================
print("\nGenerating specification curve plot...")

fig = plt.figure(figsize=(12, 10))

# Create grid with custom height ratios
gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 3])

# Create axes using axislines for better customization
ax_curve = fig.add_subplot(gs[0], axes_class=axislines.Axes)
ax_specs = fig.add_subplot(gs[1], axes_class=axislines.Axes, sharex=ax_curve)

# ---- Top Panel: Wasserstein Distance Curve ----
for j in range(len(plot_df)):
    questionnaire = plot_df.loc[j, 'Questionnaire']
    color = QUESTIONNAIRE_COLORS.get(questionnaire, 'gray')
    ax_curve.errorbar(
        x=j,
        y=plot_df.loc[j, 'wasserstein'],
        fmt='|',
        ecolor=color,
        elinewidth=1,
        capsize=0,
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=6,
        markeredgewidth=1.5
    )

# Add horizontal reference lines for mean values
overall_mean = plot_df['wasserstein'].mean()
ax_curve.axhline(overall_mean, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax_curve.text(len(plot_df)-1, overall_mean + 0.1, f'Mean = {overall_mean:.2f}',
              va='bottom', ha='right', fontsize=9, color='gray')

ax_curve.set_ylabel('Wasserstein Distance', fontsize=11)
ax_curve.set_xlim([-2, len(plot_df)])

# ---- Bottom Panel: Specification Indicators ----
var_order = GROUP_COLS.copy()
var_order.reverse()  # Bottom to top

# Assign y-positions for each unique value within each variable
y_positions = {}
base_y = 1
y_offset = 1.2
ytick_labels = []
major_ytick_labels = []
ytick_positions = []
major_ytick_positions = []

for var in var_order:
    if var == 'Health Status':
        unique_vals = HEALTH_ORDER[::-1]
    elif var == 'Income Level':
        unique_vals = INCOME_ORDER[::-1]
    elif var == 'Model':
        unique_vals = MODEL_ORDER[::-1]
    elif var == 'Country':
        unique_vals = COUNTRY_ORDER[::-1]
    elif var == 'Questionnaire':
        unique_vals = QUESTIONNAIRE_ORDER[::-1]
    else:
        unique_vals = plot_df[var].unique()[::-1]

    for k, val in enumerate(unique_vals):
        y_positions[(var, str(val))] = base_y + k * y_offset
        ytick_labels.append(str(val))
        ytick_positions.append(base_y + k * y_offset)

    major_ytick_labels.append(var)
    major_ytick_positions.append(base_y + (len(unique_vals) - 0.3) * y_offset)
    base_y += len(unique_vals) + 1.5  # gap between groups

# Plot markers for each configuration
for j in range(len(plot_df)):
    for var in var_order:
        val = str(plot_df.loc[j, var])
        y = y_positions.get((var, val))
        if y is None:
            continue

        if var == 'Questionnaire':
            color = QUESTIONNAIRE_COLORS.get(val, 'gray')
        else:
            color = 'gray'

        ax_specs.plot(j, y, marker='|', color=color, markersize=8, markeredgewidth=1.2)

# Add horizontal lines for each row
for y in ytick_positions:
    ax_specs.axhline(y, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.6, zorder=0)

# Add vertical grid lines
vlines = list(range(50, len(plot_df), 50))
for x in vlines:
    ax_specs.axvline(x, color='lightgray', ls='-', linewidth=0.5, alpha=0.6, zorder=0)
    ax_curve.axvline(x, color='lightgray', ls='-', linewidth=0.5, alpha=0.6, zorder=0)

# Format y-axis
ax_specs.set_yticks(ytick_positions, labels=ytick_labels, fontsize=9, minor=True)
ax_specs.tick_params(axis='y', which='minor', labelsize=9)
ax_specs.set_yticks(major_ytick_positions, labels=major_ytick_labels, fontsize=10)
ax_specs.axis['left'].major_ticklabels.set(ha='right', va='bottom', fontsize=10, fontweight='bold', linespacing=1.1)
ax_specs.axis['left'].minor_ticklabels.set(fontsize=9)

# Format x-axis
ax_curve.set_xticks(vlines, labels=[str(i) for i in vlines])
ax_curve.set_xlabel('Specification (ranked by Wasserstein Distance)', fontsize=10)
ax_curve.axis['bottom'].major_ticklabels.set(fontsize=9)
ax_curve.axis['bottom'].label.set(fontsize=10)
ax_curve.axis['bottom'].major_ticks.set_tick_out(True)
ax_curve.axis['left'].major_ticklabels.set(fontsize=9)
ax_curve.axis['left'].label.set(fontsize=10)
ax_curve.axis['left'].major_ticks.set_tick_out(True)
ax_specs.axis['bottom'].major_ticklabels.set_visible(False)

# Remove unnecessary spines
for spine in ['top', 'right']:
    ax_specs.axis[spine].set_visible(False)
    ax_curve.axis[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax_specs.axis[spine].line.set_visible(False)
    ax_specs.axis[spine].major_ticks.set_ticksize(0)
    ax_specs.axis[spine].minor_ticks.set_ticksize(0)

# Adjust limits
xmin, xmax = ax_curve.get_xlim()
ax_curve.set_xlim(xmin - 5, xmax)
ax_curve.set_ylim(0, plot_df['wasserstein'].max() + 0.5)

ymin, ymax = ax_specs.get_ylim()
ax_specs.set_ylim(ymin - 1, ymax + 1)

# Title
ax_curve.set_title('Specification Curve: Approximation Quality Across All Configurations\n(540 segment-level comparisons, lower Wasserstein = better approximation)',
                   fontsize=12, fontweight='bold', pad=10)

# Add legend for questionnaires (using consistent order)
legend_handles = [plt.Line2D([0], [0], marker='|', color=QUESTIONNAIRE_COLORS[q], linestyle='None',
                              markersize=10, markeredgewidth=2, label=q)
                  for q in QUESTIONNAIRE_ORDER]
ax_curve.legend(handles=legend_handles, title='Questionnaire', loc='upper right',
                fontsize=8, title_fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.subplots_adjust(hspace=0.15)

# Save
plt.savefig(FIGURES_DIR / 'fig5_specification_curve.png', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'fig5_specification_curve.pdf', bbox_inches='tight')
print("  Saved: thesis_figures/fig5_specification_curve.png/pdf")

plt.close()

# ============================================================================
# Summary Statistics by Specification Groups
# ============================================================================
print("\n" + "="*60)
print("SPECIFICATION CURVE SUMMARY")
print("="*60)

print(f"\nTotal specifications: {len(plot_df)}")
print(f"Wasserstein range: {plot_df['wasserstein'].min():.3f} to {plot_df['wasserstein'].max():.3f}")
print(f"Mean: {plot_df['wasserstein'].mean():.3f}, Median: {plot_df['wasserstein'].median():.3f}")

# Worst 10 configurations (now at head due to descending sort)
print("\nTop 10 Worst Configurations (highest Wasserstein):")
print("-" * 60)
worst_10 = plot_df.head(10)[['Model', 'Questionnaire', 'Country', 'Health Status', 'Income Level', 'wasserstein']]
for i, row in worst_10.iterrows():
    print(f"  W={row['wasserstein']:.3f}: {row['Model']}, {row['Questionnaire']}, {row['Country']}, {row['Health Status']}, {row['Income Level']}")

# Best 10 configurations (now at tail due to descending sort)
print("\nTop 10 Best Configurations (lowest Wasserstein):")
print("-" * 60)
best_10 = plot_df.tail(10)[['Model', 'Questionnaire', 'Country', 'Health Status', 'Income Level', 'wasserstein']]
for i, row in best_10.iloc[::-1].iterrows():
    print(f"  W={row['wasserstein']:.3f}: {row['Model']}, {row['Questionnaire']}, {row['Country']}, {row['Health Status']}, {row['Income Level']}")

print("\n" + "="*60)
print("DONE")
print("="*60)
