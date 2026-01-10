#!/usr/bin/env python3
"""
Calculate Weighted Metrics and Generate Bar Plots

This script:
1. Calculates weighted Kolmogorov-Smirnov (KS) and Wasserstein distance statistics
   comparing real WVS Wave 7 data with synthetic datasets
2. Generates 2x2 grouped bar plots for visualization

Comparison is performed for each combination of:
- Questionnaire type (base, cantril, reverse, swls, ohq)
- Model (llama3.1_8b, llama3.3_70b, qwen2.5_72b)
- Country (USA, IDN, NLD, MEX)

Input:
- WVS Wave 7 dataset (real data with weights)
- Synthetic datasets with weight_joint column

Output:
- Outputs/metrics_tables/weighted_metrics.csv
- Outputs/plots/barplots/wasserstein_grouped_all_countries.png
- Outputs/plots/barplots/ks_grouped_all_countries.png
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# Get the base directory (Thesis/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Data paths
WVS_DATA_PATH = os.path.join(BASE_DIR, "Datasets", "WVS_Cross-National_Wave_7_csv_v6_0.csv")
SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "synthetic_data", "weighted")

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs")
METRICS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "metrics_tables", "KS and Wasserstein metrics")
PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "plots", "barplots")

# Target countries and models
TARGET_COUNTRIES = ["USA", "IDN", "NLD", "MEX"]
MODELS = ["llama3.1_8b", "llama3.3_70b", "qwen2.5_72b"]
QUESTIONNAIRES = ["base", "reverse", "cantril", "swls", "ohq"]

# Display labels
QUESTIONNAIRE_LABELS = {
    "base": "Original WVS",
    "reverse": "Reverse",
    "cantril": "Cantril",
    "swls": "SWLS",
    "ohq": "OHQ",
}

# Colors for plots
VARIANT_COLORS = {
    "Original WVS": "#58C747D9",
    "Reverse": "#ff7e0ecd",
    "Cantril": "#66aada",
    "SWLS": "#9c27d696",
    "OHQ": "#ff5722",
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_wvs_data(file_path):
    """Load the WVS Wave 7 dataset with weights."""
    columns = {
        "Q49": "life_satisfaction",
        "Q288": "income",
        "Q47": "health",
        "B_COUNTRY_ALPHA": "country",
        "W_WEIGHT": "weight"
    }

    print(f"Loading WVS Wave 7 dataset from:\n  {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    df = df[list(columns.keys())].rename(columns=columns)

    # Replace negative values with NaN
    missing_values = [-1, -2, -4, -5]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(missing_values, np.nan)

    df = df[df["life_satisfaction"].between(0, 10) | df["life_satisfaction"].isna()]
    return df


def categorize_dataframe(df):
    """Categorize income and health into groups."""
    def categorize_income(step):
        if step in [1, 2, 3]: return "Low"
        elif step in [4, 5, 6, 7]: return "Medium"
        elif step in [8, 9, 10]: return "High"
        return None

    def categorize_health(value):
        if value in [1, 2]: return "Good"
        elif value == 3: return "Fair"
        elif value in [4, 5]: return "Poor"
        return None

    df_categorized = df.copy()
    df_categorized['income_level'] = df_categorized['income'].apply(categorize_income)
    df_categorized['health_level'] = df_categorized['health'].apply(categorize_health)
    return df_categorized


def prepare_real_data(file_path, target_countries):
    """Load and prepare real WVS data."""
    df = load_wvs_data(file_path)
    df = categorize_dataframe(df)

    essential_cols = ["life_satisfaction", "country", "income_level", "health_level", "weight"]
    df = df[df['country'].isin(target_countries)][essential_cols].copy()
    df.dropna(inplace=True)

    print(f"  Loaded {len(df):,} real WVS records for {len(target_countries)} countries")
    return df


def get_score_column(questionnaire):
    """Determine the score column name based on questionnaire type."""
    if questionnaire == 'ohq':
        return 'ohq_equated'
    elif questionnaire == 'swls':
        return 'swls_equated'
    else:
        return 'score'


def load_synthetic_data(filepath, questionnaire):
    """Load a synthetic dataset."""
    score_col = get_score_column(questionnaire)

    df = pd.read_csv(filepath)

    if score_col not in df.columns:
        raise ValueError(f"Expected column '{score_col}' not found in {filepath}")

    required_cols = ['country', score_col, 'weight_joint']
    df = df[required_cols].copy()
    df.rename(columns={score_col: 'score'}, inplace=True)
    df.dropna(inplace=True)

    return df


# ============================================================================
# Statistical Calculation Functions
# ============================================================================

def weighted_ks_statistic(data1, weights1, data2, weights2):
    """
    Calculate weighted Kolmogorov-Smirnov statistic.

    The KS statistic is the maximum absolute difference between
    the weighted empirical cumulative distribution functions (ECDFs).
    """
    data1 = np.asarray(data1)
    weights1 = np.asarray(weights1)
    data2 = np.asarray(data2)
    weights2 = np.asarray(weights2)

    # Normalize weights
    weights1 = weights1 / np.sum(weights1)
    weights2 = weights2 / np.sum(weights2)

    # Combine all unique data points
    all_data = np.concatenate([data1, data2])
    all_data = np.unique(all_data)
    all_data = np.sort(all_data)

    # Calculate weighted ECDFs
    ecdf1 = np.zeros(len(all_data))
    ecdf2 = np.zeros(len(all_data))

    for i, x in enumerate(all_data):
        ecdf1[i] = np.sum(weights1[data1 <= x])
        ecdf2[i] = np.sum(weights2[data2 <= x])

    # KS statistic is maximum absolute difference
    ks_stat = np.max(np.abs(ecdf1 - ecdf2))
    return ks_stat


def calculate_statistics(real_data, real_weights, synth_data, synth_weights):
    """Calculate weighted KS and Wasserstein statistics."""
    real_data = np.asarray(real_data)
    real_weights = np.asarray(real_weights)
    synth_data = np.asarray(synth_data)
    synth_weights = np.asarray(synth_weights)

    ks_stat = weighted_ks_statistic(real_data, real_weights, synth_data, synth_weights)

    wasserstein_dist = wasserstein_distance(
        real_data, synth_data,
        u_weights=real_weights, v_weights=synth_weights
    )

    return {
        'ks_stat': ks_stat,
        'wasserstein_dist': wasserstein_dist,
    }


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_all_metrics(real_df, synthetic_dir, target_countries, models, questionnaires):
    """Calculate metrics for all combinations."""
    print("\nCalculating metrics for all combinations...")

    results = {
        'Country': [],
        'Model': [],
        'Variant': [],
        'KS': [],
        'Wasserstein': []
    }

    # Process each combination
    for country in target_countries:
        for model in models:
            for questionnaire in questionnaires:
                variant_label = QUESTIONNAIRE_LABELS[questionnaire]

                # Build the filename
                filename = f"synthetic_{questionnaire}_{model}_w.csv"
                filepath = os.path.join(synthetic_dir, filename)

                if not os.path.exists(filepath):
                    print(f"  Warning: File not found: {filename}")
                    continue

                try:
                    # Load synthetic data
                    synth_df = load_synthetic_data(filepath, questionnaire)

                    # Filter real data for this country
                    real_country = real_df[real_df['country'] == country].copy()

                    # Filter synthetic data for this country
                    synth_country = synth_df[synth_df['country'] == country].copy()

                    if len(real_country) == 0 or len(synth_country) == 0:
                        print(f"  Warning: No data for {country}/{questionnaire}/{model}")
                        continue

                    # Extract data and weights
                    real_scores = real_country['life_satisfaction'].values
                    real_weights = real_country['weight'].values
                    synth_scores = synth_country['score'].values
                    synth_weights = synth_country['weight_joint'].values

                    # Calculate statistics
                    stats = calculate_statistics(
                        real_scores, real_weights,
                        synth_scores, synth_weights
                    )

                    # Store results
                    results['Country'].append(country)
                    results['Model'].append(model)
                    results['Variant'].append(variant_label)
                    results['KS'].append(stats['ks_stat'])
                    results['Wasserstein'].append(stats['wasserstein_dist'])

                except Exception as e:
                    print(f"  Error processing {filepath}: {e}")
                    continue

    metrics_df = pd.DataFrame(results)
    print(f"  Calculated metrics for {len(metrics_df)} combinations")

    return metrics_df


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_metric_grid(metrics_df, metric_name, metric_column, ylabel, output_path):
    """
    Create a 2x2 grid plot for a specific metric.
    """
    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 9),
        sharey=True,
        gridspec_kw={"wspace": 0.25, "hspace": 0.40},
    )

    axes = axes.ravel()

    global_ymax = 0.0
    legend_handles = None
    legend_labels = None

    for idx, country in enumerate(TARGET_COUNTRIES):
        ax = axes[idx]
        df_c = metrics_df[metrics_df["Country"] == country].copy()

        n_models = len(MODELS)
        n_vars = len(QUESTIONNAIRES)

        indices = np.arange(n_models)
        bar_width = 0.15

        label_info = []
        local_ymax = 0.0

        # Draw grouped bars
        for j, q in enumerate(QUESTIONNAIRES):
            var_label = QUESTIONNAIRE_LABELS[q]
            color = VARIANT_COLORS[var_label]

            # Center variants around each model index
            offset = (j - (n_vars - 1) / 2) * bar_width
            x_positions = indices + offset

            heights = []
            for model in MODELS:
                row = df_c[(df_c["Model"] == model) & (df_c["Variant"] == var_label)]
                value = float(row[metric_column].iloc[0]) if not row.empty else 0.0
                heights.append(value)
                local_ymax = max(local_ymax, value)

            bars = ax.bar(
                x_positions,
                heights,
                width=bar_width,
                color=color,
                edgecolor="white",
                label=var_label,
            )

            for bar, h in zip(bars, heights):
                if h > 0:
                    x = bar.get_x() + bar.get_width() / 2
                    label_info.append((x, h))

        global_ymax = max(global_ymax, local_ymax)

        # Axes formatting
        ax.set_xticks(indices)
        ax.set_xticklabels(MODELS, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(country, fontsize=15)

        if idx in [0, 2]:
            ax.set_ylabel(ylabel, fontsize=12)

        ax.tick_params(axis="y", labelsize=10)

        # Capture legend once
        if legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            seen = {}
            for h, lab in zip(handles, labels):
                if lab not in seen:
                    seen[lab] = h
            legend_labels = list(seen.keys())
            legend_handles = list(seen.values())

        # Data labels
        if label_info and local_ymax > 0:
            offset_val = local_ymax * 0.02
            for x, h in label_info:
                ax.text(
                    x,
                    h + offset_val,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="medium",
                )

    # Shared y-limit
    if global_ymax > 0:
        ylim = global_ymax * 1.18
        for ax in axes:
            ax.set_ylim(0, ylim)

    # Shared legend
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Questionnaire",
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=len(legend_labels),
            fontsize=11,
            title_fontsize=12,
        )

    fig.suptitle(
        f"{metric_name} by model and questionnaire across countries",
        fontsize=18,
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"  Saved: {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("Calculate Weighted Metrics and Generate Bar Plots")
    print("=" * 80)

    # Create output directories
    os.makedirs(METRICS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

    # Step 1: Load real WVS data
    print("\nStep 1: Loading real WVS data...")
    real_df = prepare_real_data(WVS_DATA_PATH, TARGET_COUNTRIES)

    # Step 2: Calculate metrics
    print("\nStep 2: Calculating metrics...")
    metrics_df = calculate_all_metrics(
        real_df,
        SYNTHETIC_DATA_DIR,
        TARGET_COUNTRIES,
        MODELS,
        QUESTIONNAIRES
    )

    # Save metrics to CSV
    metrics_output = os.path.join(METRICS_OUTPUT_DIR, "weighted_metrics.csv")
    metrics_df.to_csv(metrics_output, index=False)
    print(f"\n  Saved metrics to: {metrics_output}")

    # Step 3: Create visualizations
    print("\nStep 3: Creating visualizations...")

    # Wasserstein Distance plot
    plot_metric_grid(
        metrics_df,
        metric_name="Wasserstein distance",
        metric_column="Wasserstein",
        ylabel="Wasserstein distance",
        output_path=os.path.join(PLOTS_OUTPUT_DIR, "wasserstein_grouped_all_countries.png")
    )

    # KS Statistic plot
    plot_metric_grid(
        metrics_df,
        metric_name="Kolmogorov-Smirnov statistic",
        metric_column="KS",
        ylabel="KS statistic",
        output_path=os.path.join(PLOTS_OUTPUT_DIR, "ks_grouped_all_countries.png")
    )

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)

    # Display summary
    print("\nSummary Statistics:")
    print("\nWasserstein Distance by Questionnaire:")
    wass_summary = metrics_df.groupby('Variant')['Wasserstein'].agg(['mean', 'std', 'min', 'max']).round(3)
    print(wass_summary)

    print("\nKS Statistic by Questionnaire:")
    ks_summary = metrics_df.groupby('Variant')['KS'].agg(['mean', 'std', 'min', 'max']).round(3)
    print(ks_summary)


if __name__ == "__main__":
    main()
