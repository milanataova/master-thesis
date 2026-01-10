#!/usr/bin/env python3
"""
Generate Scatterplots with Average Ensemble Distributions

Creates one plot per country with 3 subplots (one per model).
Each subplot shows:
- Real WVS distribution (black line)
- All 5 synthetic questionnaire distributions (colored lines)
- Average ensemble distribution (gray line)

Uses weighted relative frequencies for both real and synthetic data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

TARGET_COUNTRIES = ["USA", "IDN", "NLD", "MEX"]

MODELS = [
    "llama3.1_8b",
    "llama3.3_70b",
    "qwen2.5_72b",
]

QUESTIONNAIRES = ["base", "reverse", "cantril", "swls", "ohq"]

QUESTIONNAIRE_LABELS = {
    "base": "Original WVS",
    "reverse": "Reverse",
    "cantril": "Cantril",
    "swls": "SWLS",
    "ohq": "OHQ",
}

# Line styles with thicker lines
LINE_STYLES = {
    "Real WVS": {
        "color": "black",
        "linestyle": "-",
        "linewidth": 2.7,
        "markersize": 7,
    },
    "Original WVS": {
        "color": "#58C747D9",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": 5,
    },
    "Reverse": {
        "color": "#ff7e0ecd",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": 5,
    },
    "Cantril": {
        "color": "#66aada",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": 5,
    },
    "SWLS": {
        "color": "#9c27d696",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": 5,
    },
    "OHQ": {
        "color": "#ff5722",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": 5,
    },
    "Average": {
        "color": "#95A5A6",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": 5,
    },
}

# Paths - using project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Thesis/

WVS_DATA_PATH = os.path.join(BASE_DIR, "Datasets", "WVS_Cross-National_Wave_7_csv_v6_0.csv")
SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "synthetic_data", "weighted")
AVERAGE_DISTRIBUTIONS_FILE = os.path.join(BASE_DIR, "Outputs", "metrics_tables", "ensemble", "average_distributions.csv")
PLOTS_OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs", "plots", "ensemble")

os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_wvs_data(file_path):
    """Load the WVS Wave 7 dataset with weights."""
    columns = {
        "Q49": "life_satisfaction",
        "Q288": "income",
        "Q47": "health",
        "Q260": "gender",
        "B_COUNTRY_ALPHA": "country",
        "Q262": "age",
        "X003R": "age_group",
        "Q270": "household_size",
        "Q273": "marital_status",
        "Q275": "education",
        "Q279": "employment_status",
        "Q287": "social_class",
        "Q289": "religion",
        "Q247": "num_children",
        "H_URBRURAL": "urban_rural",
        "Q2": "importance_friends",
        "Q71": "confidence_government",
        "W_WEIGHT": "weight",
    }

    print("Loading WVS Wave 7 dataset...")
    df = pd.read_csv(file_path, low_memory=False)
    df = df[list(columns.keys())].rename(columns=columns)

    missing_values = [-1, -2, -4, -5]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(missing_values, np.nan)

    df = df[df["life_satisfaction"].between(0, 10) | df["life_satisfaction"].isna()]
    return df


def categorize_dataframe(df):
    """Categorize income and health into groups."""
    def categorize_income(step):
        if step in [1, 2, 3]:
            return "Low"
        elif step in [4, 5, 6, 7]:
            return "Medium"
        elif step in [8, 9, 10]:
            return "High"
        return None

    def categorize_health(value):
        if value in [1, 2]:
            return "Good"
        elif value == 3:
            return "Fair"
        elif value in [4, 5]:
            return "Poor"
        return None

    df_categorized = df.copy()
    df_categorized["income_level"] = df_categorized["income"].apply(categorize_income)
    df_categorized["health_category"] = df_categorized["health"].apply(categorize_health)
    return df_categorized


def prepare_comparison_subset(df_categorized, target_countries):
    """Prepare real WVS data for comparison."""
    essential_cols = [
        "life_satisfaction", "country", "income_level", "health_category", "weight"
    ]
    df_subset = df_categorized[
        df_categorized["country"].isin(target_countries)
    ][essential_cols].copy()
    df_subset.dropna(inplace=True)
    return df_subset


def get_score_column(questionnaire):
    """Determine the score column name based on questionnaire type."""
    if questionnaire == 'ohq':
        return 'ohq_equated'
    elif questionnaire == 'swls':
        return 'swls_equated'
    else:
        return 'score'


def load_synthetic_dataset(questionnaire: str, model: str) -> pd.DataFrame:
    """Load a synthetic dataset."""
    filename = f"synthetic_{questionnaire}_{model}_w.csv"
    filepath = os.path.join(SYNTHETIC_DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Synthetic file not found: {filepath}")

    score_col = get_score_column(questionnaire)

    df = pd.read_csv(filepath)

    if score_col not in df.columns:
        raise ValueError(f"Expected column '{score_col}' not found in {filepath}")

    df = df.copy()
    df["life_satisfaction"] = df[score_col]
    df = df[df["life_satisfaction"].between(1, 10)]

    if "weight_joint" not in df.columns:
        raise ValueError(f"weight_joint column not found in {filepath}")

    df["weight"] = df["weight_joint"]

    return df[["country", "life_satisfaction", "weight"]]


def load_average_distributions(filepath):
    """Load average distributions from CSV."""
    df = pd.read_csv(filepath)

    # Pivot to get probabilities by score for each (country, model)
    average_data = {}

    for (country, model), group in df.groupby(["Country", "Model"]):
        # Create a series indexed by score (1-10)
        probs = group.set_index("Score")["Average_Prob"]
        average_data[(country, model)] = probs.reindex(range(1, 11), fill_value=0)

    return average_data


# ============================================================================
# Frequency Calculation
# ============================================================================

def weighted_relative_frequency(series: pd.Series, weights: pd.Series) -> pd.Series:
    """Calculate weighted relative frequency for life satisfaction scores 1-10."""
    mask = (series >= 1) & (series <= 10) & series.notna() & weights.notna()
    s = series[mask]
    w = weights[mask]

    if len(s) == 0:
        return pd.Series(0.0, index=range(1, 11))

    weighted_counts = {}
    for score in range(1, 11):
        score_mask = (s == score)
        weighted_counts[score] = w[score_mask].sum()

    counts = pd.Series(weighted_counts)

    total = counts.sum()
    if total == 0:
        return pd.Series(0.0, index=range(1, 11))

    return counts / total


# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("Life Satisfaction Distribution Scatterplots with Average Ensemble")
    print("="*80)

    # Step 1: Load real WVS data
    print("\nStep 1: Loading real WVS data...")
    df_base = load_wvs_data(WVS_DATA_PATH)
    df_categorized_base = categorize_dataframe(df_base)
    df_for_comparison = prepare_comparison_subset(df_categorized_base, TARGET_COUNTRIES)

    real_dist = {}
    for country in TARGET_COUNTRIES:
        country_df = df_for_comparison[df_for_comparison["country"] == country]
        real_dist[country] = weighted_relative_frequency(
            country_df["life_satisfaction"],
            country_df["weight"]
        )

    print(f"✅ Loaded real data for {len(TARGET_COUNTRIES)} countries")

    # Step 2: Load synthetic data
    print("\nStep 2: Loading synthetic data...")
    synthetic_data = {}

    for q in QUESTIONNAIRES:
        for m in MODELS:
            try:
                df = load_synthetic_dataset(q, m)
                synthetic_data[(q, m)] = df
                print(f"  ✅ Loaded {q}/{m}")
            except Exception as e:
                print(f"  ❌ Error loading {q}/{m}: {e}")

    print(f"✅ Loaded {len(synthetic_data)} synthetic datasets")

    # Step 3: Load average distributions
    print("\nStep 3: Loading average ensemble distributions...")
    average_data = load_average_distributions(AVERAGE_DISTRIBUTIONS_FILE)
    print(f"✅ Loaded average data for {len(average_data)} (country, model) combinations")

    # Step 4: Calculate global y-limit
    print("\nStep 4: Calculating global y-limit...")
    all_max = []

    for country in TARGET_COUNTRIES:
        all_max.append(real_dist[country].max())

    for (q, m), df_syn in synthetic_data.items():
        for country in TARGET_COUNTRIES:
            country_df = df_syn[df_syn["country"] == country]
            if len(country_df) > 0:
                freq = weighted_relative_frequency(
                    country_df["life_satisfaction"],
                    country_df["weight"]
                )
                all_max.append(freq.max())

    # Also check average max
    for (country, model), probs in average_data.items():
        all_max.append(probs.max())

    y_max = max(all_max) if all_max else 0.3
    y_lim = (0, y_max * 1.15)

    print(f"✅ Global y-limit: {y_lim}")

    # Step 5: Create plots
    print("\nStep 5: Creating scatterplots with average ensemble...")

    x_vals = list(range(1, 11))

    for country in TARGET_COUNTRIES:
        print(f"  Creating plot for {country}...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(
            f"Life Satisfaction Distributions with Average Ensemble – {country}",
            fontsize=20,
            y=0.98,
        )

        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        legend_handles, legend_labels = None, None

        for ax, model in zip(axes, MODELS):
            # Plot Real WVS
            y_real = real_dist[country]
            s_real = LINE_STYLES["Real WVS"]
            ax.plot(
                x_vals,
                y_real.values,
                marker="o",
                linestyle=s_real["linestyle"],
                linewidth=s_real["linewidth"],
                markersize=s_real["markersize"],
                color=s_real["color"],
                label="Real WVS",
            )

            # Plot synthetic questionnaires for this model
            for q in QUESTIONNAIRES:
                if (q, model) not in synthetic_data:
                    continue

                df_syn = synthetic_data[(q, model)]
                country_df = df_syn[df_syn["country"] == country]

                if len(country_df) == 0:
                    continue

                y = weighted_relative_frequency(
                    country_df["life_satisfaction"],
                    country_df["weight"]
                )

                label = QUESTIONNAIRE_LABELS[q]
                s = LINE_STYLES[label]

                ax.plot(
                    x_vals,
                    y.values,
                    marker="o",
                    linestyle=s["linestyle"],
                    linewidth=s["linewidth"],
                    markersize=s["markersize"],
                    color=s["color"],
                    label=label,
                )

            # Plot average ensemble
            if (country, model) in average_data:
                y_avg = average_data[(country, model)]
                s_avg = LINE_STYLES["Average"]

                ax.plot(
                    x_vals,
                    y_avg.values,
                    marker="o",
                    linestyle=s_avg["linestyle"],
                    linewidth=s_avg["linewidth"],
                    markersize=s_avg["markersize"],
                    color=s_avg["color"],
                    label="Average",
                )

            ax.set_title(model, fontsize=14)
            ax.set_xlabel("Life satisfaction (1–10)", fontsize=14)
            ax.set_xticks(x_vals)
            ax.tick_params(axis="both", labelsize=14)
            ax.set_ylim(y_lim)
            ax.grid(axis="y", alpha=0.25)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        axes[0].set_ylabel("Relative frequency", fontsize=14)
        axes[0].tick_params(axis="y", labelsize=14)

        # Shared legend
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.93),
                ncol=len(legend_labels),
                frameon=False,
                fontsize=13,
                title="",
            )

        plt.tight_layout(rect=[0, 0, 1, 0.86])

        out_path = os.path.join(PLOTS_OUTPUT_DIR, f"scatter_with_average_{country}.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"    ✅ Saved: {out_path}")

    print("\n" + "="*80)
    print("✅ All scatterplots with average ensemble complete!")
    print(f"✅ Output files saved to: {PLOTS_OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()
