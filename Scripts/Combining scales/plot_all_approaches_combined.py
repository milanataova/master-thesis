#!/usr/bin/env python3
"""
Combined Visualization: All Approaches Together

Creates visualizations showing:
- 5 individual questionnaires (Original WVS, Reverse, Cantril, SWLS, OHQ)
- KS-based weighted ensemble
- Equal weights average ensemble

Generates barplots and scatterplots with all 7 variants for direct comparison.
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

# All variants including both ensembles
ALL_VARIANTS_COMBINED = ["Original WVS", "Reverse", "Cantril", "SWLS", "OHQ", "Ensemble", "Average"]

# Colors for plots
VARIANT_COLORS = {
    "Original WVS": "#58C747D9",
    "Reverse": "#ff7e0ecd",
    "Cantril": "#66aada",
    "SWLS": "#9c27d696",
    "OHQ": "#ff5722",
    "Ensemble": "#BABDBF",  # Light gray for KS ensemble
    "Average": "#95A5A6",   # Darker gray for average
}

# Line styles for scatterplots
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
    "Ensemble": {
        "color": "#BABDBF",
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
METRICS_OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs", "metrics_tables", "ensemble")
PLOTS_OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs", "plots", "ensemble")
ENSEMBLE_DISTRIBUTIONS_FILE = os.path.join(METRICS_OUTPUT_DIR, "ensemble_distributions.csv")
AVERAGE_DISTRIBUTIONS_FILE = os.path.join(METRICS_OUTPUT_DIR, "average_distributions.csv")
ENSEMBLE_METRICS_FILE = os.path.join(METRICS_OUTPUT_DIR, "ensemble_metrics.csv")
AVERAGE_METRICS_FILE = os.path.join(METRICS_OUTPUT_DIR, "average_metrics.csv")
ORIGINAL_METRICS_FILE = os.path.join(BASE_DIR, "Outputs", "metrics_tables", "KS and Wasserstein metrics", "weighted_metrics.csv")

os.makedirs(METRICS_OUTPUT_DIR, exist_ok=True)
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
        "B_COUNTRY_ALPHA": "country",
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
    df_categorized["income_level"] = df_categorized["income"].apply(categorize_income)
    df_categorized["health_category"] = df_categorized["health"].apply(categorize_health)
    return df_categorized


def prepare_comparison_subset(df_categorized, target_countries):
    """Prepare real WVS data for comparison."""
    essential_cols = ["life_satisfaction", "country", "income_level", "health_category", "weight"]
    df_subset = df_categorized[df_categorized["country"].isin(target_countries)][essential_cols].copy()
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


def load_ensemble_distributions(filepath, prob_column):
    """Load ensemble distributions from CSV."""
    df = pd.read_csv(filepath)
    ensemble_data = {}

    for (country, model), group in df.groupby(["Country", "Model"]):
        probs = group.set_index("Score")[prob_column]
        ensemble_data[(country, model)] = probs.reindex(range(1, 11), fill_value=0)

    return ensemble_data


def weighted_relative_frequency(series: pd.Series, weights: pd.Series) -> pd.Series:
    """Calculate weighted relative frequency for scores 1-10."""
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
# Barplot Creation
# ============================================================================

def create_combined_barplot(metric_name, ylabel, title, output_file):
    """Create a 2x2 barplot showing all 7 variants."""
    print(f"\nCreating combined barplot for {metric_name}...")

    # Load metrics
    orig_metrics = pd.read_csv(ORIGINAL_METRICS_FILE)
    ensemble_metrics = pd.read_csv(ENSEMBLE_METRICS_FILE)
    average_metrics = pd.read_csv(AVERAGE_METRICS_FILE)

    # Ensure consistent formatting
    for df in [orig_metrics, ensemble_metrics, average_metrics]:
        for col in ["Country", "Variant", "Model"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    # Combine all metrics
    combined_metrics = pd.concat([
        orig_metrics[orig_metrics["Country"].isin(TARGET_COUNTRIES)][["Country", "Model", "Variant", "Wasserstein", "KS"]],
        ensemble_metrics[["Country", "Model", "Variant", "KS", "Wasserstein"]].rename(columns={"KS": "KS", "Wasserstein": "Wasserstein"}),
        average_metrics[["Country", "Model", "Variant", "KS", "Wasserstein"]].rename(columns={"KS": "KS", "Wasserstein": "Wasserstein"})
    ], ignore_index=True)

    # Ensure ordering
    combined_metrics["Variant"] = pd.Categorical(
        combined_metrics["Variant"],
        categories=ALL_VARIANTS_COMBINED,
        ordered=True
    )

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True,
                            gridspec_kw={"wspace": 0.25, "hspace": 0.40})
    axes = axes.ravel()

    global_ymax = 0.0
    legend_handles = None
    legend_labels = None

    for idx, country in enumerate(TARGET_COUNTRIES):
        ax = axes[idx]
        df_c = combined_metrics[combined_metrics["Country"] == country]

        n_models = len(MODELS)
        n_vars = len(ALL_VARIANTS_COMBINED)

        indices = np.arange(n_models)
        bar_width = 0.11  # narrower to fit 7 bars

        local_ymax = 0.0
        label_info = []

        for j, var in enumerate(ALL_VARIANTS_COMBINED):
            color = VARIANT_COLORS[var]
            offset = (j - (n_vars - 1) / 2) * bar_width
            x_pos = indices + offset

            heights = []
            for model in MODELS:
                row = df_c[(df_c["Model"] == model) & (df_c["Variant"] == var)]
                if row.empty:
                    h = 0.0
                else:
                    h = float(row[metric_name].iloc[0])
                heights.append(h)
                local_ymax = max(local_ymax, h)

            bars = ax.bar(x_pos, heights, width=bar_width, color=color,
                         edgecolor="white", label=var)

            for bar, h in zip(bars, heights):
                if h > 0:
                    x = bar.get_x() + bar.get_width() / 2
                    label_info.append((x, h))

        global_ymax = max(global_ymax, local_ymax)

        ax.set_xticks(indices)
        ax.set_xticklabels(MODELS, fontsize=11)
        ax.set_title(country, fontsize=15)
        ax.grid(axis="y", alpha=0.25)

        if idx in [0, 2]:
            ax.set_ylabel(ylabel, fontsize=12)

        ax.tick_params(axis="y", labelsize=10)

        if legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            seen = {}
            for h, lab in zip(handles, labels):
                if lab not in seen:
                    seen[lab] = h
            legend_handles = list(seen.values())
            legend_labels = list(seen.keys())

        if label_info and local_ymax > 0:
            offset = local_ymax * 0.02
            for x, h in label_info:
                ax.text(x, h + offset, f"{h:.2f}", ha="center", va="bottom",
                       fontsize=7, fontweight="medium")

    if global_ymax > 0:
        ylim = global_ymax * 1.18
        for ax in axes:
            ax.set_ylim(0, ylim)

    fig.legend(legend_handles, legend_labels, title="Questionnaire",
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.96),
              ncol=len(legend_labels), fontsize=10, title_fontsize=11)

    fig.suptitle(title, fontsize=18, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = os.path.join(PLOTS_OUTPUT_DIR, output_file)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"✅ Saved combined barplot to: {out_path}")


# ============================================================================
# Scatterplot Creation
# ============================================================================

def create_combined_scatterplots():
    """Create scatterplots showing all variants including both ensembles."""
    print("\nCreating combined scatterplots...")

    # Step 1: Load real WVS data
    print("  Loading real WVS data...")
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

    # Step 2: Load synthetic data
    print("  Loading synthetic data...")
    synthetic_data = {}
    for q in QUESTIONNAIRES:
        for m in MODELS:
            try:
                df = load_synthetic_dataset(q, m)
                synthetic_data[(q, m)] = df
            except Exception as e:
                print(f"    ⚠️ Error loading {q}/{m}: {e}")

    # Step 3: Load ensemble distributions
    print("  Loading ensemble distributions...")
    ensemble_data = load_ensemble_distributions(ENSEMBLE_DISTRIBUTIONS_FILE, "Ensemble_Prob")
    average_data = load_ensemble_distributions(AVERAGE_DISTRIBUTIONS_FILE, "Average_Prob")

    # Step 4: Calculate global y-limit
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

    for (country, model), probs in ensemble_data.items():
        all_max.append(probs.max())
    for (country, model), probs in average_data.items():
        all_max.append(probs.max())

    y_max = max(all_max) if all_max else 0.3
    y_lim = (0, y_max * 1.15)

    # Step 5: Create plots
    print("  Creating scatterplots...")

    x_vals = list(range(1, 11))

    for country in TARGET_COUNTRIES:
        print(f"    Processing {country}...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(
            f"Life Satisfaction Distributions – All Approaches – {country}",
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
            ax.plot(x_vals, y_real.values, marker="o",
                   linestyle=s_real["linestyle"], linewidth=s_real["linewidth"],
                   markersize=s_real["markersize"], color=s_real["color"],
                   label="Real WVS")

            # Plot synthetic questionnaires
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

                ax.plot(x_vals, y.values, marker="o",
                       linestyle=s["linestyle"], linewidth=s["linewidth"],
                       markersize=s["markersize"], color=s["color"],
                       label=label)

            # Plot KS ensemble
            if (country, model) in ensemble_data:
                y_ens = ensemble_data[(country, model)]
                s_ens = LINE_STYLES["Ensemble"]
                ax.plot(x_vals, y_ens.values, marker="o",
                       linestyle=s_ens["linestyle"], linewidth=s_ens["linewidth"],
                       markersize=s_ens["markersize"], color=s_ens["color"],
                       label="Ensemble")

            # Plot Average
            if (country, model) in average_data:
                y_avg = average_data[(country, model)]
                s_avg = LINE_STYLES["Average"]
                ax.plot(x_vals, y_avg.values, marker="o",
                       linestyle=s_avg["linestyle"], linewidth=s_avg["linewidth"],
                       markersize=s_avg["markersize"], color=s_avg["color"],
                       label="Average")

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
            fig.legend(legend_handles, legend_labels, loc="upper center",
                      bbox_to_anchor=(0.5, 0.93), ncol=len(legend_labels),
                      frameon=False, fontsize=12, title="")

        plt.tight_layout(rect=[0, 0, 1, 0.86])

        out_path = os.path.join(PLOTS_OUTPUT_DIR, f"scatter_all_approaches_{country}.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"      ✅ Saved: {out_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("Combined Visualization: All Approaches Together")
    print("="*80)

    # Create barplots
    create_combined_barplot(
        metric_name="Wasserstein",
        ylabel="Wasserstein distance",
        title="Wasserstein distance – All approaches comparison",
        output_file="wasserstein_all_approaches.png"
    )

    create_combined_barplot(
        metric_name="KS",
        ylabel="KS statistic",
        title="KS statistic – All approaches comparison",
        output_file="ks_all_approaches.png"
    )

    # Create scatterplots
    create_combined_scatterplots()

    print("\n" + "="*80)
    print("✅ All combined visualizations complete!")
    print(f"✅ Output files saved to: {PLOTS_OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()
