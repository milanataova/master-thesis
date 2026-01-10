#!/usr/bin/env python3
"""
Ensemble Approach: Combining Scales with KS-Based Weighting

This script combines multiple questionnaire scales into an ensemble distribution
using KS statistics to determine optimal weights for each scale.

Ensemble includes: Original WVS (Base), Cantril, SWLS, and OHQ
Excluded from ensemble: Reverse (poor performance)

Outputs:
- Ensemble distributions for each country and model
- Ensemble metrics (KS and Wasserstein)
- Combined barplots showing all variants + ensemble
"""

import os
import re
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

# Variants to include in ensemble (excluding Reverse only)
ENSEMBLE_VARIANTS = ["Original WVS", "Cantril", "SWLS", "OHQ"]

# All variants for display
ALL_VARIANTS_WITH_ENSEMBLE = ["Original WVS", "Reverse", "Cantril", "SWLS", "OHQ", "Ensemble"]

# Map variant labels to questionnaire keys
VARIANT_TO_KEY = {
    "Original WVS": "base",
    "Reverse": "reverse",
    "Cantril": "cantril",
    "SWLS": "swls",
    "OHQ": "ohq",
}

# Colors for plots
VARIANT_COLORS = {
    "Original WVS": "#58C747D9",
    "Reverse": "#ff7e0ecd",
    "Cantril": "#66aada",
    "SWLS": "#9c27d696",
    "OHQ": "#ff5722",
    "Ensemble": "#BABDBF",
}

# Paths - using project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Thesis/

WVS_DATA_PATH = os.path.join(BASE_DIR, "Datasets", "WVS_Cross-National_Wave_7_csv_v6_0.csv")
METRICS_INPUT = os.path.join(BASE_DIR, "Outputs", "metrics_tables", "KS and Wasserstein metrics", "weighted_metrics.csv")
SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "synthetic_data", "weighted")
METRICS_OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs", "metrics_tables", "ensemble")
PLOTS_OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs", "plots", "ensemble")

os.makedirs(METRICS_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Helper Functions
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
    df["country"] = df["country"].astype(str).str.strip()

    df_sub = df[df["country"].isin(TARGET_COUNTRIES)].copy()
    df_sub = df_sub.dropna(subset=["life_satisfaction", "weight"])
    df_sub = df_sub[df_sub["life_satisfaction"].between(1, 10)]

    return df_sub[["country", "life_satisfaction", "weight"]]


def get_score_column(questionnaire_key):
    """Determine the score column name based on questionnaire type."""
    if questionnaire_key == 'ohq':
        return 'ohq_equated'
    elif questionnaire_key == 'swls':
        return 'swls_equated'
    else:
        return 'score'


def load_synthetic_dataset(variant_key: str, model: str) -> pd.DataFrame:
    """Load a synthetic dataset with proper score column."""
    filename = f"synthetic_{variant_key}_{model}_w.csv"
    filepath = os.path.join(SYNTHETIC_DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Synthetic file not found: {filepath}")

    score_col = get_score_column(variant_key)

    df = pd.read_csv(filepath)

    if score_col not in df.columns:
        raise ValueError(f"Expected column '{score_col}' not found in {filepath}")

    df = df.copy()
    df["life_satisfaction"] = df[score_col]
    df = df[df["life_satisfaction"].between(1, 10)]
    df["country"] = df["country"].astype(str).str.strip()

    # Use weight_joint as weight
    if "weight_joint" not in df.columns:
        raise ValueError(f"weight_joint column not found in {filepath}")

    df["weight"] = df["weight_joint"]

    return df[["country", "life_satisfaction", "weight"]]


def weighted_relative_frequency(series: pd.Series, weights: pd.Series) -> pd.Series:
    """Calculate weighted relative frequency for scores 1-10."""
    mask = (series >= 1) & (series <= 10) & series.notna() & weights.notna()
    s = series[mask]
    w = weights[mask]

    if len(s) == 0:
        return pd.Series(0.0, index=range(1, 11))

    # Calculate weighted counts for each score
    weighted_counts = {}
    for score in range(1, 11):
        score_mask = (s == score)
        weighted_counts[score] = w[score_mask].sum()

    counts = pd.Series(weighted_counts)

    # Normalize to get relative frequencies
    total = counts.sum()
    if total == 0:
        return pd.Series(0.0, index=range(1, 11))

    return counts / total


def ks_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate KS statistic between two discrete distributions."""
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.max(np.abs(cdf_p - cdf_q)))


def wasserstein_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Wasserstein distance between two discrete distributions."""
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)))


def compute_global_weights(metrics_df: pd.DataFrame, variants: list[str]) -> dict:
    """
    Compute ensemble weights based on average KS statistics.

    weights = (1 - KS) normalized, averaged across countries

    Parameters:
        metrics_df: DataFrame with Country, Model, Variant, KS columns
        variants: List of variant labels to include in ensemble

    Returns:
        Dictionary: model -> Series(variant -> weight)
    """
    # Filter for ensemble variants
    df_ens = metrics_df[metrics_df["Variant"].isin(variants)].copy()

    # Average KS across countries for each (Model, Variant)
    avg_ks = (
        df_ens
        .groupby(["Model", "Variant"])["KS"]
        .mean()
        .reset_index()
    )

    weights_by_model = {}

    for model in MODELS:
        model_data = avg_ks[avg_ks["Model"] == model].set_index("Variant")

        # Compute weights as (1 - KS)
        scores = 1.0 - model_data["KS"]
        scores = scores.clip(lower=0)

        # Normalize
        if scores.sum() == 0:
            weights = pd.Series(1.0 / len(scores), index=scores.index)
        else:
            weights = scores / scores.sum()

        weights_by_model[model] = weights

    return weights_by_model


# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("Ensemble Approach: KS-Based Scale Combination")
    print("="*80)

    # Step 1: Load real WVS data
    print("\nStep 1: Loading real WVS data...")
    df_wvs = load_wvs_data(WVS_DATA_PATH)

    # Calculate real distributions for each country
    real_dist = {}
    for country in TARGET_COUNTRIES:
        country_df = df_wvs[df_wvs["country"] == country]
        real_dist[country] = weighted_relative_frequency(
            country_df["life_satisfaction"],
            country_df["weight"]
        )

    print(f"✅ Loaded real data for {len(TARGET_COUNTRIES)} countries")

    # Step 2: Load existing metrics
    print("\nStep 2: Loading existing metrics...")
    metrics_df = pd.read_csv(METRICS_INPUT)

    # Ensure consistent formatting
    for col in ["Country", "Variant", "Model"]:
        metrics_df[col] = metrics_df[col].astype(str).str.strip()

    print(f"✅ Loaded {len(metrics_df)} metric records")

    # Step 3: Compute ensemble weights
    print("\nStep 3: Computing ensemble weights from KS statistics...")
    weights_by_model = compute_global_weights(metrics_df, ENSEMBLE_VARIANTS)

    print("\nEnsemble weights (w ∝ 1 - KS, averaged across countries):")
    for model, w in weights_by_model.items():
        print(f"\n{model}:")
        for variant, weight in w.items():
            print(f"  {variant:15s}: {weight:.3f}")
        print(f"  {'Total':15s}: {w.sum():.3f}")

    # Step 4: Load all synthetic data
    print("\nStep 4: Loading synthetic datasets...")
    synthetic_data = {}

    for variant_label, variant_key in VARIANT_TO_KEY.items():
        for model in MODELS:
            try:
                df = load_synthetic_dataset(variant_key, model)
                synthetic_data[(variant_label, model)] = df
                print(f"  ✅ Loaded {variant_label}/{model}")
            except Exception as e:
                print(f"  ❌ Error loading {variant_label}/{model}: {e}")

    print(f"✅ Loaded {len(synthetic_data)} synthetic datasets")

    # Step 5: Build ensemble distributions
    print("\nStep 5: Building ensemble distributions...")
    ensemble_rows = []
    ensemble_metrics_rows = []

    for model in MODELS:
        w = weights_by_model[model]  # Weights for ensemble variants

        for country in TARGET_COUNTRIES:
            # Initialize ensemble distribution
            ens = np.zeros(10, dtype=float)

            # Combine distributions with weights
            for variant_label, weight in w.items():
                if (variant_label, model) not in synthetic_data:
                    print(f"  ⚠️ Missing data for {variant_label}/{model}")
                    continue

                df_syn = synthetic_data[(variant_label, model)]
                country_df = df_syn[df_syn["country"] == country]

                if len(country_df) == 0:
                    print(f"  ⚠️ No data for {variant_label}/{model}/{country}")
                    continue

                # Get weighted relative frequency
                p = weighted_relative_frequency(
                    country_df["life_satisfaction"],
                    country_df["weight"]
                )

                ens += weight * p.values

            # Normalize ensemble
            total = ens.sum()
            if total > 0:
                ens /= total

            # Store ensemble distribution
            for score, prob in zip(range(1, 11), ens):
                ensemble_rows.append({
                    "Country": country,
                    "Model": model,
                    "Score": score,
                    "Ensemble_Prob": prob,
                })

            # Calculate metrics vs real
            real_p = real_dist[country].reindex(range(1, 11), fill_value=0).values
            ks_val = ks_discrete(real_p, ens)
            wass_val = wasserstein_discrete(real_p, ens)

            ensemble_metrics_rows.append({
                "Country": country,
                "Model": model,
                "Variant": "Ensemble",
                "KS": ks_val,
                "Wasserstein": wass_val,
            })

    # Convert to DataFrames
    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_metrics_df = pd.DataFrame(ensemble_metrics_rows)

    # Save ensemble data
    ensemble_df.to_csv(
        os.path.join(METRICS_OUTPUT_DIR, "ensemble_distributions.csv"),
        index=False
    )
    ensemble_metrics_df.to_csv(
        os.path.join(METRICS_OUTPUT_DIR, "ensemble_metrics.csv"),
        index=False
    )

    print(f"✅ Saved ensemble distributions and metrics")

    # Step 6: Combine metrics
    print("\nStep 6: Creating combined metrics table...")

    # Original metrics for all 5 questionnaires
    orig_metrics = metrics_df[
        metrics_df["Country"].isin(TARGET_COUNTRIES)
    ][["Country", "Model", "Variant", "Wasserstein", "KS"]].copy()

    # Add ensemble metrics
    combined_metrics = pd.concat(
        [orig_metrics, ensemble_metrics_df],
        ignore_index=True
    )

    # Ensure ordering
    combined_metrics["Variant"] = pd.Categorical(
        combined_metrics["Variant"],
        categories=ALL_VARIANTS_WITH_ENSEMBLE,
        ordered=True
    )

    combined_metrics.to_csv(
        os.path.join(METRICS_OUTPUT_DIR, "combined_metrics_with_ensemble.csv"),
        index=False
    )

    print(f"✅ Saved combined metrics table")

    # Step 7: Create barplots
    print("\nStep 7: Creating visualizations...")

    # Wasserstein barplot
    create_barplot(
        combined_metrics,
        metric="Wasserstein",
        ylabel="Wasserstein distance",
        title="Wasserstein distance by model, questionnaire, and ensemble across countries",
        output_file="wasserstein_with_ensemble.png"
    )

    # KS barplot
    create_barplot(
        combined_metrics,
        metric="KS",
        ylabel="KS statistic",
        title="KS statistic by model, questionnaire, and ensemble across countries",
        output_file="ks_with_ensemble.png"
    )

    print("\n" + "="*80)
    print("✅ Ensemble analysis complete!")
    print(f"✅ Metrics saved to: {METRICS_OUTPUT_DIR}/")
    print(f"✅ Plots saved to: {PLOTS_OUTPUT_DIR}/")
    print("="*80)

    # Print summary
    print("\nEnsemble Performance Summary:")
    print("\nAverage Metrics by Variant (across all countries and models):")
    summary = combined_metrics.groupby("Variant")[["KS", "Wasserstein"]].mean()
    print(summary.round(3))


def create_barplot(data, metric, ylabel, title, output_file):
    """Create a 2x2 barplot for the specified metric."""
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
        df_c = data[data["Country"] == country]

        n_models = len(MODELS)
        n_vars = len(ALL_VARIANTS_WITH_ENSEMBLE)

        indices = np.arange(n_models)
        bar_width = 0.125  # narrower to fit 6 bars

        local_ymax = 0.0
        label_info = []

        for j, var in enumerate(ALL_VARIANTS_WITH_ENSEMBLE):
            color = VARIANT_COLORS[var]
            offset = (j - (n_vars - 1) / 2) * bar_width
            x_pos = indices + offset

            heights = []
            for model in MODELS:
                row = df_c[(df_c["Model"] == model) & (df_c["Variant"] == var)]
                if row.empty:
                    h = 0.0
                else:
                    h = float(row[metric].iloc[0])
                heights.append(h)
                local_ymax = max(local_ymax, h)

            bars = ax.bar(
                x_pos,
                heights,
                width=bar_width,
                color=color,
                edgecolor="white",
                label=var,
            )

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
                ax.text(
                    x,
                    h + offset,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="medium",
                )

    if global_ymax > 0:
        ylim = global_ymax * 1.18
        for ax in axes:
            ax.set_ylim(0, ylim)

    fig.legend(
        legend_handles,
        legend_labels,
        title="Questionnaire",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(legend_labels),
        fontsize=10,
        title_fontsize=11,
    )

    fig.suptitle(title, fontsize=18, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = os.path.join(PLOTS_OUTPUT_DIR, output_file)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"✅ Saved {metric} barplot to: {out_path}")


if __name__ == "__main__":
    main()
