"""
Score-Level Accuracy Analysis for Life Satisfaction Synthetic Data

This script analyzes which specific life satisfaction scores (1-10) are predicted
more accurately by LLMs compared to real WVS data. It computes:
1. Per-score absolute prediction error (difference between real and synthetic proportions)
2. Per-score relative prediction error (percentage deviation from real proportions)
3. Aggregated metrics across models, questionnaires, and countries

Author: Master's Thesis Analysis
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - use dynamic path resolution
SCRIPT_DIR = Path(__file__).parent
BASE_PATH = SCRIPT_DIR.parent  # Thesis/
DATASETS_PATH = BASE_PATH / "Datasets"
WVS_PATH = DATASETS_PATH / "WVS_Cross-National_Wave_7_csv_v6_0.csv"
SYNTHETIC_DATA_PATH = DATASETS_PATH / "synthetic_data" / "weighted"

# Output paths
TABLES_OUTPUT_PATH = BASE_PATH / "Outputs" / "metrics_tables" / "score_level_analysis"
PLOTS_OUTPUT_PATH = BASE_PATH / "Outputs" / "plots" / "score_level_analysis"
TABLES_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Countries and models
COUNTRIES = ["USA", "IDN", "NLD", "MEX"]
COUNTRY_NAMES = {"USA": "United States", "IDN": "Indonesia", "NLD": "Netherlands", "MEX": "Mexico"}
MODELS = ["llama3.1_8b", "llama3.3_70b", "qwen2.5_72b"]
MODEL_NAMES = {"llama3.1_8b": "LLaMA 3.1 8B", "llama3.3_70b": "LLaMA 3.3 70B", "qwen2.5_72b": "Qwen 2.5 72B"}
QUESTIONNAIRES = ["base", "cantril", "reverse", "swls", "ohq"]
QUESTIONNAIRE_NAMES = {"base": "Original WVS", "cantril": "Cantril", "reverse": "Reverse", "swls": "SWLS", "ohq": "OHQ"}

# Plotting style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
sns.set_theme(style="whitegrid")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_wvs_data():
    """Load and preprocess WVS Wave 7 data."""
    columns = {
        "Q49": "life_satisfaction",
        "Q288": "income",
        "Q47": "health",
        "B_COUNTRY_ALPHA": "country",
        "W_WEIGHT": "weight"
    }

    df = pd.read_csv(WVS_PATH, low_memory=False)
    df = df[list(columns.keys())].rename(columns=columns)

    # Replace missing values
    for col in ['life_satisfaction', 'income', 'health']:
        df[col] = df[col].replace([-1, -2, -4, -5], np.nan)

    # Filter valid life satisfaction (1-10 scale)
    df = df[df['life_satisfaction'].between(1, 10)]
    df = df[df['country'].isin(COUNTRIES)]

    return df

def load_synthetic_data(questionnaire, model):
    """Load synthetic data for a specific questionnaire-model combination.

    All synthetic data uses the weighted files (_w.csv) from synthetic_data/weighted/.
    For SWLS and OHQ, uses equipercentile-equated scores (swls_equated, ohq_equated columns).
    """
    file_path = SYNTHETIC_DATA_PATH / f"synthetic_{questionnaire}_{model}_w.csv"

    if not file_path.exists():
        return None

    df = pd.read_csv(file_path, on_bad_lines='skip')

    # For SWLS and OHQ, use the equated score column
    if questionnaire in ['swls', 'ohq']:
        equated_col = f'{questionnaire}_equated'
        if equated_col in df.columns:
            df['score'] = df[equated_col]
            df = df[df['score'].between(1, 10)]
    else:
        # Ensure score is in 1-10 range
        if 'score' in df.columns:
            df = df[df['score'].between(1, 10)]

    return df

# ============================================================================
# SCORE-LEVEL ANALYSIS
# ============================================================================

def compute_score_distributions(df_real, df_synthetic, country):
    """
    Compute the distribution of scores for real and synthetic data.
    Returns proportions for each score (1-10).
    """
    # Real data distribution (weighted)
    real_country = df_real[df_real['country'] == country].copy()
    real_counts = []
    for score in range(1, 11):
        weight_sum = real_country[real_country['life_satisfaction'] == score]['weight'].sum()
        real_counts.append(weight_sum)
    real_total = sum(real_counts)
    real_props = [c / real_total if real_total > 0 else 0 for c in real_counts]

    # Synthetic data distribution
    synth_country = df_synthetic[df_synthetic['country'] == country].copy()
    synth_counts = []
    for score in range(1, 11):
        count = len(synth_country[synth_country['score'] == score])
        synth_counts.append(count)
    synth_total = sum(synth_counts)
    synth_props = [c / synth_total if synth_total > 0 else 0 for c in synth_counts]

    return real_props, synth_props

def compute_score_level_metrics(real_props, synth_props):
    """
    Compute per-score accuracy metrics.

    Returns:
    - absolute_error: |real_prop - synth_prop| for each score
    - relative_error: |real_prop - synth_prop| / real_prop for each score (where real_prop > 0)
    - squared_error: (real_prop - synth_prop)^2 for each score
    """
    scores = list(range(1, 11))
    metrics = []

    for i, score in enumerate(scores):
        real = real_props[i]
        synth = synth_props[i]

        abs_error = abs(real - synth)
        rel_error = abs(real - synth) / real if real > 0.001 else np.nan
        squared_error = (real - synth) ** 2
        direction = "over" if synth > real else "under"

        metrics.append({
            'score': score,
            'real_proportion': real,
            'synthetic_proportion': synth,
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'squared_error': squared_error,
            'direction': direction
        })

    return pd.DataFrame(metrics)

def run_full_analysis(df_real):
    """Run score-level analysis across all models, questionnaires, and countries."""
    all_results = []

    for model in MODELS:
        for questionnaire in QUESTIONNAIRES:
            df_synthetic = load_synthetic_data(questionnaire, model)
            if df_synthetic is None:
                continue

            for country in COUNTRIES:
                real_props, synth_props = compute_score_distributions(df_real, df_synthetic, country)
                metrics_df = compute_score_level_metrics(real_props, synth_props)

                metrics_df['model'] = model
                metrics_df['questionnaire'] = questionnaire
                metrics_df['country'] = country

                all_results.append(metrics_df)

    return pd.concat(all_results, ignore_index=True)

# ============================================================================
# AGGREGATION AND SUMMARY STATISTICS
# ============================================================================

def compute_summary_by_score(results_df):
    """Compute summary statistics by score (averaged across all conditions)."""
    summary = results_df.groupby('score').agg({
        'absolute_error': ['mean', 'std', 'median'],
        'relative_error': ['mean', 'std', 'median'],
        'real_proportion': 'mean',
        'synthetic_proportion': 'mean'
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()

def compute_summary_by_score_and_model(results_df):
    """Compute summary by score and model."""
    summary = results_df.groupby(['score', 'model']).agg({
        'absolute_error': ['mean', 'std'],
        'relative_error': 'mean',
        'real_proportion': 'mean',
        'synthetic_proportion': 'mean'
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()

def compute_summary_by_score_and_questionnaire(results_df):
    """Compute summary by score and questionnaire."""
    summary = results_df.groupby(['score', 'questionnaire']).agg({
        'absolute_error': ['mean', 'std'],
        'relative_error': 'mean',
        'real_proportion': 'mean',
        'synthetic_proportion': 'mean'
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()

def compute_summary_by_score_and_country(results_df):
    """Compute summary by score and country."""
    summary = results_df.groupby(['score', 'country']).agg({
        'absolute_error': ['mean', 'std'],
        'relative_error': 'mean',
        'real_proportion': 'mean',
        'synthetic_proportion': 'mean'
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_score_level_accuracy_heatmap(results_df):
    """Create heatmap of absolute error by score (rows) vs model-questionnaire (cols)."""
    # Aggregate by score, model, questionnaire
    pivot_data = results_df.groupby(['score', 'model', 'questionnaire'])['absolute_error'].mean().reset_index()
    pivot_data['model_quest'] = pivot_data['model'].map(MODEL_NAMES) + '\n' + pivot_data['questionnaire'].map(QUESTIONNAIRE_NAMES)

    heatmap_data = pivot_data.pivot(index='score', columns='model_quest', values='absolute_error')

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'Mean Absolute Error'})
    ax.set_xlabel('Model - Questionnaire')
    ax.set_ylabel('Life Satisfaction Score')
    ax.set_title('Score-Level Prediction Accuracy by Model and Questionnaire\n(Lower values = better approximation)')
    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_PATH / 'score_accuracy_heatmap_full.png', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_OUTPUT_PATH / 'score_accuracy_heatmap_full.pdf', bbox_inches='tight')
    plt.close()

def plot_score_accuracy_by_model(results_df):
    """Line plot of absolute error by score for each model."""
    summary = compute_summary_by_score_and_model(results_df)

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in MODELS:
        model_data = summary[summary['model'] == model]
        ax.plot(model_data['score'], model_data['absolute_error_mean'],
                marker='o', label=MODEL_NAMES[model], linewidth=2, markersize=8)
        # Add confidence bands (using std)
        ax.fill_between(model_data['score'],
                       model_data['absolute_error_mean'] - model_data['absolute_error_std'],
                       model_data['absolute_error_mean'] + model_data['absolute_error_std'],
                       alpha=0.2)

    ax.set_xlabel('Life Satisfaction Score')
    ax.set_ylabel('Mean Absolute Error (Proportion)')
    ax.set_title('Score-Level Prediction Accuracy by Model')
    ax.set_xticks(range(1, 11))
    ax.legend(title='Model')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_PATH / 'score_accuracy_by_model.png', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_OUTPUT_PATH / 'score_accuracy_by_model.pdf', bbox_inches='tight')
    plt.close()

def plot_score_accuracy_by_questionnaire(results_df):
    """Line plot of absolute error by score for each questionnaire."""
    summary = compute_summary_by_score_and_questionnaire(results_df)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'base': '#2ecc71', 'cantril': '#3498db', 'reverse': '#e74c3c',
              'swls': '#9b59b6', 'ohq': '#e67e22'}

    for quest in QUESTIONNAIRES:
        quest_data = summary[summary['questionnaire'] == quest]
        ax.plot(quest_data['score'], quest_data['absolute_error_mean'],
                marker='o', label=QUESTIONNAIRE_NAMES[quest], linewidth=2,
                markersize=8, color=colors[quest])

    ax.set_xlabel('Life Satisfaction Score')
    ax.set_ylabel('Mean Absolute Error (Proportion)')
    ax.set_title('Score-Level Prediction Accuracy by Questionnaire Type')
    ax.set_xticks(range(1, 11))
    ax.legend(title='Questionnaire')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_PATH / 'score_accuracy_by_questionnaire.png', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_OUTPUT_PATH / 'score_accuracy_by_questionnaire.pdf', bbox_inches='tight')
    plt.close()

def plot_score_accuracy_by_country(results_df):
    """Line plot of absolute error by score for each country."""
    summary = compute_summary_by_score_and_country(results_df)

    fig, ax = plt.subplots(figsize=(10, 6))

    for country in COUNTRIES:
        country_data = summary[summary['country'] == country]
        ax.plot(country_data['score'], country_data['absolute_error_mean'],
                marker='o', label=COUNTRY_NAMES[country], linewidth=2, markersize=8)

    ax.set_xlabel('Life Satisfaction Score')
    ax.set_ylabel('Mean Absolute Error (Proportion)')
    ax.set_title('Score-Level Prediction Accuracy by Country')
    ax.set_xticks(range(1, 11))
    ax.legend(title='Country')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_PATH / 'score_accuracy_by_country.png', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_OUTPUT_PATH / 'score_accuracy_by_country.pdf', bbox_inches='tight')
    plt.close()

def plot_overall_score_accuracy(results_df):
    """Bar plot of overall accuracy by score with error bars."""
    summary = compute_summary_by_score(results_df)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = summary['score']
    y = summary['absolute_error_mean']
    yerr = summary['absolute_error_std']

    bars = ax.bar(x, y, yerr=yerr, capsize=5, color='steelblue', edgecolor='navy', alpha=0.8)

    # Color-code bars by error magnitude
    colors = plt.cm.RdYlGn_r(y / y.max())
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)

    ax.set_xlabel('Life Satisfaction Score')
    ax.set_ylabel('Mean Absolute Error (Proportion)')
    ax.set_title('Overall Score-Level Prediction Accuracy\n(Lower bars = better approximation)')
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(f'{yi:.3f}', xy=(xi, yi), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_PATH / 'overall_score_accuracy.png', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_OUTPUT_PATH / 'overall_score_accuracy.pdf', bbox_inches='tight')
    plt.close()

def plot_real_vs_synthetic_comparison(results_df):
    """
    Create a comparison plot showing real vs synthetic proportions for each score,
    faceted by model.
    """
    summary = compute_summary_by_score_and_model(results_df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        model_data = summary[summary['model'] == model]

        x = np.arange(1, 11)
        width = 0.35

        bars1 = ax.bar(x - width/2, model_data['real_proportion_mean'], width,
                       label='Real WVS', color='#2c3e50', alpha=0.8)
        bars2 = ax.bar(x + width/2, model_data['synthetic_proportion_mean'], width,
                       label='Synthetic', color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Life Satisfaction Score')
        if idx == 0:
            ax.set_ylabel('Mean Proportion')
        ax.set_title(MODEL_NAMES[model])
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Real vs Synthetic Score Distributions by Model', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_PATH / 'real_vs_synthetic_by_model.png', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_OUTPUT_PATH / 'real_vs_synthetic_by_model.pdf', bbox_inches='tight')
    plt.close()

def plot_direction_analysis(results_df):
    """
    Analyze whether LLMs tend to over- or under-predict specific scores.
    """
    # Compute mean difference (synth - real) for each score
    results_df['difference'] = results_df['synthetic_proportion'] - results_df['real_proportion']

    direction_summary = results_df.groupby('score')['difference'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in direction_summary['mean']]

    bars = ax.bar(direction_summary['score'], direction_summary['mean'],
                  yerr=direction_summary['std'], capsize=5, color=colors,
                  edgecolor='black', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Life Satisfaction Score')
    ax.set_ylabel('Mean Proportion Difference (Synthetic - Real)')
    ax.set_title('LLM Prediction Bias by Score\n(Positive = Over-prediction, Negative = Under-prediction)')
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Over-prediction'),
                       Patch(facecolor='#2ecc71', label='Under-prediction')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_PATH / 'prediction_direction_by_score.png', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_OUTPUT_PATH / 'prediction_direction_by_score.pdf', bbox_inches='tight')
    plt.close()

def plot_combined_faceted(results_df):
    """
    Create a combined faceted plot showing score-level accuracy across
    questionnaires (rows) and models (columns).
    """
    fig, axes = plt.subplots(len(QUESTIONNAIRES), len(MODELS),
                              figsize=(14, 16), sharex=True, sharey=True)

    for row_idx, quest in enumerate(QUESTIONNAIRES):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]

            # Filter data
            subset = results_df[(results_df['questionnaire'] == quest) &
                               (results_df['model'] == model)]

            # Aggregate across countries
            score_data = subset.groupby('score').agg({
                'real_proportion': 'mean',
                'synthetic_proportion': 'mean',
                'absolute_error': 'mean'
            }).reset_index()

            x = score_data['score']
            width = 0.35

            ax.bar(x - width/2, score_data['real_proportion'], width,
                   label='Real', color='#2c3e50', alpha=0.8)
            ax.bar(x + width/2, score_data['synthetic_proportion'], width,
                   label='Synthetic', color='#e74c3c', alpha=0.8)

            # Row and column labels
            if row_idx == 0:
                ax.set_title(MODEL_NAMES[model], fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(QUESTIONNAIRE_NAMES[quest], fontsize=10)
            if row_idx == len(QUESTIONNAIRES) - 1:
                ax.set_xlabel('Score')

            ax.set_xticks(range(1, 11))
            ax.grid(True, alpha=0.3, axis='y')

    # Add legend to first subplot
    axes[0, 0].legend(loc='upper left', fontsize=8)

    plt.suptitle('Score-Level Distributions: Real vs Synthetic\n(by Questionnaire and Model)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_PATH / 'combined_faceted_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_OUTPUT_PATH / 'combined_faceted_comparison.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistical_tests(results_df):
    """
    Perform statistical tests to assess significance of score-level differences.
    """
    from scipy.stats import f_oneway, kruskal

    # ANOVA: Does score significantly affect absolute error?
    score_groups = [group['absolute_error'].values for name, group in results_df.groupby('score')]
    f_stat, p_value = f_oneway(*score_groups)

    print("\n=== Statistical Analysis ===")
    print(f"One-way ANOVA (Score effect on Absolute Error):")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    # Kruskal-Wallis (non-parametric)
    h_stat, p_value_kw = kruskal(*score_groups)
    print(f"\nKruskal-Wallis Test:")
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_value_kw:.6f}")

    return {'anova_f': f_stat, 'anova_p': p_value, 'kruskal_h': h_stat, 'kruskal_p': p_value_kw}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("Score-Level Accuracy Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading WVS data...")
    df_real = load_wvs_data()
    print(f"   Loaded {len(df_real)} records from WVS")

    # Run analysis
    print("\n2. Running score-level analysis...")
    results_df = run_full_analysis(df_real)
    print(f"   Generated {len(results_df)} score-level comparisons")

    # Save raw results
    results_df.to_csv(TABLES_OUTPUT_PATH / 'score_level_results_full.csv', index=False)
    print(f"   Raw results saved to {TABLES_OUTPUT_PATH / 'score_level_results_full.csv'}")

    # Compute summaries
    print("\n3. Computing summary statistics...")
    summary_overall = compute_summary_by_score(results_df)
    summary_by_model = compute_summary_by_score_and_model(results_df)
    summary_by_quest = compute_summary_by_score_and_questionnaire(results_df)
    summary_by_country = compute_summary_by_score_and_country(results_df)

    # Save summaries
    summary_overall.to_csv(TABLES_OUTPUT_PATH / 'summary_by_score.csv', index=False)
    summary_by_model.to_csv(TABLES_OUTPUT_PATH / 'summary_by_score_model.csv', index=False)
    summary_by_quest.to_csv(TABLES_OUTPUT_PATH / 'summary_by_score_questionnaire.csv', index=False)
    summary_by_country.to_csv(TABLES_OUTPUT_PATH / 'summary_by_score_country.csv', index=False)

    # Print key findings
    print("\n4. Key Findings:")
    print("\n   Scores ranked by prediction accuracy (best to worst):")
    sorted_summary = summary_overall.sort_values('absolute_error_mean')
    for _, row in sorted_summary.iterrows():
        print(f"   Score {int(row['score'])}: Mean Abs. Error = {row['absolute_error_mean']:.4f}")

    # Generate plots
    print("\n5. Generating visualizations...")
    plot_overall_score_accuracy(results_df)
    plot_score_accuracy_by_model(results_df)
    plot_score_accuracy_by_questionnaire(results_df)
    plot_score_accuracy_by_country(results_df)
    plot_score_level_accuracy_heatmap(results_df)
    plot_real_vs_synthetic_comparison(results_df)
    plot_direction_analysis(results_df)
    plot_combined_faceted(results_df)
    print(f"   Plots saved to {PLOTS_OUTPUT_PATH}")

    # Statistical tests
    print("\n6. Statistical analysis...")
    stats = compute_statistical_tests(results_df)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return results_df, summary_overall

if __name__ == "__main__":
    results_df, summary = main()
