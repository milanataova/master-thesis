#!/usr/bin/env python3
"""
Subgroup Analysis for Life Satisfaction Synthetic Data

This script performs comprehensive subgroup analysis comparing real WVS data
with synthetic LLM-generated data across multiple aggregation levels:
- Overall (all data)
- By Country
- By Income Level
- By Health Level
- By Country × Income
- By Country × Health
- Full segments (Country × Income × Health)

Outputs (12 CSV files):
- all_segment_metrics.csv - Complete metrics at all aggregation levels
- segment_insights.csv - Best model/questionnaire per full segment
- model_performance.csv - Summary statistics by model
- questionnaire_performance.csv - Summary statistics by questionnaire
- best_performers_summary.csv - Best performer at each aggregation level
- rankings_by_segment.csv - Rankings within each segment
- win_counts.csv - Win counts by model × questionnaire combination
- table1_variance_decomposition.csv - Variance explained by each factor
- table2_summary_statistics.csv - Summary statistics by factor level
- table3_statistical_tests.csv - ANOVA test results
- table4_best_configurations.csv - Best configuration per segment (formatted)
- table4b_country_summary.csv - Country-level summary
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, f_oneway
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # Thesis/

# Input paths
WVS_DATA_PATH = BASE_DIR / "Datasets" / "WVS_Cross-National_Wave_7_csv_v6_0.csv"
SYNTHETIC_DATA_DIR = BASE_DIR / "Datasets" / "synthetic_data" / "weighted"

# Output path
OUTPUT_DIR = BASE_DIR / "Outputs" / "metrics_tables" / "subgroup_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
TARGET_COUNTRIES = ["USA", "IDN", "NLD", "MEX"]
MODELS = ["llama3.1_8b", "llama3.3_70b", "qwen2.5_72b"]
QUESTIONNAIRES = ["base", "cantril", "reverse", "swls", "ohq"]
INCOME_LEVELS = ["Low", "Medium", "High"]
HEALTH_LEVELS = ["Good", "Fair", "Poor"]


# ============================================================================
# Data Loading
# ============================================================================

def load_wvs_data():
    """Load and preprocess WVS Wave 7 data."""
    print("Loading WVS data...")

    columns = {
        "Q49": "life_satisfaction",
        "Q288": "income",
        "Q47": "health",
        "B_COUNTRY_ALPHA": "country",
        "W_WEIGHT": "weight"
    }

    df = pd.read_csv(WVS_DATA_PATH, low_memory=False)
    df = df[list(columns.keys())].rename(columns=columns)

    # Replace missing values
    for col in ['life_satisfaction', 'income', 'health']:
        df[col] = df[col].replace([-1, -2, -4, -5], np.nan)

    # Filter valid life satisfaction (1-10 scale)
    df = df[df['life_satisfaction'].between(1, 10)]
    df = df[df['country'].isin(TARGET_COUNTRIES)]

    # Categorize income
    def categorize_income(val):
        if val in [1, 2, 3]:
            return "Low"
        elif val in [4, 5, 6, 7]:
            return "Medium"
        elif val in [8, 9, 10]:
            return "High"
        return None

    # Categorize health
    def categorize_health(val):
        if val in [1, 2]:
            return "Good"
        elif val == 3:
            return "Fair"
        elif val in [4, 5]:
            return "Poor"
        return None

    df['income_level'] = df['income'].apply(categorize_income)
    df['health_level'] = df['health'].apply(categorize_health)

    # Drop rows with missing categories
    df = df.dropna(subset=['income_level', 'health_level', 'weight'])

    print(f"  Loaded {len(df)} valid WVS records")
    return df


def load_synthetic_data(questionnaire, model):
    """Load synthetic data for a specific questionnaire-model combination."""
    file_path = SYNTHETIC_DATA_DIR / f"synthetic_{questionnaire}_{model}_w.csv"

    if not file_path.exists():
        print(f"  Warning: File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    # Get the score column based on questionnaire type
    if questionnaire == "swls":
        score_col = "swls_equated"
    elif questionnaire == "ohq":
        score_col = "ohq_equated"
    else:
        score_col = "score"

    if score_col not in df.columns:
        print(f"  Warning: Column {score_col} not found in {file_path}")
        return None

    df['life_satisfaction'] = df[score_col]
    df = df[df['life_satisfaction'].between(1, 10)]

    return df


# ============================================================================
# Metric Calculation
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


def calculate_metrics(real_data, synth_data, real_weights=None, synth_weights=None):
    """Calculate weighted KS statistic and Wasserstein distance between distributions."""

    if len(real_data) == 0 or len(synth_data) == 0:
        return None

    real_data = np.asarray(real_data)
    synth_data = np.asarray(synth_data)

    # Use weights if provided, otherwise equal weights
    if real_weights is None:
        real_weights = np.ones(len(real_data))
    else:
        real_weights = np.asarray(real_weights)

    if synth_weights is None:
        synth_weights = np.ones(len(synth_data))
    else:
        synth_weights = np.asarray(synth_weights)

    # Calculate weighted KS statistic
    ks_stat = weighted_ks_statistic(real_data, real_weights, synth_data, synth_weights)

    # Calculate weighted Wasserstein distance
    real_w = real_weights / real_weights.sum()
    synth_w = synth_weights / synth_weights.sum()
    wass_dist = wasserstein_distance(real_data, synth_data, real_w, synth_w)

    # Calculate summary statistics
    real_mean = np.average(real_data, weights=real_weights) if real_weights is not None else np.mean(real_data)
    synth_mean = np.average(synth_data, weights=synth_weights) if synth_weights is not None else np.mean(synth_data)

    # Weighted std for real data
    if real_weights is not None:
        real_var = np.average((real_data - real_mean)**2, weights=real_weights)
        real_std = np.sqrt(real_var)
    else:
        real_std = np.std(real_data)

    # Weighted std for synthetic data
    if synth_weights is not None:
        synth_var = np.average((synth_data - synth_mean)**2, weights=synth_weights)
        synth_std = np.sqrt(synth_var)
    else:
        synth_std = np.std(synth_data)

    return {
        'ks_stat': ks_stat,
        'wasserstein': wass_dist,
        'n_real': len(real_data),
        'n_synth': len(synth_data),
        'real_mean': real_mean,
        'synth_mean': synth_mean,
        'real_std': real_std,
        'synth_std': synth_std,
        'mean_diff': abs(real_mean - synth_mean),
        'std_ratio': synth_std / real_std if real_std > 0 else np.nan
    }


def filter_data(df, country=None, income_level=None, health_level=None):
    """Filter dataframe by specified criteria."""
    filtered = df.copy()

    if country is not None and country != "ALL":
        filtered = filtered[filtered['country'] == country]
    if income_level is not None and income_level != "ALL":
        filtered = filtered[filtered['income_level'] == income_level]
    if health_level is not None and health_level != "ALL":
        filtered = filtered[filtered['health_level'] == health_level]

    return filtered


# ============================================================================
# Main Analysis
# ============================================================================

def run_subgroup_analysis(df_real):
    """Run comprehensive subgroup analysis across all levels."""

    print("\nRunning subgroup analysis...")
    all_results = []

    # Define aggregation levels
    levels = [
        ("Overall", [("ALL", "ALL", "ALL")]),
        ("Country", [(c, "ALL", "ALL") for c in TARGET_COUNTRIES]),
        ("Income", [("ALL", i, "ALL") for i in INCOME_LEVELS]),
        ("Health", [("ALL", "ALL", h) for h in HEALTH_LEVELS]),
        ("Country_Income", [(c, i, "ALL") for c in TARGET_COUNTRIES for i in INCOME_LEVELS]),
        ("Country_Health", [(c, "ALL", h) for c in TARGET_COUNTRIES for h in HEALTH_LEVELS]),
        ("Full", [(c, i, h) for c in TARGET_COUNTRIES for i in INCOME_LEVELS for h in HEALTH_LEVELS])
    ]

    # Load all synthetic data
    print("Loading synthetic data...")
    synthetic_datasets = {}
    for q in QUESTIONNAIRES:
        for m in MODELS:
            df_synth = load_synthetic_data(q, m)
            if df_synth is not None:
                synthetic_datasets[(q, m)] = df_synth

    print(f"  Loaded {len(synthetic_datasets)} synthetic datasets")

    # Calculate metrics for each level
    total_comparisons = 0
    for level_name, segments in levels:
        print(f"\n  Processing level: {level_name} ({len(segments)} segments)")

        for country, income, health in segments:
            # Filter real data
            real_filtered = filter_data(df_real,
                                        country if country != "ALL" else None,
                                        income if income != "ALL" else None,
                                        health if health != "ALL" else None)

            if len(real_filtered) == 0:
                continue

            real_values = real_filtered['life_satisfaction'].values
            real_weights = real_filtered['weight'].values

            # Compare with each synthetic dataset
            for (questionnaire, model), df_synth in synthetic_datasets.items():
                # Filter synthetic data
                synth_filtered = filter_data(df_synth,
                                            country if country != "ALL" else None,
                                            income if income != "ALL" else None,
                                            health if health != "ALL" else None)

                if len(synth_filtered) == 0:
                    continue

                synth_values = synth_filtered['life_satisfaction'].values
                synth_weights = synth_filtered['weight_joint'].values if 'weight_joint' in synth_filtered.columns else None

                # Calculate metrics
                metrics = calculate_metrics(real_values, synth_values, real_weights, synth_weights)

                if metrics is None:
                    continue

                # Create segment ID
                segment_id = f"{level_name}_{country}_{income}_{health}"

                result = {
                    'questionnaire': questionnaire,
                    'model': model,
                    'level': level_name,
                    'country': country,
                    'income_level': income,
                    'health_level': health,
                    'segment_id': segment_id,
                    **metrics
                }

                all_results.append(result)
                total_comparisons += 1

    print(f"\n  Total comparisons: {total_comparisons}")
    return pd.DataFrame(all_results)


def add_rankings_and_scores(df):
    """Add normalized scores, composite scores, and rankings within each segment."""

    # Group by segment and calculate normalized metrics
    df = df.copy()

    # Normalize KS and Wasserstein within each segment (0-1 scale, lower is better)
    for segment_id in df['segment_id'].unique():
        mask = df['segment_id'] == segment_id
        segment_data = df[mask]

        # Normalize KS
        ks_min, ks_max = segment_data['ks_stat'].min(), segment_data['ks_stat'].max()
        if ks_max > ks_min:
            df.loc[mask, 'ks_norm'] = (segment_data['ks_stat'] - ks_min) / (ks_max - ks_min)
        else:
            df.loc[mask, 'ks_norm'] = 0

        # Normalize Wasserstein
        wass_min, wass_max = segment_data['wasserstein'].min(), segment_data['wasserstein'].max()
        if wass_max > wass_min:
            df.loc[mask, 'wass_norm'] = (segment_data['wasserstein'] - wass_min) / (wass_max - wass_min)
        else:
            df.loc[mask, 'wass_norm'] = 0

    # Composite score (average of normalized metrics, lower is better)
    df['composite_score'] = (df['ks_norm'] + df['wass_norm']) / 2

    # Rank within each segment (1 = best)
    df['rank'] = df.groupby('segment_id')['composite_score'].rank(method='min')
    df['rank_wasserstein'] = df.groupby('segment_id')['wasserstein'].rank(method='min')

    return df


def generate_summary_tables(df):
    """Generate summary statistics tables."""

    # Full segment level for win rates and segment-specific analysis
    df_full = df[df['level'] == 'Full'].copy()

    # 1. Model performance summary (use ALL levels for mean/std/min/max)
    print("\nGenerating model_performance.csv...")
    model_perf = df.groupby('model').agg({
        'wasserstein': ['mean', 'std', 'min', 'max'],
        'ks_stat': ['mean', 'std', 'min', 'max'],
        'composite_score': ['mean', 'std'],
        'rank': 'mean'
    }).round(4)
    model_perf.columns = ['_'.join(col).strip() for col in model_perf.columns]
    model_perf = model_perf.reset_index()

    # Add win rate (from Full level only - rank == 1)
    wins_by_model = df_full[df_full['rank'] == 1].groupby('model').size()
    total_segments = df_full['segment_id'].nunique()
    model_perf['win_rate'] = model_perf['model'].map(
        lambda m: (wins_by_model.get(m, 0) / total_segments) * 100
    )

    # 2. Questionnaire performance summary (use ALL levels for mean/std/min/max)
    print("Generating questionnaire_performance.csv...")
    quest_perf = df.groupby('questionnaire').agg({
        'wasserstein': ['mean', 'std', 'min', 'max'],
        'ks_stat': ['mean', 'std', 'min', 'max'],
        'composite_score': ['mean', 'std'],
        'rank': 'mean'
    }).round(4)
    quest_perf.columns = ['_'.join(col).strip() for col in quest_perf.columns]
    quest_perf = quest_perf.reset_index()

    # Add win rate (from Full level only)
    wins_by_quest = df_full[df_full['rank'] == 1].groupby('questionnaire').size()
    quest_perf['win_rate'] = quest_perf['questionnaire'].map(
        lambda q: (wins_by_quest.get(q, 0) / total_segments) * 100
    )

    # 3. Segment insights (best performer per full segment)
    print("Generating segment_insights.csv...")
    best_per_segment = df_full.loc[df_full.groupby('segment_id')['wasserstein'].idxmin()]

    insights = []
    for _, row in best_per_segment.iterrows():
        insight_text = (
            f"For {row['country']} citizens with {row['income_level']} income and "
            f"{row['health_level']} health, the real data is best approximated by "
            f"{row['model']} using the {row['questionnaire'].upper()} questionnaire "
            f"(Wasserstein: {row['wasserstein']:.3f}, KS: {row['ks_stat']:.3f})"
        )
        insights.append({
            'country': row['country'],
            'income_level': row['income_level'],
            'health_level': row['health_level'],
            'best_model': row['model'],
            'best_questionnaire': row['questionnaire'],
            'wasserstein': row['wasserstein'],
            'ks_stat': row['ks_stat'],
            'insight': insight_text
        })

    segment_insights = pd.DataFrame(insights)

    # 4. Best performers summary (best at each aggregation level)
    print("Generating best_performers_summary.csv...")
    best_performers = df.loc[df.groupby('segment_id')['wasserstein'].idxmin()]
    best_performers = best_performers[[
        'level', 'country', 'income_level', 'health_level',
        'questionnaire', 'model', 'composite_score', 'wasserstein', 'ks_stat',
        'n_real', 'n_synth', 'mean_diff'
    ]]

    # 5. Win counts
    print("Generating win_counts.csv...")
    df_full['model_quest'] = df_full['model'] + '_' + df_full['questionnaire']
    wins = df_full[df_full['rank'] == 1].groupby(['questionnaire', 'model']).size().reset_index(name='wins')
    wins['win_rate'] = (wins['wins'] / total_segments) * 100
    wins = wins.sort_values('wins', ascending=False)

    # 6. Rankings by segment (sorted for specification curve)
    print("Generating rankings_by_segment.csv...")
    rankings = df.sort_values(['segment_id', 'rank'])

    # =========================================================================
    # Additional tables (table1-4b)
    # =========================================================================

    # 7. Variance decomposition (table1)
    print("Generating table1_variance_decomposition.csv...")
    factors = ['health_level', 'income_level', 'country', 'model', 'questionnaire']
    factor_names = ['Health Status', 'Income Level', 'Country', 'Model', 'Questionnaire Type']

    variance_results = []
    grand_mean = df_full['wasserstein'].mean()
    SS_total = ((df_full['wasserstein'] - grand_mean) ** 2).sum()

    for factor, name in zip(factors, factor_names):
        group_stats = df_full.groupby(factor)['wasserstein'].agg(['mean', 'count'])
        SS_between = ((group_stats['mean'] - grand_mean) ** 2 * group_stats['count']).sum()
        var_explained = (SS_between / SS_total) * 100

        groups = [df_full[df_full[factor] == level]['wasserstein'].values
                 for level in df_full[factor].unique()]
        f_stat, p_value = f_oneway(*groups)

        variance_results.append({
            'factor': name,
            'variance_explained': round(var_explained, 4),
            'f_stat': round(f_stat, 4),
            'p_value': round(p_value, 6),
            'significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })

    table1_variance = pd.DataFrame(variance_results)
    table1_variance = table1_variance.sort_values('variance_explained', ascending=False)

    # 8. Summary statistics by factor (table2)
    print("Generating table2_summary_statistics.csv...")
    summary_data = []

    # By Health Status
    for health in ['Good', 'Fair', 'Poor']:
        data = df_full[df_full['health_level'] == health]['wasserstein']
        summary_data.append({
            'Factor': 'Health Status', 'Level': health,
            'Mean W': f"{data.mean():.3f}", 'SD': f"{data.std():.3f}",
            'Median': f"{data.median():.3f}", 'Min': f"{data.min():.3f}",
            'Max': f"{data.max():.3f}", 'N': len(data)
        })

    # By Income Level
    for income in ['Low', 'Medium', 'High']:
        data = df_full[df_full['income_level'] == income]['wasserstein']
        summary_data.append({
            'Factor': 'Income Level', 'Level': income,
            'Mean W': f"{data.mean():.3f}", 'SD': f"{data.std():.3f}",
            'Median': f"{data.median():.3f}", 'Min': f"{data.min():.3f}",
            'Max': f"{data.max():.3f}", 'N': len(data)
        })

    # By Country
    for country in ['USA', 'IDN', 'NLD', 'MEX']:
        data = df_full[df_full['country'] == country]['wasserstein']
        summary_data.append({
            'Factor': 'Country', 'Level': country,
            'Mean W': f"{data.mean():.3f}", 'SD': f"{data.std():.3f}",
            'Median': f"{data.median():.3f}", 'Min': f"{data.min():.3f}",
            'Max': f"{data.max():.3f}", 'N': len(data)
        })

    # By Model
    model_display = {'llama3.1_8b': 'LLaMA 3.1 8B', 'llama3.3_70b': 'LLaMA 3.3 70B', 'qwen2.5_72b': 'Qwen 2.5 72B'}
    for model in ['llama3.1_8b', 'llama3.3_70b', 'qwen2.5_72b']:
        data = df_full[df_full['model'] == model]['wasserstein']
        summary_data.append({
            'Factor': 'Model', 'Level': model_display[model],
            'Mean W': f"{data.mean():.3f}", 'SD': f"{data.std():.3f}",
            'Median': f"{data.median():.3f}", 'Min': f"{data.min():.3f}",
            'Max': f"{data.max():.3f}", 'N': len(data)
        })

    # By Questionnaire
    for quest in ['base', 'cantril', 'ohq', 'reverse', 'swls']:
        data = df_full[df_full['questionnaire'] == quest]['wasserstein']
        summary_data.append({
            'Factor': 'Questionnaire', 'Level': quest.upper(),
            'Mean W': f"{data.mean():.3f}", 'SD': f"{data.std():.3f}",
            'Median': f"{data.median():.3f}", 'Min': f"{data.min():.3f}",
            'Max': f"{data.max():.3f}", 'N': len(data)
        })

    table2_summary = pd.DataFrame(summary_data)

    # 9. Statistical tests (table3)
    print("Generating table3_statistical_tests.csv...")
    test_results = []
    test_factors = [
        ('health_level', 'Health Status', ['Good', 'Fair', 'Poor']),
        ('income_level', 'Income Level', ['Low', 'Medium', 'High']),
        ('country', 'Country', ['USA', 'IDN', 'NLD', 'MEX']),
        ('model', 'Model', ['llama3.1_8b', 'llama3.3_70b', 'qwen2.5_72b']),
        ('questionnaire', 'Questionnaire', ['base', 'cantril', 'ohq', 'reverse', 'swls'])
    ]

    for factor_col, factor_name, levels in test_factors:
        groups = [df_full[df_full[factor_col] == level]['wasserstein'].dropna().values
                 for level in levels]
        f_stat, p_value = f_oneway(*groups)
        test_results.append({
            'Factor': factor_name,
            'F-statistic': f"{f_stat:.3f}",
            'p-value': f"{p_value:.4f}" if p_value >= 0.001 else "<0.001",
            'Significant': 'Yes***' if p_value < 0.001 else 'Yes**' if p_value < 0.01 else 'Yes*' if p_value < 0.05 else 'No'
        })

    table3_tests = pd.DataFrame(test_results)

    # 10. Best configurations formatted (table4)
    print("Generating table4_best_configurations.csv...")
    table4_configs = segment_insights[[
        'country', 'income_level', 'health_level',
        'best_model', 'best_questionnaire', 'wasserstein', 'ks_stat'
    ]].copy()

    table4_configs['best_model'] = table4_configs['best_model'].replace(model_display)
    table4_configs['best_questionnaire'] = table4_configs['best_questionnaire'].str.upper()
    table4_configs['wasserstein'] = table4_configs['wasserstein'].round(3)
    table4_configs['ks_stat'] = table4_configs['ks_stat'].round(3)
    table4_configs.columns = ['Country', 'Income', 'Health', 'Best Model',
                              'Best Questionnaire', 'Wasserstein', 'KS Statistic']
    table4_configs = table4_configs.sort_values(['Country', 'Income', 'Health'])

    # 11. Country summary (table4b)
    print("Generating table4b_country_summary.csv...")
    table4b_country = table4_configs.groupby('Country').agg({
        'Wasserstein': ['mean', 'std', 'min', 'max'],
        'KS Statistic': ['mean', 'std']
    }).round(3)

    return {
        'model_performance': model_perf,
        'questionnaire_performance': quest_perf,
        'segment_insights': segment_insights,
        'best_performers_summary': best_performers,
        'win_counts': wins,
        'rankings_by_segment': rankings,
        'table1_variance_decomposition': table1_variance,
        'table2_summary_statistics': table2_summary,
        'table3_statistical_tests': table3_tests,
        'table4_best_configurations': table4_configs,
        'table4b_country_summary': table4b_country
    }


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 60)
    print("Subgroup Analysis for Life Satisfaction Synthetic Data")
    print("=" * 60)

    # Load real data
    df_real = load_wvs_data()

    # Run analysis
    df_metrics = run_subgroup_analysis(df_real)

    # Add rankings and composite scores
    print("\nCalculating rankings and composite scores...")
    df_metrics = add_rankings_and_scores(df_metrics)

    # Reorder columns
    column_order = [
        'questionnaire', 'model', 'level', 'country', 'income_level', 'health_level',
        'ks_stat', 'wasserstein', 'n_real', 'n_synth', 'real_mean', 'synth_mean',
        'real_std', 'synth_std', 'mean_diff', 'std_ratio', 'segment_id',
        'ks_norm', 'wass_norm', 'composite_score', 'rank', 'rank_wasserstein'
    ]
    df_metrics = df_metrics[column_order]

    # Save main metrics file
    print(f"\nSaving all_segment_metrics.csv...")
    df_metrics.to_csv(OUTPUT_DIR / 'all_segment_metrics.csv', index=False)
    print(f"  Saved {len(df_metrics)} rows")

    # Generate and save summary tables
    summaries = generate_summary_tables(df_metrics)

    for name, df_summary in summaries.items():
        output_path = OUTPUT_DIR / f'{name}.csv'
        df_summary.to_csv(output_path, index=False)
        print(f"  Saved {name}.csv ({len(df_summary)} rows)")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    df_full = df_metrics[df_metrics['level'] == 'Full']
    print(f"\nTotal segment-level comparisons: {len(df_full)}")
    print(f"Wasserstein range: {df_full['wasserstein'].min():.3f} to {df_full['wasserstein'].max():.3f}")
    print(f"Mean Wasserstein: {df_full['wasserstein'].mean():.3f}")

    print("\nBest performing models (by win rate):")
    for _, row in summaries['model_performance'].sort_values('win_rate', ascending=False).iterrows():
        print(f"  {row['model']}: {row['win_rate']:.1f}% wins, mean W={row['wasserstein_mean']:.3f}")

    print("\nBest performing questionnaires (by win rate):")
    for _, row in summaries['questionnaire_performance'].sort_values('win_rate', ascending=False).iterrows():
        print(f"  {row['questionnaire']}: {row['win_rate']:.1f}% wins, mean W={row['wasserstein_mean']:.3f}")

    print(f"\nOutput saved to: {OUTPUT_DIR}")

    return df_metrics


if __name__ == "__main__":
    main()
