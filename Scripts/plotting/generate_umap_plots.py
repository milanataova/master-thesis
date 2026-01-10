#!/usr/bin/env python3
"""
UMAP Visualization of Life Satisfaction Distributions

This script generates 2D UMAP plots for each LLM model, where each data point
represents a (country, questionnaire) combination encoded as a 10-dimensional
relative frequency vector.

The 10D vector represents the relative frequencies for life satisfaction scores 1-10.

Output: Outputs/plots/umap/umap_{model}.png (3 files)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP

# ------------------ CONFIG ------------------

TARGET_COUNTRIES = ["USA", "IDN", "NLD", "MEX"]

# Paths - using project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Thesis/

DATA_ROOT = os.path.join(BASE_DIR, "Datasets")
WVS_FILE = "WVS_Cross-National_Wave_7_csv_v6_0.csv"
SYNTHETIC_DATA_DIR = os.path.join(DATA_ROOT, "synthetic_data", "weighted")
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs", "plots", "umap")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Colors from the scatterplots section
LINE_COLORS = {
    "Real WVS": "black",
    "Original WVS": "#58C747D9",
    "Reverse": "#ff7e0ecd",
    "Cantril": "#66aada",
    "SWLS": "#9c27d696",
    "OHQ": "#d62728cc",  # Red color for OHQ
}

# Different markers for each country
COUNTRY_MARKERS = {
    "USA": "o",      # circle
    "IDN": "s",      # square
    "NLD": "^",      # triangle up
    "MEX": "D",      # diamond
}

# ------------------ HELPERS ------------------

def load_wvs_data(file_path):
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

    df = pd.read_csv(file_path, low_memory=False)
    df = df[list(columns.keys())].rename(columns=columns)

    missing_values = [-1, -2, -4, -5]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(missing_values, np.nan)

    df = df[df["life_satisfaction"].between(0, 10) | df["life_satisfaction"].isna()]
    return df


def categorize_dataframe(df):
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
    essential_cols = [
        "life_satisfaction", "country", "income_level", "health_category", "weight"
    ]
    df_subset = df_categorized[
        df_categorized["country"].isin(target_countries)
    ][essential_cols].copy()
    df_subset.dropna(inplace=True)
    return df_subset


def load_synthetic_dataset(questionnaire: str, model: str) -> pd.DataFrame:
    fname = f"synthetic_{questionnaire}_{model}_w.csv"
    path = os.path.join(SYNTHETIC_DATA_DIR, fname)
    df = pd.read_csv(path)

    if questionnaire == "swls":
        col = "swls_equated"
    elif questionnaire == "ohq":
        col = "ohq_equated"
    else:
        col = "score"

    df = df.copy()
    df["life_satisfaction"] = df[col]
    df = df[df["life_satisfaction"].between(1, 10)]
    return df


def relative_frequency(series: pd.Series, weights: pd.Series = None) -> np.ndarray:
    """
    Returns weighted relative frequencies for scores 1..10 as a numpy array.
    If weights is None, uses equal weights (uniform weighting).
    """
    s = series.dropna()
    valid_idx = (s >= 1) & (s <= 10)
    s = s[valid_idx]

    if weights is not None:
        w = weights[valid_idx]
    else:
        w = pd.Series(1.0, index=s.index)

    # Calculate weighted counts for each score
    weighted_counts = {}
    for score in range(1, 11):
        mask = s == score
        weighted_counts[score] = w[mask].sum()

    # Convert to array
    counts = pd.Series(weighted_counts).reindex(range(1, 11), fill_value=0).sort_index()
    total = counts.sum()
    if total == 0:
        return np.zeros(10)
    return (counts / total).values


# ------------------ LOAD REAL DATA ------------------

print("Loading WVS data...")
df_base = load_wvs_data(os.path.join(DATA_ROOT, WVS_FILE))
df_categorized_base = categorize_dataframe(df_base)
df_for_comparison = prepare_comparison_subset(df_categorized_base, TARGET_COUNTRIES)

real_dist = {}
for country in TARGET_COUNTRIES:
    country_data = df_for_comparison[df_for_comparison["country"] == country]
    vals = country_data["life_satisfaction"]
    weights = country_data["weight"]
    real_dist[country] = relative_frequency(vals, weights)

# ------------------ LOAD SYNTHETIC DATA ------------------

print("Loading synthetic data...")
synthetic_data = {
    (q, m): load_synthetic_dataset(q, m)
    for q in QUESTIONNAIRES
    for m in MODELS
}

# ------------------ BUILD FEATURE MATRIX FOR EACH MODEL ------------------

print("Building feature matrices and generating UMAP plots...")

for model in MODELS:
    print(f"\nProcessing model: {model}")

    # Collect all data points for this model
    vectors = []
    labels = []
    countries = []
    questionnaires = []

    # Add real WVS data for each country (same across all models)
    for country in TARGET_COUNTRIES:
        vectors.append(real_dist[country])
        labels.append("Real WVS")
        countries.append(country)
        questionnaires.append("Real WVS")

    # Add synthetic data for each questionnaire and country
    for q in QUESTIONNAIRES:
        df_syn = synthetic_data[(q, model)]
        q_label = QUESTIONNAIRE_LABELS[q]

        for country in TARGET_COUNTRIES:
            country_data = df_syn[df_syn["country"] == country]
            if country_data.empty:
                continue

            vals = country_data["life_satisfaction"]
            weights = country_data["weight_joint"]
            freq_vector = relative_frequency(vals, weights)
            vectors.append(freq_vector)
            labels.append(q_label)
            countries.append(country)
            questionnaires.append(q_label)

    # Convert to numpy array
    X = np.array(vectors)

    print(f"  Data shape: {X.shape}")

    # Apply UMAP
    print(f"  Applying UMAP...")
    umap_model = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean'
    )
    X_embedded = umap_model.fit_transform(X)

    # Create plot with adjusted size
    print(f"  Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each combination of country and questionnaire
    for i, (country, questionnaire) in enumerate(zip(countries, questionnaires)):
        color = LINE_COLORS[questionnaire]
        marker = COUNTRY_MARKERS[country]

        ax.scatter(
            X_embedded[i, 0],
            X_embedded[i, 1],
            c=color,
            marker=marker,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
        )

    # Create custom legend with two parts: one for colors (questionnaires) and one for shapes (countries)
    from matplotlib.lines import Line2D

    # Legend for questionnaires (colors)
    color_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS["Real WVS"],
               markersize=12, label='Real WVS', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS["Original WVS"],
               markersize=12, label='Original WVS', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS["Reverse"],
               markersize=12, label='Reverse', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS["Cantril"],
               markersize=12, label='Cantril', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS["SWLS"],
               markersize=12, label='SWLS', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS["OHQ"],
               markersize=12, label='OHQ', markeredgecolor='black', markeredgewidth=1.5),
    ]

    # Legend for countries (shapes)
    shape_elements = [
        Line2D([0], [0], marker=COUNTRY_MARKERS["USA"], color='w', markerfacecolor='gray',
               markersize=12, label='USA', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker=COUNTRY_MARKERS["IDN"], color='w', markerfacecolor='gray',
               markersize=12, label='Indonesia', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker=COUNTRY_MARKERS["NLD"], color='w', markerfacecolor='gray',
               markersize=12, label='Netherlands', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker=COUNTRY_MARKERS["MEX"], color='w', markerfacecolor='gray',
               markersize=12, label='Mexico', markeredgecolor='black', markeredgewidth=1.5),
    ]

    # Place both legends next to each other outside the plot (bottom center)
    legend1 = ax.legend(handles=color_elements, loc='upper left', title='Questionnaire',
                       frameon=True, fontsize=13, title_fontsize=14,
                       bbox_to_anchor=(0.0, -0.08), ncol=3)
    ax.add_artist(legend1)

    legend2 = ax.legend(handles=shape_elements, loc='upper right', title='Country',
                       frameon=True, fontsize=13, title_fontsize=14,
                       bbox_to_anchor=(1.0, -0.08), ncol=2)

    # Set title
    ax.set_title(f'UMAP Projection of Life Satisfaction Distributions\n{model}',
                fontsize=16, pad=20)

    # Remove x- and y-axes labels and ticks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    # Keep grid for visual reference
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    out_path = os.path.join(OUTPUT_DIR, f"umap_{model}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved to: {out_path}")

print("\n✓ All UMAP plots generated successfully!")
print(f"Plots saved to: {OUTPUT_DIR}")
