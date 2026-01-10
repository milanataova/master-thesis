"""
03_weight_generation.py

Compute survey weights for synthetic data to match real WVS demographic distribution.

This script calculates weights for each synthetic response so that, when weighted,
the synthetic joint distribution over (country, income_level, health_level) matches
the real WVS survey-weighted joint distribution.

Input:
    - WVS Wave 7 data (for real distribution)
    - Raw synthetic CSV files from Datasets/synthetic_data/synthetic_raw/

Output:
    - Weighted synthetic CSV files in Datasets/synthetic_data/weighted/
    - Each file gets a 'weight_joint' column added
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "Datasets"
WVS_FILE = DATA_DIR / "WVS_Cross-National_Wave_7_csv_v6_0.csv"
SYNTHETIC_RAW_DIR = DATA_DIR / "synthetic_data" / "synthetic_raw"
SYNTHETIC_WEIGHTED_DIR = DATA_DIR / "synthetic_data" / "weighted"

TARGET_COUNTRIES = ["USA", "IDN", "NLD", "MEX"]

# Column name mappings
GROUP_COLS_REAL = ["country", "income_level", "health_category"]
GROUP_COLS_SYNTH = ["country", "income_level", "health_level"]

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_wvs_data(file_path):
    """
    Load WVS Wave 7, rename columns, apply missing codes,
    keep W_WEIGHT as 'weight'.
    """
    columns = {
        "Q49": "life_satisfaction",
        "Q288": "income",
        "Q47": "health",
        "B_COUNTRY_ALPHA": "country",
        "W_WEIGHT": "weight"
    }

    print(f"Loading WVS data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    df = df[list(columns.keys())].rename(columns=columns)

    # Replace missing value codes with NaN
    missing_values = [-1, -2, -4, -5]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(missing_values, np.nan)

    # Filter for valid life satisfaction scores
    df = df[df["life_satisfaction"].between(0, 10) | df["life_satisfaction"].isna()]
    return df


def categorize_dataframe(df):
    """Add income_level and health_category to WVS data."""
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

    df_c = df.copy()
    df_c["income_level"] = df_c["income"].apply(categorize_income)
    df_c["health_category"] = df_c["health"].apply(categorize_health)
    return df_c


def prepare_comparison_subset(df_categorized, target_countries):
    """Build real subset for comparison, including survey weight."""
    essential_cols = [
        "life_satisfaction",
        "country",
        "income_level",
        "health_category",
        "weight"
    ]

    df_subset = df_categorized[
        df_categorized["country"].isin(target_countries)
    ][essential_cols].copy()

    df_subset.dropna(inplace=True)

    print(f"Prepared comparison subset: {len(df_subset)} rows for {len(target_countries)} countries")
    return df_subset


# ============================================================================
# Weight Computation
# ============================================================================

def compute_joint_weights(
    df_real_subset,
    df_synth,
    group_cols_real=GROUP_COLS_REAL,
    group_cols_synth=GROUP_COLS_SYNTH,
    weight_col_name="weight_joint",
):
    """
    Compute weights for each synthetic row so that, when weighted,
    the synthetic joint distribution over (country, income_level, health_level)
    matches the real survey weighted joint distribution.

    The weight for each cell = p_real / p_synth, where:
    - p_real is the proportion in the real weighted distribution
    - p_synth is the proportion in the synthetic unweighted distribution
    """
    if "weight" not in df_real_subset.columns:
        raise ValueError("df_real_subset must contain 'weight' column")

    # Step 1: Calculate survey-weighted real joint distribution
    real_weighted = (
        df_real_subset
        .groupby(group_cols_real)["weight"]
        .sum()
        .rename("real_weight_sum")
        .reset_index()
    )
    total_real_weight = real_weighted["real_weight_sum"].sum()
    real_weighted["p_real"] = real_weighted["real_weight_sum"] / total_real_weight

    # Step 2: Calculate synthetic unweighted joint distribution
    synth_counts = (
        df_synth
        .groupby(group_cols_synth)
        .size()
        .rename("n_synth")
        .reset_index()
    )
    total_synth = synth_counts["n_synth"].sum()
    synth_counts["p_synth"] = synth_counts["n_synth"] / total_synth

    # Step 3: Align column names for merging
    rename_map = {s: r for r, s in zip(group_cols_real, group_cols_synth)}
    synth_for_merge = synth_counts.rename(columns=rename_map)

    # Step 4: Merge real and synthetic distributions
    merged = pd.merge(
        synth_for_merge,
        real_weighted,
        on=group_cols_real,
        how="left",
        validate="one_to_one"
    )
    merged["p_real"] = merged["p_real"].fillna(0.0)

    # Step 5: Calculate cell weight = p_real / p_synth
    merged[weight_col_name] = np.where(
        merged["p_synth"] > 0,
        merged["p_real"] / merged["p_synth"],
        0.0
    )

    cell_weights = merged[group_cols_real + [weight_col_name]]

    # Step 6: Attach weights to synthetic rows
    df_synth_tmp = df_synth.rename(columns=rename_map)
    df_synth_weighted = df_synth_tmp.merge(
        cell_weights,
        on=group_cols_real,
        how="left"
    )
    df_synth_weighted[weight_col_name] = df_synth_weighted[weight_col_name].fillna(0.0)

    # Step 7: Restore original synthetic column names
    inv_rename = {v: k for k, v in rename_map.items()}
    df_synth_weighted = df_synth_weighted.rename(columns=inv_rename)

    return df_synth_weighted


def summarize_group_distribution(df, group_cols, label="Dataset"):
    """Print count and percentage distribution for given grouping columns."""
    print(f"\n==== {label} distribution ====")

    counts = (
        df.groupby(group_cols)
          .size()
          .rename("count")
          .reset_index()
    )
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(2)

    print(counts.to_string(index=False))
    print(f"Total N = {counts['count'].sum()}")
    return counts


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Ensure output directory exists
    SYNTHETIC_WEIGHTED_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare real WVS data
    df_base = load_wvs_data(WVS_FILE)
    print(f"Loaded WVS dataset: {len(df_base)} rows")

    df_categorized = categorize_dataframe(df_base)
    print("Categorized income and health")

    df_for_comparison = prepare_comparison_subset(df_categorized, TARGET_COUNTRIES)

    # Show real data distribution
    summarize_group_distribution(
        df_for_comparison,
        GROUP_COLS_REAL,
        label="Real WVS (comparison subset)"
    )

    # List of synthetic files to process
    synthetic_files = list(SYNTHETIC_RAW_DIR.glob("synthetic_*.csv"))

    if not synthetic_files:
        print(f"\nNo synthetic files found in {SYNTHETIC_RAW_DIR}")
    else:
        print(f"\nFound {len(synthetic_files)} synthetic files to process")

    for synth_path in sorted(synthetic_files):
        print(f"\n{'='*60}")
        print(f"Processing: {synth_path.name}")
        print("="*60)

        df_synth = pd.read_csv(synth_path)

        # Check required columns
        required_cols = ["country", "income_level", "health_level"]
        missing = [c for c in required_cols if c not in df_synth.columns]
        if missing:
            print(f"  [SKIP] Missing columns: {missing}")
            continue

        # Show original distribution
        summarize_group_distribution(
            df_synth,
            GROUP_COLS_SYNTH,
            label=f"Synthetic ({synth_path.name})"
        )

        # Compute weights
        df_synth_weighted = compute_joint_weights(
            df_real_subset=df_for_comparison,
            df_synth=df_synth,
            group_cols_real=GROUP_COLS_REAL,
            group_cols_synth=GROUP_COLS_SYNTH,
            weight_col_name="weight_joint",
        )

        # Preview cell weights
        print("\nCell weights (first 12):")
        cell_weights_preview = (
            df_synth_weighted
            .groupby(["country", "income_level", "health_level"])["weight_joint"]
            .first()
            .reset_index()
            .head(12)
        )
        print(cell_weights_preview.to_string(index=False))

        # Save weighted file
        out_filename = synth_path.name.replace(".csv", "_w.csv")
        out_path = SYNTHETIC_WEIGHTED_DIR / out_filename
        df_synth_weighted.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

    print("\n" + "="*60)
    print("Weight generation complete!")
    print("="*60)
