"""
04_equipercentile_equating.py

Equipercentile equating for SWLS and OHQ scales to WVS 1-10 scale.

This script transforms scores from different life satisfaction instruments
to a common 1-10 scale using equipercentile equating. This allows fair
comparison between different questionnaire variants.

Scales transformed:
    - SWLS: 1-7 scale (from swls_total 5-35 mapped to categories) -> 1-10
    - OHQ: 1-6 scale (ohq_score average) -> 1-10

Method:
    Equipercentile equating maps scores by matching percentile ranks.
    For each source score, find its percentile in the source distribution,
    then find the target score with the same percentile.

Input:
    - WVS Wave 7 data (for target distribution)
    - Weighted synthetic SWLS/OHQ files from Datasets/synthetic_data/weighted/

Output:
    - Updated synthetic files with 'life_satisfaction_equated' column
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "Datasets"
WVS_FILE = DATA_DIR / "WVS_Cross-National_Wave_7_csv_v6_0.csv"
SYNTHETIC_WEIGHTED_DIR = DATA_DIR / "synthetic_data" / "weighted"

TARGET_COUNTRIES = ["USA", "IDN", "NLD", "MEX"]

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_wvs_data(file_path):
    """Load WVS Wave 7, rename columns, apply missing codes."""
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
    df = df[df["life_satisfaction"].between(1, 10)]
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
    return df_subset


# ============================================================================
# Target Distribution Building
# ============================================================================

def sample_wvs_for_equating(df_for_comparison, frac=0.1, random_state=42):
    """
    Take a random subset of the WVS comparison data to be used as target
    for equipercentile equating. Using a subset prevents overfitting.
    """
    df_equate = df_for_comparison.sample(frac=frac, random_state=random_state)
    print(f"Selected {len(df_equate)} rows for equating (frac={frac})")
    return df_equate


def build_weighted_target_distribution(df_equate, score_col="life_satisfaction", weight_col="weight"):
    """
    Build a weighted target distribution for WVS life satisfaction.
    Returns sorted unique scores and their weighted CDF.
    """
    # Aggregate weights per score
    agg = (
        df_equate
        .groupby(score_col)[weight_col]
        .sum()
        .rename("weight_sum")
        .reset_index()
    )

    # Keep only valid scores (1-10)
    agg = agg[agg[score_col].between(1, 10)]
    agg = agg.sort_values(score_col)

    total_weight = agg["weight_sum"].sum()
    agg["p"] = agg["weight_sum"] / total_weight
    agg["cdf"] = agg["p"].cumsum()

    target_values = agg[score_col].to_numpy()
    target_cdf = agg["cdf"].to_numpy()

    return target_values, target_cdf


# ============================================================================
# Equipercentile Mapping
# ============================================================================

def build_equip_mapping_discrete(
    source_min,
    source_max,
    target_values,
    target_cdf,
    target_range=(1, 10)
):
    """
    Build an equipercentile mapping for a discrete source scale
    with integer values from source_min to source_max (inclusive),
    assuming a uniform distribution over the source categories.

    Returns a dictionary mapping source_value -> target_value_on_WVS_scale.
    """
    source_values = np.arange(source_min, source_max + 1)
    k = len(source_values)

    # Uniform probabilities and CDF for source
    p_source = np.full(k, 1.0 / k)
    source_cdf = p_source.cumsum()

    # Interpolation from percentile to target score
    t_min, t_max = target_range
    percentile_to_target = interpolate.interp1d(
        target_cdf,
        target_values,
        kind="linear",
        bounds_error=False,
        fill_value=(t_min, t_max)
    )

    mapping = {}
    for val, cdf_val in zip(source_values, source_cdf):
        t_val = percentile_to_target(cdf_val)
        t_val_rounded = int(np.clip(np.round(t_val), t_min, t_max))
        mapping[val] = t_val_rounded

    return mapping


# ============================================================================
# SWLS Score Conversion
# ============================================================================

def convert_swls_total_to_7scale(swls_total):
    """
    Convert SWLS total scores (5-35) to 1-7 satisfaction scale.

    Uses standard SWLS interpretation categories:
        - 31-35: Highly Satisfied (7)
        - 26-30: Satisfied (6)
        - 21-25: Slightly Satisfied (5)
        - 20: Neutral (4)
        - 15-19: Slightly Dissatisfied (3)
        - 10-14: Dissatisfied (2)
        - 5-9: Highly Dissatisfied (1)
    """
    conditions = [
        swls_total >= 31,
        (swls_total >= 26) & (swls_total <= 30),
        (swls_total >= 21) & (swls_total <= 25),
        swls_total == 20,
        (swls_total >= 15) & (swls_total <= 19),
        (swls_total >= 10) & (swls_total <= 14),
        (swls_total >= 5) & (swls_total <= 9)
    ]
    choices = [7, 6, 5, 4, 3, 2, 1]

    return np.select(conditions, choices, default=4)


# ============================================================================
# Apply Mappings to Files
# ============================================================================

def apply_swls_equating(df, swls_mapping):
    """Apply SWLS equating to a dataframe."""
    # First, convert swls_total to 7-scale if needed
    if 'life_satisfaction_7scale' not in df.columns and 'swls_total' in df.columns:
        df['life_satisfaction_7scale'] = convert_swls_total_to_7scale(df['swls_total'].values)
        print("  Created 'life_satisfaction_7scale' from 'swls_total'")

    if 'life_satisfaction_7scale' in df.columns:
        df['swls_equated'] = df['life_satisfaction_7scale'].map(swls_mapping)
        print(f"  Applied SWLS mapping: 1-7 -> 1-10 (column: swls_equated)")
        return True
    return False


def apply_ohq_equating(df, ohq_mapping):
    """Apply OHQ equating to a dataframe."""
    if 'ohq_score' in df.columns:
        # Round OHQ score to nearest integer for mapping
        df['ohq_score_rounded'] = df['ohq_score'].round().astype(int)
        df['ohq_equated'] = df['ohq_score_rounded'].map(ohq_mapping)
        print(f"  Applied OHQ mapping: 1-6 -> 1-10 (column: ohq_equated)")
        return True
    return False


def process_synthetic_file(path, swls_mapping, ohq_mapping):
    """Process a single synthetic file, applying appropriate equating."""
    df = pd.read_csv(path)
    print(f"\nProcessing: {path.name} ({len(df)} rows)")

    filename_lower = path.name.lower()
    updated = False
    equated_col = None

    if 'swls' in filename_lower:
        updated = apply_swls_equating(df, swls_mapping)
        equated_col = 'swls_equated'
    elif 'ohq' in filename_lower:
        updated = apply_ohq_equating(df, ohq_mapping)
        equated_col = 'ohq_equated'
    else:
        print("  [SKIP] Not an SWLS or OHQ file")
        return False

    if updated:
        df.to_csv(path, index=False)
        print(f"  Saved updated file")

        # Show distribution of equated scores
        if equated_col and equated_col in df.columns:
            equated = df[equated_col].dropna()
            print(f"  Equated scores: mean={equated.mean():.2f}, std={equated.std():.2f}")

    return updated


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EQUIPERCENTILE EQUATING FOR SWLS AND OHQ SCALES")
    print("="*70)

    # Load and prepare real WVS data
    df_base = load_wvs_data(WVS_FILE)
    print(f"Loaded WVS dataset: {len(df_base)} rows")

    df_categorized = categorize_dataframe(df_base)
    df_for_comparison = prepare_comparison_subset(df_categorized, TARGET_COUNTRIES)
    print(f"Comparison subset: {len(df_for_comparison)} rows")

    # Take 10% subset for equating (to avoid overfitting)
    df_equate = sample_wvs_for_equating(df_for_comparison, frac=0.1, random_state=42)

    # Build weighted WVS target distribution
    target_values, target_cdf = build_weighted_target_distribution(
        df_equate,
        score_col="life_satisfaction",
        weight_col="weight"
    )

    print(f"\nTarget distribution (WVS):")
    print(f"  Values: {target_values}")
    print(f"  CDF: {np.round(target_cdf, 3)}")

    # Build equipercentile mappings
    swls_mapping = build_equip_mapping_discrete(
        source_min=1,
        source_max=7,
        target_values=target_values,
        target_cdf=target_cdf,
        target_range=(1, 10)
    )

    ohq_mapping = build_equip_mapping_discrete(
        source_min=1,
        source_max=6,
        target_values=target_values,
        target_cdf=target_cdf,
        target_range=(1, 10)
    )

    print("\n" + "="*70)
    print("EQUIPERCENTILE MAPPINGS")
    print("="*70)

    print("\nSWLS mapping (1-7 -> 1-10):")
    for k, v in swls_mapping.items():
        print(f"  {k} -> {v}")

    print("\nOHQ mapping (1-6 -> 1-10):")
    for k, v in ohq_mapping.items():
        print(f"  {k} -> {v}")

    # Process synthetic files
    print("\n" + "="*70)
    print("PROCESSING SYNTHETIC FILES")
    print("="*70)

    # Find SWLS and OHQ weighted files
    swls_files = list(SYNTHETIC_WEIGHTED_DIR.glob("synthetic_swls_*_w.csv"))
    ohq_files = list(SYNTHETIC_WEIGHTED_DIR.glob("synthetic_ohq_*_w.csv"))

    all_files = swls_files + ohq_files

    if not all_files:
        print(f"\nNo SWLS/OHQ files found in {SYNTHETIC_WEIGHTED_DIR}")
    else:
        print(f"\nFound {len(all_files)} files to process")

    processed = 0
    for path in sorted(all_files):
        if process_synthetic_file(path, swls_mapping, ohq_mapping):
            processed += 1

    print("\n" + "="*70)
    print(f"Equipercentile equating complete!")
    print(f"Processed {processed} files")
    print("="*70)
