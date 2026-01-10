"""
02_data_preparation.py

Prepare World Values Survey data for comparison with synthetic responses.

This script loads WVS Wave 7 data, categorizes income and health into groups,
and creates a clean subset for comparing real vs. synthetic life satisfaction
distributions.

Output: Prepared DataFrame with columns:
    - country
    - life_satisfaction (1-10)
    - income_level (Low/Medium/High)
    - health_level (Good/Fair/Poor)
    - weight (survey weight)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "Datasets" / "WVS_Cross-National_Wave_7_csv_v6_0.csv"
OUTPUT_DIR = BASE_DIR / "Datasets"

# Target countries for analysis
TARGET_COUNTRIES = ["USA", "IDN", "NLD", "MEX"]

# ============================================================================
# Data Loading and Processing Functions
# ============================================================================

def load_wvs_data(file_path):
    """
    Loads the WVS Wave 7 dataset, renames columns, and replaces
    special negative values with NaN.
    """
    columns = {
        # Columns for comparison
        "Q49": "life_satisfaction",
        "Q288": "income",
        "Q47": "health",
        "Q260": "gender",
        "B_COUNTRY_ALPHA": "country",

        # Additional columns (kept for potential future use)
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
        "W_WEIGHT": "weight"
    }

    print("Loading WVS Wave 7 dataset...")
    df = pd.read_csv(file_path, low_memory=False)

    # Select and rename only the columns we need
    df = df[list(columns.keys())].rename(columns=columns)

    # Replace all specified negative values with NaN
    missing_values = [-1, -2, -4, -5]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(missing_values, np.nan)

    # Filter for valid life satisfaction scores (1-10)
    df = df[df["life_satisfaction"].between(0, 10) | df["life_satisfaction"].isna()]

    return df


def categorize_dataframe(df):
    """
    Categorize income and health into groups.

    Income levels (based on 10-step scale):
        - Low: steps 1-3
        - Medium: steps 4-7
        - High: steps 8-10

    Health levels (based on 5-point scale):
        - Good: 1-2 (Very good, Good)
        - Fair: 3 (Fair)
        - Poor: 4-5 (Poor, Very poor)
    """
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
    df_categorized['income_level'] = df_categorized['income'].apply(categorize_income)
    df_categorized['health_level'] = df_categorized['health'].apply(categorize_health)
    return df_categorized


def prepare_real_data(file_path, target_countries):
    """
    Load and prepare real WVS data for comparison.

    Returns DataFrame with columns:
        - country
        - life_satisfaction
        - income_level
        - health_level
        - weight
    """
    # Load data
    df = load_wvs_data(file_path)

    # Categorize income and health
    df = categorize_dataframe(df)

    # Filter for target countries and required columns
    essential_cols = ["life_satisfaction", "country", "income_level", "health_level", "weight"]
    df = df[df['country'].isin(target_countries)][essential_cols].copy()

    # Drop NaNs
    initial_rows = len(df)
    df.dropna(inplace=True)

    print(f"Loaded {len(df)} real WVS records for {len(target_countries)} countries")
    print(f"  (dropped {initial_rows - len(df)} rows with missing values)")

    return df


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Prepare real data for comparison
    df_real = prepare_real_data(DATA_PATH, TARGET_COUNTRIES)

    # Display summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)

    print(f"\nTotal records: {len(df_real)}")

    print("\nRecords by country:")
    print(df_real['country'].value_counts().to_string())

    print("\nRecords by income level:")
    print(df_real['income_level'].value_counts().to_string())

    print("\nRecords by health level:")
    print(df_real['health_level'].value_counts().to_string())

    print("\nLife satisfaction distribution:")
    print(df_real['life_satisfaction'].describe().to_string())

    # Cross-tabulation of segments
    print("\n" + "="*60)
    print("SEGMENT SIZES (Income x Health)")
    print("="*60)
    segment_counts = df_real.groupby(['country', 'income_level', 'health_level']).size()
    print(segment_counts.unstack(fill_value=0).to_string())

    # Optionally save prepared data
    # output_path = OUTPUT_DIR / "wvs_prepared.csv"
    # df_real.to_csv(output_path, index=False)
    # print(f"\nSaved prepared data to {output_path}")
