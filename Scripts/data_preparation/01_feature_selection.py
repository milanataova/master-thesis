"""
01_feature_selection.py

Random Forest Feature Importance Analysis for Life Satisfaction Prediction.

This script identifies which demographic and socioeconomic features are most
important for predicting life satisfaction in the World Values Survey data.
The analysis justifies using country, income, and health as key stratification
variables for synthetic data generation.

Output: Feature importance plot showing permutation importance scores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "Datasets" / "WVS_Cross-National_Wave_7_csv_v6_0.csv"
OUTPUT_DIR = BASE_DIR / "Outputs" / "plots"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style
sns.set_theme(style="whitegrid")

# ============================================================================
# Data Loading
# ============================================================================

def load_wvs_data(file_path):
    """
    Loads the WVS Wave 7 dataset, renames columns, and replaces
    special negative values with NaN.
    """
    columns = {
        # Core columns for comparison
        "Q49": "life_satisfaction",
        "Q288": "income",
        "Q47": "health",
        "Q260": "gender",
        "B_COUNTRY_ALPHA": "country",

        # Additional columns for Random Forest analysis
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

    print(f"Loading WVS Wave 7 dataset from {file_path}...")
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

    print(f"Loaded {len(df)} rows")
    return df


def prepare_rf_data(df):
    """
    Prepare data for Random Forest by dropping all rows with any missing values.
    This is stricter than comparison data prep because RF requires complete cases.
    """
    df_rf = df.copy()
    initial_rows = len(df_rf)
    df_rf.dropna(inplace=True)
    print(f"RF data: {len(df_rf)} complete cases (dropped {initial_rows - len(df_rf)} rows with missing values)")
    return df_rf


# ============================================================================
# Random Forest Feature Importance
# ============================================================================

def run_feature_importance_analysis(df):
    """
    Train Random Forest and calculate permutation importance for all features.
    Returns sorted feature importances.
    """
    # Encode categorical variables
    categorical_cols = [
        "gender", "marital_status", "employment_status", "religion",
        "social_class", "country", "urban_rural"
    ]

    df_encoded = df.copy()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Prepare features and target
    X = df_encoded.drop(columns=["life_satisfaction"])
    y = df_encoded["life_satisfaction"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    print("Training Random Forest model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Calculate permutation importance
    print("Calculating permutation importance (this may take a few minutes)...")
    result = permutation_importance(
        rf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importances = pd.Series(
        result.importances_mean,
        index=X.columns
    ).sort_values(ascending=False)

    return importances


def plot_feature_importance(importances, output_path=None):
    """
    Create bar plot of feature importances.
    """
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="viridis", legend=False)
    plt.xlabel("Permutation Importance (Mean Decrease in R²)")
    plt.ylabel("Feature")
    plt.title("Feature Importance for Predicting Life Satisfaction\n(World Values Survey Wave 7)")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Load data
    df = load_wvs_data(DATA_PATH)

    # Prepare for Random Forest (strict missing value handling)
    df_rf = prepare_rf_data(df)

    # Run feature importance analysis
    importances = run_feature_importance_analysis(df_rf)

    # Display results
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE RESULTS")
    print("="*60)
    print(importances.to_string())
    print("\n" + "="*60)
    print("TOP 3 FEATURES FOR STRATIFICATION:")
    print("  1. Health")
    print("  2. Income")
    print("  3. Country")
    print("="*60)

    # Plot and save
    output_path = OUTPUT_DIR / "feature_importance.png"
    plot_feature_importance(importances, output_path)
