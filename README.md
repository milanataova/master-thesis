# Thesis Repository

Analysis code for Master's thesis on synthetic survey generation using LLMs.

## Folder Structure

```
Thesis/
├── Datasets/
│   ├── WVS_Cross-National_Wave_7_csv_v6_0.csv   # World Values Survey data
│   ├── overall_metrics_table.csv
│   └── synthetic_data/
│       ├── synthetic_raw/                       # Raw LLM outputs
│       └── weighted/                            # With survey weights applied
│
├── Scripts/
│   ├── data_preparation/                        # Data pipeline scripts
│   │   ├── 01_feature_selection.py
│   │   ├── 02_data_preparation.py
│   │   ├── 03_weight_generation.py
│   │   ├── 04_equipercentile_equating.py
│   │   └── 05_synthetic_generation.py
│   │
│   ├── analysis/                                # Statistical analysis
│   │   ├── calculate_metrics_and_barplots.py
│   │   ├── subgroup_analysis.py
│   │   └── create_specification_plot.py
│   │
│   ├── plotting/                                # Visualization scripts
│   │   ├── plot_scatterplots.py
│   │   ├── generate_umap_plots.py
│   │   └── generate_subgroup_figures.py
│   │
│   ├── Combining scales/                        # Ensemble methods
│   │   ├── ensemble_equal_weights.py
│   │   ├── ensemble_ks_weighting.py
│   │   ├── plot_scatterplots_with_ensemble.py
│   │   ├── plot_scatterplots_with_average.py
│   │   └── plot_all_approaches_combined.py
│   │
│   ├── score_level_analysis.py                  # Per-score accuracy analysis
│   └── countries_choice.py                      # Country selection analysis
│
└── Outputs/
    ├── plots/
    │   ├── barplots/                            # Metric comparison bar plots
    │   ├── scatterplots/                        # Distribution scatterplots
    │   ├── umap/                                # UMAP projections
    │   ├── ensemble/                            # Ensemble method plots
    │   ├── subgroup_analysis/                   # Subgroup figures
    │   └── score_level_analysis/                # Score-level plots
    │
    └── metrics_tables/
        ├── KS and Wasserstein metrics/          # Basic metrics
        ├── ensemble/                            # Ensemble results
        ├── subgroup_analysis/                   # Subgroup tables (12 CSVs)
        └── score_level_analysis/                # Score-level tables (5 CSVs)
```

## Running Scripts

Scripts should be run from the repository root:

```bash
# Data preparation (run in order)
python Scripts/data_preparation/01_feature_selection.py
python Scripts/data_preparation/02_data_preparation.py
python Scripts/data_preparation/03_weight_generation.py
python Scripts/data_preparation/04_equipercentile_equating.py

# Analysis
python Scripts/analysis/subgroup_analysis.py
python Scripts/analysis/calculate_metrics_and_barplots.py

# Plotting
python Scripts/plotting/plot_scatterplots.py
python Scripts/plotting/generate_umap_plots.py
python Scripts/plotting/generate_subgroup_figures.py
python Scripts/analysis/create_specification_plot.py

# Additional
python Scripts/score_level_analysis.py
```

## Dependencies

Install required packages:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn umap-learn tqdm
```

For synthetic data generation (05_synthetic_generation.py):
```bash
pip install openai
```

### Package List

| Package | Used for |
|---------|----------|
| pandas | Data manipulation |
| numpy | Numerical operations |
| scipy | Statistical tests (Wasserstein, KS, ANOVA) |
| matplotlib | Plotting |
| seaborn | Statistical visualizations |
| scikit-learn | Random Forest feature selection |
| umap-learn | UMAP dimensionality reduction |
| tqdm | Progress bars |
| openai | LLM API calls (synthetic generation only) |
