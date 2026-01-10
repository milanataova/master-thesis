# Combining Scales: Ensemble Approaches

This directory contains implementations of two ensemble approaches that combine multiple questionnaire scales:

1. **KS-Based Weighting**: Optimal weights determined by inverse KS statistics
2. **Equal Weights (Simple Averaging)**: All scales weighted equally at 25% each

## Overview

Both ensemble methods combine predictions from multiple questionnaire variants to achieve better performance than any individual scale. The key difference is how weights are assigned to each questionnaire.

## Methodology

### Ensemble Composition

**Included in Ensemble:**
- Original WVS (Base)
- Cantril Ladder
- SWLS (Satisfaction with Life Scale)
- OHQ (Oxford Happiness Questionnaire)

**Excluded from Ensemble:**
- Reverse Scale (poor performance, high cognitive difficulty)

### Weighting Strategies

#### Approach 1: KS-Based Weighting

For each model, weights are calculated as:

1. Calculate average KS statistic across all countries for each questionnaire
2. Compute scores as: `score = 1 - KS`
3. Normalize scores to sum to 1: `weight = score / Σscores`

This approach gives higher weight to questionnaires with lower KS statistics (better fit to real data).

#### Approach 2: Equal Weights (Simple Averaging)

All questionnaires receive equal weight:

- Each questionnaire: **25% weight** (1/4)
- No optimization based on performance
- Simple, transparent averaging
- Treats all measurement approaches as equally valid

### Ensemble Distribution

For each country and model, the ensemble distribution is created by:

```
ensemble_prob(score) = Σ [weight(variant) × prob(variant, score)]
```

Where:
- Probabilities are weighted relative frequencies
- Weights sum to 1.0 for each model
- Result is normalized to ensure valid probability distribution

## Files

```
Combining scales/
├── ensemble_ks_weighting.py              # KS-based weighting ensemble
├── ensemble_equal_weights.py             # Equal weights (average) ensemble
├── plot_scatterplots_with_ensemble.py    # Scatterplots with KS-based ensemble
├── plot_scatterplots_with_average.py     # Scatterplots with average ensemble
├── plot_all_approaches_combined.py       # Combined plots showing ALL 7 variants
├── README.md                              # This file
└── Outputs/
    ├── ensemble_distributions.csv                # KS-based ensemble probabilities
    ├── ensemble_metrics.csv                      # KS-based ensemble metrics
    ├── combined_metrics_with_ensemble.csv        # All variants + KS ensemble
    ├── wasserstein_with_ensemble.png             # Wasserstein barplot (KS ensemble)
    ├── ks_with_ensemble.png                      # KS barplot (KS ensemble)
    ├── scatter_with_ensemble_*.png (4 files)     # Scatterplots with KS ensemble
    ├── average_distributions.csv                 # Average ensemble probabilities
    ├── average_metrics.csv                       # Average ensemble metrics
    ├── combined_metrics_with_average.csv         # All variants + average ensemble
    ├── wasserstein_with_average.png              # Wasserstein barplot (average)
    ├── ks_with_average.png                       # KS barplot (average)
    ├── scatter_with_average_*.png (4 files)      # Scatterplots with average ensemble
    ├── wasserstein_all_approaches.png            # Combined barplot (all 7 variants)
    ├── ks_all_approaches.png                     # Combined barplot (all 7 variants)
    └── scatter_all_approaches_*.png (4 files)    # Combined scatterplots (all 7 variants)
```

## Usage

### Running the KS-Based Weighting Approach

```bash
cd "Combining scales"
python ensemble_ks_weighting.py
python plot_scatterplots_with_ensemble.py
```

The scripts will:
1. Compute optimal weights based on average KS statistics
2. Create weighted ensemble distributions
3. Calculate ensemble metrics
4. Generate barplots and scatterplots

### Running the Equal Weights Approach

```bash
python ensemble_equal_weights.py
python plot_scatterplots_with_average.py
```

The scripts will:
1. Apply equal weights (25% each) to all questionnaires
2. Create simple average ensemble distributions
3. Calculate ensemble metrics
4. Generate barplots and scatterplots

### Creating Combined Visualizations (All 7 Variants Together)

After running both ensemble approaches, create combined plots:

```bash
python plot_all_approaches_combined.py
```

This generates comprehensive comparison plots showing:
- **All 5 individual questionnaires** (Original WVS, Reverse, Cantril, SWLS, OHQ)
- **KS-based ensemble** (optimized weights)
- **Average ensemble** (equal weights)

**Total: 7 variants in a single visualization** for direct comparison

Outputs:
- `wasserstein_all_approaches.png` - 2x2 barplot with all 7 variants
- `ks_all_approaches.png` - 2x2 barplot with all 7 variants
- `scatter_all_approaches_*.png` (4 files) - Scatterplots showing all distributions

### What Each Script Does

**Main ensemble scripts:**
- Load real WVS data with weights
- Load existing metrics from `../KS & Wasserstein Stats/Outputs/calculated_metrics_table.csv`
- Compute ensemble weights (KS-based or equal)
- Load all 5 synthetic datasets
- Create ensemble distributions using weighted relative frequencies
- Calculate KS and Wasserstein metrics
- Generate 2x2 grid barplots

**Scatterplot scripts:**
- Load ensemble distributions from CSV
- Create one figure per country with 3 subplots (one per model)
- Each subplot shows:
  - Real WVS distribution (black line)
  - All 5 synthetic questionnaire distributions (colored lines)
  - **Ensemble distribution (gray line)**
- Uses **weighted relative frequencies** for all distributions
- X-axis: Life satisfaction (1-10)
- Y-axis: Relative frequency

### Expected Output

```
Ensemble weights (w ∝ 1 - KS, averaged across countries):

llama3.1_8b:
  Cantril        : 0.255
  OHQ            : 0.179
  Original WVS   : 0.288
  SWLS           : 0.277
  Total          : 1.000

✅ Ensemble analysis complete!
✅ All outputs saved to: Outputs/
```

## Results

### Performance Comparison: KS-Based vs Equal Weights

**Average across all countries and models:**

| Metric | Original WVS | Reverse | Cantril | SWLS | OHQ | **KS Ensemble** | **Average** |
|--------|--------------|---------|---------|------|-----|-----------------|-------------|
| **KS Statistic** | 0.357 | 0.532 | 0.389 | 0.342 | 0.452 | **0.249** ✓ | 0.249 ✓ |
| **Wasserstein** | 0.951 | 2.528 | 1.240 | 1.029 | 1.455 | **0.699** ✓ | 0.712 |

**Key Findings:**

1. **Both ensemble approaches significantly outperform all individual questionnaires**
2. **KS-based weighting performs marginally better than equal weights:**
   - Same KS statistic (0.249)
   - 1.8% better Wasserstein distance (0.699 vs 0.712)
3. **The similarity suggests that all 4 questionnaires contribute relatively equally**

**Improvements over best individual scale (KS Ensemble):**
- **27% better KS** statistic (0.249 vs 0.342 SWLS)
- **26% better Wasserstein** distance (0.699 vs 0.951 Original WVS)

**Improvements over best individual scale (Average Ensemble):**
- **27% better KS** statistic (0.249 vs 0.342 SWLS)
- **25% better Wasserstein** distance (0.712 vs 0.951 Original WVS)

### Model-Specific Weights

#### KS-Based Weighting

The weights vary slightly by model based on their performance:

**llama3.1_8b:**
- Original WVS: 28.8%
- SWLS: 27.7%
- Cantril: 25.5%
- OHQ: 17.9%

**llama3.3_70b:**
- Original WVS: 26.4%
- Cantril: 25.9%
- SWLS: 24.3%
- OHQ: 23.3%

**qwen2.5_72b:**
- SWLS: 28.3%
- OHQ: 26.0%
- Original WVS: 22.8%
- Cantril: 22.8%

#### Equal Weights (Average)

**All models:**
- Original WVS: 25.0%
- Cantril: 25.0%
- SWLS: 25.0%
- OHQ: 25.0%

Simple averaging treats all questionnaires identically regardless of model performance.

### Best Performance by Country

**Ensemble KS Statistic:**
- USA: 0.107 - 0.229 (excellent to very good)
- Netherlands (NLD): 0.153 - 0.224 (excellent to very good)
- Indonesia (IDN): 0.272 - 0.307 (moderate)
- Mexico (MEX): 0.340 - 0.375 (moderate)

**Ensemble Wasserstein Distance:**
- Netherlands (NLD): 0.358 - 0.494 (excellent)
- USA: 0.485 - 0.756 (good)
- Indonesia (IDN): 0.747 - 0.956 (moderate)
- Mexico (MEX): 0.798 - 0.909 (moderate)

## Visualizations

### Barplots (2x2 Grid)

Both visualizations show a 2x2 grid with one subplot per country:
- Each subplot shows 6 grouped bars per model
- Bars represent: Original WVS, Reverse, Cantril, SWLS, OHQ, and Ensemble
- Color scheme matches the main project palette
- Ensemble is shown in gray (#BABDBF)

**Key Observation:** In almost all cases, the Ensemble bar is the shortest (best performance).

### Distribution Scatterplots with Ensemble

Four scatterplot figures (one per country), each containing:
- **3 subplots** (one per model: llama3.1_8b, llama3.3_70b, qwen2.5_72b)
- **7 lines per subplot:**
  - Real WVS (black, thickest line)
  - Original WVS (green)
  - Reverse (orange)
  - Cantril (blue)
  - SWLS (purple)
  - OHQ (red-orange)
  - **Ensemble (gray)**
- Connected scatter plots showing weighted relative frequencies
- Shared y-axis across all subplots for easy comparison
- Legend showing all questionnaire types

**Key Observation:** The Ensemble line (gray) typically follows a path between the individual questionnaires, often tracking closer to the Real WVS line than any single questionnaire, demonstrating the ensemble's ability to capture the true distribution more accurately.

### Combined Comparison Plots (All 7 Variants)

The most comprehensive visualizations showing all approaches together:

**Barplots (2x2 Grid):**
- Each subplot shows 7 grouped bars per model
- Includes all 5 questionnaires + both ensemble approaches
- Color-coded for easy distinction:
  - Green: Original WVS
  - Orange: Reverse
  - Blue: Cantril
  - Purple: SWLS
  - Red-Orange: OHQ
  - Light Gray: Ensemble (KS-based)
  - Dark Gray: Average (equal weights)

**Distribution Scatterplots:**
- Four figures (one per country)
- **8 lines per subplot:**
  - Real WVS (black, thickest)
  - 5 individual questionnaires (colored)
  - Ensemble (light gray)
  - Average (dark gray)
- Allows visual comparison of how both ensemble approaches compare to individual questionnaires

**Key Observation:** Both ensemble lines (light and dark gray) track very closely together, often overlapping, confirming their similar performance. Both consistently follow the Real WVS line more closely than any individual questionnaire.

## Output Files

### 1. ensemble_distributions.csv

Contains the ensemble probability distribution for each country and model:

| Country | Model | Score | Ensemble_Prob |
|---------|-------|-------|---------------|
| USA | llama3.1_8b | 1 | 0.0037 |
| USA | llama3.1_8b | 2 | 0.0074 |
| ... | ... | ... | ... |

- One row per (country, model, score) combination
- 10 scores × 4 countries × 3 models = 120 rows

### 2. ensemble_metrics.csv

Contains KS and Wasserstein metrics for the ensemble:

| Country | Model | Variant | KS | Wasserstein |
|---------|-------|---------|-----|-------------|
| USA | llama3.1_8b | Ensemble | 0.147 | 0.726 |
| ... | ... | ... | ... | ... |

- 4 countries × 3 models = 12 rows

### 3. combined_metrics_with_ensemble.csv

Combines original metrics (5 questionnaires) with ensemble metrics:

- Original metrics: 5 variants × 4 countries × 3 models = 60 rows
- Ensemble metrics: 4 countries × 3 models = 12 rows
- **Total: 72 rows**

This file is used to generate the comparison barplots.

## Interpretation

### Why the Ensemble Works

1. **Complementary Strengths**: Different questionnaires capture different aspects of life satisfaction
   - Original WVS: Direct, simple framing
   - Cantril: Best-possible-life reference point
   - SWLS: Multi-item reliability
   - OHQ: Comprehensive happiness assessment

2. **Error Averaging**: Random errors in individual questionnaires tend to cancel out when combined

3. **Optimal Weighting**: KS-based weighting automatically adjusts for questionnaire quality

4. **Model-Specific Adaptation**: Weights are computed per model, accounting for model-specific biases

### When to Use the Ensemble

The ensemble is particularly beneficial when:
- Maximum accuracy is required
- You want to reduce model-specific bias
- Multiple data collection methods are feasible
- Robustness across different populations is important

### Limitations

- Requires collecting data with multiple questionnaires
- More complex to administer than single-scale approach
- Weights may not generalize to very different populations
- Excludes potentially useful information from Reverse scale

## Technical Details

### Weighting Formula

For model `m` and variant `v`:

```
score(m, v) = 1 - mean(KS(m, v, c) for c in countries)
weight(m, v) = score(m, v) / Σ score(m, v')
```

Where:
- `KS(m, v, c)` is the KS statistic for model m, variant v, country c
- Weights are normalized to sum to 1.0

### Statistical Metrics

**KS Statistic (Kolmogorov-Smirnov):**
```
KS = max|CDF_real(x) - CDF_synthetic(x)|
```

**Wasserstein Distance:**
```
W = Σ|CDF_real(x) - CDF_synthetic(x)|
```

Both use weighted CDFs based on survey weights (real data) and matching weights (synthetic data).

## Dependencies

The script requires the following inputs from other directories:

1. **Real WVS Data**: `../WVS_Cross-National_Wave_7_csv_v6_0.csv`
2. **Existing Metrics**: `../KS & Wasserstein Stats/Outputs/calculated_metrics_table.csv`
3. **Synthetic Datasets**: `../synthetic_*_w.csv` (15 files)

## Comparison and Recommendations

### When to Use KS-Based Weighting

**Advantages:**
- Slightly better performance (1.8% lower Wasserstein distance)
- Automatically adapts to model-specific strengths
- Data-driven optimization
- Weights reflect questionnaire quality

**Disadvantages:**
- More complex to explain
- Requires existing performance data
- May overfit to specific dataset characteristics

**Best for:**
- Maximizing predictive accuracy
- Research contexts where optimization is valued
- When model-specific adaptation is important

### When to Use Equal Weights (Average)

**Advantages:**
- Nearly identical performance (same KS, 1.8% worse Wasserstein)
- Simple, transparent, easy to explain
- No optimization required
- Treats all measurement approaches as equally valid
- More robust to overfitting

**Disadvantages:**
- Slightly suboptimal performance
- Doesn't account for questionnaire quality differences
- Same weights for all models

**Best for:**
- Simplicity and transparency are priorities
- When performance difference is negligible
- Stakeholder communication
- When avoiding optimization bias is important

### Practical Recommendation

Given that both approaches achieve nearly identical KS statistics (0.249) and the Wasserstein difference is minimal (0.699 vs 0.712, only 1.8%), **the equal weights approach is recommended for most practical applications** due to its simplicity and transparency, unless maximum optimization is critical.

The small performance gap suggests that all four questionnaires (Original WVS, Cantril, SWLS, OHQ) contribute roughly equally to ensemble performance, validating the equal weights approach.

## Summary Statistics

### Complete Outputs Generated

**CSV Files (6):**
- ensemble_distributions.csv (KS-based)
- ensemble_metrics.csv (KS-based)
- combined_metrics_with_ensemble.csv (KS-based)
- average_distributions.csv (Equal weights)
- average_metrics.csv (Equal weights)
- combined_metrics_with_average.csv (Equal weights)

**Visualization Files (18):**
- **Separate approaches:**
  - 2 barplots × 2 approaches = 4 files
  - 4 scatterplots × 2 approaches = 8 files
- **Combined approach (all 7 variants):**
  - 2 barplots (KS and Wasserstein)
  - 4 scatterplots (one per country)

**Total: 24 output files** (6 CSV + 18 PNG)

## Citation

This ensemble approach is part of:
**"Synthetic Survey Generation for Life Satisfaction Research using Large Language Models"**
Master's Thesis, University of Mannheim, 2025

---

For questions about the methodology or implementation, refer to the main thesis document or the code comments in `ensemble_ks_weighting.py` and `ensemble_equal_weights.py`.
