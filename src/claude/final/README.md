# Final Analysis Scripts

This directory contains the final analysis scripts for the domain shift and biological interpretability study.

## Scripts

### 1. `compare_linear_regressions.py`
**Purpose**: Multivariate regression analysis to identify predictors of domain shift.

**What it does**:
- Analyzes the relationship between biological/distributional features and model performance degradation
- Computes univariate, bivariate, and trivariate regression models
- Predictors:
  - `B_c`: Biological shift indicator (morphological separability)
  - `d_c`: Domain shift (distributional distance)
  - `Δp_c`: Prevalence shift (change in class prevalence)
  - `NPG`: Normalized performance gap
- Output: `RPC` (Relative Performance Change) as response variable

**Configuration**:
```python
REMOVE_OUTLIERS = False          # Enable/disable outlier removal
SCALE_PREDICTORS = False         # Enable/disable Min-Max scaling
SUBTRACT_TCGA_EFFECT_SIZE = False  # Adjust B_c by within-cohort effect
EXCLUDE_CLASSES = ['HER2-positive', 'PR-positive', 'ER-positive']
```

**Key Output** (STEP 6):
- Trivariate models with R², adjusted R², F-statistic, p-values
- Regression coefficients for each model

---

### 2. `generate_intercohort_latex_table.py`
**Purpose**: Generate LaTeX tables comparing TCGA vs CPTAC histomorphological features.

**What it does**:
- Creates two separate tables:
  - **Ordinal features**: Tubule Formation, Nuclear Pleomorphism, Mitotic Activity
    - Test: Mann-Whitney U with rank-biserial correlation
  - **Binary features**: Tumour Necrosis, Lymphocytic Infiltrate, Polymorphonuclear Infiltrate
    - Test: Chi-square with Cramér's V
- Transposes data: classes as rows, features as columns (11×3 format)

**Usage**:
```bash
python generate_intercohort_latex_table.py \
    results/biological_analysis/pam50_biological_interpretability.json \
    output_tables.tex
```

**Output**: Two `sidewaystable` environments ready for LaTeX compilation.

---

### 3. `split_intra_cohort_table.py`
**Purpose**: Split TCGA intra-cohort statistical tables into two compact tables.

**What it does**:
- Splits original 4-task table (PAM50, ER, PR, HER2) into two tables:
  - **Table 1**: PAM50 + ER (6 features × 2 tasks)
  - **Table 2**: PR + HER2 (6 features × 2 tasks)
- Hardcoded data from original analysis results
- Format: `lcccc` (5 columns total per table)

**Usage**:
```bash
python split_intra_cohort_table.py output_split_tables.tex
```

**Output**: Two `table` environments (non-sideways) for standard page layout.

---

### 4. `distributional_shift_analysis.py`

**Purpose**: Core module for analyzing distributional shifts between TCGA and CPTAC cohorts.

**What it does**:

- Computes prevalence shift metrics: **Δp** (signed difference), Δp% (relative change)
- Calculates performance degradation: **RPC** (Relative Performance Change)
- Performs correlation analysis: Pearson/Spearman correlation between Δp and RPC
- Generates visualizations:
  - Prevalence comparison (TCGA vs CPTAC)
  - Signed prevalence shift bar chart (red=decrease, green=increase)
  - Correlation scatter plot (Δp vs RPC)
  - Combined 4-panel figure

**Key Metrics**:

- **Δp** = Prevalence_CPTAC - Prevalence_TCGA (with sign, NOT absolute value)
- **RPC** = (PR_AUC_HO - PR_AUC_MCCV) / PR_AUC_MCCV × 100%
- Correlation tests include significance testing (p-values)

**Usage**:

```python
from distributional_shift_analysis import compute_distributional_shift

results = compute_distributional_shift(
    tcga_labels_path='data/dataset_csv/tcga-subtype_pam50.csv',
    cptac_labels_path='data/dataset_csv/cptac-subtype_pam50.csv',
    metrics_path='data/dataset_csv/statistics/pam50_performance_metrics_fake.csv',
    task_name='pam50',
    output_dir='results/distributional_shift'
)
```

**Output Files**:

- `{task}_prevalence_shift.csv` - Prevalence data for each class
- `{task}_performance_degradation.csv` - Performance metrics per class
- `{task}_distributional_shift_analysis.json` - Complete results with correlation statistics
- `{task}_*.png` - Visualization figures

**Important Note**:
For binary classification tasks (ER, PR, HER2), correlation r=-1.0 is a mathematical artifact (2 complementary data points), NOT a real correlation.

---

### 5. `run_all_distributional_shift.py`

**Purpose**: Master script to run distributional shift analysis for all tasks in one command.

**What it does**:

- Runs distributional shift analysis for all 4 tasks: PAM50, ER, PR, HER2
- Generates individual task outputs (CSVs, JSONs, visualizations)
- Extracts summary metrics into combined CSV table
- Creates LaTeX summary table with correlation results for all tasks

**Configuration**:

```python
TASKS = {
    'pam50': {
        'tcga_labels': 'data/dataset_csv/tcga-subtype_pam50.csv',
        'cptac_labels': 'data/dataset_csv/cptac-subtype_pam50.csv',
        'metrics': 'data/dataset_csv/statistics/pam50_performance_metrics_fake.csv',
    },
    'er': {...},
    'pr': {...},
    'her2': {...}
}
```

**Usage**:

```bash
# Run all tasks
python src/claude/final/run_all_distributional_shift.py --output results/distributional_shift

# Or use the shell wrapper
bash scripts/claude/run_distributional_shift_analysis.sh [output_dir]
```

**Output Structure**:

```
results/distributional_shift/
├── distributional_shift_summary.csv         # Summary metrics for all tasks
├── distributional_shift_summary_table.tex   # LaTeX summary table
├── pam50_*.csv, pam50_*.json, pam50_*.png  # PAM50 results
├── er_*.csv, er_*.json, er_*.png           # ER results
├── pr_*.csv, pr_*.json, pr_*.png           # PR results
└── her2_*.csv, her2_*.json, her2_*.png     # HER2 results
```

**Key Outputs**:

- **Summary CSV**: Contains n_classes, correlation coefficients (r), p-values for all tasks
- **LaTeX Table**: Ready-to-use table for manuscript with significance markers (*, **, ***)

---

## Automated Pipelines

### Distributional Shift Analysis Pipeline

Run distributional shift analysis for all tasks:

```bash
bash scripts/claude/run_distributional_shift_analysis.sh [output_dir]
```

This pipeline:

1. Runs distributional shift analysis for PAM50, ER, PR, HER2 tasks
2. Generates prevalence shift metrics (signed Δp)
3. Computes performance degradation (RPC)
4. Performs correlation analysis (Δp vs RPC)
5. Creates visualizations for each task
6. Generates summary CSV and LaTeX table

**Output Structure**:

```
results/distributional_shift/
├── distributional_shift_summary.csv
├── distributional_shift_summary_table.tex
├── pam50_prevalence_shift.csv
├── pam50_performance_degradation.csv
├── pam50_distributional_shift_analysis.json
├── pam50_prevalence_comparison.png
├── pam50_prevalence_shift.png
├── pam50_correlation.png
├── pam50_combined.png
└── (similar files for er, pr, her2)
```

---

### Final Analysis Pipeline

Run all final analysis scripts:

```bash
bash scripts/claude/run_final_analysis.sh
```

This pipeline:

1. Runs multivariate regression analysis
2. Generates inter-cohort LaTeX tables (requires biological analysis results)
3. Generates intra-cohort split tables
4. Saves all outputs to `results/final_analysis/`

**Output Structure**:

```
results/final_analysis/
├── multivariate_regression_results.txt
├── tables/
│   ├── intercohort_tables.tex
│   └── intra_cohort_split_tables.tex
└── figures/
    └── (regression plots if enabled)
```

---

## Dependencies

- Python 3.10+
- NumPy, Pandas, SciPy, Scikit-learn
- Matplotlib, Seaborn (for visualizations)

---

## Notes

- All scripts are self-contained and can be run independently
- Configuration parameters are at the top of each script
- LaTeX tables use `booktabs` package (`\toprule`, `\midrule`, `\bottomrule`)
- Statistical significance: * p<0.05, ** p<0.01, *** p<0.001
