# Final Analysis Pipeline - Usage Guide

## Quick Start

Run the complete analysis pipeline with a single command:

```bash
bash scripts/claude/run_final_analysis.sh
```

## What Gets Generated

The pipeline creates the following outputs in `results/final_analysis/`:

### 1. Multivariate Regression Results
**File**: `multivariate_regression_results.txt`

Contains:
- Univariate regression models (4 predictors individually)
- Bivariate regression models (all pairs of predictors)
- **Trivariate regression models** (all combinations of 3 predictors)
- Statistical metrics: R², Adjusted R², F-statistic, p-values
- Regression coefficients for each model

### 2. Inter-Cohort Tables (TCGA vs CPTAC)
**File**: `tables/intercohort_tables.tex`

Two LaTeX tables:
- **Table 1 - Ordinal features**: 11 classes × 3 features
  - Tubule Formation, Nuclear Pleomorphism, Mitotic Activity
  - Mann-Whitney U test with rank-biserial correlation
  
- **Table 2 - Binary features**: 11 classes × 3 features
  - Tumour Necrosis, Lymphocytic Infiltrate, Polymorphonuclear Infiltrate
  - Chi-square test with Cramér's V

### 3. Intra-Cohort Split Tables (TCGA)
**File**: `tables/intra_cohort_split_tables.tex`

Two LaTeX tables:
- **Table 1 - PAM50 + ER**: 6 features × 2 tasks
- **Table 2 - PR + HER2**: 6 features × 2 tasks

## Manual Execution

If you want to run scripts individually:

### 1. Multivariate Regression
```bash
python src/claude/final/compare_linear_regressions.py
```

**Configuration** (edit script before running):
```python
EXCLUDE_CLASSES = ['HER2-positive', 'PR-positive', 'ER-positive']
REMOVE_OUTLIERS = False
SCALE_PREDICTORS = False
```

### 2. Inter-Cohort Tables
```bash
python src/claude/final/generate_intercohort_latex_table.py \
    results/biological_analysis/pam50_biological_interpretability.json \
    output.tex
```

**Note**: Requires biological analysis to be run first:
```bash
conda run -n clam_latest python src/claude/biological_interpretability_analysis.py \
    -a data/histomorfologico/representative_images_annotation.xlsx \
    -t pam50 \
    -o results/biological_analysis/
```

### 3. Intra-Cohort Split Tables
```bash
python src/claude/final/split_intra_cohort_table.py output.tex
```

## Verification

Before running the pipeline, verify your setup:

```bash
conda run -n clam_latest python src/claude/final/verify_setup.py
```

This checks:
- Python dependencies (numpy, pandas, scipy, sklearn, matplotlib, seaborn)
- Required scripts are present
- Data files are available
- Output directories exist

## Environment

The pipeline uses the `clam_latest` conda environment. Make sure it's activated:

```bash
conda activate clam_latest
```

Or use `conda run -n clam_latest` prefix for individual commands.

## Troubleshooting

**Issue**: Biological analysis results not found
**Solution**: The pipeline will automatically run biological analysis if needed

**Issue**: NumPy version mismatch
**Solution**: Already handled - numpy 1.x is installed in clam_latest environment

**Issue**: Permission denied on script
**Solution**: Make sure the script is executable:
```bash
chmod +x scripts/claude/run_final_analysis.sh
```

## Output Structure

```
results/final_analysis/
├── multivariate_regression_results.txt   # Full regression output
├── tables/
│   ├── intercohort_tables.tex           # TCGA vs CPTAC tables
│   └── intra_cohort_split_tables.tex    # TCGA split tables
└── figures/                              # (reserved for future plots)
```

## Next Steps

After running the pipeline:

1. Review multivariate regression results in `multivariate_regression_results.txt`
2. Import LaTeX tables into your manuscript:
   ```latex
   \input{results/final_analysis/tables/intercohort_tables.tex}
   \input{results/final_analysis/tables/intra_cohort_split_tables.tex}
   ```
3. Check statistical significance markers (* p<0.05, ** p<0.01, *** p<0.001)

