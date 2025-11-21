# Domain Shift Analysis Pipeline

Comprehensive scripts for analyzing domain shift in foundation model-based histopathology classification, following the methodology described in the Materials and Methods section.

## 📋 Overview

This pipeline implements four complementary analyses to characterize domain shift:

1. **Stain Normalization Robustness** - Assess whether staining variability contributes to domain shift
2. **Feature Space Consistency** - Evaluate if feature representations are consistent across cohorts
3. **Distributional Shift** - Test if class prevalence differences explain performance degradation
4. **Biological Interpretability** - Analyze morphological patterns and cross-cohort differences
5. **Integrative Analysis** - Multiple regression to identify independent contributors

## 📂 Scripts

### 1. Stain Normalization Analysis

**Purpose:** Determine if color normalization significantly affects external validation performance.

**Files:**
- `stain_normalization_analysis.py` - Full analysis from prediction files
- `stain_normalization_from_accuracy.py` - Simplified analysis from accuracy tables

**Statistical Test:** McNemar's test

**Outputs:**
- Per-class accuracy comparison
- McNemar test results (χ², p-value)
- Visualizations (accuracy bars, delta accuracy, confusion matrices)

**Usage:**

```bash
# From accuracy data (recommended for quick analysis)
python src/claude/stain_normalization_from_accuracy.py \
  --input data/stain_analysis/pam50_accuracy.csv \
  --task pam50 \
  --output results/stain_analysis/

# From full predictions
python src/claude/stain_normalization_analysis.py \
  --original results/pam50/hold_out_predictions.pkl \
  --normalized results/pam50_macenko/hold_out_predictions.pkl \
  --task pam50 \
  --output results/stain_analysis/
```

**Input Format (CSV):**
```csv
Class,N_samples,Accuracy_Original,Accuracy_Normalized
LumA,150,0.750,0.780
LumB,80,0.650,0.680
Her2,45,0.600,0.620
Basal,70,0.700,0.710
Normal,42,0.550,0.560
```

---

### 2. Feature Space Consistency Analysis

**Purpose:** Assess whether embeddings for high-attention patches are consistent across cohorts.

**Script:** `feature_space_consistency_analysis.py`

**Metrics:**
- Centroid distances (Euclidean)
- Intra-class variability
- Silhouette coefficients

**Outputs:**
- Centroid distances vs intra-class std
- Silhouette score comparisons
- t-SNE visualizations

**Usage:**

```bash
python src/claude/feature_space_consistency_analysis.py \
  --features_dir data/aggregated_features/ \
  --task pam50 \
  --top_k 8 \
  --output results/feature_space_analysis/
```

**Expected Directory Structure:**
```
data/aggregated_features/
├── tcga/
│   ├── TCGA-A1-001_features.pt
│   ├── TCGA-A1-001_attention.pt
│   └── labels.csv
└── cptac/
    ├── CPTAC-01_features.pt
    ├── CPTAC-01_attention.pt
    └── labels.csv
```

---

### 3. Distributional Shift Analysis

**Purpose:** Test correlation between prevalence shift and performance degradation.

**Script:** `distributional_shift_analysis.py`

**Statistical Tests:**
- Pearson correlation
- Spearman correlation
- Linear regression

**Outputs:**
- Prevalence comparison (TCGA vs CPTAC)
- Correlation scatter plot
- Combined prevalence + performance analysis

**Usage:**

```bash
python src/claude/distributional_shift_analysis.py \
  --tcga_labels data/dataset_csv/tcga_pam50_labels.csv \
  --cptac_labels data/dataset_csv/cptac_pam50_labels.csv \
  --metrics results/pam50_performance_metrics.csv \
  --task pam50 \
  --output results/distributional_shift/
```

**Input Files:**

`tcga_pam50_labels.csv`:
```csv
slide_id,label
TCGA-A1-001,LumA
TCGA-A1-002,LumB
...
```

`pam50_performance_metrics.csv`:
```csv
Class,F1_MCCV,F1_HO
LumA,0.75,0.68
LumB,0.65,0.58
...
```

---

### 4. Biological Interpretability Analysis

**Purpose:** Evaluate if models attend to morphologically meaningful patterns and quantify biological shift.

**Script:** `biological_interpretability_analysis.py`

**Statistical Tests:**
- Kruskal-Wallis (ordinal features)
- Mann-Whitney U (pairwise comparisons)
- Chi-square (binary features)

**Effect Sizes:**
- ε² (epsilon-squared) for Kruskal-Wallis
- r_rb (rank-biserial) for Mann-Whitney
- Cramér's V for Chi-square

**Outputs:**
- Intra-cohort analysis (TCGA, CPTAC separately)
- Inter-cohort comparison (TCGA vs CPTAC per class)
- Biological shift indicators (B_c)
- Effect size heatmaps

**Usage:**

```bash
python src/claude/biological_interpretability_analysis.py \
  --annotations data/histomorfologico/representative_images_annotation.xlsx \
  --task pam50 \
  --output results/biological_analysis/
```

**Excel Format:**

Two sheets: `TCGA` and `CPTAC`

Required columns:
- `ETIQUETA`: Class label (e.g., 'PAM50_LumA', 'ER-positive')
- `ESTRUCTURA GLANDULAR`: [0-4] Tubule formation
- `ATIPIA NUCLEAR`: [0-4] Nuclear pleomorphism
- `MITOSIS`: [0-4] Mitotic activity
- `NECROSIS`: [0-1] Tumor necrosis
- `INFILTRADO_LI`: [0-1] Lymphocytic infiltrate
- `INFILTRADO_PMN`: [0-1] Polymorphonuclear infiltrate

---

### 5. Integrative Domain Shift Analysis

**Purpose:** Multiple regression to identify which factors independently contribute to performance degradation.

**Script:** `integrative_domain_shift_analysis.py`

**Model:**
```
RPC_c = β₀ + β₁·Δp_c + β₂·d_c + β₃·ΔAcc_norm_c + β₄·B_c + ε
```

Where:
- `RPC_c`: Relative Performance Change for class c
- `Δp_c`: Prevalence shift (distributional shift)
- `d_c`: Centroid distance (feature space shift)
- `ΔAcc_norm_c`: Stain normalization benefit
- `B_c`: Biological shift indicator

**Outputs:**
- R², adjusted R²
- Regression coefficients with p-values
- Standardized coefficients (effect sizes)
- VIF (multicollinearity diagnostics)
- Model diagnostic plots

**Usage:**

```bash
python src/claude/integrative_domain_shift_analysis.py \
  --results_dir results/ \
  --task pam50 \
  --output results/integrative_analysis/
```

**Expected Directory Structure:**
```
results/
├── distributional_shift/
│   ├── pam50_prevalence_shift.csv
│   └── pam50_performance_degradation.csv
├── feature_space/
│   └── pam50_centroid_distances.csv
├── stain_analysis/
│   └── pam50_stain_normalization_results.csv
└── biological_analysis/
    └── pam50_biological_shift.csv
```

---

## 🔄 Complete Pipeline Workflow

### Step 1: Prepare Data

Ensure you have:
- TCGA and CPTAC labels CSVs
- Performance metrics (MCCV and Hold-Out)
- Extracted features with attention scores
- Histomorphological annotations Excel

### Step 2: Run Individual Analyses

```bash
# 1. Stain normalization
python src/claude/stain_normalization_from_accuracy.py \
  --input data/stain_analysis/pam50_accuracy.csv \
  --task pam50 \
  --output results/stain_analysis/

# 2. Feature space consistency
python src/claude/feature_space_consistency_analysis.py \
  --features_dir data/aggregated_features/ \
  --task pam50 \
  --top_k 8 \
  --output results/feature_space_analysis/

# 3. Distributional shift
python src/claude/distributional_shift_analysis.py \
  --tcga_labels data/dataset_csv/tcga_pam50_labels.csv \
  --cptac_labels data/dataset_csv/cptac_pam50_labels.csv \
  --metrics results/pam50_performance_metrics.csv \
  --task pam50 \
  --output results/distributional_shift/

# 4. Biological interpretability
python src/claude/biological_interpretability_analysis.py \
  --annotations data/histomorfologico/representative_images_annotation.xlsx \
  --task pam50 \
  --output results/biological_analysis/
```

### Step 3: Integrative Analysis

```bash
python src/claude/integrative_domain_shift_analysis.py \
  --results_dir results/ \
  --task pam50 \
  --output results/integrative_analysis/
```

### Step 4: Run for All Tasks

Repeat for each task:
- `pam50` - Molecular subtyping (5 classes)
- `er` - ER status (2 classes)
- `pr` - PR status (2 classes)
- `her2` - HER2 status (2 classes)

---

## 📊 Output Summary

Each analysis generates:

### CSV Files
- Detailed per-class metrics
- Statistical test results
- Effect sizes

### JSON Files
- Complete analysis results
- Metadata and parameters
- Structured for programmatic access

### Visualizations (PNG/PDF/SVG)
- Comparison plots
- Scatter plots (correlations)
- Heatmaps
- Diagnostic plots

---

## 🔬 Interpretation Guidelines

### Stain Normalization

**McNemar's test p < 0.05:**
- ✓ Staining variability contributes to domain shift
- → Color normalization should be applied

**McNemar's test p ≥ 0.05:**
- ✗ Staining is NOT a major contributor
- → Domain shift arises from other factors

### Feature Space Consistency

**Small centroid distances (< intra-class std):**
- ✓ Embeddings are consistent across cohorts
- → Feature-level domain shift does NOT explain failure

**Large centroid distances (> intra-class std):**
- ✗ Embeddings diverge between cohorts
- → Feature representations show domain shift

### Distributional Shift

**Pearson r ≈ 0, p > 0.05:**
- ✗ Prevalence shift does NOT explain degradation
- → Class imbalance is NOT the primary driver

**Pearson |r| > 0.5, p < 0.05:**
- ✓ Significant correlation detected
- → Distributional shift contributes to failure

### Biological Interpretability

**Small effect sizes (ε² < 0.06, V < 0.10):**
- ✗ No systematic morphological patterns
- → Internal performance may reflect overfitting

**Large effect sizes + significant p-values:**
- ✓ Model attends to biologically meaningful patterns
- → Morphological differences drive shift

### Integrative Analysis

**R² < 0.30:**
- Measured factors explain < 30% of variance
- → Unmeasured factors drive generalization failure
- → Fundamental limitations in H&E-based prediction

**R² > 0.30 with significant predictors:**
- Specific factors independently contribute
- → Targeted interventions may improve generalization

---

## 📦 Dependencies

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn torch openpyxl
```

---

## 📝 Citation

If you use these scripts, please cite:

```bibtex
@article{your_paper_2025,
  title={Foundation Model-Based Molecular Subtyping: Domain Shift Analysis},
  author={Your Name et al.},
  journal={Journal Name},
  year={2025}
}
```

---

## 🤝 Contributing

For issues or improvements, please open an issue at:
https://github.com/yourusername/CLAiMemAll/issues

---

## 📄 License

This code is provided for research purposes under the MIT License.

---

## ✨ Author

Generated with Claude Code by Anthropic
Date: 2025