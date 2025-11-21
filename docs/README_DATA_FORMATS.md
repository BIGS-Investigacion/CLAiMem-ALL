# Data Format Examples for Domain Shift Analysis

This directory contains example CSV files showing the exact format required for each analysis script.

## 📋 File Descriptions

### 1. Stain Normalization Analysis

**Files:**
- `pam50_accuracy.csv`
- `er_accuracy.csv`
- `pr_accuracy.csv`
- `her2_accuracy.csv`

**Format:**
```csv
Class,N_samples,Accuracy_Original,Accuracy_Normalized
LumA,150,0.750,0.780
```

**Columns:**
- `Class`: Class label (e.g., LumA, ER-positive)
- `N_samples`: Number of test samples for this class in CPTAC
- `Accuracy_Original`: Per-class accuracy without Macenko normalization
- `Accuracy_Normalized`: Per-class accuracy with Macenko normalization

**Notes:**
- For PAM50: 5 rows (LumA, LumB, Her2, Basal, Normal)
- For IHC: 2 rows (Receptor-negative, Receptor-positive)
- Accuracies should be in [0, 1] range (not percentages)

---

### 2. Distributional Shift Analysis

**Label Files:**
- `tcga_pam50_labels.csv`
- `cptac_pam50_labels.csv`

**Format:**
```csv
slide_id,label
TCGA-A1-A0SB-01Z-00-DX1,LumA
```

**Columns:**
- `slide_id`: Unique slide identifier
- `label`: Class label matching your task

**Performance Metrics File:**
- `pam50_performance_metrics.csv`
- `er_performance_metrics.csv`

**Format (PAM50):**
```csv
Class,F1_MCCV,F1_HO,Precision_MCCV,Precision_HO,Recall_MCCV,Recall_HO
LumA,0.7500,0.6800,0.7800,0.7200,0.7200,0.6500
```

**Format (IHC - ER/PR/HER2):**
```csv
Class,PR_AUC_MCCV,PR_AUC_HO,F1_MCCV,F1_HO,Precision_MCCV,Precision_HO,Recall_MCCV,Recall_HO
ER-negative,0.8200,0.7800,0.8000,0.7600,0.8300,0.7900,0.7700,0.7300
```

**Columns:**
- `Class`: Class label
- `*_MCCV`: Metric from internal validation (Monte Carlo Cross-Validation)
- `*_HO`: Metric from external validation (Hold-Out on CPTAC)
- For PAM50: Use `F1` scores
- For IHC: Use `PR_AUC` (Precision-Recall AUC) as primary metric

**Notes:**
- The script automatically computes RPC (Relative Performance Change)
- RPC = (Metric_HO - Metric_MCCV) / Metric_MCCV

---

### 3. Feature Space Consistency Analysis

This analysis requires PyTorch `.pt` files, not CSV. Expected structure:

```
data/aggregated_features/
├── tcga/
│   ├── TCGA-A1-001_features.pt     # Tensor [N_patches, D]
│   ├── TCGA-A1-001_attention.pt    # Tensor [N_patches]
│   └── labels.csv                  # slide_id,label
└── cptac/
    ├── CPTAC-01_features.pt
    ├── CPTAC-01_attention.pt
    └── labels.csv
```

**labels.csv format:**
```csv
slide_id,label
TCGA-A1-001,LumA
```

**Feature file format (.pt):**
- Tensor shape: `[N_patches, embedding_dim]`
- Example: `[512, 1280]` for Virchow v2

**Attention file format (.pt):**
- Tensor shape: `[N_patches]`
- Contains attention scores from CLAM model

---

### 4. Biological Interpretability Analysis

This analysis requires an Excel file with histomorphological annotations.

**File:** Excel with two sheets: `TCGA` and `CPTAC`

**Columns (both sheets):**
- `ETIQUETA`: Class label (e.g., 'PAM50_LumA', 'ER-positive')
- `ESTRUCTURA GLANDULAR`: [0-4] Tubule formation
- `ATIPIA NUCLEAR`: [0-4] Nuclear pleomorphism
- `MITOSIS`: [0-4] Mitotic activity
- `NECROSIS`: [0-1] Tumor necrosis presence
- `INFILTRADO_LI`: [0-1] Lymphocytic infiltrate presence
- `INFILTRADO_PMN`: [0-1] Polymorphonuclear infiltrate presence

**Example Excel structure:**

Sheet: TCGA
```
ETIQUETA         | ESTRUCTURA GLANDULAR | ATIPIA NUCLEAR | MITOSIS | NECROSIS | INFILTRADO_LI | INFILTRADO_PMN
PAM50_LumA       | 3                    | 2              | 1       | 0        | 1             | 0
PAM50_LumA       | 4                    | 1              | 0       | 0        | 0             | 0
PAM50_LumB       | 2                    | 3              | 2       | 1        | 1             | 0
```

Sheet: CPTAC
```
ETIQUETA         | ESTRUCTURA GLANDULAR | ATIPIA NUCLEAR | MITOSIS | NECROSIS | INFILTRADO_LI | INFILTRADO_PMN
PAM50_LumA       | 3                    | 2              | 1       | 0        | 1             | 0
PAM50_LumA       | 4                    | 2              | 1       | 0        | 0             | 0
PAM50_LumB       | 2                    | 3              | 3       | 1        | 1             | 1
```

**Notes:**
- Each row represents one annotated patch
- Typically 25 patches per class (125 for PAM50, 50 for IHC)
- Annotations should be made by a board-certified pathologist
- Label format: `PAM50_<subtype>` or `<Receptor>-<status>`

---

### 5. Integrative Analysis

This analysis loads results from all previous analyses. No direct input CSV needed.

**Required directory structure:**
```
results/
├── distributional_shift/
│   ├── pam50_prevalence_shift.csv
│   └── pam50_performance_degradation.csv
├── feature_space_analysis/
│   └── pam50_centroid_distances.csv
├── stain_analysis/
│   └── pam50_stain_normalization_results.csv
└── biological_analysis/
    └── pam50_biological_shift.csv
```

---

## 🔧 How to Prepare Your Data

### Step 1: Extract Performance Metrics from Results

If you have CLAM results:

```python
import pickle
import pandas as pd

# Load results
with open('results/pam50/fold_0.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract metrics
metrics = []
for class_idx, class_name in enumerate(['LumA', 'LumB', 'Her2', 'Basal', 'Normal']):
    # Compute per-class metrics from results
    metrics.append({
        'Class': class_name,
        'F1_MCCV': results['f1'][class_idx],
        'F1_HO': results['f1_ho'][class_idx],
        # ... add other metrics
    })

df = pd.DataFrame(metrics)
df.to_csv('pam50_performance_metrics.csv', index=False)
```

### Step 2: Create Label CSVs

From your dataset splits:

```python
import pandas as pd

# TCGA labels
df_tcga = pd.read_csv('dataset_csv/tcga_pam50.csv')
df_tcga[['slide_id', 'label']].to_csv('tcga_pam50_labels.csv', index=False)

# CPTAC labels
df_cptac = pd.read_csv('dataset_csv/cptac_pam50.csv')
df_cptac[['slide_id', 'label']].to_csv('cptac_pam50_labels.csv', index=False)
```

### Step 3: Compute Accuracies with/without Normalization

```python
from sklearn.metrics import accuracy_score

# Original model results
y_true = [...]  # Ground truth labels
y_pred_orig = [...]  # Predictions from original model

# Normalized model results
y_pred_norm = [...]  # Predictions from Macenko-normalized model

# Per-class accuracy
for class_idx, class_name in enumerate(['LumA', 'LumB', 'Her2', 'Basal', 'Normal']):
    mask = (y_true == class_idx)
    acc_orig = accuracy_score(y_true[mask], y_pred_orig[mask])
    acc_norm = accuracy_score(y_true[mask], y_pred_norm[mask])

    print(f"{class_name},{mask.sum()},{acc_orig:.4f},{acc_norm:.4f}")
```

---

## 📊 Quick Validation

To validate your CSV files:

```python
import pandas as pd

# Stain normalization
df = pd.read_csv('pam50_accuracy.csv')
assert set(df.columns) == {'Class', 'N_samples', 'Accuracy_Original', 'Accuracy_Normalized'}
assert df['Accuracy_Original'].between(0, 1).all()
assert df['Accuracy_Normalized'].between(0, 1).all()
print("✓ Stain normalization CSV is valid")

# Labels
df = pd.read_csv('tcga_pam50_labels.csv')
assert set(df.columns) == {'slide_id', 'label'}
assert df['slide_id'].nunique() == len(df)  # No duplicates
print("✓ Labels CSV is valid")

# Performance metrics
df = pd.read_csv('pam50_performance_metrics.csv')
assert 'Class' in df.columns
assert 'F1_MCCV' in df.columns
assert 'F1_HO' in df.columns
print("✓ Performance metrics CSV is valid")
```

---

## 🚀 Next Steps

1. Replace example data with your actual results
2. Validate CSV formats using the code above
3. Run each analysis script
4. Check output files in `results/` directories

---

## ❓ Need Help?

Common issues:
- **Wrong column names**: Check exact spelling and capitalization
- **Wrong data types**: Accuracies should be floats (0.75), not percentages (75)
- **Missing classes**: Ensure all molecular classes are present
- **Duplicate slide_ids**: Each slide should appear only once

For more help, check the main README: `src/claude/README_DOMAIN_SHIFT_ANALYSIS.md`
