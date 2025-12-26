
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("DOMAIN SHIFT MULTIVARIATE ANALYSIS")
print("="*80)

# ============================================================================
# CONFIGURATION: OUTLIER REMOVAL AND SCALING
# ============================================================================

# Set to True to remove outliers using IQR method
REMOVE_OUTLIERS = False  # Change to True to enable outlier removal
IQR_MULTIPLIER = 3.0  # Standard IQR multiplier (1.5 for mild outliers, 3.0 for extreme)

# Set to True to scale predictors to [0, 1] range (Min-Max scaling)
SCALE_PREDICTORS = False  # Change to True to enable scaling
# Note: Scaling is applied AFTER outlier removal (if enabled)

# ============================================================================
# CONFIGURATION: EFFECT SIZE ADJUSTMENT
# ============================================================================

# Set to True to subtract the original TCGA within-cohort effect size from B_c
# This shows the CHANGE in morphological separability (cross-cohort vs within-cohort)
SUBTRACT_TCGA_EFFECT_SIZE = False  # Change to True to enable adjustment

# ============================================================================
# CONFIGURATION: CLASS EXCLUSION
# ============================================================================

# List of class names to exclude from multivariate regression analysis
# Examples: ['HER2-enriched', 'Normal-like'], ['ER-negative'], etc.
# Set to empty list [] to include all classes
EXCLUDE_CLASSES = []  # Change to exclude specific classes from regression

print(f"\nOutlier removal: {'ENABLED' if REMOVE_OUTLIERS else 'DISABLED'}")
if REMOVE_OUTLIERS:
    print(f"IQR multiplier: {IQR_MULTIPLIER}")

print(f"Predictor scaling: {'ENABLED' if SCALE_PREDICTORS else 'DISABLED'}")
if SCALE_PREDICTORS:
    print("  Predictors will be scaled to [0, 1] range using Min-Max scaling")

print(f"Effect size adjustment: {'ENABLED' if SUBTRACT_TCGA_EFFECT_SIZE else 'DISABLED'}")
if SUBTRACT_TCGA_EFFECT_SIZE:
    print("  B_c will be adjusted by subtracting TCGA transition baseline effect")

print(f"Class exclusion: {'ENABLED' if EXCLUDE_CLASSES else 'DISABLED'}")
if EXCLUDE_CLASSES:
    print(f"  Excluded classes: {', '.join(EXCLUDE_CLASSES)}")

# ============================================================================
# STEP 1: CALCULATE B_c (Morphological Separability)
# ============================================================================

print("\n" + "="*80)
print("STEP 1: CALCULATING B_c = Diagonal - Min(Other)")
print("="*80)

# Cross-cohort effect size matrices (TCGA vs CPTAC)
er_matrix = np.array([
    [0.33, 2.70],  # ER-negative row
    [1.98, 0.00]   # ER-positive row
])

pr_matrix = np.array([
    [0.00, 3.20],  # PR-negative row
    [1.12, 1.08]   # PR-positive row
])

her2_matrix = np.array([
    [0.00, 0.50],  # HER2-negative row
    [0.77, 0.00]   # HER2-positive row
])

pam50_matrix = np.array([
    [0.32, 0.56, 3.31, 3.19, 2.72],  # Her2-enriched row
    [0.32, 0.24, 2.03, 2.44, 2.43],  # Basal row
    [3.83, 3.71, 0.00, 0.00, 0.48],  # LumA row
    [3.03, 2.93, 0.00, 0.00, 1.32],  # LumB row
    [1.32, 0.59, 2.07, 1.37, 2.09]   # Normal-like row
])




def calculate_Bc(matrix, row_idx):
    """
    B_c = Diagonal - Min(other values in row)
    Positive: Greater within-cohort consistency than cross-cohort similarity
    Negative: More similar to other classes cross-cohort than within-cohort
    """
    row = matrix[row_idx, :]
    diagonal_value = row[row_idx]
    other_values = np.delete(row, row_idx)
    min_other = np.min(other_values)
    B_c = min_other - diagonal_value 
    return diagonal_value, min_other, B_c

# Calculate B_c for all classes
B_c_results = []

# PAM50
pam50_classes = ['Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B', 'Normal-like']
pam50_order = [1, 0, 2, 3, 4]  # Reorder to match RPC array
for idx, class_name in zip(pam50_order, ['Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B', 'Normal-like']):
    diag, min_oth, bc = calculate_Bc(pam50_matrix, idx)
    B_c_results.append({'Class': class_name, 'Diagonal': diag, 'Min_Other': min_oth, 'B_c': bc})
    print(f"{class_name:20} diag={diag:.2f}, min_other={min_oth:.2f}, B_c={bc:.2f}")

# ER
for i, class_name in enumerate(['ER-negative', 'ER-positive']):
    diag, min_oth, bc = calculate_Bc(er_matrix, i)
    B_c_results.append({'Class': class_name, 'Diagonal': diag, 'Min_Other': min_oth, 'B_c': bc})
    print(f"{class_name:20} diag={diag:.2f}, min_other={min_oth:.2f}, B_c={bc:.2f}")

# PR
for i, class_name in enumerate(['PR-negative', 'PR-positive']):
    diag, min_oth, bc = calculate_Bc(pr_matrix, i)
    B_c_results.append({'Class': class_name, 'Diagonal': diag, 'Min_Other': min_oth, 'B_c': bc})
    print(f"{class_name:20} diag={diag:.2f}, min_other={min_oth:.2f}, B_c={bc:.2f}")

# HER2
for i, class_name in enumerate(['HER2-negative', 'HER2-positive']):
    diag, min_oth, bc = calculate_Bc(her2_matrix, i)
    B_c_results.append({'Class': class_name, 'Diagonal': diag, 'Min_Other': min_oth, 'B_c': bc})
    print(f"{class_name:20} diag={diag:.2f}, min_other={min_oth:.2f}, B_c={bc:.2f}")

# ============================================================================
# TCGA WITHIN-COHORT MATRICES (FOR OPTIONAL B_c ADJUSTMENT)
# ============================================================================

# TCGA intra-cohort cumulative effect size matrices
# These represent the morphological differences WITHIN TCGA
# Used to adjust B_c when SUBTRACT_TCGA_EFFECT_SIZE = True
# Rows = source class, Columns = target class
# B_c_adjusted = (Diagonal_cross - Min(other)_cross) - (Diagonal_TCGA - TransitionValue_TCGA)

# PAM50 TCGA matrix (5x5: Basal, Her2-enriched, LumA, LumB, Normal)
# Extracted from TCGA within-cohort pairwise tests (cumulative effect sizes)
tcga_pam50_matrix = np.array([
    [0.00, 0.68, 2.13, 1.88, 2.41],  # Basal row (index 0)
    [0.68, 0.00, 2.14, 1.88, 2.36],  # Her2-enriched row (index 1)
    [2.13, 2.14, 0.00, 0.36, 0.96],  # LumA row (index 2)
    [1.88, 1.88, 0.36, 0.00, 1.25],  # LumB row (index 3)
    [2.41, 2.36, 0.96, 1.25, 0.00],  # Normal row (index 4)
])

# ER TCGA matrix (2x2: ER-, ER+)
tcga_er_matrix = np.array([
    [0.00, 1.71],  # ER-negative row
    [1.71, 0.00],  # ER-positive row
])

# PR TCGA matrix (2x2: PR-, PR+)
tcga_pr_matrix = np.array([
    [0.00, 1.86],  # PR-negative row
    [1.86, 0.00],  # PR-positive row
])

# HER2 TCGA matrix (2x2: HER2-, HER2+)
tcga_her2_matrix = np.array([
    [0.00, 0.45],  # HER2-negative row
    [0.45, 0.00],  # HER2-positive row
])

# ============================================================================
# STEP 2: PREPARE FULL DATASET
# ============================================================================

print("\n" + "="*80)
print("STEP 2: PREPARE FULL DATASET")
print("="*80)

data = {
    'Class': ['Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B', 'Normal-like',
              'ER-negative', 'ER-positive', 'PR-negative', 'PR-positive',
              'HER2-negative', 'HER2-positive'],
    'NPG': [0.000, 0.000, -0.761, 0.055, 0.000, 0.020, -0.433, -0.458, 0.438, 0.000, 0.000],
    'd_c': [24.88, 22.22, 16.66, 23.38, 26.31, 18.23, 17.38, 23.02, 19.78, 18.38, 20.92],
    'delta_p': [0.131, 0.023, 0.071, 0.085, 0.001, 0.159, 0.159, 0.127, 0.127, 0.049, 0.049],
    'B_c': [r['B_c'] for r in B_c_results],
    'RPC': [-0.064, -1.000, -0.078, -0.517, -1.000, 0.037, -0.441, 0.198, -0.097, 0.037, -0.441],
    'Task': ['PAM50', 'PAM50', 'PAM50', 'PAM50', 'PAM50',
             'ER', 'ER', 'PR', 'PR', 'HER2', 'HER2']
}

df = pd.DataFrame(data)

# Apply class exclusion if specified
if EXCLUDE_CLASSES:
    print("\n" + "="*80)
    print("EXCLUDING CLASSES FROM REGRESSION ANALYSIS")
    print("="*80)
    n_before = len(df)
    df = df[~df['Class'].isin(EXCLUDE_CLASSES)].copy()
    n_after = len(df)
    print(f"Excluded {n_before - n_after} classes: {', '.join(EXCLUDE_CLASSES)}")
    print(f"Remaining classes for regression: {n_after}")
    print("\nRemaining dataset:")
    print(df[['Class', 'Task', 'RPC', 'B_c', 'NPG', 'd_c', 'delta_p']].to_string(index=False))

# Apply adjustment if enabled
if SUBTRACT_TCGA_EFFECT_SIZE:
    print("\n" + "="*80)
    print("APPLYING B_c ADJUSTMENT (Subtracting TCGA transition effect)")
    print("="*80)
    print("\nFor each class C in cross-cohort matrix:")
    print("  Find the minimum cross-cohort transition value (C→C' with min effect size)")
    print("  Subtract the corresponding TCGA transition value (C_TCGA→C'_TCGA)")
    print("Formula: B_c_adjusted = B_c_original - (Diag_TCGA - Transition_TCGA)")

    # Get components from B_c_results
    diagonal_dict = {r['Class']: r['Diagonal'] for r in B_c_results}
    min_other_dict = {r['Class']: r['Min_Other'] for r in B_c_results}

    # Class to matrix mapping
    class_to_matrix = {
        'Basal-like': (tcga_pam50_matrix, 0),
        'HER2-enriched': (tcga_pam50_matrix, 1),
        'Luminal A': (tcga_pam50_matrix, 2),
        'Luminal B': (tcga_pam50_matrix, 3),
        'Normal-like': (tcga_pam50_matrix, 4),
        'ER-negative': (tcga_er_matrix, 0),
        'ER-positive': (tcga_er_matrix, 1),
        'PR-negative': (tcga_pr_matrix, 0),
        'PR-positive': (tcga_pr_matrix, 1),
        'HER2-negative': (tcga_her2_matrix, 0),
        'HER2-positive': (tcga_her2_matrix, 1)
    }

    print(f"\n{'Class':<20} {'B_c_orig':>10} {'Diag_TCGA':>10} {'Min_TCGA':>10} {'Adjustment':>12} {'B_c_adj':>10}")
    print("-"*90)

    for idx, row in df.iterrows():
        class_name = row['Class']
        b_c_original = row['B_c']

        # Get TCGA matrix and row index for this class
        tcga_matrix, row_idx = class_to_matrix[class_name]
        tcga_row = tcga_matrix[row_idx, :]

        # Get diagonal value for TCGA
        diag_tcga = tcga_row[row_idx]

        # Get minimum of other values in TCGA (same logic as B_c calculation)
        other_values_tcga = np.delete(tcga_row, row_idx)
        min_other_tcga = np.min(other_values_tcga)

        # Calculate the TCGA baseline effect (what we need to subtract)
        # This is the difference in TCGA between diagonal and minimum transition
        tcga_baseline = diag_tcga - min_other_tcga

        # Adjust B_c by subtracting the TCGA baseline
        b_c_adjusted = b_c_original - tcga_baseline

        print(f"{class_name:<20} {b_c_original:>10.2f} {diag_tcga:>10.2f} {min_other_tcga:>10.2f} {tcga_baseline:>12.2f} {b_c_adjusted:>10.2f}")

        # Update the value in the dataframe
        df.at[idx, 'B_c'] = b_c_adjusted

    print("\n✓ B_c has been adjusted by subtracting TCGA baseline transition effect")
    print("  Interpretation:")
    print("    - B_c_adjusted = B_c_cross-cohort - B_c_TCGA")
    print("    - Positive values = Cross-cohort separability BETTER than TCGA baseline")
    print("    - Negative values = Cross-cohort separability WORSE than TCGA baseline")
    print("    - Zero values = Cross-cohort separability SAME as TCGA baseline")

print("\nDataset:")
print(df.to_string(index=False))

# ============================================================================
# STEP 2.5: OUTLIER DETECTION AND REMOVAL (OPTIONAL)
# ============================================================================

if REMOVE_OUTLIERS:
    print("\n" + "="*80)
    print("STEP 2.5: OUTLIER DETECTION AND REMOVAL (IQR METHOD)")
    print("="*80)
    print("Detection based on DEPENDENT VARIABLE (RPC) only")

    def detect_outliers_iqr(data, multiplier=1.5):
        """Detect outliers using IQR method."""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers, lower_bound, upper_bound

    # Check only the dependent variable (RPC)
    original_n = len(df)

    rpc_data = df['RPC'].values
    outliers, lower, upper = detect_outliers_iqr(rpc_data, IQR_MULTIPLIER)
    Q1 = np.percentile(rpc_data, 25)
    Q3 = np.percentile(rpc_data, 75)
    IQR = Q3 - Q1
    n_outliers = np.sum(outliers)

    print("\nOutlier detection for RPC (dependent variable):")
    print(f"{'Variable':<12} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'Lower':>8} {'Upper':>8} {'Outliers':>10}")
    print("-"*80)
    print(f"{'RPC':<12} {Q1:>8.3f} {Q3:>8.3f} {IQR:>8.3f} {lower:>8.3f} {upper:>8.3f} {n_outliers:>10}")

    if n_outliers > 0:
        print(f"\n{'='*80}")
        print(f"Total observations with extreme RPC values: {n_outliers}")
        print(f"\nClasses with outliers (extreme values in RPC):")
        for idx in np.where(outliers)[0]:
            rpc_val = df.iloc[idx]['RPC']
            class_name = df.iloc[idx]['Class']
            outlier_type = "LOW" if rpc_val < lower else "HIGH"
            print(f"  - {class_name}: RPC = {rpc_val:.3f} ({outlier_type})")

        df_clean = df[~outliers].copy()
        print(f"\nDataset size: {original_n} → {len(df_clean)} (removed {n_outliers})")
        print(f"\nCleaned dataset:")
        print(df_clean.to_string(index=False))

        # Use cleaned dataset
        df = df_clean
    else:
        print(f"\nNo outliers detected in RPC. Using full dataset (n={original_n}).")

# ============================================================================
# STEP 2.6: PREDICTOR SCALING (OPTIONAL)
# ============================================================================

if SCALE_PREDICTORS:
    print("\n" + "="*80)
    print("STEP 2.6: PREDICTOR SCALING (Min-Max [0, 1])")
    print("="*80)
    
    predictors_to_scale = ['NPG', 'd_c', 'delta_p', 'B_c']
    
    print(f"{'Predictor':<15} {'Original Min':>12} {'Original Max':>12}")
    print("-" * 50)
    
    for pred in predictors_to_scale:
        min_val = df[pred].min()
        max_val = df[pred].max()
        print(f"{pred:<15} {min_val:>12.4f} {max_val:>12.4f}")
        
        # Apply Min-Max scaling
        if max_val > min_val:
            df[pred] = (df[pred] - min_val) / (max_val - min_val)
        else:
            print(f"Warning: {pred} has constant value {min_val}, cannot scale.")
            df[pred] = 0.0
        
    print("\n✓ Predictors scaled to [0, 1] range")
    print("\nScaled dataset:")
    print(df.to_string(index=False))


# ============================================================================
# STEP 3: UNIVARIATE ANALYSES
# ============================================================================

print("\n" + "="*80)
print("STEP 3: UNIVARIATE ANALYSES")
print("="*80)

X = df[['NPG', 'd_c', 'delta_p', 'B_c']].values
y = df['RPC'].values
n = len(y)

predictors = ['NPG', 'd_c', 'delta_p', 'B_c']
pred_labels = ['NPG', 'd_c', 'Δp_c', 'B_c']

univariate_results = []

for pred, label in zip(predictors, pred_labels):
    X_uni = df[[pred]].values
    r, p_corr = stats.pearsonr(X_uni.flatten(), y)
    
    model_uni = LinearRegression()
    model_uni.fit(X_uni, y)
    r2 = model_uni.score(X_uni, y)
    
    f_stat = (r2 / 1) / ((1 - r2) / (n - 2))
    f_pval = 1 - stats.f.cdf(f_stat, 1, n - 2)
    
    univariate_results.append({
        'Predictor': label,
        'r': r,
        'R²': r2,
        'β': model_uni.coef_[0],
        'p': f_pval
    })

print(f"\n{'Predictor':<10} {'r':>8} {'R²':>8} {'β':>10} {'p-value':>10} {'Sig':>6}")
print("-"*80)

for res in sorted(univariate_results, key=lambda x: x['p']):
    sig = ""
    if res['p'] < 0.001:
        sig = "***"
    elif res['p'] < 0.01:
        sig = "**"
    elif res['p'] < 0.05:
        sig = "*"
    elif res['p'] < 0.10:
        sig = "†"
    
    print(f"{res['Predictor']:<10} {res['r']:>8.3f} {res['R²']:>8.3f} {res['β']:>10.4f} {res['p']:>10.4f} {sig:>6}")

# ============================================================================
# STEP 4: MULTIVARIATE REGRESSION
# ============================================================================

print("\n" + "="*80)
print("STEP 4: MULTIVARIATE REGRESSION (All 4 predictors)")
print("="*80)

k = X.shape[1]
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
residuals = y - y_pred

ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

mse = ss_res / (n - k - 1)
se = np.sqrt(mse)

# Coefficient statistics
X_with_intercept = np.column_stack([np.ones(n), X])
XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
se_coeffs = np.sqrt(np.diagonal(XtX_inv) * mse)

coeffs_with_intercept = np.concatenate([[model.intercept_], model.coef_])
t_stats = coeffs_with_intercept / se_coeffs
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

# F-statistic
f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
f_pval = 1 - stats.f.cdf(f_stat, k, n - k - 1)

print(f"\nR² = {r_squared:.4f}")
print(f"Adjusted R² = {adj_r_squared:.4f}")
print(f"F-statistic = {f_stat:.4f} (p = {f_pval:.4f})")
print(f"Residual SE = {se:.4f}")

if f_pval < 0.05:
    print("✓ MODEL IS SIGNIFICANT")
else:
    print("✗ Model is NOT significant")

print(f"\n{'Variable':<15} {'Coefficient':>12} {'Std Error':>12} {'t-value':>10} {'p-value':>10} {'Sig':>6}")
print("-"*80)

var_names = ['(Intercept)', 'NPG', 'd_c', 'Δp_c', 'B_c']
for i, var in enumerate(var_names):
    coef = coeffs_with_intercept[i]
    se_coef = se_coeffs[i]
    t = t_stats[i]
    p = p_values[i]
    
    sig = ""
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    
    print(f"{var:<15} {coef:>12.4f} {se_coef:>12.4f} {t:>10.3f} {p:>10.4f} {sig:>6}")

# ============================================================================
# STEP 5: BIVARIATE MODELS (All combinations of 2 predictors)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: BIVARIATE MODELS (All combinations of 2 predictors)")
print("="*80)

print(f"\n{'Model':<25} {'R²':>8} {'Adj R²':>10} {'F':>8} {'p(F)':>10} {'Sig':>6}")
print("-"*80)

from itertools import combinations

# Generate all combinations of 2 predictors
predictor_combos_2 = list(combinations(range(4), 2))

for combo in predictor_combos_2:
    # Get predictor indices
    pred_indices = list(combo)
    pred_names = [pred_labels[i] for i in pred_indices]
    pred_cols = [predictors[i] for i in pred_indices]

    # Fit model
    X_bi = df[pred_cols].values
    model_bi = LinearRegression()
    model_bi.fit(X_bi, y)
    y_pred_bi = model_bi.predict(X_bi)

    # Calculate statistics
    ss_res_bi = np.sum((y - y_pred_bi)**2)
    ss_tot_bi = np.sum((y - np.mean(y))**2)
    r2_bi = 1 - (ss_res_bi / ss_tot_bi)
    adj_r2_bi = 1 - (1 - r2_bi) * (n - 1) / (n - 3)

    f_bi = (r2_bi / 2) / ((1 - r2_bi) / (n - 3))
    p_f_bi = 1 - stats.f.cdf(f_bi, 2, n - 3)

    sig = ""
    if p_f_bi < 0.001:
        sig = "***"
    elif p_f_bi < 0.01:
        sig = "**"
    elif p_f_bi < 0.05:
        sig = "*"

    model_name = " + ".join(pred_names)
    print(f"{model_name:<25} {r2_bi:>8.3f} {adj_r2_bi:>10.3f} {f_bi:>8.3f} {p_f_bi:>10.4f} {sig:>6}")

    # Print coefficients for this bivariate model
    print(f"  Coefficients: Intercept={model_bi.intercept_:.4f}", end="")
    for pred_name, coef in zip(pred_names, model_bi.coef_):
        print(f", {pred_name}={coef:.4f}", end="")
    print()  # New line

# ============================================================================
# STEP 6: TRIVARIATE MODELS (3 predictors)
# ============================================================================

print("\n" + "="*80)
print("STEP 6: TRIVARIATE MODELS (3 predictors)")
print("="*80)

print(f"\n{'Model':<30} {'R²':>8} {'Adj R²':>10} {'F':>8} {'p(F)':>10} {'Sig':>6}")
print("-"*80)

from itertools import combinations

# Generate all combinations of 3 predictors
predictor_combos_3 = list(combinations(range(4), 3))

for combo in predictor_combos_3:
    # Get predictor indices
    pred_indices = list(combo)
    pred_names = [pred_labels[i] for i in pred_indices]
    pred_cols = [predictors[i] for i in pred_indices]

    # Fit model
    X_tri = df[pred_cols].values
    model_tri = LinearRegression()
    model_tri.fit(X_tri, y)
    y_pred_tri = model_tri.predict(X_tri)

    # Calculate statistics
    ss_res_tri = np.sum((y - y_pred_tri)**2)
    ss_tot_tri = np.sum((y - np.mean(y))**2)
    r2_tri = 1 - (ss_res_tri / ss_tot_tri)
    adj_r2_tri = 1 - (1 - r2_tri) * (n - 1) / (n - 4)

    f_tri = (r2_tri / 3) / ((1 - r2_tri) / (n - 4))
    p_f_tri = 1 - stats.f.cdf(f_tri, 3, n - 4)

    sig = ""
    if p_f_tri < 0.001:
        sig = "***"
    elif p_f_tri < 0.01:
        sig = "**"
    elif p_f_tri < 0.05:
        sig = "*"

    model_name = " + ".join(pred_names)
    print(f"{model_name:<30} {r2_tri:>8.3f} {adj_r2_tri:>10.3f} {f_tri:>8.3f} {p_f_tri:>10.4f} {sig:>6}")

    # Print coefficients for this trivariate model
    print(f"  Coefficients: Intercept={model_tri.intercept_:.4f}", end="")
    for pred_name, coef in zip(pred_names, model_tri.coef_):
        print(f", {pred_name}={coef:.4f}", end="")
    print()  # New line

# ============================================================================
# STEP 7: SUMMARY AND INTERPRETATION
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n** UNIVARIATE RESULTS **")
print("Δp_c (prevalence shift): ONLY significant predictor (p = 0.029)")
print("  → Greater distributional differences → Worse RPC")
print("  → Explains 42.6% of variance")
print("\nB_c (morphological separability): Marginally significant (p = 0.065)")
print("  → Greater within-cohort stability → Better RPC")
print("  → Explains 33.0% of variance")
print("\nd_c and NPG: NOT significant")

print("\n** MULTIVARIATE MODEL **")
print("Full model (4 predictors): NOT significant (p = 0.348)")
print("  → Limited sample size (n=11) with 4 predictors")
print("  → Model is overfitted")

print("\n** KEY FINDING **")
print("Prevalence shift (Δp_c) is the primary driver of domain shift.")
print("Biological morphological heterogeneity (B_c) shows marginal association")
print("but does not significantly predict performance degradation.")

print("\n" + "="*80)
print("INTERPRETATION OF B_c VALUES")
print("="*80)
print("\nNormal-like (B_c = -1.50):")
print("  → MORE similar to other classes cross-cohort than to itself")
print("  → Severe morphological instability → Complete failure (RPC = -1.0)")
print("\nPR-negative (B_c = 3.20):")
print("  → HIGH within-cohort stability")
print("  → Good generalization (RPC = 0.198)")
print("\nLuminal A/B (B_c = 0.00):")
print("  → Equal within/cross-cohort similarity")
print("  → Mixed generalization")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 8: GENERATING REGRESSION VISUALIZATIONS")
print("="*80)

from pathlib import Path

# Create output directory
OUTPUT_DIR = Path('results/regression_comparison/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define colors for each task (magma palette)
task_colors = {
    'PAM50': '#FCFDBF',  # Light yellow (magma top)
    'ER': '#FE9F6D',     # Orange-pink (magma mid-high)
    'PR': '#DE4968',     # Pink-red (magma mid)
    'HER2': '#711F81'    # Purple (magma mid-low)
}

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("\nCreating univariate regression plots...")

# Create 2x2 subplot for the 4 univariate regressions
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (pred, label) in enumerate(zip(predictors, pred_labels)):
    ax = axes[idx]

    # Get predictor values
    X_uni = df[pred].values
    y_vals = df['RPC'].values

    # Fit regression
    model_uni = LinearRegression()
    model_uni.fit(X_uni.reshape(-1, 1), y_vals)

    # Generate prediction line
    X_range = np.linspace(X_uni.min(), X_uni.max(), 100)
    y_pred_line = model_uni.predict(X_range.reshape(-1, 1))

    # Plot points by task
    for task in ['PAM50', 'ER', 'PR', 'HER2']:
        mask = df['Task'] == task
        if mask.any():
            ax.scatter(
                df.loc[mask, pred],
                df.loc[mask, 'RPC'],
                color=task_colors[task],
                label=task,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )

    # Plot regression line
    ax.plot(X_range, y_pred_line, 'k--', linewidth=2, alpha=0.8, label='Regression')

    # Get statistics
    r2 = model_uni.score(X_uni.reshape(-1, 1), y_vals)
    r, p_corr = stats.pearsonr(X_uni, y_vals)

    # Determine significance
    sig = ""
    if p_corr < 0.001:
        sig = "***"
    elif p_corr < 0.01:
        sig = "**"
    elif p_corr < 0.05:
        sig = "*"
    elif p_corr < 0.10:
        sig = "†"

    # Add statistics to plot
    textstr = f'R² = {r2:.3f}\nr = {r:.3f}\np = {p_corr:.4f} {sig}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Add panel label (A, B, C, D)
    panel_label = chr(65 + idx)  # 65 is ASCII for 'A'
    ax.text(0.5, -0.18, f"({panel_label})", transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='center')

    # Formatting
    ax.set_xlabel(label, fontsize=12, fontweight='bold')
    ax.set_ylabel('RPC', fontsize=12, fontweight='bold')
    ax.set_title(f'RPC vs {label}', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
output_path = OUTPUT_DIR / 'univariate_regressions.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# Create bivariate regression visualizations (selected top models)
print("\nCreating bivariate regression plots...")

# Select top 3 bivariate models by R²
bivariate_models = []
for combo in predictor_combos_2:
    pred_indices = list(combo)
    pred_cols = [predictors[i] for i in pred_indices]
    pred_names = [pred_labels[i] for i in pred_indices]

    X_bi = df[pred_cols].values
    model_bi = LinearRegression()
    model_bi.fit(X_bi, y)
    r2_bi = model_bi.score(X_bi, y)

    bivariate_models.append({
        'predictors': pred_names,
        'pred_cols': pred_cols,
        'r2': r2_bi,
        'model': model_bi
    })

# Sort by R² and select top 3
bivariate_models = sorted(bivariate_models, key=lambda x: x['r2'], reverse=True)[:3]

# Create 3D scatter plots for top 3 bivariate models
fig = plt.figure(figsize=(18, 6))

for idx, model_info in enumerate(bivariate_models):
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

    pred_names = model_info['predictors']
    pred_cols = model_info['pred_cols']
    model_bi = model_info['model']
    r2_bi = model_info['r2']

    # Plot points by task
    for task in ['PAM50', 'ER', 'PR', 'HER2']:
        mask = df['Task'] == task
        if mask.any():
            ax.scatter(
                df.loc[mask, pred_cols[0]],
                df.loc[mask, pred_cols[1]],
                df.loc[mask, 'RPC'],
                color=task_colors[task],
                label=task,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )

    # Create mesh for regression plane
    x1_range = np.linspace(df[pred_cols[0]].min(), df[pred_cols[0]].max(), 20)
    x2_range = np.linspace(df[pred_cols[1]].min(), df[pred_cols[1]].max(), 20)
    X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)

    # Predict on mesh
    X_mesh = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])
    Y_mesh = model_bi.predict(X_mesh).reshape(X1_mesh.shape)

    # Plot regression plane
    ax.plot_surface(X1_mesh, X2_mesh, Y_mesh, alpha=0.3, cmap='viridis')

    # Add panel label (A, B, C)
    panel_label = chr(65 + idx)  # 65 is ASCII for 'A'
    ax.text2D(0.5, -0.06, f"({panel_label})", transform=ax.transAxes, fontsize=16,
              fontweight='bold', va='top', ha='center')

    # Formatting
    ax.set_xlabel(pred_names[0], fontsize=10, fontweight='bold', labelpad=8)
    ax.set_ylabel(pred_names[1], fontsize=10, fontweight='bold', labelpad=8)
    ax.set_zlabel('RPC', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_title(f'{pred_names[0]} + {pred_names[1]}\nR² = {r2_bi:.3f}',
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=8)

    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)

plt.tight_layout()
output_path = OUTPUT_DIR / 'bivariate_regressions_3d.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# Create full multivariate model visualization (4D using pairs)
print("\nCreating multivariate model visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Plot RPC (actual vs predicted)
ax = axes[0]
y_pred_full = model.predict(X)

for task in ['PAM50', 'ER', 'PR', 'HER2']:
    mask = df['Task'] == task
    if mask.any():
        ax.scatter(
            y[mask],
            y_pred_full[mask],
            color=task_colors[task],
            label=task,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5
        )

# Add diagonal line (perfect prediction)
min_val = min(y.min(), y_pred_full.min())
max_val = max(y.max(), y_pred_full.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)

textstr = f'R² = {r_squared:.3f}\nAdj R² = {adj_r_squared:.3f}\np(F) = {f_pval:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Add panel label A
ax.text(0.5, -0.18, '(A)', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='center')

ax.set_xlabel('Actual RPC', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted RPC', fontsize=12, fontweight='bold')
ax.set_title('Multivariate Model: Actual vs Predicted', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right')

# Plot residuals vs predicted
ax = axes[1]
for task in ['PAM50', 'ER', 'PR', 'HER2']:
    mask = df['Task'] == task
    if mask.any():
        ax.scatter(
            y_pred_full[mask],
            residuals[mask],
            color=task_colors[task],
            label=task,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5
        )

ax.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.8)

# Add panel label B
ax.text(0.5, -0.18, '(B)', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='center')

ax.set_xlabel('Predicted RPC', fontsize=12, fontweight='bold')
ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax.set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right')

# Plot residuals for each predictor
for idx, (pred, label) in enumerate(zip(predictors, pred_labels)):
    ax = axes[idx + 2]

    for task in ['PAM50', 'ER', 'PR', 'HER2']:
        mask = df['Task'] == task
        if mask.any():
            ax.scatter(
                df.loc[mask, pred],
                residuals[mask],
                color=task_colors[task],
                label=task,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )

    ax.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.8)

    # Add panel label (C, D, E, F)
    panel_label = chr(67 + idx)  # 67 is ASCII for 'C'
    ax.text(0.5, -0.18, f"({panel_label})", transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='center')

    ax.set_xlabel(label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax.set_title(f'Residuals vs {label}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')

plt.tight_layout()
output_path = OUTPUT_DIR / 'multivariate_model_diagnostics.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

print("\n✓ All visualizations saved to: results/regression_comparison/figures/")

# ============================================================================
# STEP 9: COMPLETE ANALYSIS WITHOUT COLLAPSED POINTS
# ============================================================================

print("\n" + "="*80)
print("STEP 9: COMPLETE ANALYSIS WITHOUT COLLAPSED POINTS")
print("="*80)

# Filter out collapsed points
df_no_collapse = df[df['RPC'] > -1].copy()
X_no_collapse = df_no_collapse[['NPG', 'd_c', 'delta_p', 'B_c']].values
y_no_collapse = df_no_collapse['RPC'].values
n_no_collapse = len(y_no_collapse)

print(f"\nDataset without collapsed points: n={n_no_collapse} (removed {len(df) - n_no_collapse} collapsed)")

# ============================================================================
# STEP 9.1: UNIVARIATE ANALYSES (NO COLLAPSED)
# ============================================================================

print("\n" + "="*80)
print("STEP 9.1: UNIVARIATE ANALYSES (NO COLLAPSED)")
print("="*80)

univariate_results_nc = []

for pred, label in zip(predictors, pred_labels):
    X_uni = df_no_collapse[[pred]].values
    r, p_corr = stats.pearsonr(X_uni.flatten(), y_no_collapse)

    model_uni = LinearRegression()
    model_uni.fit(X_uni, y_no_collapse)
    r2 = model_uni.score(X_uni, y_no_collapse)

    f_stat = (r2 / 1) / ((1 - r2) / (n_no_collapse - 2))
    f_pval = 1 - stats.f.cdf(f_stat, 1, n_no_collapse - 2)

    univariate_results_nc.append({
        'Predictor': label,
        'r': r,
        'R²': r2,
        'β': model_uni.coef_[0],
        'p': f_pval
    })

print(f"\n{'Predictor':<10} {'r':>8} {'R²':>8} {'β':>10} {'p-value':>10} {'Sig':>6}")
print("-"*80)

for res in sorted(univariate_results_nc, key=lambda x: x['p']):
    sig = ""
    if res['p'] < 0.001:
        sig = "***"
    elif res['p'] < 0.01:
        sig = "**"
    elif res['p'] < 0.05:
        sig = "*"
    elif res['p'] < 0.10:
        sig = "†"

    print(f"{res['Predictor']:<10} {res['r']:>8.3f} {res['R²']:>8.3f} {res['β']:>10.4f} {res['p']:>10.4f} {sig:>6}")

# ============================================================================
# STEP 9.2: MULTIVARIATE REGRESSION (NO COLLAPSED)
# ============================================================================

print("\n" + "="*80)
print("STEP 9.2: MULTIVARIATE REGRESSION (All 4 predictors, NO COLLAPSED)")
print("="*80)

k = X_no_collapse.shape[1]
model_nc = LinearRegression()
model_nc.fit(X_no_collapse, y_no_collapse)

y_pred_nc = model_nc.predict(X_no_collapse)
residuals_nc = y_no_collapse - y_pred_nc

ss_res_nc = np.sum(residuals_nc**2)
ss_tot_nc = np.sum((y_no_collapse - np.mean(y_no_collapse))**2)
r_squared_nc = 1 - (ss_res_nc / ss_tot_nc)
adj_r_squared_nc = 1 - (1 - r_squared_nc) * (n_no_collapse - 1) / (n_no_collapse - k - 1)

mse_nc = ss_res_nc / (n_no_collapse - k - 1)
se_nc = np.sqrt(mse_nc)

# Coefficient statistics
X_with_intercept_nc = np.column_stack([np.ones(n_no_collapse), X_no_collapse])
XtX_inv_nc = np.linalg.inv(X_with_intercept_nc.T @ X_with_intercept_nc)
se_coeffs_nc = np.sqrt(np.diagonal(XtX_inv_nc) * mse_nc)

coeffs_with_intercept_nc = np.concatenate([[model_nc.intercept_], model_nc.coef_])
t_stats_nc = coeffs_with_intercept_nc / se_coeffs_nc
p_values_nc = 2 * (1 - stats.t.cdf(np.abs(t_stats_nc), n_no_collapse - k - 1))

# F-statistic
f_stat_nc = (r_squared_nc / k) / ((1 - r_squared_nc) / (n_no_collapse - k - 1))
f_pval_nc = 1 - stats.f.cdf(f_stat_nc, k, n_no_collapse - k - 1)

print(f"\nR² = {r_squared_nc:.4f}")
print(f"Adjusted R² = {adj_r_squared_nc:.4f}")
print(f"F-statistic = {f_stat_nc:.4f} (p = {f_pval_nc:.4f})")
print(f"Residual SE = {se_nc:.4f}")

if f_pval_nc < 0.05:
    print("✓ MODEL IS SIGNIFICANT")
else:
    print("✗ Model is NOT significant")

print(f"\n{'Variable':<15} {'Coefficient':>12} {'Std Error':>12} {'t-value':>10} {'p-value':>10} {'Sig':>6}")
print("-"*80)

var_names = ['(Intercept)', 'NPG', 'd_c', 'Δp_c', 'B_c']
for i, var in enumerate(var_names):
    coef = coeffs_with_intercept_nc[i]
    se_coef = se_coeffs_nc[i]
    t = t_stats_nc[i]
    p = p_values_nc[i]

    sig = ""
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"

    print(f"{var:<15} {coef:>12.4f} {se_coef:>12.4f} {t:>10.3f} {p:>10.4f} {sig:>6}")

# ============================================================================
# STEP 9.3: BIVARIATE MODELS (NO COLLAPSED)
# ============================================================================

print("\n" + "="*80)
print("STEP 9.3: BIVARIATE MODELS (All combinations of 2 predictors, NO COLLAPSED)")
print("="*80)

print(f"\n{'Model':<25} {'R²':>8} {'Adj R²':>10} {'F':>8} {'p(F)':>10} {'Sig':>6}")
print("-"*80)

for combo in predictor_combos_2:
    pred_indices = list(combo)
    pred_names = [pred_labels[i] for i in pred_indices]
    pred_cols = [predictors[i] for i in pred_indices]

    X_bi = df_no_collapse[pred_cols].values
    model_bi = LinearRegression()
    model_bi.fit(X_bi, y_no_collapse)
    y_pred_bi = model_bi.predict(X_bi)

    ss_res_bi = np.sum((y_no_collapse - y_pred_bi)**2)
    ss_tot_bi = np.sum((y_no_collapse - np.mean(y_no_collapse))**2)
    r2_bi = 1 - (ss_res_bi / ss_tot_bi)
    adj_r2_bi = 1 - (1 - r2_bi) * (n_no_collapse - 1) / (n_no_collapse - 3)

    f_bi = (r2_bi / 2) / ((1 - r2_bi) / (n_no_collapse - 3))
    p_f_bi = 1 - stats.f.cdf(f_bi, 2, n_no_collapse - 3)

    sig = ""
    if p_f_bi < 0.001:
        sig = "***"
    elif p_f_bi < 0.01:
        sig = "**"
    elif p_f_bi < 0.05:
        sig = "*"

    model_name = " + ".join(pred_names)
    print(f"{model_name:<25} {r2_bi:>8.3f} {adj_r2_bi:>10.3f} {f_bi:>8.3f} {p_f_bi:>10.4f} {sig:>6}")

# ============================================================================
# STEP 9.4: TRIVARIATE MODELS (NO COLLAPSED)
# ============================================================================

print("\n" + "="*80)
print("STEP 9.4: TRIVARIATE MODELS (3 predictors, NO COLLAPSED)")
print("="*80)

print(f"\n{'Model':<30} {'R²':>8} {'Adj R²':>10} {'F':>8} {'p(F)':>10} {'Sig':>6}")
print("-"*80)

predictor_combos_3 = list(combinations(range(4), 3))

for combo in predictor_combos_3:
    pred_indices = list(combo)
    pred_names = [pred_labels[i] for i in pred_indices]
    pred_cols = [predictors[i] for i in pred_indices]

    X_tri = df_no_collapse[pred_cols].values
    model_tri = LinearRegression()
    model_tri.fit(X_tri, y_no_collapse)
    y_pred_tri = model_tri.predict(X_tri)

    ss_res_tri = np.sum((y_no_collapse - y_pred_tri)**2)
    ss_tot_tri = np.sum((y_no_collapse - np.mean(y_no_collapse))**2)
    r2_tri = 1 - (ss_res_tri / ss_tot_tri)
    adj_r2_tri = 1 - (1 - r2_tri) * (n_no_collapse - 1) / (n_no_collapse - 4)

    f_tri = (r2_tri / 3) / ((1 - r2_tri) / (n_no_collapse - 4))
    p_f_tri = 1 - stats.f.cdf(f_tri, 3, n_no_collapse - 4)

    sig = ""
    if p_f_tri < 0.001:
        sig = "***"
    elif p_f_tri < 0.01:
        sig = "**"
    elif p_f_tri < 0.05:
        sig = "*"

    model_name = " + ".join(pred_names)
    print(f"{model_name:<30} {r2_tri:>8.3f} {adj_r2_tri:>10.3f} {f_tri:>8.3f} {p_f_tri:>10.4f} {sig:>6}")

# ============================================================================
# STEP 9.5: VISUALIZATIONS (NO COLLAPSED)
# ============================================================================

print("\n" + "="*80)
print("STEP 9.5: GENERATING VISUALIZATIONS (NO COLLAPSED)")
print("="*80)

print("\nCreating univariate regression plots (no collapsed)...")

# Create 2x2 subplot for the 4 univariate regressions
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (pred, label) in enumerate(zip(predictors, pred_labels)):
    ax = axes[idx]

    # Get predictor values
    X_uni = df_no_collapse[pred].values
    y_vals = df_no_collapse['RPC'].values

    # Fit regression
    model_uni = LinearRegression()
    model_uni.fit(X_uni.reshape(-1, 1), y_vals)

    # Generate prediction line
    X_range = np.linspace(X_uni.min(), X_uni.max(), 100)
    y_pred_line = model_uni.predict(X_range.reshape(-1, 1))

    # Plot points by task
    for task in ['PAM50', 'ER', 'PR', 'HER2']:
        mask = df_no_collapse['Task'] == task
        if mask.any():
            ax.scatter(
                df_no_collapse.loc[mask, pred],
                df_no_collapse.loc[mask, 'RPC'],
                color=task_colors[task],
                label=task,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )

    # Plot regression line
    ax.plot(X_range, y_pred_line, 'k--', linewidth=2, alpha=0.8, label='Regression')

    # Get statistics
    r2 = model_uni.score(X_uni.reshape(-1, 1), y_vals)
    r, p_corr = stats.pearsonr(X_uni, y_vals)

    # Determine significance
    sig = ""
    if p_corr < 0.001:
        sig = "***"
    elif p_corr < 0.01:
        sig = "**"
    elif p_corr < 0.05:
        sig = "*"
    elif p_corr < 0.10:
        sig = "†"

    # Add statistics to plot
    textstr = f'R² = {r2:.3f}\nr = {r:.3f}\np = {p_corr:.4f} {sig}\nn = {n_no_collapse}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Add panel label (A, B, C, D)
    panel_label = chr(65 + idx)  # 65 is ASCII for 'A'
    ax.text(0.5, -0.18, f"({panel_label})", transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='center')

    # Formatting
    ax.set_xlabel(label, fontsize=12, fontweight='bold')
    ax.set_ylabel('RPC', fontsize=12, fontweight='bold')
    ax.set_title(f'RPC vs {label} (No Collapsed)', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
output_path = OUTPUT_DIR / 'univariate_regressions_no_collapsed.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# Create bivariate regression visualizations (no collapsed)
print("\nCreating bivariate regression plots (no collapsed)...")

# Select top 3 bivariate models by R²
bivariate_models_nc = []
for combo in predictor_combos_2:
    pred_indices = list(combo)
    pred_cols = [predictors[i] for i in pred_indices]
    pred_names = [pred_labels[i] for i in pred_indices]

    X_bi = df_no_collapse[pred_cols].values
    model_bi = LinearRegression()
    model_bi.fit(X_bi, y_no_collapse)
    r2_bi = model_bi.score(X_bi, y_no_collapse)

    bivariate_models_nc.append({
        'predictors': pred_names,
        'pred_cols': pred_cols,
        'r2': r2_bi,
        'model': model_bi
    })

# Sort by R² and select top 3
bivariate_models_nc = sorted(bivariate_models_nc, key=lambda x: x['r2'], reverse=True)[:3]

# Create 3D scatter plots for top 3 bivariate models
fig = plt.figure(figsize=(18, 6))

for idx, model_info in enumerate(bivariate_models_nc):
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

    pred_names = model_info['predictors']
    pred_cols = model_info['pred_cols']
    model_bi = model_info['model']
    r2_bi = model_info['r2']

    # Plot points by task
    for task in ['PAM50', 'ER', 'PR', 'HER2']:
        mask = df_no_collapse['Task'] == task
        if mask.any():
            ax.scatter(
                df_no_collapse.loc[mask, pred_cols[0]],
                df_no_collapse.loc[mask, pred_cols[1]],
                df_no_collapse.loc[mask, 'RPC'],
                color=task_colors[task],
                label=task,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )

    # Create mesh for regression plane
    x1_range = np.linspace(df_no_collapse[pred_cols[0]].min(), df_no_collapse[pred_cols[0]].max(), 20)
    x2_range = np.linspace(df_no_collapse[pred_cols[1]].min(), df_no_collapse[pred_cols[1]].max(), 20)
    X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)

    # Predict on mesh
    X_mesh = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])
    Y_mesh = model_bi.predict(X_mesh).reshape(X1_mesh.shape)

    # Plot regression plane
    ax.plot_surface(X1_mesh, X2_mesh, Y_mesh, alpha=0.3, cmap='viridis')

    # Add panel label (A, B, C)
    panel_label = chr(65 + idx)  # 65 is ASCII for 'A'
    ax.text2D(0.5, -0.06, f"({panel_label})", transform=ax.transAxes, fontsize=16,
              fontweight='bold', va='top', ha='center')

    # Formatting
    ax.set_xlabel(pred_names[0], fontsize=10, fontweight='bold', labelpad=8)
    ax.set_ylabel(pred_names[1], fontsize=10, fontweight='bold', labelpad=8)
    ax.set_zlabel('RPC', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_title(f'{pred_names[0]} + {pred_names[1]} (No Collapsed)\nR² = {r2_bi:.3f}, n = {n_no_collapse}',
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=8)

    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)

plt.tight_layout()
output_path = OUTPUT_DIR / 'bivariate_regressions_3d_no_collapsed.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# Create full multivariate model visualization (no collapsed)
print("\nCreating multivariate model visualization (no collapsed)...")

# Fit multivariate model without collapsed points
k = X_no_collapse.shape[1]
model_nc = LinearRegression()
model_nc.fit(X_no_collapse, y_no_collapse)

y_pred_full_nc = model_nc.predict(X_no_collapse)
residuals_nc = y_no_collapse - y_pred_full_nc

ss_res_nc = np.sum(residuals_nc**2)
ss_tot_nc = np.sum((y_no_collapse - np.mean(y_no_collapse))**2)
r_squared_nc = 1 - (ss_res_nc / ss_tot_nc)
adj_r_squared_nc = 1 - (1 - r_squared_nc) * (n_no_collapse - 1) / (n_no_collapse - k - 1)

f_stat_nc = (r_squared_nc / k) / ((1 - r_squared_nc) / (n_no_collapse - k - 1))
f_pval_nc = 1 - stats.f.cdf(f_stat_nc, k, n_no_collapse - k - 1)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Plot RPC (actual vs predicted)
ax = axes[0]

for task in ['PAM50', 'ER', 'PR', 'HER2']:
    mask = df_no_collapse['Task'] == task
    if mask.any():
        ax.scatter(
            y_no_collapse[mask.values],
            y_pred_full_nc[mask.values],
            color=task_colors[task],
            label=task,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5
        )

# Add diagonal line (perfect prediction)
min_val = min(y_no_collapse.min(), y_pred_full_nc.min())
max_val = max(y_no_collapse.max(), y_pred_full_nc.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)

textstr = f'R² = {r_squared_nc:.3f}\nAdj R² = {adj_r_squared_nc:.3f}\np(F) = {f_pval_nc:.4f}\nn = {n_no_collapse}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Add panel label A
ax.text(0.5, -0.18, '(A)', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='center')

ax.set_xlabel('Actual RPC', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted RPC', fontsize=12, fontweight='bold')
ax.set_title('Multivariate Model: Actual vs Predicted (No Collapsed)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right')

# Plot residuals vs predicted
ax = axes[1]
for task in ['PAM50', 'ER', 'PR', 'HER2']:
    mask = df_no_collapse['Task'] == task
    if mask.any():
        ax.scatter(
            y_pred_full_nc[mask.values],
            residuals_nc[mask.values],
            color=task_colors[task],
            label=task,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5
        )

ax.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.8)

# Add panel label B
ax.text(0.5, -0.18, '(B)', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='center')

ax.set_xlabel('Predicted RPC', fontsize=12, fontweight='bold')
ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax.set_title('Residuals vs Predicted (No Collapsed)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right')

# Plot residuals for each predictor
for idx, (pred, label) in enumerate(zip(predictors, pred_labels)):
    ax = axes[idx + 2]

    for task in ['PAM50', 'ER', 'PR', 'HER2']:
        mask = df_no_collapse['Task'] == task
        if mask.any():
            ax.scatter(
                df_no_collapse.loc[mask, pred],
                residuals_nc[mask.values],
                color=task_colors[task],
                label=task,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )

    ax.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.8)

    # Add panel label (C, D, E, F)
    panel_label = chr(67 + idx)  # 67 is ASCII for 'C'
    ax.text(0.5, -0.18, f"({panel_label})", transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='center')

    ax.set_xlabel(label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax.set_title(f'Residuals vs {label} (No Collapsed)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')

plt.tight_layout()
output_path = OUTPUT_DIR / 'multivariate_model_diagnostics_no_collapsed.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

print("\n✓ All visualizations (with and without collapsed) saved to: results/regression_comparison/figures/")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)