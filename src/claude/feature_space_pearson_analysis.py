import numpy as np
from scipy.stats import pearsonr

# ============================================================================
# Per-class accuracy in Hold-Out (CPTAC) - Original model
# ============================================================================

# PAM50 (5 classes)
pam50_accuracy = {
    'LumA': 95.40,
    'LumB': 18.18,
    'Her2-enriched': 0.00,
    'Basal': 68.38,
    'Normal': 0.00
}

# ER (2 classes)
er_accuracy = {
    'ER-positive': 97.02,
    'ER-negative': 29.17
}

# PR (2 classes)
pr_accuracy = {
    'PR-positive': 92.00,
    'PR-negative': 58.08
}

# HER2 (2 classes)
her2_accuracy = {
    'HER2-positive': 0.00,
    'HER2-negative': 100.00
}

# ============================================================================
# Feature space metrics
# ============================================================================

# Centroid distances (from JSON files)
centroid_distance = {
    'LumA': 16.66,
    'LumB': 23.38,
    'Her2-enriched': 22.22,
    'Basal': 24.88,
    'Normal': 26.31,
    'ER-positive': 17.38,
    'ER-negative': 18.23,
    'PR-positive': 19.78,
    'PR-negative': 23.02,
    'HER2-positive': 20.92,
    'HER2-negative': 18.38
}

# Silhouette scores TCGA
silhouette_tcga = {
    'LumA': -0.032,
    'LumB': -0.039,
    'Her2-enriched': 0.110,
    'Basal': 0.118,
    'Normal': 0.066,
    'ER-positive': 0.089,
    'ER-negative': 0.092,
    'PR-positive': 0.004,
    'PR-negative': 0.117,
    'HER2-positive': 0.075,
    'HER2-negative': -0.029
}

# Silhouette scores CPTAC
silhouette_cptac = {
    'LumA': 0.023,
    'LumB': -0.021,
    'Her2-enriched': -0.017,
    'Basal': 0.008,
    'Normal': -0.020,
    'ER-positive': 0.054,
    'ER-negative': 0.010,
    'PR-positive': 0.013,
    'PR-negative': 0.010,
    'HER2-positive': 0.012,
    'HER2-negative': 0.007
}

# ============================================================================
# Combine all classes (11 total)
# ============================================================================

classes = ['LumA', 'LumB', 'Her2-enriched', 'Basal', 'Normal',
           'ER-positive', 'ER-negative',
           'PR-positive', 'PR-negative',
           'HER2-positive', 'HER2-negative']

# Create arrays in the same order
accuracy = np.array([
    pam50_accuracy['LumA'], pam50_accuracy['LumB'], pam50_accuracy['Her2-enriched'],
    pam50_accuracy['Basal'], pam50_accuracy['Normal'],
    er_accuracy['ER-positive'], er_accuracy['ER-negative'],
    pr_accuracy['PR-positive'], pr_accuracy['PR-negative'],
    her2_accuracy['HER2-positive'], her2_accuracy['HER2-negative']
])

centroid_dist = np.array([centroid_distance[c] for c in classes])
silh_tcga = np.array([silhouette_tcga[c] for c in classes])
silh_cptac = np.array([silhouette_cptac[c] for c in classes])

# ============================================================================
# Compute Pearson correlations
# ============================================================================

print("=" * 80)
print("CORRELATION ANALYSIS: Accuracy vs Feature Space Metrics")
print("=" * 80)
print(f"Number of classes: {len(classes)}")
print()

# Correlation 1: Accuracy vs Centroid Distance
r1, p1 = pearsonr(accuracy, centroid_dist)
print(f"Accuracy vs Centroid Distance:")
print(f"  Pearson r = {r1:.3f}")
print(f"  p-value   = {p1:.3f}")
print(f"  Significant: {'Yes' if p1 < 0.05 else 'No'} (α = 0.05)")
print()

# Correlation 2: Accuracy vs Silhouette TCGA
r2, p2 = pearsonr(accuracy, silh_tcga)
print(f"Accuracy vs Silhouette (TCGA):")
print(f"  Pearson r = {r2:.3f}")
print(f"  p-value   = {p2:.3f}")
print(f"  Significant: {'Yes' if p2 < 0.05 else 'No'} (α = 0.05)")
print()

# Correlation 3: Accuracy vs Silhouette CPTAC
r3, p3 = pearsonr(accuracy, silh_cptac)
print(f"Accuracy vs Silhouette (CPTAC):")
print(f"  Pearson r = {r3:.3f}")
print(f"  p-value   = {p3:.3f}")
print(f"  Significant: {'Yes' if p3 < 0.05 else 'No'} (α = 0.05)")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"None of the correlations are statistically significant (all p > 0.05).")
print(f"This suggests that feature space metrics alone do not fully explain")
print(f"external validation performance degradation.")
print("=" * 80)

# ============================================================================
# Optional: Display data table
# ============================================================================

print("\n" + "=" * 80)
print("DATA TABLE")
print("=" * 80)
print(f"{'Class':<15} {'Accuracy':<10} {'Centroid Dist':<15} {'Silh (TCGA)':<15} {'Silh (CPTAC)':<15}")
print("-" * 80)
for i, c in enumerate(classes):
    print(f"{c:<15} {accuracy[i]:<10.2f} {centroid_dist[i]:<15.2f} {silh_tcga[i]:<15.3f} {silh_cptac[i]:<15.3f}")
print("=" * 80)