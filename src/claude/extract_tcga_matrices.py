#!/usr/bin/env python3
"""
Extract TCGA Within-Cohort Effect Size Matrices

Reads the TCGA pairwise Mann-Whitney test results and builds
symmetric effect size matrices for each task (PAM50, ER, PR, HER2).

Author: Claude Code
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Input path
PAIRWISE_PATH = Path('results/biological_analysis/tcga_statistics/mann_whitney_pairwise_tests.csv')

# Load pairwise data
df = pd.read_csv(PAIRWISE_PATH)

print("="*80)
print("EXTRACTING TCGA WITHIN-COHORT EFFECT SIZE MATRICES")
print("="*80)

# ============================================================================
# PAM50 MATRIX
# ============================================================================

print("\n" + "="*80)
print("PAM50 MATRIX (5x5)")
print("="*80)

df_pam50 = df[df['Task'] == 'PAM50'].copy()

# Classes in order: Basal, Her2-enriched, LumA, LumB, Normal
classes_pam50 = ['Basal', 'Her2-enriched', 'LumA', 'LumB', 'Normal']
n_pam50 = len(classes_pam50)

# Initialize symmetric matrix
pam50_matrix = np.zeros((n_pam50, n_pam50))

# Map class names to indices
class_to_idx = {cls: i for i, cls in enumerate(classes_pam50)}

# Fill matrix with cumulative effect sizes (sum across features)
for i, cls1 in enumerate(classes_pam50):
    for j, cls2 in enumerate(classes_pam50):
        if i == j:
            # Diagonal is 0 (same class)
            pam50_matrix[i, j] = 0.0
        else:
            # Find pairwise comparisons for this pair
            mask = ((df_pam50['Class_1'] == cls1) & (df_pam50['Class_2'] == cls2)) | \
                   ((df_pam50['Class_1'] == cls2) & (df_pam50['Class_2'] == cls1))

            subset = df_pam50[mask]

            if len(subset) > 0:
                # Sum absolute rank_biserial values across features
                cumulative_effect = subset['effect_size'].str.replace('Small', '0.1').str.replace('Medium', '0.3').str.replace('Large', '0.5')
                # Actually use the rank_biserial column
                cumulative_effect = np.abs(subset['rank_biserial']).sum()
                pam50_matrix[i, j] = cumulative_effect

print("\nPAM50 Matrix (Basal, Her2-enriched, LumA, LumB, Normal):")
print(pam50_matrix)
print("\nNumpy array code:")
print("tcga_pam50_matrix = np.array([")
for i, cls in enumerate(classes_pam50):
    values_str = ', '.join([f'{pam50_matrix[i, j]:.2f}' for j in range(n_pam50)])
    print(f"    [{values_str}],  # {cls} row (index {i})")
print("])")

# ============================================================================
# ER MATRIX
# ============================================================================

print("\n" + "="*80)
print("ER MATRIX (2x2)")
print("="*80)

df_er = df[df['Task'] == 'ER'].copy()

classes_er = ['ER-negative', 'ER-positive']
n_er = len(classes_er)

er_matrix = np.zeros((n_er, n_er))

for i, cls1 in enumerate(classes_er):
    for j, cls2 in enumerate(classes_er):
        if i == j:
            er_matrix[i, j] = 0.0
        else:
            mask = ((df_er['Class_1'] == cls1) & (df_er['Class_2'] == cls2)) | \
                   ((df_er['Class_1'] == cls2) & (df_er['Class_2'] == cls1))
            subset = df_er[mask]
            if len(subset) > 0:
                cumulative_effect = np.abs(subset['rank_biserial']).sum()
                er_matrix[i, j] = cumulative_effect

print("\nER Matrix (ER-negative, ER-positive):")
print(er_matrix)
print("\nNumpy array code:")
print("tcga_er_matrix = np.array([")
for i, cls in enumerate(classes_er):
    values_str = ', '.join([f'{er_matrix[i, j]:.2f}' for j in range(n_er)])
    print(f"    [{values_str}],  # {cls} row")
print("])")

# ============================================================================
# PR MATRIX
# ============================================================================

print("\n" + "="*80)
print("PR MATRIX (2x2)")
print("="*80)

df_pr = df[df['Task'] == 'PR'].copy()

classes_pr = ['PR-negative', 'PR-positive']
n_pr = len(classes_pr)

pr_matrix = np.zeros((n_pr, n_pr))

for i, cls1 in enumerate(classes_pr):
    for j, cls2 in enumerate(classes_pr):
        if i == j:
            pr_matrix[i, j] = 0.0
        else:
            mask = ((df_pr['Class_1'] == cls1) & (df_pr['Class_2'] == cls2)) | \
                   ((df_pr['Class_1'] == cls2) & (df_pr['Class_2'] == cls1))
            subset = df_pr[mask]
            if len(subset) > 0:
                cumulative_effect = np.abs(subset['rank_biserial']).sum()
                pr_matrix[i, j] = cumulative_effect

print("\nPR Matrix (PR-negative, PR-positive):")
print(pr_matrix)
print("\nNumpy array code:")
print("tcga_pr_matrix = np.array([")
for i, cls in enumerate(classes_pr):
    values_str = ', '.join([f'{pr_matrix[i, j]:.2f}' for j in range(n_pr)])
    print(f"    [{values_str}],  # {cls} row")
print("])")

# ============================================================================
# HER2 MATRIX
# ============================================================================

print("\n" + "="*80)
print("HER2 MATRIX (2x2)")
print("="*80)

df_her2 = df[df['Task'] == 'HER2'].copy()

classes_her2 = ['HER2-negative', 'HER2-positive']
n_her2 = len(classes_her2)

her2_matrix = np.zeros((n_her2, n_her2))

for i, cls1 in enumerate(classes_her2):
    for j, cls2 in enumerate(classes_her2):
        if i == j:
            her2_matrix[i, j] = 0.0
        else:
            mask = ((df_her2['Class_1'] == cls1) & (df_her2['Class_2'] == cls2)) | \
                   ((df_her2['Class_1'] == cls2) & (df_her2['Class_2'] == cls1))
            subset = df_her2[mask]
            if len(subset) > 0:
                cumulative_effect = np.abs(subset['rank_biserial']).sum()
                her2_matrix[i, j] = cumulative_effect

print("\nHER2 Matrix (HER2-negative, HER2-positive):")
print(her2_matrix)
print("\nNumpy array code:")
print("tcga_her2_matrix = np.array([")
for i, cls in enumerate(classes_her2):
    values_str = ', '.join([f'{her2_matrix[i, j]:.2f}' for j in range(n_her2)])
    print(f"    [{values_str}],  # {cls} row")
print("])")

print("\n" + "="*80)
print("EXTRACTION COMPLETE")
print("="*80)
