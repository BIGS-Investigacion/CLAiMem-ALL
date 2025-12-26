#!/usr/bin/env python3
"""
TCGA vs CPTAC: Cross-Cohort Cross-Class Comparisons

Compares TCGA classes with CPTAC classes (not within same cohort).
For example: Basal-TCGA vs LumA-CPTAC, Basal-TCGA vs Her2-CPTAC, etc.

Separated by:
- PAM50: All TCGA PAM50 classes vs all CPTAC PAM50 classes
- IHC: All TCGA IHC classes vs all CPTAC IHC classes

Author: Claude Code
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, chi2_contingency
from itertools import product

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input paths
EXCEL_PATH = Path('data/histomorfologico/representative_images_annotation.xlsx')

# Output directory
OUTPUT_DIR = Path('results/biological_analysis/tcga_vs_cptac_cross_class')

# Significance threshold
ALPHA = 0.05

# Features
ORDINAL_FEATURES = [
    'ESTRUCTURA GLANDULAR',
    'ATIPIA NUCLEAR',
    'MITOSIS'
]

BINARY_FEATURES = [
    'NECROSIS',
    'INFILTRADO_LI',
    'INFILTRADO_PMN'
]

FEATURE_NAMES = {
    'ESTRUCTURA GLANDULAR': 'Glandular\nStructure',
    'ATIPIA NUCLEAR': 'Nuclear\nAtypia',
    'MITOSIS': 'Mitosis',
    'NECROSIS': 'Necrosis',
    'INFILTRADO_LI': 'Lymphocytic\nInfiltrate',
    'INFILTRADO_PMN': 'PMN\nInfiltrate'
}

# Class mapping (IHC: negative before positive)
CLASS_MAPPING = {
    'BASAL': 'Basal',
    'HER2-enriched': 'Her2-enriched',
    'LUMINAL-A': 'LumA',
    'LUMINAL-B': 'LumB',
    'NORMAL-like': 'Normal',
    'ER-negative': 'ER-',
    'ER-positive': 'ER+',
    'PR-negative': 'PR-',
    'PR-positive': 'PR+',
    'HER2-negative': 'HER2-',
    'HER2-positive': 'HER2+'
}

# Task mapping
TASK_MAPPING = {
    'Basal': 'PAM50',
    'Her2-enriched': 'PAM50',
    'LumA': 'PAM50',
    'LumB': 'PAM50',
    'Normal': 'PAM50',
    'ER+': 'ER',
    'ER-': 'ER',
    'PR+': 'PR',
    'PR-': 'PR',
    'HER2+': 'HER2',
    'HER2-': 'HER2'
}

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data():
    """Load and process data from Excel."""
    print("\nLoading data...")

    # Load both sheets
    df_tcga = pd.read_excel(EXCEL_PATH, sheet_name='TCGA')
    df_tcga['Cohort'] = 'TCGA'

    df_cptac = pd.read_excel(EXCEL_PATH, sheet_name='CPTAC')
    df_cptac['Cohort'] = 'CPTAC'

    # Concatenate
    df = pd.concat([df_tcga, df_cptac], ignore_index=True)

    # Map class names
    df['Class'] = df['ETIQUETA'].map(CLASS_MAPPING)
    df['Task'] = df['Class'].map(TASK_MAPPING)

    print(f"  Total samples: {len(df)}")
    print(f"  TCGA: {len(df_tcga)}, CPTAC: {len(df_cptac)}")

    return df


# ==============================================================================
# CROSS-COHORT ANALYSIS
# ==============================================================================

def cross_cohort_analysis(df, task_name):
    """
    Compare all TCGA classes with all CPTAC classes for a given task.

    Args:
        df: DataFrame with all data
        task_name: 'PAM50' or 'IHC'
    """
    print(f"\n" + "="*80)
    print(f"{task_name}: TCGA vs CPTAC CROSS-CLASS COMPARISONS")
    print("="*80)

    df_task = df[df['Task'] == task_name].copy()

    # Get TCGA and CPTAC classes
    tcga_classes = sorted(df_task[df_task['Cohort'] == 'TCGA']['Class'].unique())
    cptac_classes = sorted(df_task[df_task['Cohort'] == 'CPTAC']['Class'].unique())

    print(f"\nTCGA classes: {tcga_classes}")
    print(f"CPTAC classes: {cptac_classes}")

    # Generate all cross-cohort pairs
    cross_pairs = list(product(tcga_classes, cptac_classes))
    print(f"Total cross-cohort comparisons: {len(cross_pairs)}")

    results_ordinal = []
    results_binary = []

    # Ordinal features
    print("\nProcessing ordinal features...")
    for feature in ORDINAL_FEATURES:
        print(f"  {feature}...")
        for tcga_class, cptac_class in cross_pairs:
            # Get data for TCGA class
            tcga_data = df_task[(df_task['Cohort'] == 'TCGA') &
                               (df_task['Class'] == tcga_class)][feature].values

            # Get data for CPTAC class
            cptac_data = df_task[(df_task['Cohort'] == 'CPTAC') &
                                (df_task['Class'] == cptac_class)][feature].values

            if len(tcga_data) < 3 or len(cptac_data) < 3:
                continue

            # Mann-Whitney U test
            U, p_value = mannwhitneyu(tcga_data, cptac_data, alternative='two-sided')

            # Effect size (rank-biserial correlation)
            r_rb = 1 - (2 * U) / (len(tcga_data) * len(cptac_data))

            results_ordinal.append({
                'Feature': feature,
                'TCGA_Class': tcga_class,
                'CPTAC_Class': cptac_class,
                'n_TCGA': len(tcga_data),
                'n_CPTAC': len(cptac_data),
                'Mean_TCGA': np.mean(tcga_data),
                'Mean_CPTAC': np.mean(cptac_data),
                'U_statistic': U,
                'p_value': p_value,
                'rank_biserial': r_rb,
                'effect_size_abs': abs(r_rb),
                'significant': p_value < ALPHA
            })

    # Binary features
    print("\nProcessing binary features...")
    for feature in BINARY_FEATURES:
        print(f"  {feature}...")
        for tcga_class, cptac_class in cross_pairs:
            # Get data for TCGA class
            tcga_data = df_task[(df_task['Cohort'] == 'TCGA') &
                               (df_task['Class'] == tcga_class)][feature].values

            # Get data for CPTAC class
            cptac_data = df_task[(df_task['Cohort'] == 'CPTAC') &
                                (df_task['Class'] == cptac_class)][feature].values

            if len(tcga_data) < 3 or len(cptac_data) < 3:
                continue

            # Create contingency table
            tcga_pos = np.sum(tcga_data == 1)
            tcga_neg = len(tcga_data) - tcga_pos
            cptac_pos = np.sum(cptac_data == 1)
            cptac_neg = len(cptac_data) - cptac_pos

            contingency = np.array([[tcga_pos, tcga_neg],
                                   [cptac_pos, cptac_neg]])

            # Skip if any cell has zero (chi-square not valid)
            if np.any(contingency == 0):
                continue

            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency)

            # Cramér's V
            n = len(tcga_data) + len(cptac_data)
            cramers_v = np.sqrt(chi2 / n)

            results_binary.append({
                'Feature': feature,
                'TCGA_Class': tcga_class,
                'CPTAC_Class': cptac_class,
                'n_TCGA': len(tcga_data),
                'n_CPTAC': len(cptac_data),
                'Prop_TCGA': tcga_pos / len(tcga_data),
                'Prop_CPTAC': cptac_pos / len(cptac_data),
                'chi2_statistic': chi2,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'significant': p_value < ALPHA
            })

    df_ordinal = pd.DataFrame(results_ordinal)
    df_binary = pd.DataFrame(results_binary)

    return df_ordinal, df_binary


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

def save_results(pam50_ordinal, pam50_binary, er_ordinal, er_binary, pr_ordinal, pr_binary, her2_ordinal, her2_binary):
    """Save all results to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # PAM50
    pam50_dir = OUTPUT_DIR / 'pam50'
    pam50_dir.mkdir(exist_ok=True)

    pam50_ordinal.to_csv(pam50_dir / 'pam50_cross_ordinal.csv', index=False, float_format='%.4f')
    pam50_binary.to_csv(pam50_dir / 'pam50_cross_binary.csv', index=False, float_format='%.4f')

    print(f"\nPAM50 Results:")
    print(f"  Ordinal comparisons: {len(pam50_ordinal)} ({pam50_ordinal['significant'].sum()} significant)")
    print(f"  Binary comparisons: {len(pam50_binary)} ({pam50_binary['significant'].sum()} significant)")
    print(f"  Saved to: {pam50_dir}/")

    # ER
    er_dir = OUTPUT_DIR / 'er'
    er_dir.mkdir(exist_ok=True)

    er_ordinal.to_csv(er_dir / 'er_cross_ordinal.csv', index=False, float_format='%.4f')
    er_binary.to_csv(er_dir / 'er_cross_binary.csv', index=False, float_format='%.4f')

    print(f"\nER Results:")
    print(f"  Ordinal comparisons: {len(er_ordinal)} ({er_ordinal['significant'].sum()} significant)")
    print(f"  Binary comparisons: {len(er_binary)} ({er_binary['significant'].sum()} significant)")
    print(f"  Saved to: {er_dir}/")

    # PR
    pr_dir = OUTPUT_DIR / 'pr'
    pr_dir.mkdir(exist_ok=True)

    pr_ordinal.to_csv(pr_dir / 'pr_cross_ordinal.csv', index=False, float_format='%.4f')
    pr_binary.to_csv(pr_dir / 'pr_cross_binary.csv', index=False, float_format='%.4f')

    print(f"\nPR Results:")
    print(f"  Ordinal comparisons: {len(pr_ordinal)} ({pr_ordinal['significant'].sum()} significant)")
    print(f"  Binary comparisons: {len(pr_binary)} ({pr_binary['significant'].sum()} significant)")
    print(f"  Saved to: {pr_dir}/")

    # HER2
    her2_dir = OUTPUT_DIR / 'her2'
    her2_dir.mkdir(exist_ok=True)

    her2_ordinal.to_csv(her2_dir / 'her2_cross_ordinal.csv', index=False, float_format='%.4f')
    her2_binary.to_csv(her2_dir / 'her2_cross_binary.csv', index=False, float_format='%.4f')

    print(f"\nHER2 Results:")
    print(f"  Ordinal comparisons: {len(her2_ordinal)} ({her2_ordinal['significant'].sum()} significant)")
    print(f"  Binary comparisons: {len(her2_binary)} ({her2_binary['significant'].sum()} significant)")
    print(f"  Saved to: {her2_dir}/")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("TCGA vs CPTAC: CROSS-COHORT CROSS-CLASS ANALYSIS")
    print("="*80)

    # Load data
    df = load_data()

    # PAM50: TCGA classes vs CPTAC classes
    pam50_ordinal, pam50_binary = cross_cohort_analysis(df, 'PAM50')

    # ER: TCGA classes vs CPTAC classes
    er_ordinal, er_binary = cross_cohort_analysis(df, 'ER')

    # PR: TCGA classes vs CPTAC classes
    pr_ordinal, pr_binary = cross_cohort_analysis(df, 'PR')

    # HER2: TCGA classes vs CPTAC classes
    her2_ordinal, her2_binary = cross_cohort_analysis(df, 'HER2')

    # Save results
    save_results(pam50_ordinal, pam50_binary, er_ordinal, er_binary, pr_ordinal, pr_binary, her2_ordinal, her2_binary)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
