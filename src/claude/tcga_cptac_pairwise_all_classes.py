#!/usr/bin/env python3
"""
TCGA vs CPTAC: Comprehensive Pairwise Analysis

For PAM50: All pairwise comparisons between ALL classes (cross-cohort and cross-class)
For IHC (ER, PR, HER2): Only same-class comparisons between cohorts

Author: Claude Code
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, chi2_contingency
from itertools import combinations

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input paths
EXCEL_PATH = Path('data/histomorfologico/representative_images_annotation.xlsx')

# Output directory
OUTPUT_DIR = Path('results/biological_analysis/pairwise_cohort_analysis')

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
    'ESTRUCTURA GLANDULAR': 'Glandular Structure',
    'ATIPIA NUCLEAR': 'Nuclear Atypia',
    'MITOSIS': 'Mitosis',
    'NECROSIS': 'Necrosis',
    'INFILTRADO_LI': 'Lymphocytic Infiltrate',
    'INFILTRADO_PMN': 'PMN Infiltrate'
}

# Class mapping
CLASS_MAPPING = {
    'BASAL': 'Basal',
    'HER2-enriched': 'Her2',
    'LUMINAL-A': 'LumA',
    'LUMINAL-B': 'LumB',
    'NORMAL-like': 'Normal',
    'ER-positive': 'ER+',
    'ER-negative': 'ER-',
    'PR-positive': 'PR+',
    'PR-negative': 'PR-',
    'HER2-positive': 'HER2+',
    'HER2-negative': 'HER2-'
}

# Task mapping
TASK_MAPPING = {
    'Basal': 'PAM50',
    'Her2': 'PAM50',
    'LumA': 'PAM50',
    'LumB': 'PAM50',
    'Normal': 'PAM50',
    'ER+': 'IHC',
    'ER-': 'IHC',
    'PR+': 'IHC',
    'PR-': 'IHC',
    'HER2+': 'IHC',
    'HER2-': 'IHC'
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
# PAIRWISE ANALYSIS - PAM50 (ALL vs ALL)
# ==============================================================================

def pairwise_analysis_pam50(df):
    """
    All pairwise comparisons for PAM50.
    Compares every class-cohort combination with every other.

    E.g., Basal-TCGA vs Basal-CPTAC, Basal-TCGA vs LumA-TCGA, etc.
    """
    print("\n" + "="*80)
    print("PAM50: ALL PAIRWISE COMPARISONS")
    print("="*80)

    df_pam50 = df[df['Task'] == 'PAM50'].copy()

    # Create class-cohort combinations
    df_pam50['Class_Cohort'] = df_pam50['Class'] + '-' + df_pam50['Cohort']

    class_cohort_combinations = sorted(df_pam50['Class_Cohort'].unique())

    print(f"\nTotal combinations: {len(class_cohort_combinations)}")
    print(f"Combinations: {class_cohort_combinations}")

    # Generate all pairs
    pairs = list(combinations(class_cohort_combinations, 2))
    print(f"Total pairwise comparisons: {len(pairs)}")

    results_ordinal = []
    results_binary = []

    # Ordinal features
    print("\nProcessing ordinal features...")
    for feature in ORDINAL_FEATURES:
        print(f"  {feature}...")
        for pair in pairs:
            class1, class2 = pair

            data1 = df_pam50[df_pam50['Class_Cohort'] == class1][feature].values
            data2 = df_pam50[df_pam50['Class_Cohort'] == class2][feature].values

            if len(data1) < 3 or len(data2) < 3:
                continue

            # Mann-Whitney U test
            U, p_value = mannwhitneyu(data1, data2, alternative='two-sided')

            # Effect size (rank-biserial correlation)
            r_rb = 1 - (2 * U) / (len(data1) * len(data2))

            results_ordinal.append({
                'Feature': feature,
                'Class_Cohort_1': class1,
                'Class_Cohort_2': class2,
                'n1': len(data1),
                'n2': len(data2),
                'Mean_1': np.mean(data1),
                'Mean_2': np.mean(data2),
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
        for pair in pairs:
            class1, class2 = pair

            data1 = df_pam50[df_pam50['Class_Cohort'] == class1][feature].values
            data2 = df_pam50[df_pam50['Class_Cohort'] == class2][feature].values

            if len(data1) < 3 or len(data2) < 3:
                continue

            # Create contingency table
            count1_pos = np.sum(data1 == 1)
            count1_neg = len(data1) - count1_pos
            count2_pos = np.sum(data2 == 1)
            count2_neg = len(data2) - count2_pos

            contingency = np.array([[count1_pos, count1_neg],
                                   [count2_pos, count2_neg]])

            # Skip if any cell has zero (chi-square not valid)
            if np.any(contingency == 0):
                continue

            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency)

            # Cramér's V
            n = len(data1) + len(data2)
            cramers_v = np.sqrt(chi2 / n)

            results_binary.append({
                'Feature': feature,
                'Class_Cohort_1': class1,
                'Class_Cohort_2': class2,
                'n1': len(data1),
                'n2': len(data2),
                'Prop_1': count1_pos / len(data1),
                'Prop_2': count2_pos / len(data2),
                'chi2_statistic': chi2,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'significant': p_value < ALPHA
            })

    df_ordinal = pd.DataFrame(results_ordinal)
    df_binary = pd.DataFrame(results_binary)

    return df_ordinal, df_binary


# ==============================================================================
# PAIRWISE ANALYSIS - IHC (SAME CLASS ONLY)
# ==============================================================================

def pairwise_analysis_ihc(df):
    """
    Same-class comparisons only for IHC.
    E.g., ER+-TCGA vs ER+-CPTAC, PR+-TCGA vs PR+-CPTAC, etc.
    """
    print("\n" + "="*80)
    print("IHC: SAME-CLASS COHORT COMPARISONS")
    print("="*80)

    df_ihc = df[df['Task'] == 'IHC'].copy()

    classes = sorted(df_ihc['Class'].unique())
    print(f"\nClasses: {classes}")

    results_ordinal = []
    results_binary = []

    # Ordinal features
    print("\nProcessing ordinal features...")
    for feature in ORDINAL_FEATURES:
        print(f"  {feature}...")
        for class_name in classes:
            class_df = df_ihc[df_ihc['Class'] == class_name]

            data_tcga = class_df[class_df['Cohort'] == 'TCGA'][feature].values
            data_cptac = class_df[class_df['Cohort'] == 'CPTAC'][feature].values

            if len(data_tcga) < 3 or len(data_cptac) < 3:
                continue

            # Mann-Whitney U test
            U, p_value = mannwhitneyu(data_tcga, data_cptac, alternative='two-sided')

            # Effect size
            r_rb = 1 - (2 * U) / (len(data_tcga) * len(data_cptac))

            results_ordinal.append({
                'Feature': feature,
                'Class': class_name,
                'n_TCGA': len(data_tcga),
                'n_CPTAC': len(data_cptac),
                'Mean_TCGA': np.mean(data_tcga),
                'Mean_CPTAC': np.mean(data_cptac),
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
        for class_name in classes:
            class_df = df_ihc[df_ihc['Class'] == class_name]

            data_tcga = class_df[class_df['Cohort'] == 'TCGA'][feature].values
            data_cptac = class_df[class_df['Cohort'] == 'CPTAC'][feature].values

            if len(data_tcga) < 3 or len(data_cptac) < 3:
                continue

            # Contingency table
            tcga_pos = np.sum(data_tcga == 1)
            tcga_neg = len(data_tcga) - tcga_pos
            cptac_pos = np.sum(data_cptac == 1)
            cptac_neg = len(data_cptac) - cptac_pos

            contingency = np.array([[tcga_pos, tcga_neg],
                                   [cptac_pos, cptac_neg]])

            # Skip if any cell has zero (chi-square not valid)
            if np.any(contingency == 0):
                continue

            chi2, p_value, dof, expected = chi2_contingency(contingency)

            n = len(data_tcga) + len(data_cptac)
            cramers_v = np.sqrt(chi2 / n)

            results_binary.append({
                'Feature': feature,
                'Class': class_name,
                'n_TCGA': len(data_tcga),
                'n_CPTAC': len(data_cptac),
                'Prop_TCGA': tcga_pos / len(data_tcga),
                'Prop_CPTAC': cptac_pos / len(data_cptac),
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

def save_results(pam50_ordinal, pam50_binary, ihc_ordinal, ihc_binary):
    """Save all results to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # PAM50
    pam50_dir = OUTPUT_DIR / 'pam50'
    pam50_dir.mkdir(exist_ok=True)

    pam50_ordinal.to_csv(pam50_dir / 'pam50_pairwise_ordinal.csv', index=False, float_format='%.4f')
    pam50_binary.to_csv(pam50_dir / 'pam50_pairwise_binary.csv', index=False, float_format='%.4f')

    print(f"\nPAM50 Results:")
    print(f"  Ordinal comparisons: {len(pam50_ordinal)} ({pam50_ordinal['significant'].sum()} significant)")
    print(f"  Binary comparisons: {len(pam50_binary)} ({pam50_binary['significant'].sum()} significant)")
    print(f"  Saved to: {pam50_dir}/")

    # IHC
    ihc_dir = OUTPUT_DIR / 'ihc'
    ihc_dir.mkdir(exist_ok=True)

    ihc_ordinal.to_csv(ihc_dir / 'ihc_cohort_ordinal.csv', index=False, float_format='%.4f')
    ihc_binary.to_csv(ihc_dir / 'ihc_cohort_binary.csv', index=False, float_format='%.4f')

    print(f"\nIHC Results:")
    print(f"  Ordinal comparisons: {len(ihc_ordinal)} ({ihc_ordinal['significant'].sum()} significant)")
    print(f"  Binary comparisons: {len(ihc_binary)} ({ihc_binary['significant'].sum()} significant)")
    print(f"  Saved to: {ihc_dir}/")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("TCGA vs CPTAC: COMPREHENSIVE PAIRWISE ANALYSIS")
    print("="*80)

    # Load data
    df = load_data()

    # PAM50: All pairwise comparisons
    pam50_ordinal, pam50_binary = pairwise_analysis_pam50(df)

    # IHC: Same-class only
    ihc_ordinal, ihc_binary = pairwise_analysis_ihc(df)

    # Save results
    save_results(pam50_ordinal, pam50_binary, ihc_ordinal, ihc_binary)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
