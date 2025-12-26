#!/usr/bin/env python3
"""
Statistical Analysis of Histomorphological Features - TCGA Only

Performs comprehensive statistical analysis of biological features across
molecular classes for TCGA-BRCA patches, following methodology:

For ordinal features (Tubule Formation, Nuclear Pleomorphism, Mitotic Activity):
- Kruskal-Wallis tests with epsilon-squared effect size
- Pairwise Mann-Whitney U tests with Bonferroni correction
- Rank-biserial correlation as effect size

For binary features (Necrosis, Inflammatory Infiltrates):
- Chi-square tests with Cramér's V effect size

Interpretation thresholds:
- Small effect: ε² < 0.06, |r_rb| < 0.30, V < 0.10
- Medium effect: ε² ≥ 0.06, |r_rb| ≥ 0.30, V ≥ 0.10
- Large effect: ε² ≥ 0.14, |r_rb| ≥ 0.50, V ≥ 0.30

Author: Claude Code
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from itertools import combinations
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

EXCEL_PATH = Path('data/histomorfologico/representative_images_annotation.xlsx')
OUTPUT_DIR = Path('results/biological_analysis/tcga_statistics')

# Feature categories
ORDINAL_FEATURES = {
    'ESTRUCTURA GLANDULAR': 'Tubule Formation',
    'ATIPIA NUCLEAR': 'Nuclear Pleomorphism',
    'MITOSIS': 'Mitotic Activity'
}

BINARY_FEATURES = {
    'NECROSIS': 'Tumour Necrosis',
    'INFILTRADO_LI': 'Lymphocytic Infiltrate',
    'INFILTRADO_PMN': 'Polymorphonuclear Infiltrate'
}

# Class mapping
CLASS_MAPPING = {
    'BASAL': 'Basal',
    'HER2-enriched': 'Her2-enriched',
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
# EFFECT SIZE CALCULATIONS
# ==============================================================================

def epsilon_squared(H, n, k):
    """
    Calculate epsilon-squared effect size for Kruskal-Wallis test.
    
    ε² = H / ((n² - 1) / (n + 1))
    
    Interpretation:
    - Small: ε² < 0.06
    - Medium: 0.06 ≤ ε² < 0.14
    - Large: ε² ≥ 0.14
    """
    return H / ((n**2 - 1) / (n + 1))


def rank_biserial_correlation(U, n1, n2):
    """
    Calculate rank-biserial correlation for Mann-Whitney U test.
    
    r_rb = 1 - (2U) / (n1 * n2)
    
    Interpretation:
    - Small: |r_rb| < 0.30
    - Medium: 0.30 ≤ |r_rb| < 0.50
    - Large: |r_rb| ≥ 0.50
    """
    return 1 - (2 * U) / (n1 * n2)


def cramers_v(chi2, n, r, c):
    """
    Calculate Cramér's V effect size for chi-square test.
    
    V = sqrt(χ² / (n * min(r-1, c-1)))
    
    Interpretation:
    - Small: V < 0.10
    - Medium: 0.10 ≤ V < 0.30
    - Large: V ≥ 0.30
    """
    return np.sqrt(chi2 / (n * min(r - 1, c - 1)))


def interpret_effect_size(value, effect_type):
    """Interpret effect size magnitude."""
    if effect_type == 'epsilon_squared':
        if value < 0.06:
            return 'Small'
        elif value < 0.14:
            return 'Medium'
        else:
            return 'Large'
    elif effect_type == 'rank_biserial':
        abs_val = abs(value)
        if abs_val < 0.30:
            return 'Small'
        elif abs_val < 0.50:
            return 'Medium'
        else:
            return 'Large'
    elif effect_type == 'cramers_v':
        if value < 0.10:
            return 'Small'
        elif value < 0.30:
            return 'Medium'
        else:
            return 'Large'
    return 'Unknown'


def format_pvalue(p):
    """Format p-value with significance stars."""
    if p < 0.001:
        return f"{p:.4f} ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f} n.s."


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_tcga_data(excel_path: Path) -> pd.DataFrame:
    """Load TCGA data from Excel."""
    print(f"\nLoading TCGA data from: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name='TCGA')
    
    df['Class'] = df['ETIQUETA'].map(CLASS_MAPPING)
    df['Task'] = df['Class'].map(TASK_MAPPING)
    
    print(f"  Total TCGA samples: {len(df)}")
    print(f"  Classes: {sorted(df['Class'].unique())}")
    
    return df


# ==============================================================================
# KRUSKAL-WALLIS TESTS (ORDINAL FEATURES)
# ==============================================================================

def kruskal_wallis_analysis(df: pd.DataFrame, feature: str, task: str) -> dict:
    """
    Perform Kruskal-Wallis test for ordinal feature.
    
    Returns dictionary with test statistics.
    """
    task_df = df[df['Task'] == task]
    classes = sorted(task_df['Class'].unique())
    
    # Prepare data for each class
    groups = [task_df[task_df['Class'] == cls][feature].values for cls in classes]
    
    # Remove empty groups
    groups = [g for g in groups if len(g) > 0]
    classes = [c for i, c in enumerate(classes) if len(groups[i]) > 0]
    
    if len(groups) < 2:
        return None
    
    # Kruskal-Wallis test
    H, p_value = kruskal(*groups)
    
    # Total sample size
    n = sum(len(g) for g in groups)
    k = len(groups)
    
    # Effect size
    eps_sq = epsilon_squared(H, n, k)
    effect_interp = interpret_effect_size(eps_sq, 'epsilon_squared')
    
    return {
        'Feature': feature,
        'Task': task,
        'H_statistic': H,
        'p_value': p_value,
        'epsilon_squared': eps_sq,
        'effect_size': effect_interp,
        'n_total': n,
        'n_classes': k,
        'classes': classes
    }


def pairwise_mann_whitney(df: pd.DataFrame, feature: str, task: str, 
                         alpha: float = 0.05) -> list:
    """
    Perform pairwise Mann-Whitney U tests with Bonferroni correction.
    
    Returns list of dictionaries with pairwise comparisons.
    """
    task_df = df[df['Task'] == task]
    classes = sorted(task_df['Class'].unique())
    
    results = []
    
    # Get all pairwise combinations
    pairs = list(combinations(classes, 2))
    n_comparisons = len(pairs)
    bonferroni_alpha = alpha / n_comparisons
    
    for class1, class2 in pairs:
        data1 = task_df[task_df['Class'] == class1][feature].values
        data2 = task_df[task_df['Class'] == class2][feature].values
        
        if len(data1) == 0 or len(data2) == 0:
            continue
        
        # Mann-Whitney U test
        U, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Effect size
        r_rb = rank_biserial_correlation(U, len(data1), len(data2))
        effect_interp = interpret_effect_size(r_rb, 'rank_biserial')
        
        # Significance after Bonferroni correction
        significant = p_value < bonferroni_alpha
        
        results.append({
            'Feature': feature,
            'Task': task,
            'Class_1': class1,
            'Class_2': class2,
            'n_1': len(data1),
            'n_2': len(data2),
            'U_statistic': U,
            'p_value': p_value,
            'p_adjusted': bonferroni_alpha,
            'significant': significant,
            'rank_biserial': r_rb,
            'effect_size': effect_interp
        })
    
    return results


# ==============================================================================
# CHI-SQUARE TESTS (BINARY FEATURES)
# ==============================================================================

def chi_square_analysis(df: pd.DataFrame, feature: str, task: str) -> dict:
    """
    Perform chi-square test for binary feature.
    
    Returns dictionary with test statistics.
    """
    task_df = df[df['Task'] == task]
    
    # Create contingency table
    contingency = pd.crosstab(task_df['Class'], task_df[feature])
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Effect size
    n = contingency.sum().sum()
    r, c = contingency.shape
    cramers = cramers_v(chi2, n, r, c)
    effect_interp = interpret_effect_size(cramers, 'cramers_v')
    
    return {
        'Feature': feature,
        'Task': task,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers,
        'effect_size': effect_interp,
        'n_total': n,
        'contingency_table': contingency.to_dict()
    }


# ==============================================================================
# COMPREHENSIVE ANALYSIS
# ==============================================================================

def run_comprehensive_analysis(df: pd.DataFrame) -> dict:
    """Run all statistical tests for all tasks."""
    
    results = {
        'kruskal_wallis': [],
        'mann_whitney_pairwise': [],
        'chi_square': []
    }
    
    tasks = sorted(df['Task'].unique())
    
    print("\n" + "="*80)
    print("KRUSKAL-WALLIS TESTS (Ordinal Features)")
    print("="*80)
    
    for task in tasks:
        print(f"\n{task}:")
        for feature_orig, feature_name in ORDINAL_FEATURES.items():
            print(f"  Testing {feature_name}...")
            kw_result = kruskal_wallis_analysis(df, feature_orig, task)
            if kw_result:
                results['kruskal_wallis'].append(kw_result)
                print(f"    H={kw_result['H_statistic']:.4f}, p={kw_result['p_value']:.4f}, "
                      f"ε²={kw_result['epsilon_squared']:.4f} ({kw_result['effect_size']})")
                
                # If significant, run pairwise tests
                if kw_result['p_value'] < 0.05:
                    print(f"    Running pairwise Mann-Whitney U tests...")
                    pw_results = pairwise_mann_whitney(df, feature_orig, task)
                    results['mann_whitney_pairwise'].extend(pw_results)
                    
                    sig_pairs = [r for r in pw_results if r['significant']]
                    print(f"    {len(sig_pairs)}/{len(pw_results)} pairs significant after Bonferroni")
    
    print("\n" + "="*80)
    print("CHI-SQUARE TESTS (Binary Features)")
    print("="*80)
    
    for task in tasks:
        print(f"\n{task}:")
        for feature_orig, feature_name in BINARY_FEATURES.items():
            print(f"  Testing {feature_name}...")
            chi_result = chi_square_analysis(df, feature_orig, task)
            results['chi_square'].append(chi_result)
            print(f"    χ²={chi_result['chi2_statistic']:.4f}, p={chi_result['p_value']:.4f}, "
                  f"V={chi_result['cramers_v']:.4f} ({chi_result['effect_size']})")
    
    return results


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

def save_results(results: dict, output_dir: Path):
    """Save all results to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Kruskal-Wallis results
    if results['kruskal_wallis']:
        kw_df = pd.DataFrame(results['kruskal_wallis'])
        kw_df['p_formatted'] = kw_df['p_value'].apply(format_pvalue)
        
        csv_path = output_dir / 'kruskal_wallis_tests.csv'
        kw_df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"\n✓ Saved: {csv_path}")
    
    # 2. Mann-Whitney pairwise results
    if results['mann_whitney_pairwise']:
        mw_df = pd.DataFrame(results['mann_whitney_pairwise'])
        mw_df['p_formatted'] = mw_df['p_value'].apply(format_pvalue)
        
        csv_path = output_dir / 'mann_whitney_pairwise_tests.csv'
        mw_df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"✓ Saved: {csv_path}")
    
    # 3. Chi-square results
    if results['chi_square']:
        chi_df = pd.DataFrame(results['chi_square'])
        # Remove contingency table for CSV (too complex)
        chi_df_csv = chi_df.drop(columns=['contingency_table'])
        chi_df_csv['p_formatted'] = chi_df_csv['p_value'].apply(format_pvalue)
        
        csv_path = output_dir / 'chi_square_tests.csv'
        chi_df_csv.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"✓ Saved: {csv_path}")
    
    # 4. Complete JSON with all details
    json_path = output_dir / 'statistical_analysis_complete.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Saved: {json_path}")
    
    # 5. Summary report
    create_summary_report(results, output_dir)


def create_summary_report(results: dict, output_dir: Path):
    """Create a human-readable summary report."""
    
    report_path = output_dir / 'statistical_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL ANALYSIS SUMMARY - TCGA-BRCA\n")
        f.write("="*80 + "\n\n")
        
        # Kruskal-Wallis summary
        f.write("KRUSKAL-WALLIS TESTS (Ordinal Features)\n")
        f.write("-"*80 + "\n\n")
        
        for result in results['kruskal_wallis']:
            f.write(f"Task: {result['Task']}\n")
            f.write(f"Feature: {result['Feature']}\n")
            f.write(f"  H-statistic: {result['H_statistic']:.4f}\n")
            f.write(f"  p-value: {format_pvalue(result['p_value'])}\n")
            f.write(f"  Epsilon-squared: {result['epsilon_squared']:.4f}\n")
            f.write(f"  Effect size: {result['effect_size']}\n")
            f.write(f"  Sample size: n={result['n_total']}, k={result['n_classes']} classes\n")
            f.write(f"  Classes: {', '.join(result['classes'])}\n")
            f.write("\n")
        
        # Mann-Whitney summary (only significant)
        f.write("\n" + "="*80 + "\n")
        f.write("SIGNIFICANT PAIRWISE COMPARISONS (Mann-Whitney U)\n")
        f.write("-"*80 + "\n\n")
        
        sig_mw = [r for r in results['mann_whitney_pairwise'] if r['significant']]
        
        if sig_mw:
            for result in sig_mw:
                f.write(f"Task: {result['Task']}, Feature: {result['Feature']}\n")
                f.write(f"  {result['Class_1']} vs {result['Class_2']}\n")
                f.write(f"  U-statistic: {result['U_statistic']:.4f}\n")
                f.write(f"  p-value: {format_pvalue(result['p_value'])}\n")
                f.write(f"  Rank-biserial: {result['rank_biserial']:.4f}\n")
                f.write(f"  Effect size: {result['effect_size']}\n")
                f.write("\n")
        else:
            f.write("No significant pairwise comparisons after Bonferroni correction.\n\n")
        
        # Chi-square summary
        f.write("\n" + "="*80 + "\n")
        f.write("CHI-SQUARE TESTS (Binary Features)\n")
        f.write("-"*80 + "\n\n")
        
        for result in results['chi_square']:
            f.write(f"Task: {result['Task']}\n")
            f.write(f"Feature: {result['Feature']}\n")
            f.write(f"  χ²-statistic: {result['chi2_statistic']:.4f}\n")
            f.write(f"  p-value: {format_pvalue(result['p_value'])}\n")
            f.write(f"  Cramér's V: {result['cramers_v']:.4f}\n")
            f.write(f"  Effect size: {result['effect_size']}\n")
            f.write(f"  Sample size: n={result['n_total']}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Saved: {report_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - TCGA-BRCA")
    print("="*80)
    
    # Load data
    df = load_tcga_data(EXCEL_PATH)
    
    # Run comprehensive analysis
    results = run_comprehensive_analysis(df)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_results(results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}/\n")


if __name__ == '__main__':
    main()
