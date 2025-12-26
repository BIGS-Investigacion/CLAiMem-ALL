#!/usr/bin/env python3
"""
TCGA vs CPTAC Cohort Comparison by Class

Compares histomorphological features between TCGA and CPTAC for the same molecular class.
For example: Basal (TCGA) vs Basal (CPTAC), LumA (TCGA) vs LumA (CPTAC), etc.

This analysis tests whether the same molecular class shows consistent histomorphological
features across cohorts, or if there are systematic differences that could explain
generalization failure.

Statistical tests:
- Ordinal features: Mann-Whitney U test with rank-biserial correlation
- Binary features: Chi-square test with Cramér's V

Author: Claude Code
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

EXCEL_PATH = Path('data/histomorfologico/representative_images_annotation.xlsx')
OUTPUT_DIR = Path('results/biological_analysis/cohort_comparison')

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
    'ER-positive': 'ER-positive',
    'ER-negative': 'ER-negative',
    'PR-positive': 'PR-positive',
    'PR-negative': 'PR-negative',
    'HER2-positive': 'HER2-positive',
    'HER2-negative': 'HER2-negative'
}

TASK_MAPPING = {
    'Basal': 'PAM50',
    'Her2-enriched': 'PAM50',
    'LumA': 'PAM50',
    'LumB': 'PAM50',
    'Normal': 'PAM50',
    'ER-positive': 'ER',
    'ER-negative': 'ER',
    'PR-positive': 'PR',
    'PR-negative': 'PR',
    'HER2-positive': 'HER2',
    'HER2-negative': 'HER2'
}

# ==============================================================================
# EFFECT SIZE CALCULATIONS
# ==============================================================================

def rank_biserial_correlation(U, n1, n2):
    """Calculate rank-biserial correlation for Mann-Whitney U test."""
    return 1 - (2 * U) / (n1 * n2)

def cramers_v(chi2, n, r, c):
    """Calculate Cramér's V effect size for chi-square test."""
    return np.sqrt(chi2 / (n * min(r - 1, c - 1)))

def interpret_effect_size(value, effect_type):
    """Interpret effect size magnitude."""
    if effect_type == 'rank_biserial':
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

def load_data(excel_path: Path) -> pd.DataFrame:
    """Load data from both TCGA and CPTAC sheets."""
    print(f"\nLoading data from: {excel_path}")
    
    # Load TCGA
    df_tcga = pd.read_excel(excel_path, sheet_name='TCGA')
    df_tcga['Cohort'] = 'TCGA'
    print(f"  TCGA: {len(df_tcga)} samples")
    
    # Load CPTAC
    df_cptac = pd.read_excel(excel_path, sheet_name='CPTAC')
    df_cptac['Cohort'] = 'CPTAC'
    print(f"  CPTAC: {len(df_cptac)} samples")
    
    # Combine
    df = pd.concat([df_tcga, df_cptac], ignore_index=True)
    
    # Standardize class names
    df['Class'] = df['ETIQUETA'].map(CLASS_MAPPING)
    df['Task'] = df['Class'].map(TASK_MAPPING)
    
    print(f"  Total: {len(df)} samples")
    print(f"  Classes: {sorted(df['Class'].unique())}")
    
    return df

# ==============================================================================
# MANN-WHITNEY U TESTS (ORDINAL FEATURES)
# ==============================================================================

def mann_whitney_by_class(df: pd.DataFrame, feature: str) -> list:
    """
    Perform Mann-Whitney U test comparing TCGA vs CPTAC for each class.
    """
    results = []
    
    for class_name in sorted(df['Class'].unique()):
        class_df = df[df['Class'] == class_name]
        
        tcga_data = class_df[class_df['Cohort'] == 'TCGA'][feature].values
        cptac_data = class_df[class_df['Cohort'] == 'CPTAC'][feature].values
        
        if len(tcga_data) == 0 or len(cptac_data) == 0:
            continue
        
        # Mann-Whitney U test
        U, p_value = mannwhitneyu(tcga_data, cptac_data, alternative='two-sided')
        
        # Effect size
        r_rb = rank_biserial_correlation(U, len(tcga_data), len(cptac_data))
        effect_interp = interpret_effect_size(r_rb, 'rank_biserial')
        
        # Means
        mean_tcga = np.mean(tcga_data)
        mean_cptac = np.mean(cptac_data)
        
        results.append({
            'Class': class_name,
            'Task': TASK_MAPPING[class_name],
            'Feature': feature,
            'n_TCGA': len(tcga_data),
            'n_CPTAC': len(cptac_data),
            'Mean_TCGA': mean_tcga,
            'Mean_CPTAC': mean_cptac,
            'Difference': mean_cptac - mean_tcga,
            'U_statistic': U,
            'p_value': p_value,
            'rank_biserial': r_rb,
            'effect_size': effect_interp,
            'significant': p_value < 0.05
        })
    
    return results

# ==============================================================================
# CHI-SQUARE TESTS (BINARY FEATURES)
# ==============================================================================

def chi_square_by_class(df: pd.DataFrame, feature: str) -> list:
    """
    Perform chi-square test comparing TCGA vs CPTAC for each class.
    """
    results = []
    
    for class_name in sorted(df['Class'].unique()):
        class_df = df[df['Class'] == class_name]
        
        # Create contingency table: rows=cohort, cols=feature value
        contingency = pd.crosstab(class_df['Cohort'], class_df[feature])
        
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            # Need at least 2x2 for chi-square
            continue
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Effect size
        n = contingency.sum().sum()
        r, c = contingency.shape
        cramers = cramers_v(chi2, n, r, c)
        effect_interp = interpret_effect_size(cramers, 'cramers_v')
        
        # Proportions
        tcga_positive = contingency.loc['TCGA', 1] if 1 in contingency.columns else 0
        cptac_positive = contingency.loc['CPTAC', 1] if 1 in contingency.columns else 0
        n_tcga = contingency.loc['TCGA'].sum()
        n_cptac = contingency.loc['CPTAC'].sum()
        
        prop_tcga = tcga_positive / n_tcga if n_tcga > 0 else 0
        prop_cptac = cptac_positive / n_cptac if n_cptac > 0 else 0
        
        results.append({
            'Class': class_name,
            'Task': TASK_MAPPING[class_name],
            'Feature': feature,
            'n_TCGA': int(n_tcga),
            'n_CPTAC': int(n_cptac),
            'Prop_TCGA': prop_tcga,
            'Prop_CPTAC': prop_cptac,
            'Difference': prop_cptac - prop_tcga,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'cramers_v': cramers,
            'effect_size': effect_interp,
            'significant': p_value < 0.05
        })
    
    return results

# ==============================================================================
# COMPREHENSIVE ANALYSIS
# ==============================================================================

def run_comprehensive_analysis(df: pd.DataFrame) -> dict:
    """Run all statistical tests."""
    
    results = {
        'mann_whitney': [],
        'chi_square': []
    }
    
    print("\n" + "="*80)
    print("MANN-WHITNEY U TESTS (Ordinal Features: TCGA vs CPTAC by Class)")
    print("="*80)
    
    for feature_orig, feature_name in ORDINAL_FEATURES.items():
        print(f"\n{feature_name}:")
        mw_results = mann_whitney_by_class(df, feature_orig)
        results['mann_whitney'].extend(mw_results)
        
        sig_count = sum(1 for r in mw_results if r['significant'])
        print(f"  {sig_count}/{len(mw_results)} classes show significant differences")
    
    print("\n" + "="*80)
    print("CHI-SQUARE TESTS (Binary Features: TCGA vs CPTAC by Class)")
    print("="*80)
    
    for feature_orig, feature_name in BINARY_FEATURES.items():
        print(f"\n{feature_name}:")
        chi_results = chi_square_by_class(df, feature_orig)
        results['chi_square'].extend(chi_results)
        
        sig_count = sum(1 for r in chi_results if r['significant'])
        print(f"  {sig_count}/{len(chi_results)} classes show significant differences")
    
    return results

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

def save_results(results: dict, output_dir: Path):
    """Save all results to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mann-Whitney results
    if results['mann_whitney']:
        mw_df = pd.DataFrame(results['mann_whitney'])
        mw_df['p_formatted'] = mw_df['p_value'].apply(format_pvalue)
        
        csv_path = output_dir / 'cohort_mann_whitney_tests.csv'
        mw_df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"\n✓ Saved: {csv_path}")
    
    # Chi-square results
    if results['chi_square']:
        chi_df = pd.DataFrame(results['chi_square'])
        chi_df['p_formatted'] = chi_df['p_value'].apply(format_pvalue)
        
        csv_path = output_dir / 'cohort_chi_square_tests.csv'
        chi_df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"✓ Saved: {csv_path}")
    
    # Complete JSON
    json_path = output_dir / 'cohort_comparison_complete.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Saved: {json_path}")
    
    # Summary report
    create_summary_report(results, output_dir)

def create_summary_report(results: dict, output_dir: Path):
    """Create a human-readable summary report."""
    
    report_path = output_dir / 'cohort_comparison_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TCGA vs CPTAC COHORT COMPARISON BY CLASS\n")
        f.write("="*80 + "\n\n")
        
        # Ordinal features summary
        f.write("ORDINAL FEATURES (Mann-Whitney U Tests)\n")
        f.write("-"*80 + "\n\n")
        
        for result in results['mann_whitney']:
            if result['significant']:
                f.write(f"Class: {result['Class']} | Feature: {result['Feature']}\n")
                f.write(f"  TCGA: {result['Mean_TCGA']:.3f} (n={result['n_TCGA']})\n")
                f.write(f"  CPTAC: {result['Mean_CPTAC']:.3f} (n={result['n_CPTAC']})\n")
                f.write(f"  Difference: {result['Difference']:+.3f}\n")
                f.write(f"  p-value: {format_pvalue(result['p_value'])}\n")
                f.write(f"  Effect size: {result['rank_biserial']:.3f} ({result['effect_size']})\n")
                f.write("\n")
        
        # Binary features summary
        f.write("\n" + "="*80 + "\n")
        f.write("BINARY FEATURES (Chi-Square Tests)\n")
        f.write("-"*80 + "\n\n")
        
        for result in results['chi_square']:
            if result['significant']:
                f.write(f"Class: {result['Class']} | Feature: {result['Feature']}\n")
                f.write(f"  TCGA: {result['Prop_TCGA']:.3f} (n={result['n_TCGA']})\n")
                f.write(f"  CPTAC: {result['Prop_CPTAC']:.3f} (n={result['n_CPTAC']})\n")
                f.write(f"  Difference: {result['Difference']:+.3f}\n")
                f.write(f"  p-value: {format_pvalue(result['p_value'])}\n")
                f.write(f"  Effect size: {result['cramers_v']:.3f} ({result['effect_size']})\n")
                f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Saved: {report_path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("TCGA vs CPTAC COHORT COMPARISON BY CLASS")
    print("="*80)
    
    # Load data
    df = load_data(EXCEL_PATH)
    
    # Run analysis
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
