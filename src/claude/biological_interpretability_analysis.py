#!/usr/bin/env python3
"""
==============================================================================
BIOLOGICAL INTERPRETABILITY AND CROSS-COHORT MORPHOLOGICAL ANALYSIS
==============================================================================

This script assesses whether model predictions are grounded in biologically
meaningful patterns by analyzing histomorphological features of high-attention
patches and comparing them across cohorts.

Methodology:
1. Load histomorphological annotations for representative patches from both cohorts
2. Perform statistical tests to assess:
   a) Intra-cohort: Do molecular classes show distinct histomorphology?
   b) Inter-cohort: Do corresponding classes differ morphologically between cohorts?
3. Compute effect sizes (ε², rank-biserial, Cramér's V)
4. Generate visualizations and interpretable reports

Key Questions:
- Does the model attend to histomorphologically meaningful regions?
- Are there systematic morphological differences between cohorts?

Statistical Tests:
- Ordinal features: Kruskal-Wallis + Mann-Whitney U (effect size: ε²,  rank-biserial)
- Binary features: Chi-square (effect size: Cramér's V)

Author: Claude Code
Date: 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from itertools import combinations


# ==============================================================================
# CONFIGURATION
# ==============================================================================

ORDINAL_FEATURES = [
    'ESTRUCTURA GLANDULAR',  # Tubule Formation [0-4]
    'ATIPIA NUCLEAR',         # Nuclear Pleomorphism [0-4]
    'MITOSIS'                 # Mitotic Activity [0-4]
]

BINARY_FEATURES = [
    'NECROSIS',              # Tumor Necrosis [0-1]
    'INFILTRADO_LI',         # Lymphocytic Infiltrate [0-1]
    'INFILTRADO_PMN'         # Polymorphonuclear Infiltrate [0-1]
]

ALL_FEATURES = ORDINAL_FEATURES + BINARY_FEATURES

FEATURE_LABELS = {
    'ESTRUCTURA GLANDULAR': 'Tubule Formation',
    'ATIPIA NUCLEAR': 'Nuclear Pleomorphism',
    'MITOSIS': 'Mitotic Activity',
    'NECROSIS': 'Tumor Necrosis',
    'INFILTRADO_LI': 'Lymphocytic Infiltrate',
    'INFILTRADO_PMN': 'Polymorphonuclear Infiltrate'
}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_annotations(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load histomorphological annotations from Excel file.

    Expected columns:
        - ETIQUETA: class label (e.g., 'PAM50_LumA', 'ER-positive')
        - ESTRUCTURA GLANDULAR: [0-4]
        - ATIPIA NUCLEAR: [0-4]
        - MITOSIS: [0-4]
        - NECROSIS: [0-1]
        - INFILTRADO_LI: [0-1]
        - INFILTRADO_PMN: [0-1]

    Args:
        excel_path: Path to Excel file
        sheet_name: Sheet name ('TCGA' or 'CPTAC')

    Returns:
        DataFrame with annotations
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Validate required columns
    required = ['ETIQUETA'] + ALL_FEATURES
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# ==============================================================================
# STATISTICAL ANALYSIS: INTRA-COHORT
# ==============================================================================

def kruskal_wallis_test(df: pd.DataFrame, feature: str,
                        class_col: str = 'ETIQUETA') -> Dict:
    """
    Perform Kruskal-Wallis H-test for ordinal features across multiple classes.

    H₀: All classes have the same distribution
    H₁: At least one class differs

    Args:
        df: DataFrame with annotations
        feature: Feature name (ordinal)
        class_col: Column name for class labels

    Returns:
        Dictionary with test results and effect size
    """
    classes = df[class_col].unique()
    groups = [df[df[class_col] == c][feature].dropna() for c in classes]

    # Remove empty groups
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return {
            'test': 'kruskal_wallis',
            'statistic': np.nan,
            'p_value': np.nan,
            'epsilon_squared': np.nan,
            'n_classes': len(groups),
            'warning': 'Insufficient groups for test'
        }

    # Kruskal-Wallis test
    H, p = stats.kruskal(*groups)

    # Epsilon-squared effect size
    n = sum(len(g) for g in groups)
    k = len(groups)
    epsilon_sq = (H - k + 1) / (n - k) if n > k else np.nan

    return {
        'test': 'kruskal_wallis',
        'statistic': float(H),
        'p_value': float(p),
        'epsilon_squared': float(epsilon_sq),
        'n_classes': k,
        'n_total': n
    }


def mann_whitney_u_test(df: pd.DataFrame, feature: str,
                        class1: str, class2: str,
                        class_col: str = 'ETIQUETA') -> Dict:
    """
    Perform Mann-Whitney U test for pairwise comparison.

    Args:
        df: DataFrame with annotations
        feature: Feature name
        class1, class2: Classes to compare
        class_col: Column name for class labels

    Returns:
        Dictionary with test results and rank-biserial correlation
    """
    group1 = df[df[class_col] == class1][feature].dropna()
    group2 = df[df[class_col] == class2][feature].dropna()

    if len(group1) == 0 or len(group2) == 0:
        return {
            'test': 'mann_whitney',
            'class1': class1,
            'class2': class2,
            'statistic': np.nan,
            'p_value': np.nan,
            'rank_biserial': np.nan,
            'warning': 'Empty group(s)'
        }

    # Mann-Whitney U test
    U, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    # Rank-biserial correlation (effect size)
    n1, n2 = len(group1), len(group2)
    rank_biserial = 1 - (2*U) / (n1 * n2)

    return {
        'test': 'mann_whitney',
        'class1': class1,
        'class2': class2,
        'statistic': float(U),
        'p_value': float(p),
        'rank_biserial': float(rank_biserial),
        'n1': n1,
        'n2': n2
    }


def chi_square_test(df: pd.DataFrame, feature: str,
                   class_col: str = 'ETIQUETA') -> Dict:
    """
    Perform Chi-square test for binary features.

    Args:
        df: DataFrame with annotations
        feature: Feature name (binary)
        class_col: Column name for class labels

    Returns:
        Dictionary with test results and Cramér's V
    """
    # Create contingency table
    ct = pd.crosstab(df[class_col], df[feature])

    # Chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(ct)

    # Cramér's V effect size
    n = ct.sum().sum()
    min_dim = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    return {
        'test': 'chi_square',
        'statistic': float(chi2),
        'p_value': float(p),
        'cramers_v': float(cramers_v),
        'dof': int(dof),
        'n_total': int(n)
    }


def analyze_intra_cohort(df: pd.DataFrame, cohort_name: str,
                         task: str) -> Dict:
    """
    Analyze histomorphological differences within a cohort across molecular classes.

    Args:
        df: DataFrame with annotations
        cohort_name: 'TCGA' or 'CPTAC'
        task: Task name for grouping results

    Returns:
        Dictionary with all test results
    """
    results = {
        'cohort': cohort_name,
        'task': task,
        'n_samples': len(df),
        'n_classes': df['ETIQUETA'].nunique(),
        'ordinal_features': {},
        'binary_features': {},
        'pairwise_comparisons': {}
    }

    # Ordinal features: Kruskal-Wallis
    print(f"\n  Ordinal features (Kruskal-Wallis):")
    for feature in ORDINAL_FEATURES:
        if feature not in df.columns:
            continue

        result = kruskal_wallis_test(df, feature)
        results['ordinal_features'][feature] = result

        sig = "✓" if result['p_value'] < 0.05 else "✗"
        print(f"    {sig} {FEATURE_LABELS[feature]}: "
              f"H={result['statistic']:.2f}, p={result['p_value']:.4f}, "
              f"ε²={result['epsilon_squared']:.3f}")

    # Binary features: Chi-square
    print(f"\n  Binary features (Chi-square):")
    for feature in BINARY_FEATURES:
        if feature not in df.columns:
            continue

        result = chi_square_test(df, feature)
        results['binary_features'][feature] = result

        sig = "✓" if result['p_value'] < 0.05 else "✗"
        print(f"    {sig} {FEATURE_LABELS[feature]}: "
              f"χ²={result['statistic']:.2f}, p={result['p_value']:.4f}, "
              f"V={result['cramers_v']:.3f}")

    # Pairwise comparisons for significant ordinal features
    for feature in ORDINAL_FEATURES:
        if feature not in results['ordinal_features']:
            continue

        if results['ordinal_features'][feature]['p_value'] < 0.05:
            print(f"\n  Post-hoc pairwise tests for {FEATURE_LABELS[feature]}:")

            classes = df['ETIQUETA'].unique()
            pairwise = []

            for class1, class2 in combinations(classes, 2):
                result = mann_whitney_u_test(df, feature, class1, class2)
                pairwise.append(result)

                if result['p_value'] < 0.05:  # Bonferroni correction should be applied
                    print(f"    • {class1} vs {class2}: "
                          f"U={result['statistic']:.1f}, p={result['p_value']:.4f}, "
                          f"r_rb={result['rank_biserial']:.3f}")

            results['pairwise_comparisons'][feature] = pairwise

    return results


# ==============================================================================
# STATISTICAL ANALYSIS: INTER-COHORT
# ==============================================================================

def analyze_inter_cohort(df_tcga: pd.DataFrame, df_cptac: pd.DataFrame,
                         task: str) -> Dict:
    """
    Compare histomorphological features between TCGA and CPTAC for each class.

    Args:
        df_tcga: TCGA annotations
        df_cptac: CPTAC annotations
        task: Task name

    Returns:
        Dictionary with comparison results
    """
    results = {
        'task': task,
        'per_class_comparisons': {}
    }

    # Get common classes
    classes_tcga = set(df_tcga['ETIQUETA'].unique())
    classes_cptac = set(df_cptac['ETIQUETA'].unique())
    common_classes = classes_tcga & classes_cptac

    print(f"\n  Comparing {len(common_classes)} common classes between cohorts...")

    for class_label in common_classes:
        df_tcga_class = df_tcga[df_tcga['ETIQUETA'] == class_label]
        df_cptac_class = df_cptac[df_cptac['ETIQUETA'] == class_label]

        class_results = {
            'class': class_label,
            'n_tcga': len(df_tcga_class),
            'n_cptac': len(df_cptac_class),
            'ordinal_features': {},
            'binary_features': {}
        }

        # Ordinal features: Mann-Whitney U
        for feature in ORDINAL_FEATURES:
            if feature not in df_tcga_class.columns or feature not in df_cptac_class.columns:
                continue

            group_tcga = df_tcga_class[feature].dropna()
            group_cptac = df_cptac_class[feature].dropna()

            if len(group_tcga) == 0 or len(group_cptac) == 0:
                continue

            U, p = stats.mannwhitneyu(group_tcga, group_cptac, alternative='two-sided')

            n1, n2 = len(group_tcga), len(group_cptac)
            rank_biserial = 1 - (2*U) / (n1 * n2)

            class_results['ordinal_features'][feature] = {
                'statistic': float(U),
                'p_value': float(p),
                'rank_biserial': float(rank_biserial)
            }

        # Binary features: Chi-square
        for feature in BINARY_FEATURES:
            if feature not in df_tcga_class.columns or feature not in df_cptac_class.columns:
                continue

            # Create combined dataframe
            df_combined = pd.concat([
                df_tcga_class[[feature]].assign(cohort='TCGA'),
                df_cptac_class[[feature]].assign(cohort='CPTAC')
            ])

            ct = pd.crosstab(df_combined['cohort'], df_combined[feature])

            try:
                chi2, p, dof, expected = stats.chi2_contingency(ct)

                n = ct.sum().sum()
                min_dim = min(ct.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                class_results['binary_features'][feature] = {
                    'statistic': float(chi2),
                    'p_value': float(p),
                    'cramers_v': float(cramers_v)
                }
            except:
                pass

        results['per_class_comparisons'][class_label] = class_results

    return results


# ==============================================================================
# EFFECT SIZE SUMMARY
# ==============================================================================

def compute_biological_shift_indicator(inter_cohort_results: Dict) -> pd.DataFrame:
    """
    Compute biological shift indicator (B_c) for each class.

    B_c = mean effect size across all histomorphological features

    Args:
        inter_cohort_results: Results from analyze_inter_cohort()

    Returns:
        DataFrame with biological shift indicators
    """
    records = []

    for class_label, class_data in inter_cohort_results['per_class_comparisons'].items():
        effect_sizes = []

        # Ordinal features: rank-biserial correlation
        for feature, result in class_data['ordinal_features'].items():
            effect_sizes.append(abs(result['rank_biserial']))

        # Binary features: Cramér's V
        for feature, result in class_data['binary_features'].items():
            effect_sizes.append(result['cramers_v'])

        # Mean effect size
        b_c = np.mean(effect_sizes) if effect_sizes else np.nan

        records.append({
            'Class': class_label,
            'B_c': b_c,
            'n_features': len(effect_sizes),
            'n_tcga': class_data['n_tcga'],
            'n_cptac': class_data['n_cptac']
        })

    return pd.DataFrame(records)


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_feature_distributions(df: pd.DataFrame, feature: str,
                               cohort_name: str, task: str,
                               output_path: Optional[Path] = None) -> None:
    """
    Plot distribution of a feature across classes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by class
    classes = df['ETIQUETA'].unique()
    data = [df[df['ETIQUETA'] == c][feature].dropna() for c in classes]

    # Box plot
    bp = ax.boxplot(data, labels=classes, patch_artist=True)

    # Color boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel(FEATURE_LABELS.get(feature, feature), fontsize=12, fontweight='bold')
    ax.set_title(f'{FEATURE_LABELS.get(feature, feature)} Distribution\n{cohort_name} - {task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_effect_sizes_heatmap(intra_tcga: Dict, intra_cptac: Dict,
                              task: str, output_path: Optional[Path] = None) -> None:
    """
    Heatmap of effect sizes for intra-cohort comparisons.
    """
    # Collect effect sizes
    features = ORDINAL_FEATURES + BINARY_FEATURES
    effect_tcga = []
    effect_cptac = []

    for feature in ORDINAL_FEATURES:
        if feature in intra_tcga['ordinal_features']:
            effect_tcga.append(intra_tcga['ordinal_features'][feature]['epsilon_squared'])
        else:
            effect_tcga.append(np.nan)

        if feature in intra_cptac['ordinal_features']:
            effect_cptac.append(intra_cptac['ordinal_features'][feature]['epsilon_squared'])
        else:
            effect_cptac.append(np.nan)

    for feature in BINARY_FEATURES:
        if feature in intra_tcga['binary_features']:
            effect_tcga.append(intra_tcga['binary_features'][feature]['cramers_v'])
        else:
            effect_tcga.append(np.nan)

        if feature in intra_cptac['binary_features']:
            effect_cptac.append(intra_cptac['binary_features'][feature]['cramers_v'])
        else:
            effect_cptac.append(np.nan)

    # Create dataframe
    df = pd.DataFrame({
        'Feature': [FEATURE_LABELS[f] for f in features],
        'TCGA': effect_tcga,
        'CPTAC': effect_cptac
    })

    df = df.set_index('Feature')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(df.T, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=0.5,
                cbar_kws={'label': 'Effect Size'}, linewidths=0.5, ax=ax)

    ax.set_title(f'Effect Sizes: Histomorphological Differences Across Classes\n{task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Histomorphological Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cohort', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_biological_shift(df_shift: pd.DataFrame, task: str,
                          output_path: Optional[Path] = None) -> None:
    """
    Bar plot of biological shift indicator per class.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlGn_r(df_shift['B_c'] / df_shift['B_c'].max())

    bars = ax.bar(df_shift['Class'], df_shift['B_c'],
                  color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, df_shift['B_c']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Reference lines for effect size interpretation
    ax.axhline(y=0.10, color='green', linestyle='--', linewidth=1, alpha=0.5,
               label='Small effect (0.10)')
    ax.axhline(y=0.30, color='orange', linestyle='--', linewidth=1, alpha=0.5,
               label='Medium effect (0.30)')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Biological Shift Indicator (B_c)', fontsize=12, fontweight='bold')
    ax.set_title(f'Morphological Divergence Between Cohorts\n{task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(df_shift['Class'], rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


# ==============================================================================
# REPORTING
# ==============================================================================

def print_intra_cohort_summary(results: Dict) -> None:
    """
    Print summary of intra-cohort analysis.
    """
    print("\n" + "="*80)
    print(f"INTRA-COHORT ANALYSIS: {results['cohort']}")
    print("="*80)

    print(f"\nSamples: {results['n_samples']}")
    print(f"Classes: {results['n_classes']}")

    # Ordinal features summary
    print("\n" + "-"*80)
    print("ORDINAL FEATURES (Kruskal-Wallis)")
    print("-"*80)

    for feature, result in results['ordinal_features'].items():
        sig = "✓ SIGNIFICANT" if result['p_value'] < 0.05 else "✗ Not significant"
        print(f"\n{FEATURE_LABELS[feature]}:")
        print(f"  H-statistic: {result['statistic']:.3f}")
        print(f"  p-value:     {result['p_value']:.4f}")
        print(f"  ε²:          {result['epsilon_squared']:.3f}")
        print(f"  Result:      {sig}")

    # Binary features summary
    print("\n" + "-"*80)
    print("BINARY FEATURES (Chi-square)")
    print("-"*80)

    for feature, result in results['binary_features'].items():
        sig = "✓ SIGNIFICANT" if result['p_value'] < 0.05 else "✗ Not significant"
        print(f"\n{FEATURE_LABELS[feature]}:")
        print(f"  χ²:          {result['statistic']:.3f}")
        print(f"  p-value:     {result['p_value']:.4f}")
        print(f"  Cramér's V:  {result['cramers_v']:.3f}")
        print(f"  Result:      {sig}")


def save_results(intra_tcga: Dict, intra_cptac: Dict, inter_cohort: Dict,
                 df_shift: pd.DataFrame, output_dir: Path, task: str) -> None:
    """
    Save all results to files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save biological shift indicators
    csv_path = output_dir / f'{task}_biological_shift.csv'
    df_shift.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  ✓ Saved: {csv_path.name}")

    # Save full results as JSON
    results = {
        'task': task,
        'intra_cohort_tcga': intra_tcga,
        'intra_cohort_cptac': intra_cptac,
        'inter_cohort': inter_cohort,
        'biological_shift': df_shift.to_dict(orient='records')
    }

    json_path = output_dir / f'{task}_biological_interpretability.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: {json_path.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Biological Interpretability and Cross-Cohort Morphological Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python biological_interpretability_analysis.py \\
    --annotations data/histomorfologico/representative_images_annotation.xlsx \\
    --task pam50 \\
    --output results/biological_analysis/
        """
    )

    # Input/output
    parser.add_argument('--annotations', '-a', type=str, required=True,
                        help='Path to Excel file with histomorphological annotations')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory')

    # Task
    parser.add_argument('--task', '-t', type=str, required=True,
                        help='Task name (pam50, er, pr, her2)')

    # Options
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--plot_format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'])

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("BIOLOGICAL INTERPRETABILITY ANALYSIS")
    print("="*80)
    print(f"Task:        {args.task.upper()}")
    print(f"Annotations: {args.annotations}")
    print(f"Output:      {args.output}")
    print("="*80 + "\n")

    # Load annotations
    print("Loading annotations...")
    df_tcga = load_annotations(Path(args.annotations), 'TCGA')
    df_cptac = load_annotations(Path(args.annotations), 'CPTAC')

    print(f"  ✓ TCGA:  {len(df_tcga)} patches, {df_tcga['ETIQUETA'].nunique()} classes")
    print(f"  ✓ CPTAC: {len(df_cptac)} patches, {df_cptac['ETIQUETA'].nunique()} classes")

    # Intra-cohort analysis
    print("\n" + "="*80)
    print("INTRA-COHORT ANALYSIS (TCGA)")
    print("="*80)
    intra_tcga = analyze_intra_cohort(df_tcga, 'TCGA', args.task)

    print("\n" + "="*80)
    print("INTRA-COHORT ANALYSIS (CPTAC)")
    print("="*80)
    intra_cptac = analyze_intra_cohort(df_cptac, 'CPTAC', args.task)

    # Inter-cohort analysis
    print("\n" + "="*80)
    print("INTER-COHORT ANALYSIS (TCGA vs CPTAC)")
    print("="*80)
    inter_cohort = analyze_inter_cohort(df_tcga, df_cptac, args.task)

    # Compute biological shift indicators
    print("\nComputing biological shift indicators...")
    df_shift = compute_biological_shift_indicator(inter_cohort)
    print("  ✓ Biological shift indicators computed")

    # Print summaries
    print_intra_cohort_summary(intra_tcga)
    print_intra_cohort_summary(intra_cptac)

    print("\n" + "="*80)
    print("BIOLOGICAL SHIFT INDICATORS")
    print("="*80)
    print(df_shift.to_string(index=False))

    # Save results
    print("\n\nSaving results...")
    save_results(intra_tcga, intra_cptac, inter_cohort, df_shift,
                 output_dir, args.task)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualizations...")

        plot_effect_sizes_heatmap(
            intra_tcga, intra_cptac, args.task,
            output_dir / f'{args.task}_effect_sizes_heatmap.{args.plot_format}'
        )

        plot_biological_shift(
            df_shift, args.task,
            output_dir / f'{args.task}_biological_shift.{args.plot_format}'
        )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/\n")


if __name__ == '__main__':
    main()