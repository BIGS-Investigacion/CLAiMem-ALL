#!/usr/bin/env python3
"""
Kruskal-Wallis analysis for TCGA-BRCA patches with results in table format.
Analyzes histomorphological features across PAM50 subtypes and IHC biomarkers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
from itertools import combinations


def load_annotations(annotation_path: Path) -> pd.DataFrame:
    """Load and prepare annotation data."""
    df = pd.read_excel(annotation_path)

    # Rename columns to English
    column_mapping = {
        'ETIQUETA': 'Class',
        'IMAGEN': 'Image',
        'DISTANCIA': 'Distance',
        'ESTRUCTURA GLANDULAR': 'Tubule_Formation',
        'ATIPIA NUCLEAR': 'Nuclear_Pleomorphism',
        'MITOSIS': 'Mitotic_Activity',
        'NECROSIS': 'Tumor_Necrosis',
        'INFILTRADO_LI': 'Lymphocytic_Infiltrate',
        'INFILTRADO_PMN': 'Polymorphonuclear_Infiltrate'
    }
    df = df.rename(columns=column_mapping)

    return df


def compute_kruskal_wallis(df: pd.DataFrame, feature: str, classes: List[str]) -> Dict:
    """
    Compute Kruskal-Wallis test for a feature across classes.

    Returns:
        Dictionary with H-statistic, p-value, and effect size (epsilon-squared)
    """
    # Get data for each class
    groups = [df[df['Class'] == cls][feature].dropna().values for cls in classes]

    # Remove empty groups
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return None

    # Kruskal-Wallis test
    H, p_value = stats.kruskal(*groups)

    # Compute epsilon-squared (effect size)
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    epsilon_squared = (H - k + 1) / (n_total - k)

    return {
        'H_statistic': H,
        'p_value': p_value,
        'epsilon_squared': epsilon_squared,
        'n_groups': k,
        'n_total': n_total
    }


def create_kruskal_wallis_table(df: pd.DataFrame, features: List[str],
                                 classes: List[str], task_name: str) -> pd.DataFrame:
    """
    Create comprehensive Kruskal-Wallis table for all features.
    """
    results = []

    for feature in features:
        result = compute_kruskal_wallis(df, feature, classes)

        if result is not None:
            # Significance markers
            p = result['p_value']
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            # Effect size interpretation
            eps2 = result['epsilon_squared']
            if eps2 < 0.01:
                effect = 'Negligible'
            elif eps2 < 0.06:
                effect = 'Small'
            elif eps2 < 0.14:
                effect = 'Medium'
            else:
                effect = 'Large'

            results.append({
                'Feature': feature,
                'H': f"{result['H_statistic']:.3f}",
                'p-value': f"{p:.2e}",
                'ε²': f"{eps2:.3f}",
                'Effect_Size': effect,
                'Sig': sig,
                'n_groups': result['n_groups'],
                'n_total': result['n_total']
            })

    df_results = pd.DataFrame(results)
    return df_results


def create_descriptive_stats_table(df: pd.DataFrame, feature: str,
                                   classes: List[str]) -> pd.DataFrame:
    """
    Create descriptive statistics table for a feature across classes.
    """
    stats_list = []

    for cls in classes:
        data = df[df['Class'] == cls][feature].dropna()

        if len(data) > 0:
            stats_list.append({
                'Class': cls,
                'N': len(data),
                'Mean': f"{data.mean():.2f}",
                'Median': f"{data.median():.2f}",
                'SD': f"{data.std():.2f}",
                'Min': f"{data.min():.2f}",
                'Max': f"{data.max():.2f}",
                'Q1': f"{data.quantile(0.25):.2f}",
                'Q3': f"{data.quantile(0.75):.2f}"
            })

    return pd.DataFrame(stats_list)


def create_frequency_table(df: pd.DataFrame, feature: str,
                           classes: List[str]) -> pd.DataFrame:
    """
    Create frequency table for binary features.
    """
    freq_list = []

    for cls in classes:
        data = df[df['Class'] == cls][feature].dropna()

        if len(data) > 0:
            n_total = len(data)
            n_positive = (data == 1).sum()
            n_negative = (data == 0).sum()
            pct_positive = 100 * n_positive / n_total

            freq_list.append({
                'Class': cls,
                'N': n_total,
                'Positive': n_positive,
                'Negative': n_negative,
                '% Positive': f"{pct_positive:.1f}%"
            })

    return pd.DataFrame(freq_list)


def posthoc_mannwhitney_bonferroni(df: pd.DataFrame, feature: str,
                                    classes: List[str]) -> pd.DataFrame:
    """
    Perform pairwise Mann-Whitney U tests with Bonferroni correction.
    """
    results = []
    pairs = list(combinations(classes, 2))
    n_comparisons = len(pairs)
    alpha = 0.05
    bonferroni_threshold = alpha / n_comparisons

    for class1, class2 in pairs:
        data1 = df[df['Class'] == class1][feature].dropna()
        data2 = df[df['Class'] == class2][feature].dropna()

        if len(data1) > 0 and len(data2) > 0:
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

            # Rank-biserial correlation (effect size)
            n1, n2 = len(data1), len(data2)
            r_rb = 1 - (2*u_stat) / (n1 * n2)

            # Significance
            is_significant = p_value < bonferroni_threshold
            if is_significant:
                if p_value < 0.001:
                    sig = '***'
                elif p_value < 0.01:
                    sig = '**'
                elif p_value < bonferroni_threshold:
                    sig = '*'
            else:
                sig = 'ns'

            results.append({
                'Class_1': class1,
                'Class_2': class2,
                'U': f"{u_stat:.1f}",
                'p-value': f"{p_value:.2e}",
                'p_bonf_threshold': f"{bonferroni_threshold:.2e}",
                'r_rb': f"{r_rb:.3f}",
                'Sig': sig
            })

    return pd.DataFrame(results)


def separate_pam50_ihc(df: pd.DataFrame):
    """
    Separate data into PAM50 and IHC biomarker groups.
    Returns PAM50 data and separate ER, HER2, PR data.
    """
    pam50_classes = ['BASAL', 'HER2-enriched', 'LUMINAL-A', 'LUMINAL-B', 'NORMAL-like']

    er_classes = ['ER-negative', 'ER-positive']
    her2_classes = ['HER2-negative', 'HER2-positive']
    pr_classes = ['PR-negative', 'PR-positive']

    df_pam50 = df[df['Class'].isin(pam50_classes)].copy()
    df_er = df[df['Class'].isin(er_classes)].copy()
    df_her2 = df[df['Class'].isin(her2_classes)].copy()
    df_pr = df[df['Class'].isin(pr_classes)].copy()

    return df_pam50, df_er, df_her2, df_pr, pam50_classes, er_classes, her2_classes, pr_classes


def main():
    # Paths
    annotation_path = Path('data/histomorfologico/representative_images_annotation.xlsx')
    output_dir = Path('results/biological_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("KRUSKAL-WALLIS ANALYSIS - TCGA-BRCA")
    print("="*80)

    # Load annotations
    print("\nLoading annotations...")
    df = load_annotations(annotation_path)
    print(f"✓ Loaded {len(df)} patches with {df['Class'].nunique()} classes")

    # Define features
    ordinal_features = ['Tubule_Formation', 'Nuclear_Pleomorphism', 'Mitotic_Activity']
    binary_features = ['Tumor_Necrosis', 'Lymphocytic_Infiltrate', 'Polymorphonuclear_Infiltrate']
    all_features = ordinal_features + binary_features

    # Separate PAM50 and IHC biomarkers
    print("\nSeparating PAM50 subtypes and IHC biomarkers...")
    df_pam50, df_er, df_her2, df_pr, pam50_classes, er_classes, her2_classes, pr_classes = separate_pam50_ihc(df)
    print(f"✓ PAM50: {len(df_pam50)} patches, {len(pam50_classes)} classes")
    print(f"✓ ER: {len(df_er)} patches, {len(er_classes)} classes")
    print(f"✓ HER2: {len(df_her2)} patches, {len(her2_classes)} classes")
    print(f"✓ PR: {len(df_pr)} patches, {len(pr_classes)} classes")

    # ========================================================================
    # PAM50 ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PAM50 SUBTYPES ANALYSIS")
    print("="*80)

    # Kruskal-Wallis table for PAM50
    print("\nComputing Kruskal-Wallis tests for PAM50...")
    df_kw_pam50 = create_kruskal_wallis_table(df_pam50, all_features, pam50_classes, 'PAM50')

    print("\nKruskal-Wallis Results - PAM50 Subtypes:")
    print(df_kw_pam50.to_string(index=False))

    # Save table
    output_csv = output_dir / 'tcga_kruskal_wallis_pam50.csv'
    df_kw_pam50.to_csv(output_csv, index=False)
    print(f"\n✓ Saved: {output_csv}")

    # Descriptive statistics for ordinal features
    print("\n" + "-"*80)
    print("Descriptive Statistics - PAM50 (Ordinal Features)")
    print("-"*80)

    for feature in ordinal_features:
        print(f"\n{feature}:")
        df_desc = create_descriptive_stats_table(df_pam50, feature, pam50_classes)
        print(df_desc.to_string(index=False))

        # Save
        output_csv = output_dir / f'tcga_pam50_descriptive_{feature.lower()}.csv'
        df_desc.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")

    # Frequency tables for binary features
    print("\n" + "-"*80)
    print("Frequency Tables - PAM50 (Binary Features)")
    print("-"*80)

    for feature in binary_features:
        print(f"\n{feature}:")
        df_freq = create_frequency_table(df_pam50, feature, pam50_classes)
        print(df_freq.to_string(index=False))

        # Save
        output_csv = output_dir / f'tcga_pam50_frequency_{feature.lower()}.csv'
        df_freq.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")

    # Post-hoc analysis for significant features
    print("\n" + "="*80)
    print("POST-HOC ANALYSIS - PAM50 (Mann-Whitney U with Bonferroni)")
    print("="*80)

    # Only perform post-hoc for ordinal features that showed significance
    significant_ordinal = df_kw_pam50[
        (df_kw_pam50['Sig'] != 'ns') &
        (df_kw_pam50['Feature'].isin(ordinal_features))
    ]['Feature'].tolist()

    for feature in significant_ordinal:
        print(f"\n{'-'*80}")
        print(f"Post-hoc pairwise comparisons for {feature}")
        print(f"{'-'*80}")

        df_posthoc = posthoc_mannwhitney_bonferroni(df_pam50, feature, pam50_classes)
        print(df_posthoc.to_string(index=False))

        # Save
        output_csv = output_dir / f'tcga_pam50_posthoc_{feature.lower()}.csv'
        df_posthoc.to_csv(output_csv, index=False)
        print(f"\n✓ Saved: {output_csv}")

        # Summary
        n_sig = (df_posthoc['Sig'] != 'ns').sum()
        n_total = len(df_posthoc)
        print(f"Significant pairs (Bonferroni): {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)")

    # ========================================================================
    # IHC BIOMARKERS ANALYSIS - ER
    # ========================================================================
    print("\n" + "="*80)
    print("IHC BIOMARKERS ANALYSIS - ER STATUS")
    print("="*80)

    # Kruskal-Wallis table for ER
    print("\nComputing Kruskal-Wallis tests for ER status...")
    df_kw_er = create_kruskal_wallis_table(df_er, all_features, er_classes, 'ER')

    print("\nKruskal-Wallis Results - ER Status:")
    print(df_kw_er.to_string(index=False))

    # Save table
    output_csv = output_dir / 'tcga_kruskal_wallis_er.csv'
    df_kw_er.to_csv(output_csv, index=False)
    print(f"\n✓ Saved: {output_csv}")

    # Descriptive statistics for ordinal features
    print("\n" + "-"*80)
    print("Descriptive Statistics - ER (Ordinal Features)")
    print("-"*80)

    for feature in ordinal_features:
        print(f"\n{feature}:")
        df_desc = create_descriptive_stats_table(df_er, feature, er_classes)
        print(df_desc.to_string(index=False))

        # Save
        output_csv = output_dir / f'tcga_er_descriptive_{feature.lower()}.csv'
        df_desc.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")

    # Frequency tables for binary features
    print("\n" + "-"*80)
    print("Frequency Tables - ER (Binary Features)")
    print("-"*80)

    for feature in binary_features:
        print(f"\n{feature}:")
        df_freq = create_frequency_table(df_er, feature, er_classes)
        print(df_freq.to_string(index=False))

        # Save
        output_csv = output_dir / f'tcga_er_frequency_{feature.lower()}.csv'
        df_freq.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")

    # ========================================================================
    # IHC BIOMARKERS ANALYSIS - HER2
    # ========================================================================
    print("\n" + "="*80)
    print("IHC BIOMARKERS ANALYSIS - HER2 STATUS")
    print("="*80)

    # Kruskal-Wallis table for HER2
    print("\nComputing Kruskal-Wallis tests for HER2 status...")
    df_kw_her2 = create_kruskal_wallis_table(df_her2, all_features, her2_classes, 'HER2')

    print("\nKruskal-Wallis Results - HER2 Status:")
    print(df_kw_her2.to_string(index=False))

    # Save table
    output_csv = output_dir / 'tcga_kruskal_wallis_her2.csv'
    df_kw_her2.to_csv(output_csv, index=False)
    print(f"\n✓ Saved: {output_csv}")

    # Descriptive statistics for ordinal features
    print("\n" + "-"*80)
    print("Descriptive Statistics - HER2 (Ordinal Features)")
    print("-"*80)

    for feature in ordinal_features:
        print(f"\n{feature}:")
        df_desc = create_descriptive_stats_table(df_her2, feature, her2_classes)
        print(df_desc.to_string(index=False))

        # Save
        output_csv = output_dir / f'tcga_her2_descriptive_{feature.lower()}.csv'
        df_desc.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")

    # Frequency tables for binary features
    print("\n" + "-"*80)
    print("Frequency Tables - HER2 (Binary Features)")
    print("-"*80)

    for feature in binary_features:
        print(f"\n{feature}:")
        df_freq = create_frequency_table(df_her2, feature, her2_classes)
        print(df_freq.to_string(index=False))

        # Save
        output_csv = output_dir / f'tcga_her2_frequency_{feature.lower()}.csv'
        df_freq.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")

    # ========================================================================
    # IHC BIOMARKERS ANALYSIS - PR
    # ========================================================================
    print("\n" + "="*80)
    print("IHC BIOMARKERS ANALYSIS - PR STATUS")
    print("="*80)

    # Kruskal-Wallis table for PR
    print("\nComputing Kruskal-Wallis tests for PR status...")
    df_kw_pr = create_kruskal_wallis_table(df_pr, all_features, pr_classes, 'PR')

    print("\nKruskal-Wallis Results - PR Status:")
    print(df_kw_pr.to_string(index=False))

    # Save table
    output_csv = output_dir / 'tcga_kruskal_wallis_pr.csv'
    df_kw_pr.to_csv(output_csv, index=False)
    print(f"\n✓ Saved: {output_csv}")

    # Descriptive statistics for ordinal features
    print("\n" + "-"*80)
    print("Descriptive Statistics - PR (Ordinal Features)")
    print("-"*80)

    for feature in ordinal_features:
        print(f"\n{feature}:")
        df_desc = create_descriptive_stats_table(df_pr, feature, pr_classes)
        print(df_desc.to_string(index=False))

        # Save
        output_csv = output_dir / f'tcga_pr_descriptive_{feature.lower()}.csv'
        df_desc.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")

    # Frequency tables for binary features
    print("\n" + "-"*80)
    print("Frequency Tables - PR (Binary Features)")
    print("-"*80)

    for feature in binary_features:
        print(f"\n{feature}:")
        df_freq = create_frequency_table(df_pr, feature, pr_classes)
        print(df_freq.to_string(index=False))

        # Save
        output_csv = output_dir / f'tcga_pr_frequency_{feature.lower()}.csv'
        df_freq.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nPAM50 Subtypes:")
    sig_pam50 = (df_kw_pam50['Sig'] != 'ns').sum()
    print(f"  Significant features (Kruskal-Wallis): {sig_pam50}/{len(df_kw_pam50)}")
    print(f"  Features with large effect (ε² ≥ 0.14): {(df_kw_pam50['Effect_Size'] == 'Large').sum()}")

    print("\nER Status:")
    sig_er = (df_kw_er['Sig'] != 'ns').sum()
    print(f"  Significant features: {sig_er}/{len(df_kw_er)}")
    print(f"  Features with large effect (ε² ≥ 0.14): {(df_kw_er['Effect_Size'] == 'Large').sum()}")

    print("\nHER2 Status:")
    sig_her2 = (df_kw_her2['Sig'] != 'ns').sum()
    print(f"  Significant features: {sig_her2}/{len(df_kw_her2)}")
    print(f"  Features with large effect (ε² ≥ 0.14): {(df_kw_her2['Effect_Size'] == 'Large').sum()}")

    print("\nPR Status:")
    sig_pr = (df_kw_pr['Sig'] != 'ns').sum()
    print(f"  Significant features: {sig_pr}/{len(df_kw_pr)}")
    print(f"  Features with large effect (ε² ≥ 0.14): {(df_kw_pr['Effect_Size'] == 'Large').sum()}")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nGenerated files in results/biological_analysis/:")
    print("  Kruskal-Wallis tables:")
    print("    - tcga_kruskal_wallis_pam50.csv")
    print("    - tcga_kruskal_wallis_er.csv")
    print("    - tcga_kruskal_wallis_her2.csv")
    print("    - tcga_kruskal_wallis_pr.csv")
    print("  Descriptive statistics (ordinal features):")
    print("    - tcga_pam50_descriptive_*.csv (3 files)")
    print("    - tcga_er_descriptive_*.csv (3 files)")
    print("    - tcga_her2_descriptive_*.csv (3 files)")
    print("    - tcga_pr_descriptive_*.csv (3 files)")
    print("  Frequency tables (binary features):")
    print("    - tcga_pam50_frequency_*.csv (3 files)")
    print("    - tcga_er_frequency_*.csv (3 files)")
    print("    - tcga_her2_frequency_*.csv (3 files)")
    print("    - tcga_pr_frequency_*.csv (3 files)")
    print("  Post-hoc analysis (PAM50 ordinal features):")
    print("    - tcga_pam50_posthoc_*.csv (3 files)")
    print()


if __name__ == '__main__':
    main()
