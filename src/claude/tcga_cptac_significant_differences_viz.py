#!/usr/bin/env python3
"""
TCGA vs CPTAC Cohort Significant Differences Visualization

Creates visualizations showing which classes have significant differences
between TCGA and CPTAC cohorts for biological features.

Two main visualizations:
1. Heatmap showing effect sizes for significant comparisons
2. Summary chart showing count of significant features per class

Author: Claude Code
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input paths
STATS_DIR = Path('results/biological_analysis/cohort_comparison')
MANN_WHITNEY_CSV = STATS_DIR / 'cohort_mann_whitney_tests.csv'
CHI_SQUARE_CSV = STATS_DIR / 'cohort_chi_square_tests.csv'

# Output directory
OUTPUT_DIR = STATS_DIR / 'figures'

# Significance threshold
ALPHA = 0.05

# Class order for visualization
CLASS_ORDER = [
    'Basal', 'Her2-enriched', 'LumA', 'LumB', 'Normal',
    'ER-positive', 'ER-negative',
    'PR-positive', 'PR-negative',
    'HER2-positive', 'HER2-negative'
]

# Feature display names
FEATURE_NAMES = {
    'ESTRUCTURA GLANDULAR': 'Glandular Structure',
    'ATIPIA NUCLEAR': 'Nuclear Atypia',
    'MITOSIS': 'Mitosis',
    'NECROSIS': 'Necrosis',
    'INFILTRADO_LI': 'Lymphocytic Infiltrate',
    'INFILTRADO_PMN': 'PMN Infiltrate'
}

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_comparison_data():
    """Load comparison data from CSV files."""
    print("\nLoading comparison data...")

    # Load ordinal features (Mann-Whitney)
    df_mw = pd.read_csv(MANN_WHITNEY_CSV)
    print(f"  Loaded {len(df_mw)} Mann-Whitney comparisons")

    # Load binary features (Chi-square)
    df_chi = pd.read_csv(CHI_SQUARE_CSV)
    print(f"  Loaded {len(df_chi)} Chi-square comparisons")

    return df_mw, df_chi


# ==============================================================================
# VISUALIZATION 1: EFFECT SIZE HEATMAP
# ==============================================================================

def create_effect_size_heatmap(df_mw, df_chi):
    """
    Create heatmap showing effect sizes for significant comparisons.

    Args:
        df_mw: Mann-Whitney results DataFrame
        df_chi: Chi-square results DataFrame
    """
    print("\nCreating effect size heatmap...")

    # Prepare data for heatmap
    all_classes = []
    all_features = []
    effect_sizes = []
    is_significant = []

    # Process ordinal features (Mann-Whitney)
    for _, row in df_mw.iterrows():
        class_name = row['Class']
        feature = row['Feature']
        p_value = row['p_value']
        effect_size = abs(row['rank_biserial'])  # Use absolute value

        all_classes.append(class_name)
        all_features.append(FEATURE_NAMES.get(feature, feature))
        effect_sizes.append(effect_size)
        is_significant.append(p_value < ALPHA)

    # Process binary features (Chi-square)
    for _, row in df_chi.iterrows():
        class_name = row['Class']
        feature = row['Feature']
        p_value = row['p_value']
        effect_size = row['cramers_v']

        all_classes.append(class_name)
        all_features.append(FEATURE_NAMES.get(feature, feature))
        effect_sizes.append(effect_size)
        is_significant.append(p_value < ALPHA)

    # Create DataFrame
    data = pd.DataFrame({
        'Class': all_classes,
        'Feature': all_features,
        'Effect_Size': effect_sizes,
        'Significant': is_significant
    })

    # Pivot to matrix form
    matrix = data.pivot(index='Feature', columns='Class', values='Effect_Size')
    sig_matrix = data.pivot(index='Feature', columns='Class', values='Significant')

    # Reorder columns
    ordered_cols = [c for c in CLASS_ORDER if c in matrix.columns]
    matrix = matrix[ordered_cols]
    sig_matrix = sig_matrix[ordered_cols]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create heatmap
    # Mask non-significant values
    masked_matrix = matrix.copy()
    # Convert sig_matrix to boolean and invert
    sig_mask = sig_matrix.fillna(False).astype(bool)
    masked_matrix[~sig_mask] = 0  # Set non-significant to 0

    sns.heatmap(
        masked_matrix,
        annot=True,
        fmt='.2f',
        cmap='magma_r',  # Inverted magma (darker = stronger)
        vmin=0,
        vmax=0.5,
        cbar_kws={'label': 'Effect Size'},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )

    ax.set_title(
        'TCGA vs CPTAC: Effect Sizes for Significant Differences\n' +
        '(Only significant comparisons shown, p < 0.05)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Molecular Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Biological Feature', fontsize=12, fontweight='bold')

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'cohort_effect_size_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()

    return matrix, sig_matrix


# ==============================================================================
# VISUALIZATION 2: SIGNIFICANT FEATURES COUNT
# ==============================================================================

def create_significant_count_chart(df_mw, df_chi):
    """
    Create bar chart showing number of significant features per class.

    Args:
        df_mw: Mann-Whitney results DataFrame
        df_chi: Chi-square results DataFrame
    """
    print("\nCreating significant features count chart...")

    # Count significant features per class
    sig_counts = {}

    # Process ordinal features
    for _, row in df_mw.iterrows():
        class_name = row['Class']
        if row['p_value'] < ALPHA:
            sig_counts[class_name] = sig_counts.get(class_name, 0) + 1

    # Process binary features
    for _, row in df_chi.iterrows():
        class_name = row['Class']
        if row['p_value'] < ALPHA:
            sig_counts[class_name] = sig_counts.get(class_name, 0) + 1

    # Convert to DataFrame and sort
    count_df = pd.DataFrame(list(sig_counts.items()), columns=['Class', 'Count'])
    count_df = count_df.sort_values('Count', ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create horizontal bar chart
    magma_colors = sns.color_palette('magma_r', n_colors=len(count_df))
    bars = ax.barh(count_df['Class'], count_df['Count'], color=magma_colors)

    # Add value labels
    for i, (idx, row) in enumerate(count_df.iterrows()):
        ax.text(
            row['Count'] + 0.1,
            i,
            f"{int(row['Count'])}",
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('Number of Significant Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Molecular Class', fontsize=12, fontweight='bold')
    ax.set_title(
        'TCGA vs CPTAC: Number of Features with Significant Differences\n' +
        'per Molecular Class (p < 0.05)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set x-axis limits
    ax.set_xlim(0, max(count_df['Count']) + 1)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'cohort_significant_count.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()

    return count_df


# ==============================================================================
# VISUALIZATION 3: COMBINED SUMMARY
# ==============================================================================

def create_combined_summary(df_mw, df_chi):
    """
    Create combined figure with both heatmap and count chart.

    Args:
        df_mw: Mann-Whitney results DataFrame
        df_chi: Chi-square results DataFrame
    """
    print("\nCreating combined summary figure...")

    # Prepare data for heatmap
    all_classes = []
    all_features = []
    effect_sizes = []
    is_significant = []

    # Process ordinal features
    for _, row in df_mw.iterrows():
        all_classes.append(row['Class'])
        all_features.append(FEATURE_NAMES.get(row['Feature'], row['Feature']))
        effect_sizes.append(abs(row['rank_biserial']))
        is_significant.append(row['p_value'] < ALPHA)

    # Process binary features
    for _, row in df_chi.iterrows():
        all_classes.append(row['Class'])
        all_features.append(FEATURE_NAMES.get(row['Feature'], row['Feature']))
        effect_sizes.append(row['cramers_v'])
        is_significant.append(row['p_value'] < ALPHA)

    # Create DataFrame
    data = pd.DataFrame({
        'Class': all_classes,
        'Feature': all_features,
        'Effect_Size': effect_sizes,
        'Significant': is_significant
    })

    # Pivot to matrix
    matrix = data.pivot(index='Feature', columns='Class', values='Effect_Size')
    sig_matrix = data.pivot(index='Feature', columns='Class', values='Significant')

    # Reorder columns
    ordered_cols = [c for c in CLASS_ORDER if c in matrix.columns]
    matrix = matrix[ordered_cols]
    sig_matrix = sig_matrix[ordered_cols]

    # Count significant features per class
    sig_counts = {}
    for _, row in df_mw.iterrows():
        if row['p_value'] < ALPHA:
            sig_counts[row['Class']] = sig_counts.get(row['Class'], 0) + 1
    for _, row in df_chi.iterrows():
        if row['p_value'] < ALPHA:
            sig_counts[row['Class']] = sig_counts.get(row['Class'], 0) + 1

    count_df = pd.DataFrame(list(sig_counts.items()), columns=['Class', 'Count'])
    count_df = count_df.sort_values('Count', ascending=True)

    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 8))

    # Left: Heatmap
    ax1 = plt.subplot(1, 2, 1)
    masked_matrix = matrix.copy()
    sig_mask = sig_matrix.fillna(False).astype(bool)
    masked_matrix[~sig_mask] = 0

    sns.heatmap(
        masked_matrix,
        annot=True,
        fmt='.2f',
        cmap='magma_r',
        vmin=0,
        vmax=0.5,
        cbar_kws={'label': 'Effect Size'},
        linewidths=0.5,
        linecolor='white',
        ax=ax1
    )

    ax1.set_title(
        'Effect Sizes for Significant Differences',
        fontsize=12,
        fontweight='bold',
        pad=15
    )
    ax1.set_xlabel('Molecular Class', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Biological Feature', fontsize=11, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)

    # Right: Bar chart
    ax2 = plt.subplot(1, 2, 2)
    magma_colors = sns.color_palette('magma_r', n_colors=len(count_df))
    ax2.barh(count_df['Class'], count_df['Count'], color=magma_colors)

    for i, (idx, row) in enumerate(count_df.iterrows()):
        ax2.text(
            row['Count'] + 0.1,
            i,
            f"{int(row['Count'])}",
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    ax2.set_xlabel('Number of Significant Features', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Molecular Class', fontsize=11, fontweight='bold')
    ax2.set_title(
        'Count of Significant Features per Class',
        fontsize=12,
        fontweight='bold',
        pad=15
    )
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.set_xlim(0, max(count_df['Count']) + 1)

    # Overall title
    fig.suptitle(
        'TCGA vs CPTAC: Significant Biological Differences by Molecular Class (p < 0.05)',
        fontsize=15,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = OUTPUT_DIR / 'cohort_combined_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()


# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

def create_summary_table(df_mw, df_chi):
    """
    Create summary table showing which features differ significantly per class.

    Args:
        df_mw: Mann-Whitney results DataFrame
        df_chi: Chi-square results DataFrame
    """
    print("\nCreating summary table...")

    # Collect significant differences
    sig_diffs = {}

    # Process ordinal features
    for _, row in df_mw.iterrows():
        if row['p_value'] < ALPHA:
            class_name = row['Class']
            feature = FEATURE_NAMES.get(row['Feature'], row['Feature'])

            if class_name not in sig_diffs:
                sig_diffs[class_name] = []

            sig_diffs[class_name].append({
                'Feature': feature,
                'Effect_Size': abs(row['rank_biserial']),
                'P_value': row['p_value'],
                'Test': 'Mann-Whitney'
            })

    # Process binary features
    for _, row in df_chi.iterrows():
        if row['p_value'] < ALPHA:
            class_name = row['Class']
            feature = FEATURE_NAMES.get(row['Feature'], row['Feature'])

            if class_name not in sig_diffs:
                sig_diffs[class_name] = []

            sig_diffs[class_name].append({
                'Feature': feature,
                'Effect_Size': row['cramers_v'],
                'P_value': row['p_value'],
                'Test': 'Chi-square'
            })

    # Create summary dataframe
    summary_rows = []
    for class_name in sorted(sig_diffs.keys()):
        for diff in sig_diffs[class_name]:
            summary_rows.append({
                'Class': class_name,
                'Feature': diff['Feature'],
                'Test': diff['Test'],
                'Effect_Size': diff['Effect_Size'],
                'P_value': diff['P_value']
            })

    summary_df = pd.DataFrame(summary_rows)

    # Save
    output_path = STATS_DIR / 'cohort_significant_differences_summary.csv'
    summary_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"✓ Saved: {output_path}")

    return summary_df


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("TCGA vs CPTAC: SIGNIFICANT DIFFERENCES VISUALIZATION")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_mw, df_chi = load_comparison_data()

    # Create visualizations
    print("\nGenerating visualizations...")
    create_effect_size_heatmap(df_mw, df_chi)
    create_significant_count_chart(df_mw, df_chi)
    create_combined_summary(df_mw, df_chi)

    # Create summary table
    summary_df = create_summary_table(df_mw, df_chi)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Significant Differences by Class")
    print("="*80)

    for class_name in sorted(summary_df['Class'].unique()):
        class_df = summary_df[summary_df['Class'] == class_name]
        print(f"\n{class_name}: {len(class_df)} significant differences")
        for _, row in class_df.iterrows():
            print(f"  - {row['Feature']}: effect size = {row['Effect_Size']:.3f}, p = {row['P_value']:.4f}")

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print()


if __name__ == '__main__':
    main()
