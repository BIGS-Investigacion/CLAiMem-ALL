#!/usr/bin/env python3
"""
Visualize Pairwise Comparison Results

Creates heatmaps and summary visualizations for:
- PAM50: All pairwise comparisons between all class-cohort combinations
- IHC: Same-class cohort comparisons

Author: Claude Code
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input paths
PAM50_DIR = Path('results/biological_analysis/pairwise_cohort_analysis/pam50')
IHC_DIR = Path('results/biological_analysis/pairwise_cohort_analysis/ihc')

# Output directories
PAM50_FIG_DIR = PAM50_DIR / 'figures'
IHC_FIG_DIR = IHC_DIR / 'figures'

ALPHA = 0.05

# Feature names
FEATURE_NAMES = {
    'ESTRUCTURA GLANDULAR': 'Glandular\nStructure',
    'ATIPIA NUCLEAR': 'Nuclear\nAtypia',
    'MITOSIS': 'Mitosis',
    'NECROSIS': 'Necrosis',
    'INFILTRADO_LI': 'Lymphocytic\nInfiltrate',
    'INFILTRADO_PMN': 'PMN\nInfiltrate'
}

# ==============================================================================
# PAM50 VISUALIZATIONS
# ==============================================================================

def visualize_pam50():
    """Create visualizations for PAM50 pairwise comparisons."""
    print("\n" + "="*80)
    print("PAM50 VISUALIZATIONS")
    print("="*80)

    PAM50_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_ordinal = pd.read_csv(PAM50_DIR / 'pam50_pairwise_ordinal.csv')
    df_binary = pd.read_csv(PAM50_DIR / 'pam50_pairwise_binary.csv')

    # Combine both
    df_ordinal['Effect_Size'] = df_ordinal['effect_size_abs']
    df_binary['Effect_Size'] = df_binary['cramers_v']

    df_all = pd.concat([
        df_ordinal[['Feature', 'Class_Cohort_1', 'Class_Cohort_2', 'p_value', 'Effect_Size', 'significant']],
        df_binary[['Feature', 'Class_Cohort_1', 'Class_Cohort_2', 'p_value', 'Effect_Size', 'significant']]
    ])

    # Create heatmap for each feature
    print("\nCreating per-feature heatmaps...")
    for feature in df_all['Feature'].unique():
        create_pam50_feature_heatmap(df_all, feature)

    # Create summary heatmap (mean effect size across features)
    print("\nCreating summary heatmap...")
    create_pam50_summary_heatmap(df_all)

    # Create count of significant differences
    print("\nCreating significance count heatmap...")
    create_pam50_significance_count(df_all)


def create_pam50_feature_heatmap(df, feature):
    """Create heatmap for a single feature showing all pairwise comparisons."""
    df_feat = df[df['Feature'] == feature].copy()

    # Get unique class-cohort combinations
    combinations = sorted(set(df_feat['Class_Cohort_1'].unique()) | set(df_feat['Class_Cohort_2'].unique()))

    # Create empty matrix
    n = len(combinations)
    matrix = np.zeros((n, n))
    sig_matrix = np.zeros((n, n), dtype=bool)

    combo_to_idx = {combo: i for i, combo in enumerate(combinations)}

    # Fill matrix
    for _, row in df_feat.iterrows():
        i = combo_to_idx[row['Class_Cohort_1']]
        j = combo_to_idx[row['Class_Cohort_2']]

        # Symmetric matrix
        matrix[i, j] = row['Effect_Size']
        matrix[j, i] = row['Effect_Size']
        sig_matrix[i, j] = row['significant']
        sig_matrix[j, i] = row['significant']

    # Mask non-significant and lower triangle
    mask = np.tril(np.ones_like(matrix, dtype=bool), k=0)  # Mask diagonal and lower
    masked_matrix = matrix.copy()
    masked_matrix[~sig_matrix] = 0  # Mask non-significant
    masked_matrix[mask] = np.nan  # Mask lower triangle

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        masked_matrix,
        annot=True,
        fmt='.2f',
        cmap='magma_r',
        vmin=0,
        vmax=0.8,
        cbar_kws={'label': 'Effect Size'},
        xticklabels=combinations,
        yticklabels=combinations,
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax,
        mask=np.isnan(masked_matrix)
    )

    feat_name = FEATURE_NAMES.get(feature, feature)
    ax.set_title(
        f'PAM50: Pairwise Effect Sizes - {feat_name}\n(Upper triangle, significant only)',
        fontsize=13,
        fontweight='bold',
        pad=15
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()

    output_path = PAM50_FIG_DIR / f'pam50_{feature.lower().replace(" ", "_")}_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path.name}")
    plt.close()


def create_pam50_summary_heatmap(df):
    """Create summary heatmap showing mean effect size across all features."""
    # Get unique class-cohort combinations
    combinations = sorted(set(df['Class_Cohort_1'].unique()) | set(df['Class_Cohort_2'].unique()))

    n = len(combinations)
    combo_to_idx = {combo: i for i, combo in enumerate(combinations)}

    # Aggregate by pair (mean effect size where significant)
    pair_stats = df[df['significant']].groupby(['Class_Cohort_1', 'Class_Cohort_2']).agg({
        'Effect_Size': 'mean',
        'significant': 'count'  # Count of significant features
    }).reset_index()

    # Create matrices
    matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))

    for _, row in pair_stats.iterrows():
        i = combo_to_idx[row['Class_Cohort_1']]
        j = combo_to_idx[row['Class_Cohort_2']]

        matrix[i, j] = row['Effect_Size']
        matrix[j, i] = row['Effect_Size']
        count_matrix[i, j] = row['significant']
        count_matrix[j, i] = row['significant']

    # Mask lower triangle
    mask = np.tril(np.ones_like(matrix, dtype=bool), k=0)
    masked_matrix = matrix.copy()
    masked_matrix[mask] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        masked_matrix,
        annot=True,
        fmt='.2f',
        cmap='magma_r',
        vmin=0,
        vmax=0.8,
        cbar_kws={'label': 'Mean Effect Size'},
        xticklabels=combinations,
        yticklabels=combinations,
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax,
        mask=np.isnan(masked_matrix)
    )

    ax.set_title(
        'PAM50: Mean Effect Size Across All Features\n(Upper triangle, significant comparisons only)',
        fontsize=14,
        fontweight='bold',
        pad=15
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()

    output_path = PAM50_FIG_DIR / 'pam50_summary_mean_effect_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path.name}")
    plt.close()


def create_pam50_significance_count(df):
    """Create heatmap showing count of significant features per pair."""
    combinations = sorted(set(df['Class_Cohort_1'].unique()) | set(df['Class_Cohort_2'].unique()))

    n = len(combinations)
    combo_to_idx = {combo: i for i, combo in enumerate(combinations)}

    # Count significant features per pair
    pair_counts = df[df['significant']].groupby(['Class_Cohort_1', 'Class_Cohort_2']).size().reset_index(name='count')

    # Create matrix
    matrix = np.zeros((n, n))

    for _, row in pair_counts.iterrows():
        i = combo_to_idx[row['Class_Cohort_1']]
        j = combo_to_idx[row['Class_Cohort_2']]

        matrix[i, j] = row['count']
        matrix[j, i] = row['count']

    # Mask lower triangle
    mask = np.tril(np.ones_like(matrix, dtype=bool), k=0)
    masked_matrix = matrix.copy()
    masked_matrix[mask] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        masked_matrix,
        annot=True,
        fmt='.0f',
        cmap='magma_r',
        cbar_kws={'label': 'Number of Significant Features'},
        xticklabels=combinations,
        yticklabels=combinations,
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax,
        mask=np.isnan(masked_matrix)
    )

    ax.set_title(
        'PAM50: Number of Significant Features per Pair\n(Upper triangle)',
        fontsize=14,
        fontweight='bold',
        pad=15
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()

    output_path = PAM50_FIG_DIR / 'pam50_significance_count.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path.name}")
    plt.close()


# ==============================================================================
# IHC VISUALIZATIONS
# ==============================================================================

def visualize_ihc():
    """Create visualizations for IHC same-class cohort comparisons."""
    print("\n" + "="*80)
    print("IHC VISUALIZATIONS")
    print("="*80)

    IHC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_ordinal = pd.read_csv(IHC_DIR / 'ihc_cohort_ordinal.csv')
    df_binary = pd.read_csv(IHC_DIR / 'ihc_cohort_binary.csv')

    # Combine
    df_ordinal['Effect_Size'] = df_ordinal['effect_size_abs']
    df_binary['Effect_Size'] = df_binary['cramers_v']

    df_all = pd.concat([
        df_ordinal[['Feature', 'Class', 'p_value', 'Effect_Size', 'significant']],
        df_binary[['Feature', 'Class', 'p_value', 'Effect_Size', 'significant']]
    ])

    # Create heatmap
    print("\nCreating IHC heatmap...")
    create_ihc_heatmap(df_all)

    # Create bar chart
    print("\nCreating IHC bar chart...")
    create_ihc_bar_chart(df_all)


def create_ihc_heatmap(df):
    """Create heatmap showing effect sizes for IHC comparisons."""
    # Pivot to matrix
    matrix = df.pivot(index='Feature', columns='Class', values='Effect_Size')
    sig_matrix = df.pivot(index='Feature', columns='Class', values='significant')

    # Mask non-significant
    masked_matrix = matrix.copy()
    sig_mask = sig_matrix.fillna(False).astype(bool)
    masked_matrix[~sig_mask] = 0

    # Rename features
    feature_labels = [FEATURE_NAMES.get(f, f) for f in matrix.index]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        masked_matrix,
        annot=True,
        fmt='.2f',
        cmap='magma_r',
        vmin=0,
        vmax=0.5,
        cbar_kws={'label': 'Effect Size'},
        yticklabels=feature_labels,
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )

    ax.set_title(
        'IHC: TCGA vs CPTAC Effect Sizes by Class\n(Significant comparisons only)',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Biological Feature', fontsize=11, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()

    output_path = IHC_FIG_DIR / 'ihc_cohort_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path.name}")
    plt.close()


def create_ihc_bar_chart(df):
    """Create bar chart showing count of significant features per class."""
    # Count significant per class
    sig_counts = df[df['significant']].groupby('Class').size().reset_index(name='count')
    sig_counts = sig_counts.sort_values('count', ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    magma_colors = sns.color_palette('magma_r', n_colors=len(sig_counts))

    bars = ax.barh(sig_counts['Class'], sig_counts['count'], color=magma_colors)

    # Add value labels
    for i, (idx, row) in enumerate(sig_counts.iterrows()):
        ax.text(
            row['count'] + 0.05,
            i,
            f"{int(row['count'])}",
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('Number of Significant Features', fontsize=11, fontweight='bold')
    ax.set_ylabel('Class', fontsize=11, fontweight='bold')
    ax.set_title(
        'IHC: Number of Features with Significant TCGA-CPTAC Differences',
        fontsize=12,
        fontweight='bold',
        pad=15
    )

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(sig_counts['count']) + 1)

    plt.tight_layout()

    output_path = IHC_FIG_DIR / 'ihc_cohort_bar_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path.name}")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("VISUALIZING PAIRWISE COMPARISON RESULTS")
    print("="*80)

    # PAM50 visualizations
    visualize_pam50()

    # IHC visualizations
    visualize_ihc()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
