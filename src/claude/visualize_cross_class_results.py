#!/usr/bin/env python3
"""
Visualize Cross-Cohort Cross-Class Comparison Results

Creates heatmaps for TCGA vs CPTAC cross-class comparisons.
Rows = TCGA classes, Columns = CPTAC classes

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
PAM50_DIR = Path('results/biological_analysis/tcga_vs_cptac_cross_class/pam50')
ER_DIR = Path('results/biological_analysis/tcga_vs_cptac_cross_class/er')
PR_DIR = Path('results/biological_analysis/tcga_vs_cptac_cross_class/pr')
HER2_DIR = Path('results/biological_analysis/tcga_vs_cptac_cross_class/her2')

# Output directories
PAM50_FIG_DIR = PAM50_DIR / 'figures'
ER_FIG_DIR = ER_DIR / 'figures'
PR_FIG_DIR = PR_DIR / 'figures'
HER2_FIG_DIR = HER2_DIR / 'figures'

ALPHA = 0.05

# Feature names
FEATURE_NAMES = {
    'ESTRUCTURA GLANDULAR': 'Glandular Structure',
    'ATIPIA NUCLEAR': 'Nuclear Atypia',
    'MITOSIS': 'Mitosis',
    'NECROSIS': 'Necrosis',
    'INFILTRADO_LI': 'Lymphocytic Infiltrate',
    'INFILTRADO_PMN': 'PMN Infiltrate'
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_class_sort_key(x):
    """
    Sort key for classes to ensure consistent ordering.
    
    Order:
    1. IHC Negative classes
    2. IHC Positive classes
    3. PAM50 classes (Basal, Her2, LumA, LumB, Normal)
    4. Others
    """
    # PAM50 classes: Basal, Her2-enriched, LumA, LumB, Normal
    pam50_order = {'Basal': 0, 'Her2-enriched': 1, 'LumA': 2, 'LumB': 3, 'Normal': 4}
    
    x_str = str(x)
    x_lower = x_str.lower()
    
    # IHC classes: negative before positive
    if 'negative' in x_lower or x_lower.strip().endswith('-'):
        return (0, x_str)  # Negative classes first
    elif 'positive' in x_lower or x_lower.strip().endswith('+'):
        return (1, x_str)  # Positive classes second (AFTER Negative)
    elif x_str in pam50_order:
        return (2, pam50_order[x_str])  # PAM50 classes in specified order
    else:
        return (3, x_str)  # Any other classes last


# ==============================================================================
# PAM50 VISUALIZATIONS
# ==============================================================================

def visualize_pam50():
    """Create visualizations for PAM50 cross-class comparisons."""
    print("\n" + "="*80)
    print("PAM50 VISUALIZATIONS")
    print("="*80)

    PAM50_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_ordinal = pd.read_csv(PAM50_DIR / 'pam50_cross_ordinal.csv')
    df_binary = pd.read_csv(PAM50_DIR / 'pam50_cross_binary.csv')

    # Combine both
    df_ordinal['Effect_Size'] = df_ordinal['effect_size_abs']
    df_binary['Effect_Size'] = df_binary['cramers_v']

    df_all = pd.concat([
        df_ordinal[['Feature', 'TCGA_Class', 'CPTAC_Class', 'p_value', 'Effect_Size', 'significant']],
        df_binary[['Feature', 'TCGA_Class', 'CPTAC_Class', 'p_value', 'Effect_Size', 'significant']]
    ])

    # Create heatmap for each feature
    print("\nCreating per-feature heatmaps...")
    for feature in df_all['Feature'].unique():
        create_feature_heatmap(df_all, feature, PAM50_FIG_DIR, 'PAM50')

    # Create summary heatmap (mean effect size across features)
    print("\nCreating summary heatmap...")
    create_summary_heatmap(df_all, PAM50_FIG_DIR, 'PAM50')

    # Create significance count heatmap
    print("\nCreating significance count heatmap...")
    create_significance_count_heatmap(df_all, PAM50_FIG_DIR, 'PAM50')


# ==============================================================================
# IHC VISUALIZATIONS
# ==============================================================================

def visualize_task(task_dir, fig_dir, task_name):
    """Create visualizations for a specific task."""
    print("\n" + "="*80)
    print(f"{task_name} VISUALIZATIONS")
    print("="*80)

    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_ordinal = pd.read_csv(task_dir / f'{task_name.lower()}_cross_ordinal.csv')
    df_binary = pd.read_csv(task_dir / f'{task_name.lower()}_cross_binary.csv')

    # Combine both
    df_ordinal['Effect_Size'] = df_ordinal['effect_size_abs']
    df_binary['Effect_Size'] = df_binary['cramers_v']

    df_all = pd.concat([
        df_ordinal[['Feature', 'TCGA_Class', 'CPTAC_Class', 'p_value', 'Effect_Size', 'significant']],
        df_binary[['Feature', 'TCGA_Class', 'CPTAC_Class', 'p_value', 'Effect_Size', 'significant']]
    ])

    # Create heatmap for each feature
    print("\nCreating per-feature heatmaps...")
    for feature in df_all['Feature'].unique():
        create_feature_heatmap(df_all, feature, fig_dir, task_name)

    # Create summary heatmap
    print("\nCreating summary heatmap...")
    create_summary_heatmap(df_all, fig_dir, task_name)

    # Create significance count heatmap
    print("\nCreating significance count heatmap...")
    create_significance_count_heatmap(df_all, fig_dir, task_name)


# ==============================================================================
# HEATMAP CREATION FUNCTIONS
# ==============================================================================

def create_feature_heatmap(df, feature, output_dir, task_name):
    """Create heatmap for a single feature."""
    df_feat = df[df['Feature'] == feature].copy()

    # Pivot to matrix (rows = CPTAC, columns = TCGA)
    matrix = df_feat.pivot(index='CPTAC_Class', columns='TCGA_Class', values='Effect_Size')
    sig_matrix = df_feat.pivot(index='CPTAC_Class', columns='TCGA_Class', values='significant')

    # Sort both index and columns
    sorted_index = sorted(matrix.index, key=get_class_sort_key)
    sorted_columns = sorted(matrix.columns, key=get_class_sort_key)
    matrix = matrix.reindex(index=sorted_index, columns=sorted_columns)
    sig_matrix = sig_matrix.reindex(index=sorted_index, columns=sorted_columns)

    # Mask non-significant
    masked_matrix = matrix.copy()
    sig_mask = sig_matrix.fillna(False).astype(bool)
    masked_matrix[~sig_mask] = 0

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create inverted pastel pink colormap (dark to light)
    # Low effect size (few differences) = dark pink, High effect size (many differences) = light/white
    from matplotlib.colors import LinearSegmentedColormap
    pink_cmap = LinearSegmentedColormap.from_list('pastel_pink_inv',
                                                    ['#C71585', '#FF1493', '#FF69B4', '#FFB6C1', '#FFE4E1', '#FFFFFF'])

    # Use inverted pastel pink gradient for all cells
    # Auto-scale to min/max of data
    non_zero_values = masked_matrix[masked_matrix > 0]
    vmin = 0.0#non_zero_values.min().min() if non_zero_values.size > 0 else 0
    vmax = masked_matrix.max().max()

    sns.heatmap(
        masked_matrix,
        annot=True,
        fmt='.2f',
        cmap=pink_cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Effect Size'},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )

    feat_name = FEATURE_NAMES.get(feature, feature)
    ax.set_title(
        f'{task_name}: TCGA vs CPTAC - {feat_name}\n(Significant comparisons only, p ≤ 0.05)',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('TCGA Classes', fontsize=11, fontweight='bold')
    ax.set_ylabel('CPTAC Classes', fontsize=11, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()

    output_path = output_dir / f'{task_name.lower()}_{feature.lower().replace(" ", "_")}_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path.name}")
    plt.close()


def create_summary_heatmap(df, output_dir, task_name):
    """Create summary heatmap showing cumulative effect size for significant features."""
    # Aggregate by pair (sum of effect sizes where significant)
    pair_stats = df[df['significant']].groupby(['TCGA_Class', 'CPTAC_Class']).agg({
        'Effect_Size': 'sum'  # Changed from 'mean' to 'sum'
    }).reset_index()

    # Pivot to matrix (rows = CPTAC, columns = TCGA)
    matrix = pair_stats.pivot(index='CPTAC_Class', columns='TCGA_Class', values='Effect_Size')

    # Fill NaN with 0 (no significant differences)
    matrix = matrix.fillna(0)

    # Sort both index and columns
    sorted_index = sorted(matrix.index, key=get_class_sort_key)
    sorted_columns = sorted(matrix.columns, key=get_class_sort_key)
    matrix = matrix.reindex(index=sorted_index, columns=sorted_columns)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create magenta monochrome colormap (white at 0 to dark magenta at max)
    # Low cumulative effect (few differences) = white, High cumulative effect (many differences) = dark magenta
    from matplotlib.colors import LinearSegmentedColormap
    pink_cmap = LinearSegmentedColormap.from_list('white_magenta',
                                                    ['#FFFFFF', '#FFE0F0', '#FFB3E6', '#FF66CC', '#FF00CC', '#CC0099', '#990073'])

    # Fixed scale (0-4.5)
    vmin = 0.0
    vmax = 4.5

    # Create colorbar with integer ticks only
    import matplotlib.ticker as ticker

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap=pink_cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={
            'label': 'Cumulative Effect Size',
            'ticks': [0, 1, 2, 3, 4]
        },
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )

    ax.set_title(
        f'{task_name}: TCGA vs CPTAC - Cumulative Effect Size\n(Sum of significant comparisons, p ≤ 0.05)',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('TCGA Classes', fontsize=11, fontweight='bold')
    ax.set_ylabel('CPTAC Classes', fontsize=11, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()

    output_path = output_dir / f'{task_name.lower()}_summary_cumulative_effect_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path.name}")
    plt.close()


def create_significance_count_heatmap(df, output_dir, task_name):
    """Create heatmap showing count of significant features per pair."""
    # Count significant features per pair
    pair_counts = df[df['significant']].groupby(['TCGA_Class', 'CPTAC_Class']).size().reset_index(name='count')

    # Pivot to matrix (rows = CPTAC, columns = TCGA)
    matrix = pair_counts.pivot(index='CPTAC_Class', columns='TCGA_Class', values='count')

    # Fill NaN with 0
    matrix = matrix.fillna(0)

    # Sort both index and columns
    sorted_index = sorted(matrix.index, key=get_class_sort_key)
    sorted_columns = sorted(matrix.columns, key=get_class_sort_key)
    matrix = matrix.reindex(index=sorted_index, columns=sorted_columns)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create inverted pastel pink colormap (dark to light)
    # Low count (few significant features) = dark pink, High count (many significant features) = light/white
    from matplotlib.colors import LinearSegmentedColormap
    pink_cmap = LinearSegmentedColormap.from_list('pastel_pink_inv',
                                                    ['#C71585', '#FF1493', '#FF69B4', '#FFB6C1', '#FFE4E1', '#FFFFFF'])

    # Auto-scale to min/max of data
    non_zero_values = matrix[matrix > 0]
    vmin = 0.0#non_zero_values.min().min() if non_zero_values.size > 0 else 0
    vmax = matrix.max().max()

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.0f',
        cmap=pink_cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Number of Significant Features'},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )

    ax.set_title(
        f'{task_name}: TCGA vs CPTAC - Number of Significant Features\n(p ≤ 0.05)',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('TCGA Classes', fontsize=11, fontweight='bold')
    ax.set_ylabel('CPTAC Classes', fontsize=11, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()

    output_path = output_dir / f'{task_name.lower()}_significance_count.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path.name}")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("VISUALIZING CROSS-COHORT CROSS-CLASS RESULTS")
    print("="*80)

    # PAM50 visualizations
    visualize_pam50()

    # ER visualizations
    visualize_task(ER_DIR, ER_FIG_DIR, 'ER')

    # PR visualizations
    visualize_task(PR_DIR, PR_FIG_DIR, 'PR')

    # HER2 visualizations
    visualize_task(HER2_DIR, HER2_FIG_DIR, 'HER2')

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
