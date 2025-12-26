#!/usr/bin/env python3
"""
Generate Pairwise Significance Matrices for TCGA Statistical Analysis

Creates heatmap visualizations showing effect sizes for significant pairwise
comparisons between molecular classes within each task.

For each task and feature:
- Shows rank-biserial correlation (r_rb) for ordinal features
- Shows Cramér's V for binary features (when applicable)
- Only displays comparisons that are statistically significant
- Uses magma color palette for consistency

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

STATS_DIR = Path('results/biological_analysis/tcga_statistics')
OUTPUT_DIR = Path('results/biological_analysis/tcga_statistics/figures')

# Feature display names
FEATURE_NAMES = {
    'ESTRUCTURA GLANDULAR': 'Tubule Formation',
    'ATIPIA NUCLEAR': 'Nuclear Pleomorphism',
    'MITOSIS': 'Mitotic Activity',
    'NECROSIS': 'Tumour Necrosis',
    'INFILTRADO_LI': 'Lymphocytic Infiltrate',
    'INFILTRADO_PMN': 'Polymorphonuclear Infiltrate'
}


# ==============================================================================
# LOAD DATA
# ==============================================================================

def load_statistical_results():
    """Load statistical analysis results."""
    
    print("\nLoading statistical results...")
    
    # Load Mann-Whitney pairwise results
    mw_path = STATS_DIR / 'mann_whitney_pairwise_tests.csv'
    mw_df = pd.read_csv(mw_path)
    print(f"  Loaded {len(mw_df)} pairwise Mann-Whitney comparisons")
    
    # Load Kruskal-Wallis results
    kw_path = STATS_DIR / 'kruskal_wallis_tests.csv'
    kw_df = pd.read_csv(kw_path)
    print(f"  Loaded {len(kw_df)} Kruskal-Wallis tests")
    
    # Load chi-square results
    chi_path = STATS_DIR / 'chi_square_tests.csv'
    chi_df = pd.read_csv(chi_path)
    print(f"  Loaded {len(chi_df)} chi-square tests")
    
    return mw_df, kw_df, chi_df


# ==============================================================================
# CLASS ORDERING
# ==============================================================================

def get_class_sort_key(x):
    """
    Sort key for molecular classes.
    Order: PAM50 classes (Basal, Her2-enriched, LumA, LumB, Normal),
           then IHC- classes, then IHC+ classes.
    """
    # PAM50 classes: Basal, Her2-enriched, LumA, LumB, Normal
    pam50_order = {'Basal': 0, 'Her2-enriched': 1, 'LumA': 2, 'LumB': 3, 'Normal': 4}

    # IHC classes: negative (-) before positive (+)
    if x.endswith('-'):
        return (1, x)  # IHC- classes second
    elif x.endswith('+'):
        return (2, x)  # IHC+ classes third
    elif x in pam50_order:
        return (0, pam50_order[x])  # PAM50 classes first, in specified order
    else:
        return (3, x)  # Any other classes last


# ==============================================================================
# CREATE PAIRWISE MATRICES
# ==============================================================================

def create_pairwise_matrix(mw_df: pd.DataFrame, task: str, feature: str) -> pd.DataFrame:
    """
    Create symmetric matrix of effect sizes for significant pairwise comparisons.
    
    Args:
        mw_df: Mann-Whitney pairwise results
        task: Task name (e.g., 'PAM50')
        feature: Feature name
    
    Returns:
        DataFrame with effect sizes (rank-biserial correlation)
    """
    # Filter for this task and feature
    subset = mw_df[(mw_df['Task'] == task) & (mw_df['Feature'] == feature)]
    
    if len(subset) == 0:
        return None
    
    # Get all classes
    classes = sorted(set(subset['Class_1'].unique()) | set(subset['Class_2'].unique()), key=get_class_sort_key)
    
    # Initialize matrix with NaN
    matrix = pd.DataFrame(np.nan, index=classes, columns=classes)
    
    # Fill matrix with effect sizes (only for significant comparisons)
    for _, row in subset.iterrows():
        if row['significant']:
            c1, c2 = row['Class_1'], row['Class_2']
            effect = row['rank_biserial']
            
            # Symmetric matrix
            matrix.loc[c1, c2] = effect
            matrix.loc[c2, c1] = effect
    
    # Diagonal is 0 (comparison with self)
    for c in classes:
        matrix.loc[c, c] = 0.0
    
    return matrix


def compute_mean_effect_size(mw_df: pd.DataFrame, chi_df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Compute sum of absolute effect sizes across all features (ordinal + binary).

    Args:
        mw_df: Mann-Whitney pairwise results (ordinal features)
        chi_df: Chi-square test results (binary features)
        task: Task name

    Returns:
        DataFrame with summed effect sizes
    """
    # Filter for this task
    subset = mw_df[mw_df['Task'] == task]

    if len(subset) == 0:
        return None

    # Get all classes
    classes = sorted(set(subset['Class_1'].unique()) | set(subset['Class_2'].unique()), key=get_class_sort_key)

    # Initialize matrix
    matrix = pd.DataFrame(np.nan, index=classes, columns=classes)

    # Get binary feature effect sizes for this task (if significant)
    chi_task = chi_df[(chi_df['Task'] == task) & (chi_df['p_value'] < 0.05)]
    binary_effect_sum = chi_task['cramers_v'].sum() if len(chi_task) > 0 else 0.0

    # For each pair, compute sum of absolute effect sizes across features
    for c1 in classes:
        for c2 in classes:
            if c1 == c2:
                matrix.loc[c1, c2] = 0.0
                continue

            # Get all comparisons between c1 and c2 (across all ordinal features)
            pair_data = subset[
                (((subset['Class_1'] == c1) & (subset['Class_2'] == c2)) |
                 ((subset['Class_1'] == c2) & (subset['Class_2'] == c1))) &
                (subset['significant'])
            ]

            if len(pair_data) > 0 or binary_effect_sum > 0:
                # Sum of absolute effect sizes from ordinal features
                ordinal_sum = pair_data['rank_biserial'].abs().sum() if len(pair_data) > 0 else 0.0

                # Add binary feature effect sizes (distributed to all pairs)
                total_sum = ordinal_sum + binary_effect_sum
                matrix.loc[c1, c2] = total_sum

    return matrix


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_pairwise_heatmap(matrix: pd.DataFrame, task: str, feature: str, 
                          output_path: Path):
    """
    Plot heatmap of pairwise effect sizes.
    
    Args:
        matrix: Pairwise effect size matrix
        task: Task name
        feature: Feature display name
        output_path: Output file path
    """
    if matrix is None or matrix.empty:
        print(f"  Skipping {feature} (no significant comparisons)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use magma colormap
    cmap = sns.color_palette("magma", as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Rank-Biserial Correlation (r_rb)'},
        ax=ax,
        mask=matrix.isna()  # Mask non-significant comparisons
    )
    
    ax.set_title(f'{task}: {feature}\nPairwise Effect Sizes (Significant Only)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Molecular Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Molecular Class', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_mean_effect_heatmap(matrix: pd.DataFrame, task: str, output_path: Path, vmax_global: float = None):
    """
    Plot heatmap of summed effect sizes across all features (ordinal + binary).

    Args:
        matrix: Summed effect size matrix
        task: Task name
        output_path: Output file path
        vmax_global: Global maximum value for consistent color scale across all tasks
    """
    if matrix is None or matrix.empty:
        print(f"  Skipping summed effect size for {task} (no data)")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Only mask non-significant values (NaN values)
    mask = matrix.isna()

    # Create magenta monochrome colormap (white at 0 to magenta at max)
    from matplotlib.colors import LinearSegmentedColormap
    magenta_cmap = LinearSegmentedColormap.from_list(
        'white_magenta',
        ['#FFFFFF', '#FFE0F0', '#FFB3E6', '#FF66CC', '#FF00CC', '#CC0099', '#990073']
    )

    # Use global vmax if provided, otherwise use local max
    if vmax_global is not None:
        vmax = vmax_global
    else:
        vmax = matrix.max().max() if not matrix.isna().all().all() else 3.0

    # Plot heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap=magenta_cmap,
        vmin=0,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Sum of Effect Sizes'},
        ax=ax,
        mask=mask
    )

    ax.set_title(f'{task}: Sum of Effect Sizes (Ordinal + Binary Features)\n(Significant Comparisons Only)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Molecular Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Molecular Class', fontsize=12, fontweight='bold')

    # Rotate x-axis labels, keep y-axis labels horizontal
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def generate_all_visualizations(mw_df: pd.DataFrame, kw_df: pd.DataFrame, chi_df: pd.DataFrame):
    """Generate all pairwise effect size visualizations."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = sorted(mw_df['Task'].unique())

    print("\n" + "="*80)
    print("GENERATING PAIRWISE EFFECT SIZE MATRICES")
    print("="*80)

    # First pass: compute all matrices and find global max for consistent color scale
    all_matrices = {}
    vmax_global = 0.0

    for task in tasks:
        sum_matrix = compute_mean_effect_size(mw_df, chi_df, task)
        if sum_matrix is not None and not sum_matrix.isna().all().all():
            all_matrices[task] = sum_matrix
            task_max = sum_matrix.max().max()
            if not np.isnan(task_max):
                vmax_global = max(vmax_global, task_max)

    print(f"\nGlobal vmax for consistent color scale: {vmax_global:.3f}\n")

    # Second pass: generate visualizations with consistent scale
    for task in tasks:
        print(f"\n{task}:")

        # Get ordinal features for this task
        task_features = sorted(mw_df[mw_df['Task'] == task]['Feature'].unique())

        # Generate individual feature matrices
        for feature in task_features:
            feature_name = FEATURE_NAMES.get(feature, feature)
            matrix = create_pairwise_matrix(mw_df, task, feature)

            if matrix is not None and not matrix.isna().all().all():
                output_path = OUTPUT_DIR / f'{task.lower()}_{feature.replace(" ", "_").lower()}_pairwise_matrix.png'
                plot_pairwise_heatmap(matrix, task, feature_name, output_path)

        # Generate summed effect size matrix (across all features: ordinal + binary)
        if task in all_matrices:
            output_path = OUTPUT_DIR / f'{task.lower()}_sum_effect_size_matrix.png'
            plot_mean_effect_heatmap(all_matrices[task], task, output_path, vmax_global=vmax_global)


# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

def print_summary_statistics(mw_df: pd.DataFrame):
    """Print summary of significant comparisons."""
    
    print("\n" + "="*80)
    print("SUMMARY: Significant Pairwise Comparisons")
    print("="*80)
    
    for task in sorted(mw_df['Task'].unique()):
        task_data = mw_df[mw_df['Task'] == task]
        sig_data = task_data[task_data['significant']]
        
        print(f"\n{task}:")
        print(f"  Total comparisons: {len(task_data)}")
        print(f"  Significant: {len(sig_data)} ({100*len(sig_data)/len(task_data):.1f}%)")
        
        # Effect size distribution
        if len(sig_data) > 0:
            effect_sizes = sig_data['rank_biserial'].abs()
            print(f"  Mean |effect size|: {effect_sizes.mean():.3f}")
            print(f"  Median |effect size|: {effect_sizes.median():.3f}")
            print(f"  Range: [{effect_sizes.min():.3f}, {effect_sizes.max():.3f}]")
            
            # Count by effect size category
            small = (effect_sizes < 0.30).sum()
            medium = ((effect_sizes >= 0.30) & (effect_sizes < 0.50)).sum()
            large = (effect_sizes >= 0.50).sum()
            
            print(f"  Effect sizes: Small={small}, Medium={medium}, Large={large}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("PAIRWISE SIGNIFICANCE MATRIX VISUALIZATION")
    print("="*80)

    # Load results
    mw_df, kw_df, chi_df = load_statistical_results()

    # Generate visualizations
    generate_all_visualizations(mw_df, kw_df, chi_df)

    # Print summary
    print_summary_statistics(mw_df)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Figures saved to: {OUTPUT_DIR}/\n")


if __name__ == '__main__':
    main()
