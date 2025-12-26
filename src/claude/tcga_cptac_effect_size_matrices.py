#!/usr/bin/env python3
"""
Generate Effect Size Matrices for TCGA vs CPTAC Comparisons

Creates heatmap visualizations showing effect sizes for comparisons
between TCGA and CPTAC cohorts for each molecular class.

For each task and class:
- Shows rank-biserial correlation (r_rb) for ordinal features
- Shows Cramér's V for binary features
- Only displays comparisons that are statistically significant
- Uses magenta monochrome color palette

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

COHORT_DIR = Path('results/biological_analysis/cohort_comparison')
OUTPUT_DIR = Path('results/biological_analysis/cohort_comparison/figures')

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

def load_cohort_comparison_results():
    """Load TCGA vs CPTAC comparison results."""

    print("\nLoading TCGA vs CPTAC comparison results...")

    # Load Mann-Whitney results (ordinal features)
    mw_path = COHORT_DIR / 'cohort_mann_whitney_tests.csv'
    mw_df = pd.read_csv(mw_path)
    print(f"  Loaded {len(mw_df)} Mann-Whitney comparisons")

    # Load chi-square results (binary features)
    chi_path = COHORT_DIR / 'cohort_chi_square_tests.csv'
    chi_df = pd.read_csv(chi_path)
    print(f"  Loaded {len(chi_df)} chi-square tests")

    return mw_df, chi_df


# ==============================================================================
# COMPUTE EFFECT SIZE MATRIX
# ==============================================================================

def compute_cohort_effect_size_matrix(mw_df: pd.DataFrame, chi_df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Compute sum of absolute effect sizes across all features (ordinal + binary)
    for TCGA vs CPTAC comparisons.

    Args:
        mw_df: Mann-Whitney results (ordinal features)
        chi_df: Chi-square results (binary features)
        task: Task name

    Returns:
        DataFrame with effect sizes per class (single column for TCGA-CPTAC comparison)
    """
    # Filter for this task
    mw_subset = mw_df[(mw_df['Task'] == task) & (mw_df['significant'])]
    chi_subset = chi_df[(chi_df['Task'] == task) & (chi_df['significant'])]

    # Get all classes for this task
    classes = sorted(set(mw_df[mw_df['Task'] == task]['Class'].unique()))

    # Initialize results dictionary
    results = {}

    for cls in classes:
        # Sum ordinal feature effect sizes
        ordinal_effects = mw_subset[mw_subset['Class'] == cls]['rank_biserial'].abs()
        ordinal_sum = ordinal_effects.sum() if len(ordinal_effects) > 0 else 0.0

        # Sum binary feature effect sizes
        binary_effects = chi_subset[chi_subset['Class'] == cls]['cramers_v']
        binary_sum = binary_effects.sum() if len(binary_effects) > 0 else 0.0

        # Total effect size
        total = ordinal_sum + binary_sum

        # Only include if there are significant differences
        if total > 0:
            results[cls] = total

    # Convert to DataFrame
    if results:
        df = pd.DataFrame(list(results.items()), columns=['Class', 'Effect_Size'])
        df = df.sort_values('Effect_Size', ascending=False)
        return df
    else:
        return None


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_cohort_effect_size_bar(df: pd.DataFrame, task: str, output_path: Path, vmax_global: float = None):
    """
    Plot bar chart of effect sizes for TCGA vs CPTAC comparisons.

    Args:
        df: DataFrame with Class and Effect_Size columns
        task: Task name
        output_path: Output file path
        vmax_global: Global maximum for consistent color scale
    """
    if df is None or df.empty:
        print(f"  Skipping {task} (no significant differences)")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create magenta monochrome colormap
    from matplotlib.colors import LinearSegmentedColormap
    magenta_cmap = LinearSegmentedColormap.from_list(
        'white_magenta',
        ['#FFFFFF', '#FFE0F0', '#FFB3E6', '#FF66CC', '#FF00CC', '#CC0099', '#990073']
    )

    # Normalize colors based on effect size
    if vmax_global is not None:
        norm = plt.Normalize(vmin=0, vmax=vmax_global)
    else:
        norm = plt.Normalize(vmin=0, vmax=df['Effect_Size'].max())

    colors = [magenta_cmap(norm(val)) for val in df['Effect_Size']]

    # Create horizontal bar chart
    bars = ax.barh(df['Class'], df['Effect_Size'], color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row['Effect_Size'] + 0.05, i, f"{row['Effect_Size']:.2f}",
                va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Sum of Effect Sizes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Molecular Class', fontsize=12, fontweight='bold')
    ax.set_title(f'{task}: TCGA vs CPTAC Effect Sizes (Ordinal + Binary Features)\n(Significant Comparisons Only)',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    if vmax_global is not None:
        ax.set_xlim(0, vmax_global * 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def generate_all_visualizations(mw_df: pd.DataFrame, chi_df: pd.DataFrame):
    """Generate all TCGA vs CPTAC effect size visualizations."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = sorted(mw_df['Task'].unique())

    print("\n" + "="*80)
    print("GENERATING TCGA vs CPTAC EFFECT SIZE VISUALIZATIONS")
    print("="*80)

    # First pass: compute all matrices and find global max
    all_results = {}
    vmax_global = 0.0

    for task in tasks:
        df = compute_cohort_effect_size_matrix(mw_df, chi_df, task)
        if df is not None and not df.empty:
            all_results[task] = df
            task_max = df['Effect_Size'].max()
            vmax_global = max(vmax_global, task_max)

    print(f"\nGlobal vmax for consistent color scale: {vmax_global:.3f}\n")

    # Second pass: generate visualizations with consistent scale
    for task in tasks:
        print(f"\n{task}:")
        if task in all_results:
            output_path = OUTPUT_DIR / f'{task.lower()}_tcga_cptac_effect_size.png'
            plot_cohort_effect_size_bar(all_results[task], task, output_path, vmax_global=vmax_global)


# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

def print_summary_statistics(mw_df: pd.DataFrame, chi_df: pd.DataFrame):
    """Print summary of significant TCGA vs CPTAC comparisons."""

    print("\n" + "="*80)
    print("SUMMARY: Significant TCGA vs CPTAC Comparisons")
    print("="*80)

    for task in sorted(mw_df['Task'].unique()):
        mw_task = mw_df[(mw_df['Task'] == task) & (mw_df['significant'])]
        chi_task = chi_df[(chi_df['Task'] == task) & (chi_df['significant'])]

        total_sig = len(mw_task) + len(chi_task)

        print(f"\n{task}:")
        print(f"  Significant comparisons: {total_sig}")
        print(f"    Ordinal features: {len(mw_task)}")
        print(f"    Binary features: {len(chi_task)}")

        if len(mw_task) > 0:
            effect_sizes = mw_task['rank_biserial'].abs()
            print(f"  Ordinal effect sizes: mean={effect_sizes.mean():.3f}, max={effect_sizes.max():.3f}")

        if len(chi_task) > 0:
            effect_sizes = chi_task['cramers_v']
            print(f"  Binary effect sizes: mean={effect_sizes.mean():.3f}, max={effect_sizes.max():.3f}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("TCGA vs CPTAC EFFECT SIZE VISUALIZATION")
    print("="*80)

    # Load results
    mw_df, chi_df = load_cohort_comparison_results()

    # Generate visualizations
    generate_all_visualizations(mw_df, chi_df)

    # Print summary
    print_summary_statistics(mw_df, chi_df)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Figures saved to: {OUTPUT_DIR}/\n")


if __name__ == '__main__':
    main()
