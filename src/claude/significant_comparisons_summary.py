#!/usr/bin/env python3
"""
Summary of significant comparisons with mean p-values and visualizations.
Focuses only on statistically significant results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(results_dir: Path):
    """Load all comparison data."""
    # Intra-TCGA pairwise
    df_pairwise = pd.read_csv(results_dir / 'tcga_pairwise_significant_bonferroni.csv')

    # Inter-cohort
    df_intercohort = pd.read_csv(results_dir / 'intercohort_significant_tcga_vs_cptac.csv')

    return df_pairwise, df_intercohort


def compute_mean_pvalues_intra(df_pairwise: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean p-values for intra-TCGA comparisons grouped by feature.
    """
    summary = df_pairwise.groupby(['Task', 'Feature']).agg({
        'p_value': ['mean', 'median', 'min', 'max', 'std'],
        'Effect_size_r_rb': ['mean', 'median', 'std'],
        'Class_1': 'count'  # Number of significant comparisons
    }).reset_index()

    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.rename(columns={'Class_1_count': 'n_significant'}, inplace=True)

    return summary


def compute_mean_pvalues_inter(df_intercohort: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean p-values for inter-cohort comparisons grouped by class.
    """
    summary = df_intercohort.groupby(['Task', 'Class']).agg({
        'p_value': ['mean', 'median', 'min', 'max', 'std'],
        'Effect_size': ['mean', 'median', 'std'],
        'Feature': 'count'
    }).reset_index()

    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.rename(columns={'Feature_count': 'n_significant_features'}, inplace=True)

    return summary


def plot_intra_tcga_summary(summary: pd.DataFrame, output_dir: Path):
    """
    Visualize intra-TCGA significant comparisons summary.
    """
    # Use clustering palette
    palette = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(summary['Feature'].unique()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Number of significant comparisons by feature
    ax = axes[0, 0]
    df_plot = summary.pivot(index='Task', columns='Feature', values='n_significant')
    df_plot.plot(kind='bar', ax=ax, color=palette, width=0.8, edgecolor='black')
    ax.set_title('Significant Pairwise Comparisons\n(Bonferroni corrected)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of significant comparisons', fontsize=10, fontweight='bold')
    ax.set_xlabel('Task', fontsize=10, fontweight='bold')
    ax.legend(title='Feature', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # 2. Mean p-values by feature
    ax = axes[0, 1]
    df_plot = summary.pivot(index='Task', columns='Feature', values='p_value_mean')
    df_plot.plot(kind='bar', ax=ax, color=palette, width=0.8, edgecolor='black')
    ax.axhline(y=0.001, color='red', linestyle='--', linewidth=1, alpha=0.5, label='p=0.001')
    ax.axhline(y=0.01, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='p=0.01')
    ax.set_title('Mean p-value of Significant Comparisons',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean p-value', fontsize=10, fontweight='bold')
    ax.set_xlabel('Task', fontsize=10, fontweight='bold')
    ax.legend(title='Feature', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # 3. Mean effect sizes by feature
    ax = axes[1, 0]
    df_plot = summary.pivot(index='Task', columns='Feature', values='Effect_size_r_rb_mean')
    df_plot.plot(kind='bar', ax=ax, color=palette, width=0.8, edgecolor='black')
    ax.axhline(y=0.3, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Small (0.3)')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium (0.5)')
    ax.set_title('Mean Effect Size (rank-biserial)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean |r_rb|', fontsize=10, fontweight='bold')
    ax.set_xlabel('Task', fontsize=10, fontweight='bold')
    ax.legend(title='', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # 4. Distribution of p-values (violin plot)
    ax = axes[1, 1]
    # Combine all tasks for feature comparison
    df_violin = summary[['Feature', 'p_value_mean']].copy()

    parts = ax.violinplot([summary[summary['Feature'] == f]['p_value_mean'].values
                           for f in summary['Feature'].unique()],
                          positions=range(len(summary['Feature'].unique())),
                          showmeans=True, showmedians=True)

    # Color violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')

    ax.set_xticks(range(len(summary['Feature'].unique())))
    ax.set_xticklabels(summary['Feature'].unique(), rotation=45, ha='right')
    ax.set_ylabel('Mean p-value', fontsize=10, fontweight='bold')
    ax.set_title('Distribution of Mean p-values by Feature',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Intra-TCGA Significant Pairwise Comparisons Summary',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'intra_tcga_significant_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_intercohort_summary(summary: pd.DataFrame, df_intercohort: pd.DataFrame,
                              output_dir: Path):
    """
    Visualize inter-cohort significant comparisons summary.
    """
    # Use clustering palette
    n_classes = len(summary['Class'].unique())
    palette = sns.color_palette("ch:s=-.2,r=.6", n_colors=n_classes)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Number of significant features by class
    ax = axes[0, 0]
    summary_sorted = summary.sort_values('n_significant_features', ascending=False)

    colors = [palette[i % len(palette)] for i in range(len(summary_sorted))]
    bars = ax.barh(summary_sorted['Class'], summary_sorted['n_significant_features'],
                   color=colors, edgecolor='black', alpha=0.8)

    # Highlight NORMAL-like
    for i, (idx, row) in enumerate(summary_sorted.iterrows()):
        if row['Class'] == 'NORMAL-like':
            bars[i].set_color('#8B008B')
            bars[i].set_alpha(1.0)

    ax.set_xlabel('Number of significant features (out of 6)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Class', fontsize=10, fontweight='bold')
    ax.set_title('Significant Features per Class\n(TCGA vs CPTAC)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 2. Mean p-value by class
    ax = axes[0, 1]
    summary_sorted = summary.sort_values('p_value_mean')

    colors = [palette[i % len(palette)] for i in range(len(summary_sorted))]
    bars = ax.barh(summary_sorted['Class'], summary_sorted['p_value_mean'],
                   color=colors, edgecolor='black', alpha=0.8)

    # Highlight NORMAL-like
    for i, (idx, row) in enumerate(summary_sorted.iterrows()):
        if row['Class'] == 'NORMAL-like':
            bars[i].set_color('#8B008B')
            bars[i].set_alpha(1.0)

    ax.axvline(x=0.001, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.01, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Mean p-value', fontsize=10, fontweight='bold')
    ax.set_ylabel('Class', fontsize=10, fontweight='bold')
    ax.set_title('Mean p-value of Significant Comparisons',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 3. Mean effect size by class
    ax = axes[1, 0]
    summary_sorted = summary.sort_values('Effect_size_mean', ascending=False)

    colors = [palette[i % len(palette)] for i in range(len(summary_sorted))]
    bars = ax.barh(summary_sorted['Class'], summary_sorted['Effect_size_mean'],
                   color=colors, edgecolor='black', alpha=0.8)

    # Highlight NORMAL-like
    for i, (idx, row) in enumerate(summary_sorted.iterrows()):
        if row['Class'] == 'NORMAL-like':
            bars[i].set_color('#8B008B')
            bars[i].set_alpha(1.0)

    ax.axvline(x=0.3, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Medium (0.3)')
    ax.set_xlabel('Mean effect size', fontsize=10, fontweight='bold')
    ax.set_ylabel('Class', fontsize=10, fontweight='bold')
    ax.set_title('Mean Effect Size of Significant Comparisons',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    # 4. Heatmap of significant features by class
    ax = axes[1, 1]

    # Create presence/absence matrix
    classes = df_intercohort['Class'].unique()
    features = df_intercohort['Feature'].unique()

    matrix = np.zeros((len(classes), len(features)))

    for i, cls in enumerate(classes):
        for j, feat in enumerate(features):
            subset = df_intercohort[(df_intercohort['Class'] == cls) &
                                   (df_intercohort['Feature'] == feat)]
            if len(subset) > 0:
                matrix[i, j] = subset['Effect_size'].values[0]

    # Create heatmap
    cmap_heat = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    im = ax.imshow(matrix, cmap=cmap_heat, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels([f.replace('ESTRUCTURA GLANDULAR', 'Tubule').replace('ATIPIA NUCLEAR', 'Nuclear').replace('INFILTRADO_LI', 'Lympho').replace('INFILTRADO_PMN', 'PMN').replace('NECROSIS', 'Necr').replace('MITOSIS', 'Mit') for f in features], rotation=45, ha='right')
    ax.set_yticklabels(classes)

    # Add effect size values
    for i in range(len(classes)):
        for j in range(len(features)):
            if matrix[i, j] > 0:
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black",
                             fontsize=8, fontweight='bold')

    ax.set_title('Effect Sizes of Significant Features\n(TCGA vs CPTAC)',
                 fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Effect Size', fontsize=10, fontweight='bold')

    plt.suptitle('Inter-Cohort Significant Comparisons Summary (TCGA vs CPTAC)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'intercohort_significant_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_combined_summary_table(summary_intra: pd.DataFrame,
                                   summary_inter: pd.DataFrame,
                                   output_dir: Path):
    """
    Create combined summary table.
    """
    # Intra-TCGA summary
    print("\n" + "="*80)
    print("INTRA-TCGA SIGNIFICANT COMPARISONS - MEAN P-VALUES")
    print("="*80)
    print(summary_intra.to_string(index=False, float_format='%.4f'))

    summary_intra.to_csv(output_dir / 'intra_tcga_mean_pvalues.csv',
                        index=False, float_format='%.6f')
    print(f"\n✓ Saved: {output_dir / 'intra_tcga_mean_pvalues.csv'}")

    # Inter-cohort summary
    print("\n" + "="*80)
    print("INTER-COHORT SIGNIFICANT COMPARISONS - MEAN P-VALUES")
    print("="*80)
    print(summary_inter.to_string(index=False, float_format='%.4f'))

    summary_inter.to_csv(output_dir / 'intercohort_mean_pvalues.csv',
                        index=False, float_format='%.6f')
    print(f"\n✓ Saved: {output_dir / 'intercohort_mean_pvalues.csv'}")


def main():
    results_dir = Path('results/biological_analysis')

    print("="*80)
    print("SIGNIFICANT COMPARISONS SUMMARY WITH VISUALIZATIONS")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df_pairwise, df_intercohort = load_data(results_dir)
    print(f"  ✓ Loaded {len(df_pairwise)} significant intra-TCGA comparisons")
    print(f"  ✓ Loaded {len(df_intercohort)} significant inter-cohort comparisons")

    # Compute mean p-values
    print("\nComputing mean p-values...")
    summary_intra = compute_mean_pvalues_intra(df_pairwise)
    summary_inter = compute_mean_pvalues_inter(df_intercohort)
    print("  ✓ Computed summaries")

    # Create tables
    create_combined_summary_table(summary_intra, summary_inter, results_dir)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_intra_tcga_summary(summary_intra, results_dir)
    plot_intercohort_summary(summary_inter, df_intercohort, results_dir)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  1. intra_tcga_mean_pvalues.csv")
    print("  2. intercohort_mean_pvalues.csv")
    print("  3. intra_tcga_significant_summary.png")
    print("  4. intercohort_significant_summary.png")
    print()


if __name__ == '__main__':
    main()
