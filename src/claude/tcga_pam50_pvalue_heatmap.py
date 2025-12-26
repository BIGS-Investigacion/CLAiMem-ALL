#!/usr/bin/env python3
"""
Generate heatmap showing minimum p-values for PAM50 pairwise comparisons in TCGA.
For each class pair, shows the minimum p-value across all features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_pairwise_data(results_dir: Path) -> pd.DataFrame:
    """Load TCGA pairwise comparisons."""
    csv_path = results_dir / 'tcga_pairwise_comparisons_bonferroni.csv'
    df = pd.read_csv(csv_path)
    return df


def create_min_pvalue_matrix(df_pairwise: pd.DataFrame, classes: list) -> pd.DataFrame:
    """
    Create matrix showing minimum p-value across all features for each class pair.
    """
    n = len(classes)
    matrix = np.full((n, n), np.nan)

    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes):
            if i == j:
                matrix[i, j] = np.nan  # Diagonal
                continue

            # Get all comparisons for this pair
            subset = df_pairwise[
                ((df_pairwise['Class_1'] == class1) & (df_pairwise['Class_2'] == class2)) |
                ((df_pairwise['Class_1'] == class2) & (df_pairwise['Class_2'] == class1))
            ]

            if len(subset) > 0:
                # Take minimum p-value across all features
                matrix[i, j] = subset['p_value'].min()

    return pd.DataFrame(matrix, index=classes, columns=classes)


def plot_pvalue_heatmap(matrix: pd.DataFrame, output_path: Path):
    """
    Plot heatmap of minimum p-values with clustering palette.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use clustering palette (pink/magenta)
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True).reversed()

    # Create mask for diagonal
    mask = np.isnan(matrix.values)

    # Log-transform p-values for better visualization
    log_matrix = -np.log10(matrix.replace(0, 1e-10))  # -log10(p-value)

    sns.heatmap(log_matrix, annot=matrix, fmt='.2e', cmap=cmap,
                mask=mask,
                cbar_kws={'label': '-log₁₀(p-value)'},
                linewidths=1, linecolor='white', ax=ax,
                vmin=0, square=True)

    ax.set_title('PAM50 Subtypes: Minimum p-value per Class Pair\n(TCGA-BRCA Intra-cohort, across all features)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class', fontsize=12, fontweight='bold')

    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    results_dir = Path('results/biological_analysis')

    print("="*80)
    print("PAM50 MINIMUM P-VALUE HEATMAP")
    print("="*80)

    # Load data
    print("\nLoading pairwise comparisons...")
    df_pairwise = load_pairwise_data(results_dir)

    # Filter PAM50 classes only
    pam50_classes = ['BASAL', 'HER2-enriched', 'LUMINAL-A', 'LUMINAL-B', 'NORMAL-like']

    # Filter data for PAM50 task only
    df_pam50 = df_pairwise[df_pairwise['Task'] == 'PAM50'].copy()

    # Filter for PAM50 classes only
    df_pam50 = df_pam50[
        df_pam50['Class_1'].isin(pam50_classes) &
        df_pam50['Class_2'].isin(pam50_classes)
    ]

    print(f"✓ Loaded {len(df_pam50)} comparisons for PAM50 subtypes")

    # Create minimum p-value matrix
    print("\nCreating minimum p-value matrix...")
    pvalue_matrix = create_min_pvalue_matrix(df_pam50, pam50_classes)

    print("\nMinimum p-values matrix:")
    print(pvalue_matrix.to_string(float_format='%.2e'))

    # Save matrix
    output_csv = results_dir / 'tcga_pam50_min_pvalue_matrix.csv'
    pvalue_matrix.to_csv(output_csv, float_format='%.6e')
    print(f"\n✓ Saved: {output_csv}")

    # Generate heatmap
    print("\nGenerating heatmap...")
    output_png = results_dir / 'tcga_pam50_min_pvalue_heatmap.png'
    plot_pvalue_heatmap(pvalue_matrix, output_png)

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Get upper triangle (avoid duplicates and diagonal)
    upper_triangle = []
    for i in range(len(pam50_classes)):
        for j in range(i+1, len(pam50_classes)):
            val = pvalue_matrix.iloc[i, j]
            if not np.isnan(val):
                upper_triangle.append({
                    'Pair': f"{pam50_classes[i]} vs {pam50_classes[j]}",
                    'Min_p_value': val
                })

    df_summary = pd.DataFrame(upper_triangle).sort_values('Min_p_value')

    print(f"\nAll {len(df_summary)} class pairs sorted by minimum p-value:")
    print(df_summary.to_string(index=False, float_format='%.2e'))

    # Count significant pairs
    alpha = 0.05
    n_sig = (df_summary['Min_p_value'] < alpha).sum()

    print(f"\nSignificant pairs (min p-value < {alpha}): {n_sig}/{len(df_summary)}")
    print(f"Percentage: {100*n_sig/len(df_summary):.1f}%")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {output_csv}")
    print(f"  - {output_png}")
    print()


if __name__ == '__main__':
    main()
