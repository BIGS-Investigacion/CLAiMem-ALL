#!/usr/bin/env python3
"""
Feature Space Unified Comparison with/without Outliers

Generates dual-panel scatter plots comparing:
- RPC vs Centroid Distance
- RPC vs Silhouette Scores

With and without outliers (Her2-enriched and Normal with RPC = -1.0).

Author: Claude Code
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path

# ============================================================================
# Load RPC data from distributional shift analysis
# ============================================================================

def load_rpc_data():
    """Load RPC data from performance degradation CSV files."""
    rpc_data = {}

    tasks = ['pam50', 'er', 'pr', 'her2']
    for task in tasks:
        csv_path = Path(f'results/distributional_shift/{task}_performance_degradation.csv')
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                rpc_data[row['Class']] = row['RPC']

    return rpc_data

# Load RPC data
rpc_dict = load_rpc_data()

# ============================================================================
# Feature space metrics
# ============================================================================

# Centroid distances
centroid_distance = {
    'LumA': 16.66,
    'LumB': 23.38,
    'Her2-enriched': 22.22,
    'Basal': 24.88,
    'Normal': 26.31,
    'ER-positive': 17.38,
    'ER-negative': 18.23,
    'PR-positive': 19.78,
    'PR-negative': 23.02,
    'HER2-positive': 20.92,
    'HER2-negative': 18.38
}

# Silhouette scores TCGA
silhouette_tcga = {
    'LumA': -0.032,
    'LumB': -0.039,
    'Her2-enriched': 0.110,
    'Basal': 0.118,
    'Normal': 0.066,
    'ER-positive': 0.089,
    'ER-negative': 0.092,
    'PR-positive': 0.004,
    'PR-negative': 0.117,
    'HER2-positive': 0.075,
    'HER2-negative': -0.029
}

# Silhouette scores CPTAC
silhouette_cptac = {
    'LumA': 0.023,
    'LumB': -0.021,
    'Her2-enriched': -0.017,
    'Basal': 0.008,
    'Normal': -0.020,
    'ER-positive': 0.054,
    'ER-negative': 0.010,
    'PR-positive': 0.013,
    'PR-negative': 0.010,
    'HER2-positive': 0.012,
    'HER2-negative': 0.007
}

# ============================================================================
# Organize data by task
# ============================================================================

# Task assignment
task_mapping = {
    'LumA': 'PAM50',
    'LumB': 'PAM50',
    'Her2-enriched': 'PAM50',
    'Basal': 'PAM50',
    'Normal': 'PAM50',
    'ER-positive': 'ER',
    'ER-negative': 'ER',
    'PR-positive': 'PR',
    'PR-negative': 'PR',
    'HER2-positive': 'HER2',
    'HER2-negative': 'HER2'
}

classes = ['LumA', 'LumB', 'Her2-enriched', 'Basal', 'Normal',
           'ER-positive', 'ER-negative',
           'PR-positive', 'PR-negative',
           'HER2-positive', 'HER2-negative']

# Create arrays
rpc = np.array([rpc_dict[c] for c in classes])
centroid_dist = np.array([centroid_distance[c] for c in classes])
silh_tcga = np.array([silhouette_tcga[c] for c in classes])
silh_cptac = np.array([silhouette_cptac[c] for c in classes])
tasks = np.array([task_mapping[c] for c in classes])

# ============================================================================
# Color palette (magma)
# ============================================================================

magma_colors = sns.color_palette("magma", 4)
task_colors = {
    'PAM50': magma_colors[0],  # morado oscuro
    'ER': magma_colors[1],     # rosado morado
    'PR': magma_colors[2],     # anaranjado rosado
    'HER2': magma_colors[3]    # amarillo claro
}

# ============================================================================
# Plot function
# ============================================================================

def plot_comparison(x_data, x_label, output_filename):
    """
    Create dual-panel comparison: with and without outliers

    Args:
        x_data: X-axis data (feature space metric)
        x_label: Label for X-axis
        output_filename: Output file path
    """
    # Identify outliers (RPC = -1.0)
    outliers_mask = rpc == -1.0
    outlier_classes = np.array(classes)[outliers_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ========================================================================
    # Left panel: With outliers
    # ========================================================================

    tasks_unique = ['PAM50', 'ER', 'PR', 'HER2']

    for task in tasks_unique:
        mask_task = tasks == task
        if mask_task.any():
            ax1.scatter(x_data[mask_task], rpc[mask_task],
                       color=task_colors[task], label=task, s=100, alpha=0.7,
                       edgecolors='black', linewidths=1.5)

    # Mark outliers with hollow red circles
    if outliers_mask.any():
        ax1.scatter(x_data[outliers_mask], rpc[outliers_mask],
                   facecolors='none', edgecolors='#d62728', s=300,
                   linewidths=2.5, alpha=1.0, label='Outliers', zorder=3)

    # Regression line (all data)
    r1, p1 = pearsonr(x_data, rpc)
    z1 = np.polyfit(x_data, rpc, 1)
    p_fit1 = np.poly1d(z1)
    x_line1 = np.linspace(x_data.min(), x_data.max(), 100)
    ax1.plot(x_line1, p_fit1(x_line1), "r--", alpha=0.5, linewidth=2,
             label=f'Linear fit (R²={r1**2:.3f})')

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax1.set_ylabel('Relative Performance Change (RPC)', fontsize=13, fontweight='bold')

    # Adjust X-axis limits to remove empty space
    x_margin = (x_data.max() - x_data.min()) * 0.05
    ax1.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)

    sig_str1 = "***" if p1 < 0.001 else "**" if p1 < 0.01 else "*" if p1 < 0.05 else "n.s."
    ax1.set_title(f'All data (n={len(classes)})\nPearson r = {r1:.3f}, p = {p1:.4f} {sig_str1}',
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='best', framealpha=0.9, fontsize=10, markerscale=0.8)
    ax1.grid(alpha=0.3, linestyle='--')

    # ========================================================================
    # Right panel: Without outliers
    # ========================================================================

    mask_valid = ~outliers_mask
    x_filtered = x_data[mask_valid]
    rpc_filtered = rpc[mask_valid]
    tasks_filtered = tasks[mask_valid]

    for task in tasks_unique:
        mask_task = tasks_filtered == task
        if mask_task.any():
            ax2.scatter(x_filtered[mask_task], rpc_filtered[mask_task],
                       color=task_colors[task], label=task, s=100, alpha=0.7,
                       edgecolors='black', linewidths=1.5)

    # Regression line (without outliers)
    r2, p2 = pearsonr(x_filtered, rpc_filtered)
    z2 = np.polyfit(x_filtered, rpc_filtered, 1)
    p_fit2 = np.poly1d(z2)
    x_line2 = np.linspace(x_filtered.min(), x_filtered.max(), 100)
    ax2.plot(x_line2, p_fit2(x_line2), "r--", alpha=0.5, linewidth=2,
             label=f'Linear fit (R²={r2**2:.3f})')

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax2.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax2.set_ylabel('Relative Performance Change (RPC)', fontsize=13, fontweight='bold')

    # Adjust X-axis limits to remove empty space
    x_margin_filtered = (x_filtered.max() - x_filtered.min()) * 0.05
    ax2.set_xlim(x_filtered.min() - x_margin_filtered, x_filtered.max() + x_margin_filtered)

    sig_str2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "n.s."
    outlier_names = ', '.join(outlier_classes)
    ax2.set_title(f'Without {outlier_names} (n={len(x_filtered)})\nPearson r = {r2:.3f}, p = {p2:.4f} {sig_str2}',
                 fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='best', framealpha=0.9, fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')

    fig.suptitle(f'Feature Space Analysis: {x_label} vs RPC',
                 fontsize=16, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_filename}")
    plt.close()

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    output_dir = Path("results/feature_space_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("FEATURE SPACE UNIFIED COMPARISON PLOTS")
    print("="*80)
    print()

    # Plot 1: RPC vs Centroid Distance
    plot_comparison(
        centroid_dist,
        "Centroid Distance (TCGA ↔ CPTAC)",
        output_dir / "unified_rpc_vs_centroid_distance.png"
    )

    # Plot 2: RPC vs Silhouette TCGA
    plot_comparison(
        silh_tcga,
        "Silhouette Score (TCGA)",
        output_dir / "unified_rpc_vs_silhouette_tcga.png"
    )

    # Plot 3: RPC vs Silhouette CPTAC
    plot_comparison(
        silh_cptac,
        "Silhouette Score (CPTAC)",
        output_dir / "unified_rpc_vs_silhouette_cptac.png"
    )

    print()
    print("="*80)
    print("COMPARISON PLOTS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}/")
    print()
