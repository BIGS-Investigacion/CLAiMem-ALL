#!/usr/bin/env python3
"""
Create Combined Effect Size Matrices Figure

Combines the 4 individual task effect size matrices (PAM50, ER, PR, HER2)
into a single 2x2 panel figure.

Author: Claude Code
Date: 2025
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_DIR = Path('results/biological_analysis/tcga_statistics/figures')
OUTPUT_DIR = Path('results/biological_analysis/tcga_statistics/figures')

# Individual matrix files
MATRIX_FILES = {
    'PAM50': INPUT_DIR / 'pam50_sum_effect_size_matrix.png',
    'ER': INPUT_DIR / 'er_sum_effect_size_matrix.png',
    'PR': INPUT_DIR / 'pr_sum_effect_size_matrix.png',
    'HER2': INPUT_DIR / 'her2_sum_effect_size_matrix.png'
}

# Output file
OUTPUT_FILE = OUTPUT_DIR / 'combined_effect_size_matrices.png'


# ==============================================================================
# CREATE COMBINED FIGURE
# ==============================================================================

def create_combined_figure():
    """Create 2x2 panel figure with all task effect size matrices."""

    print("\n" + "="*80)
    print("CREATING COMBINED EFFECT SIZE MATRICES FIGURE")
    print("="*80)

    # Check that all files exist
    missing = [task for task, path in MATRIX_FILES.items() if not path.exists()]
    if missing:
        print(f"\nERROR: Missing matrix files for: {missing}")
        print("Please run tcga_pairwise_significance_matrices.py first.")
        return

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('Sum of Effect Sizes Across All Features (Ordinal + Binary)\n(Significant Comparisons Only)',
                 fontsize=18, fontweight='bold', y=0.995)

    # Panel layout: [PAM50, ER]
    #               [PR, HER2]
    tasks = ['PAM50', 'ER', 'PR', 'HER2']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    panel_labels = ['A', 'B', 'C', 'D']

    for task, (row, col), label in zip(tasks, positions, panel_labels):
        img_path = MATRIX_FILES[task]
        print(f"  Loading {task} matrix from {img_path.name}...")

        # Read image
        img = mpimg.imread(img_path)

        # Display in subplot
        ax = axes[row, col]
        ax.imshow(img)
        ax.axis('off')

        # Add panel label
        ax.text(-0.05, 1.05, label, transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='right')

    plt.tight_layout()

    # Save combined figure
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n✓ Combined figure saved: {OUTPUT_FILE}")
    print(f"  Size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
    plt.close()

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    create_combined_figure()
