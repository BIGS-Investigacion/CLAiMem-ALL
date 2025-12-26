#!/usr/bin/env python3
"""
Create Combined TCGA vs CPTAC Cross-Class Effect Size Figure

Combines the 4 individual task cumulative effect size heatmaps (PAM50, ER, PR, HER2)
from cross-class comparisons into a single 2x2 panel figure.

Author: Claude Code
Date: 2025
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = Path('results/biological_analysis/tcga_vs_cptac_cross_class')
OUTPUT_DIR = Path('results/biological_analysis/tcga_vs_cptac_cross_class')

# Individual cumulative effect size heatmap files
VIZ_FILES = {
    'PAM50': BASE_DIR / 'pam50/figures/pam50_summary_cumulative_effect_size.png',
    'ER': BASE_DIR / 'er/figures/er_summary_cumulative_effect_size.png',
    'PR': BASE_DIR / 'pr/figures/pr_summary_cumulative_effect_size.png',
    'HER2': BASE_DIR / 'her2/figures/her2_summary_cumulative_effect_size.png'
}

# Output file
OUTPUT_FILE = OUTPUT_DIR / 'combined_tcga_cptac_cross_class_effect_sizes.png'


# ==============================================================================
# CREATE COMBINED FIGURE
# ==============================================================================

def create_combined_figure():
    """Create 2x2 panel figure with all task TCGA vs CPTAC cross-class cumulative effect sizes."""

    print("\n" + "="*80)
    print("CREATING COMBINED TCGA VS CPTAC CROSS-CLASS EFFECT SIZE FIGURE")
    print("="*80)

    # Check that all files exist
    missing = [task for task, path in VIZ_FILES.items() if not path.exists()]
    if missing:
        print(f"\nERROR: Missing cumulative effect size files for: {missing}")
        print("Please run visualize_cross_class_results.py first.")
        return

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('TCGA vs CPTAC Cross-Class: Cumulative Effect Sizes (Ordinal + Binary)\n(Significant Comparisons Only)',
                 fontsize=18, fontweight='bold', y=0.995)

    # Panel layout: [PAM50, ER]
    #               [PR, HER2]
    tasks = ['PAM50', 'ER', 'PR', 'HER2']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    panel_labels = ['A', 'B', 'C', 'D']

    for task, (row, col), label in zip(tasks, positions, panel_labels):
        img_path = VIZ_FILES[task]
        ax = axes[row, col]

        print(f"  Loading {task} cumulative effect size heatmap from {img_path.name}...")
        # Read image
        img = mpimg.imread(img_path)
        # Display in subplot
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
