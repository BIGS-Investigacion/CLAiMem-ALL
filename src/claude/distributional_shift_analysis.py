#!/usr/bin/env python3
"""
==============================================================================
DISTRIBUTIONAL SHIFT ANALYSIS
==============================================================================

This script assesses whether class prevalence differences between TCGA and CPTAC
cohorts contribute to performance degradation in external validation.

Methodology:
1. Compute class prevalence in both cohorts
2. Calculate prevalence shift (Δp) for each class
3. Measure performance degradation (relative change in F1/PR-AUC)
4. Compute Pearson correlation between prevalence shift and performance degradation
5. Statistical significance testing

Key Question:
Do classes with larger prevalence shifts show proportionally greater performance
degradation? If correlation ≈ 0 (p > 0.05), distributional shift is not the primary
driver of generalization failure.

Author: Claude Code
Date: 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ==============================================================================
# CONFIGURATION
# ==============================================================================

PAM50_LABELS = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
IHC_LABELS = {
    'ER': ['ER-negative', 'ER-positive'],
    'PR': ['PR-negative', 'PR-positive'],
    'HER2': ['HER2-negative', 'HER2-positive']
}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_class_distribution(csv_path: Path) -> pd.DataFrame:
    """
    Load class distribution from labels CSV file.

    Expected format:
        slide_id,label
        TCGA-A1-001,LumA
        TCGA-A1-002,LumB
        ...

    Args:
        csv_path: Path to labels CSV

    Returns:
        DataFrame with slide_id and label columns
    """
    df = pd.read_csv(csv_path)

    if 'label' not in df.columns:
        raise ValueError(f"CSV must have 'label' column. Found: {df.columns.tolist()}")

    return df


def load_performance_metrics(csv_path: Path) -> pd.DataFrame:
    """
    Load performance metrics from CSV.

    Expected columns:
        - Class: class name
        - F1_MCCV or PR_AUC_MCCV: internal validation metric
        - F1_HO or PR_AUC_HO: hold-out (external) validation metric

    Args:
        csv_path: Path to metrics CSV

    Returns:
        DataFrame with performance metrics
    """
    df = pd.read_csv(csv_path)

    required_cols = ['Class']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must have 'Class' column. Found: {df.columns.tolist()}")

    return df


# ==============================================================================
# DISTRIBUTIONAL ANALYSIS
# ==============================================================================

def compute_prevalence(df_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Compute class prevalence.

    Args:
        df_labels: DataFrame with slide_id and label columns

    Returns:
        DataFrame with class counts and prevalence
    """
    total = len(df_labels)
    prevalence = df_labels['label'].value_counts().reset_index()
    prevalence.columns = ['Class', 'Count']
    prevalence['Prevalence'] = prevalence['Count'] / total

    return prevalence.sort_values('Class').reset_index(drop=True)


def compute_prevalence_shift(prev_tcga: pd.DataFrame,
                             prev_cptac: pd.DataFrame) -> pd.DataFrame:
    """
    Compute prevalence shift between cohorts.

    Args:
        prev_tcga: TCGA prevalence DataFrame
        prev_cptac: CPTAC prevalence DataFrame

    Returns:
        DataFrame with prevalence shift metrics
    """
    # Merge prevalence data
    df = prev_tcga[['Class', 'Prevalence']].merge(
        prev_cptac[['Class', 'Prevalence']],
        on='Class',
        suffixes=('_TCGA', '_CPTAC')
    )

    # Compute shift metrics
    df['Δp_absolute'] = np.abs(df['Prevalence_CPTAC'] - df['Prevalence_TCGA'])
    df['Δp_relative_%'] = (
        (df['Prevalence_CPTAC'] - df['Prevalence_TCGA']) /
        df['Prevalence_TCGA'] * 100
    )

    # Merge counts
    df = df.merge(
        prev_tcga[['Class', 'Count']].rename(columns={'Count': 'Count_TCGA'}),
        on='Class'
    ).merge(
        prev_cptac[['Class', 'Count']].rename(columns={'Count': 'Count_CPTAC'}),
        on='Class'
    )

    return df


def compute_performance_degradation(df_metrics: pd.DataFrame,
                                    task: str) -> pd.DataFrame:
    """
    Compute relative performance change (RPC) for each class.

    Args:
        df_metrics: DataFrame with performance metrics
        task: 'pam50' or IHC task ('er', 'pr', 'her2')

    Returns:
        DataFrame with performance degradation metrics
    """
    df = df_metrics.copy()

    # Determine metric column names
    if task == 'pam50':
        # Multi-class: use F1-score
        metric_mccv = 'F1_MCCV'
        metric_ho = 'F1_HO'
        metric_name = 'F1'
    else:
        # Binary: use PR-AUC
        if 'PR_AUC_MCCV' in df.columns:
            metric_mccv = 'PR_AUC_MCCV'
            metric_ho = 'PR_AUC_HO'
            metric_name = 'PR-AUC'
        else:
            # Fallback to F1 if PR-AUC not available
            metric_mccv = 'F1_MCCV'
            metric_ho = 'F1_HO'
            metric_name = 'F1'

    # Check columns exist
    if metric_mccv not in df.columns or metric_ho not in df.columns:
        raise ValueError(
            f"Required metric columns not found. Expected: {metric_mccv}, {metric_ho}. "
            f"Found: {df.columns.tolist()}"
        )

    # Compute relative performance change (RPC)
    df['RPC'] = (df[metric_ho] - df[metric_mccv]) / df[metric_mccv]
    df['RPC_%'] = df['RPC'] * 100

    # Absolute change
    df['Δ_metric'] = df[metric_ho] - df[metric_mccv]

    df['metric_name'] = metric_name
    df['metric_mccv'] = df[metric_mccv]
    df['metric_ho'] = df[metric_ho]

    return df


def correlate_shift_and_degradation(df_shift: pd.DataFrame,
                                    df_degradation: pd.DataFrame) -> Dict:
    """
    Compute correlation between prevalence shift and performance degradation.

    Args:
        df_shift: DataFrame with prevalence shift
        df_degradation: DataFrame with performance degradation (RPC)

    Returns:
        Dictionary with correlation results
    """
    # Merge data
    df = df_shift[['Class', 'Δp_absolute']].merge(
        df_degradation[['Class', 'RPC']],
        on='Class'
    )

    # Compute Pearson correlation
    r, p_value = stats.pearsonr(df['Δp_absolute'], df['RPC'])

    # Compute Spearman correlation (non-parametric)
    r_spearman, p_spearman = stats.spearmanr(df['Δp_absolute'], df['RPC'])

    # Linear regression for trend line
    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
        df['Δp_absolute'], df['RPC']
    )

    return {
        'pearson_r': float(r),
        'pearson_p': float(p_value),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman),
        'regression_slope': float(slope),
        'regression_intercept': float(intercept),
        'regression_r2': float(r_value ** 2),
        'regression_p': float(p_value_reg),
        'n_classes': len(df),
        'data': df.to_dict(orient='records')
    }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_prevalence_comparison(df_shift: pd.DataFrame, task: str,
                               output_path: Optional[Path] = None) -> None:
    """
    Plot class prevalence comparison between cohorts.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df_shift))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_shift['Prevalence_TCGA'], width,
                   label='TCGA (Internal)', color='steelblue',
                   alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df_shift['Prevalence_CPTAC'], width,
                   label='CPTAC (External)', color='coral',
                   alpha=0.8, edgecolor='black')

    # Add count labels
    for i, (_, row) in enumerate(df_shift.iterrows()):
        ax.text(i - width/2, row['Prevalence_TCGA'] + 0.01,
                f"n={row['Count_TCGA']}",
                ha='center', va='bottom', fontsize=8, color='darkblue')
        ax.text(i + width/2, row['Prevalence_CPTAC'] + 0.01,
                f"n={row['Count_CPTAC']}",
                ha='center', va='bottom', fontsize=8, color='darkred')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prevalence', fontsize=12, fontweight='bold')
    ax.set_title(f'Class Prevalence: TCGA vs CPTAC\n{task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_shift['Class'], rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(df_shift['Prevalence_TCGA'].max(),
                        df_shift['Prevalence_CPTAC'].max()) * 1.15])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_prevalence_shift(df_shift: pd.DataFrame, task: str,
                         output_path: Optional[Path] = None) -> None:
    """
    Plot absolute prevalence shift.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlGn_r(df_shift['Δp_absolute'] / df_shift['Δp_absolute'].max())

    bars = ax.bar(df_shift['Class'], df_shift['Δp_absolute'],
                  color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, delta in zip(bars, df_shift['Δp_absolute']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{delta:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('|Δp| (Absolute Prevalence Shift)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distributional Shift Between Cohorts\n{task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(df_shift['Class'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_correlation(corr_results: Dict, task: str,
                    output_path: Optional[Path] = None) -> None:
    """
    Scatter plot of prevalence shift vs performance degradation.
    """
    df = pd.DataFrame(corr_results['data'])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    scatter = ax.scatter(df['Δp_absolute'], df['RPC'],
                        s=150, alpha=0.7, c=df['RPC'],
                        cmap='RdYlGn', edgecolor='black', linewidth=1.5)

    # Add labels
    for _, row in df.iterrows():
        ax.annotate(row['Class'],
                   (row['Δp_absolute'], row['RPC']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')

    # Regression line
    x_line = np.array([df['Δp_absolute'].min(), df['Δp_absolute'].max()])
    y_line = (corr_results['regression_slope'] * x_line +
              corr_results['regression_intercept'])
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7,
            label=f"y = {corr_results['regression_slope']:.2f}x + {corr_results['regression_intercept']:.2f}")

    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Relative Performance Change (RPC)', fontsize=11)

    ax.set_xlabel('Prevalence Shift |Δp|', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Performance Change (RPC)', fontsize=12, fontweight='bold')

    # Title with correlation stats
    title = (f'Prevalence Shift vs Performance Degradation\n{task.upper()}\n'
             f"Pearson r = {corr_results['pearson_r']:.3f}, "
             f"p = {corr_results['pearson_p']:.4f}, "
             f"R² = {corr_results['regression_r2']:.3f}")
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_combined_analysis(df_shift: pd.DataFrame, df_degradation: pd.DataFrame,
                          task: str, output_path: Optional[Path] = None) -> None:
    """
    Combined plot showing prevalence shift and performance change.
    """
    # Merge data
    df = df_shift[['Class', 'Δp_absolute']].merge(
        df_degradation[['Class', 'RPC_%']],
        on='Class'
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.35

    # Prevalence shift (left axis)
    color1 = 'steelblue'
    ax1.bar(x - width/2, df['Δp_absolute'], width,
            label='Prevalence Shift |Δp|', color=color1,
            alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Prevalence Shift |Δp|', fontsize=12, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Class'], rotation=45, ha='right')

    # Performance change (right axis)
    ax2 = ax1.twinx()
    color2 = 'coral'
    ax2.bar(x + width/2, df['RPC_%'], width,
            label='Performance Change (RPC %)', color=color2,
            alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Relative Performance Change (%)', fontsize=12, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Title
    ax1.set_title(f'Prevalence Shift vs Performance Degradation\n{task.upper()}',
                  fontsize=14, fontweight='bold', pad=20)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


# ==============================================================================
# REPORTING
# ==============================================================================

def print_results(df_shift: pd.DataFrame, df_degradation: pd.DataFrame,
                  corr_results: Dict, task: str) -> None:
    """
    Print formatted results.
    """
    print("\n" + "="*80)
    print(f"DISTRIBUTIONAL SHIFT ANALYSIS: {task.upper()}")
    print("="*80)

    # Prevalence statistics
    print("\n" + "-"*80)
    print("CLASS PREVALENCE")
    print("-"*80)
    print(f"{'Class':<15} {'TCGA':>12} {'CPTAC':>12} {'|Δp|':>10} {'Δp%':>10}")
    print("-"*80)

    for _, row in df_shift.iterrows():
        print(f"{row['Class']:<15} "
              f"{row['Prevalence_TCGA']:>12.4f} "
              f"{row['Prevalence_CPTAC']:>12.4f} "
              f"{row['Δp_absolute']:>10.4f} "
              f"{row['Δp_relative_%']:>+10.2f}%")

    print(f"\nMean |Δp|: {df_shift['Δp_absolute'].mean():.4f}")
    print(f"Max |Δp|:  {df_shift['Δp_absolute'].max():.4f} ({df_shift.loc[df_shift['Δp_absolute'].idxmax(), 'Class']})")

    # Performance degradation
    print("\n" + "-"*80)
    print("PERFORMANCE DEGRADATION")
    print("-"*80)

    metric_name = df_degradation['metric_name'].iloc[0]
    print(f"{'Class':<15} {'MCCV':>12} {'Hold-Out':>12} {'RPC':>10} {'RPC%':>10}")
    print("-"*80)

    for _, row in df_degradation.iterrows():
        print(f"{row['Class']:<15} "
              f"{row['metric_mccv']:>12.4f} "
              f"{row['metric_ho']:>12.4f} "
              f"{row['RPC']:>+10.4f} "
              f"{row['RPC_%']:>+10.2f}%")

    print(f"\nMean RPC: {df_degradation['RPC'].mean():+.4f} ({df_degradation['RPC_%'].mean():+.2f}%)")

    # Correlation analysis
    print("\n" + "-"*80)
    print("CORRELATION ANALYSIS")
    print("-"*80)
    print(f"Pearson correlation (r):     {corr_results['pearson_r']:+.4f}")
    print(f"  p-value:                   {corr_results['pearson_p']:.4f}")
    print(f"  Significance (α=0.05):     {'YES' if corr_results['pearson_p'] < 0.05 else 'NO'}")
    print(f"\nSpearman correlation (ρ):    {corr_results['spearman_r']:+.4f}")
    print(f"  p-value:                   {corr_results['spearman_p']:.4f}")
    print(f"\nLinear regression:")
    print(f"  Slope:                     {corr_results['regression_slope']:+.4f}")
    print(f"  Intercept:                 {corr_results['regression_intercept']:+.4f}")
    print(f"  R²:                        {corr_results['regression_r2']:.4f}")
    print(f"  p-value:                   {corr_results['regression_p']:.4f}")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    r = corr_results['pearson_r']
    p = corr_results['pearson_p']

    if abs(r) < 0.3:
        strength = "WEAK"
    elif abs(r) < 0.7:
        strength = "MODERATE"
    else:
        strength = "STRONG"

    print(f"\nCorrelation strength: {strength} (r = {r:+.3f})")

    if p >= 0.05:
        print(f"Statistical significance: NOT SIGNIFICANT (p = {p:.4f} ≥ 0.05)")
        print("\n✓ CONCLUSION:")
        print("  Distributional shift does NOT explain performance degradation.")
        print("  Classes with larger prevalence shifts do not show proportionally")
        print("  greater performance drops, ruling out class imbalance as the")
        print("  primary driver of generalization failure.")
    else:
        print(f"Statistical significance: SIGNIFICANT (p = {p:.4f} < 0.05)")
        print("\n✗ CONCLUSION:")
        print("  Distributional shift DOES contribute to performance degradation.")
        print("  Classes with larger prevalence differences between cohorts show")
        print("  greater performance drops in external validation.")

    print("="*80 + "\n")


def save_results(df_shift: pd.DataFrame, df_degradation: pd.DataFrame,
                 corr_results: Dict, output_dir: Path, task: str) -> None:
    """
    Save results to files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save prevalence shift
    csv_path = output_dir / f'{task}_prevalence_shift.csv'
    df_shift.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  ✓ Saved: {csv_path.name}")

    # Save performance degradation
    csv_path = output_dir / f'{task}_performance_degradation.csv'
    df_degradation.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  ✓ Saved: {csv_path.name}")

    # Save correlation results
    json_path = output_dir / f'{task}_distributional_shift_analysis.json'
    results = {
        'task': task,
        'prevalence_shift': df_shift.to_dict(orient='records'),
        'performance_degradation': df_degradation.to_dict(orient='records'),
        'correlation': corr_results
    }

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: {json_path.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Distributional Shift Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python distributional_shift_analysis.py \\
    --tcga_labels data/dataset_csv/tcga_pam50_labels.csv \\
    --cptac_labels data/dataset_csv/cptac_pam50_labels.csv \\
    --metrics results/pam50_performance_metrics.csv \\
    --task pam50 \\
    --output results/distributional_shift/
        """
    )

    # Input/output
    parser.add_argument('--tcga_labels', type=str, required=True,
                        help='Path to TCGA labels CSV')
    parser.add_argument('--cptac_labels', type=str, required=True,
                        help='Path to CPTAC labels CSV')
    parser.add_argument('--metrics', type=str, required=True,
                        help='Path to performance metrics CSV')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory')

    # Task
    parser.add_argument('--task', '-t', type=str, required=True,
                        help='Task name (pam50, er, pr, her2)')

    # Options
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--plot_format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'])

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("DISTRIBUTIONAL SHIFT ANALYSIS")
    print("="*80)
    print(f"Task:         {args.task.upper()}")
    print(f"TCGA labels:  {args.tcga_labels}")
    print(f"CPTAC labels: {args.cptac_labels}")
    print(f"Metrics:      {args.metrics}")
    print(f"Output:       {args.output}")
    print("="*80 + "\n")

    # Load data
    print("Loading data...")
    df_tcga = load_class_distribution(Path(args.tcga_labels))
    df_cptac = load_class_distribution(Path(args.cptac_labels))
    df_metrics = load_performance_metrics(Path(args.metrics))

    print(f"  ✓ TCGA:  {len(df_tcga)} samples")
    print(f"  ✓ CPTAC: {len(df_cptac)} samples")

    # Compute prevalence
    print("\nComputing prevalence...")
    prev_tcga = compute_prevalence(df_tcga)
    prev_cptac = compute_prevalence(df_cptac)

    df_shift = compute_prevalence_shift(prev_tcga, prev_cptac)
    print("  ✓ Prevalence shift computed")

    # Compute performance degradation
    print("\nComputing performance degradation...")
    df_degradation = compute_performance_degradation(df_metrics, args.task)
    print("  ✓ Performance degradation computed")

    # Correlation analysis
    print("\nPerforming correlation analysis...")
    corr_results = correlate_shift_and_degradation(df_shift, df_degradation)
    print("  ✓ Correlation analysis complete")

    # Print results
    print_results(df_shift, df_degradation, corr_results, args.task)

    # Save results
    print("Saving results...")
    save_results(df_shift, df_degradation, corr_results, output_dir, args.task)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualizations...")

        plot_prevalence_comparison(
            df_shift, args.task,
            output_dir / f'{args.task}_prevalence_comparison.{args.plot_format}'
        )

        plot_prevalence_shift(
            df_shift, args.task,
            output_dir / f'{args.task}_prevalence_shift.{args.plot_format}'
        )

        plot_correlation(
            corr_results, args.task,
            output_dir / f'{args.task}_correlation.{args.plot_format}'
        )

        plot_combined_analysis(
            df_shift, df_degradation, args.task,
            output_dir / f'{args.task}_combined_analysis.{args.plot_format}'
        )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/\n")


if __name__ == '__main__':
    main()