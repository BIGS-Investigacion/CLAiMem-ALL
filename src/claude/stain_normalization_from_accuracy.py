#!/usr/bin/env python3
"""
==============================================================================
STAIN NORMALIZATION ANALYSIS FROM ACCURACY DATA
==============================================================================

Simplified script that takes per-class accuracy data directly and performs
McNemar's test analysis to assess stain normalization impact.

This script is designed for cases where you already have computed accuracies
and just need to perform statistical testing and visualization.

Input format (CSV):
    Class,N_samples,Accuracy_Original,Accuracy_Normalized,N_correct_orig,N_correct_norm
    LumA,150,0.75,0.78,112,117
    LumB,80,0.65,0.68,52,54
    ...

Or JSON format:
    {
        "pam50": {
            "LumA": {"n_samples": 150, "acc_original": 0.75, "acc_normalized": 0.78},
            ...
        }
    }

Author: Claude Code
Date: 2025
"""

import argparse
import json
import sys
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

def load_accuracy_data_csv(file_path: Path) -> pd.DataFrame:
    """
    Load accuracy data from CSV file.

    Expected columns:
        - Class: class name (e.g., 'LumA', 'ER-positive')
        - N_samples: number of test samples for this class
        - Accuracy_Original: accuracy without normalization
        - Accuracy_Normalized: accuracy with Macenko normalization

    Optional columns (for exact McNemar test):
        - N_correct_orig: number correctly classified by original model
        - N_correct_norm: number correctly classified by normalized model
        - N_both_correct: samples correctly classified by both
        - N_only_orig: samples only original got correct
        - N_only_norm: samples only normalized got correct
        - N_both_wrong: samples both got wrong

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with accuracy data
    """
    df = pd.read_csv(file_path)

    required_cols = ['Class', 'N_samples', 'Accuracy_Original', 'Accuracy_Normalized']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def load_accuracy_data_json(file_path: Path) -> pd.DataFrame:
    """
    Load accuracy data from JSON file.

    Expected format:
        {
            "Class1": {
                "n_samples": 100,
                "acc_original": 0.75,
                "acc_normalized": 0.78
            },
            ...
        }

    Args:
        file_path: Path to JSON file

    Returns:
        DataFrame with accuracy data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    records = []
    for class_name, metrics in data.items():
        records.append({
            'Class': class_name,
            'N_samples': metrics['n_samples'],
            'Accuracy_Original': metrics['acc_original'],
            'Accuracy_Normalized': metrics['acc_normalized']
        })

    return pd.DataFrame(records)


def create_dataframe_from_dict(data: Dict) -> pd.DataFrame:
    """
    Create DataFrame from dictionary of accuracy data.

    Args:
        data: Dictionary with structure:
            {
                'classes': ['LumA', 'LumB', ...],
                'n_samples': [150, 80, ...],
                'acc_original': [0.75, 0.65, ...],
                'acc_normalized': [0.78, 0.68, ...]
            }

    Returns:
        DataFrame with accuracy data
    """
    df = pd.DataFrame({
        'Class': data['classes'],
        'N_samples': data['n_samples'],
        'Accuracy_Original': data['acc_original'],
        'Accuracy_Normalized': data['acc_normalized']
    })

    return df


# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================

def compute_delta_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta (change) metrics.

    Args:
        df: DataFrame with original and normalized accuracies

    Returns:
        DataFrame with added delta columns
    """
    df = df.copy()

    # Absolute change
    df['ΔAccuracy'] = df['Accuracy_Normalized'] - df['Accuracy_Original']

    # Relative change (%)
    df['ΔAccuracy_Relative_%'] = (
        (df['Accuracy_Normalized'] - df['Accuracy_Original']) /
        df['Accuracy_Original'] * 100
    )

    # Benefit indicator
    df['Improvement'] = df['ΔAccuracy'] > 0

    return df


def compute_mcnemar_from_accuracies(df: pd.DataFrame,
                                    total_samples: Optional[int] = None) -> Dict:
    """
    Approximate McNemar's test from per-class accuracies.

    This is an approximation when we don't have the full contingency table.
    We estimate the number of discordant pairs (cases where models disagree).

    For exact test, provide contingency table columns in the input DataFrame.

    Args:
        df: DataFrame with per-class accuracies
        total_samples: Total number of test samples (if known)

    Returns:
        Dictionary with test results
    """
    # Check if exact contingency data is available
    has_exact = all(col in df.columns for col in
                    ['N_both_correct', 'N_only_orig', 'N_only_norm', 'N_both_wrong'])

    if has_exact:
        # Use exact contingency table
        both_correct = int(df['N_both_correct'].sum())
        only_orig = int(df['N_only_orig'].sum())
        only_norm = int(df['N_only_norm'].sum())
        both_wrong = int(df['N_both_wrong'].sum())

        # McNemar statistic (with continuity correction)
        numerator = (abs(only_orig - only_norm) - 1) ** 2
        denominator = only_orig + only_norm

        if denominator == 0:
            chi2 = 0.0
            p_value = 1.0
        else:
            chi2 = numerator / denominator
            p_value = 1 - stats.chi2.cdf(chi2, df=1)

        method = 'exact'

    else:
        # Approximate from accuracies
        if total_samples is None:
            total_samples = int(df['N_samples'].sum())

        # Estimate correctly classified samples
        n_correct_orig = (df['Accuracy_Original'] * df['N_samples']).sum()
        n_correct_norm = (df['Accuracy_Normalized'] * df['N_samples']).sum()

        # Conservative estimate: assume maximum disagreement
        # This gives us an upper bound on the test statistic
        diff = abs(n_correct_norm - n_correct_orig)

        # McNemar approximation (assumes discordant pairs)
        chi2 = (diff ** 2) / (n_correct_orig + n_correct_norm - diff)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        both_correct = int(min(n_correct_orig, n_correct_norm))
        only_orig = int(max(0, n_correct_orig - n_correct_norm))
        only_norm = int(max(0, n_correct_norm - n_correct_orig))
        both_wrong = int(total_samples - max(n_correct_orig, n_correct_norm))

        method = 'approximate'

    return {
        'method': method,
        'statistic': float(chi2),
        'p_value': float(p_value),
        'both_correct': both_correct,
        'only_original_correct': only_orig,
        'only_normalized_correct': only_norm,
        'both_wrong': both_wrong,
        'significant': p_value < 0.05,
        'interpretation': (
            'Stain normalization SIGNIFICANTLY affects performance (p < 0.05)'
            if p_value < 0.05
            else 'Stain normalization DOES NOT significantly affect performance (p ≥ 0.05)'
        ),
        'warning': '' if method == 'exact' else
                  'Using approximate McNemar test. For exact results, provide contingency table.'
    }


def compute_global_metrics(df: pd.DataFrame) -> Dict:
    """
    Compute weighted global metrics.

    Args:
        df: DataFrame with per-class accuracies

    Returns:
        Dictionary with global metrics
    """
    total_samples = df['N_samples'].sum()

    # Weighted average accuracy
    acc_orig = (df['Accuracy_Original'] * df['N_samples']).sum() / total_samples
    acc_norm = (df['Accuracy_Normalized'] * df['N_samples']).sum() / total_samples

    return {
        'total_samples': int(total_samples),
        'accuracy_original': float(acc_orig),
        'accuracy_normalized': float(acc_norm),
        'delta_accuracy': float(acc_norm - acc_orig),
        'delta_accuracy_relative_%': float((acc_norm - acc_orig) / acc_orig * 100)
    }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_accuracy_comparison(df: pd.DataFrame, task: str,
                              output_path: Optional[Path] = None) -> None:
    """
    Create bar plot comparing per-class accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width/2, df['Accuracy_Original'], width,
                   label='Original', alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, df['Accuracy_Normalized'], width,
                   label='Macenko Normalized', alpha=0.8, color='coral', edgecolor='black')

    # Add sample sizes as text
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, -0.08, f'n={row["N_samples"]}',
                ha='center', va='top', fontsize=9, color='gray')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class Accuracy: Original vs Macenko Normalized\n{task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Class'], rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0, color='black', linewidth=0.8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_delta_accuracy(df: pd.DataFrame, task: str,
                        output_path: Optional[Path] = None) -> None:
    """
    Create bar plot showing change in accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray'
              for x in df['ΔAccuracy']]

    bars = ax.bar(df['Class'], df['ΔAccuracy'], color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, delta in zip(bars, df['ΔAccuracy']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{delta:+.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('ΔAccuracy (Normalized - Original)', fontsize=12, fontweight='bold')
    ax.set_title(f'Change in Accuracy with Stain Normalization\n{task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(df['Class'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Improvement'),
        Patch(facecolor='red', alpha=0.7, label='Degradation')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_accuracy_heatmap(df: pd.DataFrame, task: str,
                          output_path: Optional[Path] = None) -> None:
    """
    Create heatmap showing accuracies for each class.
    """
    # Prepare data for heatmap
    data = df[['Accuracy_Original', 'Accuracy_Normalized']].T
    data.columns = df['Class']

    fig, ax = plt.subplots(figsize=(12, 4))

    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Accuracy'}, linewidths=0.5,
                yticklabels=['Original', 'Macenko Normalized'], ax=ax)

    ax.set_title(f'Accuracy Heatmap: {task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')

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

def print_results(df: pd.DataFrame, global_metrics: Dict,
                  mcnemar_results: Dict, task: str) -> None:
    """
    Print formatted results to console.
    """
    print("\n" + "="*80)
    print(f"STAIN NORMALIZATION ANALYSIS: {task.upper()}")
    print("="*80)

    # Global metrics
    print("\n" + "-"*80)
    print("GLOBAL METRICS")
    print("-"*80)
    print(f"Total samples:  {global_metrics['total_samples']}")
    print(f"Accuracy:       {global_metrics['accuracy_original']:.4f} → "
          f"{global_metrics['accuracy_normalized']:.4f} "
          f"(Δ = {global_metrics['delta_accuracy']:+.4f}, "
          f"{global_metrics['delta_accuracy_relative_%']:+.2f}%)")

    # Per-class results
    print("\n" + "-"*80)
    print("PER-CLASS ACCURACY")
    print("-"*80)
    print(f"{'Class':<15} {'N':>6} {'Original':>10} {'Normalized':>10} "
          f"{'ΔAcc':>10} {'ΔAcc%':>10}")
    print("-"*80)

    for _, row in df.iterrows():
        print(f"{row['Class']:<15} {row['N_samples']:>6} "
              f"{row['Accuracy_Original']:>10.4f} {row['Accuracy_Normalized']:>10.4f} "
              f"{row['ΔAccuracy']:>+10.4f} {row['ΔAccuracy_Relative_%']:>+9.2f}%")

    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)
    n_improved = (df['ΔAccuracy'] > 0).sum()
    n_degraded = (df['ΔAccuracy'] < 0).sum()
    n_unchanged = (df['ΔAccuracy'] == 0).sum()

    print(f"Classes improved:  {n_improved}/{len(df)}")
    print(f"Classes degraded:  {n_degraded}/{len(df)}")
    print(f"Classes unchanged: {n_unchanged}/{len(df)}")
    print(f"\nMean ΔAccuracy:    {df['ΔAccuracy'].mean():+.4f}")
    print(f"Median ΔAccuracy:  {df['ΔAccuracy'].median():+.4f}")
    print(f"Max improvement:   {df['ΔAccuracy'].max():+.4f} ({df.loc[df['ΔAccuracy'].idxmax(), 'Class']})")
    if df['ΔAccuracy'].min() < 0:
        print(f"Max degradation:   {df['ΔAccuracy'].min():+.4f} ({df.loc[df['ΔAccuracy'].idxmin(), 'Class']})")

    # McNemar's test
    print("\n" + "-"*80)
    print("McNEMAR'S TEST")
    print("-"*80)
    print(f"Method:              {mcnemar_results['method'].upper()}")
    print(f"Test statistic (χ²): {mcnemar_results['statistic']:.4f}")
    print(f"p-value:             {mcnemar_results['p_value']:.4f}")

    print(f"\nContingency Table (estimated):")
    print(f"  Both correct:            {mcnemar_results['both_correct']}")
    print(f"  Only original correct:   {mcnemar_results['only_original_correct']}")
    print(f"  Only normalized correct: {mcnemar_results['only_normalized_correct']}")
    print(f"  Both wrong:              {mcnemar_results['both_wrong']}")

    print(f"\nResult: {'SIGNIFICANT' if mcnemar_results['significant'] else 'NOT SIGNIFICANT'} (α = 0.05)")

    if mcnemar_results['warning']:
        print(f"\n⚠ WARNING: {mcnemar_results['warning']}")

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print("="*80)
    print(f"{mcnemar_results['interpretation']}")

    if not mcnemar_results['significant']:
        print("\nConclusion: Staining variability does NOT appear to be a major contributor")
        print("to domain shift, as eliminating color differences did not significantly")
        print("improve external validation performance.")
    else:
        print("\nConclusion: Staining variability DOES contribute to domain shift.")
        print("Color normalization significantly affects model performance.")

    print("="*80 + "\n")


def save_results(df: pd.DataFrame, global_metrics: Dict,
                 mcnemar_results: Dict, output_dir: Path, task: str) -> None:
    """
    Save results to files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    csv_path = output_dir / f'{task}_stain_normalization_results.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  ✓ Saved: {csv_path.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stain Normalization Analysis from Accuracy Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. From CSV file:
     python stain_normalization_from_accuracy.py \\
       --input accuracies_pam50.csv \\
       --task pam50 \\
       --output results/stain_analysis/

  2. From JSON file:
     python stain_normalization_from_accuracy.py \\
       --input accuracies_er.json \\
       --task er \\
       --output results/stain_analysis/

  3. Interactive mode (will prompt for data):
     python stain_normalization_from_accuracy.py \\
       --interactive \\
       --task pam50 \\
       --output results/stain_analysis/

CSV format:
  Class,N_samples,Accuracy_Original,Accuracy_Normalized
  LumA,150,0.75,0.78
  LumB,80,0.65,0.68
  ...

JSON format:
  {
    "LumA": {"n_samples": 150, "acc_original": 0.75, "acc_normalized": 0.78},
    "LumB": {"n_samples": 80, "acc_original": 0.65, "acc_normalized": 0.68},
    ...
  }
        """
    )

    # Input/output
    parser.add_argument('--input', '-i', type=str,
                        help='Path to CSV or JSON file with accuracy data')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for results')

    # Task
    parser.add_argument('--task', '-t', type=str, required=True,
                        help='Task name (e.g., pam50, er, pr, her2)')

    # Options
    parser.add_argument('--interactive', action='store_true',
                        help='Enter data interactively')
    parser.add_argument('--total_samples', type=int,
                        help='Total number of test samples (for McNemar approximation)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--plot_format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format for plots')

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.interactive:
        parser.error("Either --input or --interactive must be specified")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("STAIN NORMALIZATION ANALYSIS FROM ACCURACY DATA")
    print("="*80)
    print(f"Task: {args.task.upper()}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    # Load data
    if args.interactive:
        print("Interactive mode not yet implemented.")
        print("Please provide data via --input CSV or JSON file.")
        sys.exit(1)
    else:
        print(f"Loading data from: {args.input}")
        input_path = Path(args.input)

        if not input_path.exists():
            print(f"ERROR: File not found: {args.input}")
            sys.exit(1)

        if input_path.suffix == '.csv':
            df = load_accuracy_data_csv(input_path)
        elif input_path.suffix == '.json':
            df = load_accuracy_data_json(input_path)
        else:
            print(f"ERROR: Unsupported file format: {input_path.suffix}")
            print("Supported formats: .csv, .json")
            sys.exit(1)

        print(f"  ✓ Loaded {len(df)} classes\n")

    # Compute metrics
    print("Computing metrics...")
    df = compute_delta_metrics(df)
    global_metrics = compute_global_metrics(df)
    mcnemar_results = compute_mcnemar_from_accuracies(df, args.total_samples)
    print("  ✓ Metrics computed\n")

    # Print results
    print_results(df, global_metrics, mcnemar_results, args.task)

    # Save results
    print("Saving results...")
    save_results(df, global_metrics, mcnemar_results, output_dir, args.task)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualizations...")

        plot_accuracy_comparison(
            df, args.task,
            output_dir / f'{args.task}_accuracy_comparison.{args.plot_format}'
        )

        plot_delta_accuracy(
            df, args.task,
            output_dir / f'{args.task}_delta_accuracy.{args.plot_format}'
        )

        plot_accuracy_heatmap(
            df, args.task,
            output_dir / f'{args.task}_accuracy_heatmap.{args.plot_format}'
        )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nKey finding: {mcnemar_results['interpretation']}")
    print("\n")


if __name__ == '__main__':
    main()
