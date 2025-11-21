#!/usr/bin/env python3
"""
==============================================================================
STAIN NORMALIZATION ROBUSTNESS ANALYSIS
==============================================================================

This script performs statistical comparison between models trained/tested on
original vs Macenko-normalized H&E images to assess whether staining variability
contributes to domain shift.

Methodology:
- Compares predictions from two CLAM models:
  1. Original: trained on TCGA, tested on CPTAC (both unnormalized)
  2. Normalized: trained on normalized TCGA, tested on normalized CPTAC
- Uses McNemar's test to assess if stain normalization significantly affects performance
- Computes per-class accuracy changes (ΔAcc) for downstream regression analysis

References:
- McNemar's test: https://en.wikipedia.org/wiki/McNemar%27s_test
- Macenko normalization: Macenko et al. 2009, IEEE ISBI

Author: Claude Code
Date: 2025
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report
)
from statsmodels.stats.contingency_tables import mcnemar


# ==============================================================================
# CONFIGURATION
# ==============================================================================

PAM50_LABELS = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
IHC_RECEPTORS = ['ER', 'PR', 'HER2']
IHC_STATUS_LABELS = {
    'ER': ['ER-negative', 'ER-positive'],
    'PR': ['PR-negative', 'PR-positive'],
    'HER2': ['HER2-negative', 'HER2-positive']
}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_predictions(file_path: Path) -> Dict:
    """
    Load predictions from pickle or numpy file.

    Expected format (pickle dict):
        {
            'slide_ids': list of slide identifiers,
            'Y': array of true labels (N,),
            'Y_hat': array of predicted labels (N,),
            'probs': array of prediction probabilities (N, n_classes)
        }

    Args:
        file_path: Path to predictions file (.pkl or .npy)

    Returns:
        Dictionary with predictions data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {file_path}")

    if file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.suffix == '.npy':
        data = np.load(file_path, allow_pickle=True).item()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Validate required keys
    required_keys = ['Y', 'Y_hat']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys in predictions: {missing_keys}")

    return data


def validate_predictions(pred_original: Dict, pred_normalized: Dict) -> None:
    """
    Validate that both prediction sets are compatible.

    Args:
        pred_original: Predictions from original model
        pred_normalized: Predictions from normalized model

    Raises:
        ValueError: If predictions are incompatible
    """
    n_orig = len(pred_original['Y'])
    n_norm = len(pred_normalized['Y'])

    if n_orig != n_norm:
        raise ValueError(
            f"Prediction arrays have different lengths: "
            f"original={n_orig}, normalized={n_norm}"
        )

    # Check if labels match
    if not np.array_equal(pred_original['Y'], pred_normalized['Y']):
        print("WARNING: True labels differ between original and normalized predictions!")
        print("This may indicate mismatched test sets.")


# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================

def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: List[str]) -> pd.DataFrame:
    """
    Compute per-class accuracy, precision, recall, F1.

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        class_names: List of class names

    Returns:
        DataFrame with per-class metrics
    """
    metrics = []

    for class_idx, class_name in enumerate(class_names):
        mask = (y_true == class_idx)
        n_samples = mask.sum()

        if n_samples == 0:
            metrics.append({
                'Class': class_name,
                'N_samples': 0,
                'Accuracy': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1': np.nan
            })
            continue

        # Per-class accuracy
        acc = accuracy_score(y_true[mask], y_pred[mask])

        # Binary classification metrics for this class
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)

        # Compute precision, recall, F1 (handle warnings for classes with no predictions)
        with np.errstate(divide='ignore', invalid='ignore'):
            report = classification_report(
                y_true_binary, y_pred_binary,
                output_dict=True,
                zero_division=0
            )

        metrics.append({
            'Class': class_name,
            'N_samples': int(n_samples),
            'Accuracy': acc,
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1': report['1']['f1-score']
        })

    return pd.DataFrame(metrics)


def compute_pr_auc(y_true: np.ndarray, probs: np.ndarray,
                   positive_class: int = 1) -> float:
    """
    Compute Precision-Recall AUC for binary classification.

    Args:
        y_true: True binary labels (N,)
        probs: Prediction probabilities (N,) or (N, 2)
        positive_class: Index of positive class (default: 1)

    Returns:
        PR-AUC score
    """
    # Handle multi-dimensional probability arrays
    if probs.ndim == 2:
        probs = probs[:, positive_class]

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, probs)

    # Compute AUC
    pr_auc = auc(recall, precision)

    return pr_auc


def compute_metrics_comparison(pred_original: Dict, pred_normalized: Dict,
                                task: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute and compare metrics between original and normalized models.

    Args:
        pred_original: Predictions from original model
        pred_normalized: Predictions from normalized model
        task: Task type ('pam50', 'er', 'pr', 'her2')

    Returns:
        Tuple of (comparison DataFrame, global metrics dict)
    """
    y_true = pred_original['Y']
    y_pred_orig = pred_original['Y_hat']
    y_pred_norm = pred_normalized['Y_hat']

    # Determine class names
    if task == 'pam50':
        class_names = PAM50_LABELS
    else:
        class_names = IHC_STATUS_LABELS[task.upper()]

    # Compute per-class metrics
    metrics_orig = compute_per_class_metrics(y_true, y_pred_orig, class_names)
    metrics_norm = compute_per_class_metrics(y_true, y_pred_norm, class_names)

    # Merge and compute deltas
    comparison = metrics_orig.merge(
        metrics_norm,
        on='Class',
        suffixes=('_Original', '_Normalized')
    )

    # Compute delta for each metric
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        comparison[f'Δ{metric}'] = (
            comparison[f'{metric}_Normalized'] - comparison[f'{metric}_Original']
        )

    # Global metrics
    global_metrics = {
        'accuracy_original': accuracy_score(y_true, y_pred_orig),
        'accuracy_normalized': accuracy_score(y_true, y_pred_norm),
        'macro_f1_original': f1_score(y_true, y_pred_orig, average='macro'),
        'macro_f1_normalized': f1_score(y_true, y_pred_norm, average='macro')
    }

    # For binary classification, compute PR-AUC
    if task != 'pam50' and 'probs' in pred_original and 'probs' in pred_normalized:
        try:
            global_metrics['pr_auc_original'] = compute_pr_auc(
                y_true, pred_original['probs'], positive_class=1
            )
            global_metrics['pr_auc_normalized'] = compute_pr_auc(
                y_true, pred_normalized['probs'], positive_class=1
            )
        except Exception as e:
            print(f"Warning: Could not compute PR-AUC: {e}")

    return comparison, global_metrics


# ==============================================================================
# STATISTICAL TESTING
# ==============================================================================

def mcnemar_test(y_true: np.ndarray, y_pred_1: np.ndarray,
                 y_pred_2: np.ndarray, exact: bool = True) -> Dict:
    """
    Perform McNemar's test to compare two classifiers.

    McNemar's test assesses whether two models have significantly different
    error rates on the same test set by comparing patterns of correct/incorrect
    predictions.

    H0: Both models have equal error rates
    H1: Models have different error rates

    Args:
        y_true: True labels (N,)
        y_pred_1: Predictions from model 1 (N,)
        y_pred_2: Predictions from model 2 (N,)
        exact: Use exact binomial test (recommended for small samples)

    Returns:
        Dictionary with test results
    """
    # Compute correctness for each model
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)

    # Build contingency table
    # Rows: model 1 (correct/incorrect)
    # Cols: model 2 (correct/incorrect)
    both_correct = np.sum(correct_1 & correct_2)
    only_1_correct = np.sum(correct_1 & ~correct_2)
    only_2_correct = np.sum(~correct_1 & correct_2)
    both_wrong = np.sum(~correct_1 & ~correct_2)

    contingency = np.array([
        [both_correct, only_1_correct],
        [only_2_correct, both_wrong]
    ])

    # Perform McNemar's test
    result = mcnemar(contingency, exact=exact)

    return {
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'contingency_table': contingency.tolist(),
        'both_correct': int(both_correct),
        'only_original_correct': int(only_1_correct),
        'only_normalized_correct': int(only_2_correct),
        'both_wrong': int(both_wrong),
        'significant': result.pvalue < 0.05,
        'interpretation': (
            'Stain normalization SIGNIFICANTLY affects performance (p < 0.05)'
            if result.pvalue < 0.05
            else 'Stain normalization DOES NOT significantly affect performance (p ≥ 0.05)'
        )
    }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_accuracy_comparison(comparison_df: pd.DataFrame, task: str,
                              output_path: Optional[Path] = None) -> None:
    """
    Create bar plot comparing per-class accuracy between original and normalized.

    Args:
        comparison_df: DataFrame with comparison metrics
        task: Task name for title
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(comparison_df))
    width = 0.35

    ax.bar(x - width/2, comparison_df['Accuracy_Original'], width,
           label='Original', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, comparison_df['Accuracy_Normalized'], width,
           label='Macenko Normalized', alpha=0.8, color='coral')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Per-Class Accuracy Comparison: {task.upper()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Class'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved figure: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_delta_accuracy(comparison_df: pd.DataFrame, task: str,
                        output_path: Optional[Path] = None) -> None:
    """
    Create bar plot showing ΔAccuracy for each class.

    Args:
        comparison_df: DataFrame with comparison metrics
        task: Task name for title
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if x > 0 else 'red' for x in comparison_df['ΔAccuracy']]

    ax.bar(comparison_df['Class'], comparison_df['ΔAccuracy'],
           color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('ΔAccuracy (Normalized - Original)', fontsize=12)
    ax.set_title(f'Change in Accuracy with Stain Normalization: {task.upper()}',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(comparison_df['Class'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved figure: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrices(y_true: np.ndarray, y_pred_orig: np.ndarray,
                            y_pred_norm: np.ndarray, class_names: List[str],
                            task: str, output_path: Optional[Path] = None) -> None:
    """
    Create side-by-side confusion matrices for original vs normalized.

    Args:
        y_true: True labels
        y_pred_orig: Predictions from original model
        y_pred_norm: Predictions from normalized model
        class_names: List of class names
        task: Task name for title
        output_path: Optional path to save figure
    """
    cm_orig = confusion_matrix(y_true, y_pred_orig)
    cm_norm = confusion_matrix(y_true, y_pred_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Original Model', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_title('Macenko Normalized Model', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)

    fig.suptitle(f'Confusion Matrices: {task.upper()}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved figure: {output_path}")
    else:
        plt.show()

    plt.close()


# ==============================================================================
# REPORTING
# ==============================================================================

def print_results(comparison_df: pd.DataFrame, global_metrics: Dict,
                  mcnemar_results: Dict, task: str) -> None:
    """
    Print formatted results to console.

    Args:
        comparison_df: DataFrame with metric comparisons
        global_metrics: Dictionary of global metrics
        mcnemar_results: Dictionary with McNemar's test results
        task: Task name
    """
    print("\n" + "="*80)
    print(f"STAIN NORMALIZATION ANALYSIS: {task.upper()}")
    print("="*80)

    # Global metrics
    print("\n" + "-"*80)
    print("GLOBAL METRICS")
    print("-"*80)
    print(f"Accuracy:    {global_metrics['accuracy_original']:.4f} → "
          f"{global_metrics['accuracy_normalized']:.4f} "
          f"(Δ = {global_metrics['accuracy_normalized'] - global_metrics['accuracy_original']:+.4f})")
    print(f"Macro F1:    {global_metrics['macro_f1_original']:.4f} → "
          f"{global_metrics['macro_f1_normalized']:.4f} "
          f"(Δ = {global_metrics['macro_f1_normalized'] - global_metrics['macro_f1_original']:+.4f})")

    if 'pr_auc_original' in global_metrics:
        print(f"PR-AUC:      {global_metrics['pr_auc_original']:.4f} → "
              f"{global_metrics['pr_auc_normalized']:.4f} "
              f"(Δ = {global_metrics['pr_auc_normalized'] - global_metrics['pr_auc_original']:+.4f})")

    # Per-class metrics
    print("\n" + "-"*80)
    print("PER-CLASS ACCURACY")
    print("-"*80)
    for _, row in comparison_df.iterrows():
        print(f"{row['Class']:15s}: {row['Accuracy_Original']:.4f} → "
              f"{row['Accuracy_Normalized']:.4f} (Δ = {row['ΔAccuracy']:+.4f})")

    # McNemar's test
    print("\n" + "-"*80)
    print("McNEMAR'S TEST")
    print("-"*80)
    print(f"Test statistic (χ²): {mcnemar_results['statistic']:.4f}")
    print(f"p-value:             {mcnemar_results['p_value']:.4f}")
    print(f"\nContingency Table:")
    print(f"  Both correct:           {mcnemar_results['both_correct']}")
    print(f"  Only original correct:  {mcnemar_results['only_original_correct']}")
    print(f"  Only normalized correct:{mcnemar_results['only_normalized_correct']}")
    print(f"  Both wrong:             {mcnemar_results['both_wrong']}")
    print(f"\nResult: {'SIGNIFICANT' if mcnemar_results['significant'] else 'NOT SIGNIFICANT'} (α = 0.05)")
    print(f"\nInterpretation:")
    print(f"  {mcnemar_results['interpretation']}")

    print("\n" + "="*80 + "\n")


def save_results(comparison_df: pd.DataFrame, global_metrics: Dict,
                 mcnemar_results: Dict, output_dir: Path, task: str) -> None:
    """
    Save results to files.

    Args:
        comparison_df: DataFrame with metric comparisons
        global_metrics: Dictionary of global metrics
        mcnemar_results: Dictionary with McNemar's test results
        output_dir: Output directory
        task: Task name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    csv_path = output_dir / f'{task}_per_class_comparison.csv'
    comparison_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  ✓ Saved per-class comparison: {csv_path}")

    # Save all results as JSON
    results = {
        'task': task,
        'global_metrics': global_metrics,
        'mcnemar_test': mcnemar_results,
        'per_class_metrics': comparison_df.to_dict(orient='records')
    }

    json_path = output_dir / f'{task}_stain_normalization_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results JSON: {json_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stain Normalization Robustness Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PAM50 subtyping
  python stain_normalization_analysis.py \\
    --original results/pam50/hold_out_predictions.pkl \\
    --normalized results/pam50_macenko/hold_out_predictions.pkl \\
    --task pam50 \\
    --output results/stain_analysis/

  # ER status prediction
  python stain_normalization_analysis.py \\
    --original results/er/hold_out_predictions.pkl \\
    --normalized results/er_macenko/hold_out_predictions.pkl \\
    --task er \\
    --output results/stain_analysis/
        """
    )

    # Input/output arguments
    parser.add_argument('--original', '-o', type=str, required=True,
                        help='Path to predictions from original model (.pkl or .npy)')
    parser.add_argument('--normalized', '-n', type=str, required=True,
                        help='Path to predictions from Macenko-normalized model (.pkl or .npy)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for results and figures')

    # Task configuration
    parser.add_argument('--task', '-t', type=str, required=True,
                        choices=['pam50', 'er', 'pr', 'her2'],
                        help='Classification task')

    # Statistical testing options
    parser.add_argument('--mcnemar_exact', action='store_true', default=True,
                        help='Use exact binomial test for McNemar (recommended)')

    # Visualization options
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--plot_format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format for plots (default: png)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("STAIN NORMALIZATION ROBUSTNESS ANALYSIS")
    print("="*80)
    print(f"Task: {args.task.upper()}")
    print(f"Original predictions:   {args.original}")
    print(f"Normalized predictions: {args.normalized}")
    print(f"Output directory:       {args.output}")
    print("="*80 + "\n")

    # Load predictions
    print("Loading predictions...")
    pred_original = load_predictions(args.original)
    pred_normalized = load_predictions(args.normalized)

    # Validate
    validate_predictions(pred_original, pred_normalized)

    print(f"  ✓ Loaded {len(pred_original['Y'])} samples")

    # Compute metrics
    print("\nComputing metrics...")
    comparison_df, global_metrics = compute_metrics_comparison(
        pred_original, pred_normalized, args.task
    )
    print("  ✓ Metrics computed")

    # McNemar's test
    print("\nPerforming McNemar's test...")
    mcnemar_results = mcnemar_test(
        pred_original['Y'],
        pred_original['Y_hat'],
        pred_normalized['Y_hat'],
        exact=args.mcnemar_exact
    )
    print("  ✓ Statistical test completed")

    # Print results
    print_results(comparison_df, global_metrics, mcnemar_results, args.task)

    # Save results
    print("Saving results...")
    save_results(comparison_df, global_metrics, mcnemar_results, output_dir, args.task)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualizations...")

        # Determine class names
        if args.task == 'pam50':
            class_names = PAM50_LABELS
        else:
            class_names = IHC_STATUS_LABELS[args.task.upper()]

        # Accuracy comparison
        plot_accuracy_comparison(
            comparison_df, args.task,
            output_dir / f'{args.task}_accuracy_comparison.{args.plot_format}'
        )

        # Delta accuracy
        plot_delta_accuracy(
            comparison_df, args.task,
            output_dir / f'{args.task}_delta_accuracy.{args.plot_format}'
        )

        # Confusion matrices
        plot_confusion_matrices(
            pred_original['Y'],
            pred_original['Y_hat'],
            pred_normalized['Y_hat'],
            class_names,
            args.task,
            output_dir / f'{args.task}_confusion_matrices.{args.plot_format}'
        )

        print("  ✓ Visualizations saved")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nKey finding:")
    print(f"  {mcnemar_results['interpretation']}")
    print("\n")


if __name__ == '__main__':
    main()
