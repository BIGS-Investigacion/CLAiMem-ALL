#!/usr/bin/env python3
"""
Extract Performance Metrics for Distributional Shift Analysis

Extracts PR-AUC and F1-score metrics from CLAM results (fold CSV files) and
generates summary CSV files for distributional shift analysis.

For binary tasks (ER, PR, HER2): Computes PR-AUC for both positive and negative classes
For multiclass tasks (PAM50): Computes F1-score per class

Author: Claude Code
Date: 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve


# ==============================================================================
# BINARY CLASSIFICATION METRICS (IHC)
# ==============================================================================

def calculate_pr_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Calculate PR-AUC for both positive and negative classes in binary classification.

    Args:
        y_true: True labels (0 or 1)
        y_score: Predicted probabilities for positive class

    Returns:
        Dictionary with PR-AUC for positive and negative classes
    """
    # PR-AUC for positive class (label = 1)
    pr_auc_positive = average_precision_score(y_true, y_score)

    # PR-AUC for negative class (label = 0)
    # Invert: treat 0 as positive class
    y_true_negative = 1 - y_true
    y_score_negative = 1 - y_score
    pr_auc_negative = average_precision_score(y_true_negative, y_score_negative)

    return {
        'positive': pr_auc_positive,
        'negative': pr_auc_negative
    }


def process_binary_fold(csv_path: Path,
                       label_col: str = 'Y',
                       prob_col: str = 'Y_hat') -> Dict[str, float]:
    """
    Process a single fold CSV for binary classification.

    Args:
        csv_path: Path to fold CSV file
        label_col: Column name for true labels
        prob_col: Column name for predicted probabilities

    Returns:
        Dictionary with PR-AUC for both classes
    """
    df = pd.read_csv(csv_path)

    y_true = df[label_col].values
    y_score = df[prob_col].values

    return calculate_pr_auc_binary(y_true, y_score)


def aggregate_binary_folds(fold_files: List[Path],
                           label_col: str = 'Y',
                           prob_col: str = 'Y_hat') -> Dict[str, Dict[str, float]]:
    """
    Aggregate PR-AUC across multiple folds for binary classification.

    Args:
        fold_files: List of paths to fold CSV files
        label_col: Column name for true labels
        prob_col: Column name for predicted probabilities

    Returns:
        Dictionary with mean and std PR-AUC for both classes
    """
    pr_auc_pos_list = []
    pr_auc_neg_list = []

    for fold_file in fold_files:
        results = process_binary_fold(fold_file, label_col, prob_col)
        pr_auc_pos_list.append(results['positive'])
        pr_auc_neg_list.append(results['negative'])

    return {
        'positive': {
            'mean': np.mean(pr_auc_pos_list),
            'std': np.std(pr_auc_pos_list),
            'values': pr_auc_pos_list
        },
        'negative': {
            'mean': np.mean(pr_auc_neg_list),
            'std': np.std(pr_auc_neg_list),
            'values': pr_auc_neg_list
        }
    }


# ==============================================================================
# MULTICLASS CLASSIFICATION METRICS (PAM50)
# ==============================================================================

def calculate_f1_multiclass(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           labels: List[str]) -> Dict[str, float]:
    """
    Calculate per-class F1-score for multiclass classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels

    Returns:
        Dictionary mapping class names to F1-scores
    """
    # Compute F1-score per class
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=range(len(labels)))

    return {label: f1 for label, f1 in zip(labels, f1_per_class)}


def process_multiclass_fold(csv_path: Path,
                            label_col: str = 'Y',
                            pred_col: str = 'Y_hat',
                            class_names: List[str] = None) -> Dict[str, float]:
    """
    Process a single fold CSV for multiclass classification.

    Args:
        csv_path: Path to fold CSV file
        label_col: Column name for true labels
        pred_col: Column name for predictions (or use argmax of probabilities)
        class_names: List of class names

    Returns:
        Dictionary with F1-score per class
    """
    df = pd.read_csv(csv_path)

    y_true = df[label_col].values

    # If pred_col contains probabilities, find the predicted class
    if pred_col in df.columns:
        y_pred = df[pred_col].values
    else:
        # Assume probability columns: p_0, p_1, p_2, ...
        prob_cols = [col for col in df.columns if col.startswith('p_')]
        if prob_cols:
            y_pred = df[prob_cols].values.argmax(axis=1)
        else:
            raise ValueError(f"Cannot find prediction or probability columns in {csv_path}")

    return calculate_f1_multiclass(y_true, y_pred, class_names)


def aggregate_multiclass_folds(fold_files: List[Path],
                               label_col: str = 'Y',
                               pred_col: str = 'Y_hat',
                               class_names: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Aggregate F1-scores across multiple folds for multiclass classification.

    Args:
        fold_files: List of paths to fold CSV files
        label_col: Column name for true labels
        pred_col: Column name for predictions
        class_names: List of class names

    Returns:
        Dictionary with mean and std F1-score per class
    """
    f1_per_class = {class_name: [] for class_name in class_names}

    for fold_file in fold_files:
        results = process_multiclass_fold(fold_file, label_col, pred_col, class_names)
        for class_name, f1 in results.items():
            f1_per_class[class_name].append(f1)

    return {
        class_name: {
            'mean': np.mean(f1_values),
            'std': np.std(f1_values),
            'values': f1_values
        }
        for class_name, f1_values in f1_per_class.items()
    }


# ==============================================================================
# RESULTS AGGREGATION
# ==============================================================================

def create_metrics_csv_binary(mccv_results: Dict,
                              ho_results: Dict,
                              task: str,
                              class_names: Tuple[str, str]) -> pd.DataFrame:
    """
    Create metrics CSV for binary classification task.

    Args:
        mccv_results: MCCV aggregated results
        ho_results: Hold-out results
        task: Task name (e.g., 'ER', 'PR', 'HER2')
        class_names: Tuple of (negative_class_name, positive_class_name)

    Returns:
        DataFrame with columns: Class, PR_AUC_MCCV, PR_AUC_HO
    """
    negative_class, positive_class = class_names

    data = [
        {
            'Class': negative_class,
            'PR_AUC_MCCV': mccv_results['negative']['mean'],
            'PR_AUC_HO': ho_results['negative']
        },
        {
            'Class': positive_class,
            'PR_AUC_MCCV': mccv_results['positive']['mean'],
            'PR_AUC_HO': ho_results['positive']
        }
    ]

    return pd.DataFrame(data)


def create_metrics_csv_multiclass(mccv_results: Dict,
                                  ho_results: Dict,
                                  class_names: List[str]) -> pd.DataFrame:
    """
    Create metrics CSV for multiclass classification task.

    Args:
        mccv_results: MCCV aggregated results
        ho_results: Hold-out results
        class_names: List of class names

    Returns:
        DataFrame with columns: Class, F1_MCCV, F1_HO
    """
    data = []

    for class_name in class_names:
        data.append({
            'Class': class_name,
            'F1_MCCV': mccv_results[class_name]['mean'],
            'F1_HO': ho_results[class_name]
        })

    return pd.DataFrame(data)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract metrics for distributional shift analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Binary task (IHC)
  python extract_metrics_for_distributional_analysis.py \\
    --task her2 \\
    --mccv_dir results/her2/mccv/split_0/ \\
    --ho_file results/her2/holdout/fold_0.csv \\
    --output results/distributional_shift/her2_metrics.csv

  # Multiclass task (PAM50)
  python extract_metrics_for_distributional_analysis.py \\
    --task pam50 \\
    --mccv_dir results/pam50/mccv/split_0/ \\
    --ho_file results/pam50/holdout/fold_0.csv \\
    --output results/distributional_shift/pam50_metrics.csv \\
    --class_names Basal Her2-enriched LumA LumB Normal
        """
    )

    # Task specification
    parser.add_argument('--task', type=str, required=True,
                       choices=['pam50', 'er', 'pr', 'her2'],
                       help='Task name')

    # Input paths
    parser.add_argument('--mccv_dir', type=str, required=True,
                       help='Directory containing MCCV fold CSV files (fold_0.csv, fold_1.csv, ...)')
    parser.add_argument('--ho_file', type=str, required=True,
                       help='Path to hold-out CSV file')

    # Output
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output CSV file path')

    # Column names
    parser.add_argument('--label_col', type=str, default='Y',
                       help='Column name for true labels (default: Y)')
    parser.add_argument('--prob_col', type=str, default='Y_hat',
                       help='Column name for predictions/probabilities (default: Y_hat)')

    # Class names (for multiclass)
    parser.add_argument('--class_names', type=str, nargs='+',
                       help='Class names for multiclass tasks (e.g., Basal Her2-enriched LumA LumB Normal)')

    args = parser.parse_args()

    mccv_dir = Path(args.mccv_dir)
    ho_file = Path(args.ho_file)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXTRACTING METRICS FOR DISTRIBUTIONAL SHIFT ANALYSIS")
    print("="*80)
    print(f"Task:     {args.task.upper()}")
    print(f"MCCV dir: {mccv_dir}")
    print(f"HO file:  {ho_file}")
    print(f"Output:   {output_file}")
    print("="*80 + "\n")

    # Find all fold files in MCCV directory
    fold_files = sorted(mccv_dir.glob('fold_*.csv'))

    if not fold_files:
        raise FileNotFoundError(f"No fold_*.csv files found in {mccv_dir}")

    print(f"Found {len(fold_files)} MCCV fold files")

    # Process based on task type
    if args.task in ['er', 'pr', 'her2']:
        # Binary classification
        print("\nProcessing binary classification task...")

        # Aggregate MCCV results
        print("  Aggregating MCCV folds...")
        mccv_results = aggregate_binary_folds(fold_files, args.label_col, args.prob_col)

        # Process hold-out
        print("  Processing hold-out...")
        ho_results = process_binary_fold(ho_file, args.label_col, args.prob_col)

        # Define class names
        receptor = args.task.upper()
        class_names = (f"{receptor}-negative", f"{receptor}-positive")

        # Create metrics CSV
        df_metrics = create_metrics_csv_binary(mccv_results, ho_results, args.task, class_names)

        # Print results
        print("\n" + "-"*80)
        print("RESULTS")
        print("-"*80)
        print(f"{'Class':<20} {'PR-AUC (MCCV)':>15} {'PR-AUC (HO)':>15} {'Δ':>10}")
        print("-"*80)
        for _, row in df_metrics.iterrows():
            delta = row['PR_AUC_HO'] - row['PR_AUC_MCCV']
            print(f"{row['Class']:<20} {row['PR_AUC_MCCV']:>15.4f} "
                  f"{row['PR_AUC_HO']:>15.4f} {delta:>+10.4f}")
        print("-"*80)

    else:
        # Multiclass classification (PAM50)
        print("\nProcessing multiclass classification task...")

        if not args.class_names:
            args.class_names = ['Basal', 'Her2-enriched', 'LumA', 'LumB', 'Normal']
            print(f"  Using default PAM50 class names: {args.class_names}")

        # Aggregate MCCV results
        print("  Aggregating MCCV folds...")
        mccv_results = aggregate_multiclass_folds(
            fold_files, args.label_col, args.prob_col, args.class_names
        )

        # Process hold-out
        print("  Processing hold-out...")
        ho_results = process_multiclass_fold(
            ho_file, args.label_col, args.prob_col, args.class_names
        )

        # Create metrics CSV
        df_metrics = create_metrics_csv_multiclass(mccv_results, ho_results, args.class_names)

        # Print results
        print("\n" + "-"*80)
        print("RESULTS")
        print("-"*80)
        print(f"{'Class':<15} {'F1 (MCCV)':>12} {'F1 (HO)':>12} {'Δ':>10}")
        print("-"*80)
        for _, row in df_metrics.iterrows():
            delta = row['F1_HO'] - row['F1_MCCV']
            print(f"{row['Class']:<15} {row['F1_MCCV']:>12.4f} "
                  f"{row['F1_HO']:>12.4f} {delta:>+10.4f}")
        print("-"*80)

    # Save metrics
    df_metrics.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n✓ Metrics saved to: {output_file}")

    # Save detailed JSON
    json_file = output_file.with_suffix('.json')

    if args.task in ['er', 'pr', 'her2']:
        detailed_results = {
            'task': args.task,
            'task_type': 'binary',
            'mccv': {
                'n_folds': len(fold_files),
                'positive': {
                    'mean': float(mccv_results['positive']['mean']),
                    'std': float(mccv_results['positive']['std']),
                    'values': [float(v) for v in mccv_results['positive']['values']]
                },
                'negative': {
                    'mean': float(mccv_results['negative']['mean']),
                    'std': float(mccv_results['negative']['std']),
                    'values': [float(v) for v in mccv_results['negative']['values']]
                }
            },
            'holdout': {
                'positive': float(ho_results['positive']),
                'negative': float(ho_results['negative'])
            }
        }
    else:
        detailed_results = {
            'task': args.task,
            'task_type': 'multiclass',
            'class_names': args.class_names,
            'mccv': {
                'n_folds': len(fold_files),
                'per_class': {
                    class_name: {
                        'mean': float(results['mean']),
                        'std': float(results['std']),
                        'values': [float(v) for v in results['values']]
                    }
                    for class_name, results in mccv_results.items()
                }
            },
            'holdout': {
                class_name: float(f1)
                for class_name, f1 in ho_results.items()
            }
        }

    with open(json_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"✓ Detailed results saved to: {json_file}")

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
