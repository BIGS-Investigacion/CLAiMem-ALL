#!/usr/bin/env python3
"""
==============================================================================
INTEGRATIVE DOMAIN SHIFT ANALYSIS
==============================================================================

This script integrates findings from all four domain shift analyses to evaluate
which factors independently contribute to performance degradation.

Multiple Linear Regression Model:
    RPC_c = β₀ + β₁·Δp_c + β₂·d_c + β₃·ΔAcc_norm_c + β₄·B_c + ε

Where:
    - RPC_c: Relative Performance Change for class c
    - Δp_c: Prevalence shift (distributional shift)
    - d_c: Centroid distance in embedding space (feature shift)
    - ΔAcc_norm_c: Stain normalization benefit
    - B_c: Biological shift indicator (morphological differences)

All predictors are min-max scaled to [0,1] for comparable coefficients.

Outputs:
    - R²: Proportion of variance explained
    - Coefficients (β): Relative importance of each factor
    - p-values: Statistical significance
    - VIF: Multicollinearity diagnostics

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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_all_analyses_results(results_dir: Path, task: str) -> Dict:
    """
    Load results from all four domain shift analyses.

    Expected files:
        - {task}_prevalence_shift.csv (distributional shift)
        - {task}_centroid_distances.csv (feature space consistency)
        - {task}_stain_normalization_results.csv (stain normalization)
        - {task}_biological_shift.csv (biological interpretability)
        - {task}_performance_degradation.csv (RPC values)

    Args:
        results_dir: Directory containing all analysis results
        task: Task name

    Returns:
        Dictionary with merged data
    """
    results_dir = Path(results_dir)

    # Load distributional shift
    df_prevalence = pd.read_csv(
        results_dir / 'distributional_shift' / f'{task}_prevalence_shift.csv'
    )

    # Load feature space consistency
    df_centroids = pd.read_csv(
        results_dir / 'feature_space' / f'{task}_centroid_distances.csv'
    )

    # Load stain normalization
    df_stain = pd.read_csv(
        results_dir / 'stain_analysis' / f'{task}_stain_normalization_results.csv'
    )

    # Load biological shift
    df_biological = pd.read_csv(
        results_dir / 'biological_analysis' / f'{task}_biological_shift.csv'
    )

    # Load performance degradation
    df_performance = pd.read_csv(
        results_dir / 'distributional_shift' / f'{task}_performance_degradation.csv'
    )

    return {
        'prevalence': df_prevalence,
        'centroids': df_centroids,
        'stain': df_stain,
        'biological': df_biological,
        'performance': df_performance
    }


def merge_all_data(data_dict: Dict) -> pd.DataFrame:
    """
    Merge all analysis results into a single DataFrame.

    Args:
        data_dict: Dictionary from load_all_analyses_results()

    Returns:
        Merged DataFrame with all predictors and response variable
    """
    # Start with performance (contains RPC)
    df = data_dict['performance'][['Class', 'RPC']].copy()

    # Merge prevalence shift (Δp)
    df = df.merge(
        data_dict['prevalence'][['Class', 'Δp_absolute']],
        on='Class',
        how='left'
    )

    # Merge centroid distance (d_c)
    df = df.merge(
        data_dict['centroids'][['Class', 'Centroid_Distance']],
        on='Class',
        how='left'
    )

    # Merge stain normalization benefit (ΔAcc_norm)
    df = df.merge(
        data_dict['stain'][['Class', 'ΔAccuracy']],
        on='Class',
        how='left'
    ).rename(columns={'ΔAccuracy': 'ΔAcc_norm'})

    # Merge biological shift (B_c)
    df = df.merge(
        data_dict['biological'][['Class', 'B_c']],
        on='Class',
        how='left'
    )

    # Drop rows with missing data
    df_clean = df.dropna()

    if len(df_clean) < len(df):
        print(f"  Warning: Dropped {len(df) - len(df_clean)} classes with missing data")

    return df_clean


# ==============================================================================
# REGRESSION ANALYSIS
# ==============================================================================

def scale_predictors(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply min-max scaling to all predictors to [0,1] range.

    This enables direct comparison of regression coefficients.

    Args:
        df: DataFrame with predictors

    Returns:
        Tuple of (scaled DataFrame, scaler params dict)
    """
    df_scaled = df.copy()

    predictors = ['Δp_absolute', 'Centroid_Distance', 'ΔAcc_norm', 'B_c']

    scaler_params = {}

    for pred in predictors:
        min_val = df[pred].min()
        max_val = df[pred].max()
        range_val = max_val - min_val

        if range_val > 0:
            df_scaled[f'{pred}_scaled'] = (df[pred] - min_val) / range_val
        else:
            df_scaled[f'{pred}_scaled'] = 0.5  # Constant predictor

        scaler_params[pred] = {'min': min_val, 'max': max_val, 'range': range_val}

    return df_scaled, scaler_params


def fit_multiple_regression(df: pd.DataFrame) -> Dict:
    """
    Fit multiple linear regression model.

    Model:
        RPC = β₀ + β₁·Δp + β₂·d + β₃·ΔAcc_norm + β₄·B + ε

    Args:
        df: DataFrame with scaled predictors

    Returns:
        Dictionary with regression results
    """
    # Prepare data
    predictors_scaled = [
        'Δp_absolute_scaled',
        'Centroid_Distance_scaled',
        'ΔAcc_norm_scaled',
        'B_c_scaled'
    ]

    X = df[predictors_scaled].values
    y = df['RPC'].values

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)
    residuals = y - y_pred

    # R-squared
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(residuals ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Adjusted R-squared
    n = len(y)
    p = X.shape[1]
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    # Standard error
    mse = ss_res / (n - p - 1)
    se = np.sqrt(mse)

    # Coefficient standard errors and p-values
    # Using simple formula (for proper inference, use statsmodels)
    X_with_intercept = np.column_stack([np.ones(n), X])
    cov_matrix = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * mse

    se_coeffs = np.sqrt(np.diag(cov_matrix))
    coeffs_with_intercept = np.concatenate([[model.intercept_], model.coef_])

    t_stats = coeffs_with_intercept / se_coeffs
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))

    # F-statistic for overall model significance
    f_stat = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
    f_p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)

    return {
        'intercept': float(model.intercept_),
        'coefficients': {
            'prevalence_shift': float(model.coef_[0]),
            'centroid_distance': float(model.coef_[1]),
            'stain_normalization': float(model.coef_[2]),
            'biological_shift': float(model.coef_[3])
        },
        'p_values': {
            'intercept': float(p_values[0]),
            'prevalence_shift': float(p_values[1]),
            'centroid_distance': float(p_values[2]),
            'stain_normalization': float(p_values[3]),
            'biological_shift': float(p_values[4])
        },
        'r_squared': float(r_squared),
        'adj_r_squared': float(adj_r_squared),
        'f_statistic': float(f_stat),
        'f_p_value': float(f_p_value),
        'residual_se': float(se),
        'n_obs': int(n),
        'n_predictors': int(p),
        'predictions': y_pred.tolist(),
        'residuals': residuals.tolist()
    }


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor to detect multicollinearity.

    VIF interpretation:
        - VIF < 5: Low multicollinearity
        - 5 ≤ VIF < 10: Moderate multicollinearity
        - VIF ≥ 10: High multicollinearity (problematic)

    Args:
        df: DataFrame with scaled predictors

    Returns:
        DataFrame with VIF values
    """
    predictors_scaled = [
        'Δp_absolute_scaled',
        'Centroid_Distance_scaled',
        'ΔAcc_norm_scaled',
        'B_c_scaled'
    ]

    X = df[predictors_scaled].values

    vif_data = []
    for i, col in enumerate(predictors_scaled):
        vif = variance_inflation_factor(X, i)
        vif_data.append({
            'Predictor': col.replace('_scaled', ''),
            'VIF': vif
        })

    return pd.DataFrame(vif_data)


def compute_standardized_coefficients(df: pd.DataFrame,
                                      regression_results: Dict) -> Dict:
    """
    Compute standardized (beta) coefficients for interpretation.

    Standardized coefficients show the change in y (in standard deviations)
    for a 1-SD change in x, allowing direct comparison of effect sizes.

    Args:
        df: DataFrame with original (unscaled) predictors
        regression_results: Results from fit_multiple_regression()

    Returns:
        Dictionary with standardized coefficients
    """
    predictors_original = ['Δp_absolute', 'Centroid_Distance', 'ΔAcc_norm', 'B_c']

    std_y = df['RPC'].std()
    std_x = {pred: df[pred].std() for pred in predictors_original}

    coef_names = ['prevalence_shift', 'centroid_distance',
                  'stain_normalization', 'biological_shift']

    standardized = {}

    for i, (pred_name, coef_name) in enumerate(zip(predictors_original, coef_names)):
        beta = regression_results['coefficients'][coef_name]
        standardized[coef_name] = beta * (std_x[pred_name] / std_y)

    return standardized


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_regression_coefficients(regression_results: Dict, task: str,
                                 output_path: Optional[Path] = None) -> None:
    """
    Bar plot of regression coefficients with significance indicators.
    """
    coef_names = ['Prevalence\nShift', 'Centroid\nDistance',
                  'Stain\nNormalization', 'Biological\nShift']

    coeffs = [
        regression_results['coefficients']['prevalence_shift'],
        regression_results['coefficients']['centroid_distance'],
        regression_results['coefficients']['stain_normalization'],
        regression_results['coefficients']['biological_shift']
    ]

    p_vals = [
        regression_results['p_values']['prevalence_shift'],
        regression_results['p_values']['centroid_distance'],
        regression_results['p_values']['stain_normalization'],
        regression_results['p_values']['biological_shift']
    ]

    # Color by significance
    colors = ['green' if p < 0.05 else 'gray' for p in p_vals]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(coef_names, coeffs, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, coef, p in zip(bars, coeffs, p_vals):
        height = bar.get_height()
        sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{coef:+.3f}\n{sig_marker}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_ylabel('Regression Coefficient (β)', fontsize=12, fontweight='bold')
    ax.set_title(f'Multiple Regression: Predictors of Performance Degradation\n'
                 f'{task.upper()} (R² = {regression_results["r_squared"]:.3f}, '
                 f'p = {regression_results["f_p_value"]:.4f})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Significant (p < 0.05)'),
        Patch(facecolor='gray', alpha=0.7, label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_predicted_vs_actual(df: pd.DataFrame, regression_results: Dict,
                             task: str, output_path: Optional[Path] = None) -> None:
    """
    Scatter plot of predicted vs actual RPC.
    """
    y_actual = df['RPC'].values
    y_pred = np.array(regression_results['predictions'])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    scatter = ax.scatter(y_actual, y_pred, s=150, alpha=0.7,
                        c=y_actual, cmap='RdYlGn', edgecolor='black', linewidth=1.5)

    # Add labels
    for i, class_name in enumerate(df['Class']):
        ax.annotate(class_name, (y_actual[i], y_pred[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')

    # Diagonal line (perfect prediction)
    lims = [min(y_actual.min(), y_pred.min()) - 0.05,
            max(y_actual.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.5, label='Perfect prediction')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Actual RPC', fontsize=11)

    ax.set_xlabel('Actual RPC', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted RPC', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Predictions vs Actual Performance Change\n'
                 f'{task.upper()} (R² = {regression_results["r_squared"]:.3f})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_residuals(df: pd.DataFrame, regression_results: Dict,
                  task: str, output_path: Optional[Path] = None) -> None:
    """
    Residual plot for model diagnostics.
    """
    y_pred = np.array(regression_results['predictions'])
    residuals = np.array(regression_results['residuals'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Residuals vs fitted
    ax1.scatter(y_pred, residuals, s=100, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)

    for i, class_name in enumerate(df['Class']):
        ax1.annotate(class_name, (y_pred[i], residuals[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax1.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax1.set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')

    fig.suptitle(f'Regression Diagnostics: {task.upper()}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, task: str,
                            output_path: Optional[Path] = None) -> None:
    """
    Correlation matrix heatmap of all variables.
    """
    vars_of_interest = ['RPC', 'Δp_absolute', 'Centroid_Distance', 'ΔAcc_norm', 'B_c']
    var_labels = ['RPC', 'Prevalence\nShift', 'Centroid\nDistance',
                  'Stain Norm\nBenefit', 'Biological\nShift']

    corr_matrix = df[vars_of_interest].corr()
    corr_matrix.index = var_labels
    corr_matrix.columns = var_labels

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation'}, ax=ax)

    ax.set_title(f'Correlation Matrix: Domain Shift Factors\n{task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)

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

def print_results(regression_results: Dict, df_vif: pd.DataFrame,
                  standardized_coef: Dict, task: str) -> None:
    """
    Print formatted regression results.
    """
    print("\n" + "="*80)
    print(f"INTEGRATIVE DOMAIN SHIFT ANALYSIS: {task.upper()}")
    print("="*80)

    # Model summary
    print("\n" + "-"*80)
    print("MODEL SUMMARY")
    print("-"*80)
    print(f"R²:              {regression_results['r_squared']:.4f}")
    print(f"Adjusted R²:     {regression_results['adj_r_squared']:.4f}")
    print(f"F-statistic:     {regression_results['f_statistic']:.4f}")
    print(f"F p-value:       {regression_results['f_p_value']:.6f}")
    print(f"Residual SE:     {regression_results['residual_se']:.4f}")
    print(f"Observations:    {regression_results['n_obs']}")

    # Coefficients
    print("\n" + "-"*80)
    print("REGRESSION COEFFICIENTS (Scaled Predictors)")
    print("-"*80)
    print(f"{'Predictor':<25} {'Coefficient':>12} {'p-value':>12} {'Sig.':>6}")
    print("-"*80)

    print(f"{'Intercept':<25} {regression_results['intercept']:>12.4f} "
          f"{regression_results['p_values']['intercept']:>12.6f} "
          f"{'***' if regression_results['p_values']['intercept'] < 0.001 else '**' if regression_results['p_values']['intercept'] < 0.01 else '*' if regression_results['p_values']['intercept'] < 0.05 else 'ns':>6}")

    coef_info = [
        ('Prevalence Shift (Δp)', 'prevalence_shift'),
        ('Centroid Distance (d)', 'centroid_distance'),
        ('Stain Norm Benefit (ΔAcc)', 'stain_normalization'),
        ('Biological Shift (B)', 'biological_shift')
    ]

    for name, key in coef_info:
        coef = regression_results['coefficients'][key]
        p_val = regression_results['p_values'][key]
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

        print(f"{name:<25} {coef:>12.4f} {p_val:>12.6f} {sig:>6}")

    # Standardized coefficients
    print("\n" + "-"*80)
    print("STANDARDIZED COEFFICIENTS (Effect Sizes)")
    print("-"*80)

    for name, key in coef_info:
        std_coef = standardized_coef[key]
        print(f"{name:<25} {std_coef:>12.4f}")

    # Multicollinearity
    print("\n" + "-"*80)
    print("MULTICOLLINEARITY DIAGNOSTICS (VIF)")
    print("-"*80)
    print(df_vif.to_string(index=False))

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    r2 = regression_results['r_squared']
    p_model = regression_results['f_p_value']

    if p_model >= 0.05:
        print("\n✗ The overall model is NOT statistically significant (p ≥ 0.05).")
        print("  The measured domain shift factors do not collectively explain")
        print("  performance degradation.")
    else:
        print("\n✓ The overall model is statistically significant (p < 0.05).")

        if r2 < 0.30:
            print(f"\n  However, the model explains only {r2*100:.1f}% of variance (R² < 0.30).")
            print("  This suggests that unmeasured factors drive generalization failure.")
        else:
            print(f"\n  The model explains {r2*100:.1f}% of variance (R² = {r2:.3f}).")

            # Identify significant predictors
            sig_predictors = []
            for name, key in coef_info:
                if regression_results['p_values'][key] < 0.05:
                    sig_predictors.append((name, key))

            if sig_predictors:
                print("\n  Significant predictors:")
                for name, key in sig_predictors:
                    coef = regression_results['coefficients'][key]
                    direction = "increases" if coef > 0 else "decreases"
                    print(f"    • {name}: {direction} performance degradation")
            else:
                print("\n  No individual predictors are statistically significant,")
                print("  suggesting complex interactions or multicollinearity.")

    # VIF interpretation
    max_vif = df_vif['VIF'].max()
    if max_vif >= 10:
        print("\n  ⚠ WARNING: High multicollinearity detected (VIF ≥ 10).")
        print("  Coefficient estimates may be unstable.")
    elif max_vif >= 5:
        print("\n  ⚠ Note: Moderate multicollinearity detected (5 ≤ VIF < 10).")

    print("="*80 + "\n")


def save_results(df: pd.DataFrame, regression_results: Dict,
                 df_vif: pd.DataFrame, standardized_coef: Dict,
                 output_dir: Path, task: str) -> None:
    """
    Save all results to files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save merged data with predictions
    df_output = df.copy()
    df_output['Predicted_RPC'] = regression_results['predictions']
    df_output['Residuals'] = regression_results['residuals']

    csv_path = output_dir / f'{task}_regression_data.csv'
    df_output.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  ✓ Saved: {csv_path.name}")

    # Save VIF
    csv_path = output_dir / f'{task}_vif.csv'
    df_vif.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  ✓ Saved: {csv_path.name}")

    # Save full results as JSON
    results = {
        'task': task,
        'regression': regression_results,
        'standardized_coefficients': standardized_coef,
        'vif': df_vif.to_dict(orient='records')
    }

    json_path = output_dir / f'{task}_integrative_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: {json_path.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Integrative Domain Shift Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python integrative_domain_shift_analysis.py \\
    --results_dir results/ \\
    --task pam50 \\
    --output results/integrative_analysis/
        """
    )

    # Input/output
    parser.add_argument('--results_dir', '-r', type=str, required=True,
                        help='Directory containing all analysis results')
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
    print("INTEGRATIVE DOMAIN SHIFT ANALYSIS")
    print("="*80)
    print(f"Task:        {args.task.upper()}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output:      {args.output}")
    print("="*80 + "\n")

    # Load all data
    print("Loading analysis results...")
    data_dict = load_all_analyses_results(Path(args.results_dir), args.task)
    print("  ✓ All results loaded")

    # Merge data
    print("\nMerging data...")
    df = merge_all_data(data_dict)
    print(f"  ✓ Merged {len(df)} classes")

    # Scale predictors
    print("\nScaling predictors...")
    df_scaled, scaler_params = scale_predictors(df)
    print("  ✓ Predictors scaled to [0, 1]")

    # Fit regression
    print("\nFitting multiple linear regression...")
    regression_results = fit_multiple_regression(df_scaled)
    print("  ✓ Regression complete")

    # Compute VIF
    print("\nComputing VIF...")
    df_vif = compute_vif(df_scaled)
    print("  ✓ VIF computed")

    # Standardized coefficients
    print("\nComputing standardized coefficients...")
    standardized_coef = compute_standardized_coefficients(df, regression_results)
    print("  ✓ Standardized coefficients computed")

    # Print results
    print_results(regression_results, df_vif, standardized_coef, args.task)

    # Save results
    print("\nSaving results...")
    save_results(df, regression_results, df_vif, standardized_coef,
                 output_dir, args.task)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualizations...")

        plot_regression_coefficients(
            regression_results, args.task,
            output_dir / f'{args.task}_regression_coefficients.{args.plot_format}'
        )

        plot_predicted_vs_actual(
            df, regression_results, args.task,
            output_dir / f'{args.task}_predicted_vs_actual.{args.plot_format}'
        )

        plot_residuals(
            df, regression_results, args.task,
            output_dir / f'{args.task}_residuals.{args.plot_format}'
        )

        plot_correlation_matrix(
            df, args.task,
            output_dir / f'{args.task}_correlation_matrix.{args.plot_format}'
        )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/\n")


if __name__ == '__main__':
    main()