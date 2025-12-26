#!/usr/bin/env python3
"""
Run distributional shift analysis for all tasks (PAM50, ER, PR, HER2)

This script executes the complete distributional shift analysis pipeline,
generates all metrics, visualizations, and summary tables.

Usage:
    python run_all_distributional_shift.py [--output OUTPUT_DIR]
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import json
import argparse


# Configuration for each task
TASKS = {
    'pam50': {
        'tcga_labels': 'data/dataset_csv/tcga-subtype_pam50.csv',
        'cptac_labels': 'data/dataset_csv/cptac-subtype_pam50.csv',
        'metrics': 'data/dataset_csv/statistics/pam50_performance_metrics_fake.csv',
    },
    'er': {
        'tcga_labels': 'data/dataset_csv/tcga-er.csv',
        'cptac_labels': 'data/dataset_csv/cptac-er.csv',
        'metrics': 'data/dataset_csv/statistics/er_performance_metrics_fake.csv',
    },
    'pr': {
        'tcga_labels': 'data/dataset_csv/tcga-pr.csv',
        'cptac_labels': 'data/dataset_csv/cptac-pr.csv',
        'metrics': 'data/dataset_csv/statistics/pr_performance_metrics_fake.csv',
    },
    'her2': {
        'tcga_labels': 'data/dataset_csv/tcga-erbb2.csv',
        'cptac_labels': 'data/dataset_csv/cptac-erbb2.csv',
        'metrics': 'data/dataset_csv/statistics/her2_performance_metrics_fake.csv',
    }
}


def run_distributional_shift(task_name, config, output_dir):
    """Run distributional shift analysis for a single task"""

    print(f"\n{'='*80}")
    print(f"Running distributional shift analysis: {task_name.upper()}")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        'python', 'src/claude/final/distributional_shift_analysis.py',
        '--tcga_labels', config['tcga_labels'],
        '--cptac_labels', config['cptac_labels'],
        '--metrics', config['metrics'],
        '--task', task_name,
        '--output', str(output_dir)
    ]

    # Run command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {task_name}:")
        print(e.stdout)
        print(e.stderr)
        return False


def extract_key_metrics(output_dir):
    """Extract key metrics from all task results into a summary table"""

    print(f"\n{'='*80}")
    print("EXTRACTING KEY METRICS")
    print(f"{'='*80}\n")

    summary_data = []

    for task_name in TASKS.keys():
        json_file = output_dir / f'{task_name}_distributional_shift_analysis.json'

        if not json_file.exists():
            print(f"  ⚠ Missing results for {task_name}")
            continue

        # Load JSON results
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract correlation metrics
        corr = data['correlation']

        # Extract prevalence shift stats
        prev_shift = pd.DataFrame(data['prevalence_shift'])
        mean_abs_shift = prev_shift['Δp'].abs().mean()
        max_abs_shift = prev_shift['Δp'].abs().max()

        # Extract performance degradation stats
        perf_deg = pd.DataFrame(data['performance_degradation'])
        mean_rpc = perf_deg['RPC'].mean()

        summary_data.append({
            'Task': task_name.upper(),
            'N_classes': len(prev_shift),
            'Mean_|Δp|': f"{mean_abs_shift:.4f}",
            'Max_|Δp|': f"{max_abs_shift:.4f}",
            'Mean_RPC': f"{mean_rpc:.4f}",
            'Pearson_r': f"{corr['pearson_r']:.4f}" if not pd.isna(corr['pearson_r']) else 'N/A',
            'p_value': f"{corr['pearson_p']:.4f}" if not pd.isna(corr['pearson_p']) else 'N/A',
            'Significant': 'Yes' if corr['pearson_p'] < 0.05 else 'No'
        })

    # Create summary DataFrame
    df_summary = pd.DataFrame(summary_data)

    # Save summary
    summary_file = output_dir / 'distributional_shift_summary.csv'
    df_summary.to_csv(summary_file, index=False)
    print(f"  ✓ Summary saved: {summary_file}")

    # Print summary table
    print("\n" + "="*80)
    print("DISTRIBUTIONAL SHIFT SUMMARY")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80 + "\n")

    return df_summary


def generate_combined_latex_table(output_dir):
    """Generate a combined LaTeX table with all tasks"""

    print(f"\n{'='*80}")
    print("GENERATING LATEX SUMMARY TABLE")
    print(f"{'='*80}\n")

    summary_data = []

    for task_name in TASKS.keys():
        json_file = output_dir / f'{task_name}_distributional_shift_analysis.json'

        if not json_file.exists():
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        corr = data['correlation']
        prev_shift = pd.DataFrame(data['prevalence_shift'])
        perf_deg = pd.DataFrame(data['performance_degradation'])

        mean_abs_shift = prev_shift['Δp'].abs().mean()
        mean_rpc = perf_deg['RPC'].mean()

        r = corr['pearson_r']
        p = corr['pearson_p']

        # Format p-value with stars
        if pd.isna(p) or p >= 0.05:
            p_str = f"{p:.3f}" if not pd.isna(p) else "N/A"
        elif p < 0.001:
            p_str = r"$< 0.001$***"
        elif p < 0.01:
            p_str = f"{p:.3f}**"
        elif p < 0.05:
            p_str = f"{p:.3f}*"
        else:
            p_str = f"{p:.3f}"

        summary_data.append({
            'task': task_name.upper(),
            'n_classes': len(prev_shift),
            'mean_shift': f"{mean_abs_shift:.4f}",
            'mean_rpc': f"{mean_rpc:.4f}",
            'r': f"{r:.3f}" if not pd.isna(r) else "N/A",
            'p': p_str
        })

    # Generate LaTeX table
    latex = []
    latex.append(r"\begin{table}[h!]")
    latex.append(r"\centering")
    latex.append(r"\caption{Distributional shift analysis summary across tasks. " +
                 r"Pearson correlation between signed prevalence shift ($\Delta p$) and " +
                 r"relative performance change (RPC). * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$.}")
    latex.append(r"\label{tab:distributional-shift-summary}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Task} & \textbf{Classes} & \textbf{Mean $|\Delta p|$} & " +
                 r"\textbf{Mean RPC} & \textbf{$r$} & \textbf{$p$} \\")
    latex.append(r"\midrule")

    for row in summary_data:
        latex.append(f"{row['task']} & {row['n_classes']} & {row['mean_shift']} & " +
                    f"{row['mean_rpc']} & {row['r']} & {row['p']} " + r"\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    # Save LaTeX table
    latex_file = output_dir / 'distributional_shift_summary_table.tex'
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex))

    print(f"  ✓ LaTeX table saved: {latex_file}\n")

    return latex_file


def main():
    parser = argparse.ArgumentParser(
        description='Run distributional shift analysis for all tasks'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/distributional_shift',
        help='Output directory (default: results/distributional_shift)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DISTRIBUTIONAL SHIFT ANALYSIS - ALL TASKS")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Tasks: {', '.join([t.upper() for t in TASKS.keys()])}")
    print("="*80)

    # Run analysis for each task
    success_count = 0
    for task_name, config in TASKS.items():
        if run_distributional_shift(task_name, config, output_dir):
            success_count += 1
        else:
            print(f"  ✗ Failed: {task_name}")

    print(f"\n{'='*80}")
    print(f"COMPLETED: {success_count}/{len(TASKS)} tasks successful")
    print(f"{'='*80}\n")

    # Extract key metrics
    if success_count > 0:
        extract_key_metrics(output_dir)
        generate_combined_latex_table(output_dir)

    print("="*80)
    print("ALL ANALYSES COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  • *_prevalence_shift.csv - Prevalence data for each task")
    print("  • *_performance_degradation.csv - Performance metrics")
    print("  • *_distributional_shift_analysis.json - Complete results")
    print("  • *_prevalence_comparison.png - Prevalence bar charts")
    print("  • *_prevalence_shift.png - Signed Δp visualizations")
    print("  • *_correlation.png - Scatter plots Δp vs RPC")
    print("  • *_combined_analysis.png - Combined visualizations")
    print("  • distributional_shift_summary.csv - Summary table")
    print("  • distributional_shift_summary_table.tex - LaTeX summary")
    print("="*80)


if __name__ == "__main__":
    main()
