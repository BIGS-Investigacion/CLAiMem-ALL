#!/usr/bin/env python3
"""
Extract Mean Biological Features by Class, Task, and Cohort

Reads annotated biological features from Excel file and computes mean values
for each class, organized by task (PAM50, ER, PR, HER2) and cohort (TCGA, CPTAC).

Features analyzed:
- ESTRUCTURA GLANDULAR (Glandular structure)
- ATIPIA NUCLEAR (Nuclear atypia)
- MITOSIS (Mitosis)
- NECROSIS (Necrosis)
- INFILTRADO_LI (Lymphocytic infiltrate)
- INFILTRADO_PMN (Polymorphonuclear infiltrate)

Author: Claude Code
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Excel file path
EXCEL_PATH = Path('data/histomorfologico/representative_images_annotation.xlsx')

# Output directory
OUTPUT_DIR = Path('results/biological_analysis')

# Feature columns to analyze
FEATURES = [
    'ESTRUCTURA GLANDULAR',
    'ATIPIA NUCLEAR',
    'MITOSIS',
    'NECROSIS',
    'INFILTRADO_LI',
    'INFILTRADO_PMN'
]

# Class mapping to standardized names
CLASS_MAPPING = {
    'BASAL': 'Basal',
    'HER2-enriched': 'Her2-enriched',
    'LUMINAL-A': 'LumA',
    'LUMINAL-B': 'LumB',
    'NORMAL-like': 'Normal',
    'ER-positive': 'ER-positive',
    'ER-negative': 'ER-negative',
    'PR-positive': 'PR-positive',
    'PR-negative': 'PR-negative',
    'HER2-positive': 'HER2-positive',
    'HER2-negative': 'HER2-negative'
}

# Task assignment
TASK_MAPPING = {
    'Basal': 'PAM50',
    'Her2-enriched': 'PAM50',
    'LumA': 'PAM50',
    'LumB': 'PAM50',
    'Normal': 'PAM50',
    'ER-positive': 'ER',
    'ER-negative': 'ER',
    'PR-positive': 'PR',
    'PR-negative': 'PR',
    'HER2-positive': 'HER2',
    'HER2-negative': 'HER2'
}


# ==============================================================================
# DATA LOADING AND PROCESSING
# ==============================================================================

def load_and_process_data(excel_path: Path) -> pd.DataFrame:
    """
    Load Excel file (both TCGA and CPTAC sheets) and process data.

    Args:
        excel_path: Path to Excel file

    Returns:
        Processed DataFrame
    """
    print(f"\nLoading data from: {excel_path}")
    
    all_data = []
    
    # Load TCGA sheet
    df_tcga = pd.read_excel(excel_path, sheet_name='TCGA')
    df_tcga['Cohort'] = 'TCGA'
    print(f"  TCGA sheet: {len(df_tcga)} rows")
    all_data.append(df_tcga)
    
    # Load CPTAC sheet
    df_cptac = pd.read_excel(excel_path, sheet_name='CPTAC')
    df_cptac['Cohort'] = 'CPTAC'
    print(f"  CPTAC sheet: {len(df_cptac)} rows")
    all_data.append(df_cptac)
    
    # Concatenate both datasets
    df = pd.concat(all_data, ignore_index=True)
    
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")

    # Standardize class names
    df['Class'] = df['ETIQUETA'].map(CLASS_MAPPING)

    # Assign task
    df['Task'] = df['Class'].map(TASK_MAPPING)

    # Verify all features exist
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in Excel: {missing_features}")

    print(f"  Unique classes: {sorted(df['Class'].unique())}")
    print(f"  Cohorts: {df['Cohort'].value_counts().to_dict()}")

    return df


def compute_mean_features_by_class_and_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean feature values for each class and cohort.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with mean values
    """
    results = []

    for task in sorted(df['Task'].unique()):
        task_df = df[df['Task'] == task]

        for cohort in ['TCGA', 'CPTAC']:
            cohort_df = task_df[task_df['Cohort'] == cohort]

            for class_name in sorted(cohort_df['Class'].unique()):
                class_df = cohort_df[cohort_df['Class'] == class_name]

                if len(class_df) == 0:
                    continue

                # Compute mean for each feature
                row = {
                    'Task': task,
                    'Cohort': cohort,
                    'Class': class_name,
                    'N_samples': len(class_df)
                }

                for feature in FEATURES:
                    mean_val = class_df[feature].mean()
                    std_val = class_df[feature].std()
                    row[f'{feature}_mean'] = mean_val
                    row[f'{feature}_std'] = std_val

                results.append(row)

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, output_dir: Path):
    """
    Save results to CSV and JSON files.

    Args:
        df: Results DataFrame
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save complete results
    csv_path = output_dir / 'biological_features_by_class_cohort.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✓ Saved: {csv_path}")

    # Save per-task results
    for task in df['Task'].unique():
        task_df = df[df['Task'] == task]
        task_csv = output_dir / f'{task.lower()}_biological_features.csv'
        task_df.to_csv(task_csv, index=False, float_format='%.4f')
        print(f"✓ Saved: {task_csv}")

    # Save as JSON for easy loading in other scripts
    json_data = {}
    for task in df['Task'].unique():
        json_data[task] = {}
        task_df = df[df['Task'] == task]

        for cohort in ['TCGA', 'CPTAC']:
            json_data[task][cohort] = {}
            cohort_df = task_df[task_df['Cohort'] == cohort]

            for _, row in cohort_df.iterrows():
                class_name = row['Class']
                json_data[task][cohort][class_name] = {
                    'n_samples': int(row['N_samples'])
                }

                for feature in FEATURES:
                    json_data[task][cohort][class_name][feature] = {
                        'mean': float(row[f'{feature}_mean']),
                        'std': float(row[f'{feature}_std'])
                    }

    json_path = output_dir / 'biological_features_summary.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved: {json_path}")


def print_summary(df: pd.DataFrame):
    """
    Print summary statistics.

    Args:
        df: Results DataFrame
    """
    print("\n" + "="*80)
    print("SUMMARY: Biological Features by Class and Cohort")
    print("="*80)

    for task in sorted(df['Task'].unique()):
        print(f"\n{task}:")
        print("-" * 80)
        task_df = df[df['Task'] == task]

        for cohort in ['TCGA', 'CPTAC']:
            cohort_df = task_df[task_df['Cohort'] == cohort]
            if len(cohort_df) == 0:
                continue

            print(f"\n  {cohort}:")
            for _, row in cohort_df.iterrows():
                print(f"    {row['Class']}: n={row['N_samples']}")
                for feature in FEATURES:
                    mean_val = row[f'{feature}_mean']
                    std_val = row[f'{feature}_std']
                    print(f"      {feature}: {mean_val:.3f} ± {std_val:.3f}")

    print("\n" + "="*80)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("BIOLOGICAL FEATURES EXTRACTION")
    print("="*80)

    # Load and process data
    df = load_and_process_data(EXCEL_PATH)

    # Compute mean features
    print("\nComputing mean features by class and cohort...")
    results_df = compute_mean_features_by_class_and_cohort(df)

    # Save results
    print("\nSaving results...")
    save_results(results_df, OUTPUT_DIR)

    # Print summary
    print_summary(results_df)

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print()


if __name__ == '__main__':
    main()
