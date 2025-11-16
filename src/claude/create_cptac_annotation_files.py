#!/usr/bin/env python3
"""
Create CSV and copy selected representative images for CPTAC dataset.
"""

import json
import pandas as pd
import shutil
import os
from pathlib import Path

def find_image_path(filename, base_dirs):
    """
    Find the full path of an image file by searching in base directories.

    Args:
        filename: Image filename to find
        base_dirs: List of base directories to search in

    Returns:
        Full path if found, None otherwise
    """
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
    return None

def create_annotation_files(database='cptac', output_dir='selected_representative_images_cptac'):
    """
    Create CSV file and copy all selected images for a database.

    Args:
        database: Database name (e.g., 'cptac')
        output_dir: Directory where images will be copied
    """

    # Base directories where images are stored
    base_dirs = [
        f'data/histomorfologico/{database}/er',
        f'data/histomorfologico/{database}/her2',
        f'data/histomorfologico/{database}/pr',
        f'data/histomorfologico/{database}/pam50'
    ]

    # Define datasets and their labels
    datasets = {
        'ER': {
            'negative': f'selected_gt_{database}_er_negative_with_distances.json',
            'positive': f'selected_gt_{database}_er_positive_with_distances.json',
        },
        'HER2': {
            'negative': f'selected_gt_{database}_her2_negative_with_distances.json',
            'positive': f'selected_gt_{database}_her2_positive_with_distances.json',
        },
        'PR': {
            'negative': f'selected_gt_{database}_pr_negative_with_distances.json',
            'positive': f'selected_gt_{database}_pr_positive_with_distances.json',
        },
        'PAM50': {
            'basal': f'selected_gt_{database}_pam50_basal_with_distances.json',
            'her2': f'selected_gt_{database}_pam50_her2_with_distances.json',
            'luma': f'selected_gt_{database}_pam50_luma_with_distances.json',
            'lumb': f'selected_gt_{database}_pam50_lumb_with_distances.json',
            'normal': f'selected_gt_{database}_pam50_normal_with_distances.json',
        }
    }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Collect all data for CSV
    all_rows = []
    copied = 0
    not_found = 0

    print(f"\nProcessing images...")

    for dataset_name, labels in datasets.items():
        for label_name, json_file in labels.items():
            if not Path(json_file).exists():
                print(f"Warning: {json_file} not found, skipping {dataset_name}/{label_name}")
                continue

            # Load selected images
            with open(json_file, 'r') as f:
                images = json.load(f)

            # Create subdirectory for each dataset-label combination
            label_dir = os.path.join(output_dir, f'{dataset_name}_{label_name}')
            os.makedirs(label_dir, exist_ok=True)

            for img_data in images:
                filename = img_data['filename']
                distance = img_data['distance']

                # Find the image
                src_path = find_image_path(filename, base_dirs)

                if src_path:
                    # Copy image
                    dst_path = os.path.join(label_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    copied += 1

                    # Add to CSV data
                    row = {
                        'DATASET': dataset_name,
                        'LABEL': label_name,
                        'IMAGEN': filename,
                        'DISTANCIA': round(distance, 4),
                        'RUTA_RELATIVA': f'{dataset_name}_{label_name}/{filename}',
                        'ESTRUCTURA_GLANDULAR': '',
                        'ATIPIA_NUCLEAR': '',
                        'MITOSIS': '',
                        'NECROSIS': '',
                        'INFILTRADO_LI': ''
                    }
                    all_rows.append(row)
                else:
                    print(f"  Warning: Image not found: {filename}")
                    not_found += 1

    # Create DataFrame and save CSV
    df = pd.DataFrame(all_rows)
    csv_file = f'{output_dir}/annotation_list_{database}.csv'
    df.to_csv(csv_file, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"Images copied: {copied}")
    print(f"Images not found: {not_found}")
    print(f"\nFiles created:")
    print(f"  CSV: {csv_file}")
    print(f"  Images directory: {output_dir}")

    # Show breakdown by dataset/label
    print(f"\nBreakdown by dataset and label:")
    for dataset_name in datasets.keys():
        dataset_count = df[df['DATASET'] == dataset_name].shape[0]
        if dataset_count > 0:
            print(f"\n  {dataset_name}: {dataset_count} images")
            for label in df[df['DATASET'] == dataset_name]['LABEL'].unique():
                count = df[(df['DATASET'] == dataset_name) & (df['LABEL'] == label)].shape[0]
                print(f"    - {label}: {count} images")
    print(f"{'='*60}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create annotation CSV and copy images')
    parser.add_argument('--database', type=str, default='cptac', help='Database name (default: cptac)')
    parser.add_argument('--output-dir', type=str, default='selected_representative_images_cptac',
                       help='Output directory (default: selected_representative_images_cptac)')
    args = parser.parse_args()

    create_annotation_files(database=args.database, output_dir=args.output_dir)