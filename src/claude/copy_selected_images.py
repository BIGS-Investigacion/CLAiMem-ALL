#!/usr/bin/env python3
"""
Copy selected representative images to a specific directory.
"""

import pandas as pd
import shutil
from pathlib import Path
import os

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

def copy_selected_images(excel_file='representative_images_annotation.xlsx',
                        output_dir='selected_representative_images'):
    """
    Copy all images from Excel file to a specific directory.

    Args:
        excel_file: Path to the Excel file with selected images
        output_dir: Directory where images will be copied
    """

    # Base directories where images are stored
    base_dirs = [
        'data/histomorfologico/tcga/er',
        'data/histomorfologico/tcga/her2',
        'data/histomorfologico/tcga/pr',
        'data/histomorfologico/tcga/pam50'
    ]

    # Read Excel file
    print(f"Reading {excel_file}...")
    df = pd.read_excel(excel_file, sheet_name='Anotaciones')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Copy images
    total = len(df)
    copied = 0
    not_found = 0

    print(f"\nCopying {total} images...")

    for idx, row in df.iterrows():
        filename = row['IMAGEN']
        label = row['ETIQUETA']

        # Create subdirectory for each label
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # Find the image
        src_path = find_image_path(filename, base_dirs)

        if src_path:
            dst_path = os.path.join(label_dir, filename)
            shutil.copy2(src_path, dst_path)
            copied += 1
            if (copied % 25) == 0:
                print(f"  Progress: {copied}/{total} images copied...")
        else:
            print(f"  Warning: Image not found: {filename}")
            not_found += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"Total images in Excel: {total}")
    print(f"Images copied: {copied}")
    print(f"Images not found: {not_found}")
    print(f"\nImages organized in: {output_dir}")

    # Show directory structure
    print(f"\nDirectory structure:")
    for label in df['ETIQUETA'].unique():
        label_dir = os.path.join(output_dir, label)
        count = len([f for f in os.listdir(label_dir) if f.endswith('.png')])
        print(f"  {label}: {count} images")
    print(f"{'='*60}")

if __name__ == '__main__':
    copy_selected_images()