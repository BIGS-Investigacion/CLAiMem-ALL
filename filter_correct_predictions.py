#!/usr/bin/env python3
"""
Script to filter correctly classified images based on CSV labels.
"""

import os
import re
import argparse
import pandas as pd
import shutil
from pathlib import Path


def extract_slide_id(filename):
    """
    Extract slide_id from image filename.

    Example:
        Input: 0_TCGA-A1-A0SK-01A-01-BSA.29fd9144-1886-47ad-b075-7800cb57c9f5_x_34928_y_30501_a_100.000.png
        Output: TCGA-A1-A0SK-01A-01-BSA.29fd9144-1886-47ad-b075-7800cb57c9f5
    """
    # Remove the ordinal prefix (e.g., "0_", "1_", etc.) - this is just generation order, not prediction
    match = re.match(r'^\d+_(.+?)_x_\d+_y_\d+_a_[\d.]+\.png$', filename)
    if match:
        return match.group(1)
    return None


def get_predicted_label_from_dir(image_dir):
    """
    Extract predicted label from directory name.

    Examples:
        label_negative_pred_0 -> 0
        label_positive_pred_1 -> 1
    """
    dir_name = os.path.basename(os.path.normpath(image_dir))
    match = re.search(r'pred_(\d+)$', dir_name)
    if match:
        return int(match.group(1))
    return None


def filter_correct_predictions(image_dir, csv_path, output_dir=None, copy=False):
    """
    Filter images that were correctly classified.

    Args:
        image_dir: Directory containing predicted images
        csv_path: Path to CSV with ground truth labels
        output_dir: Optional directory to copy/move correct predictions
        copy: If True, copy files; if False, just list them
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} entries")

    # Create mapping: slide_id -> label
    label_map = {}
    for _, row in df.iterrows():
        slide_id = row['slide_id']
        label = row['label']
        # Convert label to numeric: negative=0, positive=1
        label_numeric = 0 if label == 'negative' else 1
        label_map[slide_id] = label_numeric

    print(f"Created label mapping for {len(label_map)} slide IDs")

    # Get predicted label from directory name
    predicted_label = get_predicted_label_from_dir(image_dir)
    if predicted_label is None:
        print(f"Error: Could not extract prediction label from directory name: {image_dir}")
        return [], [], []

    print(f"Directory prediction label: {predicted_label}")

    # Get all images in directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    print(f"Found {len(image_files)} images in {image_dir}")

    # Filter correct predictions
    correct_predictions = []
    incorrect_predictions = []
    not_found_in_csv = []

    for img_file in image_files:
        slide_id = extract_slide_id(img_file)

        if slide_id is None:
            print(f"Warning: Could not parse filename: {img_file}")
            continue

        if slide_id not in label_map:
            not_found_in_csv.append(img_file)
            continue

        true_label = label_map[slide_id]

        if predicted_label == true_label:
            correct_predictions.append(img_file)
        else:
            incorrect_predictions.append(img_file)

    # Print statistics
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Correct predictions: {len(correct_predictions)}")
    print(f"Incorrect predictions: {len(incorrect_predictions)}")
    print(f"Not found in CSV: {len(not_found_in_csv)}")
    print(f"Total processed: {len(correct_predictions) + len(incorrect_predictions)}")

    if len(correct_predictions) > 0:
        accuracy = len(correct_predictions) / (len(correct_predictions) + len(incorrect_predictions)) * 100
        print(f"Accuracy: {accuracy:.2f}%")

    # Copy/move files if output directory specified
    if output_dir and copy:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCopying {len(correct_predictions)} correctly classified images to {output_dir}")
        for img_file in correct_predictions:
            src = os.path.join(image_dir, img_file)
            dst = os.path.join(output_dir, img_file)
            shutil.copy2(src, dst)
        print("Done!")

    return correct_predictions, incorrect_predictions, not_found_in_csv


def main():
    parser = argparse.ArgumentParser(
        description='Filter correctly classified images based on CSV labels'
    )
    parser.add_argument(
        'image_dir',
        help='Directory containing predicted images (e.g., data/histomorfologico/tcga/er/label_negative_pred_0)'
    )
    parser.add_argument(
        'csv_path',
        help='Path to CSV with ground truth labels (e.g., data/dataset_csv/tcga-er.csv)'
    )
    parser.add_argument(
        '--output-dir',
        help='Optional: Directory to copy correctly classified images'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files to output directory (requires --output-dir)'
    )
    parser.add_argument(
        '--save-list',
        help='Optional: Save list of correct predictions to a text file'
    )

    args = parser.parse_args()

    correct, incorrect, not_found = filter_correct_predictions(
        args.image_dir,
        args.csv_path,
        args.output_dir,
        args.copy
    )

    # Save list if requested
    if args.save_list:
        with open(args.save_list, 'w') as f:
            for img in correct:
                f.write(f"{img}\n")
        print(f"\nSaved list of {len(correct)} correct predictions to {args.save_list}")


if __name__ == '__main__':
    main()
