#!/usr/bin/env python3
"""
Select representative images based on ground truth labels (from CSV),
regardless of model predictions.
"""

import os
import re
import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import json


def extract_slide_id(filename):
    """
    Extract slide_id from image filename.

    Example:
        Input: 0_TCGA-A1-A0SK-01A-01-BSA.29fd9144-1886-47ad-b075-7800cb57c9f5_x_34928_y_30501_a_100.000.png
        Output: TCGA-A1-A0SK-01A-01-BSA.29fd9144-1886-47ad-b075-7800cb57c9f5
    """
    match = re.match(r'^\d+_(.+?)_x_\d+_y_\d+_a_[\d.]+\.png$', filename)
    if match:
        return match.group(1)
    return None


def load_virchow_model():
    """Load Virchow v2 model from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModel
    import timm

    print("Loading Virchow v2 model...")
    processor = AutoImageProcessor.from_pretrained("paige-ai/Virchow2", trust_remote_code=True)
    model = AutoModel.from_pretrained("paige-ai/Virchow2", trust_remote_code=True)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, processor, device


def compute_embeddings(model, processor, device, image_list, image_dir, batch_size=8):
    """
    Compute embeddings for a list of images.

    Args:
        model: Virchow model
        processor: Image processor
        device: torch device
        image_list: List of image filenames
        image_dir: Directory containing images
        batch_size: Batch size for processing

    Returns:
        torch.Tensor: Embeddings of shape [num_images, embedding_dim]
    """
    from PIL import Image

    embeddings_list = []

    num_batches = (len(image_list) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Computing embeddings"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(image_list))
            batch_images = image_list[batch_start:batch_end]

            # Load and process images
            pil_images = []
            for img_file in batch_images:
                img_path = os.path.join(image_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                pil_images.append(img)

            # Process batch
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            outputs = model(**inputs)
            # Use CLS token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

            embeddings_list.append(batch_embeddings.cpu())

    # Concatenate all embeddings
    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings


def select_representative_images(embeddings, image_list, n_images=25):
    """
    Select n representative images closest to the mean embedding (centroid).

    Args:
        embeddings: Tensor of embeddings [num_images, embedding_dim]
        image_list: List of image filenames
        n_images: Number of representative images to select

    Returns:
        List of selected image filenames with distances
    """
    # Compute mean embedding (centroid)
    mean_embedding = embeddings.mean(dim=0, keepdim=True)

    # Compute distances to mean
    distances = torch.norm(embeddings - mean_embedding, dim=1)

    # Get indices of n_images closest to mean
    _, indices = torch.topk(distances, k=min(n_images, len(image_list)), largest=False)

    # Get selected images with distances
    selected = []
    for idx in indices:
        selected.append({
            'filename': image_list[idx],
            'distance': float(distances[idx])
        })

    return selected


def group_images_by_label(image_dirs, csv_path):
    """
    Group all images by their ground truth label from CSV.

    Args:
        image_dirs: List of directories containing images
        csv_path: Path to CSV with ground truth labels

    Returns:
        dict: {label: [list of (filename, full_path)]}
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} entries")

    # Create mapping: slide_id -> label
    label_map = {}
    for _, row in df.iterrows():
        slide_id = row['slide_id']
        label = row['label']
        label_map[slide_id] = label

    print(f"Created label mapping for {len(label_map)} slide IDs")

    # Group images by label
    images_by_label = {}
    total_images = 0
    not_found_in_csv = 0

    for image_dir in image_dirs:
        if not os.path.exists(image_dir):
            print(f"Warning: Directory not found: {image_dir}")
            continue

        print(f"\nScanning directory: {image_dir}")
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        print(f"  Found {len(image_files)} images")

        for img_file in image_files:
            slide_id = extract_slide_id(img_file)

            if slide_id is None:
                continue

            if slide_id not in label_map:
                not_found_in_csv += 1
                continue

            label = label_map[slide_id]

            if label not in images_by_label:
                images_by_label[label] = []

            images_by_label[label].append((img_file, image_dir))
            total_images += 1

    print(f"\n{'='*60}")
    print(f"Images grouped by ground truth label:")
    for label, images in sorted(images_by_label.items()):
        print(f"  {label}: {len(images)} images")
    print(f"Total images: {total_images}")
    print(f"Not found in CSV: {not_found_in_csv}")
    print(f"{'='*60}\n")

    return images_by_label


def main():
    parser = argparse.ArgumentParser(
        description='Select representative images based on ground truth labels'
    )
    parser.add_argument(
        'base_dir',
        help='Base directory containing subdirectories with images (e.g., data/histomorfologico/tcga/er)'
    )
    parser.add_argument(
        'csv_path',
        help='Path to CSV with ground truth labels (e.g., data/dataset_csv/tcga-er.csv)'
    )
    parser.add_argument(
        '--n-images',
        type=int,
        default=25,
        help='Number of representative images to select per label (default: 25)'
    )
    parser.add_argument(
        '--output-prefix',
        default='selected_gt',
        help='Prefix for output files (default: selected_gt)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for embedding computation (default: 8)'
    )

    args = parser.parse_args()

    # Find all subdirectories with images
    image_dirs = []
    for item in os.listdir(args.base_dir):
        item_path = os.path.join(args.base_dir, item)
        if os.path.isdir(item_path) and item.startswith('label_'):
            image_dirs.append(item_path)

    if not image_dirs:
        print(f"Error: No label directories found in {args.base_dir}")
        return

    print(f"Found {len(image_dirs)} label directories")

    # Group images by ground truth label
    images_by_label = group_images_by_label(image_dirs, args.csv_path)

    if not images_by_label:
        print("Error: No images found")
        return

    # Load model once
    model, processor, device = load_virchow_model()

    # Process each label
    for label, image_list_with_paths in sorted(images_by_label.items()):
        print(f"\n{'='*60}")
        print(f"Processing label: {label}")
        print(f"Total images: {len(image_list_with_paths)}")
        print(f"{'='*60}")

        # We need to compute embeddings from different directories
        # Create a temporary directory mapping for efficient access
        images_by_dir = {}
        for img_file, img_dir in image_list_with_paths:
            if img_dir not in images_by_dir:
                images_by_dir[img_dir] = []
            images_by_dir[img_dir].append(img_file)

        # Compute embeddings for all images of this label
        all_embeddings = []
        all_filenames = []

        for img_dir, img_files in images_by_dir.items():
            print(f"\nComputing embeddings for {len(img_files)} images from {img_dir}...")
            embeddings = compute_embeddings(model, processor, device, img_files, img_dir, args.batch_size)
            all_embeddings.append(embeddings)
            all_filenames.extend(img_files)

        # Concatenate all embeddings for this label
        label_embeddings = torch.cat(all_embeddings, dim=0)

        print(f"\nSelecting {args.n_images} representative images...")
        selected = select_representative_images(label_embeddings, all_filenames, args.n_images)

        # Save results
        label_safe = str(label).replace('/', '_').replace(' ', '_')
        output_txt = f"{args.output_prefix}_{label_safe}.txt"
        output_json = f"{args.output_prefix}_{label_safe}_with_distances.json"

        # Save list
        with open(output_txt, 'w') as f:
            for item in selected:
                f.write(f"{item['filename']}\n")

        # Save with distances
        with open(output_json, 'w') as f:
            json.dump(selected, f, indent=2)

        print(f"\nResults for label '{label}':")
        print(f"  Selected {len(selected)} images")
        print(f"  Distance range: {selected[0]['distance']:.4f} - {selected[-1]['distance']:.4f}")
        print(f"  Saved to: {output_txt}")
        print(f"  Saved to: {output_json}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()