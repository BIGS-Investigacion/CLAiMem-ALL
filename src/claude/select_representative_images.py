#!/usr/bin/env python3
"""
Select representative images based on distance to centroid using Virchow v2.

Supports three selection modes:
- closest: Images nearest to the centroid (most representative/typical)
- farthest: Images farthest from centroid (outliers/atypical)
- midrange: Images at medium distance (moderately representative)
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

# Add src to path
import sys
# Go up two levels from src/claude/ to project root, then add src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'src'))

from models.builder import get_encoder


def load_image(image_path, transform):
    """Load and preprocess image for Virchow2."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor


def compute_embeddings(image_dir, image_list, model, transform, batch_size=16):
    """
    Compute embeddings for all images in the list.

    Args:
        image_dir: Directory containing the images
        image_list: List of image filenames
        model: Virchow model
        transform: Image transform function
        batch_size: Batch size for processing

    Returns:
        embeddings: Tensor of shape (N, embedding_dim)
    """
    embeddings = []

    print(f"Computing embeddings for {len(image_list)} images...")

    # Process in batches
    for i in tqdm(range(0, len(image_list), batch_size)):
        batch_files = image_list[i:i + batch_size]
        batch_images = []

        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
            try:
                img_tensor = load_image(img_path, transform)
                batch_images.append(img_tensor)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue

        if len(batch_images) == 0:
            continue

        # Get embeddings for batch
        with torch.no_grad():
            batch_embeddings = model(batch_images)
            embeddings.append(batch_embeddings.cpu())

    # Concatenate all embeddings
    embeddings = torch.cat(embeddings, dim=0)

    # Remove extra dimensions if present (e.g., [N, 1, D] -> [N, D])
    if embeddings.dim() == 3 and embeddings.shape[1] == 1:
        embeddings = embeddings.squeeze(1)

    return embeddings


def select_representative_images(embeddings, image_list, n_images=25, mode='closest'):
    """
    Select n_images according to different selection criteria.

    Args:
        embeddings: Tensor of shape (N, embedding_dim)
        image_list: List of image filenames
        n_images: Number of representative images to select
        mode: Selection mode - 'closest', 'farthest', or 'midrange'
            - 'closest': Images closest to the centroid (mean embedding)
            - 'farthest': Images farthest from the centroid (outliers)
            - 'midrange': Images at medium distance from centroid

    Returns:
        selected_images: List of selected image filenames
        distances: Distances to mean for selected images
    """
    # Compute mean embedding (centroid)
    mean_embedding = embeddings.mean(dim=0, keepdim=True)

    # Compute distances to mean
    distances = torch.norm(embeddings - mean_embedding, dim=1)

    if mode == 'closest':
        # Get indices of n_images closest to mean
        _, indices = torch.topk(distances, k=min(n_images, len(image_list)), largest=False)

    elif mode == 'farthest':
        # Get indices of n_images farthest from mean
        _, indices = torch.topk(distances, k=min(n_images, len(image_list)), largest=True)

    elif mode == 'midrange':
        # Get images at medium distance (around the median distance)
        sorted_distances, sorted_indices = torch.sort(distances)

        # Find the median position
        median_pos = len(sorted_distances) // 2

        # Select n_images around the median
        half_n = n_images // 2
        start_idx = max(0, median_pos - half_n)
        end_idx = min(len(sorted_distances), median_pos + (n_images - half_n))

        indices = sorted_indices[start_idx:end_idx]

    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'closest', 'farthest', or 'midrange'.")

    # Get selected images - indices is 1D tensor
    indices_np = indices.cpu().numpy()
    selected_images = [image_list[int(idx)] for idx in indices_np]
    selected_distances = [float(distances[int(idx)].item()) for idx in indices_np]

    return selected_images, selected_distances


def main():
    parser = argparse.ArgumentParser(
        description='Select representative images using Virchow v2 embeddings'
    )
    parser.add_argument(
        'image_dir',
        help='Directory containing the images'
    )
    parser.add_argument(
        'image_list_file',
        help='Text file with list of image filenames (one per line)'
    )
    parser.add_argument(
        '--n-images',
        type=int,
        default=25,
        help='Number of representative images to select (default: 25)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='closest',
        choices=['closest', 'farthest', 'midrange'],
        help='Selection mode: closest (near centroid), farthest (outliers), midrange (medium distance) (default: closest)'
    )
    parser.add_argument(
        '--output',
        help='Output file to save selected images list'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for processing (default: 16)'
    )
    parser.add_argument(
        '--save-embeddings',
        help='Optional: Save embeddings to file (.npy)'
    )

    args = parser.parse_args()

    # Load image list
    with open(args.image_list_file, 'r') as f:
        image_list = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(image_list)} images from {args.image_list_file}")

    # Initialize Virchow model
    print("Loading Virchow v2 model...")
    model, transform = get_encoder("virchow")

    # Compute embeddings
    embeddings = compute_embeddings(
        args.image_dir,
        image_list,
        model,
        transform,
        batch_size=args.batch_size
    )

    print(f"Computed embeddings shape: {embeddings.shape}")

    # Save embeddings if requested
    if args.save_embeddings:
        np.save(args.save_embeddings, embeddings.numpy())
        print(f"Saved embeddings to {args.save_embeddings}")

    # Select representative images
    selected_images, distances = select_representative_images(
        embeddings,
        image_list,
        n_images=args.n_images,
        mode=args.mode
    )

    # Compute statistics for context
    mean_embedding = embeddings.mean(dim=0, keepdim=True)
    all_distances = torch.norm(embeddings - mean_embedding, dim=1)
    min_dist = all_distances.min().item()
    max_dist = all_distances.max().item()
    median_dist = all_distances.median().item()
    mean_dist = all_distances.mean().item()

    # Print results
    print("\n" + "="*60)
    print(f"Selection mode: {args.mode.upper()}")
    print(f"Distance statistics (all images):")
    print(f"  Min: {min_dist:.4f}, Max: {max_dist:.4f}")
    print(f"  Mean: {mean_dist:.4f}, Median: {median_dist:.4f}")
    print("="*60)
    print(f"Selected {len(selected_images)} representative images:")
    print("="*60)
    for i, (img, dist) in enumerate(zip(selected_images, distances), 1):
        print(f"{i:2d}. {img} (distance: {dist:.4f})")

    # Save to output file if specified
    if args.output:
        with open(args.output, 'w') as f:
            for img in selected_images:
                f.write(f"{img}\n")
        print(f"\nSaved selected images to {args.output}")

        # Also save with distances and metadata as JSON
        output_json = args.output.replace('.txt', '_with_distances.json')
        output_data = {
            'mode': args.mode,
            'n_images': len(selected_images),
            'statistics': {
                'min_distance': min_dist,
                'max_distance': max_dist,
                'mean_distance': mean_dist,
                'median_distance': median_dist
            },
            'selected_images': [
                {'filename': img, 'distance': dist}
                for img, dist in zip(selected_images, distances)
            ]
        }
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved detailed results to {output_json}")


if __name__ == '__main__':
    main()