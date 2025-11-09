#!/usr/bin/env python3
"""
Select representative images closest to the mean embedding using Virchow v2.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import get_encoder


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


def select_representative_images(embeddings, image_list, n_images=25):
    """
    Select n_images closest to the mean embedding.

    Args:
        embeddings: Tensor of shape (N, embedding_dim)
        image_list: List of image filenames
        n_images: Number of representative images to select

    Returns:
        selected_images: List of selected image filenames
        distances: Distances to mean for selected images
    """
    # Compute mean embedding
    mean_embedding = embeddings.mean(dim=0, keepdim=True)

    # Compute distances to mean
    distances = torch.norm(embeddings - mean_embedding, dim=1)

    # Get indices of n_images closest to mean
    _, indices = torch.topk(distances, k=min(n_images, len(image_list)), largest=False)

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
        n_images=args.n_images
    )

    # Print results
    print("\n" + "="*60)
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

        # Also save with distances as JSON
        output_json = args.output.replace('.txt', '_with_distances.json')
        with open(output_json, 'w') as f:
            json.dump([
                {'filename': img, 'distance': dist}
                for img, dist in zip(selected_images, distances)
            ], f, indent=2)
        print(f"Saved detailed results to {output_json}")


if __name__ == '__main__':
    main()