#!/usr/bin/env python3
"""
==============================================================================
FEATURE SPACE CONSISTENCY ANALYSIS FROM PATCH IMAGES
==============================================================================

This script evaluates whether performance degradation stems from domain shift
in feature representations by analyzing consistency of foundation model embeddings
for diagnostically important patches (high-attention patches).

This version works directly with extracted patch images (PNG) and computes
Virchow v2 embeddings on-the-fly.

Methodology:
1. Load high-attention patch images from both TCGA and CPTAC cohorts
2. Extract Virchow v2 embeddings on-the-fly for each patch
3. Compute class centroids in embedding space for each cohort
4. Measure Euclidean distance between corresponding centroids
5. Compute silhouette coefficients to quantify class separability
6. Compare against intra-class variability as baseline

Key Question:
Do models attend to regions with consistent feature representations across cohorts?
If yes but performance still degrades, this rules out feature-level domain shift.

Expected directory structure:
    data/histomorfologico/
        tcga/
            pam50/
                label_basal_pred_0/
                    {rank}_{slide_id}_x_{x}_y_{y}_a_{attention}.png
                label_her2_pred_1/
                    ...
        cptac/
            pam50/
                label_basal_pred_0/
                    ...

References:
- Silhouette analysis: Rousseeuw, P. J. (1987). Journal of Computational and Applied Mathematics
- CLAM attention mechanism: Lu et al. (2021). Data-efficient and weakly supervised...
- Virchow v2: Vorontsov et al. (2024). A foundation model for clinical-grade pathology

Author: Claude Code
Date: 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.builder import get_encoder

# Suppress warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Color palette: Pink/Magenta shades (pastel tones with good contrast)
PINK_MAGENTA_PALETTE = {
    'dark_magenta': '#8B008B',      # DarkMagenta (original, for reference)
    'dark_orchid': '#9932CC',       # DarkOrchid
    'medium_orchid': '#BA55D3',     # MediumOrchid (pastel)
    'orchid': '#DA70D6',            # Orchid (pastel)
    'plum': '#DDA0DD',              # Plum (pastel)
    'violet': '#EE82EE',            # Violet (pastel)
    'medium_violet_red': '#C71585', # MediumVioletRed (medium intensity)
    'pale_violet_red': '#DB7093',   # PaleVioletRed (pastel)
    'deep_pink': '#FF1493',         # DeepPink
    'hot_pink': '#FF69B4',          # HotPink (medium)
    'light_pink': '#FFB6C1',        # LightPink (very pastel)
    'pink': '#FFC0CB',              # Pink (very pastel)
    'thistle': '#D8BFD8',           # Thistle (very light pastel)
    'lavender': '#E6E6FA'           # Lavender (very light)
}

PAM50_LABELS = {
    'label_basal_pred_0': 'Basal',
    'label_her2_pred_1': 'Her2',
    'label_luma_pred_2': 'LumA',
    'label_lumb_pred_3': 'LumB',
    'label_normal_pred_4': 'Normal'
}

IHC_LABELS = {
    'er': {
        'label_negative_pred_0': 'ER-negative',
        'label_positive_pred_1': 'ER-positive'
    },
    'pr': {
        'label_negative_pred_0': 'PR-negative',
        'label_positive_pred_1': 'PR-positive'
    },
    'her2': {
        'label_negative_pred_0': 'HER2-negative',
        'label_positive_pred_1': 'HER2-positive'
    }
}


# ==============================================================================
# VIRCHOW V2 FEATURE EXTRACTOR
# ==============================================================================

class VirchowV2Extractor:
    """
    Feature extractor using Virchow v2 foundation model.
    """

    def __init__(self, device='cuda'):
        """
        Initialize Virchow v2 model using the project's model builder.

        Args:
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*80}")
        print("LOADING VIRCHOW V2 MODEL")
        print(f"{'='*80}")
        print(f"Device: {self.device}")

        try:
            # Load model using get_encoder from builder
            print("Loading Virchow v2 model using project builder...")
            self.model, self.transforms = get_encoder('virchow', target_img_size=224)
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"✓ Model loaded successfully")
            print(f"  Embedding dimension: 2560")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"ERROR loading Virchow v2: {e}")
            raise e

    def extract_features(self, image_path: Path) -> np.ndarray:
        """
        Extract features from a single patch image.

        Args:
            image_path: Path to patch image

        Returns:
            Feature vector [2560]
        """
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Apply transforms
        img_tensor = self.transforms(img)

        # Extract features (VirchowWrapper expects a list of images)
        with torch.no_grad():
            features = self.model([img_tensor])

        # Return as numpy array [2560]
        return features.cpu().numpy().squeeze()

    def extract_batch_features(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
        """
        Extract features from multiple images in batches.

        Args:
            image_paths: List of paths to patch images
            batch_size: Batch size for processing

        Returns:
            Feature matrix [N, 2560]
        """
        all_features = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="  Extracting features"):
            batch_paths = image_paths[i:i+batch_size]

            # Load and transform images
            batch_tensors = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transforms(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"    Warning: Failed to load {path}: {e}")
                    continue

            if not batch_tensors:
                continue

            # Extract features (VirchowWrapper expects a list of tensors)
            with torch.no_grad():
                features = self.model(batch_tensors)

                # Ensure features are 2D: [batch_size, 2560]
                if features.dim() == 3:
                    features = features.squeeze(1)

                # Convert to numpy and check for invalid values immediately
                features_np = features.cpu().numpy()

                # Check for NaN or Inf in this batch
                if np.isnan(features_np).any() or np.isinf(features_np).any():
                    print(f"    Warning: Batch contains invalid values, filtering...")
                    # Filter out invalid rows
                    valid_mask = ~(np.isnan(features_np).any(axis=1) | np.isinf(features_np).any(axis=1))
                    features_np = features_np[valid_mask]
                    if len(features_np) == 0:
                        continue

                all_features.append(features_np)

        if not all_features:
            return np.array([])

        result = np.vstack(all_features)

        # Final check: ensure 2D output [N, 2560]
        if result.ndim == 3:
            result = result.reshape(result.shape[0], -1)

        return result


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_patch_paths(base_dir: Path, cohort: str, task: str) -> Dict[str, List[Path]]:
    """
    Load paths to all patch images organized by GROUND TRUTH class.

    Folder structure: label_{ground_truth}_pred_{prediction}/
    We extract patches based on ground_truth, not prediction.

    Args:
        base_dir: Base directory (data/histomorfologico)
        cohort: Cohort name ('tcga' or 'cptac')
        task: Task name ('pam50', 'er', 'pr', 'her2')

    Returns:
        Dictionary mapping ground truth class labels to lists of patch image paths
    """
    cohort_dir = base_dir / cohort / task

    if not cohort_dir.exists():
        raise FileNotFoundError(f"Directory not found: {cohort_dir}")

    # Get label mapping for ground truth
    if task == 'pam50':
        # Map folder prefix to clean label name
        gt_label_map = {
            'basal': 'Basal',
            'her2': 'Her2',
            'luma': 'LumA',
            'lumb': 'LumB',
            'normal': 'Normal'
        }
    elif task == 'er':
        gt_label_map = {
            'negative': 'ER-negative',
            'positive': 'ER-positive'
        }
    elif task == 'pr':
        gt_label_map = {
            'negative': 'PR-negative',
            'positive': 'PR-positive'
        }
    elif task == 'her2':
        gt_label_map = {
            'negative': 'HER2-negative',
            'positive': 'HER2-positive'
        }
    else:
        raise ValueError(f"Unknown task: {task}")

    # Collect patches by ground truth class
    # Folders have format: label_{ground_truth}_pred_{prediction}
    class_patches = {}

    for folder_path in cohort_dir.iterdir():
        if not folder_path.is_dir():
            continue

        folder_name = folder_path.name

        # Parse folder name: label_{ground_truth}_pred_{prediction}
        if not folder_name.startswith('label_'):
            continue

        # Extract ground truth from folder name
        # Format: label_basal_pred_0 -> ground_truth = "basal"
        parts = folder_name.split('_pred_')
        if len(parts) != 2:
            print(f"  Warning: Cannot parse folder name: {folder_name}")
            continue

        # Get ground truth part (e.g., "label_basal" -> "basal")
        gt_part = parts[0].replace('label_', '')

        # Map to clean label name
        if gt_part not in gt_label_map:
            print(f"  Warning: Unknown ground truth '{gt_part}' in {folder_name}")
            continue

        class_label = gt_label_map[gt_part]

        # Get all PNG files in this folder
        patches = list(folder_path.glob("*.png"))

        if patches:
            # Add to class_patches, combining all predictions for same ground truth
            if class_label not in class_patches:
                class_patches[class_label] = []
            class_patches[class_label].extend(patches)

    # Print summary
    for class_label in sorted(class_patches.keys()):
        print(f"  ✓ {class_label}: {len(class_patches[class_label])} patches")

    return class_patches


def extract_features_for_cohort(patch_paths: Dict[str, List[Path]],
                                extractor: VirchowV2Extractor,
                                cohort: str,
                                task: str,
                                cache_dir: Optional[Path] = None,
                                batch_size: int = 32) -> Dict[str, np.ndarray]:
    """
    Extract Virchow v2 features for all patches in a cohort.
    Uses caching to avoid recomputing features.

    Args:
        patch_paths: Dictionary mapping class labels to patch paths
        extractor: VirchowV2Extractor instance
        cohort: Cohort name (for cache file naming)
        task: Task name (for cache file naming)
        cache_dir: Directory to store cached features (default: None, no caching)
        batch_size: Batch size for feature extraction

    Returns:
        Dictionary mapping class labels to feature matrices [N, 2560]
    """
    class_features = {}

    for class_label, paths in patch_paths.items():
        print(f"\n  Processing class: {class_label} ({len(paths)} patches)")

        # Check cache first
        if cache_dir is not None:
            cache_file = cache_dir / f"{cohort}_{task}_{class_label}_features.npz"
            if cache_file.exists():
                print(f"    Loading cached features from {cache_file.name}")
                cached_data = np.load(cache_file)
                features = cached_data['features']
                print(f"    ✓ Loaded cached features: {features.shape}")

                # Verify cached features are valid
                has_nan = np.isnan(features).any()
                has_inf = np.isinf(features).any()

                if has_nan or has_inf:
                    print(f"    ⚠ Cached features contain invalid values, recomputing...")
                else:
                    class_features[class_label] = features
                    continue

        # Extract features
        features = extractor.extract_batch_features(paths, batch_size=batch_size)

        if len(features) > 0:
            # Check for and remove patches with NaN or Inf values
            has_nan = np.isnan(features).any()
            has_inf = np.isinf(features).any()

            if has_nan or has_inf:
                print(f"    ⚠ Features contain: NaN={has_nan}, Inf={has_inf}")
                valid_mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
                n_invalid = (~valid_mask).sum()
                print(f"    ⚠ Discarding {n_invalid}/{len(features)} patches with invalid values")
                features = features[valid_mask]

            if len(features) > 0:
                # Final verification
                if np.isnan(features).any() or np.isinf(features).any():
                    print(f"    ✗ ERROR: Features still contain invalid values after filtering!")
                    print(f"       NaN count: {np.isnan(features).sum()}, Inf count: {np.isinf(features).sum()}")
                else:
                    class_features[class_label] = features
                    print(f"    ✓ Extracted features: {features.shape}")
                    print(f"    Stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}, std={features.std():.4f}")

                    # Save to cache
                    if cache_dir is not None:
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        cache_file = cache_dir / f"{cohort}_{task}_{class_label}_features.npz"
                        np.savez_compressed(cache_file, features=features)
                        print(f"    ✓ Cached features to {cache_file.name}")
            else:
                print(f"    ✗ All patches discarded for {class_label}")
        else:
            print(f"    ✗ No features extracted for {class_label}")

    return class_features


# ==============================================================================
# CENTROID ANALYSIS
# ==============================================================================

def compute_class_centroids(class_features: Dict[str, np.ndarray]) -> Dict:
    """
    Compute class centroids.

    Args:
        class_features: Dictionary mapping class labels to feature matrices

    Returns:
        centroids_dict: {class_label: centroid vector [D]}
    """
    centroids = {}

    for label, features in class_features.items():
        # Compute centroid
        centroid = np.mean(features, axis=0)

        # Check for NaN or Inf in centroid
        if np.isnan(centroid).any() or np.isinf(centroid).any():
            print(f"  Warning: Centroid for {label} contains NaN/Inf values, skipping")
            continue

        centroids[label] = centroid
        print(f"  {label}: n_samples={len(features)}, centroid computed")

    return centroids


def compute_inter_cohort_distances(class_features_tcga: Dict,
                                   class_features_cptac: Dict) -> pd.DataFrame:
    """
    Compute distances between centroids of TCGA and CPTAC within the same class.

    Args:
        class_features_tcga: {class_label: feature matrix [N_tcga, D]}
        class_features_cptac: {class_label: feature matrix [N_cptac, D]}

    Returns:
        DataFrame with centroid distance per class
    """
    results = []

    for label in class_features_tcga.keys():
        if label not in class_features_cptac:
            print(f"  Warning: Class {label} not in CPTAC, skipping")
            continue

        features_tcga = class_features_tcga[label]
        features_cptac = class_features_cptac[label]

        n_tcga = len(features_tcga)
        n_cptac = len(features_cptac)

        print(f"  Computing centroid distance for {label}: {n_tcga} TCGA samples, {n_cptac} CPTAC samples")

        # Compute centroids for each cohort
        centroid_tcga = np.mean(features_tcga, axis=0)
        centroid_cptac = np.mean(features_cptac, axis=0)

        # Compute Euclidean distance between centroids
        centroid_distance = np.linalg.norm(centroid_tcga - centroid_cptac)

        results.append({
            'Class': label,
            'Centroid_Distance': centroid_distance,
            'N_TCGA': n_tcga,
            'N_CPTAC': n_cptac
        })

    return pd.DataFrame(results)


def compute_intra_class_variability(class_features_tcga: Dict,
                                    class_features_cptac: Dict) -> pd.DataFrame:
    """
    Compute mean intra-class standard deviation for each cohort.

    Args:
        class_features_tcga: TCGA class features
        class_features_cptac: CPTAC class features

    Returns:
        DataFrame with intra-class variability metrics
    """
    results = []

    for label in class_features_tcga.keys():
        if label not in class_features_cptac:
            continue

        # Compute intra-class std for each cohort
        features_tcga = class_features_tcga[label]
        features_cptac = class_features_cptac[label]

        # Check if we have enough samples (need at least 2 for meaningful std)
        if len(features_tcga) < 2:
            print(f"  Warning: {label} in TCGA has only {len(features_tcga)} sample(s), cannot compute std")
            std_tcga = np.nan
        else:
            # Compute std step by step to avoid overflow
            # 1. Compute mean
            mean_tcga = np.mean(features_tcga, axis=0)
            # 2. Compute variance manually
            variance_tcga = np.mean((features_tcga - mean_tcga) ** 2, axis=0)
            # 3. Take square root to get std per dimension
            std_per_dim_tcga = np.sqrt(variance_tcga)
            # 4. Average across dimensions
            std_tcga = np.mean(std_per_dim_tcga)

            # Check for invalid values
            if np.isnan(std_tcga) or np.isinf(std_tcga):
                print(f"  Warning: {label} TCGA std is {std_tcga}, setting to NaN")
                std_tcga = np.nan

        if len(features_cptac) < 2:
            print(f"  Warning: {label} in CPTAC has only {len(features_cptac)} sample(s), cannot compute std")
            std_cptac = np.nan
        else:
            # Compute std step by step to avoid overflow
            # 1. Compute mean
            mean_cptac = np.mean(features_cptac, axis=0)
            # 2. Compute variance manually
            variance_cptac = np.mean((features_cptac - mean_cptac) ** 2, axis=0)
            # 3. Take square root to get std per dimension
            std_per_dim_cptac = np.sqrt(variance_cptac)
            # 4. Average across dimensions
            std_cptac = np.mean(std_per_dim_cptac)

            # Check for invalid values
            if np.isnan(std_cptac) or np.isinf(std_cptac):
                print(f"  Warning: {label} CPTAC std is {std_cptac}, setting to NaN")
                std_cptac = np.nan

        # Compute mean std, handling NaN values
        if np.isnan(std_tcga) or np.isnan(std_cptac):
            mean_std = np.nan
        else:
            mean_std = (std_tcga + std_cptac) / 2

        results.append({
            'Class': label,
            'Std_TCGA': std_tcga,
            'Std_CPTAC': std_cptac,
            'Mean_Std': mean_std
        })

    return pd.DataFrame(results)


# ==============================================================================
# SILHOUETTE ANALYSIS
# ==============================================================================

def compute_silhouette_scores(class_features: Dict, cohort_name: str) -> Dict:
    """
    Compute silhouette coefficients to quantify class separability.

    Args:
        class_features: Dictionary mapping class labels to feature matrices
        cohort_name: 'TCGA' or 'CPTAC'

    Returns:
        Dictionary with silhouette metrics
    """
    # Prepare data
    all_features = []
    all_labels = []

    for label, features in class_features.items():
        all_features.append(features)
        all_labels.extend([label] * len(features))

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    # Check if we have at least 2 classes (required for silhouette score)
    unique_labels = np.unique(all_labels)
    if len(unique_labels) < 2:
        print(f"  Warning: {cohort_name} has only {len(unique_labels)} class(es). Silhouette score requires at least 2 classes.")
        return {
            'cohort': cohort_name,
            'silhouette_avg': 0.0,
            'per_class_scores': {label: 0.0 for label in class_features.keys()},
            'n_samples': len(all_features),
            'n_classes': len(class_features),
            'error': f'Only {len(unique_labels)} class(es) available'
        }

    # Compute global silhouette score
    silhouette_avg = silhouette_score(all_features, all_labels)

    # Compute per-class silhouette scores
    silhouette_vals = silhouette_samples(all_features, all_labels)

    per_class_scores = {}
    for label in class_features.keys():
        mask = all_labels == label
        per_class_scores[label] = silhouette_vals[mask].mean()

    return {
        'cohort': cohort_name,
        'silhouette_avg': float(silhouette_avg),
        'per_class_scores': per_class_scores,
        'n_samples': len(all_features),
        'n_classes': len(class_features)
    }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_centroid_distances(df_distances: pd.DataFrame,
                            df_variability: pd.DataFrame,
                            task: str,
                            output_path: Optional[Path] = None) -> None:
    """
    Plot centroid distances vs intra-class variability.
    """
    # Merge dataframes
    df = df_distances.merge(df_variability, on='Class')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars with magma palette
    x = np.arange(len(df))
    width = 0.35

    # Use magma color palette
    magma_palette = sns.color_palette("magma", n_colors=2)

    bars1 = ax.bar(x - width/2, df['Centroid_Distance'], width,
                   label='Centroid Distance (TCGA ↔ CPTAC)',
                   color=magma_palette[0], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df['Mean_Std'], width,
                   label='Mean Intra-Class Std',
                   color=magma_palette[1], alpha=0.8, edgecolor='black')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance / Std in Embedding Space', fontsize=12, fontweight='bold')
    ax.set_title(f'Inter-Cohort Distances vs Intra-Class Variability\n{task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Class'], rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_silhouette_comparison(silhouette_tcga: Dict,
                               silhouette_cptac: Dict,
                               task: str,
                               output_path: Optional[Path] = None) -> None:
    """
    Compare silhouette scores between cohorts.
    """
    # Extract per-class scores (only for common classes)
    common_classes = sorted(set(silhouette_tcga['per_class_scores'].keys()) &
                           set(silhouette_cptac['per_class_scores'].keys()))

    if not common_classes:
        print("  Warning: No common classes between cohorts for plotting")
        return

    classes = common_classes
    scores_tcga = [silhouette_tcga['per_class_scores'][c] for c in classes]
    scores_cptac = [silhouette_cptac['per_class_scores'][c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(classes))
    width = 0.35

    # Use magma color palette
    magma_palette = sns.color_palette("magma", n_colors=2)

    bars1 = ax.bar(x - width/2, scores_tcga, width,
                   label=f"TCGA (avg={silhouette_tcga['silhouette_avg']:.3f})",
                   color=magma_palette[0], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, scores_cptac, width,
                   label=f"CPTAC (avg={silhouette_cptac['silhouette_avg']:.3f})",
                   color=magma_palette[1], alpha=0.8, edgecolor='black')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(f'Class Separability (Silhouette Scores)\n{task.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylim([-0.2, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


def plot_tsne_embeddings(class_features_tcga: Dict, class_features_cptac: Dict,
                        task: str, output_path: Optional[Path] = None) -> None:
    """
    Visualize embeddings using t-SNE.
    """
    # Prepare data
    features_list = []
    labels_list = []
    cohort_list = []

    for label, features in class_features_tcga.items():
        features_list.append(features)
        labels_list.extend([label] * len(features))
        cohort_list.extend(['TCGA'] * len(features))

    for label, features in class_features_cptac.items():
        features_list.append(features)
        labels_list.extend([label] * len(features))
        cohort_list.extend(['CPTAC'] * len(features))

    features = np.vstack(features_list)
    labels = np.array(labels_list)
    cohorts = np.array(cohort_list)

    # Subsample if too large (t-SNE is slow)
    if len(features) > 5000:
        print("  Subsampling to 5000 points for t-SNE...")
        indices = np.random.choice(len(features), 5000, replace=False)
        features = features[indices]
        labels = labels[indices]
        cohorts = cohorts[indices]

    # Run t-SNE
    print("  Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(features)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot by class - use magma palette for biological classes
    unique_labels = np.unique(labels)

    # Use magma palette for biological classes
    class_colors = sns.color_palette("magma", n_colors=len(unique_labels))

    # Define different marker styles for each class
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        marker_style = markers[idx % len(markers)]  # Cycle through markers if needed
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[class_colors[idx]], label=label, alpha=0.6, s=30,
                   marker=marker_style, edgecolors='black', linewidths=0.5)

    ax1.set_xlabel('t-SNE 1', fontsize=11)
    ax1.set_ylabel('t-SNE 2', fontsize=11)
    ax1.set_title('Colored by Class', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(alpha=0.3)

    # Plot by cohort - use magma palette for cohort distinction
    cohort_palette = sns.color_palette("magma", n_colors=2)
    for idx, (cohort, marker) in enumerate([('TCGA', 'o'), ('CPTAC', '^')]):
        mask = cohorts == cohort
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[cohort_palette[idx]], label=cohort, alpha=0.6, s=30,
                   marker=marker, edgecolors='black', linewidths=0.5)

    ax2.set_xlabel('t-SNE 1', fontsize=11)
    ax2.set_ylabel('t-SNE 2', fontsize=11)
    ax2.set_title('Colored by Cohort', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(alpha=0.3)

    fig.suptitle(f't-SNE Visualization of High-Attention Patches\n{task.upper()}',
                 fontsize=14, fontweight='bold', y=1.02)
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

def print_results(df_distances: pd.DataFrame, df_variability: pd.DataFrame,
                  silhouette_tcga: Dict, silhouette_cptac: Dict,
                  task: str, n_patches_tcga: int, n_patches_cptac: int) -> None:
    """
    Print formatted results.
    """
    print("\n" + "="*80)
    print(f"FEATURE SPACE CONSISTENCY ANALYSIS: {task.upper()}")
    print("="*80)

    print(f"\nTotal patches analyzed:")
    print(f"  TCGA:  {n_patches_tcga:,} high-attention patches")
    print(f"  CPTAC: {n_patches_cptac:,} high-attention patches")

    # Inter-cohort distances
    print("\n" + "-"*80)
    print("CENTROID DISTANCES (TCGA ↔ CPTAC)")
    print("-"*80)
    print(f"{'Class':<15} {'Centroid Dist':>15} {'TCGA Std':>12} {'CPTAC Std':>12} {'Ratio':>10}")
    print("-"*80)

    df_merged = df_distances.merge(df_variability, on='Class')
    for _, row in df_merged.iterrows():
        # Handle division by zero or NaN
        if np.isnan(row['Mean_Std']) or row['Mean_Std'] == 0:
            ratio = np.nan
        else:
            ratio = row['Centroid_Distance'] / row['Mean_Std']

        ratio_str = f"{ratio:>10.2f}" if not np.isnan(ratio) else "       N/A"
        print(f"{row['Class']:<15} {row['Centroid_Distance']:>15.4f} "
              f"{row['Std_TCGA']:>12.4f} {row['Std_CPTAC']:>12.4f} {ratio_str}")

    mean_distance = np.nanmean(df_distances['Centroid_Distance'].values)
    mean_std = np.nanmean(df_variability['Mean_Std'].values)

    print(f"\nMean centroid distance:     {mean_distance:.4f}")
    print(f"Mean intra-class std:       {mean_std:.4f}")

    # Check if we can compute ratio
    if np.isnan(mean_std) or mean_std == 0:
        print(f"Ratio (dist/std):           N/A (insufficient data)")
    else:
        print(f"Ratio (dist/std):           {mean_distance/mean_std:.2f}")

    # Silhouette scores
    print("\n" + "-"*80)
    print("SILHOUETTE SCORES")
    print("-"*80)
    print(f"{'Class':<15} {'TCGA':>12} {'CPTAC':>12} {'Δ':>10}")
    print("-"*80)

    # Get common classes between both cohorts
    common_classes = set(silhouette_tcga['per_class_scores'].keys()) & set(silhouette_cptac['per_class_scores'].keys())

    for label in sorted(common_classes):
        score_tcga = silhouette_tcga['per_class_scores'][label]
        score_cptac = silhouette_cptac['per_class_scores'][label]
        delta = score_cptac - score_tcga

        print(f"{label:<15} {score_tcga:>12.4f} {score_cptac:>12.4f} {delta:>+10.4f}")

    # Print warning for missing classes
    tcga_only = set(silhouette_tcga['per_class_scores'].keys()) - set(silhouette_cptac['per_class_scores'].keys())
    cptac_only = set(silhouette_cptac['per_class_scores'].keys()) - set(silhouette_tcga['per_class_scores'].keys())

    if tcga_only:
        print(f"\n  Note: Classes only in TCGA: {', '.join(sorted(tcga_only))}")
    if cptac_only:
        print(f"\n  Note: Classes only in CPTAC: {', '.join(sorted(cptac_only))}")

    # Print errors if any
    if 'error' in silhouette_tcga:
        print(f"\n  ⚠ TCGA: {silhouette_tcga['error']}")
    if 'error' in silhouette_cptac:
        print(f"\n  ⚠ CPTAC: {silhouette_cptac['error']}")

    if 'error' not in silhouette_tcga and 'error' not in silhouette_cptac:
        print(f"\nAverage silhouette (TCGA):  {silhouette_tcga['silhouette_avg']:.4f}")
        print(f"Average silhouette (CPTAC): {silhouette_cptac['silhouette_avg']:.4f}")
        print(f"Difference:                 {silhouette_cptac['silhouette_avg'] - silhouette_tcga['silhouette_avg']:+.4f}")
    else:
        print(f"\nSilhouette analysis incomplete due to insufficient classes.")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    # Check if centroid distances are small relative to intra-class variability
    if mean_distance < mean_std:
        print("✓ Centroid distances are SMALLER than intra-class variability.")
        print("  → Feature representations are CONSISTENT across cohorts.")
    else:
        print("✗ Centroid distances are LARGER than intra-class variability.")
        print("  → Feature representations show DOMAIN SHIFT.")

    # Check silhouette scores (only if both are valid)
    if 'error' not in silhouette_tcga and 'error' not in silhouette_cptac:
        if abs(silhouette_tcga['silhouette_avg'] - silhouette_cptac['silhouette_avg']) < 0.1:
            print("\n✓ Silhouette scores are SIMILAR between cohorts.")
            print("  → Class separability is comparable.")
        else:
            print("\n✗ Silhouette scores DIFFER between cohorts.")
            print("  → Class separability has changed.")

    # Overall conclusion
    print("\n" + "-"*80)
    print("CONCLUSION")
    print("-"*80)

    if 'error' in silhouette_tcga or 'error' in silhouette_cptac:
        print("⚠ Silhouette analysis could not be completed due to insufficient classes.")
        print("Analysis limited to centroid distances only.")
        if mean_distance < mean_std:
            print("\nBased on centroid analysis:")
            print("Feature representations appear CONSISTENT across cohorts.")
        else:
            print("\nBased on centroid analysis:")
            print("Feature representations show DOMAIN SHIFT.")
    elif mean_distance < mean_std and abs(silhouette_tcga['silhouette_avg'] - silhouette_cptac['silhouette_avg']) < 0.1:
        print("Feature-level domain shift does NOT explain generalization failure.")
        print("Models attend to similar patterns in both cohorts, but these patterns")
        print("may lack sufficient discriminative power for molecular classification.")
    else:
        print("Evidence of feature-level domain shift detected.")
        print("Attended regions show different feature representations across cohorts.")

    print("="*80 + "\n")


def save_results(df_distances: pd.DataFrame, df_variability: pd.DataFrame,
                 silhouette_tcga: Dict, silhouette_cptac: Dict,
                 output_dir: Path, task: str) -> None:
    """
    Save results to files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save inter-cohort distances and variability
    df_merged = df_distances.merge(df_variability, on='Class')

    # Select only relevant columns for CSV output
    columns_to_save = ['Class', 'Centroid_Distance', 'Std_TCGA', 'Std_CPTAC', 'Mean_Std']
    df_output = df_merged[columns_to_save].copy()

    # Filter out rows with NaN or Inf values
    n_before = len(df_output)
    df_output = df_output.replace([np.inf, -np.inf], np.nan)
    df_output = df_output.dropna()
    n_after = len(df_output)

    if n_before > n_after:
        print(f"  Warning: Dropped {n_before - n_after} classes with invalid values")

    csv_path = output_dir / f'{task}_centroid_distances.csv'
    df_output.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  ✓ Saved: {csv_path.name}")

    # Save silhouette scores (only for common classes)
    common_classes = set(silhouette_tcga['per_class_scores'].keys()) & set(silhouette_cptac['per_class_scores'].keys())

    silhouette_df = pd.DataFrame({
        'Class': sorted(common_classes),
        'Silhouette_TCGA': [silhouette_tcga['per_class_scores'][c] for c in sorted(common_classes)],
        'Silhouette_CPTAC': [silhouette_cptac['per_class_scores'][c] for c in sorted(common_classes)]
    })
    csv_path = output_dir / f'{task}_silhouette_scores.csv'
    silhouette_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  ✓ Saved: {csv_path.name}")

    # Save summary JSON
    results = {
        'task': task,
        'inter_cohort_analysis': {
            'mean_centroid_distance': float(np.nanmean(df_distances['Centroid_Distance'].values)),
            'mean_std': float(np.nanmean(df_variability['Mean_Std'].values)),
            'per_class': df_merged.to_dict(orient='records')
        },
        'silhouette_analysis': {
            'tcga': silhouette_tcga,
            'cptac': silhouette_cptac
        }
    }

    json_path = output_dir / f'{task}_feature_space_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: {json_path.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Feature Space Consistency Analysis from Patch Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python feature_space_consistency_from_patches.py \\
    --patches_dir data/histomorfologico/ \\
    --task pam50 \\
    --batch_size 32 \\
    --output results/feature_space_analysis/
        """
    )

    # Input/output
    parser.add_argument('--patches_dir', '-p', type=str, 
                        help='Directory containing extracted patches (data/histomorfologico)', default='data/histomorfologico/')
    parser.add_argument('--output', '-o', type=str, 
                        help='Output directory', default='results/feature_space_analysis/')

    # Task
    parser.add_argument('--task', '-t', type=str, 
                        choices=['pam50', 'er', 'pr', 'her2'],
                        help='Task name', default='her2')

    # Processing options
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='Batch size for feature extraction (default: 32)')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for feature extraction (default: cuda)')
    parser.add_argument('--cache_dir', '-c', type=str, default='cache/virchow_features',
                        help='Directory to cache extracted features (default: cache/virchow_features)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable feature caching')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--plot_format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'])
    parser.add_argument('--skip_tsne', action='store_true',
                        help='Skip t-SNE visualization (saves time)')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = Path(args.patches_dir)

    print("\n" + "="*80)
    print("FEATURE SPACE CONSISTENCY ANALYSIS")
    print("="*80)
    print(f"Task:        {args.task.upper()}")
    print(f"Patches dir: {patches_dir}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Device:      {args.device}")
    print(f"Output:      {output_dir}")
    print("="*80 + "\n")

    # Initialize Virchow v2 extractor
    extractor = VirchowV2Extractor(device=args.device)

    # Load patch paths
    print("Loading patch paths...")
    print("\nTCGA:")
    tcga_patches = load_patch_paths(patches_dir, 'tcga', args.task)
    print("\nCPTAC:")
    cptac_patches = load_patch_paths(patches_dir, 'cptac', args.task)

    # Count total patches
    n_patches_tcga = sum(len(paths) for paths in tcga_patches.values())
    n_patches_cptac = sum(len(paths) for paths in cptac_patches.values())

    print(f"\n{'='*80}")
    print(f"Total patches to process:")
    print(f"  TCGA:  {n_patches_tcga:,} patches")
    print(f"  CPTAC: {n_patches_cptac:,} patches")
    print(f"{'='*80}\n")

    # Setup cache directory
    cache_dir = None if args.no_cache else Path(args.cache_dir)
    if cache_dir is not None:
        print(f"\nFeature caching enabled: {cache_dir}")

    # Extract features
    print("Extracting Virchow v2 features from TCGA patches...")
    tcga_features = extract_features_for_cohort(
        tcga_patches, extractor,
        cohort='tcga', task=args.task,
        cache_dir=cache_dir,
        batch_size=args.batch_size
    )

    print("\nExtracting Virchow v2 features from CPTAC patches...")
    cptac_features = extract_features_for_cohort(
        cptac_patches, extractor,
        cohort='cptac', task=args.task,
        cache_dir=cache_dir,
        batch_size=args.batch_size
    )

    print("\n✓ Feature extraction complete")

    # Compute inter-cohort distances and intra-class variability
    print("\nComputing inter-cohort distances...")
    df_distances = compute_inter_cohort_distances(tcga_features, cptac_features)

    print("\nComputing intra-class variability...")
    df_variability = compute_intra_class_variability(tcga_features, cptac_features)

    print("  ✓ Distance analysis complete")

    # Silhouette analysis
    print("\nComputing silhouette scores...")
    silhouette_tcga = compute_silhouette_scores(tcga_features, 'TCGA')
    silhouette_cptac = compute_silhouette_scores(cptac_features, 'CPTAC')

    print("  ✓ Silhouette scores computed")

    # Print results
    print_results(df_distances, df_variability,
                  silhouette_tcga, silhouette_cptac, args.task,
                  n_patches_tcga, n_patches_cptac)

    # Save results
    print("Saving results...")
    save_results(df_distances, df_variability,
                 silhouette_tcga, silhouette_cptac,
                 output_dir, args.task)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualizations...")

        plot_centroid_distances(
            df_distances, df_variability, args.task,
            output_dir / f'{args.task}_centroid_distances.{args.plot_format}'
        )

        plot_silhouette_comparison(
            silhouette_tcga, silhouette_cptac, args.task,
            output_dir / f'{args.task}_silhouette_comparison.{args.plot_format}'
        )

        if not args.skip_tsne:
            plot_tsne_embeddings(
                tcga_features, cptac_features, args.task,
                output_dir / f'{args.task}_tsne_embeddings.{args.plot_format}'
            )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/\n")


if __name__ == '__main__':
    main()
