#!/usr/bin/env python3
"""
==============================================================================
FEATURE SPACE CONSISTENCY ANALYSIS WITH FEATURE SELECTION
==============================================================================

This script extends the feature space consistency analysis by first performing
feature selection on TCGA data, then applying the selected dimensions to both
TCGA and CPTAC for fair comparison.

Feature Selection Strategy:
1. Use TCGA data to identify most discriminative feature dimensions
2. Select top-K dimensions based on:
   - Variance across samples (filter low-variance features)
   - Class separability (Fisher score, ANOVA F-score)
   - Or use PCA to find principal components
3. Apply selected dimensions to both TCGA and CPTAC
4. Perform same inter-cohort distance analysis on reduced features

This approach tests whether domain shift persists even when using only
the most discriminative dimensions learned from the source domain (TCGA).

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
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from tqdm import tqdm

# Import from the main script
sys.path.insert(0, str(Path(__file__).parent.parent))
from claude.feature_space_consistency_from_patches import (
    VirchowV2Extractor,
    load_patch_paths,
    extract_features_for_cohort,
    compute_intra_class_variability,
    compute_silhouette_scores,
    PINK_MAGENTA_PALETTE
)

warnings.filterwarnings('ignore')


# ==============================================================================
# FEATURE SELECTION
# ==============================================================================

def select_features_variance(features: np.ndarray,
                             n_features: int = 500,
                             variance_threshold: float = 0.0) -> np.ndarray:
    """
    Select features based on variance.

    Args:
        features: Feature matrix [N, D]
        n_features: Number of features to select
        variance_threshold: Minimum variance threshold

    Returns:
        Indices of selected features
    """
    variances = np.var(features, axis=0)

    # Filter by threshold
    valid_features = variances > variance_threshold

    # Sort by variance and select top-K
    sorted_indices = np.argsort(variances)[::-1]

    # Take top n_features that pass threshold
    selected = []
    for idx in sorted_indices:
        if valid_features[idx]:
            selected.append(idx)
        if len(selected) >= n_features:
            break

    return np.array(selected)


def select_features_anova(features: Dict[str, np.ndarray],
                          n_features: int = 500) -> np.ndarray:
    """
    Select features using ANOVA F-test (class separability).

    Args:
        features: Dictionary mapping class labels to feature matrices
        n_features: Number of features to select

    Returns:
        Indices of selected features
    """
    # Prepare data for ANOVA
    all_features = []
    all_labels = []

    for label, feats in features.items():
        all_features.append(feats)
        all_labels.extend([label] * len(feats))

    X = np.vstack(all_features)
    y = np.array(all_labels)

    # Use sklearn's f_classif
    selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
    selector.fit(X, y)

    selected_indices = selector.get_support(indices=True)

    return selected_indices


def select_features_mutual_info(features: Dict[str, np.ndarray],
                                n_features: int = 500) -> np.ndarray:
    """
    Select features using mutual information.

    Args:
        features: Dictionary mapping class labels to feature matrices
        n_features: Number of features to select

    Returns:
        Indices of selected features
    """
    # Prepare data
    all_features = []
    all_labels = []

    for label, feats in features.items():
        all_features.append(feats)
        all_labels.extend([label] * len(feats))

    X = np.vstack(all_features)
    y = np.array(all_labels)

    # Use mutual information
    selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
    selector.fit(X, y)

    selected_indices = selector.get_support(indices=True)

    return selected_indices


def select_features_pca(features: Dict[str, np.ndarray],
                       n_components: int = 500) -> Tuple[PCA, np.ndarray]:
    """
    Select features using PCA (dimensionality reduction).

    Args:
        features: Dictionary mapping class labels to feature matrices
        n_components: Number of principal components

    Returns:
        Tuple of (fitted PCA object, explained variance ratio)
    """
    # Prepare data
    all_features = []

    for label, feats in features.items():
        all_features.append(feats)

    X = np.vstack(all_features)

    # Fit PCA - n_components must be <= min(n_samples, n_features)
    max_components = min(X.shape[0], X.shape[1])
    n_components_actual = min(n_components, max_components)

    pca = PCA(n_components=n_components_actual)
    pca.fit(X)

    return pca, pca.explained_variance_ratio_


def apply_feature_selection(features: Dict[str, np.ndarray],
                            selected_indices: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply feature selection to all classes.

    Args:
        features: Dictionary mapping class labels to feature matrices
        selected_indices: Indices of selected features

    Returns:
        Dictionary with reduced features
    """
    reduced_features = {}

    for label, feats in features.items():
        reduced_features[label] = feats[:, selected_indices]

    return reduced_features


def apply_pca_transform(features: Dict[str, np.ndarray],
                       pca: PCA) -> Dict[str, np.ndarray]:
    """
    Apply PCA transformation to all classes.

    Args:
        features: Dictionary mapping class labels to feature matrices
        pca: Fitted PCA object

    Returns:
        Dictionary with PCA-transformed features
    """
    transformed_features = {}

    for label, feats in features.items():
        transformed_features[label] = pca.transform(feats)

    return transformed_features


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_inter_cohort_distances(df_distances: pd.DataFrame,
                                df_variability: pd.DataFrame,
                                task: str,
                                method: str,
                                output_path: Optional[Path] = None) -> None:
    """
    Plot inter-cohort distances vs intra-class variability.
    """
    df = df_distances.merge(df_variability, on='Class')

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    pastel_palette = sns.color_palette("ch:s=-.2,r=.6", n_colors=2)

    bars1 = ax.bar(x - width/2, df['Centroid_Distance'], width,
                   label='Centroid Distance (TCGA ↔ CPTAC)',
                   color=pastel_palette[0], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df['Mean_Std'], width,
                   label='Mean Intra-Class Std',
                   color=pastel_palette[1], alpha=0.8, edgecolor='black')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance / Std in Embedding Space', fontsize=12, fontweight='bold')
    ax.set_title(f'Inter-Cohort Distances vs Intra-Class Variability\n{task.upper()} - {method.upper()}',
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
                               method: str,
                               output_path: Optional[Path] = None) -> None:
    """
    Compare silhouette scores between cohorts.
    """
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

    pastel_palette = sns.color_palette("ch:s=-.2,r=.6", n_colors=2)

    bars1 = ax.bar(x - width/2, scores_tcga, width,
                   label=f"TCGA (avg={silhouette_tcga['silhouette_avg']:.3f})",
                   color=pastel_palette[0], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, scores_cptac, width,
                   label=f"CPTAC (avg={silhouette_cptac['silhouette_avg']:.3f})",
                   color=pastel_palette[1], alpha=0.8, edgecolor='black')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(f'Class Separability (Silhouette Scores)\n{task.upper()} - {method.upper()}',
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
                        task: str, method: str, output_path: Optional[Path] = None) -> None:
    """
    Visualize embeddings using t-SNE.
    """
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

    # Subsample if too large
    if len(features) > 5000:
        print("  Subsampling to 5000 points for t-SNE...")
        indices = np.random.choice(len(features), 5000, replace=False)
        features = features[indices]
        labels = labels[indices]
        cohorts = cohorts[indices]

    # Run t-SNE
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(features)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot by class
    unique_labels = np.unique(labels)
    class_colors = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(unique_labels))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        marker_style = markers[idx % len(markers)]
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[class_colors[idx]], label=label, alpha=0.6, s=30,
                   marker=marker_style, edgecolors='black', linewidths=0.5)

    ax1.set_xlabel('t-SNE 1', fontsize=11)
    ax1.set_ylabel('t-SNE 2', fontsize=11)
    ax1.set_title('Colored by Class', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(alpha=0.3)

    # Plot by cohort
    cohort_palette = sns.color_palette("ch:s=.25,rot=-.25", n_colors=2)
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

    plt.suptitle(f't-SNE Visualization: {task.upper()} - {method.upper()}',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
    else:
        plt.show()

    plt.close()


# ==============================================================================
# INTER-COHORT DISTANCE COMPUTATION (same as main script but copied for clarity)
# ==============================================================================

def compute_inter_cohort_distances(class_features_tcga: Dict,
                                   class_features_cptac: Dict) -> pd.DataFrame:
    """
    Compute distances between centroids of TCGA and CPTAC within the same class.
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


# ==============================================================================
# MAIN ANALYSIS PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Feature Space Consistency Analysis with Feature Selection'
    )

    # Input/output
    parser.add_argument('--patches_dir', type=str, default='data/histomorfologico',
                        help='Directory containing extracted patches')
    parser.add_argument('--output', type=str, default='results/feature_space_analysis_selected',
                        help='Output directory')

    # Task
    parser.add_argument('--task', type=str, choices=['pam50', 'er', 'pr', 'her2'],
                        default='her2', help='Task name')

    # Feature selection
    parser.add_argument('--selection_method', type=str,
                        choices=['variance', 'anova', 'mutual_info', 'pca'],
                        default='anova',
                        help='Feature selection method')
    parser.add_argument('--n_features', type=int, default=500,
                        help='Number of features/components to select')

    # Processing options
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--cache_dir', type=str, default='cache/virchow_features',
                        help='Directory to cache extracted features')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable feature caching')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.patches_dir)

    print("="*80)
    print("FEATURE SPACE CONSISTENCY ANALYSIS WITH FEATURE SELECTION")
    print("="*80)
    print(f"Task:               {args.task.upper()}")
    print(f"Selection method:   {args.selection_method}")
    print(f"N features:         {args.n_features}")
    print(f"Patches dir:        {args.patches_dir}")
    print(f"Output:             {args.output}")
    print("="*80)
    print()

    # Load or extract features (using cache from main script)
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    print("Loading Virchow v2 model...")
    extractor = VirchowV2Extractor(device=args.device)

    print("\nLoading patches...")
    tcga_patches = load_patch_paths(base_dir, 'tcga', args.task)
    cptac_patches = load_patch_paths(base_dir, 'cptac', args.task)

    print("\nExtracting features (using cache if available)...")
    tcga_features = extract_features_for_cohort(
        tcga_patches, extractor,
        cohort='tcga', task=args.task,
        cache_dir=cache_dir, batch_size=args.batch_size
    )
    cptac_features = extract_features_for_cohort(
        cptac_patches, extractor,
        cohort='cptac', task=args.task,
        cache_dir=cache_dir, batch_size=args.batch_size
    )

    # Feature selection on TCGA
    print(f"\n{'='*80}")
    print(f"FEATURE SELECTION ON TCGA ({args.selection_method.upper()})")
    print("="*80)

    if args.selection_method == 'pca':
        pca, explained_var = select_features_pca(tcga_features, args.n_features)
        print(f"PCA fitted with {pca.n_components_} components")
        print(f"Explained variance: {explained_var.sum():.4f} ({explained_var.sum()*100:.2f}%)")

        # Transform both datasets
        tcga_reduced = apply_pca_transform(tcga_features, pca)
        cptac_reduced = apply_pca_transform(cptac_features, pca)

        # Save PCA model
        import pickle
        pca_path = output_dir / f'{args.task}_pca_model.pkl'
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"✓ Saved PCA model to {pca_path.name}")

    else:
        # Select features
        if args.selection_method == 'variance':
            selected_indices = select_features_variance(
                np.vstack([f for f in tcga_features.values()]),
                n_features=args.n_features
            )
        elif args.selection_method == 'anova':
            selected_indices = select_features_anova(tcga_features, args.n_features)
        elif args.selection_method == 'mutual_info':
            selected_indices = select_features_mutual_info(tcga_features, args.n_features)

        print(f"Selected {len(selected_indices)} features from {next(iter(tcga_features.values())).shape[1]} total")

        # Apply selection to both datasets
        tcga_reduced = apply_feature_selection(tcga_features, selected_indices)
        cptac_reduced = apply_feature_selection(cptac_features, selected_indices)

        # Save selected indices
        indices_path = output_dir / f'{args.task}_selected_features.npy'
        np.save(indices_path, selected_indices)
        print(f"✓ Saved selected feature indices to {indices_path.name}")

    # Compute analysis on reduced features
    print(f"\n{'='*80}")
    print("ANALYSIS ON SELECTED FEATURES")
    print("="*80)

    print("\nComputing inter-cohort distances...")
    df_distances = compute_inter_cohort_distances(tcga_reduced, cptac_reduced)

    print("\nComputing intra-class variability...")
    df_variability = compute_intra_class_variability(tcga_reduced, cptac_reduced)

    print("\nComputing silhouette scores...")
    silhouette_tcga = compute_silhouette_scores(tcga_reduced, 'TCGA')
    silhouette_cptac = compute_silhouette_scores(cptac_reduced, 'CPTAC')

    # Save results
    print("\nSaving results...")

    df_merged = df_distances.merge(df_variability, on='Class')
    columns_to_save = ['Class', 'Centroid_Distance', 'Std_TCGA', 'Std_CPTAC', 'Mean_Std']
    df_output = df_merged[columns_to_save].copy()

    csv_path = output_dir / f'{args.task}_selected_{args.selection_method}_distances.csv'
    df_output.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"✓ Saved: {csv_path.name}")

    # Save JSON summary
    results = {
        'task': args.task,
        'selection_method': args.selection_method,
        'n_features': args.n_features,
        'original_dimensions': next(iter(tcga_features.values())).shape[1],
        'selected_dimensions': next(iter(tcga_reduced.values())).shape[1],
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

    json_path = output_dir / f'{args.task}_selected_{args.selection_method}_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {json_path.name}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Plot 1: Inter-cohort distances
    plot_path = output_dir / f'{args.task}_selected_{args.selection_method}_distances.png'
    plot_inter_cohort_distances(df_distances, df_variability, args.task, args.selection_method, plot_path)

    # Plot 2: Silhouette comparison
    plot_path = output_dir / f'{args.task}_selected_{args.selection_method}_silhouette.png'
    plot_silhouette_comparison(silhouette_tcga, silhouette_cptac, args.task, args.selection_method, plot_path)

    # Plot 3: t-SNE embeddings
    plot_path = output_dir / f'{args.task}_selected_{args.selection_method}_tsne.png'
    plot_tsne_embeddings(tcga_reduced, cptac_reduced, args.task, args.selection_method, plot_path)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
