"""
Script para comparar embeddings de dos archivos .pt (original vs Macenko)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
import torch
from tqdm import tqdm
import seaborn as sns


def load_embeddings(pt_path):
    """Cargar embeddings desde archivo .pt"""
    print(f"📂 Loading: {os.path.basename(pt_path)}")
    embeddings = torch.load(pt_path, map_location='cpu')
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")
    
    # Validar que es 2D
    if embeddings.ndim != 2:
        print(f"   ⚠️  Warning: Expected 2D array, got {embeddings.ndim}D")
        if embeddings.ndim == 1:
            # Si es 1D, asumir que es un solo embedding
            embeddings = embeddings.reshape(1, -1)
            print(f"   Reshaped to: {embeddings.shape}")
        elif embeddings.ndim > 2:
            # Si tiene más dimensiones, aplanar
            original_shape = embeddings.shape
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
            print(f"   Reshaped from {original_shape} to: {embeddings.shape}")
    
    # Asegurar que sea formato (N_patches, embedding_dim)
    # Si tiene más columnas que filas, probablemente está transpuesto
    if embeddings.shape[1] > embeddings.shape[0] and embeddings.shape[0] < 100:
        print(f"   ⚠️  Array might be transposed. Transposing...")
        embeddings = embeddings.T
        print(f"   New shape: {embeddings.shape}")
    
    # Eliminar filas con NaN o infinitos
    nan_mask = np.isnan(embeddings).any(axis=1) | np.isinf(embeddings).any(axis=1)
    n_invalid = np.sum(nan_mask)
    if n_invalid > 0:
        print(f"   ⚠️  Removing {n_invalid} rows with NaN/Inf values")
        embeddings = embeddings[~nan_mask]
        print(f"   New shape after cleaning: {embeddings.shape}")
    
    return embeddings


def compute_similarity_metrics(emb_original, emb_macenko):
    """Calcular métricas de similitud entre embeddings"""
    
    print(f"\n📊 Computing similarity metrics...")
    print(f"   Original shape: {emb_original.shape}")
    print(f"   Macenko shape:  {emb_macenko.shape}")
    
    # Verificar que tienen el mismo número de patches
    if emb_original.shape[0] != emb_macenko.shape[0]:
        print(f"⚠️  Warning: Different number of patches!")
        print(f"   Original: {emb_original.shape[0]} patches")
        print(f"   Macenko:  {emb_macenko.shape[0]} patches")
        n = min(emb_original.shape[0], emb_macenko.shape[0])
        print(f"   Using first {n} patches for comparison")
        emb_original = emb_original[:n]
        emb_macenko = emb_macenko[:n]
    
    # Verificar que tienen las mismas dimensiones
    if emb_original.shape[1] != emb_macenko.shape[1]:
        raise ValueError(
            f"Embedding dimensions don't match!\n"
            f"   Original: {emb_original.shape[1]} dims\n"
            f"   Macenko:  {emb_macenko.shape[1]} dims"
        )
    
    n_patches = emb_original.shape[0]
    n_dims = emb_original.shape[1]
    
    print(f"   Comparing {n_patches} patches with {n_dims} dimensions each")
    
    # Distancia coseno y euclidiana para cada par de patches
    cosine_dists = []
    euclidean_dists = []
    
    for i in tqdm(range(n_patches), desc="Computing distances"):
        try:
            # Asegurar que son vectores 1D
            vec1 = emb_original[i].ravel()
            vec2 = emb_macenko[i].ravel()
            
            cos_dist = cosine(vec1, vec2)
            euc_dist = euclidean(vec1, vec2)
            
            # Validar resultados
            if not np.isnan(cos_dist) and not np.isinf(cos_dist):
                cosine_dists.append(cos_dist)
            if not np.isnan(euc_dist) and not np.isinf(euc_dist):
                euclidean_dists.append(euc_dist)
        except Exception as e:
            print(f"\n⚠️  Error computing distance for patch {i}: {e}")
            continue
    
    # Similitud coseno (1 - distancia)
    cosine_sims = [1 - d for d in cosine_dists]
    
    # Correlación por dimensión
    print("Computing dimension correlations...")
    correlations = []
    
    for dim in tqdm(range(n_dims), desc="Correlations"):
        try:
            # Extraer columna (una dimensión específica de todos los patches)
            vec1 = emb_original[:, dim].ravel()
            vec2 = emb_macenko[:, dim].ravel()
            
            # Verificar que no son constantes
            if np.std(vec1) > 1e-10 and np.std(vec2) > 1e-10:
                corr, _ = pearsonr(vec1, vec2)
                if not np.isnan(corr):
                    correlations.append(corr)
        except Exception as e:
            if dim < 5:  # Solo mostrar los primeros errores
                print(f"\n⚠️  Error computing correlation for dim {dim}: {e}")
            continue
    
    if len(correlations) == 0:
        print("⚠️  Warning: No valid correlations computed!")
        correlations = [0.0]  # Valor por defecto
    
    # Norma L2 (magnitud del vector)
    norm_original = np.linalg.norm(emb_original, axis=1)
    norm_macenko = np.linalg.norm(emb_macenko, axis=1)
    norm_diff = np.abs(norm_original - norm_macenko)
    
    metrics = {
        'cosine_distance': {
            'mean': np.mean(cosine_dists),
            'std': np.std(cosine_dists),
            'min': np.min(cosine_dists),
            'max': np.max(cosine_dists),
            'median': np.median(cosine_dists),
            'values': cosine_dists
        },
        'cosine_similarity': {
            'mean': np.mean(cosine_sims),
            'std': np.std(cosine_sims),
            'min': np.min(cosine_sims),
            'max': np.max(cosine_sims),
            'median': np.median(cosine_sims),
            'values': cosine_sims
        },
        'euclidean_distance': {
            'mean': np.mean(euclidean_dists),
            'std': np.std(euclidean_dists),
            'min': np.min(euclidean_dists),
            'max': np.max(euclidean_dists),
            'median': np.median(euclidean_dists),
            'values': euclidean_dists
        },
        'dimension_correlation': {
            'mean': np.mean(correlations),
            'std': np.std(correlations),
            'min': np.min(correlations),
            'max': np.max(correlations),
            'median': np.median(correlations),
            'values': correlations
        },
        'norm_difference': {
            'mean': np.mean(norm_diff),
            'std': np.std(norm_diff),
            'min': np.min(norm_diff),
            'max': np.max(norm_diff),
            'values': norm_diff
        },
        'norm_original': {
            'mean': np.mean(norm_original),
            'values': norm_original
        },
        'norm_macenko': {
            'mean': np.mean(norm_macenko),
            'values': norm_macenko
        }
    }
    
    print(f"   ✓ Computed {len(cosine_sims)} valid similarity scores")
    print(f"   ✓ Computed {len(correlations)} valid dimension correlations")
    
    return metrics


def visualize_embeddings(emb_original, emb_macenko, metrics, slide_id, save_path=None):
    """Crear visualización comparativa de embeddings"""
    
    print(f"\n🎨 Creating visualization...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    fig.suptitle(f'Embedding Comparison: Original vs Macenko\n{slide_id}', 
                 fontsize=16, fontweight='bold')
    
    # 1. PCA 2D - Original
    print("   Computing PCA...")
    ax1 = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    emb_orig_pca = pca.fit_transform(emb_original)
    ax1.scatter(emb_orig_pca[:, 0], emb_orig_pca[:, 1], alpha=0.6, c='blue', s=50)
    ax1.set_title('Original - PCA 2D', fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.grid(True, alpha=0.3)
    
    # 2. PCA 2D - Macenko
    ax2 = fig.add_subplot(gs[0, 1])
    emb_mac_pca = pca.transform(emb_macenko)
    ax2.scatter(emb_mac_pca[:, 0], emb_mac_pca[:, 1], alpha=0.6, c='red', s=50)
    ax2.set_title('Macenko - PCA 2D', fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA 2D - Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(emb_orig_pca[:, 0], emb_orig_pca[:, 1], alpha=0.5, c='blue', 
                s=50, label='Original')
    ax3.scatter(emb_mac_pca[:, 0], emb_mac_pca[:, 1], alpha=0.5, c='red', 
                s=50, label='Macenko')
    # Conectar pares (mostrar solo primeros 100 para claridad)
    n_lines = min(100, len(emb_orig_pca))
    for i in range(n_lines):
        ax3.plot([emb_orig_pca[i, 0], emb_mac_pca[i, 0]], 
                [emb_orig_pca[i, 1], emb_mac_pca[i, 1]], 
                'k-', alpha=0.1, linewidth=0.5)
    ax3.set_title('Overlay - Paired Patches', fontweight='bold')
    ax3.set_xlabel(f'PC1')
    ax3.set_ylabel(f'PC2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Variance Explained
    ax4 = fig.add_subplot(gs[0, 3])
    n_components = min(50, emb_original.shape[1])
    pca_full = PCA(n_components=n_components).fit(emb_original)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ax4.plot(cumsum, 'b-', linewidth=2, label='Cumulative')
    ax4.plot(pca_full.explained_variance_ratio_, 'r--', linewidth=1, alpha=0.7, label='Individual')
    ax4.axhline(y=0.95, color='green', linestyle='--', linewidth=1, label='95%')
    ax4.set_title('PCA - Variance Explained', fontweight='bold')
    ax4.set_xlabel('Component Number')
    ax4.set_ylabel('Variance Explained')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Cosine Similarity Distribution
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(metrics['cosine_similarity']['values'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax5.axvline(metrics['cosine_similarity']['mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {metrics['cosine_similarity']['mean']:.4f}")
    ax5.axvline(metrics['cosine_similarity']['median'], color='orange', linestyle=':', 
                linewidth=2, label=f"Median: {metrics['cosine_similarity']['median']:.4f}")
    ax5.set_title('Cosine Similarity Distribution', fontweight='bold')
    ax5.set_xlabel('Cosine Similarity')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Euclidean Distance Distribution
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(metrics['euclidean_distance']['values'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax6.axvline(metrics['euclidean_distance']['mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {metrics['euclidean_distance']['mean']:.2f}")
    ax6.axvline(metrics['euclidean_distance']['median'], color='orange', linestyle=':', 
                linewidth=2, label=f"Median: {metrics['euclidean_distance']['median']:.2f}")
    ax6.set_title('Euclidean Distance Distribution', fontweight='bold')
    ax6.set_xlabel('Euclidean Distance')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Dimension Correlation
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(metrics['dimension_correlation']['values'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax7.axvline(metrics['dimension_correlation']['mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {metrics['dimension_correlation']['mean']:.4f}")
    ax7.set_title('Per-Dimension Correlation', fontweight='bold')
    ax7.set_xlabel('Pearson Correlation')
    ax7.set_ylabel('Frequency')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Vector Norm Comparison
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.scatter(metrics['norm_original']['values'], 
               metrics['norm_macenko']['values'], alpha=0.4, s=30)
    # Línea identidad
    min_norm = min(metrics['norm_original']['values'].min(), 
                   metrics['norm_macenko']['values'].min())
    max_norm = max(metrics['norm_original']['values'].max(), 
                   metrics['norm_macenko']['values'].max())
    ax8.plot([min_norm, max_norm], [min_norm, max_norm], 'r--', linewidth=2, label='y=x')
    ax8.set_title('Vector Magnitude Comparison', fontweight='bold')
    ax8.set_xlabel('Original L2 Norm')
    ax8.set_ylabel('Macenko L2 Norm')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Cosine Similarity vs Euclidean Distance
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.scatter(metrics['cosine_similarity']['values'], 
               metrics['euclidean_distance']['values'], alpha=0.4, s=30, c='teal')
    ax9.set_title('Similarity vs Distance', fontweight='bold')
    ax9.set_xlabel('Cosine Similarity')
    ax9.set_ylabel('Euclidean Distance')
    ax9.grid(True, alpha=0.3)
    
    # 10. Percentile plot - Cosine Similarity
    ax10 = fig.add_subplot(gs[2, 1])
    sorted_sims = np.sort(metrics['cosine_similarity']['values'])
    percentiles = np.linspace(0, 100, len(sorted_sims))
    ax10.plot(percentiles, sorted_sims, 'b-', linewidth=2)
    ax10.axhline(y=0.9, color='orange', linestyle='--', label='0.9 threshold')
    ax10.axhline(y=0.95, color='red', linestyle='--', label='0.95 threshold')
    ax10.set_title('Cosine Similarity - Percentile Plot', fontweight='bold')
    ax10.set_xlabel('Percentile')
    ax10.set_ylabel('Cosine Similarity')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Top/Bottom similarities
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.axis('off')
    
    sorted_indices = np.argsort(metrics['cosine_similarity']['values'])
    bottom_5 = sorted_indices[:5]
    top_5 = sorted_indices[-5:]
    
    comparison_text = f"""Most/Least Similar Patches:

TOP 5 (Most Similar):
"""
    for idx in reversed(top_5):
        comparison_text += f"  Patch {idx}: {metrics['cosine_similarity']['values'][idx]:.4f}\n"
    
    comparison_text += f"\nBOTTOM 5 (Least Similar):\n"
    for idx in bottom_5:
        comparison_text += f"  Patch {idx}: {metrics['cosine_similarity']['values'][idx]:.4f}\n"
    
    ax11.text(0.05, 0.5, comparison_text, fontsize=10, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # 12. Norm difference distribution
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.hist(metrics['norm_difference']['values'], bins=50, alpha=0.7, color='brown', edgecolor='black')
    ax12.axvline(metrics['norm_difference']['mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {metrics['norm_difference']['mean']:.2f}")
    ax12.set_title('Vector Magnitude Difference', fontweight='bold')
    ax12.set_xlabel('|Norm_original - Norm_macenko|')
    ax12.set_ylabel('Frequency')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    # 13. Summary Statistics
    ax13 = fig.add_subplot(gs[3, :2])
    ax13.axis('off')
    
    summary_text = f"""
SIMILARITY METRICS SUMMARY:

Dataset Info:
  Number of patches:  {len(emb_original)}
  Embedding dimension: {emb_original.shape[1]}

Cosine Similarity:
  Mean:   {metrics['cosine_similarity']['mean']:.4f} ± {metrics['cosine_similarity']['std']:.4f}
  Median: {metrics['cosine_similarity']['median']:.4f}
  Range:  [{metrics['cosine_similarity']['min']:.4f}, {metrics['cosine_similarity']['max']:.4f}]

Euclidean Distance:
  Mean:   {metrics['euclidean_distance']['mean']:.2f} ± {metrics['euclidean_distance']['std']:.2f}
  Median: {metrics['euclidean_distance']['median']:.2f}
  Range:  [{metrics['euclidean_distance']['min']:.2f}, {metrics['euclidean_distance']['max']:.2f}]

Dimension Correlation:
  Mean:   {metrics['dimension_correlation']['mean']:.4f} ± {metrics['dimension_correlation']['std']:.4f}
  Median: {metrics['dimension_correlation']['median']:.4f}
  Range:  [{metrics['dimension_correlation']['min']:.4f}, {metrics['dimension_correlation']['max']:.4f}]

Vector Norm:
  Original mean: {metrics['norm_original']['mean']:.2f}
  Macenko mean:  {metrics['norm_macenko']['mean']:.2f}
  Difference:    {abs(metrics['norm_original']['mean'] - metrics['norm_macenko']['mean']):.2f}
"""
    
    # Interpretación
    cos_sim_mean = metrics['cosine_similarity']['mean']
    if cos_sim_mean > 0.98:
        interpretation = "✅ EXTREMELY HIGH SIMILARITY\n   Embeddings are nearly identical - Macenko has negligible impact"
        box_color = 'lightgreen'
    elif cos_sim_mean > 0.95:
        interpretation = "✅ VERY HIGH SIMILARITY\n   Macenko has minimal impact on embeddings"
        box_color = 'lightgreen'
    elif cos_sim_mean > 0.90:
        interpretation = "✅ HIGH SIMILARITY\n   Macenko causes only minor changes"
        box_color = 'lightblue'
    elif cos_sim_mean > 0.80:
        interpretation = "⚠️  MODERATE SIMILARITY\n   Macenko causes noticeable changes"
        box_color = 'lightyellow'
    else:
        interpretation = "❌ LOW SIMILARITY\n   Macenko significantly affects embeddings"
        box_color = 'lightcoral'
    
    summary_text += f"\n\nINTERPRETATION:\n  {interpretation}"
    
    ax13.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7))
    
    # 14. Cross-Correlation Heatmap
    print("   Computing cross-correlation heatmap...")
    ax14 = fig.add_subplot(gs[3, 2:])
    n_dims_to_show = min(50, emb_original.shape[1])
    
    # Muestrear dimensiones si hay muchas
    if emb_original.shape[1] > 50:
        dim_indices = np.linspace(0, emb_original.shape[1]-1, 50, dtype=int)
    else:
        dim_indices = np.arange(emb_original.shape[1])
    
    corr_matrix = np.corrcoef(emb_original[:, dim_indices].T, 
                              emb_macenko[:, dim_indices].T)
    n = len(dim_indices)
    corr_submatrix = corr_matrix[:n, n:]
    
    im = ax14.imshow(corr_submatrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax14.set_title(f'Cross-Correlation Heatmap\n({"All" if n == emb_original.shape[1] else "Sampled"} {n} dimensions)', 
                   fontweight='bold')
    ax14.set_xlabel('Macenko Dimensions')
    ax14.set_ylabel('Original Dimensions')
    cbar = plt.colorbar(im, ax=ax14, label='Correlation', shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compare embeddings from two .pt files (original vs Macenko)'
    )
    
    parser.add_argument('--original', type=str, required=True,
                        help='Path to original embeddings .pt file')
    parser.add_argument('--macenko', type=str, required=True,
                        help='Path to Macenko-normalized embeddings .pt file')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save comparison figure')
    
    args = parser.parse_args()
    
    # Validar archivos
    if not os.path.exists(args.original):
        print(f"❌ Error: Original file not found: {args.original}")
        return
    
    if not os.path.exists(args.macenko):
        print(f"❌ Error: Macenko file not found: {args.macenko}")
        return
    
    print("\n" + "="*70)
    print("EMBEDDING COMPARISON: ORIGINAL vs MACENKO")
    print("="*70)
    
    # Cargar embeddings
    emb_original = load_embeddings(args.original)
    emb_macenko = load_embeddings(args.macenko)
    
    # Verificar compatibilidad
    if emb_original.shape[1] != emb_macenko.shape[1]:
        print(f"\n❌ Error: Embedding dimensions don't match!")
        print(f"   Original: {emb_original.shape[1]} dimensions")
        print(f"   Macenko:  {emb_macenko.shape[1]} dimensions")
        return
    
    # Extraer slide ID del nombre del archivo
    slide_id = os.path.splitext(os.path.basename(args.original))[0]
    
    # Computar métricas
    metrics = compute_similarity_metrics(emb_original, emb_macenko)
    
    # Mostrar resultados
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"\n📊 Cosine Similarity:")
    print(f"   Mean:   {metrics['cosine_similarity']['mean']:.4f} ± {metrics['cosine_similarity']['std']:.4f}")
    print(f"   Median: {metrics['cosine_similarity']['median']:.4f}")
    print(f"   Range:  [{metrics['cosine_similarity']['min']:.4f}, {metrics['cosine_similarity']['max']:.4f}]")
    
    print(f"\n📏 Euclidean Distance:")
    print(f"   Mean:   {metrics['euclidean_distance']['mean']:.2f} ± {metrics['euclidean_distance']['std']:.2f}")
    print(f"   Median: {metrics['euclidean_distance']['median']:.2f}")
    print(f"   Range:  [{metrics['euclidean_distance']['min']:.2f}, {metrics['euclidean_distance']['max']:.2f}]")
    
    print(f"\n🔗 Dimension Correlation:")
    print(f"   Mean:   {metrics['dimension_correlation']['mean']:.4f} ± {metrics['dimension_correlation']['std']:.4f}")
    print(f"   Median: {metrics['dimension_correlation']['median']:.4f}")
    
    print(f"\n📐 Vector Norms:")
    print(f"   Original: {metrics['norm_original']['mean']:.2f}")
    print(f"   Macenko:  {metrics['norm_macenko']['mean']:.2f}")
    print(f"   Difference: {abs(metrics['norm_original']['mean'] - metrics['norm_macenko']['mean']):.2f}")
    
    # Interpretación
    cos_sim_mean = metrics['cosine_similarity']['mean']
    print(f"\n{'='*70}")
    if cos_sim_mean > 0.98:
        print(f"✅ INTERPRETATION: EXTREMELY HIGH SIMILARITY")
        print(f"   Embeddings are nearly identical")
        print(f"   Macenko normalization has negligible impact")
    elif cos_sim_mean > 0.95:
        print(f"✅ INTERPRETATION: VERY HIGH SIMILARITY")
        print(f"   Macenko has minimal impact on feature representation")
        print(f"   Safe to use either version for downstream tasks")
    elif cos_sim_mean > 0.90:
        print(f"✅ INTERPRETATION: HIGH SIMILARITY")
        print(f"   Macenko causes only minor changes to embeddings")
    elif cos_sim_mean > 0.80:
        print(f"⚠️  INTERPRETATION: MODERATE SIMILARITY")
        print(f"   Macenko causes noticeable but not drastic changes")
        print(f"   Consider consistent preprocessing for downstream tasks")
    else:
        print(f"❌ INTERPRETATION: LOW SIMILARITY")
        print(f"   Macenko significantly affects embeddings")
        print(f"   CRITICAL: Use same normalization for training and inference")
    print(f"{'='*70}")
    
    # Visualizar
    visualize_embeddings(emb_original, emb_macenko, metrics, slide_id, args.save)
    
    print(f"\n{'='*70}")
    print("✅ Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
