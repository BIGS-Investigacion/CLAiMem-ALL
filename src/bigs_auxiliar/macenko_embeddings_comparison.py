import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from tqdm import tqdm
import os



# ============================================================================
# FUNCIÓN PARA CARGAR EMBEDDINGS DE UN WSI
# ============================================================================
def load_wsi_embeddings(pt_file):
    """
    Carga embeddings de un archivo .pt
    Asume formato: torch.load(file) devuelve tensor [num_patches, embedding_dim]
    O dict con keys como 'features', 'embeddings', etc.
    """
    try:
        data = torch.load(pt_file, map_location='cpu')
        
        # Si es un tensor directamente
        if isinstance(data, torch.Tensor):
            return data.numpy()
        
        # Si es un dict, buscar el key correcto
        # Ajusta según tu formato específico
        if isinstance(data, dict):
            # Intenta diferentes keys comunes
            for key in ['features', 'embeddings', 'feats', 'z']:
                if key in data:
                    emb = data[key]
                    if isinstance(emb, torch.Tensor):
                        return emb.numpy()
                    return np.array(emb)
            
            # Si no encuentra ninguno, imprimir keys disponibles
            print(f"Keys disponibles en {pt_file}: {data.keys()}")
            raise KeyError("No se encontró key de embeddings")
        
        return np.array(data)
    
    except Exception as e:
        print(f"Error cargando {pt_file}: {e}")
        return None

# ============================================================================
# FUNCIÓN PARA CALCULAR MÉTRICAS ENTRE DOS CONJUNTOS DE EMBEDDINGS
# ============================================================================
def calculate_pairwise_metrics(emb_original, emb_macenko):
    """
    Calcula métricas patch-by-patch entre embeddings original y Macenko
    
    Args:
        emb_original: numpy array [num_patches, 1, embedding_dim] o [num_patches, embedding_dim]
        emb_macenko: numpy array [num_patches, 1, embedding_dim] o [num_patches, embedding_dim]
    
    Returns:
        dict con listas de métricas por patch
    """
    # Asegurar formato 2D [num_patches, embedding_dim]
    if emb_original.ndim == 3:
        emb_original = emb_original.squeeze(1)  # [num_patches, 1, dim] → [num_patches, dim]
    
    if emb_macenko.ndim == 3:
        emb_macenko = emb_macenko.squeeze(1)
    
    assert emb_original.shape == emb_macenko.shape, \
        f"Shape mismatch: {emb_original.shape} vs {emb_macenko.shape}"
    
    assert emb_original.ndim == 2, f"Debe ser 2D, es {emb_original.ndim}D"
    
    num_patches, embedding_dim = emb_original.shape
    
    metrics = {
        'cosine_similarity': [],
        'euclidean_distance': [],
        'pearson_correlation': []
    }
    
    for i in range(num_patches):
        z_orig = emb_original[i]  # Shape: [embedding_dim,] después del squeeze
        z_mack = emb_macenko[i]
        
        # IMPORTANTE: Asegurar que sean 1D
        z_orig = z_orig.squeeze()  # Por si aún tiene shape [1, dim]
        z_mack = z_mack.squeeze()
        
        # 1. Cosine similarity - reshape a 2D para sklearn
        cos_sim = cosine_similarity(
            z_orig.reshape(1, -1), 
            z_mack.reshape(1, -1)
        )[0][0]
        metrics['cosine_similarity'].append(cos_sim)
        
        # 2. Euclidean distance
        euc_dist = np.linalg.norm(z_orig - z_mack)
        metrics['euclidean_distance'].append(euc_dist)
        
        # 3. Pearson correlation
        pearson_corr, _ = pearsonr(z_orig, z_mack)
        metrics['pearson_correlation'].append(pearson_corr)
    
    return metrics

# ============================================================================
# FUNCIÓN PARA PROCESAR UN COHORT COMPLETO
# ============================================================================
def process_cohort(original_dir, macenko_dir, cohort_name):
    """
    Procesa todos los WSIs de un cohort y calcula métricas
    """
    print(f"\n{'='*60}")
    print(f"Procesando cohort: {cohort_name}")
    print(f"{'='*60}")
    
    # Listar archivos .pt
    original_files = sorted(list(Path(original_dir).glob("*.pt")))
    macenko_files = sorted(list(Path(macenko_dir).glob("*.pt")))
    
    # Crear dict por nombre de archivo
    original_dict = {f.stem: f for f in original_files}
    macenko_dict = {f.stem: f for f in macenko_files}
    
    # Encontrar WSIs comunes
    common_wsis = set(original_dict.keys()) & set(macenko_dict.keys())
    print(f"WSIs con embeddings en ambos: {len(common_wsis)}")
    
    if len(common_wsis) == 0:
        print("⚠️ No se encontraron WSIs comunes!")
        return pd.DataFrame()
    
    # Procesar cada WSI
    all_results = []
    
    for wsi_id in tqdm(common_wsis, desc=f"Procesando {cohort_name}"):
        # Cargar embeddings
        emb_orig = load_wsi_embeddings(original_dict[wsi_id])
        emb_mack = load_wsi_embeddings(macenko_dict[wsi_id])
        
        if emb_orig is None or emb_mack is None:
            continue
        
        # Verificar shapes
        if emb_orig.shape != emb_mack.shape:
            print(f"⚠️ Shape mismatch en {wsi_id}: {emb_orig.shape} vs {emb_mack.shape}")
            continue
        
        # Calcular métricas
        metrics = calculate_pairwise_metrics(emb_orig, emb_mack)
        
        # Guardar resultados por patch
        num_patches = emb_orig.shape[0]
        for i in range(num_patches):
            all_results.append({
                'wsi_id': wsi_id,
                'patch_idx': i,
                'cohort': cohort_name,
                'cosine_similarity': metrics['cosine_similarity'][i],
                'euclidean_distance': metrics['euclidean_distance'][i],
                'pearson_correlation': metrics['pearson_correlation'][i]
            })
    
    df = pd.DataFrame(all_results)
    
    print(f"\n✅ Procesado: {len(common_wsis)} WSIs, {len(df)} patches totales")
    
    return df

def analysis_by_wsi():
    # ============================================================================
    # CONFIGURACIÓN
    # ============================================================================
    # Directorios con embeddings
    TCGA_ORIGINAL_DIR = "/media/jorge/MASIVO_PORTATIL_1/macenko/.features_20x/tcga/features_virchow/pt_files"
    TCGA_MACENKO_DIR = "/media/jorge/MASIVO_PORTATIL_1/macenko/.features_20x/tcga/features_virchow_macenko/pt_files"
    CPTAC_ORIGINAL_DIR = "/media/jorge/MASIVO_PORTATIL_1/macenko/.features_20x/cptac/features_virchow/pt_files"
    CPTAC_MACENKO_DIR = "/media/jorge/MASIVO_PORTATIL_1/macenko/.features_20x/cptac/features_virchow_macenko/pt_files"

    OUTPUT_DIR = "stain_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # ============================================================================
    # PROCESAR AMBOS COHORTS
    # ============================================================================
    print("Iniciando análisis de robustez de Macenko normalization...")

    df_tcga = process_cohort(TCGA_ORIGINAL_DIR, TCGA_MACENKO_DIR, "TCGA")
    df_cptac = process_cohort(CPTAC_ORIGINAL_DIR, CPTAC_MACENKO_DIR, "CPTAC")

    # Combinar
    df_all = pd.concat([df_tcga, df_cptac], ignore_index=True)

    # Guardar
    output_file = f"{OUTPUT_DIR}/stain_robustness_metrics.csv"
    df_all.to_csv(output_file, index=False)
    print(f"\n✅ Métricas guardadas en: {output_file}")

    # ============================================================================
    # ESTADÍSTICAS DESCRIPTIVAS
    # ============================================================================
    print("\n" + "="*70)
    print("ESTADÍSTICAS DESCRIPTIVAS - ROBUSTEZ A MACENKO NORMALIZATION")
    print("="*70)

    for cohort in ['TCGA', 'CPTAC']:
        df_cohort = df_all[df_all['cohort'] == cohort]
        
        if len(df_cohort) == 0:
            continue
        
        print(f"\n{cohort} (n={len(df_cohort)} patches):")
        print("-" * 60)
        
        for metric, name in [
            ('cosine_similarity', 'Cosine Similarity'),
            ('euclidean_distance', 'Euclidean Distance'),
            ('pearson_correlation', 'Pearson Correlation')
        ]:
            values = df_cohort[metric]
            print(f"\n  {name}:")
            print(f"    Mean   : {values.mean():.4f}")
            print(f"    Median : {values.median():.4f}")
            print(f"    Std    : {values.std():.4f}")
            print(f"    Q1     : {values.quantile(0.25):.4f}")
            print(f"    Q3     : {values.quantile(0.75):.4f}")
            print(f"    IQR    : {values.quantile(0.75) - values.quantile(0.25):.4f}")


    # ============================================================================
    # ANÁLISIS POR WSI (agregado)
    # ============================================================================
    print("\n" + "="*70)
    print("MÉTRICAS AGREGADAS POR WSI")
    print("="*70)

    df_wsi = df_all.groupby(['wsi_id', 'cohort']).agg({
        'cosine_similarity': ['mean', 'std', 'min'],
        'euclidean_distance': ['mean', 'std', 'max'],
        'pearson_correlation': ['mean', 'std', 'min']
    }).reset_index()

    df_wsi.columns = ['_'.join(col).strip('_') for col in df_wsi.columns.values]
    df_wsi.to_csv(f"{OUTPUT_DIR}/stain_robustness_per_wsi.csv", index=False)

    print(f"\n✅ Métricas por WSI guardadas en: {OUTPUT_DIR}/stain_robustness_per_wsi.csv")

    # Mostrar top 5 WSIs más variables
    print("\nTop 5 WSIs con MAYOR variabilidad (std de cosine similarity):")
    df_wsi_sorted = df_wsi.sort_values('cosine_similarity_std', ascending=False)
    print(df_wsi_sorted[['wsi_id', 'cohort', 'cosine_similarity_mean', 'cosine_similarity_std']].head())

    print("\nTop 5 WSIs con MENOR variabilidad:")
    df_wsi_sorted = df_wsi.sort_values('cosine_similarity_std', ascending=True)
    print(df_wsi_sorted[['wsi_id', 'cohort', 'cosine_similarity_mean', 'cosine_similarity_std']].head())

    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 100

INPUT_FILE = "stain_analysis/stain_robustness_metrics.csv"
OUTPUT_DIR = "stain_analysis/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CARGAR DATOS
# ============================================================================
df = pd.read_csv(INPUT_FILE)
print(f"📊 Datos cargados: {len(df):,} patches de {df['wsi_id'].nunique()} WSIs")
print(f"   TCGA: {len(df[df['cohort']=='TCGA']):,} patches")
print(f"   CPTAC: {len(df[df['cohort']=='CPTAC']):,} patches")

# ============================================================================
# FIGURA PRINCIPAL: BOX PLOTS (3 métricas)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

metrics_config = [
    ('cosine_similarity', 'Cosine Similarity', (0.90, 1.0)),
    ('euclidean_distance', 'Euclidean Distance', None),
    ('pearson_correlation', 'Pearson Correlation', (0.90, 1.0))
]

colors = {'TCGA': '#3498db', 'CPTAC': '#e74c3c'}

for idx, (metric, title, ylim) in enumerate(metrics_config):
    ax = axes[idx]
    
    data_tcga = df[df['cohort'] == 'TCGA'][metric].values
    data_cptac = df[df['cohort'] == 'CPTAC'][metric].values
    
    # Box plot
    bp = ax.boxplot([data_tcga, data_cptac], 
                     labels=['TCGA-BRCA', 'CPTAC-BRCA'],
                     patch_artist=True, 
                     showmeans=True,
                     showfliers=False,  # No mostrar outliers para claridad
                     meanprops=dict(marker='D', markerfacecolor='green', 
                                   markeredgecolor='black', markersize=5))
    
    # Colorear
    for patch, cohort in zip(bp['boxes'], ['TCGA', 'CPTAC']):
        patch.set_facecolor(colors[cohort])
        patch.set_alpha(0.7)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_ylabel('Value', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if ylim:
        ax.set_ylim(ylim)
    
    # Añadir estadísticas
    for i, (data, cohort) in enumerate([(data_tcga, 'TCGA'), (data_cptac, 'CPTAC')], 1):
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        
        y_text = ax.get_ylim()[1] * 0.98
        ax.text(i, y_text, f'Med: {median:.3f}\nIQR: [{q1:.3f}, {q3:.3f}]', 
               ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='gray', alpha=0.8))

plt.suptitle('Virchow v2 Robustness to Macenko Stain Normalization', 
            fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()

plt.savefig(f'{OUTPUT_DIR}/fig_stain_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig_stain_robustness.pdf', bbox_inches='tight')
print("\n✅ Figura principal guardada")
plt.show()

# ============================================================================
# TABLA PARA EL PAPER (LaTeX)
# ============================================================================
summary_data = []

for cohort in ['TCGA', 'CPTAC']:
    df_cohort = df[df['cohort'] == cohort]
    
    row = {
        'Cohort': f'{cohort}-BRCA',
        'N WSIs': df_cohort['wsi_id'].nunique(),
        'N patches': f"{len(df_cohort):,}",
        'Cosine Sim': f"{df_cohort['cosine_similarity'].median():.4f} ({df_cohort['cosine_similarity'].quantile(0.25):.4f}–{df_cohort['cosine_similarity'].quantile(0.75):.4f})",
        'Euclidean Dist': f"{df_cohort['euclidean_distance'].median():.2f} ({df_cohort['euclidean_distance'].quantile(0.25):.2f}–{df_cohort['euclidean_distance'].quantile(0.75):.2f})",
        'Pearson Corr': f"{df_cohort['pearson_correlation'].median():.4f} ({df_cohort['pearson_correlation'].quantile(0.25):.4f}–{df_cohort['pearson_correlation'].quantile(0.75):.4f})",
    }
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)

# Mostrar
print("\n" + "="*100)
print("TABLA PARA EL PAPER")
print("="*100)
print(df_summary.to_string(index=False))
print("="*100)

# Guardar LaTeX
latex_table = df_summary.to_latex(
    index=False, 
    escape=False,
    column_format='lrrrrr',
    caption='Embedding stability under Macenko stain normalization. Values show median (IQR).',
    label='tab:stain_robustness'
)

with open(f"{OUTPUT_DIR}/table_stain_robustness.tex", 'w') as f:
    f.write(latex_table)

print(f"\n✅ Tabla LaTeX guardada en: {OUTPUT_DIR}/table_stain_robustness.tex")

# ============================================================================
# INTERPRETACIÓN AUTOMÁTICA
# ============================================================================
print("\n" + "="*100)
print("INTERPRETACIÓN")
print("="*100)

COSINE_THRESHOLD = 0.95

for cohort in ['TCGA', 'CPTAC']:
    df_cohort = df[df['cohort'] == cohort]
    
    cos_median = df_cohort['cosine_similarity'].median()
    pct_high = (df_cohort['cosine_similarity'] > COSINE_THRESHOLD).mean() * 100
    
    print(f"\n{cohort}-BRCA:")
    print(f"  • Cosine similarity median: {cos_median:.4f}")
    print(f"  • % patches with cosine > {COSINE_THRESHOLD}: {pct_high:.1f}%")
    
    if cos_median > COSINE_THRESHOLD:
        print(f"  ✅ ROBUST: Virchow v2 embeddings are highly stable under stain normalization")
        print(f"  → Stain variability is NOT a primary driver of domain shift")
    elif cos_median > 0.90:
        print(f"  ⚠️ MODERATE: Some sensitivity to stain normalization detected")
        print(f"  → Stain variability may contribute partially to domain shift")
    else:
        print(f"  ❌ SENSITIVE: Significant embedding changes under stain normalization")
        print(f"  → Stain variability is a major contributor to domain shift")

print("\n" + "="*100)