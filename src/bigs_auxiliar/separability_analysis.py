import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from pathlib import Path
from tqdm import tqdm
import os

# ============================================================================
# ARGUMENTOS DE LÍNEA DE COMANDOS
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Feature space separability analysis across cohorts'
    )
    
    # Directorios H5
    parser.add_argument(
        '--tcga_h5_dir',
        type=str,
        required=True,
        help='Path to TCGA H5 files directory'
    )
    parser.add_argument(
        '--cptac_h5_dir',
        type=str,
        required=True,
        help='Path to CPTAC H5 files directory'
    )

    
    return parser.parse_args()

# ============================================================================
# CARGAR CONFIGURACIÓN
# ============================================================================
args = parse_args()

TCGA_H5_DIR = args.tcga_h5_dir
CPTAC_H5_DIR = args.cptac_h5_dir


# CSVs por tarea
TCGA_LABELS = {
    'PAM50': "data/dataset_csv/tcga-subtype_pam50.csv",
    'ER': "data/dataset_csv/tcga-er.csv",
    'PR': "data/dataset_csv/tcga-pr.csv",
    'HER2': "data/dataset_csv/tcga-erbb2.csv"
}

CPTAC_LABELS = {
    'PAM50': "data/dataset_csv/cptac-subtype_pam50.csv",
    'ER': "data/dataset_csv/cptac-er.csv",
    'PR': "data/dataset_csv/cptac-pr.csv",
    'HER2': "data/dataset_csv/cptac-erbb2.csv"
}

OUTPUT_DIR = "models/separability_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de tareas
TASKS = {
    'PAM50': {
        'classes': ['LumA', 'LumB', 'Her2', 'Basal', 'Normal'],
        'label_mapping': {  # Mapeo de labels en CSV a nombres estándar
            'luma': 'LumA',
            'lumb': 'LumB',
            'her2': 'Her2',
            'basal': 'Basal',
            'normal': 'Normal'
        },
        'colors': {
            'LumA': '#3498db',
            'LumB': '#e74c3c',
            'Her2': '#2ecc71',
            'Basal': '#f39c12',
            'Normal': '#9b59b6'
        }
    },
    'ER': {
        'classes': ['Negative', 'Positive'],
        'label_mapping': {
            'negative': 'Negative',
            'positive': 'Positive',
            '0': 'Negative',
            '1': 'Positive'
        },
        'colors': {
            'Negative': '#e74c3c',
            'Positive': '#3498db'
        }
    },
    'PR': {
        'classes': ['Negative', 'Positive'],
        'label_mapping': {
            'negative': 'Negative',
            'positive': 'Positive',
            '0': 'Negative',
            '1': 'Positive'
        },
        'colors': {
            'Negative': '#e74c3c',
            'Positive': '#3498db'
        }
    },
    'HER2': {
        'classes': ['Negative', 'Positive'],
        'label_mapping': {
            'negative': 'Negative',
            'positive': 'Positive',
            '0': 'Negative',
            '1': 'Positive'
        },
        'colors': {
            'Negative': '#e74c3c',
            'Positive': '#3498db'
        }
    }
}

# ============================================================================
# FUNCIONES
# ============================================================================
def load_labels(csv_path, label_mapping):
    """
    Carga labels desde CSV y mapea a nombres estándar
    
    Returns:
        DataFrame con columns: slide_id, label (mapped)
    """
    df = pd.read_csv(csv_path)
    
    # Verificar columnas
    if 'slide_id' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"CSV must have 'slide_id' and 'label' columns. Found: {df.columns}")
    
    # Mapear labels
    df['label'] = df['label'].astype(str).str.lower()  # Normalizar a minúsculas
    df['label_mapped'] = df['label'].map(label_mapping)
    
    # Filtrar labels sin mapeo
    unmapped = df[df['label_mapped'].isna()]['label'].unique()
    if len(unmapped) > 0:
        print(f"⚠️ Unmapped labels found: {unmapped}")
    
    df = df.dropna(subset=['label_mapped'])
    
    return df[['slide_id', 'label_mapped']].rename(columns={'label_mapped': 'label'})

def extract_wsi_id_from_slide_id(slide_id):
    """
    Extrae el wsi_id del slide_id para hacer match con archivos .h5
    
    Ejemplo: TCGA-3C-AAAU-01A-01-TS1.2F52DD63... → TCGA-3C-AAAU-01A-01-TS1_2F52DD63...
    
    Ajusta esta función según tu convención de nombres
    """
    # Si el slide_id ya es el nombre del archivo (sin extensión), devolver tal cual
    # Si tiene formato con punto, convertir a guion bajo
    if '.' in slide_id:
        wsi_id = slide_id.replace('.', '_')
    else:
        wsi_id = slide_id
    
    return wsi_id

def load_h5_embeddings(h5_path):
    """Carga features desde archivo H5 y verifica integridad"""
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
    
    # Eliminar dimensión extra: (N, 1, 2560) -> (N, 2560)
    features = features.squeeze()
    if features.ndim == 1:  # Si solo hay 1 patch
        features = features.reshape(1, -1)
    
    # Verificar NaN/Inf
    if np.any(np.isnan(features)):
        num_nan = np.sum(np.isnan(features))
        print(f"⚠️ {h5_path.name}: {num_nan} NaN values in features")
        return None
    
    if np.any(np.isinf(features)):
        num_inf = np.sum(np.isinf(features))
        print(f"⚠️ {h5_path.name}: {num_inf} Inf values in features")
        return None
    
    return features

def aggregate_wsi_embeddings(h5_dir, labels_df, cohort_name=""):
    """
    Agrega patch-level embeddings a WSI-level (mean pooling)
    """
    wsi_embeddings = []
    wsi_labels = []
    wsi_ids = []
    failed = 0
    failed_nan = 0
    
    print(f"\nAggregating WSI embeddings for {cohort_name}...")
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=cohort_name):
        slide_id = row['slide_id']
        label = row['label']
        
        try:
            # Convertir slide_id a nombre de archivo
            wsi_id = extract_wsi_id_from_slide_id(slide_id)
            
            # Buscar archivo H5
            h5_file = Path(h5_dir) / f"{wsi_id}.h5"
            
            if not h5_file.exists():
                # Intentar sin modificar
                h5_file = Path(h5_dir) / f"{slide_id}.h5"
                
            if not h5_file.exists():
                failed += 1
                continue
            
            # Cargar embeddings
            features = load_h5_embeddings(h5_file)
            
            # Verificar si hubo error (NaN/Inf)
            if features is None:
                failed_nan += 1
                continue
            
            # Agregar embeddings
            wsi_embedding = features.mean(axis=0)  # [embedding_dim]
            
            # Verificar que el agregado no tenga NaN
            if np.any(np.isnan(wsi_embedding)) or np.any(np.isinf(wsi_embedding)):
                print(f"⚠️ {slide_id}: NaN/Inf after aggregation")
                failed_nan += 1
                continue
            
            wsi_embeddings.append(wsi_embedding)
            wsi_labels.append(label)
            wsi_ids.append(slide_id)
            
        except Exception as e:
            print(f"\n❌ Error processing {slide_id}: {e}")
            failed += 1
            continue
    
    if len(wsi_embeddings) == 0:
        raise ValueError(f"No WSIs processed for {cohort_name}!")
    
    wsi_embeddings = np.vstack(wsi_embeddings)
    wsi_labels = np.array(wsi_labels)
    
    print(f"✅ {cohort_name}: {len(wsi_embeddings)} WSIs aggregated")
    if failed > 0:
        print(f"⚠️ Failed (file not found): {failed} WSIs")
    if failed_nan > 0:
        print(f"⚠️ Failed (NaN/Inf): {failed_nan} WSIs")
    
    # Verificación final
    print(f"Final check: NaN={np.any(np.isnan(wsi_embeddings))}, Inf={np.any(np.isinf(wsi_embeddings))}")
    
    return wsi_embeddings, wsi_labels, wsi_ids

def compute_silhouette_for_task(X, labels, task_name):
    """Calcula silhouette score para una tarea específica"""
    # Verificar que hay al menos 2 clases
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"⚠️ {task_name}: Only {len(unique_labels)} class(es), cannot compute silhouette")
        return None, None
    
    # Global silhouette
    sil_global = silhouette_score(X, labels, metric='euclidean')
    
    # Per-class silhouette
    sil_per_class = {}
    sample_silhouettes = silhouette_samples(X, labels, metric='euclidean')
    
    for class_name in unique_labels:
        class_mask = labels == class_name
        class_sil = sample_silhouettes[class_mask].mean()
        sil_per_class[class_name] = class_sil
    
    return sil_global, sil_per_class

# ============================================================================
# ANÁLISIS PRINCIPAL
# ============================================================================
print("="*80)
print("FEATURE SPACE SEPARABILITY ANALYSIS")
print("="*80)

results_all = []

for task_name, task_config in TASKS.items():
    print("\n" + "="*80)
    print(f"TASK: {task_name}")
    print("="*80)
    
    classes = task_config['classes']
    label_mapping = task_config['label_mapping']
    colors = task_config['colors']
    
    # Verificar que existen los CSVs
    if task_name not in TCGA_LABELS or task_name not in CPTAC_LABELS:
        print(f"⚠️ CSV not configured for {task_name}, skipping...")
        continue
    
    tcga_csv = TCGA_LABELS[task_name]
    cptac_csv = CPTAC_LABELS[task_name]
    
    if not Path(tcga_csv).exists() or not Path(cptac_csv).exists():
        print(f"⚠️ CSV files not found for {task_name}, skipping...")
        continue
    
    # Cargar labels
    print(f"\nLoading labels...")
    tcga_labels_df = load_labels(tcga_csv, label_mapping)
    cptac_labels_df = load_labels(cptac_csv, label_mapping)
    
    print(f"  TCGA-BRCA:  {len(tcga_labels_df)} WSIs with labels")
    print(f"  CPTAC-BRCA: {len(cptac_labels_df)} WSIs with labels")
    
    # Distribución de clases
    print(f"\nTCGA-BRCA class distribution:")
    print(tcga_labels_df['label'].value_counts().sort_index())
    
    print(f"\nCPTAC-BRCA class distribution:")
    print(cptac_labels_df['label'].value_counts().sort_index())
    
    # Agregar embeddings
    X_tcga, y_tcga, ids_tcga = aggregate_wsi_embeddings(
        TCGA_H5_DIR, tcga_labels_df, f"TCGA-{task_name}"
    )
    
    X_cptac, y_cptac, ids_cptac = aggregate_wsi_embeddings(
        CPTAC_H5_DIR, cptac_labels_df, f"CPTAC-{task_name}"
    )
    
    # Guardar embeddings y metadata
    np.savez_compressed(
        f"{OUTPUT_DIR}/embeddings_{task_name.lower()}.npz",
        X_tcga=X_tcga, y_tcga=y_tcga,
        X_cptac=X_cptac, y_cptac=y_cptac
    )
    
    pd.DataFrame({'slide_id': ids_tcga, 'label': y_tcga}).to_csv(
        f"{OUTPUT_DIR}/metadata_tcga_{task_name.lower()}.csv", index=False
    )
    pd.DataFrame({'slide_id': ids_cptac, 'label': y_cptac}).to_csv(
        f"{OUTPUT_DIR}/metadata_cptac_{task_name.lower()}.csv", index=False
    )
    
    # t-SNE
    print(f"\nRunning t-SNE...")
    
    print("  TCGA-BRCA...")
    tsne_tcga = TSNE(
        n_components=2, 
        perplexity=10, 
        learning_rate=200,
        max_iter=1000,  # ← Cambiado de n_iter a max_iter
        random_state=42, 
        verbose=0
    )
    X_tcga_2d = tsne_tcga.fit_transform(X_tcga)
    
    print("  CPTAC-BRCA...")
    tsne_cptac = TSNE(
        n_components=2, 
        perplexity=30, 
        learning_rate=200,
        max_iter=1000,  # ← Cambiado de n_iter a max_iter
        random_state=42, 
        verbose=0
    )
    X_cptac_2d = tsne_cptac.fit_transform(X_cptac)
    
    # Guardar proyecciones
    np.savez_compressed(
        f"{OUTPUT_DIR}/tsne_{task_name.lower()}.npz",
        X_tcga_2d=X_tcga_2d, X_cptac_2d=X_cptac_2d
    )
    
    # Silhouette analysis
    print(f"\nSilhouette analysis...")
    
    sil_tcga, sil_tcga_per_class = compute_silhouette_for_task(
        X_tcga, y_tcga, f"TCGA-{task_name}"
    )
    sil_cptac, sil_cptac_per_class = compute_silhouette_for_task(
        X_cptac, y_cptac, f"CPTAC-{task_name}"
    )
    
    if sil_tcga is not None and sil_cptac is not None:
        print(f"\nGlobal Silhouette Scores:")
        print(f"  TCGA-BRCA:  {sil_tcga:.4f}")
        print(f"  CPTAC-BRCA: {sil_cptac:.4f}")
        print(f"  Difference: {sil_tcga - sil_cptac:+.4f}")
        
        # Guardar resultados
        results_all.append({
            'Task': task_name,
            'Cohort': 'TCGA-BRCA',
            'N_WSIs': len(X_tcga),
            'Silhouette': sil_tcga
        })
        results_all.append({
            'Task': task_name,
            'Cohort': 'CPTAC-BRCA',
            'N_WSIs': len(X_cptac),
            'Silhouette': sil_cptac
        })
        
        # Per-class silhouette
        if sil_tcga_per_class and sil_cptac_per_class:
            print(f"\nPer-class Silhouette:")
            print(f"{'Class':<15} {'TCGA':>8} {'CPTAC':>8} {'Δ':>8}")
            print("-" * 43)
            for class_name in classes:
                if class_name in sil_tcga_per_class and class_name in sil_cptac_per_class:
                    tcga_val = sil_tcga_per_class[class_name]
                    cptac_val = sil_cptac_per_class[class_name]
                    delta = tcga_val - cptac_val
                    print(f"{class_name:<15} {tcga_val:>8.4f} {cptac_val:>8.4f} {delta:>+8.4f}")
    
    # Visualización
    print(f"\nGenerating t-SNE visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for X_2d, y, sil, cohort, ax in [
        (X_tcga_2d, y_tcga, sil_tcga, 'TCGA-BRCA', axes[0]),
        (X_cptac_2d, y_cptac, sil_cptac, 'CPTAC-BRCA', axes[1])
    ]:
        for class_name in classes:
            mask = y == class_name
            if mask.sum() > 0:
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=colors[class_name],
                    label=f'{class_name} (n={mask.sum()})',
                    alpha=0.6,
                    s=30,
                    edgecolors='none'
                )
        
        sil_text = f"{sil:.3f}" if sil is not None else "N/A"
        ax.set_title(f'{cohort} - {task_name}\nSilhouette: {sil_text}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/tsne_{task_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/tsne_{task_name.lower()}.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: tsne_{task_name.lower()}.png/pdf")

# ============================================================================
# RESUMEN DE RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

if len(results_all) > 0:
    results_df = pd.DataFrame(results_all)
    
    # Tabla resumen
    print("\n")
    print(results_df.to_string(index=False))
    
    # Guardar
    results_df.to_csv(f"{OUTPUT_DIR}/silhouette_all_tasks.csv", index=False)
    
    # Tabla pivotada
    pivot_df = results_df.pivot(index='Task', columns='Cohort', values='Silhouette')
    pivot_df['Δ (TCGA - CPTAC)'] = pivot_df['TCGA-BRCA'] - pivot_df['CPTAC-BRCA']
    
    print("\n" + "="*80)
    print("SILHOUETTE COMPARISON BY TASK")
    print("="*80)
    print("\n")
    print(pivot_df.to_string())
    
    # LaTeX table
    latex_table = pivot_df.to_latex(float_format="%.3f")
    with open(f"{OUTPUT_DIR}/silhouette_comparison.tex", 'w') as f:
        f.write(latex_table)
    
    print(f"\n✅ All results saved to {OUTPUT_DIR}/")
    
    # Interpretación
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    for task in TASKS.keys():
        task_df = results_df[results_df['Task'] == task]
        if len(task_df) == 2:
            tcga_sil = task_df[task_df['Cohort'] == 'TCGA-BRCA']['Silhouette'].values[0]
            cptac_sil = task_df[task_df['Cohort'] == 'CPTAC-BRCA']['Silhouette'].values[0]
            diff = tcga_sil - cptac_sil
            
            print(f"\n{task}:")
            print(f"  TCGA: {tcga_sil:.3f}, CPTAC: {cptac_sil:.3f}, Δ: {diff:+.3f}")
            
            if abs(diff) < 0.05:
                print(f"  ✅ Similar separability → Domain shift NOT at feature level")
            elif diff > 0.05:
                print(f"  ❌ TCGA better separated → Domain shift at feature level")
            else:
                print(f"  ⚠️ CPTAC better separated → Unexpected pattern")
else:
    print("\n⚠️ No results to summarize")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)