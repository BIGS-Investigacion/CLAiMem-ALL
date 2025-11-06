import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CARGAR DATOS PROCESADOS
# ============================================================================
OUTPUT_DIR = "models/separability_analysis"

# Cargar embeddings y labels de PAM50
data = np.load(f"{OUTPUT_DIR}/embeddings_pam50.npz")
X_tcga = data['X_tcga']
y_tcga = data['y_tcga']
X_cptac = data['X_cptac']
y_cptac = data['y_cptac']

PAM50_CLASSES = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
PAM50_COLORS = {
    'LumA': '#3498db',
    'LumB': '#e74c3c',
    'Her2': '#2ecc71',
    'Basal': '#f39c12',
    'Normal': '#9b59b6'
}

# ============================================================================
# ANÁLISIS 1: DISTANCIAS INTRA-CLASS VS INTER-CLASS
# ============================================================================
def analyze_distances_per_class(X, y, cohort_name):
    """
    Analiza distancias intra-class vs inter-class por cada clase
    """
    print(f"\n{'='*80}")
    print(f"{cohort_name} - DISTANCE ANALYSIS PER CLASS")
    print(f"{'='*80}\n")
    
    results = []
    
    # Calcular matriz de distancias completa
    dist_matrix = euclidean_distances(X, X)
    
    for class_name in PAM50_CLASSES:
        class_mask = y == class_name
        n_samples = class_mask.sum()
        
        if n_samples < 2:
            print(f"⚠️ {class_name}: Only {n_samples} sample(s), skipping")
            continue
        
        # Índices de esta clase
        class_indices = np.where(class_mask)[0]
        
        # Distancias intra-class (dentro de la misma clase)
        intra_distances = []
        for i in class_indices:
            for j in class_indices:
                if i < j:  # Evitar duplicados y auto-distancias
                    intra_distances.append(dist_matrix[i, j])
        
        # Distancias inter-class (a otras clases)
        inter_distances = []
        for i in class_indices:
            for j in range(len(y)):
                if not class_mask[j]:  # j es de otra clase
                    inter_distances.append(dist_matrix[i, j])
        
        intra_mean = np.mean(intra_distances)
        intra_std = np.std(intra_distances)
        inter_mean = np.mean(inter_distances)
        inter_std = np.std(inter_distances)
        
        # Ratio: queremos inter >> intra para buena separación
        separation_ratio = inter_mean / intra_mean if intra_mean > 0 else np.nan
        
        results.append({
            'Class': class_name,
            'N': n_samples,
            'Intra_mean': intra_mean,
            'Intra_std': intra_std,
            'Inter_mean': inter_mean,
            'Inter_std': inter_std,
            'Ratio': separation_ratio
        })
        
        print(f"{class_name:<10} (n={n_samples:>4})")
        print(f"  Intra-class distance: {intra_mean:.3f} ± {intra_std:.3f}")
        print(f"  Inter-class distance: {inter_mean:.3f} ± {inter_std:.3f}")
        print(f"  Separation ratio:     {separation_ratio:.3f}")
        print(f"  {'✅ Well separated' if separation_ratio > 1.2 else '❌ Poor separation'}")
        print()
    
    return pd.DataFrame(results)

print("="*80)
print("DETAILED PAM50 ANALYSIS")
print("="*80)

df_tcga = analyze_distances_per_class(X_tcga, y_tcga, "TCGA-BRCA")
df_cptac = analyze_distances_per_class(X_cptac, y_cptac, "CPTAC-BRCA")

# ============================================================================
# ANÁLISIS 2: MATRICES DE CONFUSIÓN POR NEAREST NEIGHBOR
# ============================================================================
def compute_nn_confusion_matrix(X, y, classes):
    """
    Para cada WSI, encuentra su nearest neighbor y ve si es de la misma clase
    """
    from sklearn.metrics.pairwise import euclidean_distances
    
    # Calcular distancias
    dist_matrix = euclidean_distances(X, X)
    
    # Para cada muestra, ignorar distancia a sí misma
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Encontrar nearest neighbor
    nn_indices = np.argmin(dist_matrix, axis=1)
    
    # Crear matriz de confusión: "mi clase" vs "clase del NN"
    confusion = np.zeros((len(classes), len(classes)))
    
    for i, true_class in enumerate(y):
        nn_class = y[nn_indices[i]]
        true_idx = classes.index(true_class)
        nn_idx = classes.index(nn_class)
        confusion[true_idx, nn_idx] += 1
    
    return confusion

print("\n" + "="*80)
print("NEAREST NEIGHBOR CONFUSION MATRICES")
print("="*80)

# TCGA
print("\nTCGA-BRCA:")
conf_tcga = compute_nn_confusion_matrix(X_tcga, y_tcga, PAM50_CLASSES)
df_conf_tcga = pd.DataFrame(
    conf_tcga, 
    index=PAM50_CLASSES, 
    columns=PAM50_CLASSES
)
print("\nNearest neighbor is of class (columns):")
print(df_conf_tcga.to_string())

# Calcular % en diagonal (correctos)
correct_tcga = np.diag(conf_tcga).sum()
total_tcga = conf_tcga.sum()
pct_correct_tcga = (correct_tcga / total_tcga) * 100
print(f"\nSamples with NN of same class: {correct_tcga:.0f}/{total_tcga:.0f} ({pct_correct_tcga:.1f}%)")

# CPTAC
print("\n" + "-"*80)
print("CPTAC-BRCA:")
conf_cptac = compute_nn_confusion_matrix(X_cptac, y_cptac, PAM50_CLASSES)
df_conf_cptac = pd.DataFrame(
    conf_cptac, 
    index=PAM50_CLASSES, 
    columns=PAM50_CLASSES
)
print("\nNearest neighbor is of class (columns):")
print(df_conf_cptac.to_string())

correct_cptac = np.diag(conf_cptac).sum()
total_cptac = conf_cptac.sum()
pct_correct_cptac = (correct_cptac / total_cptac) * 100
print(f"\nSamples with NN of same class: {correct_cptac:.0f}/{total_cptac:.0f} ({pct_correct_cptac:.1f}%)")

# ============================================================================
# VISUALIZACIÓN: HEATMAPS DE CONFUSIÓN
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Normalizar por filas (% de cada clase verdadera)
conf_tcga_norm = conf_tcga / conf_tcga.sum(axis=1, keepdims=True) * 100
conf_cptac_norm = conf_cptac / conf_cptac.sum(axis=1, keepdims=True) * 100

# TCGA
sns.heatmap(
    conf_tcga_norm, 
    annot=True, 
    fmt='.1f',
    cmap='YlOrRd',
    xticklabels=PAM50_CLASSES,
    yticklabels=PAM50_CLASSES,
    cbar_kws={'label': '% of samples'},
    vmin=0,
    vmax=100,
    ax=axes[0]
)
axes[0].set_title(f'TCGA-BRCA\nNearest Neighbor Confusion\n({pct_correct_tcga:.1f}% same class)', 
                  fontweight='bold')
axes[0].set_xlabel('NN Class')
axes[0].set_ylabel('True Class')

# CPTAC
sns.heatmap(
    conf_cptac_norm,
    annot=True,
    fmt='.1f',
    cmap='YlOrRd',
    xticklabels=PAM50_CLASSES,
    yticklabels=PAM50_CLASSES,
    cbar_kws={'label': '% of samples'},
    vmin=0,
    vmax=100,
    ax=axes[1]
)
axes[1].set_title(f'CPTAC-BRCA\nNearest Neighbor Confusion\n({pct_correct_cptac:.1f}% same class)',
                  fontweight='bold')
axes[1].set_xlabel('NN Class')
axes[1].set_ylabel('True Class')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pam50_nn_confusion.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/pam50_nn_confusion.pdf', bbox_inches='tight')
print(f"\n✅ Saved: pam50_nn_confusion.png/pdf")
plt.close()

# ============================================================================
# ANÁLISIS 3: PAIRWISE CLASS OVERLAP
# ============================================================================
def compute_class_overlap_matrix(X, y, classes):
    """
    Para cada par de clases, calcula qué % de muestras de clase A
    tienen su NN en clase B
    """
    overlap = np.zeros((len(classes), len(classes)))
    
    dist_matrix = euclidean_distances(X, X)
    np.fill_diagonal(dist_matrix, np.inf)
    
    for i, class_a in enumerate(classes):
        mask_a = y == class_a
        indices_a = np.where(mask_a)[0]
        
        for j, class_b in enumerate(classes):
            if i == j:
                continue
            
            mask_b = y == class_b
            
            # Para cada muestra de clase A, ¿su NN es de clase B?
            count = 0
            for idx_a in indices_a:
                # Distancias de esta muestra a todas las de clase B
                dists_to_b = dist_matrix[idx_a, mask_b]
                # Distancia mínima a clase B
                min_dist_to_b = dists_to_b.min() if len(dists_to_b) > 0 else np.inf
                
                # Distancia al NN global (cualquier clase)
                min_dist_global = dist_matrix[idx_a].min()
                
                # ¿El NN es de clase B?
                if min_dist_to_b == min_dist_global:
                    count += 1
            
            overlap[i, j] = (count / len(indices_a) * 100) if len(indices_a) > 0 else 0
    
    return overlap

print("\n" + "="*80)
print("CLASS OVERLAP ANALYSIS")
print("="*80)
print("(% of class A samples whose nearest neighbor is in class B)")

print("\nTCGA-BRCA:")
overlap_tcga = compute_class_overlap_matrix(X_tcga, y_tcga, PAM50_CLASSES)
df_overlap_tcga = pd.DataFrame(
    overlap_tcga,
    index=PAM50_CLASSES,
    columns=PAM50_CLASSES
)
print(df_overlap_tcga.to_string())

print("\nCPTAC-BRCA:")
overlap_cptac = compute_class_overlap_matrix(X_cptac, y_cptac, PAM50_CLASSES)
df_overlap_cptac = pd.DataFrame(
    overlap_cptac,
    index=PAM50_CLASSES,
    columns=PAM50_CLASSES
)
print(df_overlap_cptac.to_string())

# Identificar pares más problemáticos
print("\n" + "="*80)
print("MOST CONFUSED CLASS PAIRS")
print("="*80)

def get_top_confusions(overlap_matrix, classes, top_n=5):
    """Encuentra los pares de clases más confundidos"""
    confusions = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j:
                confusions.append({
                    'Class_A': classes[i],
                    'Class_B': classes[j],
                    'Overlap_pct': overlap_matrix[i, j]
                })
    
    df = pd.DataFrame(confusions)
    df = df.sort_values('Overlap_pct', ascending=False)
    return df.head(top_n)

print("\nTCGA-BRCA:")
print(get_top_confusions(overlap_tcga, PAM50_CLASSES).to_string(index=False))

print("\nCPTAC-BRCA:")
print(get_top_confusions(overlap_cptac, PAM50_CLASSES).to_string(index=False))

# ============================================================================
# RESUMEN COMPARATIVO
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: TCGA VS CPTAC BY CLASS")
print("="*80)

summary = pd.DataFrame({
    'Class': PAM50_CLASSES,
    'TCGA_n': [np.sum(y_tcga == c) for c in PAM50_CLASSES],
    'CPTAC_n': [np.sum(y_cptac == c) for c in PAM50_CLASSES],
    'TCGA_sil': df_tcga['Intra_mean'].values / df_tcga['Inter_mean'].values,
    'CPTAC_sil': df_cptac['Intra_mean'].values / df_cptac['Inter_mean'].values,
})

print("\n")
print(summary.to_string(index=False))

# Guardar resultados
df_conf_tcga.to_csv(f'{OUTPUT_DIR}/pam50_nn_confusion_tcga.csv')
df_conf_cptac.to_csv(f'{OUTPUT_DIR}/pam50_nn_confusion_cptac.csv')
df_overlap_tcga.to_csv(f'{OUTPUT_DIR}/pam50_class_overlap_tcga.csv')
df_overlap_cptac.to_csv(f'{OUTPUT_DIR}/pam50_class_overlap_cptac.csv')

print(f"\n✅ Detailed analysis saved to {OUTPUT_DIR}/")
print("\n" + "="*80)