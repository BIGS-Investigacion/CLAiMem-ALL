import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "models/separability_analysis"

# ============================================================================
# CARGAR DATOS
# ============================================================================
pam50 = np.load(f"{OUTPUT_DIR}/tsne_pam50.npz")
er = np.load(f"{OUTPUT_DIR}/tsne_er.npz")
pr = np.load(f"{OUTPUT_DIR}/tsne_pr.npz")
her2 = np.load(f"{OUTPUT_DIR}/tsne_her2.npz")

pam50_data = np.load(f"{OUTPUT_DIR}/embeddings_pam50.npz")
er_data = np.load(f"{OUTPUT_DIR}/embeddings_er.npz")
pr_data = np.load(f"{OUTPUT_DIR}/embeddings_pr.npz")
her2_data = np.load(f"{OUTPUT_DIR}/embeddings_her2.npz")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
TASKS = {
    'PAM50': {
        'data': pam50,
        'labels': (pam50_data['y_tcga'], pam50_data['y_cptac']),
        'colors': {
            'LumA': '#3498db',
            'LumB': '#e74c3c',
            'Her2': '#2ecc71',
            'Basal': '#f39c12',
            'Normal': '#9b59b6'
        },
        'silhouettes': (-0.030, -0.035)
    },
    'ER': {
        'data': er,
        'labels': (er_data['y_tcga'], er_data['y_cptac']),
        'colors': {
            'Negative': '#e74c3c',
            'Positive': '#3498db'
        },
        'silhouettes': (0.035, 0.042)
    },
    'PR': {
        'data': pr,
        'labels': (pr_data['y_tcga'], pr_data['y_cptac']),
        'colors': {
            'Negative': '#e74c3c',
            'Positive': '#3498db'
        },
        'silhouettes': (0.028, 0.031)
    },
    'HER2': {
        'data': her2,
        'labels': (her2_data['y_tcga'], her2_data['y_cptac']),
        'colors': {
            'Negative': '#e74c3c',
            'Positive': '#3498db'
        },
        'silhouettes': (-0.024, -0.026)
    }
}

# ============================================================================
# CALCULAR LÍMITES GLOBALES POR TAREA
# ============================================================================
def compute_axis_limits(X_tcga_2d, X_cptac_2d, margin=0.05):
    """
    Calcula límites globales para ambos cohorts con margen
    """
    # Combinar ambos datasets
    all_data = np.vstack([X_tcga_2d, X_cptac_2d])
    
    # Límites X
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    x_range = x_max - x_min
    x_lim = [x_min - margin * x_range, x_max + margin * x_range]
    
    # Límites Y
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    y_range = y_max - y_min
    y_lim = [y_min - margin * y_range, y_max + margin * y_range]
    
    return x_lim, y_lim

# Calcular límites para cada tarea
axis_limits = {}
for task_name, task in TASKS.items():
    X_tcga = task['data']['X_tcga_2d']
    X_cptac = task['data']['X_cptac_2d']
    axis_limits[task_name] = compute_axis_limits(X_tcga, X_cptac)

# ============================================================================
# CREAR FIGURA MULTIPANEL
# ============================================================================
fig, axes = plt.subplots(4, 2, figsize=(14, 20))

task_names = ['PAM50', 'ER', 'PR', 'HER2']
task_labels = ['(A)', '(B)', '(C)', '(D)']

for row, (task_name, task_label) in enumerate(zip(task_names, task_labels)):
    task = TASKS[task_name]
    x_lim, y_lim = axis_limits[task_name]
    
    # TCGA (columna izquierda)
    ax_tcga = axes[row, 0]
    X_2d_tcga = task['data']['X_tcga_2d']
    y_tcga = task['labels'][0]
    sil_tcga = task['silhouettes'][0]
    colors = task['colors']
    
    for class_name in sorted(colors.keys()):
        mask = y_tcga == class_name
        if mask.sum() > 0:
            ax_tcga.scatter(
                X_2d_tcga[mask, 0], X_2d_tcga[mask, 1],
                c=colors[class_name],
                label=f'{class_name} (n={mask.sum()})',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
    
    ax_tcga.set_xlim(x_lim)
    ax_tcga.set_ylim(y_lim)
    ax_tcga.set_aspect('equal', adjustable='box')  # Aspecto cuadrado
    
    ax_tcga.set_title(
        f'{task_label} TCGA-BRCA - {task_name}\nSilhouette: {sil_tcga:.3f}',
        fontsize=12,
        fontweight='bold'
    )
    ax_tcga.set_xlabel('t-SNE 1', fontsize=10)
    ax_tcga.set_ylabel('t-SNE 2', fontsize=10)
    ax_tcga.legend(loc='best', frameon=True, fontsize=8)
    ax_tcga.grid(True, alpha=0.3)
    
    # CPTAC (columna derecha)
    ax_cptac = axes[row, 1]
    X_2d_cptac = task['data']['X_cptac_2d']
    y_cptac = task['labels'][1]
    sil_cptac = task['silhouettes'][1]
    
    for class_name in sorted(colors.keys()):
        mask = y_cptac == class_name
        if mask.sum() > 0:
            ax_cptac.scatter(
                X_2d_cptac[mask, 0], X_2d_cptac[mask, 1],
                c=colors[class_name],
                label=f'{class_name} (n={mask.sum()})',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
    
    ax_cptac.set_xlim(x_lim)  # MISMO LÍMITE QUE TCGA
    ax_cptac.set_ylim(y_lim)  # MISMO LÍMITE QUE TCGA
    ax_cptac.set_aspect('equal', adjustable='box')  # Aspecto cuadrado
    
    ax_cptac.set_title(
        f'{task_label} CPTAC-BRCA - {task_name}\nSilhouette: {sil_cptac:.3f}',
        fontsize=12,
        fontweight='bold'
    )
    ax_cptac.set_xlabel('t-SNE 1', fontsize=10)
    ax_cptac.set_ylabel('t-SNE 2', fontsize=10)
    ax_cptac.legend(loc='best', frameon=True, fontsize=8)
    ax_cptac.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tsne_all_tasks.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/tsne_all_tasks.pdf', bbox_inches='tight')
print(f"✅ Combined t-SNE visualization saved (shared axes)")

# ============================================================================
# IMPRIMIR RANGOS PARA VERIFICACIÓN
# ============================================================================
print("\n" + "="*80)
print("AXIS LIMITS PER TASK (shared between TCGA and CPTAC)")
print("="*80)

for task_name in task_names:
    x_lim, y_lim = axis_limits[task_name]
    print(f"\n{task_name}:")
    print(f"  X: [{x_lim[0]:.2f}, {x_lim[1]:.2f}] (range: {x_lim[1]-x_lim[0]:.2f})")
    print(f"  Y: [{y_lim[0]:.2f}, {y_lim[1]:.2f}] (range: {y_lim[1]-y_lim[0]:.2f})")

plt.close()

# ============================================================================
# VERSIÓN ALTERNATIVA: LADO A LADO (HORIZONTAL)
# ============================================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for col, (task_name, task_label) in enumerate(zip(task_names, task_labels)):
    task = TASKS[task_name]
    x_lim, y_lim = axis_limits[task_name]
    colors = task['colors']
    
    # TCGA (fila superior)
    ax_tcga = axes[0, col]
    X_2d_tcga = task['data']['X_tcga_2d']
    y_tcga = task['labels'][0]
    sil_tcga = task['silhouettes'][0]
    
    for class_name in sorted(colors.keys()):
        mask = y_tcga == class_name
        if mask.sum() > 0:
            ax_tcga.scatter(
                X_2d_tcga[mask, 0], X_2d_tcga[mask, 1],
                c=colors[class_name],
                label=f'{class_name}' if col == 0 else '',  # Legend solo primera columna
                alpha=0.6,
                s=15,
                edgecolors='none'
            )
    
    ax_tcga.set_xlim(x_lim)
    ax_tcga.set_ylim(y_lim)
    ax_tcga.set_aspect('equal', adjustable='box')
    
    ax_tcga.set_title(
        f'{task_label} {task_name}\nSil: {sil_tcga:.3f}',
        fontsize=11,
        fontweight='bold'
    )
    ax_tcga.set_xlabel('t-SNE 1', fontsize=9)
    if col == 0:
        ax_tcga.set_ylabel('TCGA-BRCA\nt-SNE 2', fontsize=9, fontweight='bold')
    ax_tcga.grid(True, alpha=0.3)
    ax_tcga.tick_params(labelsize=8)
    
    if col == 0:
        ax_tcga.legend(loc='upper left', frameon=True, fontsize=7, 
                      bbox_to_anchor=(0, 1), ncol=1)
    
    # CPTAC (fila inferior)
    ax_cptac = axes[1, col]
    X_2d_cptac = task['data']['X_cptac_2d']
    y_cptac = task['labels'][1]
    sil_cptac = task['silhouettes'][1]
    
    for class_name in sorted(colors.keys()):
        mask = y_cptac == class_name
        if mask.sum() > 0:
            ax_cptac.scatter(
                X_2d_cptac[mask, 0], X_2d_cptac[mask, 1],
                c=colors[class_name],
                alpha=0.6,
                s=15,
                edgecolors='none'
            )
    
    ax_cptac.set_xlim(x_lim)
    ax_cptac.set_ylim(y_lim)
    ax_cptac.set_aspect('equal', adjustable='box')
    
    ax_cptac.set_title(f'Sil: {sil_cptac:.3f}', fontsize=11, fontweight='bold')
    ax_cptac.set_xlabel('t-SNE 1', fontsize=9)
    if col == 0:
        ax_cptac.set_ylabel('CPTAC-BRCA\nt-SNE 2', fontsize=9, fontweight='bold')
    ax_cptac.grid(True, alpha=0.3)
    ax_cptac.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tsne_all_tasks_horizontal.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/tsne_all_tasks_horizontal.pdf', bbox_inches='tight')
print(f"✅ Horizontal layout saved (2 rows × 4 columns)")

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE")
print("="*80)
print(f"\nFiles in {OUTPUT_DIR}/:")
print("  - tsne_all_tasks.pdf (4×2 vertical, shared axes)")
print("  - tsne_all_tasks_horizontal.pdf (2×4 horizontal, shared axes)")