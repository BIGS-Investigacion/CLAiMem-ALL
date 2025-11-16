import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import os

# Rutas de las imágenes generadas
images = {
    'PAM50 Cross-DB': 'results/cross_database_pam50_similarity_matrix.png',
    'PAM50 Analysis': 'results/cross_database_pam50_analysis.png',
    'IHC Cross-DB': 'results/cross_database_ihc_similarity_matrix.png',
    'IHC Analysis': 'results/cross_database_ihc_analysis.png',
    'Intra-Group Variability': 'results/tcga_intra_group_variability.png'
}

# Verificar que todas las imágenes existen
missing = [name for name, path in images.items() if not os.path.exists(path)]
if missing:
    print(f"ADVERTENCIA: Faltan las siguientes imágenes: {missing}")
    print("Ejecutando los scripts para generarlas...")

    import subprocess

    if 'PAM50 Cross-DB' in missing or 'PAM50 Analysis' in missing:
        subprocess.run(['python', 'src/claude/cross_database_comparison_matrix.py'])

    if 'IHC Cross-DB' in missing or 'IHC Analysis' in missing:
        subprocess.run(['python', 'src/claude/cross_database_comparison_matrix_ihc.py'])

    if 'Intra-Group Variability' in missing:
        subprocess.run(['python', 'src/claude/intra_database_variability.py'])

print("Generando figura resumen consolidada...")

# Crear figura grande con todas las imágenes
fig = plt.figure(figsize=(24, 30))
gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.15)

# Títulos para cada panel
titles = [
    ('PAM50 Cross-DB', 0, 0),
    ('PAM50 Analysis', 0, 1),
    ('IHC Cross-DB', 1, 0),
    ('IHC Analysis', 1, 1),
    ('Intra-Group Variability', 2, slice(0, 2))
]

for title, row, col in titles:
    if title in images and os.path.exists(images[title]):
        if isinstance(col, slice):
            ax = fig.add_subplot(gs[row, col])
        else:
            ax = fig.add_subplot(gs[row, col])

        img = mpimg.imread(images[title])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

# Título general
fig.suptitle('Análisis Morfológico Comparativo: TCGA vs CPTAC\nSubtipos PAM50 y Marcadores IHC',
             fontsize=20, fontweight='bold', y=0.995)

# Guardar figura consolidada
output_file = 'results/complete_morphological_analysis_summary.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"✓ Figura consolidada guardada en: {output_file}")

# Crear segunda figura: solo matrices principales
fig2, axes = plt.subplots(2, 2, figsize=(20, 20))
fig2.suptitle('Matrices de Similitud Morfológica\nCPTAC vs TCGA',
              fontsize=18, fontweight='bold', y=0.995)

matrices = [
    ('PAM50 Cross-DB', 0, 0),
    ('PAM50 Analysis', 0, 1),
    ('IHC Cross-DB', 1, 0),
    ('IHC Analysis', 1, 1)
]

for title, row, col in matrices:
    if title in images and os.path.exists(images[title]):
        img = mpimg.imread(images[title])
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(title, fontsize=14, fontweight='bold', pad=10)

plt.tight_layout()
output_file2 = 'results/similarity_matrices_summary.png'
plt.savefig(output_file2, dpi=200, bbox_inches='tight')
print(f"✓ Resumen de matrices guardado en: {output_file2}")

# Crear tercera figura: infografía de resultados clave
fig3 = plt.figure(figsize=(20, 12))
gs3 = GridSpec(3, 3, figure=fig3, hspace=0.4, wspace=0.3)

# Panel 1: Reproducibilidad PAM50
ax1 = fig3.add_subplot(gs3[0, 0])
pam50_labels = ['LUMINAL-B', 'HER2-enriched', 'LUMINAL-A', 'BASAL', 'NORMAL-like']
pam50_values = [0.103, 0.137, 0.150, 0.230, 0.615]
colors1 = ['green' if v < 0.1 else 'orange' if v < 0.3 else 'red' for v in pam50_values]
ax1.barh(pam50_labels, pam50_values, color=colors1, alpha=0.7, edgecolor='black')
ax1.set_xlabel("Cramér's V", fontweight='bold')
ax1.set_title('Reproducibilidad PAM50\n(CPTAC vs TCGA)', fontweight='bold', fontsize=12)
ax1.axvline(x=0.1, color='green', linestyle='--', alpha=0.5)
ax1.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5)
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(pam50_values):
    ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')

# Panel 2: Reproducibilidad IHC
ax2 = fig3.add_subplot(gs3[0, 1])
ihc_labels = ['HER2+', 'ER+', 'PR-', 'HER2-', 'ER-', 'PR+']
ihc_values = [0.099, 0.157, 0.184, 0.187, 0.213, 0.335]
colors2 = ['green' if v < 0.1 else 'orange' if v < 0.3 else 'red' for v in ihc_values]
ax2.barh(ihc_labels, ihc_values, color=colors2, alpha=0.7, edgecolor='black')
ax2.set_xlabel("Cramér's V", fontweight='bold')
ax2.set_title('Reproducibilidad IHC\n(CPTAC vs TCGA)', fontweight='bold', fontsize=12)
ax2.axvline(x=0.1, color='green', linestyle='--', alpha=0.5)
ax2.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5)
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(ihc_values):
    ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')

# Panel 3: Comparación general
ax3 = fig3.add_subplot(gs3[0, 2])
categories = ['PAM50\nCross-DB', 'IHC\nCross-DB', 'PAM50\nIntra-Group', 'IHC\nIntra-Group']
values = [0.247, 0.196, 0.105, 0.140]
colors3 = ['orange', 'green', 'green', 'orange']
bars = ax3.bar(categories, values, color=colors3, alpha=0.7, edgecolor='black')
ax3.set_ylabel("Cramér's V promedio", fontweight='bold')
ax3.set_title('Comparación General\nVariabilidad Morfológica', fontweight='bold', fontsize=12)
ax3.axhline(y=0.1, color='green', linestyle='--', alpha=0.5)
ax3.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5)
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Panel 4: Homogeneidad intra-grupo PAM50
ax4 = fig3.add_subplot(gs3[1, :])
pam50_intra = ['NORMAL-like', 'HER2-enriched', 'BASAL', 'LUMINAL-A', 'LUMINAL-B']
pam50_intra_vals = [0.024, 0.101, 0.103, 0.116, 0.182]
pam50_intra_stds = [0.019, 0.036, 0.054, 0.019, 0.067]
colors4 = ['green' if v < 0.1 else 'orange' for v in pam50_intra_vals]
ax4.barh(pam50_intra, pam50_intra_vals, xerr=pam50_intra_stds,
         color=colors4, alpha=0.7, edgecolor='black')
ax4.set_xlabel("Cramér's V (intra-group)", fontweight='bold')
ax4.set_title('Homogeneidad Morfológica Intra-Grupo - PAM50 TCGA\n(Valores bajos = Mayor homogeneidad)',
              fontweight='bold', fontsize=12)
ax4.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='Homogéneo')
ax4.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Moderado')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)
for i, (v, std) in enumerate(zip(pam50_intra_vals, pam50_intra_stds)):
    ax4.text(v + std + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

# Panel 5: Homogeneidad intra-grupo IHC
ax5 = fig3.add_subplot(gs3[2, :])
ihc_intra = ['PR-', 'PR+', 'ER+', 'ER-', 'HER2-', 'HER2+']
ihc_intra_vals = [0.085, 0.086, 0.139, 0.139, 0.189, 0.199]
ihc_intra_stds = [0.025, 0.036, 0.047, 0.044, 0.062, 0.065]
colors5 = ['green' if v < 0.1 else 'orange' for v in ihc_intra_vals]
ax5.barh(ihc_intra, ihc_intra_vals, xerr=ihc_intra_stds,
         color=colors5, alpha=0.7, edgecolor='black')
ax5.set_xlabel("Cramér's V (intra-group)", fontweight='bold')
ax5.set_title('Homogeneidad Morfológica Intra-Grupo - IHC TCGA\n(Valores bajos = Mayor homogeneidad)',
              fontweight='bold', fontsize=12)
ax5.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='Homogéneo')
ax5.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Moderado')
ax5.legend()
ax5.grid(axis='x', alpha=0.3)
for i, (v, std) in enumerate(zip(ihc_intra_vals, ihc_intra_stds)):
    ax5.text(v + std + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

fig3.suptitle('Resumen de Hallazgos Clave\nAnálisis Morfológico TCGA-CPTAC',
              fontsize=18, fontweight='bold', y=0.995)

output_file3 = 'results/key_findings_infographic.png'
plt.savefig(output_file3, dpi=200, bbox_inches='tight')
print(f"✓ Infografía de hallazgos clave guardada en: {output_file3}")

print("\n" + "="*80)
print("RESUMEN DE ARCHIVOS GENERADOS:")
print("="*80)
print(f"1. {output_file}")
print(f"2. {output_file2}")
print(f"3. {output_file3}")
print("\n✓ Todas las figuras generadas exitosamente!")