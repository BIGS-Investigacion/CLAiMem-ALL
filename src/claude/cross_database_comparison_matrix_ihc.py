import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
excel_file = 'data/histomorfologico/representative_images_annotation.xlsx'

# Leer ambas bases de datos
df_tcga = pd.read_excel(excel_file, sheet_name='TCGA')
df_cptac = pd.read_excel(excel_file, sheet_name='CPTAC')

# Etiquetas IHC
ihc_labels = ['ER-positive', 'ER-negative', 'PR-positive', 'PR-negative',
              'HER2-positive', 'HER2-negative']

# Variables morfológicas
variables = ['ESTRUCTURA GLANDULAR', 'ATIPIA NUCLEAR',
             'MITOSIS', 'NECROSIS', 'INFILTRADO_LI', 'INFILTRADO_PMN']

# Función para calcular Cramér's V promedio entre dos grupos
def calculate_cramers_v(df1, label1, df2, label2, variables):
    """Calcula la MEDIANA del Cramér's V entre dos grupos (medida robusta)"""
    group1 = df1[df1['ETIQUETA'] == label1]
    group2 = df2[df2['ETIQUETA'] == label2]

    if len(group1) == 0 or len(group2) == 0:
        return np.nan

    cramers_values = []

    for var in variables:
        try:
            # Crear tabla de contingencia
            combined = pd.concat([
                group1[[var]].assign(grupo='G1'),
                group2[[var]].assign(grupo='G2')
            ])

            ct = pd.crosstab(combined['grupo'], combined[var])

            # Test chi-cuadrado
            chi2, p_val, dof, expected = stats.chi2_contingency(ct)

            # Cramér's V
            n = ct.sum().sum()
            min_dim = min(ct.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

            cramers_values.append(cramers_v)
        except:
            continue

    return np.median(cramers_values) if cramers_values else np.nan

# Crear matriz de comparación
print("Calculando matriz de comparación CPTAC vs TCGA (IHC markers)...")
matrix = np.zeros((len(ihc_labels), len(ihc_labels)))

for i, cptac_label in enumerate(ihc_labels):
    for j, tcga_label in enumerate(ihc_labels):
        cramers = calculate_cramers_v(df_cptac, cptac_label, df_tcga, tcga_label, variables)
        matrix[i, j] = cramers
        print(f"CPTAC {cptac_label:15s} vs TCGA {tcga_label:15s}: V={cramers:.3f}")

# Crear DataFrame para mejor visualización
df_matrix = pd.DataFrame(matrix,
                         index=[f'CPTAC\n{label}' for label in ihc_labels],
                         columns=[f'TCGA\n{label}' for label in ihc_labels])

# Crear visualización principal
fig, ax = plt.subplots(figsize=(14, 12))

# Heatmap con valores anotados
sns.heatmap(df_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',  # Rojo=diferente, Verde=similar
            vmin=0,
            vmax=1,
            cbar_kws={'label': "Cramér's V\n(0=idéntico, 1=muy diferente)"},
            linewidths=0.5,
            linecolor='gray',
            square=True,
            ax=ax)

# Configuración de etiquetas
ax.set_xlabel('TCGA IHC Status', fontsize=12, fontweight='bold')
ax.set_ylabel('CPTAC IHC Status', fontsize=12, fontweight='bold')
ax.set_title('Cross-Database Morphological Similarity Matrix\nCPTAC vs TCGA IHC Markers (ER/PR/HER2)',
             fontsize=14, fontweight='bold', pad=20)

# Rotar etiquetas para mejor legibilidad
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Ajustar layout
plt.tight_layout()

# Guardar figura
output_file = 'results/cross_database_ihc_similarity_matrix.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Matriz IHC guardada en: {output_file}")

# Crear segunda visualización: análisis por marcador
fig2, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: Valores diagonales (mismo marcador entre bases de datos)
ax1 = axes[0, 0]
diagonal_values = np.diag(matrix)
colors = ['green' if v < 0.1 else 'orange' if v < 0.3 else 'red' for v in diagonal_values]

ax1.barh(ihc_labels, diagonal_values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel("Cramér's V", fontsize=11, fontweight='bold')
ax1.set_title('Same IHC Status Similarity\nCPTAC vs TCGA', fontsize=12, fontweight='bold')
ax1.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='V<0.1: Muy similar')
ax1.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='V<0.3: Moderado')
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)

# Añadir valores
for i, (label, val) in enumerate(zip(ihc_labels, diagonal_values)):
    ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Panel 2: Comparaciones ER
ax2 = axes[0, 1]
er_labels = ['ER-positive', 'ER-negative']
er_indices = [ihc_labels.index(l) for l in er_labels]
er_matrix = matrix[np.ix_(er_indices, er_indices)]

sns.heatmap(er_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=1,
            xticklabels=['TCGA\nER+', 'TCGA\nER-'],
            yticklabels=['CPTAC\nER+', 'CPTAC\nER-'],
            cbar_kws={'label': "Cramér's V"},
            linewidths=1,
            square=True,
            ax=ax2)
ax2.set_title('ER Status Comparison', fontsize=12, fontweight='bold')

# Panel 3: Comparaciones PR
ax3 = axes[1, 0]
pr_labels = ['PR-positive', 'PR-negative']
pr_indices = [ihc_labels.index(l) for l in pr_labels]
pr_matrix = matrix[np.ix_(pr_indices, pr_indices)]

sns.heatmap(pr_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=1,
            xticklabels=['TCGA\nPR+', 'TCGA\nPR-'],
            yticklabels=['CPTAC\nPR+', 'CPTAC\nPR-'],
            cbar_kws={'label': "Cramér's V"},
            linewidths=1,
            square=True,
            ax=ax3)
ax3.set_title('PR Status Comparison', fontsize=12, fontweight='bold')

# Panel 4: Comparaciones HER2
ax4 = axes[1, 1]
her2_labels = ['HER2-positive', 'HER2-negative']
her2_indices = [ihc_labels.index(l) for l in her2_labels]
her2_matrix = matrix[np.ix_(her2_indices, her2_indices)]

sns.heatmap(her2_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=1,
            xticklabels=['TCGA\nHER2+', 'TCGA\nHER2-'],
            yticklabels=['CPTAC\nHER2+', 'CPTAC\nHER2-'],
            cbar_kws={'label': "Cramér's V"},
            linewidths=1,
            square=True,
            ax=ax4)
ax4.set_title('HER2 Status Comparison', fontsize=12, fontweight='bold')

plt.tight_layout()

# Guardar segunda figura
output_file2 = 'results/cross_database_ihc_analysis.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✓ Análisis IHC detallado guardado en: {output_file2}")

# Generar reporte en texto
print("\n" + "="*80)
print("REPORTE DE SIMILITUD CROSS-DATABASE - MARCADORES IHC")
print("="*80)

print("\n1. REPRODUCIBILIDAD (mismo marcador IHC entre bases de datos):")
print("-" * 80)
for i, label in enumerate(ihc_labels):
    val = diagonal_values[i]
    if val < 0.1:
        interpretation = "✓ MUY SIMILAR (idéntico)"
    elif val < 0.3:
        interpretation = "≈ MODERADAMENTE SIMILAR"
    else:
        interpretation = "✗ DIFERENTE"
    print(f"  {label:15s}: V={val:.3f}  {interpretation}")

print("\n2. ANÁLISIS POR MARCADOR:")
print("-" * 80)

# ER
er_pos_pos = matrix[ihc_labels.index('ER-positive'), ihc_labels.index('ER-positive')]
er_neg_neg = matrix[ihc_labels.index('ER-negative'), ihc_labels.index('ER-negative')]
er_pos_neg = matrix[ihc_labels.index('ER-positive'), ihc_labels.index('ER-negative')]
print(f"\n  ER Status:")
print(f"    CPTAC ER+ vs TCGA ER+: V={er_pos_pos:.3f}")
print(f"    CPTAC ER- vs TCGA ER-: V={er_neg_neg:.3f}")
print(f"    CPTAC ER+ vs TCGA ER-: V={er_pos_neg:.3f} (debe ser alto)")

# PR
pr_pos_pos = matrix[ihc_labels.index('PR-positive'), ihc_labels.index('PR-positive')]
pr_neg_neg = matrix[ihc_labels.index('PR-negative'), ihc_labels.index('PR-negative')]
pr_pos_neg = matrix[ihc_labels.index('PR-positive'), ihc_labels.index('PR-negative')]
print(f"\n  PR Status:")
print(f"    CPTAC PR+ vs TCGA PR+: V={pr_pos_pos:.3f}")
print(f"    CPTAC PR- vs TCGA PR-: V={pr_neg_neg:.3f}")
print(f"    CPTAC PR+ vs TCGA PR-: V={pr_pos_neg:.3f} (debe ser alto)")

# HER2
her2_pos_pos = matrix[ihc_labels.index('HER2-positive'), ihc_labels.index('HER2-positive')]
her2_neg_neg = matrix[ihc_labels.index('HER2-negative'), ihc_labels.index('HER2-negative')]
her2_pos_neg = matrix[ihc_labels.index('HER2-positive'), ihc_labels.index('HER2-negative')]
print(f"\n  HER2 Status:")
print(f"    CPTAC HER2+ vs TCGA HER2+: V={her2_pos_pos:.3f}")
print(f"    CPTAC HER2- vs TCGA HER2-: V={her2_neg_neg:.3f}")
print(f"    CPTAC HER2+ vs TCGA HER2-: V={her2_pos_neg:.3f} (debe ser alto)")

print("\n3. COMPARACIONES CROSS-MARKER INTERESANTES:")
print("-" * 80)
# Encontrar pares similares que no sean la diagonal
similar_pairs = []
for i, cptac_label in enumerate(ihc_labels):
    for j, tcga_label in enumerate(ihc_labels):
        if i != j and matrix[i, j] < 0.3:  # No diagonal y similar
            similar_pairs.append((cptac_label, tcga_label, matrix[i, j]))

similar_pairs.sort(key=lambda x: x[2])
if similar_pairs:
    for cptac_l, tcga_l, val in similar_pairs[:10]:  # Top 10
        print(f"  CPTAC {cptac_l:15s} ≈ TCGA {tcga_l:15s}  V={val:.3f}")
else:
    print("  No hay comparaciones cross-marker similares (V<0.3)")

print("\n4. RESUMEN GLOBAL:")
print("-" * 80)
avg_diagonal = np.mean(diagonal_values)
print(f"  Cramér's V promedio (diagonal): {avg_diagonal:.3f}")
print(f"  Marcadores con V<0.1 (muy similares): {np.sum(diagonal_values < 0.1)}/{len(ihc_labels)}")
print(f"  Marcadores con V<0.3 (moderados): {np.sum(diagonal_values < 0.3)}/{len(ihc_labels)}")

# Análisis de discriminación
print("\n5. PODER DISCRIMINANTE DE CADA MARCADOR:")
print("-" * 80)
print("  (¿Cuánto difiere positive vs negative del mismo marcador?)")

markers = [('ER', er_pos_neg), ('PR', pr_pos_neg), ('HER2', her2_pos_neg)]
for marker_name, cross_value in markers:
    if cross_value > 0.5:
        interpretation = "✓ ALTA discriminación"
    elif cross_value > 0.3:
        interpretation = "≈ MODERADA discriminación"
    else:
        interpretation = "✗ BAJA discriminación"
    print(f"  {marker_name:5s}: V(+/-) = {cross_value:.3f}  {interpretation}")

# plt.show()  # Comentado para ejecución no-interactiva

print("\n✓ Análisis IHC completado!")