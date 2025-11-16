import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Configuración
excel_file = 'data/histomorfologico/representative_images_annotation.xlsx'

# Leer ambas bases de datos
df_tcga = pd.read_excel(excel_file, sheet_name='TCGA')
df_cptac = pd.read_excel(excel_file, sheet_name='CPTAC')

# Obtener subtipos PAM50 (excluyendo ER/PR/HER2)
pam50_labels = ['BASAL', 'HER2-enriched', 'LUMINAL-A', 'LUMINAL-B', 'NORMAL-like']

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
print("Calculando matriz de comparación CPTAC vs TCGA...")
matrix = np.zeros((len(pam50_labels), len(pam50_labels)))

for i, cptac_label in enumerate(pam50_labels):
    for j, tcga_label in enumerate(pam50_labels):
        cramers = calculate_cramers_v(df_cptac, cptac_label, df_tcga, tcga_label, variables)
        matrix[i, j] = cramers
        print(f"CPTAC {cptac_label} vs TCGA {tcga_label}: V={cramers:.3f}")

# Crear DataFrame para mejor visualización
df_matrix = pd.DataFrame(matrix,
                         index=[f'CPTAC\n{label}' for label in pam50_labels],
                         columns=[f'TCGA\n{label}' for label in pam50_labels])

# Crear visualización
fig, ax = plt.subplots(figsize=(12, 10))

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
ax.set_xlabel('TCGA PAM50 Subtypes', fontsize=12, fontweight='bold')
ax.set_ylabel('CPTAC PAM50 Subtypes', fontsize=12, fontweight='bold')
ax.set_title('Cross-Database Morphological Similarity Matrix\nCPTAC vs TCGA PAM50 Subtypes',
             fontsize=14, fontweight='bold', pad=20)

# Rotar etiquetas para mejor legibilidad
plt.xticks(rotation=0, ha='center')
plt.yticks(rotation=0)

# Ajustar layout
plt.tight_layout()

# Guardar figura
output_file = 'results/cross_database_pam50_similarity_matrix.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Matriz guardada en: {output_file}")

# Crear segunda visualización: solo diagonal y casos interesantes
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: Valores diagonales (mismo subtipo entre bases de datos)
diagonal_values = np.diag(matrix)
colors = ['green' if v < 0.1 else 'orange' if v < 0.3 else 'red' for v in diagonal_values]

ax1.barh(pam50_labels, diagonal_values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel("Cramér's V", fontsize=11, fontweight='bold')
ax1.set_title('Same Subtype Similarity\nCPTAC vs TCGA', fontsize=12, fontweight='bold')
ax1.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='V<0.1: Muy similar')
ax1.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='V<0.3: Moderado')
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)

# Añadir valores
for i, (label, val) in enumerate(zip(pam50_labels, diagonal_values)):
    ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Panel 2: Mapa de calor solo con casos más similares (V < 0.3)
similar_matrix = matrix.copy()
similar_matrix[similar_matrix >= 0.3] = np.nan

sns.heatmap(similar_matrix,
            annot=True,
            fmt='.3f',
            cmap='Greens_r',
            vmin=0,
            vmax=0.3,
            cbar_kws={'label': "Cramér's V"},
            linewidths=0.5,
            linecolor='gray',
            square=True,
            ax=ax2,
            mask=np.isnan(similar_matrix))

ax2.set_xlabel('TCGA PAM50 Subtypes', fontsize=11, fontweight='bold')
ax2.set_ylabel('CPTAC PAM50 Subtypes', fontsize=11, fontweight='bold')
ax2.set_title("Similar Comparisons Only\n(Cramér's V < 0.3)", fontsize=12, fontweight='bold')

# Ajustar etiquetas
ax2.set_xticklabels(pam50_labels, rotation=45, ha='right')
ax2.set_yticklabels(pam50_labels, rotation=0)

plt.tight_layout()

# Guardar segunda figura
output_file2 = 'results/cross_database_pam50_analysis.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✓ Análisis detallado guardado en: {output_file2}")

# Generar reporte en texto
print("\n" + "="*80)
print("REPORTE DE SIMILITUD CROSS-DATABASE")
print("="*80)

print("\n1. REPRODUCIBILIDAD (mismo subtipo entre bases de datos):")
print("-" * 80)
for i, label in enumerate(pam50_labels):
    val = diagonal_values[i]
    if val < 0.1:
        interpretation = "✓ MUY SIMILAR (idéntico)"
    elif val < 0.3:
        interpretation = "≈ MODERADAMENTE SIMILAR"
    else:
        interpretation = "✗ DIFERENTE"
    print(f"  {label:15s}: V={val:.3f}  {interpretation}")

print("\n2. COMPARACIONES CROSS-TYPE MÁS SIMILARES:")
print("-" * 80)
# Encontrar pares similares que no sean la diagonal
similar_pairs = []
for i, cptac_label in enumerate(pam50_labels):
    for j, tcga_label in enumerate(pam50_labels):
        if i != j and matrix[i, j] < 0.3:  # No diagonal y similar
            similar_pairs.append((cptac_label, tcga_label, matrix[i, j]))

similar_pairs.sort(key=lambda x: x[2])
for cptac_l, tcga_l, val in similar_pairs[:10]:  # Top 10
    print(f"  CPTAC {cptac_l:15s} ≈ TCGA {tcga_l:15s}  V={val:.3f}")

print("\n3. RESUMEN GLOBAL:")
print("-" * 80)
avg_diagonal = np.mean(diagonal_values)
print(f"  Cramér's V promedio (diagonal): {avg_diagonal:.3f}")
print(f"  Subtipos con V<0.1 (muy similares): {np.sum(diagonal_values < 0.1)}/{len(pam50_labels)}")
print(f"  Subtipos con V<0.3 (moderados): {np.sum(diagonal_values < 0.3)}/{len(pam50_labels)}")

# plt.show()  # Comentado para ejecución no-interactiva

print("\n✓ Análisis completado!")