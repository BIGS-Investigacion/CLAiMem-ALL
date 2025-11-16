import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
excel_file = 'data/histomorfologico/representative_images_annotation.xlsx'

# Leer TCGA
df_tcga = pd.read_excel(excel_file, sheet_name='TCGA')

# Definir grupos
pam50_labels = ['BASAL', 'HER2-enriched', 'LUMINAL-A', 'LUMINAL-B', 'NORMAL-like']
ihc_labels = ['ER-positive', 'ER-negative', 'PR-positive', 'PR-negative',
              'HER2-positive', 'HER2-negative']

# Variables morfológicas
variables = ['ESTRUCTURA GLANDULAR', 'ATIPIA NUCLEAR',
             'MITOSIS', 'NECROSIS', 'INFILTRADO_LI', 'INFILTRADO_PMN']

# Función para calcular Cramér's V entre dos grupos
def calculate_cramers_v(df, label1, label2, variables):
    """Calcula la MEDIANA del Cramér's V entre dos grupos (medida robusta)"""
    group1 = df[df['ETIQUETA'] == label1]
    group2 = df[df['ETIQUETA'] == label2]

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

print("="*80)
print("MATRIZ DE CONFUSIÓN/SIMILITUD MORFOLÓGICA - TCGA")
print("="*80)

# Crear matriz PAM50
print("\n1. Calculando matriz PAM50...")
pam50_matrix = np.zeros((len(pam50_labels), len(pam50_labels)))

for i, label1 in enumerate(pam50_labels):
    for j, label2 in enumerate(pam50_labels):
        if i == j:
            pam50_matrix[i, j] = 0  # Diagonal = 0 (mismo grupo)
        else:
            cramers = calculate_cramers_v(df_tcga, label1, label2, variables)
            pam50_matrix[i, j] = cramers
            print(f"  {label1:15s} vs {label2:15s}: V={cramers:.3f}")

# Crear matriz IHC
print("\n2. Calculando matriz IHC...")
ihc_matrix = np.zeros((len(ihc_labels), len(ihc_labels)))

for i, label1 in enumerate(ihc_labels):
    for j, label2 in enumerate(ihc_labels):
        if i == j:
            ihc_matrix[i, j] = 0  # Diagonal = 0 (mismo grupo)
        else:
            cramers = calculate_cramers_v(df_tcga, label1, label2, variables)
            ihc_matrix[i, j] = cramers
            print(f"  {label1:15s} vs {label2:15s}: V={cramers:.3f}")

# Crear DataFrames
df_pam50_matrix = pd.DataFrame(pam50_matrix, index=pam50_labels, columns=pam50_labels)
df_ihc_matrix = pd.DataFrame(ihc_matrix, index=ihc_labels, columns=ihc_labels)

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Matriz PAM50
ax1 = axes[0]
mask1 = np.triu(np.ones_like(pam50_matrix, dtype=bool))  # Máscara triangular superior
sns.heatmap(df_pam50_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',  # Rojo=diferente, Verde=similar
            vmin=0,
            vmax=1,
            cbar_kws={'label': "Cramér's V\n(0=idéntico, 1=muy diferente)"},
            linewidths=0.5,
            linecolor='gray',
            square=True,
            ax=ax1,
            mask=mask1)

ax1.set_xlabel('PAM50 Subtypes', fontsize=12, fontweight='bold')
ax1.set_ylabel('PAM50 Subtypes', fontsize=12, fontweight='bold')
ax1.set_title('Matriz de Similitud Morfológica\nSubtipos PAM50 - TCGA\n(Valores bajos = Morfológicamente similares)',
             fontsize=13, fontweight='bold', pad=15)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

# Matriz IHC
ax2 = axes[1]
mask2 = np.triu(np.ones_like(ihc_matrix, dtype=bool))  # Máscara triangular superior
sns.heatmap(df_ihc_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=1,
            cbar_kws={'label': "Cramér's V\n(0=idéntico, 1=muy diferente)"},
            linewidths=0.5,
            linecolor='gray',
            square=True,
            ax=ax2,
            mask=mask2)

ax2.set_xlabel('IHC Markers', fontsize=12, fontweight='bold')
ax2.set_ylabel('IHC Markers', fontsize=12, fontweight='bold')
ax2.set_title('Matriz de Similitud Morfológica\nMarcadores IHC - TCGA\n(Valores bajos = Morfológicamente similares)',
             fontsize=13, fontweight='bold', pad=15)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

plt.tight_layout()

output_file = 'results/tcga_intra_confusion_matrix.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Matriz de confusión guardada en: {output_file}")

# Análisis de pares más similares y más diferentes
print("\n" + "="*80)
print("ANÁLISIS DE SIMILITUDES Y DIFERENCIAS")
print("="*80)

# PAM50 - Pares más similares
print("\n3. SUBTIPOS PAM50 MÁS SIMILARES (riesgo de confusión):")
print("-" * 80)
pam50_pairs = []
for i, label1 in enumerate(pam50_labels):
    for j, label2 in enumerate(pam50_labels):
        if i < j:  # Solo triángulo inferior
            pam50_pairs.append((label1, label2, pam50_matrix[i, j]))

pam50_pairs_sorted = sorted(pam50_pairs, key=lambda x: x[2])
for i, (l1, l2, val) in enumerate(pam50_pairs_sorted[:5]):
    interpretation = "¡ALTO RIESGO!" if val < 0.3 else "RIESGO MODERADO" if val < 0.5 else "BAJO RIESGO"
    print(f"  {i+1}. {l1:15s} ↔ {l2:15s}: V={val:.3f}  [{interpretation}]")

# PAM50 - Pares más diferentes
print("\n4. SUBTIPOS PAM50 MÁS DIFERENTES (fácil discriminación):")
print("-" * 80)
for i, (l1, l2, val) in enumerate(reversed(pam50_pairs_sorted[-5:])):
    print(f"  {i+1}. {l1:15s} ↔ {l2:15s}: V={val:.3f}")

# IHC - Pares más similares
print("\n5. MARCADORES IHC MÁS SIMILARES (riesgo de confusión):")
print("-" * 80)
ihc_pairs = []
for i, label1 in enumerate(ihc_labels):
    for j, label2 in enumerate(ihc_labels):
        if i < j:  # Solo triángulo inferior
            ihc_pairs.append((label1, label2, ihc_matrix[i, j]))

ihc_pairs_sorted = sorted(ihc_pairs, key=lambda x: x[2])
for i, (l1, l2, val) in enumerate(ihc_pairs_sorted[:5]):
    interpretation = "¡ALTO RIESGO!" if val < 0.3 else "RIESGO MODERADO" if val < 0.5 else "BAJO RIESGO"
    print(f"  {i+1}. {l1:15s} ↔ {l2:15s}: V={val:.3f}  [{interpretation}]")

# IHC - Pares más diferentes
print("\n6. MARCADORES IHC MÁS DIFERENTES (fácil discriminación):")
print("-" * 80)
for i, (l1, l2, val) in enumerate(reversed(ihc_pairs_sorted[-5:])):
    print(f"  {i+1}. {l1:15s} ↔ {l2:15s}: V={val:.3f}")

# Crear visualización adicional: Top confusiones
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 12))

# Panel 1: Top 10 pares PAM50 más similares
ax1 = axes2[0]
top_pam50 = pam50_pairs_sorted[:10]
labels_pam50 = [f"{l1} ↔ {l2}" for l1, l2, _ in top_pam50]
values_pam50 = [val for _, _, val in top_pam50]
colors_pam50 = ['red' if v < 0.3 else 'orange' if v < 0.5 else 'green' for v in values_pam50]

ax1.barh(labels_pam50, values_pam50, color=colors_pam50, alpha=0.7, edgecolor='black')
ax1.set_xlabel("Cramér's V", fontsize=11, fontweight='bold')
ax1.set_title('Top 10 Pares PAM50 Más Similares - TCGA\n(Rojo = Alto riesgo de confusión)',
              fontsize=12, fontweight='bold')
ax1.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Alto riesgo')
ax1.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Riesgo moderado')
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

for i, (label, val) in enumerate(zip(labels_pam50, values_pam50)):
    ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

# Panel 2: Top 10 pares IHC más similares
ax2 = axes2[1]
top_ihc = ihc_pairs_sorted[:10]
labels_ihc = [f"{l1} ↔ {l2}" for l1, l2, _ in top_ihc]
values_ihc = [val for _, _, val in top_ihc]
colors_ihc = ['red' if v < 0.3 else 'orange' if v < 0.5 else 'green' for v in values_ihc]

ax2.barh(labels_ihc, values_ihc, color=colors_ihc, alpha=0.7, edgecolor='black')
ax2.set_xlabel("Cramér's V", fontsize=11, fontweight='bold')
ax2.set_title('Top 10 Pares IHC Más Similares - TCGA\n(Rojo = Alto riesgo de confusión)',
              fontsize=12, fontweight='bold')
ax2.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Alto riesgo')
ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Riesgo moderado')
ax2.legend(fontsize=9)
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

for i, (label, val) in enumerate(zip(labels_ihc, values_ihc)):
    ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()

output_file2 = 'results/tcga_top_confusions.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"\n✓ Top confusiones guardadas en: {output_file2}")

# Resumen estadístico
print("\n7. RESUMEN ESTADÍSTICO:")
print("-" * 80)

# Promedio de similitud
avg_pam50 = np.mean([v for _, _, v in pam50_pairs])
avg_ihc = np.mean([v for _, _, v in ihc_pairs])

print(f"  PAM50 - Similitud promedio inter-subtipo: {avg_pam50:.3f}")
print(f"  IHC   - Similitud promedio inter-marcador: {avg_ihc:.3f}")

# Contar pares con riesgo
pam50_high_risk = sum(1 for _, _, v in pam50_pairs if v < 0.3)
pam50_mod_risk = sum(1 for _, _, v in pam50_pairs if 0.3 <= v < 0.5)
ihc_high_risk = sum(1 for _, _, v in ihc_pairs if v < 0.3)
ihc_mod_risk = sum(1 for _, _, v in ihc_pairs if 0.3 <= v < 0.5)

print(f"\n  PAM50 - Pares con ALTO riesgo (V<0.3): {pam50_high_risk}/{len(pam50_pairs)}")
print(f"  PAM50 - Pares con riesgo MODERADO (0.3≤V<0.5): {pam50_mod_risk}/{len(pam50_pairs)}")
print(f"  IHC   - Pares con ALTO riesgo (V<0.3): {ihc_high_risk}/{len(ihc_pairs)}")
print(f"  IHC   - Pares con riesgo MODERADO (0.3≤V<0.5): {ihc_mod_risk}/{len(ihc_pairs)}")

print("\n" + "="*80)
print("INTERPRETACIÓN:")
print("="*80)
print("""
Cramér's V entre grupos diferentes mide el riesgo de confusión morfológica:

- V < 0.3: ALTO RIESGO de confusión - Morfológicamente muy similares
- 0.3 ≤ V < 0.5: RIESGO MODERADO - Algunas similitudes morfológicas
- V ≥ 0.5: BAJO RIESGO - Morfológicamente distintos

Valores bajos indican que dos subtipos/marcadores diferentes son difíciles
de distinguir solo por morfología, lo que justifica el uso de marcadores
moleculares para su discriminación.
""")

print("\n✓ Análisis de matriz de confusión completado!")