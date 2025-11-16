import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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

# Función para calcular Cramér's V entre dos mitades del mismo grupo
def calculate_intra_variability(df, label, variables, n_iterations=10):
    """
    Divide el grupo en dos mitades aleatorias y calcula Cramér's V entre ellas.
    Repite múltiples veces y devuelve el promedio.
    """
    group = df[df['ETIQUETA'] == label]

    if len(group) < 10:  # Necesitamos al menos 10 casos
        return np.nan, np.nan

    all_cramers = []

    for iteration in range(n_iterations):
        # Dividir en dos mitades
        half1, half2 = train_test_split(group, test_size=0.5, random_state=iteration)

        cramers_values = []

        for var in variables:
            try:
                # Crear tabla de contingencia
                combined = pd.concat([
                    half1[[var]].assign(grupo='G1'),
                    half2[[var]].assign(grupo='G2')
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

        if cramers_values:
            all_cramers.append(np.mean(cramers_values))

    if all_cramers:
        return np.mean(all_cramers), np.std(all_cramers)
    else:
        return np.nan, np.nan

# Calcular variabilidad intra-grupo para PAM50
print("="*80)
print("ANÁLISIS DE VARIABILIDAD INTRA-COHORTE - TCGA")
print("="*80)

print("\n1. SUBTIPOS PAM50:")
print("-" * 80)
pam50_variability = []
for label in pam50_labels:
    n_cases = len(df_tcga[df_tcga['ETIQUETA'] == label])
    mean_v, std_v = calculate_intra_variability(df_tcga, label, variables)
    pam50_variability.append((label, mean_v, std_v, n_cases))

    if not np.isnan(mean_v):
        interpretation = "✓ Homogéneo" if mean_v < 0.1 else "≈ Moderado" if mean_v < 0.2 else "✗ Heterogéneo"
        print(f"  {label:15s}: V={mean_v:.3f}±{std_v:.3f} (n={n_cases:2d})  {interpretation}")
    else:
        print(f"  {label:15s}: No suficientes datos (n={n_cases:2d})")

# Calcular variabilidad intra-grupo para IHC
print("\n2. MARCADORES IHC:")
print("-" * 80)
ihc_variability = []
for label in ihc_labels:
    n_cases = len(df_tcga[df_tcga['ETIQUETA'] == label])
    mean_v, std_v = calculate_intra_variability(df_tcga, label, variables)
    ihc_variability.append((label, mean_v, std_v, n_cases))

    if not np.isnan(mean_v):
        interpretation = "✓ Homogéneo" if mean_v < 0.1 else "≈ Moderado" if mean_v < 0.2 else "✗ Heterogéneo"
        print(f"  {label:15s}: V={mean_v:.3f}±{std_v:.3f} (n={n_cases:2d})  {interpretation}")
    else:
        print(f"  {label:15s}: No suficientes datos (n={n_cases:2d})")

# Crear visualización comparativa
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: PAM50 variabilidad intra-grupo
ax1 = axes[0, 0]
labels_pam50 = [x[0] for x in pam50_variability if not np.isnan(x[1])]
means_pam50 = [x[1] for x in pam50_variability if not np.isnan(x[1])]
stds_pam50 = [x[2] for x in pam50_variability if not np.isnan(x[1])]
colors_pam50 = ['green' if v < 0.1 else 'orange' if v < 0.2 else 'red' for v in means_pam50]

ax1.barh(labels_pam50, means_pam50, xerr=stds_pam50, color=colors_pam50, alpha=0.7, edgecolor='black')
ax1.set_xlabel("Cramér's V (intra-group)", fontsize=11, fontweight='bold')
ax1.set_title('PAM50 Intra-Group Variability\n(Lower = More Homogeneous)', fontsize=12, fontweight='bold')
ax1.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='V<0.1: Homogéneo')
ax1.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='V<0.2: Moderado')
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)

for i, (label, val, std) in enumerate(zip(labels_pam50, means_pam50, stds_pam50)):
    ax1.text(val + std + 0.01, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Panel 2: IHC variabilidad intra-grupo
ax2 = axes[0, 1]
labels_ihc = [x[0] for x in ihc_variability if not np.isnan(x[1])]
means_ihc = [x[1] for x in ihc_variability if not np.isnan(x[1])]
stds_ihc = [x[2] for x in ihc_variability if not np.isnan(x[1])]
colors_ihc = ['green' if v < 0.1 else 'orange' if v < 0.2 else 'red' for v in means_ihc]

ax2.barh(labels_ihc, means_ihc, xerr=stds_ihc, color=colors_ihc, alpha=0.7, edgecolor='black')
ax2.set_xlabel("Cramér's V (intra-group)", fontsize=11, fontweight='bold')
ax2.set_title('IHC Markers Intra-Group Variability\n(Lower = More Homogeneous)', fontsize=12, fontweight='bold')
ax2.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='V<0.1: Homogéneo')
ax2.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='V<0.2: Moderado')
ax2.legend(fontsize=9)
ax2.grid(axis='x', alpha=0.3)

for i, (label, val, std) in enumerate(zip(labels_ihc, means_ihc, stds_ihc)):
    ax2.text(val + std + 0.01, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Panel 3: Comparación PAM50 vs IHC
ax3 = axes[1, 0]
avg_pam50 = np.mean(means_pam50)
avg_ihc = np.mean(means_ihc)
categories = ['PAM50\nSubtypes', 'IHC\nMarkers']
avg_values = [avg_pam50, avg_ihc]
colors_avg = ['green' if v < 0.1 else 'orange' if v < 0.2 else 'red' for v in avg_values]

bars = ax3.bar(categories, avg_values, color=colors_avg, alpha=0.7, edgecolor='black', width=0.6)
ax3.set_ylabel("Average Cramér's V", fontsize=11, fontweight='bold')
ax3.set_title('Average Intra-Group Variability\nPAM50 vs IHC', fontsize=12, fontweight='bold')
ax3.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Threshold: Homogéneo')
ax3.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Threshold: Moderado')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, avg_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Panel 4: Distribución de variabilidad
ax4 = axes[1, 1]
all_means = means_pam50 + means_ihc
all_labels = ['PAM50']*len(means_pam50) + ['IHC']*len(means_ihc)
df_plot = pd.DataFrame({'Cramers_V': all_means, 'Type': all_labels})

# Violin plot
parts = ax4.violinplot([means_pam50, means_ihc], positions=[0, 1],
                        showmeans=True, showmedians=True)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['PAM50', 'IHC'])
ax4.set_ylabel("Cramér's V", fontsize=11, fontweight='bold')
ax4.set_title('Distribution of Intra-Group Variability', fontsize=12, fontweight='bold')
ax4.axhline(y=0.1, color='green', linestyle='--', alpha=0.5)
ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Guardar figura
output_file = 'results/tcga_intra_group_variability.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualización guardada en: {output_file}")

# Resumen estadístico
print("\n3. RESUMEN ESTADÍSTICO:")
print("-" * 80)
print(f"  PAM50 - Variabilidad promedio: {avg_pam50:.3f}")
print(f"  IHC   - Variabilidad promedio: {avg_ihc:.3f}")

if avg_pam50 < avg_ihc:
    print(f"\n  → PAM50 es MÁS HOMOGÉNEO que IHC (Δ={avg_ihc-avg_pam50:.3f})")
else:
    print(f"\n  → IHC es MÁS HOMOGÉNEO que PAM50 (Δ={avg_pam50-avg_ihc:.3f})")

# Top 3 más homogéneos
print("\n4. GRUPOS MÁS HOMOGÉNEOS (menor variabilidad):")
print("-" * 80)
all_variability = pam50_variability + ihc_variability
all_variability_sorted = sorted([x for x in all_variability if not np.isnan(x[1])],
                                key=lambda x: x[1])
for i, (label, mean_v, std_v, n) in enumerate(all_variability_sorted[:5]):
    print(f"  {i+1}. {label:15s}: V={mean_v:.3f}±{std_v:.3f} (n={n})")

# Top 3 más heterogéneos
print("\n5. GRUPOS MÁS HETEROGÉNEOS (mayor variabilidad):")
print("-" * 80)
for i, (label, mean_v, std_v, n) in enumerate(reversed(all_variability_sorted[-5:])):
    print(f"  {i+1}. {label:15s}: V={mean_v:.3f}±{std_v:.3f} (n={n})")

print("\n" + "="*80)
print("INTERPRETACIÓN:")
print("="*80)
print("""
Cramér's V intra-grupo mide la variabilidad morfológica DENTRO de cada subtipo.

- V < 0.1: Grupo HOMOGÉNEO - Casos morfológicamente muy similares
- 0.1 ≤ V < 0.2: MODERADA variabilidad - Alguna heterogeneidad morfológica
- V ≥ 0.2: Grupo HETEROGÉNEO - Alta variabilidad morfológica interna

Valores bajos indican que el subtipo tiene un perfil morfológico consistente.
Valores altos sugieren heterogeneidad morfológica dentro del subtipo.
""")

print("\n✓ Análisis de variabilidad intra-cohorte completado!")