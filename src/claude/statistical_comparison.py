import pandas as pd
import numpy as np
from scipy import stats
import argparse
import sys

# Configurar argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Comparación estadística entre dos grupos de imágenes histomorfológicas')
parser.add_argument('--database', '-d', type=str,
                    choices=['TCGA', 'CPTAC'],
                    help='Base de datos a analizar (TCGA o CPTAC). Si se especifica, ambas etiquetas se buscan en esta BD.')
parser.add_argument('--database1', '-d1', type=str,
                    choices=['TCGA', 'CPTAC'],
                    help='Base de datos para label1 (permite comparación entre BDs)')
parser.add_argument('--database2', '-d2', type=str,
                    choices=['TCGA', 'CPTAC'],
                    help='Base de datos para label2 (permite comparación entre BDs)')
parser.add_argument('--label1', '-l1', type=str, required=True,
                    help='Primera etiqueta a comparar (ej: HER2-positive, ER-positive, PAM50_Basal)')
parser.add_argument('--label2', '-l2', type=str, required=True,
                    help='Segunda etiqueta a comparar (ej: HER2-negative, ER-negative, PAM50_LumA)')
parser.add_argument('--excel', '-e', type=str,
                    default='data/histomorfologico/representative_images_annotation.xlsx',
                    help='Ruta al archivo Excel con las anotaciones')

args = parser.parse_args()

# Validar argumentos
if args.database:
    db1 = db2 = args.database
elif args.database1 and args.database2:
    db1 = args.database1
    db2 = args.database2
else:
    print("ERROR: Debe especificar --database O ambos --database1 y --database2")
    sys.exit(1)

# Leer datos de las bases de datos especificadas
try:
    df1 = pd.read_excel(args.excel, sheet_name=db1)
    df2 = pd.read_excel(args.excel, sheet_name=db2)
except FileNotFoundError:
    print(f"ERROR: No se encuentra el archivo {args.excel}")
    sys.exit(1)
except ValueError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Filtrar grupos según las etiquetas especificadas
df_group_1 = df1[df1['ETIQUETA'] == args.label1]
df_group_2 = df2[df2['ETIQUETA'] == args.label2]

# Verificar que existan datos para ambas etiquetas
if len(df_group_1) == 0:
    print(f"ERROR: No se encontraron datos para la etiqueta '{args.label1}' en {db1}")
    sys.exit(1)
if len(df_group_2) == 0:
    print(f"ERROR: No se encontraron datos para la etiqueta '{args.label2}' en {db2}")
    sys.exit(1)

print("DISTRIBUCIÓN DE MUESTRAS:")
print("="*80)
print(f"{db1} {args.label1}: {len(df_group_1)} casos")
print(f"{db2} {args.label2}: {len(df_group_2)} casos")

# Variables a analizar (excluyendo DISTANCIA)
variables = ['ESTRUCTURA GLANDULAR', 'ATIPIA NUCLEAR', 
             'MITOSIS', 'NECROSIS', 'INFILTRADO_LI', 'INFILTRADO_PMN']

results = []

for var in variables:
    # Crear tabla de contingencia
    combined = pd.concat([
        df_group_1[[var]].assign(grupo=args.label1),
        df_group_2[[var]].assign(grupo=args.label2)
    ])

    ct = pd.crosstab(combined['grupo'], combined[var])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    # Test chi-cuadrado
    try:
        chi2, p_val, dof, expected = stats.chi2_contingency(ct)

        # Cramér's V
        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        # Diferencia máxima entre porcentajes
        max_diff = abs(ct_pct.loc[args.label1] - ct_pct.loc[args.label2]).max()

        # Distribuciones como string
        dist_1 = ', '.join([f'{cat}:{ct_pct.loc[args.label1, cat]:.1f}%' for cat in ct_pct.columns])
        dist_2 = ', '.join([f'{cat}:{ct_pct.loc[args.label2, cat]:.1f}%' for cat in ct_pct.columns])

        results.append({
            'Variable': var,
            'Chi2': chi2,
            'p-value': p_val,
            'Cramers_V': cramers_v,
            'Max_Diff_%': max_diff,
            'Significativo': 'Sí' if p_val < 0.05 else 'No',
            args.label1: dist_1,
            args.label2: dist_2
        })
        
        # Mostrar tabla de contingencia detallada
        print(f"\n\n{var}")
        print("-" * 80)
        print("Tabla de contingencia (frecuencias):")
        print(ct)
        print("\nDistribución porcentual:")
        print(ct_pct.round(1))
        
    except Exception as e:
        results.append({
            'Variable': var,
            'Chi2': np.nan,
            'p-value': np.nan,
            'Cramers_V': np.nan,
            'Max_Diff_%': np.nan,
            'Significativo': 'Error',
            args.label1: 'Error',
            args.label2: 'Error'
        })

# Crear DataFrame de resultados
df_results = pd.DataFrame(results)

print("\n\n" + "="*80)
print(f"TABLA DE RESULTADOS: {db1} {args.label1} vs {db2} {args.label2}")
print("="*80)

# Tabla principal
print("\nEstadísticos:")
print(df_results[['Variable', 'Chi2', 'p-value', 'Cramers_V', 'Max_Diff_%', 'Significativo']].to_string(index=False))

print("\n\n" + "="*80)
print("DISTRIBUCIONES PORCENTUALES")
print("="*80)
for idx, row in df_results.iterrows():
    print(f"\n{row['Variable']}:")
    print(f"  {args.label1}: {row[args.label1]}")
    print(f"  {args.label2}: {row[args.label2]}")

# Resumen
print("\n\n" + "="*80)
print("RESUMEN")
print("="*80)

sig_count = (df_results['Significativo'] == 'Sí').sum()
total = len(variables)

print(f"\nVariables con diferencias significativas (p<0.05): {sig_count}/{total}")

if sig_count > 0:
    print("\nVariables DIFERENTES:")
    for _, row in df_results[df_results['Significativo'] == 'Sí'].iterrows():
        print(f"  • {row['Variable']}: p={row['p-value']:.4f}, V={row['Cramers_V']:.3f}, Δmax={row['Max_Diff_%']:.1f}%")
else:
    print("\n✓ NO hay diferencias significativas")

nosig_count = (df_results['Significativo'] == 'No').sum()
if nosig_count > 0:
    print(f"\nVariables SIMILARES (p≥0.05): {nosig_count}/{total}")
    for _, row in df_results[df_results['Significativo'] == 'No'].iterrows():
        print(f"  • {row['Variable']}: p={row['p-value']:.4f}, V={row['Cramers_V']:.3f}")

# Interpretación global
avg_cramers = df_results['Cramers_V'].mean()
print(f"\n\nINTERPRETACIÓN GLOBAL:")
print(f"Cramér's V promedio: {avg_cramers:.3f}")

if avg_cramers < 0.1:
    print(f"→ {args.label1} y {args.label2} son MORFOLÓGICAMENTE MUY SIMILARES")
elif avg_cramers < 0.3:
    print(f"→ {args.label1} y {args.label2} muestran SIMILITUD MODERADA")
else:
    print(f"→ {args.label1} y {args.label2} son MORFOLÓGICAMENTE DIFERENTES")