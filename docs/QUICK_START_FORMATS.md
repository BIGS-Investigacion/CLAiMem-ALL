# Quick Start Guide - Domain Shift Analysis

## 🚀 Ejecutar Análisis Completo en 3 Pasos

### Paso 1: Preparar tus datos

Reemplaza los archivos de ejemplo con tus datos reales:

```bash
# Para PAM50
cp tus_datos/pam50_accuracy.csv data/examples/pam50_accuracy.csv
cp tus_datos/tcga_pam50_labels.csv data/examples/tcga_pam50_labels.csv
cp tus_datos/cptac_pam50_labels.csv data/examples/cptac_pam50_labels.csv
cp tus_datos/pam50_performance_metrics.csv data/examples/pam50_performance_metrics.csv
```

### Paso 2: Ejecutar cada análisis

```bash
# 1. Normalización de tinción (RÁPIDO - solo necesita accuracy)
python src/claude/stain_normalization_from_accuracy.py \
  --input data/examples/pam50_accuracy.csv \
  --task pam50 \
  --output results/stain_analysis/

# 2. Distributional Shift (necesita labels + performance metrics)
python src/claude/distributional_shift_analysis.py \
  --tcga_labels data/examples/tcga_pam50_labels.csv \
  --cptac_labels data/examples/cptac_pam50_labels.csv \
  --metrics data/examples/pam50_performance_metrics.csv \
  --task pam50 \
  --output results/distributional_shift/
```

### Paso 3: Ver resultados

Los resultados se guardan en:
- **CSV**: Tablas con métricas detalladas
- **JSON**: Resultados completos estructurados
- **PNG**: Visualizaciones (gráficos, heatmaps)

---

## 📋 Formatos de Entrada Requeridos

### 1️⃣ Normalización de Tinción
**Archivo:** `pam50_accuracy.csv`

```csv
Class,N_samples,Accuracy_Original,Accuracy_Normalized
LumA,150,0.750,0.780
LumB,80,0.650,0.680
Her2,45,0.600,0.620
Basal,70,0.700,0.710
Normal,42,0.550,0.560
```

**¿Cómo obtener estos datos?**
```python
# Si tienes predicciones de ambos modelos:
from sklearn.metrics import accuracy_score

classes = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']

with open('pam50_accuracy.csv', 'w') as f:
    f.write('Class,N_samples,Accuracy_Original,Accuracy_Normalized\n')

    for class_idx, class_name in enumerate(classes):
        # Filtrar por clase
        mask = (y_true == class_idx)
        n = mask.sum()

        # Calcular accuracy
        acc_orig = accuracy_score(y_true[mask], y_pred_original[mask])
        acc_norm = accuracy_score(y_true[mask], y_pred_normalized[mask])

        f.write(f'{class_name},{n},{acc_orig:.4f},{acc_norm:.4f}\n')
```

---

### 2️⃣ Distributional Shift
**Archivos necesarios:**

**A) Labels TCGA:** `tcga_pam50_labels.csv`
```csv
slide_id,label
TCGA-A1-A0SB-01Z-00-DX1,LumA
TCGA-A1-A0SD-01Z-00-DX1,LumA
```

**B) Labels CPTAC:** `cptac_pam50_labels.csv`
```csv
slide_id,label
C3L-00004-21,LumA
C3L-00010-21,LumA
```

**C) Performance Metrics:** `pam50_performance_metrics.csv`
```csv
Class,F1_MCCV,F1_HO
LumA,0.7500,0.6800
LumB,0.6500,0.5800
Her2,0.6000,0.5200
Basal,0.7000,0.6300
Normal,0.5500,0.4800
```

**¿Cómo obtener estos datos?**
```python
# A y B: Extraer desde tus dataset CSVs
import pandas as pd

df_tcga = pd.read_csv('dataset_csv/tcga_all.csv')
df_tcga[['slide_id', 'label']].to_csv('tcga_pam50_labels.csv', index=False)

df_cptac = pd.read_csv('dataset_csv/cptac_all.csv')
df_cptac[['slide_id', 'label']].to_csv('cptac_pam50_labels.csv', index=False)

# C: Desde resultados de cross-validation
# F1_MCCV = promedio de F1 en 10 folds de MCCV
# F1_HO = F1 en hold-out (CPTAC)
```

---

## 🔍 Checklist de Validación

Antes de ejecutar los scripts, verifica:

```python
import pandas as pd

# ✓ Normalización de tinción
df = pd.read_csv('data/examples/pam50_accuracy.csv')
print("Columnas:", df.columns.tolist())
print("N clases:", len(df))
print("Accuracies en [0,1]:", df['Accuracy_Original'].between(0,1).all())

# ✓ Labels
df = pd.read_csv('data/examples/tcga_pam50_labels.csv')
print("Columnas:", df.columns.tolist())
print("N slides:", len(df))
print("Clases únicas:", df['label'].unique())

# ✓ Performance metrics
df = pd.read_csv('data/examples/pam50_performance_metrics.csv')
print("Columnas:", df.columns.tolist())
print("Todas las clases:", set(df['Class']))
```

**Salida esperada:**
```
Columnas: ['Class', 'N_samples', 'Accuracy_Original', 'Accuracy_Normalized']
N clases: 5
Accuracies en [0,1]: True

Columnas: ['slide_id', 'label']
N slides: 1522
Clases únicas: ['LumA' 'LumB' 'Her2' 'Basal' 'Normal']

Columnas: ['Class', 'F1_MCCV', 'F1_HO']
Todas las clases: {'LumA', 'LumB', 'Her2', 'Basal', 'Normal'}
```

---

## ⚡ Comandos Rápidos

### Ejecutar solo lo esencial (sin plots para ir rápido)

```bash
# Análisis de normalización de tinción
python src/claude/stain_normalization_from_accuracy.py \
  -i data/examples/pam50_accuracy.csv \
  -t pam50 \
  -o results/stain/ \
  --no_plots

# Distributional shift
python src/claude/distributional_shift_analysis.py \
  --tcga_labels data/examples/tcga_pam50_labels.csv \
  --cptac_labels data/examples/cptac_pam50_labels.csv \
  --metrics data/examples/pam50_performance_metrics.csv \
  -t pam50 \
  -o results/distrib/ \
  --no_plots
```

### Para todas las tareas (PAM50, ER, PR, HER2)

```bash
for task in pam50 er pr her2; do
    echo "Processing ${task}..."

    # Stain normalization
    python src/claude/stain_normalization_from_accuracy.py \
      -i data/examples/${task}_accuracy.csv \
      -t ${task} \
      -o results/stain/

    # Distributional shift (si tienes los datos)
    if [ -f "data/examples/${task}_performance_metrics.csv" ]; then
        python src/claude/distributional_shift_analysis.py \
          --tcga_labels data/examples/tcga_${task}_labels.csv \
          --cptac_labels data/examples/cptac_${task}_labels.csv \
          --metrics data/examples/${task}_performance_metrics.csv \
          -t ${task} \
          -o results/distrib/
    fi
done
```

---

## 📊 Salidas Esperadas

Cada script genera:

### Stain Normalization
```
results/stain_analysis/
├── pam50_stain_normalization_results.csv
├── pam50_stain_normalization_summary.json
├── pam50_accuracy_comparison.png
├── pam50_delta_accuracy.png
└── pam50_accuracy_heatmap.png
```

### Distributional Shift
```
results/distributional_shift/
├── pam50_prevalence_shift.csv
├── pam50_performance_degradation.csv
├── pam50_distributional_shift_analysis.json
├── pam50_prevalence_comparison.png
├── pam50_correlation.png
└── pam50_combined_analysis.png
```

---

## 🆘 Problemas Comunes

### Error: "Missing required columns"
**Causa:** Nombres de columnas incorrectos
**Solución:** Verifica mayúsculas/minúsculas y espacios exactos

### Error: "Dropped X classes with missing data"
**Causa:** Clases no coinciden entre archivos
**Solución:** Asegúrate que todas las clases aparecen en todos los CSVs

### Error: "File not found"
**Causa:** Ruta incorrecta
**Solución:** Usa rutas absolutas o ejecuta desde el directorio raíz del proyecto

---

## 📖 Documentación Completa

Para más detalles, consulta:
- [README_DOMAIN_SHIFT_ANALYSIS.md](../../src/claude/README_DOMAIN_SHIFT_ANALYSIS.md) - Pipeline completo
- [README_DATA_FORMATS.md](README_DATA_FORMATS.md) - Formatos detallados

---

## ✅ Próximos Pasos

1. ✓ Crear tus CSVs con datos reales
2. ✓ Validar formatos con el checklist
3. ✓ Ejecutar stain normalization
4. ✓ Ejecutar distributional shift
5. ✓ Revisar resultados en `results/`
6. ✓ (Opcional) Ejecutar otros análisis si tienes los datos

¡Listo! Con estos datos ya puedes empezar tu análisis.