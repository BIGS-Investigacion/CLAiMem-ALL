# Usar Tus Datos Existentes

Esta guía muestra cómo usar los scripts de análisis con **tus archivos existentes** en lugar de los ejemplos.

## 📂 Tus Archivos Actuales

Ya tienes los archivos de labels en `data/dataset_csv/`:

```bash
# PAM50 molecular subtyping
data/dataset_csv/tcga-subtype_pam50.csv
data/dataset_csv/cptac-subtype_pam50.csv

# IHC biomarkers
data/dataset_csv/tcga-er.csv
data/dataset_csv/cptac-er.csv
data/dataset_csv/tcga-pr.csv
data/dataset_csv/cptac-pr.csv
data/dataset_csv/tcga-erbb2.csv
data/dataset_csv/cptac-erbb2.csv
```

**Formato de estos archivos:**
```csv
case_id,slide_id,label
TCGA-3C-AAAU-01A,TCGA-3C-AAAU-01A-01-TS1...,luma
```

¡Estos archivos **ya están en el formato correcto**! 🎉

## 🚀 Cómo Usar Tus Datos

### 1. Análisis de Distributional Shift

Puedes usar **directamente** tus archivos existentes:

```bash
# PAM50
python src/claude/distributional_shift_analysis.py \
  --tcga_labels data/dataset_csv/tcga-subtype_pam50.csv \
  --cptac_labels data/dataset_csv/cptac-subtype_pam50.csv \
  --metrics results/pam50_performance_metrics.csv \
  --task pam50 \
  --output results/distributional_shift/

# ER
python src/claude/distributional_shift_analysis.py \
  --tcga_labels data/dataset_csv/tcga-er.csv \
  --cptac_labels data/dataset_csv/cptac-er.csv \
  --metrics results/er_performance_metrics.csv \
  --task er \
  --output results/distributional_shift/

# PR
python src/claude/distributional_shift_analysis.py \
  --tcga_labels data/dataset_csv/tcga-pr.csv \
  --cptac_labels data/dataset_csv/cptac-pr.csv \
  --metrics results/pr_performance_metrics.csv \
  --task pr \
  --output results/distributional_shift/

# HER2
python src/claude/distributional_shift_analysis.py \
  --tcga_labels data/dataset_csv/tcga-erbb2.csv \
  --cptac_labels data/dataset_csv/cptac-erbb2.csv \
  --metrics results/her2_performance_metrics.csv \
  --task her2 \
  --output results/distributional_shift/
```

### 2. Lo Que Necesitas Crear

Solo necesitas crear **2 tipos de archivos** para completar todos los análisis:

#### A) Archivos de Accuracy (para Stain Normalization)

**Formato:** `{task}_accuracy.csv`

```csv
Class,N_samples,Accuracy_Original,Accuracy_Normalized
LumA,150,0.750,0.780
LumB,80,0.650,0.680
```

**Script para generar desde tus resultados:**

```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Cargar predicciones
# Asume que tienes:
# - y_true: array con labels verdaderos de CPTAC
# - y_pred_original: predicciones del modelo sin normalización
# - y_pred_normalized: predicciones del modelo con Macenko

# Cargar labels para obtener nombres de clases
df_labels = pd.read_csv('data/dataset_csv/cptac-subtype_pam50.csv')

# Mapeo de labels a índices (ajusta según tu codificación)
label_map = {'luma': 0, 'lumb': 1, 'her2': 2, 'basal': 3, 'normal': 4}
class_names = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']

results = []
for class_idx, class_name in enumerate(class_names):
    # Filtrar por clase
    mask = (y_true == class_idx)
    n_samples = mask.sum()

    # Calcular accuracy
    acc_orig = accuracy_score(y_true[mask], y_pred_original[mask])
    acc_norm = accuracy_score(y_true[mask], y_pred_normalized[mask])

    results.append({
        'Class': class_name,
        'N_samples': int(n_samples),
        'Accuracy_Original': acc_orig,
        'Accuracy_Normalized': acc_norm
    })

df = pd.DataFrame(results)
df.to_csv('data/accuracy/pam50_accuracy.csv', index=False)
print(df)
```

#### B) Archivos de Performance Metrics (para Distributional Shift)

**Formato:** `{task}_performance_metrics.csv`

```csv
Class,F1_MCCV,F1_HO
LumA,0.7500,0.6800
LumB,0.6500,0.5800
```

**Script para generar desde resultados de cross-validation:**

```python
import pandas as pd
import numpy as np

# Supón que tienes resultados de 10 folds de MCCV
# cv_results = lista de dicts con métricas por fold
# ho_results = dict con métricas de hold-out en CPTAC

class_names = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']

results = []
for class_idx, class_name in enumerate(class_names):
    # Promedio de F1 en cross-validation
    f1_mccv_values = [fold['f1_per_class'][class_idx] for fold in cv_results]
    f1_mccv = np.mean(f1_mccv_values)

    # F1 en hold-out
    f1_ho = ho_results['f1_per_class'][class_idx]

    results.append({
        'Class': class_name,
        'F1_MCCV': f1_mccv,
        'F1_HO': f1_ho
    })

df = pd.DataFrame(results)
df.to_csv('results/pam50_performance_metrics.csv', index=False)
print(df)
```

**Para IHC (usa PR-AUC):**

```python
# Para ER, PR, HER2 (clasificación binaria)
class_names = ['ER-negative', 'ER-positive']  # o PR, HER2

results = []
for class_idx, class_name in enumerate(class_names):
    pr_auc_mccv = np.mean([fold['pr_auc_per_class'][class_idx] for fold in cv_results])
    pr_auc_ho = ho_results['pr_auc_per_class'][class_idx]

    results.append({
        'Class': class_name,
        'PR_AUC_MCCV': pr_auc_mccv,
        'PR_AUC_HO': pr_auc_ho
    })

df = pd.DataFrame(results)
df.to_csv('results/er_performance_metrics.csv', index=False)
```

## 🔧 Script Helper: Extraer Métricas de tus Resultados

He creado un script helper en `src/claude/extract_metrics_from_results.py` que puedes usar.

### 3. Verificar Labels Existentes

```bash
# Ver estructura de tus archivos
head -5 data/dataset_csv/tcga-subtype_pam50.csv
head -5 data/dataset_csv/cptac-subtype_pam50.csv

# Contar clases
python3 << EOF
import pandas as pd

df_tcga = pd.read_csv('data/dataset_csv/tcga-subtype_pam50.csv')
df_cptac = pd.read_csv('data/dataset_csv/cptac-subtype_pam50.csv')

print("TCGA PAM50:")
print(df_tcga['label'].value_counts().sort_index())
print(f"Total: {len(df_tcga)}")

print("\nCPTAC PAM50:")
print(df_cptac['label'].value_counts().sort_index())
print(f"Total: {len(df_cptac)}")
EOF
```

## 📋 Checklist de Preparación

### ✅ Ya tienes (no necesitas crear):
- [x] Labels TCGA/CPTAC para PAM50
- [x] Labels TCGA/CPTAC para ER
- [x] Labels TCGA/CPTAC para PR
- [x] Labels TCGA/CPTAC para HER2

### 📝 Necesitas crear:

#### Para Stain Normalization:
- [ ] `pam50_accuracy.csv` - Accuracy con/sin normalización por clase
- [ ] `er_accuracy.csv`
- [ ] `pr_accuracy.csv`
- [ ] `her2_accuracy.csv`

#### Para Distributional Shift:
- [ ] `pam50_performance_metrics.csv` - F1 MCCV vs HO por clase
- [ ] `er_performance_metrics.csv` - PR-AUC MCCV vs HO por clase
- [ ] `pr_performance_metrics.csv`
- [ ] `her2_performance_metrics.csv`

#### Para Feature Space Consistency:
- [ ] Embeddings de Virchow v2 para patches top-8 attention
- [ ] Archivos `.pt` con features y attention scores

#### Para Biological Interpretability:
- [ ] Excel con anotaciones histomorfológicas (ya debes tenerlo)

## 🎯 Workflow Completo

```bash
# 1. Crear directorios
mkdir -p results/{stain_analysis,distributional_shift,feature_space_analysis,biological_analysis,integrative_analysis}
mkdir -p data/accuracy

# 2. Generar archivos de accuracy (modifica según tus resultados)
# python tu_script_accuracy.py

# 3. Generar archivos de performance metrics (modifica según tus resultados)
# python tu_script_metrics.py

# 4. Ejecutar análisis distributional shift (usa tus archivos existentes)
python src/claude/distributional_shift_analysis.py \
  --tcga_labels data/dataset_csv/tcga-subtype_pam50.csv \
  --cptac_labels data/dataset_csv/cptac-subtype_pam50.csv \
  --metrics results/pam50_performance_metrics.csv \
  --task pam50 \
  --output results/distributional_shift/

# 5. Ejecutar stain normalization
python src/claude/stain_normalization_from_accuracy.py \
  --input data/accuracy/pam50_accuracy.csv \
  --task pam50 \
  --output results/stain_analysis/

# 6. (Opcional) Otros análisis si tienes los datos
```

## 💡 Tip: Batch Processing

Para procesar todas las tareas a la vez:

```bash
#!/bin/bash
# Script: run_all_tasks.sh

for task in pam50 er pr her2; do
    echo "Processing ${task}..."

    # Determinar archivos de labels
    if [ "$task" = "pam50" ]; then
        tcga_file="data/dataset_csv/tcga-subtype_pam50.csv"
        cptac_file="data/dataset_csv/cptac-subtype_pam50.csv"
    elif [ "$task" = "her2" ]; then
        tcga_file="data/dataset_csv/tcga-erbb2.csv"
        cptac_file="data/dataset_csv/cptac-erbb2.csv"
    else
        tcga_file="data/dataset_csv/tcga-${task}.csv"
        cptac_file="data/dataset_csv/cptac-${task}.csv"
    fi

    # Distributional shift
    if [ -f "results/${task}_performance_metrics.csv" ]; then
        python src/claude/distributional_shift_analysis.py \
          --tcga_labels "${tcga_file}" \
          --cptac_labels "${cptac_file}" \
          --metrics "results/${task}_performance_metrics.csv" \
          --task "${task}" \
          --output results/distributional_shift/
    fi

    # Stain normalization
    if [ -f "data/accuracy/${task}_accuracy.csv" ]; then
        python src/claude/stain_normalization_from_accuracy.py \
          --input "data/accuracy/${task}_accuracy.csv" \
          --task "${task}" \
          --output results/stain_analysis/
    fi
done
```

Hazlo ejecutable y corre:
```bash
chmod +x run_all_tasks.sh
./run_all_tasks.sh
```

## 🆘 Ayuda

Si tienes dudas sobre cómo extraer las métricas de tus resultados de CLAM, dime:
- ¿Dónde guardas los resultados de cross-validation?
- ¿Qué formato tienen (pkl, json, etc.)?
- ¿Ya tienes los modelos entrenados con/sin Macenko?

Y te ayudo a crear los scripts de extracción específicos para tu caso.
