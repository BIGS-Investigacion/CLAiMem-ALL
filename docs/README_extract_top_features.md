# Extracción de Top Features de WSI por Atención

Scripts para extraer los top-K features con mayor score de atención de archivos WSI y agregarlos por etiqueta.

## Archivos

1. **extract_top_attention_features.py** - Script principal para extraer y agregar features
2. **create_labels_file_example.py** - Utilidad para crear archivo CSV de etiquetas

## Modos de operación

### Modo 1: Auto-descubrimiento (más simple)

Organiza tus archivos en subcarpetas por etiqueta y el script las detecta automáticamente:

```bash
python src/claude/extract_top_attention_features.py \
    --input_dir /data/features/ \
    --output aggregated.pt \
    --top_k 100 \
    --auto_discover
```

**Estructura esperada:**
```
input_dir/
├── LABEL1/
│   ├── file1.pt
│   ├── file2.pt
│   └── ...
├── LABEL2/
│   ├── file3.pt
│   └── ...
└── LABEL3/
    └── ...
```

### Modo 2: Archivo CSV de etiquetas (más flexible)

Crea un CSV que mapea archivos a etiquetas (útil cuando los archivos no están en subcarpetas):

```bash
python src/claude/extract_top_attention_features.py \
    --input_dir /data/features/ \
    --output aggregated.pt \
    --labels labels.csv \
    --top_k 100
```

## Flujo de trabajo

### 1. Preparar archivo de etiquetas

El script necesita un CSV con dos columnas: `filename` y `label`.

**Opción A: Crear desde directorio** (nombres de archivo contienen la etiqueta)

```bash
python src/claude/create_labels_file_example.py \
    --mode directory \
    --input /path/to/features/ \
    --output labels.csv \
    --extension .pt \
    --pattern underscore \
    --label_position -2
```

Ejemplos de nombres de archivo:
- `TCGA_BRCA_BASAL_001.pt` → etiqueta: `BASAL` (posición -2)
- `patient_456_LUMINAL_A_features.pt` → etiqueta: `LUMINAL_A` (posición -2)

**Opción B: Crear desde Excel existente**

```bash
python src/claude/create_labels_file_example.py \
    --mode excel \
    --input data/histomorfologico/annotations.xlsx \
    --sheet TCGA \
    --filename_col IMAGEN \
    --label_col ETIQUETA \
    --output labels.csv
```

**Opción C: Crear manualmente**

```csv
filename,label
TCGA_001.pt,BASAL
TCGA_002.pt,LUMINAL-A
TCGA_003.pt,BASAL
TCGA_004.pt,HER2-enriched
TCGA_005.pt,LUMINAL-B
```

### 2. Extraer y agregar features

```bash
python src/claude/extract_top_attention_features.py \
    --input_dir /path/to/wsi/features/ \
    --output aggregated_features.pt \
    --labels labels.csv \
    --top_k 100 \
    --selection_method attention \
    --aggregation_method mean
```

## Parámetros detallados

### extract_top_attention_features.py

| Parámetro | Descripción | Default | Opciones |
|-----------|-------------|---------|----------|
| `--input_dir`, `-i` | Directorio con archivos .pt de features | *requerido* | - |
| `--output`, `-o` | Archivo de salida (.pt o .npz) | *requerido* | - |
| `--labels`, `-l` | Archivo CSV con filename,label | opcional* | - |
| `--top_k`, `-k` | Número de features top a extraer por WSI | 100 | cualquier entero |
| `--selection_method`, `-s` | Método para seleccionar top features | `attention` | `attention`, `norm`, `variance`, `random` |
| `--aggregation_method`, `-a` | Método para agregar features entre WSI | `concat` | `concat`, `mean`, `max`, `sum` |
| `--file_extension`, `-e` | Extensión de archivos de features | `.pt` | cualquier extensión |
| `--save_metadata` | Guardar metadata con info de agregación | False | flag |
| `--device`, `-dev` | Dispositivo a usar | `auto` | `auto`, `cuda`, `cpu` |
| `--batch_process`, `-b` | Procesar archivos en batch en GPU | False | flag |
| `--auto_discover`, `-auto` | Descubrir subcarpetas como etiquetas | False | flag |

\* `--labels` es opcional si se usa `--auto_discover`

### Métodos de selección (--selection_method)

- **attention**: Selecciona patches con mayor score de atención (requiere que el .pt contenga scores)
- **norm**: Selecciona patches con mayor norma L2 (features con mayor magnitud)
- **variance**: Selecciona patches con mayor varianza (más informativos)
- **random**: Muestreo aleatorio de patches

### Métodos de agregación (--aggregation_method)

- **concat** (default): Concatenar todos los features → shape: [N_wsi * K, D]
  - Ejemplo: 50 WSI × top-100 = [5000, 512]
  - Acumula TODOS los features sin pérdida de información
- **mean**: Promedio de los top-k features de cada WSI de la etiqueta → shape: [K, D]
- **max**: Máximo elemento a elemento → shape: [K, D]
- **sum**: Suma de features → shape: [K, D]

## Formatos de archivo soportados

### Entrada (.pt files)

El script acepta varios formatos de archivos PyTorch:

**Formato 1: Diccionario con 'features' y 'attention'**
```python
{
    'features': torch.Tensor([N_patches, feature_dim]),  # features de patches
    'attention': torch.Tensor([N_patches])               # scores de atención
}
```

**Formato 2: Solo tensor de features**
```python
torch.Tensor([N_patches, feature_dim])
```

**Formato 3: Nombres alternativos**
```python
{
    'feat': torch.Tensor([N_patches, feature_dim]),
    'att': torch.Tensor([N_patches])
}
# o
{
    'embeddings': torch.Tensor([N_patches, feature_dim]),
    'scores': torch.Tensor([N_patches])
}
```

### Salida

**Formato PyTorch (.pt)**
```python
{
    'features': {
        'BASAL': torch.Tensor([5000, 512]),        # 50 WSI × 100 features
        'LUMINAL-A': torch.Tensor([8000, 512]),    # 80 WSI × 100 features
        'HER2-enriched': torch.Tensor([3000, 512]), # 30 WSI × 100 features
        ...
    },
    'metadata': {
        'top_k': 100,
        'selection_method': 'attention',
        'aggregation_method': 'concat',
        'labels': {
            'BASAL': {
                'n_files': 45,
                'files': ['file1.pt', 'file2.pt', ...],
                'feature_shape': [5000, 512]  # 45 × 100 + extras
            },
            ...
        }
    }
}
```

**Formato NumPy (.npz)**
```python
{
    'BASAL': np.array([5000, 512]),      # 50 WSI × 100 features
    'LUMINAL-A': np.array([8000, 512]),  # 80 WSI × 100 features
    ...
}
# Metadata guardada en archivo .json separado
```

## Ejemplos de uso

### Ejemplo 1: Auto-descubrimiento de subcarpetas (recomendado)

Si tus features están organizados en subcarpetas por etiqueta:

```
/data/features/
├── BASAL/
│   ├── patient_001.pt
│   ├── patient_002.pt
│   └── ...
├── LUMINAL-A/
│   ├── patient_050.pt
│   ├── patient_051.pt
│   └── ...
├── LUMINAL-B/
│   └── ...
└── HER2-enriched/
    └── ...
```

```bash
# Auto-descubre subcarpetas y las usa como etiquetas
python src/claude/extract_top_attention_features.py \
    --input_dir /data/features/ \
    --output results/aggregated_features.pt \
    --top_k 100 \
    --auto_discover \
    --save_metadata
```

### Ejemplo 2: Pipeline completo con PAM50 usando CSV (usando GPU)

```bash
# 1. Crear archivo de etiquetas desde directorio
python src/claude/create_labels_file_example.py \
    --mode directory \
    --input /data/features/TCGA/ \
    --output labels_pam50.csv \
    --pattern underscore \
    --label_position -1

# 2. Extraer top-100 features por atención y concatenar (usa GPU automáticamente)
python src/claude/extract_top_attention_features.py \
    --input_dir /data/features/TCGA/ \
    --output results/aggregated_pam50_top100.pt \
    --labels labels_pam50.csv \
    --top_k 100 \
    --selection_method attention \
    --save_metadata
```

### Ejemplo 3: Extraer por norma (alternativa sin scores de atención)

```bash
python src/claude/extract_top_attention_features.py \
    --input_dir /data/features/CPTAC/ \
    --output results/aggregated_cptac_top50.pt \
    --labels labels_cptac.csv \
    --top_k 50 \
    --selection_method norm
```

### Ejemplo 4: Crear desde Excel y guardar en NumPy

```bash
# 1. Crear labels desde Excel
python src/claude/create_labels_file_example.py \
    --mode excel \
    --input data/histomorfologico/representative_images_annotation.xlsx \
    --sheet TCGA \
    --filename_col IMAGEN \
    --label_col ETIQUETA \
    --output labels_tcga.csv

# 2. Extraer y guardar como .npz
python src/claude/extract_top_attention_features.py \
    --input_dir /data/features/TCGA/ \
    --output results/aggregated_tcga.npz \
    --labels labels_tcga.csv \
    --top_k 200 \
    --selection_method variance \
    --save_metadata
```

## Cargar resultados

### PyTorch
```python
import torch

# Cargar archivo
data = torch.load('aggregated_features.pt')

# Acceder a features por etiqueta
basal_features = data['features']['BASAL']  # shape: [5000, 512] (50 WSI × 100)
luminal_features = data['features']['LUMINAL-A']  # shape: [8000, 512] (80 WSI × 100)

# Ver metadata
metadata = data['metadata']
print(f"Top-K usado: {metadata['top_k']}")
print(f"Método de selección: {metadata['selection_method']}")
print(f"Archivos procesados para BASAL: {metadata['labels']['BASAL']['n_files']}")
```

### NumPy
```python
import numpy as np
import json

# Cargar features
data = np.load('aggregated_features.npz')
basal_features = data['BASAL']  # shape: [5000, 512] (50 WSI × 100)

# Cargar metadata
with open('aggregated_features.json') as f:
    metadata = json.load(f)
```

## Aceleración GPU

El script detecta automáticamente si hay GPU disponible y la usa para acelerar el procesamiento.

### Uso automático de GPU

```bash
# Por defecto detecta y usa GPU si está disponible
python src/claude/extract_top_attention_features.py \
    --input_dir /data/features/ \
    --output aggregated.pt \
    --labels labels.csv \
    --top_k 100
```

### Forzar uso de CPU

```bash
# Forzar CPU (útil si la GPU tiene poca memoria)
python src/claude/extract_top_attention_features.py \
    --input_dir /data/features/ \
    --output aggregated.pt \
    --labels labels.csv \
    --top_k 100 \
    --device cpu
```

### Verificar GPU

El script imprime información de la GPU al inicio:
```
================================================================================
CONFIGURACIÓN DE EJECUCIÓN
================================================================================
Dispositivo: cuda:0
GPU detectada: NVIDIA GeForce RTX 3090
Memoria GPU disponible: 24.00 GB
Batch processing: Desactivado
================================================================================
```

### Beneficios de GPU

- **Ordenamiento más rápido**: `torch.argsort` en GPU es mucho más rápido para tensores grandes
- **Cálculos vectorizados**: Normas L2 y varianza se calculan en paralelo
- **Agregación acelerada**: Stack, concat, mean en GPU son más eficientes
- **Speedup típico**: 5-10x más rápido con GPU vs CPU para archivos grandes

### Gestión de memoria

El script libera automáticamente memoria GPU después de procesar cada etiqueta:
```python
torch.cuda.empty_cache()  # Ejecutado automáticamente
```

Si tienes problemas de memoria GPU, reduce `--top_k` o usa `--device cpu`.

### Ejemplo de salida con GPU, progreso y auto-discovery

```
================================================================================
CONFIGURACIÓN DE EJECUCIÓN
================================================================================
Dispositivo: cuda:0
GPU detectada: NVIDIA GeForce RTX 3090
Memoria GPU disponible: 24.00 GB
Batch processing: Desactivado
================================================================================

Descubriendo etiquetas automáticamente desde subcarpetas de: /data/features/
Modo: Auto-descubrimiento de subcarpetas

Encontradas 5 etiquetas únicas:
  - BASAL: 45 archivos
  - LUMINAL-A: 82 archivos
  - LUMINAL-B: 53 archivos
  - HER2-enriched: 28 archivos
  - NORMAL-like: 12 archivos

================================================================================
Procesando etiqueta: BASAL (45 archivos)
================================================================================
  [1/45] Restantes: 44 ✓ TCGA_BASAL_001.pt: 4523 patches → top-100
  [2/45] Restantes: 43 ✓ TCGA_BASAL_002.pt: 3891 patches → top-100
  [3/45] Restantes: 42 ✓ TCGA_BASAL_003.pt: 5234 patches → top-100
  ...
  [45/45] Restantes: 0 ✓ TCGA_BASAL_045.pt: 4102 patches → top-100

  → Agregado final para 'BASAL': torch.Size([4500, 512])
```

## Casos de uso típicos

### Reducir tamaño de dataset
Si tienes 1000 WSI con 5000 patches cada uno, puedes reducir a top-100 patches:
- Antes: 1000 × 5000 = 5M patches
- Después: 1000 × 100 = 100K patches (reducción de 50x)

### Crear representación por clase
Si tienes 50 WSI BASAL, puedes crear 1 representación agregada:
- Antes: 50 instancias (archivos .pt separados)
- Después: 1 instancia (tensor [100, 512] con los top-100 features promediados)

### Analizar features más relevantes
Usando `--selection_method attention` puedes identificar qué patches son más importantes según el modelo de atención.

## Ventajas del auto-descubrimiento

1. **No requiere archivo CSV**: Detecta automáticamente las etiquetas desde los nombres de las subcarpetas
2. **Más simple**: Solo necesitas organizar archivos en carpetas
3. **Menos errores**: Evita errores de tipeo en archivos CSV
4. **Más rápido**: No necesitas crear y mantener un archivo CSV separado
5. **Estructura clara**: Organización visual clara de tus datos

## Notas importantes

1. **Scores de atención**: Si usas `--selection_method attention` pero tus archivos .pt no contienen scores de atención, el script fallará. Usa `norm` o `variance` en su lugar.

2. **Memoria**: El método `concat` puede generar tensores muy grandes si tienes muchos WSI por etiqueta.

3. **Reproducibilidad**: Para `random`, los resultados cambiarán en cada ejecución. Añade semilla si necesitas reproducibilidad.

4. **Validación**: El script imprime estadísticas de cada archivo procesado. Revisa la salida para detectar problemas.
