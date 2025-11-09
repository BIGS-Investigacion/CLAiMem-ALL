# Configuración de Técnicas MIL

Este documento describe cómo configurar y utilizar diferentes técnicas MIL (Multiple Instance Learning) en el proyecto.

## Técnicas Disponibles

1. **clam_sb** - CLAM Single Branch (por defecto para tareas binarias)
2. **clam_mb** - CLAM Multi Branch (por defecto para tareas multiclase)
3. **mil** - MIL básico con fully connected
4. **abmil** - Attention-Based MIL
5. **rrt** - Region-Representative Transformer
6. **transmil** - Transformer MIL con Nystrom attention
7. **wikg** - Weakly-supervised Instance-level Knowledge Graph

## Uso en Scripts de Entrenamiento

### Cross-Validation (cv_trainer.sh)

Sintaxis:
```bash
bash scripts/cv_trainer.sh <database> <task> <features_dir> <patient_strat> <k_folds> <encoder> <diversity> [mil_technique]
```

Ejemplos:

```bash
# Usando CLAM-SB (por defecto para ER)
bash scripts/cv_trainer.sh tcga er /path/to/features YES 5 2 NO

# Usando ABMIL para clasificación ER
bash scripts/cv_trainer.sh tcga er /path/to/features YES 5 2 NO abmil

# Usando TransMIL con encoder CONCH
bash scripts/cv_trainer.sh tcga er /path/to/features YES 5 2 NO transmil

# Usando RRT para PAM50
bash scripts/cv_trainer.sh tcga pam50 /path/to/features YES 5 11 NO rrt
```

### Hold-out Training (ho_trainer.sh)

Sintaxis:
```bash
bash scripts/ho_trainer.sh <database> <task> <features_dir> <patient_strat> <her2_virtual> <encoder> <diversity> [mil_technique]
```

Ejemplos:

```bash
# Usando CLAM-SB (por defecto)
bash scripts/ho_trainer.sh tcga er /path/to/features YES NO 2 NO

# Usando WiKG
bash scripts/ho_trainer.sh tcga er /path/to/features YES NO 2 NO wikg

# Usando TransMIL
bash scripts/ho_trainer.sh cptac pr /path/to/features YES NO 11 NO transmil
```

## Uso Directo en Python

### cv_main.py

```bash
python src/cv_main.py \
  --model_type abmil \
  --hidden_dim 512 \
  --abmil_is_norm \
  --embed_dim 512 \
  --drop_out 0.3 \
  --k 5 \
  --data_root_dir /path/to/features \
  --csv_path data/dataset_csv/tcga-er.csv \
  --label_dict "{'negative':0,'positive':1}" \
  --split_dir .splits/tcga/cv-5-xxx \
  --results_dir .results/tcga/er/abmil/
```

### ho_main.py

```bash
python src/ho_main.py \
  --model_type transmil \
  --hidden_dim 512 \
  --embed_dim 1024 \
  --data_root_dir_train /path/to/train/features \
  --data_root_dir_test /path/to/test/features \
  --csv_path_train data/dataset_csv/tcga-er.csv \
  --csv_path_test data/dataset_csv/cptac-er.csv \
  --label_dict "{'negative':0,'positive':1}" \
  --results_dir .results/tcga/er/transmil/
```

## Parámetros de Configuración

### Parámetros Comunes (ABMIL, RRT, TransMIL, WiKG)
- `--hidden_dim`: Dimensión oculta para todas las técnicas MIL (default: 512)
  - Recomendación: Usar el mismo valor que `--embed_dim` para mejores resultados

### Parámetros Específicos de ABMIL
- `--abmil_is_norm`: Normalizar pesos de atención (default: True)

### CLAM (clam_sb, clam_mb)
- `--B`: Número de patches positivos/negativos a muestrear (default: 8)
- `--inst_loss`: Loss a nivel de instancia ('svm', 'ce', None)
- `--topo`: Agregar diversidad topológica (CLAM Enhanced)
- `--model_size`: Tamaño del modelo ('small', 'big')

### Notas sobre Técnicas Específicas

**RRT**: Tiene 42 parámetros internos adicionales configurados en `src/models/mils/RRT.py` (region_num, n_layers, n_heads, etc.)

**TransMIL**: Utiliza Nystrom attention y PPEG position encoding internamente

**WiKG**: Utiliza construcción de grafos top-k y bi-interaction aggregation

## Configuración en builder.py

El factory pattern está implementado en `src/models/builder.py` en la función `build_mil_model()`:

```python
from models.builder import build_mil_model

# El factory automáticamente crea el modelo correcto basado en args.model_type
model = build_mil_model(args, model_dict, device)
```

## Compatibilidad con Encoders

Todas las técnicas MIL son compatibles con todos los encoders foundation model disponibles:

1. ResNet50 truncado (embed_dim: 1024)
2. CONCH (embed_dim: 512)
3. CTransPath (embed_dim: 768)
4. Hibou-B (embed_dim: 768)
5. Hibou-L (embed_dim: 1024)
6. H-optimus-0 (embed_dim: 1536)
7. MUSK (embed_dim: 2048)
8. Phikon (embed_dim: 1024)
9. Prov-GigaPath (embed_dim: 1536)
10. RetCCL (embed_dim: 2048)
11. UNI v1 (embed_dim: 1024)
12. UNI v2 (embed_dim: 1536)
13. Virchow (embed_dim: 2560)

El parámetro `embed_dim` se ajusta automáticamente según el encoder seleccionado.

## Notas Importantes

1. **Compatibilidad**: Algunas técnicas MIL requieren GPU con suficiente memoria (especialmente TransMIL y RRT)
2. **Dropout**: El parámetro `--drop_out` afecta principalmente a CLAM. Para otras técnicas, el dropout está configurado internamente
3. **Hidden dimension**: Para obtener mejores resultados, configura `--hidden_dim` igual a `--embed_dim`
4. **Subtyping**: El flag `--subtyping` es principalmente para CLAM y puede no tener efecto en otras técnicas

## Ejemplos de Experimentos Completos

### Comparación de técnicas MIL con CONCH (encoder #2)

```bash
# CLAM-SB
bash scripts/cv_trainer.sh tcga er /data/features YES 5 2 NO clam_sb

# ABMIL
bash scripts/cv_trainer.sh tcga er /data/features YES 5 2 NO abmil

# TransMIL
bash scripts/cv_trainer.sh tcga er /data/features YES 5 2 NO transmil

# RRT
bash scripts/cv_trainer.sh tcga er /data/features YES 5 2 NO rrt

# WiKG
bash scripts/cv_trainer.sh tcga er /data/features YES 5 2 NO wikg
```

Todos los resultados se guardarán en directorios separados bajo `.results/tcga/er/{technique}/`
