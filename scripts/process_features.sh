#!/bin/bash

# Script para procesar features de WSI y generar archivos agregados por etiqueta
# Uso: bash scripts/process_features.sh <database> <subtype> <features_dir> <top_k> <selection_method>

if [ "$1" != "cptac" ] && [ "$1" != "tcga" ]; then
    echo "Invalid database name. Use 'cptac' or 'tcga'."
    exit 1
else
    DATABASE=$1
fi

if [ "$2" == "pam50" ]; then
    CSV_FILE=data/dataset_csv/$DATABASE-subtype_pam50.csv
elif [ "$2" == "ihc" ]; then
    CSV_FILE=data/dataset_csv/$DATABASE-subtype_ihc.csv
elif [ "$2" == "ihc_simple" ]; then
    CSV_FILE=data/dataset_csv/$DATABASE-subtype_ihc_simple.csv
elif [ "$2" == "erbb2" ]; then
    CSV_FILE=data/dataset_csv/$DATABASE-erbb2.csv
elif [ "$2" == "pr" ]; then
    CSV_FILE=data/dataset_csv/$DATABASE-pr.csv
elif [ "$2" == "er" ]; then
    CSV_FILE=data/dataset_csv/$DATABASE-er.csv
else
    echo "Invalid parameter. Use 'ihc', 'ihc_simple', 'pam50', 'er', 'pr' or 'erbb2'."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Please provide the directory where the features are located."
    exit 1
else
    FEATURES_DIR=$3
fi

if [ -z "$4" ]; then
    echo "Using default top_k=100"
    TOP_K=100
else
    TOP_K=$4
fi

if [ -z "$5" ]; then
    echo "Using default selection_method=attention"
    SELECTION_METHOD="attention"
else
    SELECTION_METHOD=$5
fi

# Validar método de selección
if [ "$SELECTION_METHOD" != "attention" ] && [ "$SELECTION_METHOD" != "norm" ] && [ "$SELECTION_METHOD" != "variance" ] && [ "$SELECTION_METHOD" != "random" ]; then
    echo "Invalid selection method. Use 'attention', 'norm', 'variance', or 'random'."
    exit 1
fi

# Crear directorio de salida
OUTPUT_DIR=data/aggregated_features/$DATABASE/$2/top${TOP_K}_${SELECTION_METHOD}
mkdir -p $OUTPUT_DIR

echo "================================================================================"
echo "PROCESSING FEATURES"
echo "================================================================================"
echo "Database:         $DATABASE"
echo "Subtype:          $2"
echo "CSV file:         $CSV_FILE"
echo "Features dir:     $FEATURES_DIR"
echo "Output dir:       $OUTPUT_DIR"
echo "Top-K:            $TOP_K"
echo "Selection method: $SELECTION_METHOD"
echo "================================================================================"

# Ejecutar script de extracción
python src/claude/extract_top_attention_features.py \
    --input_dir $FEATURES_DIR \
    --output $OUTPUT_DIR \
    --labels $CSV_FILE \
    --top_k $TOP_K \
    --selection_method $SELECTION_METHOD \
    --aggregation_method concat \
    --save_metadata

echo ""
echo "================================================================================"
echo "PROCESSING COMPLETED"
echo "================================================================================"
echo "Output files saved to: $OUTPUT_DIR"
echo "CSV mapping:           $OUTPUT_DIR/labels.csv"
echo "================================================================================"
