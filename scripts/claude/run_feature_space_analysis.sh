#!/bin/bash
# Script para ejecutar el análisis de consistencia del feature space
# Analiza patches de alta atención con Virchow v2 embeddings

set -e  # Exit on error

# Verificar que estamos en el entorno correcto
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "ERROR: No conda environment activated"
    echo "Please activate: conda activate clam_latest"
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV}" != "clam_latest" ]]; then
    echo "WARNING: Current environment is ${CONDA_DEFAULT_ENV}, expected clam_latest"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Configuración
PATCHES_DIR="data/histomorfologico"
OUTPUT_DIR="results/feature_space_analysis"
BATCH_SIZE=32
DEVICE="cuda"

# Verificar que el directorio de patches existe
if [ ! -d "$PATCHES_DIR" ]; then
    echo "ERROR: Patches directory not found: $PATCHES_DIR"
    exit 1
fi

# Crear directorio de salida
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "FEATURE SPACE CONSISTENCY ANALYSIS"
echo "================================================================================"
echo "Patches directory: $PATCHES_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo "Batch size:        $BATCH_SIZE"
echo "Device:            $DEVICE"
echo "Environment:       $CONDA_DEFAULT_ENV"
echo "================================================================================"
echo ""

# Función para ejecutar análisis
run_analysis() {
    local task=$1
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "ANALYZING TASK: ${task^^}"
    echo "--------------------------------------------------------------------------------"

    python src/claude/feature_space_consistency_from_patches.py \
        --patches_dir "$PATCHES_DIR" \
        --task "$task" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR" \
        --plot_format png \
        --skip_tsne

    if [ $? -eq 0 ]; then
        echo "✓ Analysis completed successfully for $task"
    else
        echo "✗ Analysis failed for $task"
        return 1
    fi
}

# Ejecutar análisis para cada tarea
# Puedes comentar las que no quieras ejecutar

echo "Select which tasks to analyze:"
echo "  1) PAM50 only"
echo "  2) IHC markers only (ER, PR, HER2)"
echo "  3) All tasks"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        run_analysis "pam50"
        ;;
    2)
        run_analysis "er"
        run_analysis "pr"
        run_analysis "her2"
        ;;
    3)
        run_analysis "pam50"
        run_analysis "er"
        run_analysis "pr"
        run_analysis "her2"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "ALL ANALYSES COMPLETED"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Summary:"
ls -lh "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.png 2>/dev/null || echo "No output files found"
echo ""
