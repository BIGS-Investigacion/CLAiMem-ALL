#!/bin/bash

# Run feature space consistency analysis WITH feature selection

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate clam_latest

DATASET="data/histomorfologico"
OUTPUT="results/feature_space_analysis_selected"
CACHE="cache/virchow_features"
BATCH_SIZE=32
N_FEATURES=500

echo "=========================================="
echo "Feature Space Consistency Analysis"
echo "WITH FEATURE SELECTION"
echo "Using conda environment: clam_latest"
echo "=========================================="
echo ""

# Array of tasks
TASKS=("her2" "er" "pr" "pam50")

# Array of selection methods
METHODS=("anova" "mutual_info" "pca")

for task in "${TASKS[@]}"; do
    for method in "${METHODS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Task: $task | Method: $method"
        echo "=========================================="
        echo ""

        python src/claude/feature_space_consistency_with_selection.py \
            --patches_dir "$DATASET" \
            --task "$task" \
            --output "$OUTPUT" \
            --cache_dir "$CACHE" \
            --selection_method "$method" \
            --n_features "$N_FEATURES" \
            --batch_size "$BATCH_SIZE"

        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Task $task with $method completed successfully"
        else
            echo ""
            echo "✗ Task $task with $method failed"
            exit 1
        fi
    done
done

echo ""
echo "=========================================="
echo "All analyses completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT"
