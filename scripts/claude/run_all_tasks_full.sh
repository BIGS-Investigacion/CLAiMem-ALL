#!/bin/bash

# Run feature space consistency analysis for all tasks using FULL dataset

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate clam_latest

DATASET="data/histomorfologico"
OUTPUT="results/feature_space_analysis"
CACHE="cache/virchow_features"
BATCH_SIZE=32

echo "=========================================="
echo "Feature Space Consistency Analysis"
echo "Running all tasks on FULL dataset"
echo "Using conda environment: clam_latest"
echo "=========================================="
echo ""

# Array of tasks
TASKS=("her2" "er" "pr" "pam50")

for task in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing task: $task"
    echo "=========================================="
    echo ""

    python src/claude/feature_space_consistency_from_patches.py \
        --patches_dir "$DATASET" \
        --task "$task" \
        --output "$OUTPUT" \
        --cache_dir "$CACHE" \
        --batch_size "$BATCH_SIZE"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Task $task completed successfully"
    else
        echo ""
        echo "✗ Task $task failed"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT"
echo "Cache saved to: $CACHE"
