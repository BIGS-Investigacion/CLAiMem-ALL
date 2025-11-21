#!/bin/bash

# Test feature space consistency analysis with test dataset

echo "Testing Feature Space Consistency Analysis"
echo "=========================================="
echo ""

# Run with test dataset
python src/claude/feature_space_consistency_from_patches.py \
    --base-dir data/histomorfologico_test \
    --task her2 \
    --output-dir results/feature_space_analysis_test \
    --batch-size 32

echo ""
echo "=========================================="
echo "Test complete!"
echo ""
echo "Check results in: results/feature_space_analysis_test/"
