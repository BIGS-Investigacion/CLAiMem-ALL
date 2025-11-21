#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate clam_latest

# Run selection for label 0 (negative)
echo "Processing label_negative_pred_0..."
python select_representative_images.py \
    data/histomorfologico/tcga/er/label_negative_pred_0 \
    correct_predictions_er_label_0.txt \
    --n-images 25 \
    --output selected_images_er_label_0.txt \
    --batch-size 8

echo ""
echo "================================"
echo ""

# Run selection for label 1 (positive)
echo "Processing label_positive_pred_1..."
python select_representative_images.py \
    data/histomorfologico/tcga/er/label_positive_pred_1 \
    correct_predictions_er_label_1.txt \
    --n-images 25 \
    --output selected_images_er_label_1.txt \
    --batch-size 8

echo ""
echo "Done! Results saved to:"
echo "  - selected_images_er_label_0.txt"
echo "  - selected_images_er_label_1.txt"