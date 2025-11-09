#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate clam_latest

echo "==================================="
echo "Processing HER2 (ERBB2)"
echo "==================================="

# Filter correct predictions for HER2
echo "Filtering correct predictions for label 0..."
python filter_correct_predictions.py \
    data/histomorfologico/tcga/her2/label_negative_pred_0 \
    data/dataset_csv/tcga-erbb2.csv \
    --save-list correct_predictions_her2_label_0.txt

echo ""
echo "Filtering correct predictions for label 1..."
python filter_correct_predictions.py \
    data/histomorfologico/tcga/her2/label_positive_pred_1 \
    data/dataset_csv/tcga-erbb2.csv \
    --save-list correct_predictions_her2_label_1.txt

echo ""
echo "================================"
echo ""

# Select representative images
echo "Selecting representatives for label 0..."
python select_representative_images.py \
    data/histomorfologico/tcga/her2/label_negative_pred_0 \
    correct_predictions_her2_label_0.txt \
    --n-images 25 \
    --output selected_images_her2_label_0.txt \
    --batch-size 8

echo ""
echo "================================"
echo ""

echo "Selecting representatives for label 1..."
python select_representative_images.py \
    data/histomorfologico/tcga/her2/label_positive_pred_1 \
    correct_predictions_her2_label_1.txt \
    --n-images 25 \
    --output selected_images_her2_label_1.txt \
    --batch-size 8

echo ""
echo "Done! HER2 results saved to:"
echo "  - selected_images_her2_label_0.txt"
echo "  - selected_images_her2_label_1.txt"