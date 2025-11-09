#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate clam_latest

echo "==================================="
echo "Processing PAM50 (5 subtypes)"
echo "==================================="

# Filter correct predictions for PAM50 (5 labels)
echo "Filtering correct predictions for label basal (0)..."
python filter_correct_predictions.py \
    data/histomorfologico/tcga/pam50/label_basal_pred_0 \
    data/dataset_csv/tcga-subtype_pam50.csv \
    --save-list correct_predictions_pam50_label_0.txt

echo ""
echo "Filtering correct predictions for label her2 (1)..."
python filter_correct_predictions.py \
    data/histomorfologico/tcga/pam50/label_her2_pred_1 \
    data/dataset_csv/tcga-subtype_pam50.csv \
    --save-list correct_predictions_pam50_label_1.txt

echo ""
echo "Filtering correct predictions for label luma (2)..."
python filter_correct_predictions.py \
    data/histomorfologico/tcga/pam50/label_luma_pred_2 \
    data/dataset_csv/tcga-subtype_pam50.csv \
    --save-list correct_predictions_pam50_label_2.txt

echo ""
echo "Filtering correct predictions for label lumb (3)..."
python filter_correct_predictions.py \
    data/histomorfologico/tcga/pam50/label_lumb_pred_3 \
    data/dataset_csv/tcga-subtype_pam50.csv \
    --save-list correct_predictions_pam50_label_3.txt

echo ""
echo "Filtering correct predictions for label normal (4)..."
python filter_correct_predictions.py \
    data/histomorfologico/tcga/pam50/label_normal_pred_4 \
    data/dataset_csv/tcga-subtype_pam50.csv \
    --save-list correct_predictions_pam50_label_4.txt

echo ""
echo "================================"
echo ""

# Select representative images for each label
echo "Selecting representatives for label 0 (basal)..."
python select_representative_images.py \
    data/histomorfologico/tcga/pam50/label_basal_pred_0 \
    correct_predictions_pam50_label_0.txt \
    --n-images 25 \
    --output selected_images_pam50_label_0.txt \
    --batch-size 8

echo ""
echo "================================"
echo ""

echo "Selecting representatives for label 1 (her2)..."
python select_representative_images.py \
    data/histomorfologico/tcga/pam50/label_her2_pred_1 \
    correct_predictions_pam50_label_1.txt \
    --n-images 25 \
    --output selected_images_pam50_label_1.txt \
    --batch-size 8

echo ""
echo "================================"
echo ""

echo "Selecting representatives for label 2 (luma)..."
python select_representative_images.py \
    data/histomorfologico/tcga/pam50/label_luma_pred_2 \
    correct_predictions_pam50_label_2.txt \
    --n-images 25 \
    --output selected_images_pam50_label_2.txt \
    --batch-size 8

echo ""
echo "================================"
echo ""

echo "Selecting representatives for label 3 (lumb)..."
python select_representative_images.py \
    data/histomorfologico/tcga/pam50/label_lumb_pred_3 \
    correct_predictions_pam50_label_3.txt \
    --n-images 25 \
    --output selected_images_pam50_label_3.txt \
    --batch-size 8

echo ""
echo "================================"
echo ""

echo "Selecting representatives for label 4 (normal)..."
python select_representative_images.py \
    data/histomorfologico/tcga/pam50/label_normal_pred_4 \
    correct_predictions_pam50_label_4.txt \
    --n-images 25 \
    --output selected_images_pam50_label_4.txt \
    --batch-size 8

echo ""
echo "Done! PAM50 results saved to:"
echo "  - selected_images_pam50_label_0.txt (basal)"
echo "  - selected_images_pam50_label_1.txt (her2)"
echo "  - selected_images_pam50_label_2.txt (luma)"
echo "  - selected_images_pam50_label_3.txt (lumb)"
echo "  - selected_images_pam50_label_4.txt (normal)"