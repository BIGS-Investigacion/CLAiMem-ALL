#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate clam_latest

echo "==================================="
echo "Processing by GROUND TRUTH labels"
echo "==================================="

# ER
echo ""
echo "Processing ER dataset..."
python select_by_ground_truth.py \
    data/histomorfologico/tcga/er \
    data/dataset_csv/tcga-er.csv \
    --n-images 25 \
    --output-prefix selected_gt_er \
    --batch-size 8

# HER2
echo ""
echo "Processing HER2 dataset..."
python select_by_ground_truth.py \
    data/histomorfologico/tcga/her2 \
    data/dataset_csv/tcga-erbb2.csv \
    --n-images 25 \
    --output-prefix selected_gt_her2 \
    --batch-size 8

# PR
echo ""
echo "Processing PR dataset..."
python select_by_ground_truth.py \
    data/histomorfologico/tcga/pr \
    data/dataset_csv/tcga-pr.csv \
    --n-images 25 \
    --output-prefix selected_gt_pr \
    --batch-size 8

# PAM50
echo ""
echo "Processing PAM50 dataset..."
python select_by_ground_truth.py \
    data/histomorfologico/tcga/pam50 \
    data/dataset_csv/tcga-subtype_pam50.csv \
    --n-images 25 \
    --output-prefix selected_gt_pam50 \
    --batch-size 8

echo ""
echo "==================================="
echo "All datasets processed!"
echo "==================================="