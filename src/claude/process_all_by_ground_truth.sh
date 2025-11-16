#!/bin/bash

# Usage: bash process_all_by_ground_truth.sh <database>
# Example: bash process_all_by_ground_truth.sh tcga
# Example: bash process_all_by_ground_truth.sh cptac

if [ -z "$1" ]; then
    echo "Error: Database name required"
    echo "Usage: bash process_all_by_ground_truth.sh <database>"
    echo "Example: bash process_all_by_ground_truth.sh tcga"
    echo "Example: bash process_all_by_ground_truth.sh cptac"
    exit 1
fi

DATABASE=$1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate clam_latest

echo "==================================="
echo "Processing by GROUND TRUTH labels"
echo "Database: $DATABASE"
echo "==================================="

# ER
echo ""
echo "Processing ER dataset..."
python select_by_ground_truth.py \
    data/histomorfologico/${DATABASE}/er \
    data/dataset_csv/${DATABASE}-er.csv \
    --n-images 25 \
    --output-prefix selected_gt_${DATABASE}_er \
    --batch-size 8

# HER2
echo ""
echo "Processing HER2 dataset..."
python select_by_ground_truth.py \
    data/histomorfologico/${DATABASE}/her2 \
    data/dataset_csv/${DATABASE}-erbb2.csv \
    --n-images 25 \
    --output-prefix selected_gt_${DATABASE}_her2 \
    --batch-size 8

# PR
echo ""
echo "Processing PR dataset..."
python select_by_ground_truth.py \
    data/histomorfologico/${DATABASE}/pr \
    data/dataset_csv/${DATABASE}-pr.csv \
    --n-images 25 \
    --output-prefix selected_gt_${DATABASE}_pr \
    --batch-size 8

# PAM50
echo ""
echo "Processing PAM50 dataset..."
python select_by_ground_truth.py \
    data/histomorfologico/${DATABASE}/pam50 \
    data/dataset_csv/${DATABASE}-subtype_pam50.csv \
    --n-images 25 \
    --output-prefix selected_gt_${DATABASE}_pam50 \
    --batch-size 8

echo ""
echo "==================================="
echo "All datasets processed for $DATABASE!"
echo "==================================="