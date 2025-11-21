#!/bin/bash

# Create a test subset of the histomorfologico data
# Selects 50 random patches from each subdirectory

SOURCE_DIR="data/histomorfologico"
TEST_DIR="data/histomorfologico_test"
N_PATCHES=50

echo "Creating test dataset: $TEST_DIR"
echo "Selecting $N_PATCHES random patches from each subdirectory"
echo "=========================================="

# Remove existing test directory if it exists
if [ -d "$TEST_DIR" ]; then
    echo "Removing existing test directory..."
    rm -rf "$TEST_DIR"
fi

# Create test directory structure
mkdir -p "$TEST_DIR"

# Find all subdirectories containing PNG files
# Structure: data/histomorfologico/{cohort}/{task}/{label_pred_folders}/
for cohort_dir in "$SOURCE_DIR"/*; do
    if [ ! -d "$cohort_dir" ]; then
        continue
    fi

    cohort=$(basename "$cohort_dir")
    echo ""
    echo "Processing cohort: $cohort"

    for task_dir in "$cohort_dir"/*; do
        if [ ! -d "$task_dir" ]; then
            continue
        fi

        task=$(basename "$task_dir")
        echo "  Processing task: $task"

        for label_dir in "$task_dir"/*; do
            if [ ! -d "$label_dir" ]; then
                continue
            fi

            label=$(basename "$label_dir")

            # Count total patches
            total_patches=$(find "$label_dir" -maxdepth 1 -name "*.png" | wc -l)

            if [ "$total_patches" -eq 0 ]; then
                echo "    Skipping $label (no patches found)"
                continue
            fi

            # Create target directory
            target_dir="$TEST_DIR/$cohort/$task/$label"
            mkdir -p "$target_dir"

            # Select random patches
            if [ "$total_patches" -le "$N_PATCHES" ]; then
                # If total patches <= N_PATCHES, copy all
                echo "    Copying all $total_patches patches from $label"
                find "$label_dir" -maxdepth 1 -name "*.png" -exec cp {} "$target_dir/" \;
            else
                # Otherwise, randomly select N_PATCHES
                echo "    Selecting $N_PATCHES/$total_patches random patches from $label"
                find "$label_dir" -maxdepth 1 -name "*.png" | shuf -n "$N_PATCHES" | while read patch; do
                    cp "$patch" "$target_dir/"
                done
            fi

            # Verify
            copied=$(find "$target_dir" -maxdepth 1 -name "*.png" | wc -l)
            echo "      ✓ Copied $copied patches to $target_dir"
        done
    done
done

echo ""
echo "=========================================="
echo "Test dataset created successfully!"
echo ""
echo "Summary:"
find "$TEST_DIR" -type d -name "label_*" | while read dir; do
    n_patches=$(find "$dir" -maxdepth 1 -name "*.png" | wc -l)
    echo "  $(echo $dir | sed "s|$TEST_DIR/||"): $n_patches patches"
done

echo ""
echo "Total patches in test dataset:"
total=$(find "$TEST_DIR" -name "*.png" | wc -l)
echo "  $total patches"
