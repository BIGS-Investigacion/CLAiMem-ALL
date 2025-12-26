#!/bin/bash
################################################################################
# DISTRIBUTIONAL SHIFT ANALYSIS - ALL TASKS
#
# This script runs distributional shift analysis for all tasks:
# - PAM50 (5 classes)
# - ER (2 classes)
# - PR (2 classes)
# - HER2 (2 classes)
#
# Generates:
# - Prevalence shift metrics (Δp with sign)
# - Performance degradation metrics (RPC)
# - Correlation analysis (Δp vs RPC)
# - Visualizations (bar charts, scatter plots)
# - Summary tables (CSV and LaTeX)
#
# Usage: bash scripts/claude/run_distributional_shift_analysis.sh [output_dir]
################################################################################

set -e  # Exit on error

# Default output directory
OUTPUT_DIR="${1:-results/distributional_shift}"

echo "================================================================================"
echo "DISTRIBUTIONAL SHIFT ANALYSIS - ALL TASKS"
echo "================================================================================"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run the analysis
echo "Running analysis for all tasks (PAM50, ER, PR, HER2)..."
echo ""

python src/claude/final/run_all_distributional_shift.py --output "${OUTPUT_DIR}"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ ANALYSIS COMPLETED SUCCESSFULLY"
    echo "================================================================================"
    echo ""
    echo "Results saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Key files:"
    echo "  • distributional_shift_summary.csv - Summary metrics for all tasks"
    echo "  • distributional_shift_summary_table.tex - LaTeX summary table"
    echo ""
    echo "Per-task files (pam50, er, pr, her2):"
    echo "  • {task}_prevalence_shift.csv - Prevalence data"
    echo "  • {task}_performance_degradation.csv - Performance metrics"
    echo "  • {task}_distributional_shift_analysis.json - Complete results"
    echo "  • {task}_*.png - Visualizations"
    echo ""
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "✗ ANALYSIS FAILED"
    echo "================================================================================"
    exit 1
fi
