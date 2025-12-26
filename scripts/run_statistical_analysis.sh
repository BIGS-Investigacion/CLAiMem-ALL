#!/bin/bash
################################################################################
# FINAL ANALYSIS PIPELINE
#
# This script runs the complete final analysis pipeline including:
# 1. Multivariate regression analysis (domain shift predictors)
# 2. Inter-cohort LaTeX table generation (TCGA vs CPTAC)
# 3. Intra-cohort LaTeX table generation (split tables)
#
# Usage: bash scripts/claude/run_final_analysis.sh
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "FINAL ANALYSIS PIPELINE"
echo "================================================================================"
echo ""

# Define output directories
OUTPUT_DIR="results/final_analysis"
TABLES_DIR="${OUTPUT_DIR}/tables"
FIGURES_DIR="${OUTPUT_DIR}/figures"

# Create output directories
echo "Creating output directories..."
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TABLES_DIR}"
mkdir -p "${FIGURES_DIR}"
echo "  ✓ Created: ${OUTPUT_DIR}"
echo "  ✓ Created: ${TABLES_DIR}"
echo "  ✓ Created: ${FIGURES_DIR}"
echo ""

# ============================================================================
# STEP 1: MULTIVARIATE REGRESSION ANALYSIS
# ============================================================================

echo "================================================================================"
echo "STEP 1: MULTIVARIATE REGRESSION ANALYSIS"
echo "================================================================================"
echo ""
echo "Running domain shift multivariate analysis..."
echo "Script: src/claude/final/compare_linear_regressions.py"
echo ""

python src/claude/final/compare_linear_regressions.py > "${OUTPUT_DIR}/multivariate_regression_results.txt"

if [ $? -eq 0 ]; then
    echo "  ✓ Multivariate regression completed successfully"
    echo "  ✓ Results saved to: ${OUTPUT_DIR}/multivariate_regression_results.txt"
else
    echo "  ✗ Error in multivariate regression"
    exit 1
fi
echo ""

# ============================================================================
# STEP 2: INTER-COHORT LATEX TABLES (TCGA vs CPTAC)
# ============================================================================

echo "================================================================================"
echo "STEP 2: INTER-COHORT LATEX TABLES (TCGA vs CPTAC)"
echo "================================================================================"
echo ""
echo "Generating LaTeX tables for inter-cohort comparisons..."
echo "Script: src/claude/final/generate_intercohort_latex_table.py"
echo ""

# Check if biological analysis results exist
BIOLOGICAL_JSON="results/biological_analysis/pam50_biological_interpretability.json"

if [ ! -f "${BIOLOGICAL_JSON}" ]; then
    echo "  ⚠ Biological analysis results not found: ${BIOLOGICAL_JSON}"
    echo "  Running biological interpretability analysis first..."

    python src/claude/biological_interpretability_analysis.py \
        -a data/histomorfologico/representative_images_annotation.xlsx \
        -t pam50 \
        -o results/biological_analysis/

    if [ $? -eq 0 ]; then
        echo "  ✓ Biological analysis completed"
    else
        echo "  ✗ Error in biological analysis"
        exit 1
    fi
fi

# Generate inter-cohort tables
python src/claude/final/generate_intercohort_latex_table.py \
    "${BIOLOGICAL_JSON}" \
    "${TABLES_DIR}/intercohort_tables.tex"

if [ $? -eq 0 ]; then
    echo "  ✓ Inter-cohort LaTeX tables generated"
    echo "  ✓ Saved to: ${TABLES_DIR}/intercohort_tables.tex"
else
    echo "  ✗ Error generating inter-cohort tables"
    exit 1
fi
echo ""

# ============================================================================
# STEP 3: INTRA-COHORT LATEX TABLES (TCGA split tables)
# ============================================================================

echo "================================================================================"
echo "STEP 3: INTRA-COHORT LATEX TABLES (TCGA split tables)"
echo "================================================================================"
echo ""
echo "Generating split LaTeX tables for intra-cohort analysis..."
echo "Script: src/claude/final/split_intra_cohort_table.py"
echo ""

python src/claude/final/split_intra_cohort_table.py \
    "${TABLES_DIR}/intra_cohort_split_tables.tex"

if [ $? -eq 0 ]; then
    echo "  ✓ Intra-cohort split tables generated"
    echo "  ✓ Saved to: ${TABLES_DIR}/intra_cohort_split_tables.tex"
else
    echo "  ✗ Error generating intra-cohort tables"
    exit 1
fi
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "================================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Output files:"
echo "  • Multivariate regression results: ${OUTPUT_DIR}/multivariate_regression_results.txt"
echo "  • Inter-cohort tables (TCGA vs CPTAC): ${TABLES_DIR}/intercohort_tables.tex"
echo "  • Intra-cohort split tables: ${TABLES_DIR}/intra_cohort_split_tables.tex"
echo ""
echo "All results saved to: ${OUTPUT_DIR}"
echo ""
echo "================================================================================"
