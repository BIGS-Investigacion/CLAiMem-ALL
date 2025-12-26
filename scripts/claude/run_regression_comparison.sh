#!/bin/bash

# Run Linear Regression Comparison Analysis
# Usage: bash scripts/claude/run_regression_comparison.sh

echo "================================================================================"
echo "LINEAR REGRESSION COMPARISON ANALYSIS"
echo "================================================================================"
echo ""

python src/claude/compare_linear_regressions.py

echo ""
echo "Results saved to: results/regression_comparison/"
echo ""
