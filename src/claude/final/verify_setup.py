#!/usr/bin/env python3
"""
Verification script to check if all dependencies and data files are available
for the final analysis pipeline.
"""

import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    path = Path(filepath)
    if path.exists():
        print(f"  ✓ {description}: {filepath}")
        return True
    else:
        print(f"  ✗ MISSING {description}: {filepath}")
        return False

def check_module(module_name):
    """Check if a Python module is available"""
    try:
        __import__(module_name)
        print(f"  ✓ {module_name}")
        return True
    except ImportError:
        print(f"  ✗ MISSING {module_name}")
        return False

def main():
    print("="*80)
    print("FINAL ANALYSIS PIPELINE - SETUP VERIFICATION")
    print("="*80)
    print()

    all_ok = True

    # Check Python modules
    print("Checking Python dependencies...")
    modules = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn']
    for module in modules:
        if not check_module(module):
            all_ok = False
    print()

    # Check scripts
    print("Checking analysis scripts...")
    scripts = [
        ('src/claude/final/compare_linear_regressions.py', 'Multivariate regression script'),
        ('src/claude/final/generate_intercohort_latex_table.py', 'Inter-cohort table generator'),
        ('src/claude/final/split_intra_cohort_table.py', 'Intra-cohort table splitter'),
        ('scripts/claude/run_final_analysis.sh', 'Pipeline launcher script')
    ]
    for filepath, desc in scripts:
        if not check_file_exists(filepath, desc):
            all_ok = False
    print()

    # Check data files (optional - will be generated if missing)
    print("Checking data files (will be generated if missing)...")
    data_files = [
        ('data/histomorfologico/representative_images_annotation.xlsx', 'Biological annotations'),
        ('results/biological_analysis/pam50_biological_interpretability.json', 'Biological analysis results (optional)')
    ]
    for filepath, desc in data_files:
        check_file_exists(filepath, desc)
    print()

    # Check output directories
    print("Checking output directories...")
    dirs = ['results', 'results/biological_analysis', 'results/final_analysis']
    for dirpath in dirs:
        path = Path(dirpath)
        if path.exists():
            print(f"  ✓ {dirpath}")
        else:
            print(f"  ℹ Creating {dirpath}")
            path.mkdir(parents=True, exist_ok=True)
    print()

    # Final verdict
    print("="*80)
    if all_ok:
        print("✓ SETUP VERIFICATION PASSED")
        print()
        print("You can now run the pipeline:")
        print("  bash scripts/claude/run_final_analysis.sh")
    else:
        print("✗ SETUP VERIFICATION FAILED")
        print()
        print("Please install missing dependencies or fix missing files.")
        sys.exit(1)
    print("="*80)

if __name__ == "__main__":
    main()
