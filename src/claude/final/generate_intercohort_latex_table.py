#!/usr/bin/env python3
"""
Generate LaTeX table for inter-cohort statistical tests (TCGA vs CPTAC)
"""

import json
import sys
from pathlib import Path

def format_pvalue(p):
    """Format p-value with significance stars"""
    if p < 0.001:
        return f"$< 0.001$***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"

def format_effect_ordinal(stat, rb, p):
    """Format U statistic with rank-biserial correlation"""
    return f"{stat:.1f} (${rb:+.2f}$)"

def format_effect_binary(stat, v, p):
    """Format Chi-square with Cramér's V"""
    return f"{stat:.1f} ({v:.2f})"

def generate_subtable(inter_cohort, classes, features, feature_mapping, is_ordinal, table_label, table_num):
    """Generate a single subtable for either ordinal or binary features (TRANSPOSED)"""

    num_features = len(features)
    latex = []

    # Table column specification: Class column + 2 columns per feature (stat + p-value)
    col_spec = "l" + "cc" * num_features
    latex.append(r"\begin{tabular}{" + col_spec + "}")
    latex.append(r"\toprule")

    # First header row: feature names spanning two columns each
    header1 = " & " + " & ".join([r"\multicolumn{2}{c}{\textbf{" + feature_mapping[f] + "}}" for f in features]) + r" \\"
    latex.append(header1)

    # Cmidrule for each feature
    cmidrules = []
    for i, _ in enumerate(features):
        start = 2 + i * 2
        end = start + 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    latex.append(" ".join(cmidrules))

    # Second header row: Test statistic and p-value
    if is_ordinal:
        header2 = r"\textbf{Class} & " + " & ".join([r"$U$ ($r_{rb}$)", r"$p$"] * num_features) + r" \\"
    else:
        header2 = r"\textbf{Class} & " + " & ".join([r"$\chi^2$ (V)", r"$p$"] * num_features) + r" \\"
    latex.append(header2)
    latex.append(r"\midrule")

    # Data rows: one row per class
    for class_name in classes:
        class_data = inter_cohort[class_name]
        row = [class_name.replace('_', r'\_')]

        for feature_key in features:
            if is_ordinal:
                # Ordinal feature: Mann-Whitney U
                stats = class_data['ordinal_features'][feature_key]
                u_stat = stats['statistic']
                rb = stats['rank_biserial']
                p = stats['p_value']

                row.append(format_effect_ordinal(u_stat, rb, p))
                row.append(format_pvalue(p))
            else:
                # Binary feature: Chi-square
                stats = class_data['binary_features'][feature_key]
                chi2 = stats['statistic']
                v = stats['cramers_v']
                p = stats['p_value']

                row.append(format_effect_binary(chi2, v, p))
                row.append(format_pvalue(p))

        latex.append(" & ".join(row) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")

    return "\n".join(latex)


def generate_latex_table(json_file, output_file=None):
    """Generate LaTeX tables from biological interpretability JSON results"""

    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)

    inter_cohort = data['inter_cohort']['per_class_comparisons']

    # Define feature names (mapping from JSON to display names)
    feature_mapping = {
        'ESTRUCTURA GLANDULAR': 'Tubule Formation',
        'ATIPIA NUCLEAR': 'Nuclear Pleomorphism',
        'MITOSIS': 'Mitotic Activity',
        'NECROSIS': 'Tumour Necrosis',
        'INFILTRADO_LI': 'Lymphocytic Infiltrate',
        'INFILTRADO_PMN': 'Polymorphonuclear Infiltrate'
    }

    # Define class order (for columns)
    class_order = [
        'BASAL', 'HER2-enriched', 'LUMINAL-A', 'LUMINAL-B', 'NORMAL-like',
        'ER-negative', 'ER-positive',
        'PR-negative', 'PR-positive',
        'HER2-negative', 'HER2-positive'
    ]

    # Filter to only classes present in data
    classes = [c for c in class_order if c in inter_cohort]

    # Features in order
    ordinal_features = ['ESTRUCTURA GLANDULAR', 'ATIPIA NUCLEAR', 'MITOSIS']
    binary_features = ['NECROSIS', 'INFILTRADO_LI', 'INFILTRADO_PMN']

    # Build complete table with two subtables
    latex = []

    # Table 1: Ordinal features
    latex.append(r"\begin{sidewaystable}[h!]")
    latex.append(r"\centering")
    latex.append(r"\caption{Statistical tests comparing ordinal histomorphological features between TCGA and CPTAC cohorts across molecular classes. " +
                 r"Mann-Whitney U test with rank-biserial correlation ($r_{rb}$). " +
                 r"* $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$.}")
    latex.append(r"\label{tab:inter-cohort-ordinal}")
    latex.append(r"\small")

    # Generate ordinal subtable
    ordinal_table = generate_subtable(inter_cohort, classes, ordinal_features,
                                      feature_mapping, is_ordinal=True,
                                      table_label="ordinal", table_num=1)
    latex.append(ordinal_table)
    latex.append(r"\end{sidewaystable}")

    latex.append("")
    latex.append("")

    # Table 2: Binary features
    latex.append(r"\begin{sidewaystable}[h!]")
    latex.append(r"\centering")
    latex.append(r"\caption{Statistical tests comparing binary histomorphological features between TCGA and CPTAC cohorts across molecular classes. " +
                 r"Chi-square test ($\chi^2$) with Cramér's V. " +
                 r"* $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$.}")
    latex.append(r"\label{tab:inter-cohort-binary}")
    latex.append(r"\small")

    # Generate binary subtable
    binary_table = generate_subtable(inter_cohort, classes, binary_features,
                                     feature_mapping, is_ordinal=False,
                                     table_label="binary", table_num=2)
    latex.append(binary_table)
    latex.append(r"\end{sidewaystable}")

    # Join and return
    table_latex = "\n".join(latex)

    # Save or print
    if output_file:
        with open(output_file, 'w') as f:
            f.write(table_latex)
        print(f"LaTeX tables saved to: {output_file}")
    else:
        print(table_latex)

    return table_latex

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_intercohort_latex_table.py <json_file> [output_file]")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    generate_latex_table(json_file, output_file)
