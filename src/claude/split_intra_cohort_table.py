#!/usr/bin/env python3
"""
Split intra-cohort TCGA table into two subtables
"""

def generate_split_tables():
    """Generate two subtables from the original intra-cohort table"""

    # Data from the original table
    features = [
        'Tubule Formation',
        'Nuclear Pleomorphism',
        'Mitotic Activity',
        'Tumour Necrosis',
        'Lymphocytic Infiltrate',
        'Polymorphonuclear Infiltrate'
    ]

    # Data: [PAM50_stat, PAM50_p, ER_stat, ER_p, PR_stat, PR_p, HER2_stat, HER2_p]
    data = {
        'Tubule Formation': ['74.6 (0.588)', '$< 0.001$***', '151.0 ($-0.81$)', '$< 0.001$***',
                            '123.5 ($-0.84$)', '$< 0.001$***', '372.5 ($-0.30$)', '0.131'],
        'Nuclear Pleomorphism': ['98.2 (0.785)', '$< 0.001$***', '42.5 ($-0.92$)', '$< 0.001$***',
                                '48.0 ($-0.90$)', '$< 0.001$***', '454.5 ($-0.45$)', '$< 0.01$**'],
        'Mitotic Activity': ['30.0 (0.216)', '$< 0.001$***', '209.5 ($-0.58$)', '$< 0.05$*',
                           '184.5 ($-0.63$)', '$< 0.01$**', '381.5 ($-0.35$)', '0.105'],
        'Tumour Necrosis': ['88.9 (0.708)', '$< 0.001$***', '125.0 ($-0.80$)', '$< 0.001$***',
                          '25.0 ($-0.92$)', '$< 0.001$***', '450.0 ($-0.44$)', '$< 0.01$**'],
        'Lymphocytic Infiltrate': ['26.9 (0.191)', '$< 0.001$***', '412.5 ($-0.48$)', '$< 0.01$**',
                                  '200.0 ($-0.60$)', '$< 0.01$**', '350.0 ($-0.32$)', '0.317'],
        'Polymorphonuclear Infiltrate': ['74.3 (0.588)', '$< 0.001$***', '125.0 ($-0.80$)', '$< 0.001$***',
                                        '25.0 ($-0.92$)', '$< 0.001$***', '350.0 ($-0.32$)', '0.406']
    }

    latex = []

    # ========================================================================
    # Table 1: PAM50 and ER
    # ========================================================================
    latex.append(r"\begin{table}[h!]")
    latex.append(r"\centering")
    latex.append(r"\caption{Statistical tests for histomorphological features in TCGA-BRCA: PAM50 and ER status. " +
                 r"PAM50: Kruskal-Wallis test (H) with epsilon-squared ($\varepsilon^2$). " +
                 r"ER: Mann-Whitney U test with rank-biserial correlation ($r_{rb}$). " +
                 r"* $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$.}")
    latex.append(r"\label{tab:intra-cohort-pam50-er}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r" & \multicolumn{2}{c}{\textbf{PAM50}} & \multicolumn{2}{c}{\textbf{ER}} \\")
    latex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    latex.append(r"\textbf{Feature} & $H$ ($\varepsilon^2$) & $p$ & $U$ ($r_{rb}$) & $p$ \\")
    latex.append(r"\midrule")

    for feature in features:
        values = data[feature]
        # PAM50: indices 0,1; ER: indices 2,3
        row = f"{feature} & {values[0]} & {values[1]} & {values[2]} & {values[3]} " + r"\\"
        latex.append(row)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex.append("")
    latex.append("")

    # ========================================================================
    # Table 2: PR and HER2
    # ========================================================================
    latex.append(r"\begin{table}[h!]")
    latex.append(r"\centering")
    latex.append(r"\caption{Statistical tests for histomorphological features in TCGA-BRCA: PR and HER2 status. " +
                 r"Mann-Whitney U test with rank-biserial correlation ($r_{rb}$). " +
                 r"* $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$.}")
    latex.append(r"\label{tab:intra-cohort-pr-her2}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r" & \multicolumn{2}{c}{\textbf{PR}} & \multicolumn{2}{c}{\textbf{HER2}} \\")
    latex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    latex.append(r"\textbf{Feature} & $U$ ($r_{rb}$) & $p$ & $U$ ($r_{rb}$) & $p$ \\")
    latex.append(r"\midrule")

    for feature in features:
        values = data[feature]
        # PR: indices 4,5; HER2: indices 6,7
        row = f"{feature} & {values[4]} & {values[5]} & {values[6]} & {values[7]} " + r"\\"
        latex.append(row)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


if __name__ == "__main__":
    import sys

    output_file = sys.argv[1] if len(sys.argv) > 1 else None

    table_latex = generate_split_tables()

    if output_file:
        with open(output_file, 'w') as f:
            f.write(table_latex)
        print(f"LaTeX tables saved to: {output_file}")
    else:
        print(table_latex)
