import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# === Datos originales ===
data = {
    "LABEL": ["BASAL"] * 5 + ["HER2-enriched"] * 5 + ["LUMINAL-A"] * 5 +
                ["LUMINAL-B"] * 5 + ["NORMAL-like"] * 5,
    "TUBULE_FORMATION_DEGREE": [4,4,4,4,4, 3,2,2,2,1, 2,2,3,2,3,
                             3,2,2,1,2, 1,0,1,1,1],
    "CELLULAR_POLYMORPHISM": [4,4,4,4,4, 3,3,3,3,3, 1,1,2,2,3,
                       3,1,2,1,1, 0,0,0,0,1],
    "MITOTIC_RATE": [0,0,2,3,2, 0,0,0,0,0, 0,0,1,1,0,
                0,0,1,0,0, 1,0,1,0,0],
    "TOMOUR_NECROSIS": [1]*5 + [1,0,1,0,0] + [0,0,1,1,1] +
                [0,0,1,0,0] + [0]*5,
    "LYMPHOCYTE_INFILTRATION": [1,1,0,1,1,
                                            1,0,0,1,1, 0,0,0,1,1,
                                            1,0,0,0,0,
                                            0,0,0,0,0]
}


df = pd.DataFrame(data)
variables = ["TUBULE_FORMATION_DEGREE","CELLULAR_POLYMORPHISM","MITOTIC_RATE","TOMOUR_NECROSIS","LYMPHOCYTE_INFILTRATION"]
etiquetas = sorted(set(df["LABEL"]))

# === Tomar 5 valores por variable y grupo, rellenando con NaN si faltan ===
filtered_data = []
max_vals = 5

for etiqueta in etiquetas:
    indices = [i for i, x in enumerate(data["LABEL"]) if x == etiqueta]
    row = {"LABEL": [etiqueta]*max_vals}
    for var in variables:
        values = [data[var][i] for i in indices]
        if len(values) < max_vals:
            values += [np.nan] * (max_vals - len(values))
        else:
            values = values[:max_vals]
        row[var] = values
    df_partial = pd.DataFrame(row)
    filtered_data.append(df_partial)

df_new = pd.concat(filtered_data, ignore_index=True)

# === ANOVA + Tukey por variable ===
tukey_results = {}
for column in variables:
    model = ols(f'{column} ~ C(LABEL)', data=df_new).fit()
    print(sm.stats.anova_lm(model, typ=2))
    tukey = pairwise_tukeyhsd(endog=df_new[column], groups=df_new['LABEL'], alpha=0.05)
    result_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_results[column] = result_df
print(tukey_results)
# === Calcular matriz de menor p-valor ===
groups = sorted(df_new['LABEL'].unique())
combined_matrix = pd.DataFrame(np.nan, index=groups, columns=groups)

for g1, g2 in combinations(groups, 2):
    min_p = 1.0
    for result_df in tukey_results.values():
        match = result_df[((result_df['group1'] == g1) & (result_df['group2'] == g2)) |
                          ((result_df['group1'] == g2) & (result_df['group2'] == g1))]
        if not match.empty:
            pval = float(match['p-adj'].values[0])
            min_p = min(min_p, pval)
    combined_matrix.loc[g1, g2] = min_p
    combined_matrix.loc[g2, g1] = min_p

# === Graficar heatmap ===
mask = combined_matrix >= 0.05

plt.figure(figsize=(8, 6))
sns.heatmap(combined_matrix, annot=False, cmap="Reds_r", vmin=0, vmax=0.05,
            mask=mask, cbar_kws={'label': 'minimum p-value in statistical tests'})
plt.title("")
plt.tight_layout()
plt.show()
