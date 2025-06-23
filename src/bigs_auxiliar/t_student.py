from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns

data = {
    "LABEL": ["HER2_POSITIVE"] * 5 + ["HER2_NEGATIVE"] * 5,
    "TUBULE_FORMATION_DEGREE": [3, 2, 2, 3, 3, 3, 3, 2, 2, 2],
    "CELLULAR_POLYMORPHISM": [3, 3, 2, 2, 3, 4, 4, 3, 2, 3],
    "MITOTIC_RATE": [0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    "TOMOUR_NECROSIS": [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    "LYMPHOCYTE_INFILTRATION": [1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
}
'''
data = {
    "LABEL": ["ER_POSITIVE"] * 5 + ["ER_NEGATIVE"] * 5,
    "TUBULE_FORMATION_DEGREE": [1, 3, 3, 3, 4, 1, 2, 2, 2, 2],
    "CELLULAR_POLYMORPHISM": [2, 3, 4, 4, 4, 2, 2, 1, 2, 3],
    "MITOTIC_RATE": [1, 1, 2, 0, 2, 1, 0, 0, 0, 0],
    "TOMOUR_NECROSIS": [1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    "LYMPHOCYTE_INFILTRATION": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
}
data = {
    "LABEL": ["PR_POSITIVE"] * 5 + ["PR_NEGATIVE"] * 5,
    "TUBULE_FORMATION_DEGREE": [4, 4, 2, 4, 3, 2, 3, 3, 2, 2],
    "CELLULAR_POLYMORPHISM": [4, 4, 3, 4, 4, 1, 2, 2, 1, 2],
    "MITOTIC_RATE": [0, 2, 0, 0, 0, 0, 1, 0, 0, 0],
    "TOMOUR_NECROSIS": [1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    "LYMPHOCYTE_INFILTRATION": [1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
}
'''

df = pd.DataFrame(data)


# Separar los grupos
group_pos = df[df["LABEL"] == "HER2_POSITIVE"]
group_neg = df[df["LABEL"] == "HER2_NEGATIVE"]

# Variables a comparar
variables = ["TUBULE_FORMATION_DEGREE", "CELLULAR_POLYMORPHISM", "MITOTIC_RATE",
             "TOMOUR_NECROSIS", "LYMPHOCYTE_INFILTRATION"]

# Calcular p-values
p_values = {}
for var in variables:
    stat, p = ttest_ind(group_pos[var], group_neg[var], equal_var=False)
    p_values[var] = p

# Mostrar resultados
for var, p in p_values.items():
    print(f"{var}: p-value = {p:.4f}")


