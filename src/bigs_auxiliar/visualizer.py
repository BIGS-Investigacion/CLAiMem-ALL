import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Datos extraídos de las tablas LaTeX (PATIENT_STRAT)
tasks_data_full_ho = {
    "ER": [
        ["CNN", 0.821, 0.760, 0.837, 0.792],
        ["CONCH", 0.879, 0.839, 0.909, 0.866],
        ["CTRANSPATH", 0.879, 0.818, 0.914, 0.843],
        ["HIBOU-B", 0.890, 0.836, 0.910, 0.869],
        ["HIBOU-L", 0.882, 0.828, 0.912, 0.866],
        ["HOPTIMUS-0", 0.915, 0.866, 0.922, 0.887],
        ["MUSK", 0.871, 0.815, 0.901, 0.853],
        ["PHIKON", 0.905, 0.863, 0.921, 0.872],
        ["PROVGIGAPATH", 0.898, 0.853, 0.925, 0.884],
        ["RETCCL", 0.872, 0.807, 0.898, 0.837],
        ["UNI", 0.901, 0.858, 0.932, 0.889],
        ["UNI2", 0.898, 0.846, 0.933, 0.897],
        ["VIRCHOW", 0.905, 0.861, 0.910, 0.869],
    ],
    "PR": [
        ["CNN", 0.751, 0.743, 0.779, 0.759],
        ["CONCH", 0.779, 0.762, 0.810, 0.789],
        ["CTRANSPATH", 0.770, 0.759, 0.802, 0.781],
        ["HIBOU-B", 0.756, 0.746, 0.798, 0.792],
        ["HIBOU-L", 0.767, 0.756, 0.790, 0.764],
        ["HOPTIMUS-0", 0.822, 0.803, 0.829, 0.808],
        ["MUSK", 0.764, 0.753, 0.800, 0.789],
        ["PHIKON", 0.789, 0.782, 0.817, 0.802],
        ["PROVGIGAPATH", 0.798, 0.788, 0.813, 0.799],
        ["RETCCL", 0.768, 0.745, 0.787, 0.760],
        ["UNI", 0.795, 0.786, 0.824, 0.807],
        ["UNI2", 0.804, 0.786, 0.827, 0.813],
        ["VIRCHOW", 0.793, 0.783, 0.838, 0.823],
    ],
    "ERBB2": [
        ["CNN", 0.695, 0.603, 0.734, 0.640],
        ["CONCH", 0.698, 0.595, 0.746, 0.646],
        ["CTRANSPATH", 0.759, 0.642, 0.767, 0.673],
        ["HIBOU-B", 0.716, 0.612, 0.757, 0.659],
        ["HIBOU-L", 0.617, 0.549, 0.695, 0.594],
        ["HOPTIMUS-0", 0.743, 0.629, 0.759, 0.653],
        ["MUSK", 0.722, 0.616, 0.759, 0.656],
        ["PHIKON", 0.727, 0.616, 0.795, 0.708],
        ["PROVGIGAPATH", 0.757, 0.631, 0.773, 0.670],
        ["RETCCL", 0.732, 0.622, 0.748, 0.654],
        ["UNI", 0.752, 0.626, 0.795, 0.699],
        ["UNI2", 0.730, 0.621, 0.783, 0.682],
        ["VIRCHOW", 0.722, 0.635, 0.772, 0.670]
    ],
    "PAM-50":[ 
            ["CNN",0.530,0.218,0.587,0.310,0.540,0.226,0.633,0.319],
    ["CONCH",0.579,0.335,0.796,0.442,0.532,0.328,0.768,0.413],
    ["CTRANSPATH",0.584,0.342,0.777,0.415,0.574,0.330,0.768,0.418],
    ["HIBOU-B",0.548,0.289,0.711,0.348,0.481,0.307,0.683,0.350],
    ["HIBOU-L",0.530,0.297,0.696,0.353,0.525,0.291,0.672,0.347],
    ["HOPTIMUS-0",0.612,0.304,0.796,0.450,0.587,0.309,0.787,0.458],
    ["MUSK",0.592,0.305,0.735,0.367,0.556,0.339,0.745,0.377],
    ["PHIKON",0.605,0.345,0.730,0.411,0.633,0.355,0.770,0.477],
    ["PROVGIGAPATH",0.677,0.379,0.769,0.458,0.656,0.356,0.782,0.477],
    ["RETCCL",0.587,0.272,0.697,0.358,0.592,0.274,0.697,0.363],
    ["UNI",0.643,0.365,0.792,0.453,0.633,0.304,0.778,0.421],
    ["UNI2",0.607,0.325,0.766,0.447,0.612,0.308,0.780,0.452],
    ["VIRCHOW",0.661,0.358,0.813,0.498,0.630,0.378,0.774,0.462],
    ]
}

# Datos por tarea y modelo (solo PATIENT_STRAT, resumen de 4 tareas)
tasks_data_full_cv = {
    "ER": [
        ["CNN", 0.821, 0.760, 0.837, 0.792],
        ["CONCH", 0.879, 0.839, 0.909, 0.866],
        ["CTRANSPATH", 0.879, 0.818, 0.914, 0.843],
        ["HIBOU-B", 0.890, 0.836, 0.910, 0.869],
        ["HIBOU-L", 0.882, 0.828, 0.912, 0.866],
        ["HOPTIMUS-0", 0.915, 0.866, 0.922, 0.887],
        ["MUSK", 0.871, 0.815, 0.901, 0.853],
        ["PHIKON", 0.905, 0.863, 0.921, 0.872],
        ["PROVGIGAPATH", 0.898, 0.853, 0.925, 0.884],
        ["RETCCL", 0.872, 0.807, 0.898, 0.837],
        ["UNI", 0.901, 0.858, 0.932, 0.889],
        ["UNI2", 0.898, 0.846, 0.933, 0.897],
        ["VIRCHOW", 0.905, 0.861, 0.910, 0.869],
        ["AVERAGE", 0.886, 0.835, 0.910, 0.863],
    ],
    "PR": [
        ["CNN", 0.751, 0.743, 0.779, 0.759],
        ["CONCH", 0.779, 0.762, 0.810, 0.789],
        ["CTRANSPATH", 0.770, 0.759, 0.802, 0.781],
        ["HIBOU-B", 0.756, 0.746, 0.798, 0.792],
        ["HIBOU-L", 0.767, 0.756, 0.790, 0.764],
        ["HOPTIMUS-0", 0.822, 0.803, 0.829, 0.808],
        ["MUSK", 0.764, 0.753, 0.800, 0.789],
        ["PHIKON", 0.789, 0.782, 0.817, 0.802],
        ["PROVGIGAPATH", 0.798, 0.788, 0.813, 0.799],
        ["RETCCL", 0.768, 0.745, 0.787, 0.760],
        ["UNI", 0.795, 0.786, 0.824, 0.807],
        ["UNI2", 0.804, 0.786, 0.827, 0.813],
        ["VIRCHOW", 0.793, 0.783, 0.838, 0.823],
        ["AVERAGE", 0.781, 0.769, 0.809, 0.791],
    ],
    "HER2": [
        ["CNN", 0.695, 0.603, 0.734, 0.640],
        ["CONCH", 0.698, 0.595, 0.746, 0.646],
        ["CTRANSPATH", 0.759, 0.642, 0.767, 0.673],
        ["HIBOU-B", 0.716, 0.612, 0.757, 0.659],
        ["HIBOU-L", 0.617, 0.549, 0.695, 0.594],
        ["HOPTIMUS-0", 0.743, 0.629, 0.759, 0.653],
        ["MUSK", 0.722, 0.616, 0.759, 0.656],
        ["PHIKON", 0.727, 0.616, 0.795, 0.708],
        ["PROVGIGAPATH", 0.757, 0.631, 0.773, 0.670],
        ["RETCCL", 0.732, 0.622, 0.748, 0.654],
        ["UNI", 0.752, 0.626, 0.795, 0.699],
        ["UNI2", 0.730, 0.621, 0.783, 0.682],
        ["VIRCHOW", 0.722, 0.635, 0.772, 0.670],
        ["AVERAGE", 0.721, 0.615, 0.760, 0.662],
    ],
    "PAM-50": [
        ["CNN", 0.635, 0.349, 0.783, 0.434, 0.653, 0.380, 0.818, 0.480],
        ["CONCH", 0.701, 0.496, 0.836, 0.524, 0.707, 0.482, 0.867, 0.548],
        ["CTRANSPATH", 0.679, 0.454, 0.828, 0.521, 0.696, 0.489, 0.864, 0.554],
        ["HIBOU-B", 0.682, 0.474, 0.819, 0.510, 0.701, 0.504, 0.853, 0.536],
        ["HIBOU-L", 0.656, 0.405, 0.800, 0.458, 0.672, 0.432, 0.821, 0.478],
        ["HOPTIMUS-0", 0.740, 0.574, 0.882, 0.611, 0.736, 0.577, 0.896, 0.627],
        ["MUSK", 0.680, 0.460, 0.822, 0.490, 0.696, 0.502, 0.854, 0.541],
        ["PHIKON", 0.711, 0.513, 0.838, 0.547, 0.728, 0.557, 0.897, 0.629],
        ["PROVGIGAPATH", 0.698, 0.510, 0.847, 0.560, 0.733, 0.561, 0.887, 0.618],
        ["RETCCL", 0.656, 0.418, 0.821, 0.477, 0.674, 0.449, 0.847, 0.500],
        ["UNI", 0.718, 0.535, 0.847, 0.571, 0.742, 0.544, 0.888, 0.619],
        ["UNI2", 0.731, 0.583, 0.860, 0.597, 0.742, 0.582, 0.891, 0.628],
        ["VIRCHOW", 0.730, 0.552, 0.879, 0.598, 0.721, 0.540, 0.890, 0.620],
        ["AVERAGE", 0.694, 0.486, 0.836, 0.531, 0.708, 0.508, 0.867, 0.567],
    ]
}

# Nombres de métricas
metric_names_pam50 = ['Stratified Accuracy', 'Stratified  F1', 'Stratified  AUROC', 'Stratified AUPRC','Accuracy', 'F1', 'AUROC', 'AUPRC']
metric_names_ihc = ['Stratified  AUROC', 'Stratified AUPRC','AUROC', 'AUPRC']
task_pam50 = ["PAM-50"]
task_ihc = ["HER2", "PR", "ER"]


def calculate_rankings(tasks_data, task_names, metric_names):
    ranking_data = {metric: defaultdict(list) for metric in metric_names}
    for task, rows in tasks_data.items():
        if task in task_names:
            df = pd.DataFrame(rows, columns=["Model"] + metric_names)
            for metric in metric_names:
                ranked = df[["Model", metric]].copy()
                ranked["Rank"] = ranked[metric].rank(ascending=False, method='min')
                for _, row in ranked.iterrows():
                    ranking_data[metric][row["Model"]].append(row["Rank"])

    # Promediar rankings por modelo y métrica
    average_rankings = {
        metric: {
            model: sum(ranks) / len(ranks)
            for model, ranks in models.items()
        } for metric, models in ranking_data.items()
    }

    # Convertir a DataFrame y ordenar alfabéticamente
    ranking_df = pd.DataFrame(average_rankings).sort_index()

    # Crear clustermap
    g = sns.clustermap(
        ranking_df,
        annot=False,
        cmap="magma_r",
        figsize=(10, 8),
        cbar_kws={'label': 'Average Rank'},
        metric='euclidean',
        method='average',
        col_cluster=False
    )
    # Modifica la leyenda para mostrar los valores mínimo y máximo
    cbar = g.cax
    cbar.set_yticks([1, 7 ,14])
    cbar.set_yticklabels([1, 7, 14])

    plt.suptitle("Clustered Model Rankings per Metric (PATIENT_STRAT)", y=1.02)
    plt.show()

# Calcular rankings por tarea
calculate_rankings(tasks_data_full_cv, task_pam50, metric_names_pam50)
calculate_rankings(tasks_data_full_cv, task_ihc, metric_names_ihc)