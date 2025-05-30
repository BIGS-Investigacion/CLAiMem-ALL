# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict



# Nombres de métricas
#metric_names = ['Stratified AUROC', 'Stratified AUPRC', 'AUROC', 'AUPRC']
metric_names = ['Stratified Accuracy', 'Stratified F1', 'Stratified AUROC', 'Stratified AUPRC','Accuracy', 'F1', 'AUROC', 'AUPRC']

# Calcular rankings
ranking_data = {metric: defaultdict(list) for metric in metric_names}
for task, rows in tasks_data_full.items():
    df = pd.DataFrame(rows, columns=["Model"] + metric_names)
    for metric in metric_names:
        ranked = df[["Model", metric]].copy()
        ranked["Rank"] = ranked[metric].rank(ascending=False, method='min')
        for _, row in ranked.iterrows():
            ranking_data[metric][row["Model"]].append(row["Rank"])

average_rankings = {
    metric: {
        model: sum(ranks) / len(ranks)
        for model, ranks in models.items()
    } for metric, models in ranking_data.items()
}

# Convertir a DataFrame y ordenar alfabéticamente
ranking_df = pd.DataFrame(average_rankings).sort_index()

# Crear clustermap
sns.clustermap(
    ranking_df,
    col_cluster=False,
    annot=False,
    cmap="rocket",
    figsize=(10, 8),
    cbar_kws={'label': 'Average Rank'},
    metric='euclidean',
    method='ward'
)

plt.suptitle("", y=1.02)
plt.show()
