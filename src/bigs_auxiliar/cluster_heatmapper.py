import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Definir los datos de rankings
ranking_data = {
    'PAM50_MCCV': [3.0, 2.0, 6.0, 1.0, 4.0, 5.0, 7.0, 9.0, 8.0, 10.0, 11.0, 12.0, 13.0],
    'PAM50_HO': [3.0, 9.0, 1.0, 7.0, 2.0, 5.0, 4.0, 6.0, 11.0, 8.0, 10.0, 12.0, 13.0],
    'ER_MCCV': [3.0, 1.0, 5.0, 6.0, 4.0, 2.0, 2.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0],
    'ER_HO': [1.0, 5.0, 4.0, 2.0, 6.0, 3.0, 7.0, 8.0, 10.0, 11.0, 9.0, 12.0, 13.0],
    'PR_MCCV': [5.0, 1.0, 2.0, 3.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0],
    'PR_HO': [1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 8.0, 13.0],
    'HER2_MCCV': [2.0, 4.0, 3.0, 7.0, 5.0, 6.0, 10.0, 1.0, 8.0, 6.0, 12.0, 9.0, 11.0],
    'HER2_HO': [1.0, 4.0, 5.0, 3.0, 7.0, 6.0, 8.0, 10.0, 9.0, 11.0, 12.0, 13.0, 14.0]
}
models = ['Virchow v2', 'H-optimus-0', 'Prov-Gigapath', 'UNI-2', 'UNI', 'Phikon v2',
          'CONCH', 'CTransPath', 'Hibou-B', 'Musk', 'Hibou-L', 'RetCCL', 'ResNet-50']

# Crear el DataFrame
ranking_df = pd.DataFrame(ranking_data, index=models)

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
#plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
# Modifica la leyenda para mostrar los valores mínimo y máximo
cbar = g.cax
cbar.set_yticks([1, 7 ,14])
cbar.set_yticklabels([1, 7, 14])
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.suptitle("", y=1.02)
plt.show()
