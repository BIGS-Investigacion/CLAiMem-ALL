
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

limites = (-0.45, 0)
metricas_pam50 = ['Stratified Accuracy', 'Stratified F1', 'Stratified AUROC', 'Stratified AUPRC',
            'Accuracy', 'F1', 'AUROC', 'AUPRC']
valores_pam50 = {
    'PHIKONv2': [-0.1498, -0.3271, -0.1290, -0.2489, -0.1304, -0.3624, -0.1416, -0.2415],
    'UNI': [-0.1036, -0.3173, -0.0653, -0.2065, -0.1469, -0.4409, -0.1234, -0.3198],
    'UNI2': [-0.1692, -0.4430, -0.1087, -0.2507, -0.1747, -0.4715, -0.1246, -0.2798],
    'HOPTIMUS-0': [-0.1726, -0.4694, -0.0979, -0.2635, -0.2035, -0.4650, -0.1222, -0.2689],
    'PROVGIGAPATH': [-0.0301, -0.2568, -0.0925, -0.1821, -0.1042, -0.3655, -0.1189, -0.2282],
    'VIRCHOW2': [-0.0933, -0.3523, -0.0752, -0.1658, -0.1261, -0.3004, -0.1296, -0.2550],
    'AVERAGE': [-0.1205, -0.3645, -0.0946, -0.2198, -0.1478, -0.4022, -0.1267, -0.2655],
}

# Datos
metricas_ihc = [
    'Stratified ER AUROC', 'Stratified ER AUPRC', 'ER AUROC', 'ER AUPRC',
    'Stratified PR AUROC', 'Stratified PR AUPRC', 'PR AUROC', 'PR AUPRC',
    'Stratified HER2 AUROC', 'Stratified HER2 AUPRC', 'HER2 AUROC', 'HER2 AUPRC'
]

valores_ihc = {
    'PHIKONv2': [-0.057, -0.037, -0.075, -0.055, -0.027, -0.037, -0.191, -0.179, -0.240, -0.113, -0.264, -0.213],
    'UNI': [-0.080, -0.055, -0.127, -0.112, 0.017, 0.016, -0.109, -0.087, -0.197, -0.142, -0.203, -0.200],
    'UNI2': [-0.029, -0.005, -0.067, -0.065, 0.030, 0.048, -0.010, 0.004, -0.114, -0.115, -0.216, -0.211],
    'HOPTIMUS-0': [-0.084, -0.058, -0.085, -0.078, -0.065, -0.032, -0.146, -0.112, -0.124, -0.129, -0.199, -0.176],
    'PROVGIGAPATH': [-0.070, -0.036, -0.103, -0.089, -0.029, -0.016, -0.153, -0.143, -0.156, -0.135, -0.176, -0.167],
    'VIRCHOW2': [-0.037, -0.015, -0.037, -0.015, 0.067, 0.089, -0.018, 0.000, -0.079, -0.102, -0.110, -0.087],
    'AVERAGE': [-0.060, -0.035, -0.082, -0.069, -0.002, 0.011, -0.104, -0.086, -0.152, -0.123, -0.195, -0.176]
}

def visualize_heatmap(valores, metricas):
    # Crear DataFrame
    df_ihc = pd.DataFrame(valores, index=metricas).T


    # Crear biclustermap
    g = sns.clustermap(
        df_ihc,
        metric="euclidean",
        method="ward",
        cmap="magma",  # ya invertimos los valores
        annot=False,
        linewidths=0.5,
        figsize=(12, 8),
    )
    # Ajustar la escala de colores para que llegue hasta -0.45
    g.ax_heatmap.collections[0].set_clim(vmin=0, vmax=-0.45)
    # Eliminar título
    plt.title("")
    plt.show()


def visualize_boxplot(valores, metricas):
    # Crear DataFrame
    df_ihc = pd.DataFrame(valores, index=metricas).T

    # Reorganizar para boxplot
    df_ihc_long = df_ihc.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
    df_ihc_long.rename(columns={'index': 'Model'}, inplace=True)

    # Crear diagrama de caja en blanco y negro
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_ihc_long, x='Metric', y='Value', palette="magma")

    plt.xticks(rotation=45, ha='right')
    plt.title("")
    plt.xlabel("Metric")
    plt.ylabel("Decrease ratio regarding MCCV Metric")
    plt.tight_layout()
    plt.show()

def visualize_boxplot_models(valores, metricas):
    # Crear DataFrame
    df_ihc = pd.DataFrame(valores, index=metricas).T

    # Reorganizar para boxplot por modelo
    df_ihc_long = df_ihc.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
    df_ihc_long.rename(columns={'index': 'Model'}, inplace=True)

    # Crear boxplot con paleta de grises clara
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_ihc_long, x='Model', y='Value', palette="magma")

    plt.xticks(rotation=45, ha='right')
    plt.title("")
    plt.xlabel("Model")
    plt.ylabel("Decrease ratio regarding MCCV Metric")
    plt.tight_layout()
    plt.show()


visualize_boxplot_models(valores_ihc, metricas_ihc)
visualize_boxplot_models(valores_pam50, metricas_pam50)
