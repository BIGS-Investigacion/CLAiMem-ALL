import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *


# Simulación de datos
np.random.seed(42)
modelos = ['Modelo1', 'Modelo2', 'Modelo3']
folds = range(1, 11)
data = []

for modelo in modelos:
    for fold in folds:
        y_real = np.random.randint(0, 2, 100)  # 100 instancias por fold
        y_predicho = np.random.rand(100)  # Probabilidades de 0 a 1
        data.extend(zip([modelo]*100, [fold]*100, y_real, y_predicho))

df = pd.DataFrame(data, columns=['modelo', 'fold', 'y_real', 'y_predicho'])

# Convertir predicciones a etiquetas binarias (umbral 0.5)
df['y_pred_label'] = (df['y_predicho'] >= 0.5).astype(int)

# Función para graficar curvas ROC
def plot_roc_curve(df):
    plt.figure(figsize=(8, 6))
    for modelo in df['modelo'].unique():
        fpr, tpr, _ = roc_curve(df[df['modelo'] == modelo]['y_real'], df[df['modelo'] == modelo]['y_predicho'])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{modelo} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC por Modelo')
    plt.legend()
    plt.grid()
    plt.show()

# Función para graficar curvas de Precisión-Recuperación
def plot_precision_recall_curve(df):
    plt.figure(figsize=(8, 6))
    for modelo in df['modelo'].unique():
        precision, recall, _ = precision_recall_curve(df[df['modelo'] == modelo]['y_real'], df[df['modelo'] == modelo]['y_predicho'])
        plt.plot(recall, precision, label=f'{modelo}')

    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.title('Curva Precisión-Recall por Modelo')
    plt.legend()
    plt.grid()
    plt.show()

# Matriz de confusión promediada
def plot_avg_confusion_matrix(df):
    modelos = df['modelo'].unique()
    cm_sum = np.zeros((2, 2))

    for modelo in modelos:
        for fold in df['fold'].unique():
            y_true = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_real']
            y_pred = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_pred_label']
            cm = confusion_matrix(y_true, y_pred)
            cm_sum += cm

    cm_avg = cm_sum / (len(modelos) * len(folds))  # Promedio de matrices de confusión

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Clase 0", "Clase 1"], yticklabels=["Clase 0", "Clase 1"])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión Promediada')
    plt.show()

# Boxplot de métricas de los modelos
def plot_boxplot_metrics(df):
    resultados = []
    for modelo in df['modelo'].unique():
        for fold in df['fold'].unique():
            y_true = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_real']
            y_pred = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_pred_label']
            report = classification_report(y_true, y_pred, output_dict=True)
            resultados.append([modelo, fold, report['accuracy'], report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']])

    df_metrics = pd.DataFrame(resultados, columns=['modelo', 'fold', 'accuracy', 'precision', 'recall', 'f1-score'])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='modelo', y='f1-score', data=df_metrics, palette='Set2')
    plt.xlabel('Modelo')
    plt.ylabel('F1-Score')
    plt.title('Distribución del F1-Score por Modelo')
    plt.grid()
    plt.show()

# Violin plot de métricas
def plot_violin_metrics(df):
    resultados = []
    for modelo in df['modelo'].unique():
        for fold in df['fold'].unique():
            y_true = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_real']
            y_pred = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_pred_label']
            report = classification_report(y_true, y_pred, output_dict=True)
            resultados.append([modelo, fold, report['accuracy'], report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']])

    df_metrics = pd.DataFrame(resultados, columns=['modelo', 'fold', 'accuracy', 'precision', 'recall', 'f1-score'])

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='modelo', y='f1-score', data=df_metrics, palette='Pastel1')
    plt.xlabel('Modelo')
    plt.ylabel('F1-Score')
    plt.title('Distribución de F1-Score por Modelo')
    plt.grid()
    plt.show()

# Gráfico de barras de métricas promedio
def plot_bar_metrics(df):
    resultados = []
    for modelo in df['modelo'].unique():
        for fold in df['fold'].unique():
            y_true = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_real']
            y_pred = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_pred_label']
            report = classification_report(y_true, y_pred, output_dict=True)
            resultados.append([modelo, report['accuracy'], report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']])

    df_metrics = pd.DataFrame(resultados, columns=['modelo', 'accuracy', 'precision', 'recall', 'f1-score']).groupby('modelo').mean().reset_index()

    df_metrics.plot(x='modelo', kind='bar', figsize=(10, 6), colormap='viridis')
    plt.ylabel('Valor')
    plt.title('Métricas Promedio por Modelo')
    plt.grid(axis='y')
    plt.show()

# Ejecutar las visualizaciones
plot_roc_curve(df)
plot_precision_recall_curve(df)
plot_avg_confusion_matrix(df)
plot_boxplot_metrics(df)
plot_violin_metrics(df)
plot_bar_metrics(df)



# 🔹 Simulación de datos desbalanceados (1:10)
np.random.seed(42)
modelos = ['Modelo1', 'Modelo2', 'Modelo3']
folds = range(1, 11)
data = []

for modelo in modelos:
    for fold in folds:
        y_real = np.concatenate((np.ones(10), np.zeros(90)))  # 10% Clase 1, 90% Clase 0
        np.random.shuffle(y_real)
        y_predicho = np.random.rand(100)  # Probabilidades de 0 a 1
        data.extend(zip([modelo]*100, [fold]*100, y_real, y_predicho))

df = pd.DataFrame(data, columns=['modelo', 'fold', 'y_real', 'y_predicho'])

# 🔹 Buscar el mejor umbral basado en F1-score
best_thresholds = {}
for modelo in df['modelo'].unique():
    precision, recall, thresholds = precision_recall_curve(df[df['modelo'] == modelo]['y_real'], df[df['modelo'] == modelo]['y_predicho'])
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    best_thresholds[modelo] = thresholds[np.argmax(f1_scores)]  # Umbral con mejor F1-score

# Aplicamos umbral óptimo para cada modelo
df['y_pred_label'] = df.apply(lambda row: int(row['y_predicho'] >= best_thresholds[row['modelo']]), axis=1)

# 🔹 Curva Precisión-Recall
def plot_precision_recall_curve(df):
    plt.figure(figsize=(8, 6))
    for modelo in df['modelo'].unique():
        precision, recall, _ = precision_recall_curve(df[df['modelo'] == modelo]['y_real'], df[df['modelo'] == modelo]['y_predicho'])
        plt.plot(recall, precision, label=f'{modelo}')

    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.title('Curva Precisión-Recall (PR) por Modelo')
    plt.legend()
    plt.grid()
    plt.show()

# 🔹 Matriz de Confusión Normalizada
def plot_confusion_matrix(df):
    modelos = df['modelo'].unique()
    cm_sum = np.zeros((2, 2))

    for modelo in modelos:
        for fold in df['fold'].unique():
            y_true = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_real']
            y_pred = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_pred_label']
            cm = confusion_matrix(y_true, y_pred)
            cm_sum += cm

    cm_avg = cm_sum / (len(modelos) * len(folds))  # Promedio
    cm_norm = cm_avg / cm_avg.sum(axis=1, keepdims=True)  # Normalización

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Clase 0", "Clase 1"], yticklabels=["Clase 0", "Clase 1"])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión Normalizada')
    plt.show()

# 🔹 Boxplot de Balanced Accuracy y F1-score
def plot_boxplot_metrics(df):
    resultados = []
    for modelo in df['modelo'].unique():
        for fold in df['fold'].unique():
            y_true = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_real']
            y_pred = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_pred_label']
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            resultados.append([modelo, fold, bal_acc, f1])

    df_metrics = pd.DataFrame(resultados, columns=['modelo', 'fold', 'balanced_accuracy', 'f1-score'])

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='modelo', y='f1-score', data=df_metrics, palette='Set2')
    plt.xlabel('Modelo')
    plt.ylabel('F1-Score')
    plt.title('Distribución del F1-Score por Modelo')
    plt.grid()
    plt.show()

# 🔹 Violin Plot de Balanced Accuracy
def plot_violin_metrics(df):
    resultados = []
    for modelo in df['modelo'].unique():
        for fold in df['fold'].unique():
            y_true = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_real']
            y_pred = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_pred_label']
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            resultados.append([modelo, fold, bal_acc, f1])

    df_metrics = pd.DataFrame(resultados, columns=['modelo', 'fold', 'balanced_accuracy', 'f1-score'])

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='modelo', y='balanced_accuracy', data=df_metrics, palette='Pastel1')
    plt.xlabel('Modelo')
    plt.ylabel('Balanced Accuracy')
    plt.title('Distribución de Balanced Accuracy por Modelo')
    plt.grid()
    plt.show()

# 🔹 Gráfico de Barras de Métricas Promedio
def plot_bar_metrics(df):
    resultados = []
    for modelo in df['modelo'].unique():
        for fold in df['fold'].unique():
            y_true = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_real']
            y_pred = df[(df['modelo'] == modelo) & (df['fold'] == fold)]['y_pred_label']
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            resultados.append([modelo, bal_acc, f1])

    df_metrics = pd.DataFrame(resultados, columns=['modelo', 'balanced_accuracy', 'f1-score']).groupby('modelo').mean().reset_index()

    df_metrics.plot(x='modelo', kind='bar', figsize=(10, 6), colormap='viridis')
    plt.ylabel('Valor')
    plt.title('Métricas Promedio por Modelo')
    plt.grid(axis='y')
    plt.show()

# 🔥 Ejecutar las visualizaciones
plot_precision_recall_curve(df)
plot_confusion_matrix(df)
plot_boxplot_metrics(df)
plot_violin_metrics(df)
plot_bar_metrics(df)
