import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

def get_n_classes(path):
    """
    Devuelve el número de clases a partir del nombre del directorio.
    Se espera que el nombre del directorio contenga un número seguido de un guion bajo.
    """
    if 'ihc_simple' in path :
        return 4, ['TNBC', 'Her2-enriched', 'Luminal-A', 'Luminal-B']
    elif 'ihc' in path or 'pam50' in path:
        return 5, ['Basal', 'Her2-enriched', 'Luminal-A', 'Luminal-B', 'Normal-like']
    else:
        return 2, ['Negative', 'Positive']
#name = 'EVAL_.results-tcga-pam50-clam_mb-10-cv---patient_strat-1744016425-virchow-virchow-summary'
#name = 'EVAL_.results-tcga-erbb2-clam_sb-10-cv---patient_strat-1744160542-virchow-virchow-summary'
#name = 'EVAL_.results-tcga-er-clam_sb-10-cv---patient_strat-1744070274-virchow-virchow-summary'
#name = 'EVAL_.results-tcga-pr-clam_sb-10-cv---patient_strat-1744116573-virchow-virchow-summary'

#name = 'EVAL_.results-tcga-pam50-clam_mb-ho---patient_strat-tcga-1744054479-cptac-1744054484-virchow-virchow-summary'
#name = 'EVAL_.results-tcga-erbb2-clam_sb-ho---patient_strat-tcga-1744059773-cptac-1744059778-virchow-virchow-summary'
#name = 'EVAL_.results-tcga-er-clam_sb-ho---patient_strat-tcga-1744064757-cptac-1744064762-virchow-virchow-summary'
name = 'EVAL_.results-tcga-pr-clam_sb-ho---patient_strat-tcga-1744069675-cptac-1744069719-virchow-virchow-summary'
full_path = "./.eval_results/" + name
results = []
    

# Acumuladores
all_y_true = []
all_y_pred = []
all_y_scores = []

files = os.listdir(full_path) 
fold_files = [f for f in files if f.startswith("fold_") and  f.endswith(".csv")]
n_classes, labels = get_n_classes(full_path)

for fold_file in fold_files:
    fold_path = os.path.join(full_path, fold_file)
    df = pd.read_csv(fold_path, header=None)
    df = df.iloc[1:]
    y_true = df.iloc[:, 1].values.astype(float)
    y_pred = df.iloc[:, 2].values.astype(float)
    y_scores = df.iloc[:, 3:3 + n_classes:].values.astype(float)
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)
    all_y_scores.extend(y_scores)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_scores = np.array(all_y_scores)

y_true_bin = label_binarize(all_y_true, classes=range(n_classes+1))

confussion_matrix = confusion_matrix(all_y_true, all_y_pred)
print(confussion_matrix)


# Forzar que las celdas de la matriz de confusión sean valores numéricos enteros
plt.figure(figsize=(8, 6))
sns.heatmap(
    confussion_matrix.astype(int),
    annot=True,  # Anotaciones booleanas
    fmt='d',
    cmap='PuRd',       # Gama de rosas
    linewidths=0.5,    # Bordes entre celdas
    linecolor='white',
    cbar_kws={'label': ''},  # Sin etiqueta de barra
    vmin=0,             # El blanco será 0
    xticklabels=labels,
    yticklabels=labels
)
plt.title('Confussion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
