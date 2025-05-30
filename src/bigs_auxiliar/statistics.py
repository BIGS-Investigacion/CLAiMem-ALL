import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize

def get_n_classes(path):
    """
    Devuelve el número de clases a partir del nombre del directorio.
    Se espera que el nombre del directorio contenga un número seguido de un guion bajo.
    """
    if 'ihc_simple' in path :
        return 4
    elif 'ihc' in path or 'pam50' in path:
        return 5
    else:
        return 2


root_dir = "./.eval_results"
results = []
    

# Recorre los subdirectorios DIRECTOS
for subdir in next(os.walk(root_dir))[1]:
    if True:#"pam50" in subdir:
        # Acumuladores
        all_y_true = []
        all_y_pred = []
        all_y_scores = []
        full_path = os.path.join(root_dir, subdir)
        files = os.listdir(full_path) 
        fold_files = [f for f in files if f.startswith("fold_") and  f.endswith(".csv")]
        n_classes = get_n_classes(full_path)
        for fold_file in fold_files:
            fold_path = os.path.join(full_path, fold_file)
            df = pd.read_csv(fold_path, header=None)
            df = df.iloc[1:]
            
            # Filtrar filas donde df.iloc[:, 1] == 4 o df.iloc[:, 2] == 4
            n_classes =4
            df = df[(df.iloc[:, 1] != 4) & (df.iloc[:, 2] != 4)]
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

        acc = accuracy_score(all_y_true, all_y_pred)
        f1 = f1_score(all_y_true, all_y_pred, average='macro')
        confussion_matrix = confusion_matrix(all_y_true, all_y_pred)
        print(confussion_matrix)
        auroc = roc_auc_score(y_true_bin, all_y_scores, average='macro', multi_class='ovr')
        auprc = average_precision_score(y_true_bin, all_y_scores, average='macro')
        # Calcular F1_macro eliminando una etiqueta específica (por ejemplo, etiqueta 3)
        excluded_label = 0
        filtered_indices = all_y_true != excluded_label
        filtered_y_true = all_y_true[filtered_indices]
        filtered_y_pred = all_y_pred[filtered_indices]
        f1_filtered_0 = f1_score(filtered_y_true, filtered_y_pred, average='macro')
        excluded_label = 1
        filtered_indices = all_y_true != excluded_label
        filtered_y_true = all_y_true[filtered_indices]
        filtered_y_pred = all_y_pred[filtered_indices]
        f1_filtered_1 = f1_score(filtered_y_true, filtered_y_pred, average='macro')
        excluded_label = 2
        filtered_indices = all_y_true != excluded_label
        filtered_y_true = all_y_true[filtered_indices]
        filtered_y_pred = all_y_pred[filtered_indices]
        f1_filtered_2 = f1_score(filtered_y_true, filtered_y_pred, average='macro')
        excluded_label = 3
        filtered_indices = all_y_true != excluded_label
        filtered_y_true = all_y_true[filtered_indices]
        filtered_y_pred = all_y_pred[filtered_indices]
        f1_filtered_3 = f1_score(filtered_y_true, filtered_y_pred, average='macro')
        excluded_label = 4
        filtered_indices = all_y_true != excluded_label
        filtered_y_true = all_y_true[filtered_indices]
        filtered_y_pred = all_y_pred[filtered_indices]
        f1_filtered_4 = f1_score(filtered_y_true, filtered_y_pred, average='macro')


        results.append({
            "Experiment": subdir,
            "Accuracy": acc,
            "F1_macro": f1,
            "F1_macro_excluding_0": f1_filtered_0,
            "F1_macro_excluding_1": f1_filtered_1,
            "F1_macro_excluding_2": f1_filtered_2,
            "F1_macro_excluding_3": f1_filtered_3,
            "F1_macro_excluding_4": f1_filtered_4,
            "AUROC_macro": auroc,
            "AUPRC_macro": auprc
        })

    

# Guardar resultado
summary_df = pd.DataFrame(results)
summary_df.to_csv("metrics_summary.csv", index=False)
print("Resumen guardado en metrics_summary.csv")

