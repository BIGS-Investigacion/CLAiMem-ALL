import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef
)
import seaborn as sns
from sklearn.preprocessing import label_binarize
import os


def get_statistics(df: pd.DataFrame) -> tuple :
    '''
    Esta función toma un DataFrame `df` que contiene las columnas:
    - 'Y': Etiquetas reales (valores enteros).
    - 'Y_hat': Etiquetas predichas (valores enteros).
    - 'p_0', 'p_1', ..., 'p_n': Probabilidades predichas para cada clase.

    Realiza los siguientes pasos:
    1. Convierte las columnas 'Y' y 'Y_hat' a enteros.
    2. Calcula métricas de evaluación como Accuracy, Balanced Accuracy, F1 (macro, micro, weighted),
       Matthews Correlation Coefficient (MCC) y AUC Macro.
    3. Genera un reporte de clasificación por clase.
    4. Muestra una matriz de confusión gráfica.
    5. Traza las curvas ROC para cada clase.
    '''
    # Procesar datos
    df['Y'] = df['Y'].astype(int)
    df['Y_hat'] = df['Y_hat'].astype(int)
    y_true = df['Y']
    y_pred = df['Y_hat']
    num_classes = len(df[[col for col in df.columns if col.startswith('p_')]].columns)
    y_proba = df[[f'p_{i}' for i in range(num_classes)]]

    # Métricas
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro')
    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {'acc.': acc, 'balanced_acc.': balanced_acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'f1_weighted': f1_weighted, 'mcc': mcc
            , 'auc_macro': auc_macro, 'report':report_df, 'confusion_matrix':conf_matrix}
    '''

    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"Matthews CorrCoef: {mcc:.4f}")
    print(f"AUC Macro: {auc_macro:.4f}")
    print("\nClassification Report:\n", report_df)

    # Matriz de confusión gráfica
    plt.figure(figsize=(8,6))
    sns.heatmap(, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # ROC Curves
    fpr = {}
    tpr = {}
    plt.figure(figsize=(10, 8))
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba.iloc[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Clase {i}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Curvas ROC por Clase')
    plt.xlabel('FPR (Tasa de Falsos Positivos)')
    plt.ylabel('TPR (Tasa de Verdaderos Positivos)')
    plt.legend()
    plt.grid(True)
    plt.show()'''


# Recorrer cada subdirectorio del directorio .eval_results
eval_results_dir = '.eval_results'
for root, dirs, _ in os.walk(eval_results_dir):
    for dir in dirs:
        files = os.listdir(os.path.join(root,dir))
        print(f"Procesando directorio: {dir}")
        for file in files:
            file_path = os.path.join(root, dir, file)
            # Verificar si el archivo comienza con "fold"
            if file.startswith("fold"):
                # Leer el archivo como DataFrame
                df_temp = pd.read_csv(file_path)
                
                # Si no existe un DataFrame acumulador, inicializarlo
                if 'df_combined' not in locals():
                    df_combined = df_temp
                else:
                    # Concatenar los datos al DataFrame acumulador
                    df_combined = pd.concat([df_combined, df_temp], ignore_index=True)
            
        print(get_statistics(df_combined))

            
