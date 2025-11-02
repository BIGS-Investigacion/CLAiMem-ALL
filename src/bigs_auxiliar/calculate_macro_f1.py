import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import sys
import os

# Directorio con los archivos
data_dir = ".eval_results/pam50/ho"
subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

FOLDS = 1

for subdir in subdirs:
    print(f"  - {subdir}")
    # Nombres de clases PAM50 (ajustar según tu codificación)
    # Asumiendo orden: Basal, HER2-enriched, Luminal A, Luminal B, Normal-like
    CLASS_NAMES = ['Basal', 'HER2-enriched', 'Luminal A', 'Luminal B', 'Normal-like']

    # Calcular métricas para cada fold
    macro_f1_scores = []
    weighted_f1_scores = []
    per_class_f1_scores = {class_name: [] for class_name in CLASS_NAMES}
    fold_details = []

    print("="*90)
    print("MACRO F1-SCORE PARA CLASIFICACIÓN MULTI-CLASE (PAM50 MOLECULAR SUBTYPING)")
    print("="*90)
    print()

    for fold_num in range(FOLDS):
        file_path = f"{data_dir}/{subdir}/fold_{fold_num}.csv"
        
        try:
            # Leer CSV
            df = pd.read_csv(file_path)
            
            # Verificar que es multi-clase
            if 'p_2' not in df.columns:
                print(f"ERROR: fold_{fold_num}.csv parece ser clasificación binaria, no multi-clase.")
                print("Se esperaban columnas: Y, Y_hat, p_0, p_1, p_2, p_3, p_4")
                print(f"Columnas encontradas: {list(df.columns)}")
                sys.exit(1)
            
            # Extraer etiquetas verdaderas y predichas
            y_true = df['Y'].values.astype(int)
            y_pred = df['Y_hat'].values.astype(int)
            
            # Verificar número de clases
            n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
            if n_classes != 5:
                print(f"WARNING: Se esperaban 5 clases (PAM50), pero hay {n_classes} clases")
            
            # Calcular Macro F1-score (promedio simple de F1 por clase)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            macro_f1_scores.append(macro_f1)
            
            # Calcular Weighted F1-score (promedio ponderado por soporte)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            weighted_f1_scores.append(weighted_f1)
            
            # Calcular F1-score por clase
            per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            for i, class_name in enumerate(CLASS_NAMES[:len(per_class_f1)]):
                per_class_f1_scores[class_name].append(per_class_f1[i])
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Distribution
            n_samples = len(df)
            class_distribution = [np.sum(y_true == i) for i in range(n_classes)]
            
            fold_details.append({
                'fold': fold_num,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'per_class_f1': per_class_f1,
                'n_samples': n_samples,
                'class_distribution': class_distribution,
                'confusion_matrix': cm
            })
            
            print(f"Fold {fold_num}:")
            print(f"  Macro F1-Score:    {macro_f1:.4f}")
            print(f"  Weighted F1-Score: {weighted_f1:.4f}")
            print(f"  Per-class F1:")
            for i, class_name in enumerate(CLASS_NAMES[:len(per_class_f1)]):
                print(f"    {class_name:20s}: {per_class_f1[i]:.4f}")
            print(f"  Class distribution: {class_distribution}")
            print(f"  Total samples: {n_samples}")
            print()
            
        except FileNotFoundError:
            print(f"ERROR: No se encontró {file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR procesando fold {fold_num}: {e}")
            sys.exit(1)

    # Calcular estadísticas para Macro F1
    macro_f1_mean = np.mean(macro_f1_scores)
    macro_f1_std = np.std(macro_f1_scores, ddof=1)
    macro_f1_min = np.min(macro_f1_scores)
    macro_f1_max = np.max(macro_f1_scores)

    # Calcular estadísticas para Weighted F1
    weighted_f1_mean = np.mean(weighted_f1_scores)
    weighted_f1_std = np.std(weighted_f1_scores, ddof=1)

    # Calcular estadísticas por clase
    per_class_stats = {}
    for class_name in CLASS_NAMES:
        if len(per_class_f1_scores[class_name]) > 0:
            per_class_stats[class_name] = {
                'mean': np.mean(per_class_f1_scores[class_name]),
                'std': np.std(per_class_f1_scores[class_name], ddof=1),
                'min': np.min(per_class_f1_scores[class_name]),
                'max': np.max(per_class_f1_scores[class_name])
            }

    print("="*90)
    print("RESUMEN ESTADÍSTICO")
    print("="*90)
    print()

    print("Macro F1-Score (promedio simple por clase):")
    print(f"  Mean:  {macro_f1_mean:.4f}")
    print(f"  Std:   {macro_f1_std:.4f}")
    print(f"  Range: [{macro_f1_min:.4f}, {macro_f1_max:.4f}]")
    print()

    print("Weighted F1-Score (promedio ponderado por soporte):")
    print(f"  Mean:  {weighted_f1_mean:.4f}")
    print(f"  Std:   {weighted_f1_std:.4f}")
    print()

    print("F1-Score por clase (promedio a través de folds):")
    for class_name in CLASS_NAMES:
        if class_name in per_class_stats:
            stats = per_class_stats[class_name]
            print(f"  {class_name:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}  " + 
                f"[{stats['min']:.4f}, {stats['max']:.4f}]")
    print()

    # Identificar clase más difícil y más fácil
    if per_class_stats:
        easiest_class = max(per_class_stats.items(), key=lambda x: x[1]['mean'])
        hardest_class = min(per_class_stats.items(), key=lambda x: x[1]['mean'])
        print(f"Clase más fácil de predecir: {easiest_class[0]} (F1 = {easiest_class[1]['mean']:.3f})")
        print(f"Clase más difícil de predecir: {hardest_class[0]} (F1 = {hardest_class[1]['mean']:.3f})")
        print()

    print("="*90)
    print("RESULTADOS PARA REPORTAR EN PAPER")
    print("="*90)
    print(f"Macro F1-Score    = {macro_f1_mean:.3f} ± {macro_f1_std:.3f}")
    print(f"Weighted F1-Score = {weighted_f1_mean:.3f} ± {weighted_f1_std:.3f}")
    print()
    print("Per-class F1 (MCCV, M=10):")
    for class_name in CLASS_NAMES:
        if class_name in per_class_stats:
            stats = per_class_stats[class_name]
            print(f"  {class_name:20s}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print("="*90)
    print(subdir)
    input("Seguimos?")

