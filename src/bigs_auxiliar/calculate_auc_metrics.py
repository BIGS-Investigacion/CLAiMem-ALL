import os
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# Directorio con los archivos
data_dir = ".eval_results/erbb2/ho"
subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

FOLDS = 1

for subdir in subdirs:
    print(f"  - {subdir}")

    # Calcular métricas para cada fold
    pr_auc_scores = []
    roc_auc_scores = []
    fold_details = []

    print("="*80)
    print("MÉTRICAS BASADAS EN PROBABILIDADES (PR-AUC y ROC-AUC)")
    print("="*80)
    print()

    for fold_num in range(FOLDS):
        file_path = f"{data_dir}/{subdir}/fold_{fold_num}.csv"
        
        # Leer CSV
        df = pd.read_csv(file_path)
        
        # Extraer etiquetas verdaderas y probabilidades predichas
        y_true = df['Y'].values
        y_pred_proba = df['p_1'].values  # Probabilidad de la clase positiva (1)
        
        # Calcular PR-AUC (Average Precision)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        pr_auc_scores.append(pr_auc)
        
        # Calcular ROC-AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        roc_auc_scores.append(roc_auc)
        
        # Guardar detalles
        n_samples = len(df)
        n_positive = (y_true == 1).sum()
        n_negative = (y_true == 0).sum()
        prevalence = n_positive / n_samples
        
        fold_details.append({
            'fold': fold_num,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'n_samples': n_samples,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'prevalence': prevalence
        })
        
        print(f"Fold {fold_num}:")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Samples: n={n_samples}, pos={n_positive} ({prevalence:.1%}), neg={n_negative}")
        print()

    # Calcular estadísticas para PR-AUC
    pr_auc_mean = np.mean(pr_auc_scores)
    pr_auc_std = np.std(pr_auc_scores, ddof=1)
    pr_auc_min = np.min(pr_auc_scores)
    pr_auc_max = np.max(pr_auc_scores)

    # Calcular estadísticas para ROC-AUC
    roc_auc_mean = np.mean(roc_auc_scores)
    roc_auc_std = np.std(roc_auc_scores, ddof=1)
    roc_auc_min = np.min(roc_auc_scores)
    roc_auc_max = np.max(roc_auc_scores)

    # Prevalencia promedio
    avg_prevalence = np.mean([d['prevalence'] for d in fold_details])

    print("="*80)
    print("RESUMEN ESTADÍSTICO")
    print("="*80)
    print()
    print("PR-AUC (Precision-Recall AUC):")
    print(f"  Mean:  {pr_auc_mean:.4f}")
    print(f"  Std:   {pr_auc_std:.4f}")
    print(f"  Range: [{pr_auc_min:.4f}, {pr_auc_max:.4f}]")
    print()

    print("ROC-AUC:")
    print(f"  Mean:  {roc_auc_mean:.4f}")
    print(f"  Std:   {roc_auc_std:.4f}")
    print(f"  Range: [{roc_auc_min:.4f}, {roc_auc_max:.4f}]")
    print()

    print("Información de clase:")
    print(f"  Prevalencia promedio: {avg_prevalence:.1%}")
    print(f"  Baseline PR-AUC (random): {avg_prevalence:.3f}")
    print()

    print("="*80)
    print("RESULTADOS PARA REPORTAR EN PAPER")
    print("="*80)
    print(f"PR-AUC  = {pr_auc_mean:.3f} ± {pr_auc_std:.3f}")
    print(f"ROC-AUC = {roc_auc_mean:.3f} ± {roc_auc_std:.3f}")
    print("="*80)
    print(subdir)
    input("Seguimos?")