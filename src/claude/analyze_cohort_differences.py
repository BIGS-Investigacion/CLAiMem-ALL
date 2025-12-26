import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2):
    """
    Calculate t-test from summary statistics.
    Returns t-statistic and p-value.
    """
    # Welch's t-test (unequal variances)
    # Standard error of the difference between means
    se_diff = np.sqrt((std1**2 / nobs1) + (std2**2 / nobs2))
    
    # t-statistic
    t_stat = (mean1 - mean2) / se_diff
    
    # Degrees of freedom (Welch-Satterthwaite equation)
    df_num = ((std1**2 / nobs1) + (std2**2 / nobs2))**2
    df_den = ((std1**2 / nobs1)**2 / (nobs1 - 1)) + ((std2**2 / nobs2)**2 / (nobs2 - 1))
    df = df_num / df_den
    
    # p-value (two-tailed)
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return t_stat, p_val

def cohens_d_from_stats(mean1, std1, nobs1, mean2, std2, nobs2):
    """
    Calculate Cohen's d from summary statistics.
    """
    # Pooled standard deviation
    n1, n2 = nobs1, nobs2
    s1, s2 = std1, std2
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    d = (mean1 - mean2) / s_pooled
    return d

def main():
    # Paths
    base_dir = '/media/jorge/investigacion/software/CLAiMemAll'
    input_file = os.path.join(base_dir, 'results/biological_analysis/biological_features_by_class_cohort.csv')
    output_dir = os.path.join(base_dir, 'results/biological_analysis/cohort_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Features to analyze
    features = [
        'ESTRUCTURA GLANDULAR', 'ATIPIA NUCLEAR', 'MITOSIS', 
        'NECROSIS', 'INFILTRADO_LI', 'INFILTRADO_PMN'
    ]
    
    results = []
    
    # Iterate through unique Task and Class combinations
    for task in df['Task'].unique():
        task_df = df[df['Task'] == task]
        for cls in task_df['Class'].unique():
            cls_df = task_df[task_df['Class'] == cls]
            
            # Get TCGA and CPTAC data
            tcga_row = cls_df[cls_df['Cohort'] == 'TCGA']
            cptac_row = cls_df[cls_df['Cohort'] == 'CPTAC']
            
            if tcga_row.empty or cptac_row.empty:
                print(f"Skipping {task} - {cls}: Missing cohort data")
                continue
                
            tcga_row = tcga_row.iloc[0]
            cptac_row = cptac_row.iloc[0]
            
            for feature in features:
                mean_tcga = tcga_row[f'{feature}_mean']
                std_tcga = tcga_row[f'{feature}_std']
                n_tcga = tcga_row['N_samples']
                
                mean_cptac = cptac_row[f'{feature}_mean']
                std_cptac = cptac_row[f'{feature}_std']
                n_cptac = cptac_row['N_samples']
                
                # Perform t-test
                t_stat, p_val = ttest_ind_from_stats(
                    mean_tcga, std_tcga, n_tcga,
                    mean_cptac, std_cptac, n_cptac
                )
                
                # Calculate Cohen's d
                d = cohens_d_from_stats(
                    mean_tcga, std_tcga, n_tcga,
                    mean_cptac, std_cptac, n_cptac
                )
                
                results.append({
                    'Task': task,
                    'Class': cls,
                    'Feature': feature,
                    'TCGA_Mean': mean_tcga,
                    'TCGA_Std': std_tcga,
                    'CPTAC_Mean': mean_cptac,
                    'CPTAC_Std': std_cptac,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'Cohens_d': d
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, 'cohort_statistical_comparison.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Saved statistical results to {output_csv}")
    
    # Plotting
    # Create a plot for each Task
    for task in df['Task'].unique():
        task_results = results_df[results_df['Task'] == task]
        
        # Melt for plotting
        # We need to reconstruct the data for plotting bars with error bars
        # Since we have summary stats, we can't use sns.barplot directly with data
        # We'll construct a custom plot
        
        features_list = task_results['Feature'].unique()
        classes_list = task_results['Class'].unique()
        
        n_features = len(features_list)
        n_classes = len(classes_list)
        
        fig, axes = plt.subplots(n_classes, 1, figsize=(12, 4 * n_classes), sharex=True)
        if n_classes == 1:
            axes = [axes]
            
        for i, cls in enumerate(classes_list):
            ax = axes[i]
            cls_data = task_results[task_results['Class'] == cls]
            
            x = np.arange(n_features)
            width = 0.35
            
            rects1 = ax.bar(x - width/2, cls_data['TCGA_Mean'], width, yerr=cls_data['TCGA_Std'], label='TCGA', capsize=5)
            rects2 = ax.bar(x + width/2, cls_data['CPTAC_Mean'], width, yerr=cls_data['CPTAC_Std'], label='CPTAC', capsize=5)
            
            ax.set_ylabel('Score')
            ax.set_title(f'{task} - {cls}')
            ax.set_xticks(x)
            ax.set_xticklabels(features_list, rotation=45, ha='right')
            ax.legend()
            
            # Add significance stars
            for j, feature in enumerate(features_list):
                row = cls_data[cls_data['Feature'] == feature].iloc[0]
                p_val = row['p_value']
                
                if p_val < 0.05:
                    # Calculate height for star
                    max_height = max(row['TCGA_Mean'] + row['TCGA_Std'], row['CPTAC_Mean'] + row['CPTAC_Std'])
                    ax.text(j, max_height * 1.05, '*', ha='center', va='bottom', fontsize=15, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{task}_cohort_comparison.png')
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
        plt.close()

if __name__ == "__main__":
    main()
