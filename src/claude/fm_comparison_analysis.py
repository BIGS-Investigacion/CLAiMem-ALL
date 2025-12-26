import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# ============================================================================
# DATA: Performance metrics from table
# ============================================================================

data = {
    'Model': [
        'ResNet-50', 'CTransPath', 'RetCCL', 'CONCH', 'UNI', 
        'Prov-GigaPath', 'Hibou-B', 'Hibou-L', 'H-optimus-0', 
        'Virchow v2', 'Phikon v2', 'Musk', 'UNI-2'
    ],
    'PAM50_MCCV': [0.342, 0.446, 0.414, 0.493, 0.527, 0.504, 0.457, 0.399, 0.565, 0.542, 0.508, 0.450, 0.575],
    'PAM50_HO': [0.218, 0.342, 0.272, 0.335, 0.365, 0.379, 0.289, 0.297, 0.304, 0.358, 0.345, 0.305, 0.325],
    'ER_MCCV': [0.933, 0.962, 0.956, 0.957, 0.967, 0.967, 0.964, 0.952, 0.973, 0.972, 0.971, 0.955, 0.969],
    'ER_HO': [0.722, 0.870, 0.804, 0.885, 0.885, 0.900, 0.803, 0.858, 0.897, 0.916, 0.906, 0.774, 0.917],
    'PR_MCCV': [0.822, 0.845, 0.837, 0.853, 0.870, 0.875, 0.835, 0.826, 0.883, 0.874, 0.861, 0.832, 0.868],
    'PR_HO': [0.595, 0.757, 0.736, 0.777, 0.833, 0.822, 0.696, 0.697, 0.803, 0.862, 0.802, 0.700, 0.858],
    'HER2_MCCV': [0.326, 0.395, 0.368, 0.306, 0.396, 0.368, 0.354, 0.246, 0.377, 0.399, 0.359, 0.364, 0.353],
    'HER2_HO': [0.104, 0.156, 0.130, 0.190, 0.148, 0.160, 0.133, 0.107, 0.153, 0.219, 0.191, 0.126, 0.164],
    'Mean_Rank': [12.75, 7.00, 9.63, 7.13, 4.38, 4.13, 9.63, 11.38, 4.25, 2.00, 4.63, 9.88, 4.25]
}

df = pd.DataFrame(data)

# ============================================================================
# OPTION 1: Biclustered Heatmap with actual performance values
# ============================================================================

def plot_performance_heatmap(df, output_file='performance_heatmap.png'):
    """
    Create biclustered heatmap showing performance values across all tasks
    Highlights Virchow v2 (best mean rank) with bold labels
    """
    # Select only performance columns
    perf_cols = ['PAM50_MCCV', 'PAM50_HO', 'ER_MCCV', 'ER_HO',
                 'PR_MCCV', 'PR_HO', 'HER2_MCCV', 'HER2_HO']

    # Create DataFrame for biclustering
    heatmap_df = df[perf_cols].copy()
    heatmap_df.index = df['Model'].values
    heatmap_df.columns = ['PAM50\nMCCV', 'PAM50\nHO', 'ER\nMCCV', 'ER\nHO',
                          'PR\nMCCV', 'PR\nHO', 'HER2\nMCCV', 'HER2\nHO']

    # Use clustering palette (pink/magenta)
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    # Create biclustered heatmap
    g = sns.clustermap(heatmap_df,
                       annot=True,
                       fmt='.3f',
                       cmap=cmap,
                       vmin=0.0,
                       vmax=1.0,
                       cbar_kws={'label': 'Performance (F1 / PR-AUC)'},
                       linewidths=0.5,
                       linecolor='white',
                       figsize=(13, 10),
                       dendrogram_ratio=0.15,
                       cbar_pos=(0.02, 0.83, 0.03, 0.15))

    # Highlight Virchow v2 row with bold text
    virchow_idx = list(heatmap_df.index).index('Virchow v2')
    reordered_idx = g.dendrogram_row.reordered_ind
    virchow_pos = list(reordered_idx).index(virchow_idx)

    # Make Virchow v2 label bold and add star
    yticklabels = g.ax_heatmap.get_yticklabels()
    yticklabels[virchow_pos].set_weight('bold')
    yticklabels[virchow_pos].set_color('#8B008B')  # Dark magenta
    yticklabels[virchow_pos].set_text('★ ' + yticklabels[virchow_pos].get_text())

    g.fig.suptitle('Foundation Model Performance Across Tasks and Cohorts\n(Biclustered)',
                   fontsize=16, fontweight='bold', y=0.98)
    g.ax_heatmap.set_xlabel('Task', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

# ============================================================================
# OPTION 2: Biclustered Heatmap with rankings
# ============================================================================

def plot_ranking_heatmap(df, output_file='ranking_heatmap.png'):
    """
    Create biclustered heatmap showing rankings (1=best, 13=worst)
    Highlights Virchow v2 (best mean rank)
    """
    # Select performance columns
    perf_cols = ['PAM50_MCCV', 'PAM50_HO', 'ER_MCCV', 'ER_HO',
                 'PR_MCCV', 'PR_HO', 'HER2_MCCV', 'HER2_HO']

    # Compute rankings (1=best)
    rankings_df = df[perf_cols].rank(ascending=False, method='min')
    rankings_df.index = df['Model'].values
    rankings_df.columns = ['PAM50\nMCCV', 'PAM50\nHO', 'ER\nMCCV', 'ER\nHO',
                           'PR\nMCCV', 'PR\nHO', 'HER2\nMCCV', 'HER2\nHO']

    # Use clustering palette reversed (better ranks = darker)
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True).reversed()

    # Create biclustered heatmap
    g = sns.clustermap(rankings_df,
                       annot=True,
                       fmt='.0f',
                       cmap=cmap,
                       vmin=1,
                       vmax=13,
                       cbar_kws={'label': 'Rank (1=Best, 13=Worst)'},
                       linewidths=0.5,
                       linecolor='white',
                       figsize=(13, 10),
                       dendrogram_ratio=0.15,
                       cbar_pos=(0.02, 0.83, 0.03, 0.15))

    # Highlight Virchow v2 row
    virchow_idx = list(rankings_df.index).index('Virchow v2')
    reordered_idx = g.dendrogram_row.reordered_ind
    virchow_pos = list(reordered_idx).index(virchow_idx)

    yticklabels = g.ax_heatmap.get_yticklabels()
    yticklabels[virchow_pos].set_weight('bold')
    yticklabels[virchow_pos].set_color('#8B008B')  # Dark magenta
    yticklabels[virchow_pos].set_text('★ ' + yticklabels[virchow_pos].get_text())

    g.fig.suptitle('Foundation Model Rankings Across Tasks and Cohorts\n(Biclustered)',
                   fontsize=16, fontweight='bold', y=0.98)
    g.ax_heatmap.set_xlabel('Task', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

# ============================================================================
# OPTION 3: Biclustered Split view (MCCV vs HO)
# ============================================================================

def plot_split_heatmap(df, output_file='split_heatmap.png'):
    """
    Create biclustered heatmaps for MCCV and HO separately
    """
    mccv_cols = ['PAM50_MCCV', 'ER_MCCV', 'PR_MCCV', 'HER2_MCCV']
    ho_cols = ['PAM50_HO', 'ER_HO', 'PR_HO', 'HER2_HO']

    # Create DataFrames
    mccv_df = df[mccv_cols].copy()
    mccv_df.index = df['Model'].values
    mccv_df.columns = ['PAM50', 'ER', 'PR', 'HER2']

    ho_df = df[ho_cols].copy()
    ho_df.index = df['Model'].values
    ho_df.columns = ['PAM50', 'ER', 'PR', 'HER2']

    # Use clustering palette
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    # MCCV biclustered heatmap
    g1 = sns.clustermap(mccv_df,
                        annot=True, fmt='.3f', cmap=cmap,
                        vmin=0.0, vmax=1.0,
                        cbar_kws={'label': 'Performance'},
                        linewidths=0.5, linecolor='white',
                        figsize=(8, 9),
                        dendrogram_ratio=0.15,
                        cbar_pos=(0.02, 0.83, 0.03, 0.15))

    # Highlight Virchow v2
    virchow_idx = list(mccv_df.index).index('Virchow v2')
    reordered_idx = g1.dendrogram_row.reordered_ind
    virchow_pos = list(reordered_idx).index(virchow_idx)

    yticklabels = g1.ax_heatmap.get_yticklabels()
    yticklabels[virchow_pos].set_weight('bold')
    yticklabels[virchow_pos].set_color('#8B008B')
    yticklabels[virchow_pos].set_text('★ ' + yticklabels[virchow_pos].get_text())

    g1.fig.suptitle('Internal Validation (MCCV) - Biclustered',
                    fontsize=14, fontweight='bold', y=0.98)
    g1.ax_heatmap.set_ylabel('Model', fontsize=11, fontweight='bold')

    plt.savefig(output_file.replace('.png', '_mccv.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file.replace('.png', '_mccv.png')}")
    plt.close()

    # HO biclustered heatmap
    g2 = sns.clustermap(ho_df,
                        annot=True, fmt='.3f', cmap=cmap,
                        vmin=0.0, vmax=1.0,
                        cbar_kws={'label': 'Performance'},
                        linewidths=0.5, linecolor='white',
                        figsize=(8, 9),
                        dendrogram_ratio=0.15,
                        cbar_pos=(0.02, 0.83, 0.03, 0.15))

    # Highlight Virchow v2
    virchow_idx = list(ho_df.index).index('Virchow v2')
    reordered_idx = g2.dendrogram_row.reordered_ind
    virchow_pos = list(reordered_idx).index(virchow_idx)

    yticklabels = g2.ax_heatmap.get_yticklabels()
    yticklabels[virchow_pos].set_weight('bold')
    yticklabels[virchow_pos].set_color('#8B008B')
    yticklabels[virchow_pos].set_text('★ ' + yticklabels[virchow_pos].get_text())

    g2.fig.suptitle('External Validation (Hold-Out) - Biclustered',
                    fontsize=14, fontweight='bold', y=0.98)
    g2.ax_heatmap.set_ylabel('Model', fontsize=11, fontweight='bold')

    plt.savefig(output_file.replace('.png', '_ho.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file.replace('.png', '_ho.png')}")
    plt.close()

# ============================================================================
# OPTION 4: Biclustered Performance degradation heatmap (MCCV - HO)
# ============================================================================

def plot_degradation_heatmap(df, output_file='degradation_heatmap.png'):
    """
    Show performance drop from MCCV to HO with biclustering
    """
    # Calculate degradation
    degradation = {
        'PAM50': df['PAM50_MCCV'] - df['PAM50_HO'],
        'ER': df['ER_MCCV'] - df['ER_HO'],
        'PR': df['PR_MCCV'] - df['PR_HO'],
        'HER2': df['HER2_MCCV'] - df['HER2_HO']
    }

    degrad_df = pd.DataFrame(degradation)
    degrad_df.index = df['Model'].values

    # Use clustering palette
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    # Create biclustered heatmap
    g = sns.clustermap(degrad_df,
                       annot=True, fmt='.3f', cmap=cmap,
                       cbar_kws={'label': 'Performance Degradation (MCCV - HO)'},
                       linewidths=0.5, linecolor='white',
                       figsize=(9, 10),
                       dendrogram_ratio=0.15,
                       cbar_pos=(0.02, 0.83, 0.03, 0.15))

    # Highlight Virchow v2
    virchow_idx = list(degrad_df.index).index('Virchow v2')
    reordered_idx = g.dendrogram_row.reordered_ind
    virchow_pos = list(reordered_idx).index(virchow_idx)

    yticklabels = g.ax_heatmap.get_yticklabels()
    yticklabels[virchow_pos].set_weight('bold')
    yticklabels[virchow_pos].set_color('#8B008B')
    yticklabels[virchow_pos].set_text('★ ' + yticklabels[virchow_pos].get_text())

    g.fig.suptitle('Performance Degradation: Internal to External Validation\n(Biclustered)',
                   fontsize=14, fontweight='bold', y=0.98)
    g.ax_heatmap.set_xlabel('Task', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

# ============================================================================
# RUN ALL OPTIONS
# ============================================================================

if __name__ == "__main__":
    print("Generating heatmaps...")
    
    # Option 1: Performance values
    plot_performance_heatmap(df, 'performance_heatmap.png')
    
    # Option 2: Rankings
    plot_ranking_heatmap(df, 'ranking_heatmap.png')
    
    # Option 3: Split MCCV vs HO
    plot_split_heatmap(df, 'split_heatmap.png')
    
    # Option 4: Performance degradation
    plot_degradation_heatmap(df, 'degradation_heatmap.png')
    
    print("\nAll heatmaps generated successfully!")