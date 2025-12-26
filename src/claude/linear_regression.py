import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import seaborn as sns

# ============================================================================
# Data
# ============================================================================

classes = [
    'Basal', 'Her2-enriched', 'LumA', 'LumB', 'Normal',
    'ER-negative', 'ER-positive',
    'PR-negative', 'PR-positive',
    'HER2-negative', 'HER2-positive'
]

# Task assignments for coloring
tasks = ['PAM50']*5 + ['ER']*2 + ['PR']*2 + ['HER2']*2

normalized_gain = np.array([
    0.000, 0.000, -0.761, 0.055, 0.000,  # PAM50
    0.020, -0.433,  # ER
    -0.458, 0.438,  # PR
    0.000, 0.000   # HER2
])

RPC = np.array([
    -0.064, -1.000, -0.078, -0.517, -1.000,  # PAM50
    0.037, -0.441,  # ER
    0.198, -0.097,  # PR
    0.037, -0.441   # HER2
])

# Create DataFrame
df = pd.DataFrame({
    'Class': classes,
    'Task': tasks,
    'Normalized_Gain': normalized_gain,
    'RPC': RPC
})

# ============================================================================
# OPTION 1: Single panel with all data
# ============================================================================

def plot_single_regression(df, output_file='normalization_regression.png', exclude_classes=None):
    """
    Single scatter plot with regression line

    Args:
        df: DataFrame with Class, Task, Normalized_Gain, and RPC columns
        output_file: Output filename for the plot
        exclude_classes: List of class names to exclude from analysis (default: None)
    """
    # Filter classes if requested
    if exclude_classes is not None:
        df = df[~df['Class'].isin(exclude_classes)].copy()
        print(f"Excluded classes: {', '.join(exclude_classes)}")
        print(f"Remaining samples: {len(df)}")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color mapping - using magma palette from clustering
    magma_colors = sns.color_palette("magma", 4)
    colors = {'PAM50': magma_colors[0], 'ER': magma_colors[1], 'PR': magma_colors[2], 'HER2': magma_colors[3]}

    # Scatter plot by task
    task_order = ['PAM50', 'ER', 'PR', 'HER2']
    for task in task_order:
        mask = df['Task'] == task
        if mask.any():
            ax.scatter(df[mask]['Normalized_Gain'], df[mask]['RPC'],
                      color=colors[task], label=task, s=100, alpha=0.7, edgecolors='black', linewidth=1)

    # Regression line
    slope, intercept, r_value, p_value, std_err = linregress(df['Normalized_Gain'], df['RPC'])
    x_line = np.linspace(df['Normalized_Gain'].min(), df['Normalized_Gain'].max(), 100)
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8, 
            label=f'Linear fit (R²={r_value**2:.3f})')
    
    # Statistics text
    r, p = pearsonr(df['Normalized_Gain'], df['RPC'])
    stats_text = f'Pearson r = {r:.3f}\np = {p:.3f}'
    if p < 0.001:
        stats_text = f'Pearson r = {r:.3f}\np < 0.001'
    elif p < 0.05:
        stats_text += ' *'
    else:
        stats_text += ' n.s.'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels and formatting
    ax.set_xlabel('Normalized Performance Gain (NPG)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative Performance Change (RPC)', fontsize=14, fontweight='bold')
    ax.set_title('Stain Normalization Effect vs Performance Degradation', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

# ============================================================================
# OPTION 2: Two-panel (all data vs excluding outliers)
# ============================================================================

def plot_dual_regression(df, output_file='normalization_regression_dual.png', exclude_classes=None):
    """
    Two-panel figure: left=all data with outliers marked, right=excluding outliers

    Args:
        df: DataFrame with Class, Task, Normalized_Gain, and RPC columns
        output_file: Output filename for the plot
        exclude_classes: List of class names to exclude from analysis (default: None)
                        If None, automatically excludes classes with RPC == -1.0
    """
    # Filter classes if requested
    if exclude_classes is not None:
        print(f"Excluded classes: {', '.join(exclude_classes)}")
        mask_excluded = df['Class'].isin(exclude_classes)
        df_filtered = df[~mask_excluded].copy()
        excluded_classes = exclude_classes
        print(f"Remaining samples: {len(df_filtered)}")
    else:
        # Identify outliers: Her2-enriched and Normal both have RPC = -1.0 (extreme minimum value)
        mask_excluded = df['RPC'] == -1.0
        df_filtered = df[~mask_excluded].copy()
        # Get excluded class names for title
        excluded_classes = df[mask_excluded]['Class'].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color mapping - using magma palette from clustering
    magma_colors = sns.color_palette("magma", 4)
    colors = {'PAM50': magma_colors[0], 'ER': magma_colors[1], 'PR': magma_colors[2], 'HER2': magma_colors[3]}

    # ========================================================================
    # Panel 1: All data WITH outliers (included in regression but marked)
    # ========================================================================

    # Plot ALL points by task (fixed order for consistent colors)
    task_order = ['PAM50', 'ER', 'PR', 'HER2']
    for task in task_order:
        mask_task = df['Task'] == task
        if mask_task.any():
            ax1.scatter(df[mask_task]['Normalized_Gain'], df[mask_task]['RPC'],
                       color=colors[task], label=task, s=150, alpha=0.8,
                       edgecolors='black', linewidth=1.5, zorder=2)

    # Mark outliers with hollow red circles (on top) - appears last in legend
    ax1.scatter(df[mask_excluded]['Normalized_Gain'],
               df[mask_excluded]['RPC'],
               facecolors='none', edgecolors='#d62728', s=300,
               linewidth=2.5, alpha=1.0,
               label='Outliers', zorder=3)

    # Regression for ALL data (including outliers)
    slope1, intercept1, r1, p1, _ = linregress(df['Normalized_Gain'], df['RPC'])
    x_line = np.linspace(df['Normalized_Gain'].min(), df['Normalized_Gain'].max(), 100)
    y_line = slope1 * x_line + intercept1
    ax1.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8,
             label=f'Linear fit (R²={r1**2:.3f})', zorder=4)

    # ========================================================================
    # Panel 2: WITHOUT outliers (excluded from regression)
    # ========================================================================

    # Plot included points by task (outliers are excluded from this panel) - same order as left panel
    task_order = ['PAM50', 'ER', 'PR', 'HER2']
    for task in task_order:
        mask = df_filtered['Task'] == task
        if mask.any():
            ax2.scatter(df_filtered[mask]['Normalized_Gain'],
                       df_filtered[mask]['RPC'],
                       color=colors[task], label=task, s=100, alpha=0.7,
                       edgecolors='black', linewidth=1, zorder=2)

    # Regression for filtered data ONLY (without outliers)
    if len(df_filtered) > 2:
        slope2, intercept2, r2, p2, _ = linregress(
            df_filtered['Normalized_Gain'], df_filtered['RPC'])
        x_line2 = np.linspace(df_filtered['Normalized_Gain'].min(),
                             df_filtered['Normalized_Gain'].max(), 100)
        y_line2 = slope2 * x_line2 + intercept2
        ax2.plot(x_line2, y_line2, 'r--', linewidth=2, alpha=0.8,
                label=f'Linear fit (R²={r2**2:.3f})', zorder=4)

    # Formatting for both panels
    for ax in [ax1, ax2]:
        ax.set_xlabel('Normalized Performance Gain (NPG)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Performance Change (RPC)', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9, markerscale=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # Create titles with statistics integrated
    sig1 = ' n.s.' if p1 >= 0.05 else ''
    title1 = f'All data (n={len(df)})\nPearson r = {r1:.3f}, p = {p1:.3f}{sig1}'
    ax1.set_title(title1, fontsize=11, fontweight='bold')

    if len(df_filtered) > 2:
        sig2 = ' n.s.' if p2 >= 0.05 else ''
        excluded_str = ', '.join(excluded_classes)
        title2 = f'Without {excluded_str} (n={len(df_filtered)})\nPearson r = {r2:.3f}, p = {p2:.3f}{sig2}'
    else:
        excluded_str = ', '.join(excluded_classes)
        title2 = f'Without {excluded_str} (n={len(df_filtered)})\nInsufficient data'
    ax2.set_title(title2, fontsize=11, fontweight='bold')
    
    fig.suptitle('Stain Normalization Effect vs Performance Degradation', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

# ============================================================================
# OPTION 3: Annotated scatter with class labels
# ============================================================================

def plot_annotated_regression(df, output_file='normalization_regression_annotated.png', exclude_classes=None):
    """
    Scatter plot with class labels for each point

    Args:
        df: DataFrame with Class, Task, Normalized_Gain, and RPC columns
        output_file: Output filename for the plot
        exclude_classes: List of class names to exclude from analysis (default: None)
    """
    # Filter classes if requested
    if exclude_classes is not None:
        df = df[~df['Class'].isin(exclude_classes)].copy()
        print(f"Excluded classes: {', '.join(exclude_classes)}")
        print(f"Remaining samples: {len(df)}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color mapping - using magma palette from clustering
    magma_colors = sns.color_palette("magma", 4)
    colors = {'PAM50': magma_colors[0], 'ER': magma_colors[1], 'PR': magma_colors[2], 'HER2': magma_colors[3]}

    # Scatter plot
    task_order = ['PAM50', 'ER', 'PR', 'HER2']
    for task in task_order:
        mask = df['Task'] == task
        if mask.any():
            ax.scatter(df[mask]['Normalized_Gain'], df[mask]['RPC'],
                      color=colors[task], label=task, s=120, alpha=0.7,
                      edgecolors='black', linewidth=1.5)

    # Annotate points
    for idx, row in df.iterrows():
        ax.annotate(row['Class'],
                   (row['Normalized_Gain'], row['RPC']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    # Regression line
    slope, intercept, r_value, p_value, _ = linregress(df['Normalized_Gain'], df['RPC'])
    x_line = np.linspace(df['Normalized_Gain'].min(), df['Normalized_Gain'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8, 
            label=f'Linear fit (R²={r_value**2:.3f})')
    
    # Statistics
    r, p = pearsonr(df['Normalized_Gain'], df['RPC'])
    stats_text = f'Pearson r = {r:.3f}\np = {p:.3f}'
    if p >= 0.05:
        stats_text += ' n.s.'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Formatting
    ax.set_xlabel('Normalized Performance Gain (NPG)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative Performance Change (RPC)', fontsize=14, fontweight='bold')
    ax.set_title('Stain Normalization Effect vs Performance Degradation', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

# ============================================================================
# Run all options
# ============================================================================

if __name__ == "__main__":
    print("Generating normalization regression plot...")

    # Dual panel plot (default: auto-exclude outliers with RPC == -1.0)
    plot_dual_regression(df, 'normalization_regression_dual.png')

    # Example: Manually exclude specific classes
    # plot_dual_regression(df, 'normalization_regression_dual_custom.png',
    #                     exclude_classes=['Her2-enriched', 'Normal', 'LumB'])

    # Example: Single regression without specific classes
    # plot_single_regression(df, 'normalization_regression_no_pam50.png',
    #                       exclude_classes=['Basal', 'Her2-enriched', 'LumA', 'LumB', 'Normal'])

    # Example: Annotated regression excluding outliers
    # plot_annotated_regression(df, 'normalization_regression_annotated_filtered.png',
    #                          exclude_classes=['Her2-enriched', 'Normal'])

    print("\nPlot generated successfully!")