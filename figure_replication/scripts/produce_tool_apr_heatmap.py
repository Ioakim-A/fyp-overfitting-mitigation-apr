import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from matplotlib import rcParams

def produce_tool_apr_heatmap(csv_file, output_file='apr_tool_heatmap.pdf', figsize=(10, 8), 
                             cmap='RdBu_r', center=0, font_scale=1.2):
    """
    Create a publication-quality heatmap for APR tools evaluation results.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file.
    output_file : str
        Path to save the output figure.
    figsize : tuple
        Size of the figure.
    cmap : str
        Colormap name for the heatmap.
    center : float
        Center value for the colormap.
    font_scale : float
        Scaling factor for font sizes.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Exclude the final 'Overall' row
    df = df.iloc[:-1]
    
    # Set the 'Approach' column as index
    df = df.set_index('Approach')
    
    # Convert all values to float and normalize -0.0 to 0.0
    df = df.astype(float)
    df[df == -0.0] = 0.0
    
    name_conversion = {
    'entropy_delta': 'Yang et al.',
    'Invalidator': 'Invalidator',
    'FIXCHECK': 'FIXCHECK',
    'LLM4PatchCorrectness': 'LLM4Patch\nCorrect',
    'DL4PatchCorrectness': 'Tian et al.'
    }
    df.rename(columns=name_conversion, inplace=True)

    print("MCC prediction distribution per tool:")
    print(df)
    
    # Replace "-ir4" in the row index "repairllama-ir4" with just "repairllama" if present
    df.index = df.index.map(lambda x: str(x).replace("-ir4", "") if "repairllama-ir4" in str(x) else x)

    # Set seaborn style for scientific plots
    sns.set_context("paper", font_scale=font_scale)
    sns.set_style("whitegrid")
    
    # Configure matplotlib for publication quality
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']
    rcParams['axes.grid'] = False
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute vmin and vmax for better color mapping
    abs_max = max(abs(df.values.min()), abs(df.values.max()))
    if center == 0:
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = df.values.min(), df.values.max()
    
    # Create the heatmap
    heatmap = sns.heatmap(
        df, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap,
        linewidths=.5,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar_kws={'label': 'MCC', 'shrink': 0.8}
    )

    # Shrink x-axis tick label font size and rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

    # Set axis labels
    ax.set_xlabel('Overfitting Detection Tool', fontweight='bold')
    ax.set_ylabel('APR Tool', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, ha='center')
    
    # Adjust the y-axis tick labels to be centered
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_verticalalignment('center')
    
    # Add a tight layout to maximize space usage
    plt.tight_layout()
    
    # Create directory for output file if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Save the figure with high quality
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"APR tools heatmap saved to {output_file}")
    
    # Close the figure
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a publication-quality heatmap for APR tools evaluation")
    parser.add_argument("csv_file", help="Path to the CSV file with APR tools evaluation data")
    parser.add_argument("--output", "-o", default="apr_tool_heatmap.pdf", 
                        help="Output file name (supports .pdf, .png, .svg, etc.)")
    parser.add_argument("--cmap", default="RdBu_r", 
                        help="Colormap for the heatmap (default: RdBu_r)")
    parser.add_argument("--width", type=float, default=10.5, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=8.0, help="Figure height in inches")
    parser.add_argument("--center", type=float, default=0, 
                        help="Center value for the colormap (default: 0)")
    parser.add_argument("--font-scale", type=float, default=2.0, 
                        help="Font scale factor (default: 2.0)")
    
    args = parser.parse_args()
    
    produce_tool_apr_heatmap(
        args.csv_file,
        output_file=args.output,
        figsize=(args.width, args.height),
        cmap=args.cmap,
        center=args.center,
        font_scale=args.font_scale
    )
